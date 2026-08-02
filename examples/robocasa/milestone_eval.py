"""Eval — MILESTONE-level evaluation of a System1 policy (oracle-referenced sim-check).

Between subtask (hard-reset per CHILD subgoal) and episode (reset ONCE, roll the whole episode):
this HARD-RESETS to each MILESTONE's start frame, then rolls the policy CONTINUOUSLY through that
milestone's child subgoals (feeding each child's prompt, handing off on the shared STOP rule), and
evaluates a sim-grounded MILESTONE goal predicate at the settled end.

Why: a milestone ("pick up the pan", "open the fridge door") is a semantically complete state change
whose motion has SETTLED by its end — so the sim predicate is reliable, unlike the child-level check
(which fires mid-motion -> false negatives even on a perfect oracle replay). See
docs/robocasa_milestone_sim_check_design.md and milestone_sim_check.py.

ORACLE-REFERENCED: run once with --oracle to capture the GT end-state signals per milestone
(finger opening, object lift Δz, gripper→obj distance, door joint qpos, fixture flag, base pose);
those `ref` dicts are written into the oracle method's episode.json. A policy run then LOADS the
matching oracle ref for each milestone and compares (no self-chosen absolute thresholds).

Output layout MIRRORS subtask/episode eval (per-milestone dir <mNN_<goalprim>>/ with clean.mp4 +
steps.npz + steps_meta.json, plus episode.json), so the GUI reads it unchanged. episode.json
carries per-milestone `milestone_sim_check` (verdict/rule/target/detail) + `ref` (oracle only) +
episode-level `sim_success_final`.

Run (robocasa micromamba env):
    PY=/home/yinpei.dai/micromamba/envs/robocasa/bin/python
    # 1) oracle reference (no server):
    $PY examples/robocasa/milestone_eval.py --episode-list eps.txt --oracle \
        --out-root m0717_eval_results/milestone --method oracle
    # 2) policy (reads the oracle refs):
    $PY examples/robocasa/milestone_eval.py --episode-list eps.txt --host 127.0.0.1 --port 8020 \
        --norm-stats checkpoints/<exp>/49999/assets/robocasa_system1/norm_stats.json \
        --out-root m0717_eval_results/milestone --method v4_progreg_noexec --oracle-method oracle
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import imageio
import numpy as np
from openpi_client import websocket_client_policy as _wcp

from robocasa.scripts.eval.subtask_common import EpisodeSim, load_subgoals, DEFAULT_SUBGOAL_METHOD
from robocasa.scripts.eval.subtask_env import (
    make_camera_env,
    raw_state_from_obs,
    images_from_obs,
    base_reference,
    lean_state_from_raw16,
)
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

import subtask_eval as SE
from episode_eval import _rollout_subgoal_continuous  # continuous multi-child rollout
import milestone_sim_check as MS
from stop_criterion import StopConfig


def _group_milestones(subgoals):
    """Ordered list of (milestone_index, milestone_text, [child subgoals]) preserving child order."""
    out = []
    cur = None
    for sg in subgoals:
        if cur is None or sg.milestone_index != cur[0]:
            cur = (sg.milestone_index, sg.milestone_subgoal, [])
            out.append(cur)
        cur[2].append(sg)
    return out


def eval_episode(episode_dir: Path, client, args, out_root: Path, method: str,
                 norm_stats=None, oracle_refs: dict | None = None) -> dict:
    ann = load_subgoals(episode_dir, args.subgoal_method)
    env = make_camera_env(ann.lerobot_dir)
    flat = ann.episode_id.replace("/", "__")
    ep_out = out_root / method / flat
    ep_out.mkdir(parents=True, exist_ok=True)
    fps = int(round(ann.fps))
    is_oracle = client is None
    stop_cfg = StopConfig(progress_thresh=args.stop_progress, eps=args.stop_eps, window=args.stop_window)
    try:
        ld = Path(ann.lerobot_dir)
        states = LU.get_episode_states(ld, ann.episode_index)
        actions = LU.get_episode_actions(ld, ann.episode_index)
        if not ann.subgoals:
            raise ValueError("no subgoals")
        model_xml = LU.get_episode_model_xml(ld, ann.episode_index)
        ep_meta = json.dumps(LU.get_episode_meta(ld, ann.episode_index))
        milestones = _group_milestones(ann.subgoals)
        # FULL reset (model+ep_meta) ONCE per episode to load the MJCF; per-milestone resets are
        # STATES-ONLY (like finestep/episode eval). A full reset per milestone re-runs env.reset()
        # which re-inits the gripper/objects and DROPS a held object (e.g. DeliverStraw: the straw,
        # gripped since the pick-up milestone, fell during the navigate milestone's reset). The
        # recorded state already carries everything, so states-only is correct and non-disturbing.
        reset_to(env, dict(states=states[milestones[0][2][0].start], model=model_xml, ep_meta=ep_meta))

        records = []
        for ms_idx, ms_text, kids in milestones:
            m_start = kids[0].start
            goal_prim = MS.milestone_goal_primitive([k.primitive for k in kids])
            # HARD-RESET to this milestone's start frame — STATES ONLY (no model/ep_meta -> no
            # env.reset(), so a held object is not disturbed).
            reset_to(env, dict(states=states[m_start]))
            obs0 = env._get_observations(force_update=True)
            base_pos_ref, base_yaw_ref = base_reference(obs0)
            sim = EpisodeSim(env=env, lerobot_dir=ann.lerobot_dir, episode_index=ann.episode_index)
            sim._model_loaded = True
            # object z of EVERY env object at milestone START (for pick_up lift Δz); index by the
            # held object once it's known (resolved by contact during the oracle replay).
            start_z_by_obj = {n: MS._obj_z(env, n) for n in (getattr(env, "objects", {}) or {})}

            # Per-CHILD outputs (SAME layout as finestep/episode: one dir + video + steps.npz per
            # fine subgoal). Presentation is identical across modes; only the RESET differs (here we
            # reset ONCE to the milestone start, then roll children continuously). The milestone
            # sim-check verdict attaches to the milestone's LAST child; other children -> unknown.
            last_cmd_grip = 0.0
            per_child = []   # list of (sg, frames, step_records, advanced, budinfo)
            child_refs = {}  # child_index -> per-span oracle ref (oracle branch fills it)
            if is_oracle:
                # ORACLE = STATE-DRIVEN replay. NO RENDER (drop images for speed — the oracle video is
                # not used; only the per-span REF scalars are). We step recorded states per child span
                # and CAPTURE a subtask REF at EACH MEANINGFUL span end (keyed by child_index, using
                # that span's OWN primitive), so finestep can score every meaningful span, and
                # milestone can score its end child. Objects held during a span resolved by contact.
                child_refs = {}   # child_index -> ref dict (only for meaningful spans)
                for sg in kids:
                    z_start = {n: MS._obj_z(env, n) for n in (getattr(env, "objects", {}) or {})}
                    held_counts: dict[str, int] = {}
                    for fr in range(sg.start, sg.end + 1):
                        reset_to(env, dict(states=states[fr]))
                        env._get_observations(force_update=True)
                        hc = MS._gripper_contact_obj(env)
                        if hc:
                            held_counts[hc] = held_counts.get(hc, 0) + 1
                    # env is now at this span's END frame. Capture the ref for its own primitive.
                    gp = MS.span_goal_primitive(sg.primitive)
                    if gp is not None:
                        hobj = max(held_counts, key=held_counts.get) if held_counts else None
                        sz = z_start.get(hobj) if hobj else None
                        child_refs[sg.child_index] = MS.capture_ref_state(
                            env, sg.subgoal, gp, sz, held_obj=hobj)
                    per_child.append((sg, [], [], True, {"budget": sg.end - sg.start + 1, "est_length": None}))
                # milestone-level ref = the milestone-end child's ref (for /milestone scoring)
                held_obj = None; start_obj_z = None
                ref = None  # milestone ref set below from child_refs[last_ci] if present
            else:
                # POLICY: roll continuously through the children (state carries over between them);
                # keep EACH child's own frames/records (its native local frame_step) -> per-child dir.
                for j, sg in enumerate(kids):
                    anchor_obs = env._get_observations(force_update=True)
                    anchor_imgs = images_from_obs(anchor_obs)
                    anchor_state = raw_state_from_obs(anchor_obs)
                    roll = _rollout_subgoal_continuous(
                        env, sim, client, states, actions, sg, ep_len=len(actions),
                        base_pos_ref=base_pos_ref, base_yaw_ref=base_yaw_ref,
                        anchor_imgs=anchor_imgs, anchor_state=anchor_state,
                        task_goal=ann.instruction, resize=args.resize_size,
                        replan_steps=args.replan_steps, horizon_mult=args.horizon_mult,
                        max_steps_cap=args.max_steps_cap, stop_cfg=stop_cfg, norm_stats=norm_stats,
                        last_cmd_grip_init=last_cmd_grip, do_settle=(j == 0), settle_steps=args.settle_steps,
                        zero_arm_in_base=not args.no_zero_arm_in_base, oracle=False,
                        budget_formula=getattr(args, "budget_formula", SE.DEFAULT_BUDGET_FORMULA))
                    last_cmd_grip = roll["last_cmd_grip"]
                    per_child.append((sg, roll.pop("_clean_frames", []), roll.pop("_step_records", []),
                                      bool(roll.get("advanced")),
                                      {"budget": roll.get("budget"), "est_length": roll.get("est_length")}))

            # ---- settled milestone end: verdict (attached to the milestone's LAST child) ----
            last_ci = kids[-1].child_index
            is_term = any(k.is_terminal for k in kids) or (ms_idx == milestones[-1][0])
            if is_oracle:
                ms_verdict = {"verdict": "reference", "rule": "oracle_ref", "target": "", "detail": ""}
            else:
                # milestone scoring uses the milestone-END child's per-span oracle ref.
                mref = (oracle_refs or {}).get(str(last_ci))
                sub = MS.subtask_check(env, ms_text, mref, last_grip=last_cmd_grip) if mref else \
                    {"verdict": "unknown", "rule": "no_oracle_ref", "target": "", "detail": ""}
                if is_term:
                    # TERMINAL milestone: AND the (loose) subtask verdict with the episode-level env
                    # _check_success — task-level ground truth. A span is the final SUCCESS only if
                    # BOTH agree; either failing -> failure. (subtask unknown -> fall back to episode.)
                    ep_ok = bool(sim.check_full_success())
                    sv = sub.get("verdict")
                    if sv in ("success", "failure"):
                        v = "success" if (sv == "success" and ep_ok) else "failure"
                        ms_verdict = {"verdict": v, "rule": "subtask_AND_episode",
                                      "target": sub.get("target", ""),
                                      "detail": f"subtask={sv} AND episode_check_success={ep_ok} -> {v}. {sub.get('detail','')}"}
                    else:
                        ms_verdict = {"verdict": "success" if ep_ok else "failure",
                                      "rule": "episode_check_success",
                                      "target": "", "detail": f"env _check_success at end = {ep_ok} (subtask unknown)"}
                else:
                    ms_verdict = sub

            # write one dir per CHILD; the milestone verdict lands on the LAST child, others -> unknown.
            for sg, frames, recs, advanced, budinfo in per_child:
                sub_dir = ep_out / f"child{sg.child_index:02d}_{sg.primitive}"
                sub_dir.mkdir(parents=True, exist_ok=True)
                if frames:
                    imageio.mimwrite(sub_dir / "clean.mp4", frames, fps=fps, codec="libx264",
                                     macro_block_size=1, output_params=["-g", "1", "-movflags", "+faststart"])
                is_ms_end = (sg.child_index == last_ci)
                child_verdict = ms_verdict if is_ms_end else \
                    {"verdict": "unknown", "rule": "not_milestone_end", "target": "",
                     "detail": "not the last fine subgoal of its milestone — no goal event here"}
                # ANCHOR (subgoal-start) state, so the GUI's ANCHOR block matches finestep. The anchor
                # is the child's FIRST logged frame (its start state); recs[0]["cur_raw16"/"cur_lean"].
                a_raw16 = recs[0].get("cur_raw16") if recs else None
                a_lean = recs[0].get("cur_lean") if recs else None
                doc_meta = dict(
                    method=method, episode_id=ann.episode_id, task_name=ann.task_name,
                    child_index=sg.child_index, milestone_index=ms_idx, goal_primitive=goal_prim,
                    primitive=sg.primitive, subgoal=sg.subgoal, subgoal_detail=sg.subgoal_detail,
                    milestone_subgoal=ms_text, task_goal=ann.instruction,
                    span=[sg.start, sg.end], fps=ann.fps, eval_kind="milestone",
                    clean_video="clean.mp4", n_steps=len(recs),
                    anchor_state_raw16=a_raw16, anchor_state_lean14=a_lean,
                    base_pos_ref=np.round(np.asarray(base_pos_ref), 4).tolist(),
                    base_yaw_ref=round(float(base_yaw_ref), 4),
                    budget=budinfo.get("budget"), est_length=budinfo.get("est_length"),
                    settle_steps=(args.settle_steps if sg.child_index == kids[0].child_index else 0),
                    is_terminal=sg.is_terminal, is_milestone_end=is_ms_end)
                SE._write_steps_npz(sub_dir, doc_meta, recs)
                rec = dict(
                    child_index=sg.child_index, milestone_index=ms_idx, goal_primitive=goal_prim,
                    primitive=sg.primitive, subgoal=sg.subgoal, milestone_subgoal=ms_text,
                    span=[sg.start, sg.end], out_dir=str(sub_dir.relative_to(out_root)),
                    is_terminal=sg.is_terminal, is_milestone_end=is_ms_end, advanced=advanced,
                    milestone_sim_check=child_verdict)
                # store the PER-SPAN oracle ref on EACH meaningful child (keyed by child_index) so
                # BOTH finestep (per-span) and milestone (end child) can load it.
                if is_oracle and sg.child_index in child_refs:
                    rec["ref"] = child_refs[sg.child_index]
                records.append(rec)

        sim_success_final = bool(sim.check_full_success())
        # records are now PER-CHILD (same shape as finestep/episode: subgoals[]). The milestone
        # verdict sits on each milestone's LAST child (is_milestone_end=True); others are unknown.
        ep_doc = dict(method=method, episode_id=ann.episode_id, task_name=ann.task_name,
                      instruction=ann.instruction, eval_kind="milestone",
                      n_subgoals=len(records), n_milestones=len({r["milestone_index"] for r in records}),
                      sim_success_final=sim_success_final,
                      subgoals=records, error=None)
        (ep_out / "episode.json").write_text(json.dumps(ep_doc, indent=1))
        dec = [r["milestone_sim_check"]["verdict"] for r in records
               if r.get("is_milestone_end") and r["milestone_sim_check"]["verdict"] in ("success", "failure")]
        print(f"  {ann.task_name}: sim_success_final={sim_success_final}  "
              f"milestones decided {dec.count('success')}✓/{len(dec)}", flush=True)
        return ep_doc
    finally:
        try:
            env.close()
        except Exception:
            pass


def _load_oracle_refs(out_root: Path, oracle_method: str, episode_id: str) -> dict:
    """Load {str(child_index): ref} from the oracle episode.json — PER-SPAN refs (one per meaningful
    fine subgoal). Milestone scoring looks up the milestone-end child_index; finestep looks up each."""
    flat = episode_id.replace("/", "__")
    f = out_root / oracle_method / flat / "episode.json"
    if not f.is_file():
        return {}
    try:
        doc = json.loads(f.read_text())
    except Exception:
        return {}
    return {str(m["child_index"]): m.get("ref")
            for m in doc.get("subgoals", []) if m.get("ref") and m.get("child_index") is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-list", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--out-root", type=Path, default=Path("m0717_eval_results/milestone"))
    ap.add_argument("--method", required=True)
    ap.add_argument("--oracle", action="store_true",
                    help="ORACLE mode: replay GT actions + CAPTURE the per-milestone reference "
                         "end-state (finger/lift/dist/joint/flag/base-pose) into episode.json.")
    ap.add_argument("--oracle-method", default="oracle",
                    help="method dir holding the oracle reference refs (policy runs load these)")
    ap.add_argument("--norm-stats", type=Path, default=None)
    ap.add_argument("--subgoal-method", default=DEFAULT_SUBGOAL_METHOD)
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--horizon-mult", type=float, default=2.0)
    ap.add_argument("--max-steps-cap", type=int, default=400)
    ap.add_argument("--budget-formula", choices=SE.BUDGET_FORMULAS, default=SE.DEFAULT_BUDGET_FORMULA)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--no-zero-arm-in-base", action="store_true")
    ap.add_argument("--stop-progress", type=float, default=0.95)
    ap.add_argument("--stop-eps", type=float, default=0.02)
    ap.add_argument("--stop-window", type=int, default=5)
    args = ap.parse_args()

    if args.oracle:
        client = None
        norm_stats = None
        if args.norm_stats is not None:
            norm_stats = json.loads(Path(args.norm_stats).read_text())
            norm_stats = norm_stats.get("norm_stats", norm_stats)
        print(f"ORACLE milestone reference; method '{args.method}' -> {args.out_root}")
    else:
        if args.norm_stats is None:
            raise SystemExit("--norm-stats required for a policy run (or pass --oracle)")
        norm_stats = json.loads(Path(args.norm_stats).read_text())
        norm_stats = norm_stats.get("norm_stats", norm_stats)
        client = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)

    eps = [ln.strip() for ln in Path(args.episode_list).read_text().splitlines() if ln.strip()]
    idx = []
    idx_path = args.out_root / args.method / "index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_index():
        idx_path.write_text(json.dumps(dict(
            eval_kind="milestone", method=args.method, n_episodes=len(eps), n_done=len(idx),
            horizon_mult=args.horizon_mult, max_steps_cap=args.max_steps_cap,
            budget_formula=args.budget_formula,
            stop=dict(progress=args.stop_progress, eps=args.stop_eps, window=args.stop_window),
            episodes=idx), indent=1))

    t_run = time.time()
    for i, ep in enumerate(eps):
        print(f"=== [{i+1}/{len(eps)}] MILESTONE {ep} ===", flush=True)
        ep_t0 = time.time()
        try:
            # For a policy run, load the oracle reference refs keyed by the resolved episode_id.
            oracle_refs = None
            if not args.oracle:
                ann = load_subgoals(Path(ep), args.subgoal_method)
                oracle_refs = _load_oracle_refs(args.out_root, args.oracle_method, ann.episode_id)
            doc = eval_episode(Path(ep), client, args, args.out_root, args.method,
                               norm_stats=norm_stats, oracle_refs=oracle_refs)
            secs = round(time.time() - ep_t0, 2)
            idx.append(dict(episode_id=doc["episode_id"], task_name=doc["task_name"],
                            sim_success_final=doc["sim_success_final"],
                            n_milestones=doc["n_milestones"], seconds=secs))
        except Exception as e:
            traceback.print_exc()
            secs = round(time.time() - ep_t0, 2)
            idx.append(dict(episode=ep, error=repr(e), seconds=secs))
        done = [e for e in idx if e.get("seconds") is not None]
        avg = sum(e["seconds"] for e in done) / max(1, len(done))
        eta = avg * (len(eps) - (i + 1))
        print(f"    [{i+1}/{len(eps)}] took {secs}s  (avg {avg:.1f}s/ep, ETA {eta/60:.1f}m)", flush=True)
        _write_index()
    print(f"wrote {idx_path}  (total {round(time.time() - t_run, 1)}s over {len(eps)} episodes)")


if __name__ == "__main__":
    main()
