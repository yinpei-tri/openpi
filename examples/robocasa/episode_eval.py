"""Eval #2 — OPEN-LOOP EPISODE-level evaluation of a System1 policy.

Unlike the subtask eval (subtask_eval.py, which HARD-RESETS to each subgoal's start), this
resets ONLY ONCE to the FIRST subgoal's start frame, then rolls the policy CONTINUOUSLY through
the ordered subgoal list: the sim state carries over from one subgoal to the next (no reset).
The policy is fed each subgoal's prompt in turn; we advance to the next subgoal when the shared
STOP RULE fires (progress-at-threshold AND action-quiescence — see stop_criterion.py), because a
System1 policy is trained to stop/settle at a subgoal boundary. This mimics how System2 would
hand subgoals to System1 one-by-one, with System1 self-terminating each.

Episode success = the RoboCasa task env's own ``_check_success()`` at the end (the ground-truth
signal, per the plan). We also record which subgoals the model advanced past (stop fired) vs
timed out (hit the per-subgoal budget cap without stopping).

Output layout MIRRORS subtask_eval.py so the GUI reads it unchanged (per-subgoal dir with
clean.mp4 + steps.npz + steps_meta.json, plus episode.json). Extra episode-level fields:
``episode_success``, and per-subgoal ``advanced`` / ``stop_reason``.

Reuses subtask_eval.py's faithful logging helpers (obs assembly, per-frame fields, steps.npz
writer, progress reader) so the two evals stay byte-compatible in the GUI.

Run (robocasa micromamba env; needs openpi_client):
    PY=/home/yinpeidai/micromamba/envs/robocasa/bin/python
    $PY openpi/examples/robocasa/episode_eval.py \
        --episode-list eps.txt --host 127.0.0.1 --port 8010 \
        --norm-stats checkpoints/<exp>/49999/assets/robocasa_system1/norm_stats.json \
        --out-root episode_rollouts --method reg
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
    lerobot_action_to_sim,
    settle_action_from_first,
    lean_state_from_raw16,
    quantile_norm,
)
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

# Reuse the subtask driver's faithful logging + obs helpers verbatim (single source of truth).
import subtask_eval as SE
from subtask_eval import HORIZON, SIM_GRIP_IDX, SIM_CTRL_IDX
from stop_criterion import StopConfig, StopTracker
from robocasa.scripts.eval.subtask_env import build_prompt_text


def _rollout_subgoal_continuous(env, sim, client, sim_states, sim_actions, sg, *, ep_len,
                                base_pos_ref, base_yaw_ref, anchor_imgs, anchor_state,
                                task_goal, resize, replan_steps, horizon_mult, max_steps_cap,
                                stop_cfg: StopConfig, norm_stats, last_cmd_grip_init,
                                do_settle: bool, settle_steps: int, zero_arm_in_base: bool = True,
                                oracle: bool = False):
    """Roll out ONE subgoal WITHOUT resetting the sim (state carries over from the previous
    subgoal). Terminates when the stop rule fires (advanced=True) or the budget cap is hit
    (advanced=False). Returns clean frames + step records (subtask_eval layout) + advance info.

    ``do_settle``: only the FIRST subgoal runs a gripper-settle warmup (it followed a hard
    reset_to); later subgoals continue from live sim state, so NO settle.

    ``oracle``: replay the RECORDED GT actions for this subgoal's span instead of querying the
    policy (client is None). Reproduces the ground-truth episode as a reference: budget = span_len,
    no stop rule (advance exactly at the recorded span boundary), no settle. Same log/video layout.
    """
    span_len = sg.end - sg.start + 1
    est_len = SE.estimated_length(span_len)
    # Oracle replays exactly the recorded span; policy gets slack beyond it (clamped).
    budget = span_len if oracle else int(min(max_steps_cap, max(HORIZON, round(span_len * horizon_mult))))

    q01s = q99s = q01a = q99a = None
    if norm_stats is not None:
        q01s = np.asarray(norm_stats["state"]["q01"], np.float32)
        q99s = np.asarray(norm_stats["state"]["q99"], np.float32)
        q01a = np.asarray(norm_stats["actions"]["q01"], np.float32)
        q99a = np.asarray(norm_stats["actions"]["q99"], np.float32)
    anchor_lean = lean_state_from_raw16(anchor_state, base_pos_ref, base_yaw_ref)
    anchor_lean_norm = quantile_norm(anchor_lean, q01s, q99s) if q01s is not None else None

    clean_frames, step_records = [], []
    tracker = StopTracker(stop_cfg)

    def _log_step(step, phase, action_sim, prog, replanned, query=None):
        obs = env._get_observations(force_update=True)
        fld = SE._frame_fields(obs, task_goal, sg, step, budget, est_len, base_pos_ref, base_yaw_ref,
                               anchor_lean, anchor_lean_norm, action_sim, prog, q01s, q99s, q01a, q99a, phase)
        oracle_idx = min(sg.start + max(step, 0), ep_len - 1)
        oracle_a = np.asarray(sim_actions[oracle_idx], np.float64)
        fld["oracle_action_raw12"] = np.round(oracle_a, 4).tolist()
        fld["action_mse_vs_oracle"] = round(float(np.mean((np.asarray(action_sim) - oracle_a) ** 2)), 6)
        fld["eef_pos_world"] = np.round(sim.eef_pose()["pos"], 4).tolist()
        fld["gripper_width"] = round(float(fld["cur_raw16"][14] - fld["cur_raw16"][15]), 4)
        fld["sim_check_success"] = bool(sim.check_full_success())
        fld["replanned"] = bool(replanned)
        if query is not None:
            fld["query"] = query
        clean_frames.append(SE._stacked_from_obs(obs))
        step_records.append(fld)

    # First subgoal only: gripper-settle warmup (mirrors subtask_eval; later subgoals skip it).
    # Oracle replays the recorded actions verbatim, so it never settles.
    last_cmd_grip = last_cmd_grip_init
    if do_settle and not oracle:
        settle = settle_action_from_first(sim_actions[sg.start])
        last_cmd_grip = float(settle[SIM_GRIP_IDX])
        for si in range(settle_steps):
            _log_step(-settle_steps + si, "settle", settle, None, replanned=False)
            env.step(settle)

    from robocasa.scripts.eval.subtask_env import sim_action_to_lean11, sim_to_lerobot12 as _sim_to_lerobot12

    import collections
    action_plan = collections.deque()
    executed = 0
    advanced = False
    stop_reason = None
    last_replan_prog = None
    while executed < budget:
        replanned = False
        query = None
        prog = None
        if not action_plan:
            replanned = True
            gripper_flag = "Close" if last_cmd_grip > 0 else "Open"
            if oracle:
                # ORACLE: "predict" the recorded action chunk from this step (robosuite-native
                # order already). Same structured logs/video as a policy run, as a GT reference.
                oi = min(sg.start + executed, ep_len - 1)
                chunk_sim = np.stack([sim_actions[min(oi + j, ep_len - 1)]
                                      for j in range(HORIZON)], axis=0)
                result = {}
                prog = None
            else:
                element = SE._obs_dict(
                    env, base_pos_ref=base_pos_ref, base_yaw_ref=base_yaw_ref,
                    anchor_imgs=anchor_imgs, anchor_state=anchor_state,
                    subgoal_text=sg.subgoal, task_goal=task_goal, est_length=est_len,
                    executed_step=executed, gripper_flag=gripper_flag, resize=resize)
                result = client.infer(element)
                chunk = np.asarray(result["actions"])          # (H,12) LeRobot order, unnormalized
                chunk_sim = np.stack([lerobot_action_to_sim(a) for a in chunk], axis=0)
                prog = SE._read_progress(result)
            if prog:
                prog["at_step"] = executed
            action_plan.extend(chunk_sim[:replan_steps])
            chunk_lean = np.stack([sim_action_to_lean11(a) for a in chunk_sim], axis=0)
            chunk_lean_norm = (quantile_norm(chunk_lean, q01a, q99a) if q01a is not None else None)
            # Prefer the REAL assembled prompt the server tokenized (real discretized state
            # ints); fall back to the placeholder template only if absent.
            real_prompt = (result.get("prompt_text") if isinstance(result, dict) else None) \
                or build_prompt_text(task_goal, sg.subgoal, "Success", est_len, executed, gripper_flag)
            query = dict(
                prompt=real_prompt,
                gripper_flag=gripper_flag, executed_step=int(executed),
                replan_steps=int(replan_steps), horizon=int(HORIZON),
                chunk_lean11=np.round(chunk_lean, 4).tolist(),
                chunk_lean11_norm=(np.round(chunk_lean_norm, 4).tolist() if chunk_lean_norm is not None else None),
                chunk_progress=(prog.get("progress_chunk") if prog else None))
            last_replan_prog = prog  # freshest progress readout for the stop tracker
        action_sim = action_plan.popleft()
        last_cmd_grip = float(action_sim[SIM_GRIP_IDX])
        _log_step(executed, "act", action_sim, prog, replanned, query=query)
        # Match subtask_eval: in MOBILE/base mode (control_mode>0) zero the arm eef delta on the
        # STEPPED action (whole-body IK otherwise integrates the residual into large arm drift).
        a_step = action_sim
        if zero_arm_in_base and action_sim[SIM_CTRL_IDX] > 0.0:
            a_step = action_sim.copy(); a_step[0:6] = 0.0
        env.step(a_step)
        executed += 1
        # Oracle replays the whole recorded span (budget=span_len) with NO early stop — it just
        # advances at the boundary. Policy uses the progress+quiescence stop rule.
        if not oracle:
            tracker.update(action_sim, prog if replanned else last_replan_prog)
            if tracker.should_stop():   # only after a full replan window (quiescence has data)
                advanced = True
                stop_reason = tracker.reason()
                break

    if oracle:
        # Oracle "advances" by construction at the recorded span end (it IS the GT boundary).
        advanced = True
        stop_reason = dict(oracle=True, span_end=True)
    elif not advanced:
        stop_reason = dict(timeout=True, budget=budget, **tracker.reason())

    return dict(
        _clean_frames=clean_frames, _step_records=step_records,
        steps=executed, budget=budget, span_len=span_len, est_length=est_len,
        advanced=advanced, stop_reason=stop_reason,
        last_cmd_grip=last_cmd_grip,
    )


def eval_episode(episode_dir: Path, client, args, out_root: Path, method: str, norm_stats=None) -> dict:
    ann = load_subgoals(episode_dir, args.subgoal_method)
    env = make_camera_env(ann.lerobot_dir)
    flat = ann.episode_id.replace("/", "__")
    ep_out = out_root / method / flat
    ep_out.mkdir(parents=True, exist_ok=True)
    fps = int(round(ann.fps))
    stop_cfg = StopConfig(progress_thresh=args.stop_progress, eps=args.stop_eps, window=args.stop_window)
    try:
        ld = Path(ann.lerobot_dir)
        states = LU.get_episode_states(ld, ann.episode_index)
        actions = LU.get_episode_actions(ld, ann.episode_index)
        if not ann.subgoals:
            raise ValueError("no subgoals")
        first = ann.subgoals[0]
        # RESET ONCE to the first subgoal's start frame (only-first-timestep reset for #2).
        init = dict(states=states[first.start],
                    model=LU.get_episode_model_xml(ld, ann.episode_index),
                    ep_meta=json.dumps(LU.get_episode_meta(ld, ann.episode_index)))
        reset_to(env, init)
        obs0 = env._get_observations(force_update=True)
        base_pos_ref, base_yaw_ref = base_reference(obs0)
        sim = EpisodeSim(env=env, lerobot_dir=ann.lerobot_dir, episode_index=ann.episode_index)
        sim._model_loaded = True

        records = []
        last_cmd_grip = 0.0
        for i, sg in enumerate(ann.subgoals):
            # anchor snapshot = CURRENT live obs at the moment we start this subgoal (NOT a reset).
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
                last_cmd_grip_init=last_cmd_grip, do_settle=(i == 0), settle_steps=args.settle_steps,
                zero_arm_in_base=not args.no_zero_arm_in_base, oracle=(client is None))
            last_cmd_grip = roll["last_cmd_grip"]

            clean = roll.pop("_clean_frames", [])
            step_records = roll.pop("_step_records", [])
            sub_dir = ep_out / f"child{sg.child_index:02d}_{sg.primitive}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            if clean:
                imageio.mimwrite(sub_dir / "clean.mp4", clean, fps=fps, codec="libx264",
                                 macro_block_size=1, output_params=["-g", "1", "-movflags", "+faststart"])
            anchor_img_files = {}
            for cam_key, img in anchor_imgs.items():
                fn = f"anchor_{cam_key}.jpg"
                imageio.imwrite(sub_dir / fn, np.ascontiguousarray(img))
                anchor_img_files[cam_key] = fn
            anchor_lean = lean_state_from_raw16(anchor_state, base_pos_ref, base_yaw_ref)
            doc_meta = dict(
                method=method, episode_id=ann.episode_id, task_name=ann.task_name,
                child_index=sg.child_index, milestone_index=sg.milestone_index,
                is_terminal=sg.is_terminal, primitive=sg.primitive,
                subgoal=sg.subgoal, subgoal_detail=sg.subgoal_detail,
                milestone_subgoal=sg.milestone_subgoal, task_goal=ann.instruction,
                span=[sg.start, sg.end], fps=ann.fps, eval_kind="episode",
                budget=roll["budget"], clean_video="clean.mp4", n_steps=len(step_records),
                anchor_images=anchor_img_files,
                anchor_state_raw16=np.round(np.asarray(anchor_state), 4).tolist(),
                anchor_state_lean14=np.round(anchor_lean, 4).tolist(),
                base_pos_ref=np.round(np.asarray(base_pos_ref), 4).tolist(),
                base_yaw_ref=round(float(base_yaw_ref), 4),
                advanced=roll["advanced"], stop_reason=roll["stop_reason"],
                summary=dict(steps=roll["steps"], budget=roll["budget"], span_len=roll["span_len"],
                             est_length=roll["est_length"], advanced=roll["advanced"]))
            SE._write_steps_npz(sub_dir, doc_meta, step_records)
            records.append(dict(
                child_index=sg.child_index, milestone_index=sg.milestone_index,
                is_terminal=sg.is_terminal, span=[sg.start, sg.end], primitive=sg.primitive,
                subgoal=sg.subgoal, out_dir=str(sub_dir.relative_to(out_root)),
                steps=roll["steps"], budget=roll["budget"], advanced=roll["advanced"],
                stop_reason=roll["stop_reason"]))

        # Episode success = env _check_success() at the END (ground-truth signal).
        episode_success = bool(sim.check_full_success())
        n_advanced = sum(1 for r in records if r["advanced"])
        ep_doc = dict(method=method, episode_id=ann.episode_id, task_name=ann.task_name,
                      instruction=ann.instruction, n_subgoals=len(ann.subgoals),
                      eval_kind="episode", episode_success=episode_success,
                      n_advanced=n_advanced, subgoals=records, error=None)
        (ep_out / "episode.json").write_text(json.dumps(ep_doc, indent=1))
        print(f"  {ann.task_name}: success={episode_success}  advanced={n_advanced}/{len(records)}")
        return ep_doc
    finally:
        try:
            env.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-list", required=True, help="text file: one data_annotation episode dir per line")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--out-root", type=Path, default=Path("episode_rollouts"))
    ap.add_argument("--method", required=True, help="label for this run (proact/procls/proreg or exp tag)")
    ap.add_argument("--oracle", action="store_true",
                    help="ORACLE mode: replay the recorded GT actions continuously through the "
                         "subgoal list (no server, no norm-stats needed) — a ground-truth reference "
                         "episode in the SAME layout as a policy run (method dir e.g. 'oracle').")
    ap.add_argument("--norm-stats", type=Path, default=None,
                    help="ckpt norm_stats.json (required for a policy run; omit for --oracle)")
    ap.add_argument("--subgoal-method", default=DEFAULT_SUBGOAL_METHOD)
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--horizon-mult", type=float, default=2.0)
    ap.add_argument("--max-steps-cap", type=int, default=400)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--no-zero-arm-in-base", action="store_true",
                    help="disable zeroing the arm eef delta in mobile/base mode (matches subtask_eval default ON)")
    # stop rule
    ap.add_argument("--stop-progress", type=float, default=0.95)
    ap.add_argument("--stop-eps", type=float, default=0.02)
    ap.add_argument("--stop-window", type=int, default=5)
    args = ap.parse_args()

    if args.oracle:
        client = None
        norm_stats = None
        if args.norm_stats is not None:   # optional: normalized state/action logging in oracle too
            norm_stats = json.loads(Path(args.norm_stats).read_text())
            norm_stats = norm_stats.get("norm_stats", norm_stats)
        print(f"ORACLE replay; rolling out episodes as method '{args.method}' into {args.out_root}")
    else:
        if args.norm_stats is None:
            raise SystemExit("--norm-stats is required for a policy run (or pass --oracle)")
        norm_stats = json.loads(Path(args.norm_stats).read_text())
        norm_stats = norm_stats.get("norm_stats", norm_stats)
        client = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)

    eps = [ln.strip() for ln in Path(args.episode_list).read_text().splitlines() if ln.strip()]
    idx = []
    idx_path = args.out_root / args.method / "index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_index():
        # Per-task timing rollup so we can read the avg wall-time per task at a glance.
        by_task: dict[str, list[float]] = {}
        for e in idx:
            t, s = e.get("task_name"), e.get("seconds")
            if t and s is not None:
                by_task.setdefault(t, []).append(s)
        task_timing = {t: dict(n=len(v), total_s=round(sum(v), 1), avg_s=round(sum(v) / len(v), 1))
                       for t, v in sorted(by_task.items())}
        done = [e for e in idx if e.get("seconds") is not None]
        total_s = round(sum(e["seconds"] for e in done), 1)
        idx_path.write_text(json.dumps(dict(
            eval_kind="episode", method=args.method, n_episodes=len(eps), n_done=len(idx),
            stop=dict(progress=args.stop_progress, eps=args.stop_eps, window=args.stop_window),
            total_seconds=total_s,
            avg_seconds_per_episode=round(total_s / len(done), 1) if done else None,
            task_timing=task_timing,
            episodes=idx), indent=1))

    t_run = time.time()
    for i, ep in enumerate(eps):
        print(f"=== [{i+1}/{len(eps)}] EPISODE {ep} ===", flush=True)
        ep_t0 = time.time()
        try:
            doc = eval_episode(Path(ep), client, args, args.out_root, args.method, norm_stats=norm_stats)
            secs = round(time.time() - ep_t0, 2)
            idx.append(dict(episode_id=doc["episode_id"], task_name=doc["task_name"],
                            episode_success=doc["episode_success"], n_advanced=doc["n_advanced"],
                            n_subgoals=doc["n_subgoals"], seconds=secs))
            print(f"    -> success={doc['episode_success']} advanced={doc['n_advanced']}/{doc['n_subgoals']} "
                  f"({secs}s)", flush=True)
        except Exception as e:
            traceback.print_exc()
            secs = round(time.time() - ep_t0, 2)
            idx.append(dict(episode=ep, error=repr(e), seconds=secs))
        # Write incrementally after EACH episode so a crash (OOM / bad ep) keeps completed timing.
        _write_index()
    print(f"wrote {idx_path}  (total {round(time.time() - t_run, 1)}s over {len(eps)} episodes)")


if __name__ == "__main__":
    main()
