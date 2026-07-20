"""
Phase-2 SUBTASK evaluation of a served RoboCasa System1 (subgoal-conditioned pi0.5) policy.

Runs in the ROBOCASA micromamba env (needs robosuite/mujoco AND the openpi_client
websocket package). Talks to a policy served by openpi ``scripts/serve_policy.py``:

    # in the OPENPI env, on a free GPU (6 or 7; 80GB, low mem-fraction is plenty). --policy.config
    # defaults to "auto": the config is reconstructed from the checkpoint (config.json, else the
    # dir-name settings tag progcls/progreg/progact...), so no per-variant config is needed.
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=6 \
    .venv/bin/python scripts/serve_policy.py --port 8010 \
        policy:checkpoint \
        --policy.dir checkpoints/m0717-50k-bs512-v1/pi05_robocasa_system1/m0717-50k-bs512-v1__progcls_granfine_verbsimp/20000

For each fine (child) subgoal of an episode:
  1. reset the sim to the subgoal-START frame (exact recorded MuJoCo state), snapshot the
     ANCHOR (3 cam images + raw 16-d state) there,
  2. capture the frame-0 base reference (ONCE per episode) for relative base pose,
  3. roll out the policy CLOSED-LOOP with the child's ``subgoal`` text as the prompt (+ the
     full training conditioning: task_goal, Quality/Estimated Length/Executed Step, gripper
     flag, anchor images+state), for a step budget = span_len * horizon_mult (clamped),
     replanning every ``replan_steps``,
  4. evaluate the SAME per-subtask criterion as the oracle phase; also record progress head /
     progress-as-action output and action-similarity (MSE vs the oracle actions from the
     identical start state).

TRAIN/TEST-MATCH contract (see docs/robocasa_system1_training.md + robocasa_policy.py):
  - anchor = the subgoal-START frame; images + raw state, base pose relative to episode frame-0.
  - conditioning is DECISION-TRANSFORMER at inference: quality="Success" (desired),
    est_length = the annotated span-length bucket (`_estimated_length(span_len)`, frames),
    executed_step = env steps since the subgoal reset (0, then +1 per executed action),
    gripper_flag = "Close" if the last COMMANDED gripper dim > 0 else "Open" ("Open" @ step 0).
  - images sent as uint8 224x224 resize_with_pad; server converts to [-1,1] + normalizes.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
import traceback
from pathlib import Path

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

from robocasa.scripts.eval.subtask_common import (
    DEFAULT_SUBGOAL_METHOD,
    EpisodeSim,
    load_subgoals,
    Subgoal,
    estimated_length,
)
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
from robocasa.scripts.eval.subtask_overlay import render_header
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

HORIZON = 20            # policy action chunk length (pi05 System1)
# gripper_close index in ROBOSUITE-NATIVE order (after lerobot_action_to_sim): eef_pos[0:3],
# eef_rot[3:6], grip[6], base[7:11], control[11]. gripper_flag reads the last commanded value
# here, matching training (which read the raw command's gripper dim).
SIM_GRIP_IDX = 6
SIM_CTRL_IDX = 11       # control_mode / base_mode dim in robosuite-native order (>0 = mobile/base mode)


def _resize(img, size):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


def _obs_dict(env, *, base_pos_ref, base_yaw_ref, anchor_imgs, anchor_state,
              subgoal_text, task_goal, est_length, executed_step, gripper_flag, resize, obs=None):
    """Assemble the FULL infer dict the served RobocasaInputs transform expects.

    Includes anchor images (subgoal-start), anchor raw state, base ref, and the
    decision-transformer conditioning fields — exactly the training contract.
    """
    if obs is None:
        obs = env._get_observations(force_update=True)
    imgs = images_from_obs(obs)
    element = {
        # current views
        "observation/scene_left": _resize(imgs["scene_left"], resize),
        "observation/scene_right": _resize(imgs["scene_right"], resize),
        "observation/wrist": _resize(imgs["wrist"], resize),
        # anchor (subgoal-start) views — REQUIRED for the anchor config
        "observation/anchor_scene_left": _resize(anchor_imgs["scene_left"], resize),
        "observation/anchor_scene_right": _resize(anchor_imgs["scene_right"], resize),
        "observation/anchor_wrist": _resize(anchor_imgs["wrist"], resize),
        # state: RAW 16-d (server -> lean); anchor raw state + frame-0 base ref
        "observation/state": raw_state_from_obs(obs),
        "observation/anchor_state": np.asarray(anchor_state, dtype=np.float32),
        "observation/base_pos_ref": np.asarray(base_pos_ref, dtype=np.float32),
        "observation/base_yaw_ref": np.float32(base_yaw_ref),
        # prompt text + conditioning (all consumed by build_prompt)
        "prompt": subgoal_text,
        "task_goal": task_goal,
        "quality": "Success",
        "est_length": int(est_length),
        "executed_step": int(executed_step),
        "gripper_flag": gripper_flag,
    }
    return element


def _read_progress(result: dict) -> dict:
    """Normalize whatever progress signal the served policy returned into a summary.

    - progress-as-action: ``result["progress"]`` = per-horizon-step [0,1] array.
    - classes head: ``result["progress_head"]`` = softmax over K deciles -> argmax bucket +
      expected fraction.
    - continuous head: ``result["progress_head"]`` = scalar [0,1].
    """
    out = {}
    if "progress" in result:  # progress-as-action: per-horizon-step [0,1] chunk (12th action dim)
        p = np.asarray(result["progress"]).reshape(-1)
        out["progress_kind"] = "action"
        out["progress_now"] = float(p[0])
        out["progress_end"] = float(p[-1])
        out["progress_chunk"] = p.tolist()  # full per-step progress = the model's 12th action dim
    if "progress_head" in result:
        ph = np.asarray(result["progress_head"]).reshape(-1)
        if ph.size > 1:  # classes softmax
            k = ph.size
            out["progress_kind"] = "classes"
            out["progress_argmax"] = int(np.argmax(ph))
            out["progress_expected_frac"] = float(np.dot(np.arange(k), ph) / (k - 1))
        else:            # continuous scalar
            out["progress_kind"] = "continuous"
            out["progress_now"] = float(ph[0])
    return out


def _stacked_from_obs(obs):
    """3 cam views (native res) stacked side-by-side into one RGB frame, from an obs dict."""
    imgs = images_from_obs(obs)  # already top-down uint8
    return np.concatenate([imgs["scene_left"], imgs["scene_right"], imgs["wrist"]], axis=1)




def _write_steps_npz(sub_dir, doc_meta, step_records):
    """Write per-step logs as a compact steps.npz (float16 arrays, replan-only chunks) + a small
    steps_meta.json sidecar. ~15x smaller than the old per-step steps.json; fp16 is plenty for
    visualization. The GUI reconstructs per-step objects from these (client-side npz reader).

    npz arrays (all float16 unless noted), row i = step i:
      frame_step(int16), phase(0=settle,1=act int8), replanned(int8), sim_check_success(int8),
      cur_lean_norm(N,14), cur_raw16(N,16), action_raw12(N,12), oracle_action_raw12(N,12),
      eef_pos_world(N,3), action_norms(N,3)=[eef_pos,eef_rot,base], gripper_width(N),
      action_mse_vs_oracle(N), progress_scalar(N) [carried-forward [0,1] or NaN].
      replan-only: q_step(int16 K)=step indices with a query, q_chunk_norm(K,H,11),
      q_chunk_progress(K,H) [NaN if none].
    sidecar json: doc_meta + per-replan {prompt, progress_raw, gripper_flag, executed_step,
      replan_steps, horizon}.
    """
    n = len(step_records)
    f16 = lambda key, dim: np.asarray(
        [(s.get(key) if s.get(key) is not None else [0.0] * dim) for s in step_records], np.float16)
    arrs = {
        # frame_step can be negative (settle, e.g. -10) and >255 → int16; the 0/1 flags → uint8
        "frame_step": np.asarray([s["frame_step"] for s in step_records], np.int16),
        "phase": np.asarray([1 if s["phase"] == "act" else 0 for s in step_records], np.uint8),
        "replanned": np.asarray([1 if s.get("replanned") else 0 for s in step_records], np.uint8),
        "sim_check_success": np.asarray([1 if s.get("sim_check_success") else 0 for s in step_records], np.uint8),
        "cur_lean_norm": f16("cur_lean_norm", 14),
        "cur_lean": f16("cur_lean", 14),          # unnormalized (for the raw⇄norm toggle)
        "cur_raw16": f16("cur_raw16", 16),
        "action_raw12": f16("action_raw12", 12),
        "oracle_action_raw12": f16("oracle_action_raw12", 12),
        "eef_pos_world": f16("eef_pos_world", 3),
        "action_norms": np.asarray([[s.get("action_eef_pos_norm", 0), s.get("action_eef_rot_norm", 0),
                                     s.get("action_base_norm", 0)] for s in step_records], np.float16),
        "gripper_width": np.asarray([s.get("gripper_width", 0) for s in step_records], np.float16),
        "action_mse_vs_oracle": np.asarray([s.get("action_mse_vs_oracle", 0) for s in step_records], np.float16),
    }
    # carried-forward scalar progress (what the curve plots), computed here so the GUI needn't
    prog = np.full(n, np.nan, np.float16); last = np.nan
    for i, s in enumerate(step_records):
        pr = s.get("progress_raw")
        if pr:
            k = pr.get("progress_kind")
            v = (pr.get("progress_expected_frac") if k == "classes" else pr.get("progress_now"))
            if v is not None:
                last = v
        prog[i] = last
    arrs["progress_scalar"] = prog
    # replan-only chunks + prompt sidecar
    qi = [i for i, s in enumerate(step_records) if s.get("query")]
    # Store the FRAME step of each query (not the record index): the GUI reader keys q_step
    # by frame_step (rp_by_step / q_pos[fs]), and the settle warmup logs `settle_steps` records
    # with NEGATIVE frame_step before the main loop, so record index != frame_step whenever
    # settle_steps > 0 (the default). qi order is preserved, so q_chunk_* stay positionally aligned.
    arrs["q_step"] = np.asarray([int(step_records[i]["frame_step"]) for i in qi], np.int16)
    if qi:
        H = len(step_records[qi[0]]["query"]["chunk_lean11"])
        arrs["q_chunk_norm"] = np.asarray(
            [step_records[i]["query"].get("chunk_lean11_norm") or np.zeros((H, 11)) for i in qi], np.float16)
        arrs["q_chunk_lean"] = np.asarray(   # unnormalized chunk (real → env.step) for the toggle
            [step_records[i]["query"].get("chunk_lean11") or np.zeros((H, 11)) for i in qi], np.float16)
        prog_chunks = [step_records[i]["query"].get("chunk_progress") for i in qi]
        arrs["q_chunk_progress"] = np.asarray(
            [pc if pc is not None else [np.nan] * H for pc in prog_chunks], np.float16)
    np.savez_compressed(sub_dir / "steps.npz", **arrs)
    side = dict(doc_meta)
    side["replan"] = [dict(step=int(step_records[i]["frame_step"]),
                           prompt=step_records[i]["query"].get("prompt"),
                           gripper_flag=step_records[i]["query"].get("gripper_flag"),
                           executed_step=step_records[i]["query"].get("executed_step"),
                           replan_steps=step_records[i]["query"].get("replan_steps"),
                           horizon=step_records[i]["query"].get("horizon"),
                           progress_raw=step_records[i].get("progress_raw")) for i in qi]
    (sub_dir / "steps_meta.json").write_text(json.dumps(side))


def _frame_fields(obs, task_goal, sg, step, budget, est_len, base_pos_ref, base_yaw_ref,
                  anchor_lean, anchor_lean_norm, action_sim, prog, q01s, q99s, q01a, q99a, phase):
    """Compute the per-frame field dict rendered on the video AND saved to the paired JSON.

    Takes a PRE-FETCHED obs dict (one render per step; the same obs supplies both the state
    and the camera images). Single source of truth so the overlay and the sidecar never diverge.
    """
    from robocasa.scripts.eval.subtask_env import sim_action_to_lean11
    cur_raw16 = raw_state_from_obs(obs)
    cur_lean = lean_state_from_raw16(cur_raw16, base_pos_ref, base_yaw_ref)
    cur_lean_norm = quantile_norm(cur_lean, q01s, q99s) if q01s is not None else None
    action_lean = sim_action_to_lean11(action_sim)
    action_norm = quantile_norm(action_lean, q01a, q99a) if q01a is not None else None
    # per-component action norms (magnitude of the movement command) — quick health read
    eef_pos_norm = float(np.linalg.norm(action_sim[0:3]))
    eef_rot_norm = float(np.linalg.norm(action_sim[3:6]))
    base_norm = float(np.linalg.norm(action_sim[7:11]))
    gflag = "Close" if float(action_sim[6]) > 0 else "Open"
    # Per-step log: only fields the GUI reads, rounded to 4 dp (anchor_* / action_lean11 /
    # action_norm11 dropped — anchor state is constant per-subtask in the doc header; the chunk
    # already carries lean/norm). Halves the per-step payload.
    r4 = lambda a: np.round(np.asarray(a), 4).tolist()
    fields = dict(
        phase=phase, task_goal=task_goal, subgoal=sg.subgoal, primitive=sg.primitive,
        frame_step=step, budget=budget, est_length=est_len, gripper_flag=gflag,
        cur_lean=r4(cur_lean),
        cur_lean_norm=(r4(cur_lean_norm) if cur_lean_norm is not None else None),
        cur_raw16=r4(cur_raw16),
        action_raw12=r4(action_sim),
        action_eef_pos_norm=round(eef_pos_norm, 4), action_eef_rot_norm=round(eef_rot_norm, 4),
        action_base_norm=round(base_norm, 4),
        progress=_progress_str(prog), progress_raw=prog,
    )
    return fields




def _progress_str(prog: dict | None) -> str:
    if not prog:
        return "-"
    k = prog.get("progress_kind")
    if k == "classes":
        return f"class {prog['progress_argmax']}/9 (E[frac]={prog['progress_expected_frac']:.2f})"
    if k == "continuous":
        return f"{prog['progress_now']:.3f} (continuous)"
    if k == "action":
        return f"now={prog['progress_now']:.3f} end={prog['progress_end']:.3f} (as-action)"
    return "-"


def rollout_subgoal(env, sim, client, sim_states, sim_actions, sg: Subgoal, *,
                    base_pos_ref, base_yaw_ref, anchor_imgs, anchor_state,
                    task_goal, resize, replan_steps, horizon_mult, max_steps_cap,
                    settle_steps=10, norm_stats=None, ep_len=None, zero_arm_in_base=True):
    """Reset to the subgoal start, gripper-settle, roll out the policy, and LOG EVERYTHING.

    No success criterion is applied here (Gemini judges later). We just faithfully record,
    per step: the executed action (raw12 / lean11 / normalized / component norms), the ORACLE
    reference action at the same step + its MSE, the current state (raw16 / lean14 / norm),
    the anchor state, the progress prediction, the live EEF pose, gripper width, and RoboCasa's
    own ``_check_success`` flag (kept only as an auxiliary signal, NOT our verdict).

    Returns clean 3-view frames, overlay frames, and per-step records.
    """
    reset_to(env, dict(states=sim_states[sg.start]))
    span_len = sg.end - sg.start + 1
    est_len = estimated_length(span_len)
    ep_len = ep_len if ep_len is not None else len(sim_actions)
    if client is None:
        # ORACLE: replay exactly the recorded subgoal span (stop at the true end), no extra budget.
        budget = span_len
    else:
        # policy: give slack beyond the recorded span (2x per the eval design), clamped.
        budget = int(min(max_steps_cap, max(HORIZON, round(span_len * horizon_mult))))

    q01s = q99s = q01a = q99a = None
    if norm_stats is not None:
        q01s = np.asarray(norm_stats["state"]["q01"], dtype=np.float32)
        q99s = np.asarray(norm_stats["state"]["q99"], dtype=np.float32)
        q01a = np.asarray(norm_stats["actions"]["q01"], dtype=np.float32)
        q99a = np.asarray(norm_stats["actions"]["q99"], dtype=np.float32)
    anchor_lean = lean_state_from_raw16(anchor_state, base_pos_ref, base_yaw_ref)
    anchor_lean_norm = quantile_norm(anchor_lean, q01s, q99s) if q01s is not None else None

    clean_frames, step_records = [], []
    timers = {"render_s": 0.0, "sim_s": 0.0, "infer_s": 0.0, "n_render": 0, "n_sim": 0, "n_infer": 0}

    def _log_step(step, phase, action_sim, prog, replanned, query=None):
        # ONE render per step: this obs supplies BOTH the logged state and the camera frame.
        _t = time.monotonic()
        obs = env._get_observations(force_update=True)
        timers["render_s"] += time.monotonic() - _t; timers["n_render"] += 1
        fld = _frame_fields(obs, task_goal, sg, step, budget, est_len, base_pos_ref, base_yaw_ref,
                            anchor_lean, anchor_lean_norm, action_sim, prog, q01s, q99s, q01a, q99a, phase)
        # oracle reference action at this step (what the human did from the same start), + MSE
        oracle_idx = min(sg.start + max(step, 0), ep_len - 1)
        oracle_a = np.asarray(sim_actions[oracle_idx], dtype=np.float64)
        fld["oracle_action_raw12"] = np.round(oracle_a, 4).tolist()
        fld["action_mse_vs_oracle"] = round(float(np.mean((np.asarray(action_sim) - oracle_a) ** 2)), 6)
        # auxiliary sim signals (NOT our success verdict). eef_pose/check_success read live sim
        # data (no extra render); check_full_success calls update_state (cheap, no camera).
        fld["eef_pos_world"] = np.round(sim.eef_pose()["pos"], 4).tolist()
        fld["gripper_width"] = round(float(fld["cur_raw16"][14] - fld["cur_raw16"][15]), 4)
        fld["sim_check_success"] = bool(sim.check_full_success())
        fld["replanned"] = bool(replanned)
        # On a replan step, attach the FULL model query (prompt + displayed action chunks).
        if query is not None:
            fld["query"] = query
        clean_frames.append(_stacked_from_obs(obs))   # clean video only (overlay dropped)
        step_records.append(fld)

    # --- gripper-settle warmup: zero motion, keep control_mode + gripper of the 1st action ---
    settle = settle_action_from_first(sim_actions[sg.start])
    for si in range(settle_steps):
        _log_step(-settle_steps + si, "settle", settle, None, replanned=False)
        env.step(settle)

    from robocasa.scripts.eval.subtask_env import (
        sim_action_to_lean11, build_prompt_text, sim_to_lerobot12 as _sim_to_lerobot12)

    action_plan = collections.deque()
    progress_trace = []
    last_cmd_grip = float(settle[SIM_GRIP_IDX])
    last_prog = None
    first_chunk_mse = None
    executed = 0
    query = None
    while executed < budget:
        replanned = False
        query = None
        if not action_plan:
            replanned = True
            gripper_flag = "Close" if last_cmd_grip > 0 else "Open"
            if client is None:
                # ORACLE mode: "predict" the recorded action chunk from this step (already
                # robosuite-native order). Gives the same structured logs/videos as a policy
                # run, for a ground-truth reference in the GUI. No server, no images sent.
                oi = min(sg.start + executed, ep_len - 1)
                chunk_sim = np.stack([sim_actions[min(oi + j, ep_len - 1)]
                                      for j in range(HORIZON)], axis=0)
                # present the raw chunk in LeRobot order for the query log (inverse reorder)
                chunk = np.stack([_sim_to_lerobot12(a) for a in chunk_sim], axis=0)
                result = {}
                prog = None
            else:
                element = _obs_dict(
                    env, base_pos_ref=base_pos_ref, base_yaw_ref=base_yaw_ref,
                    anchor_imgs=anchor_imgs, anchor_state=anchor_state,
                    subgoal_text=sg.subgoal, task_goal=task_goal, est_length=est_len,
                    executed_step=executed, gripper_flag=gripper_flag, resize=resize)
                _t = time.monotonic()
                result = client.infer(element)
                timers["infer_s"] += time.monotonic() - _t; timers["n_infer"] += 1
                chunk = np.asarray(result["actions"])  # (H,12) LEROBOT order, unnormalized
                chunk_sim = np.stack([lerobot_action_to_sim(a) for a in chunk], axis=0)
                prog = _read_progress(result)
            if prog:
                prog["at_step"] = executed
                progress_trace.append(prog)
                last_prog = prog
            if first_chunk_mse is None:  # first-chunk similarity vs oracle from the same start
                k = min(len(chunk_sim), span_len)
                if k > 0:
                    first_chunk_mse = float(np.mean(
                        (chunk_sim[:k] - sim_actions[sg.start:sg.start + k]) ** 2))
            action_plan.extend(chunk_sim[:replan_steps])
            # capture the FULL query for the GUI (prompt text + the whole action chunk)
            chunk_lean = np.stack([sim_action_to_lean11(a) for a in chunk_sim], axis=0)
            chunk_lean_norm = (quantile_norm(chunk_lean, q01a, q99a) if q01a is not None else None)
            prog_full = _read_progress(result) or None
            query = dict(
                prompt=build_prompt_text(task_goal, sg.subgoal, "Success", est_len, executed, gripper_flag),
                gripper_flag=gripper_flag, executed_step=executed,
                # Only the chunks the GUI displays: lean-11 (real, → env.step after unnorm) + its
                # normalized form (model's direct output). Dropped chunk_raw_lerobot12 / chunk_sim12
                # (never rendered) to shrink steps.json ~1/3.
                chunk_lean11=np.round(chunk_lean, 4).tolist(),
                chunk_lean11_norm=(np.round(chunk_lean_norm, 4).tolist() if chunk_lean_norm is not None else None),
                # progact: the model's 12th action dim = per-step progress (part of the action output)
                chunk_progress=(prog_full.get("progress_chunk") if prog_full else None),
                replan_steps=replan_steps, horizon=len(chunk),
                progress=prog_full)
        a = np.asarray(action_plan.popleft(), dtype=np.float64)  # robosuite-native order
        _log_step(executed, "act", a, last_prog, replanned, query=query)   # log the RAW model action
        last_cmd_grip = float(a[SIM_GRIP_IDX])
        # In MOBILE/base mode (control_mode > 0) teleop forces the arm eef delta to ZERO
        # (robosuite devices/device.py: base_mode -> arm_norm_delta = zeros(6)); the whole-body IK
        # otherwise INTEGRATES the model's small nonzero eef residual into a large arm drift. Apply
        # the same convention to the STEPPED action (log keeps the raw so the residual stays visible).
        a_step = a
        if zero_arm_in_base and a[SIM_CTRL_IDX] > 0.0:
            a_step = a.copy(); a_step[0:6] = 0.0   # eef_pos[0:3] + eef_rot[3:6] -> 0
        _t = time.monotonic()
        env.step(a_step)
        timers["sim_s"] += time.monotonic() - _t; timers["n_sim"] += 1
        executed += 1

    # end-of-rollout final state signals (post last step)
    final_success = bool(sim.check_full_success())
    timing = dict(
        render_fps=round(timers["n_render"] / timers["render_s"], 1) if timers["render_s"] else None,
        sim_fps=round(timers["n_sim"] / timers["sim_s"], 1) if timers["sim_s"] else None,
        infer_ms=round(1000 * timers["infer_s"] / timers["n_infer"], 1) if timers["n_infer"] else None,
        render_s=round(timers["render_s"], 2), sim_s=round(timers["sim_s"], 2),
        infer_s=round(timers["infer_s"], 2), n_render=timers["n_render"], n_infer=timers["n_infer"])
    return dict(steps=executed, budget=budget, est_length=est_len, settle_steps=settle_steps,
                span_len=span_len, first_chunk_action_mse=first_chunk_mse,
                mean_step_action_mse=float(np.mean([r["action_mse_vs_oracle"] for r in step_records
                                                    if r["phase"] == "act"])) if step_records else None,
                progress_final=progress_trace[-1] if progress_trace else None,
                progress_trace=progress_trace, n_progress_reads=len(progress_trace),
                sim_success_final=final_success,
                sim_success_any=any(r["sim_check_success"] for r in step_records),
                timing=timing,
                _clean_frames=clean_frames,
                _step_records=step_records)


def eval_episode(episode_dir: Path, client, args, out_root: Path, method: str,
                 norm_stats: dict | None = None) -> dict:
    """Roll out every fine subgoal of an episode and write structured, faithful logs.

    Output layout (per the "structured dir" request):
      <out_root>/<method>/<episode_flat>/child<NN>_<primitive>/
          clean.mp4     -- 3-view stacked, NO overlay (what we send to Gemini)
          overlay.mp4   -- clean + the text header (for human inspection)
          steps.json    -- per-step faithful log (actions/state/progress/oracle-ref/sim-signals)
    NO success criterion is computed here; Gemini judges finish separately.
    """
    ann = load_subgoals(episode_dir, args.subgoal_method)
    env = make_camera_env(ann.lerobot_dir)
    flat = ann.episode_id.replace("/", "__")
    ep_out = out_root / method / flat
    ep_out.mkdir(parents=True, exist_ok=True)
    fps = int(round(ann.fps))
    try:
        ld = Path(ann.lerobot_dir)
        states = LU.get_episode_states(ld, ann.episode_index)
        actions = LU.get_episode_actions(ld, ann.episode_index)
        init = dict(states=states[0],
                    model=LU.get_episode_model_xml(ld, ann.episode_index),
                    ep_meta=json.dumps(LU.get_episode_meta(ld, ann.episode_index)))
        reset_to(env, init)
        obs0 = env._get_observations(force_update=True)
        base_pos_ref, base_yaw_ref = base_reference(obs0)

        sim = EpisodeSim(env=env, lerobot_dir=ann.lerobot_dir, episode_index=ann.episode_index)
        sim._model_loaded = True

        records = []
        for sg in ann.subgoals:
            # anchor snapshot at subgoal START: 3 cam images + raw 16-d state (states-only reset).
            sim.reset_to_frame(sg.start)
            anchor_obs = env._get_observations(force_update=True)
            anchor_imgs = images_from_obs(anchor_obs)
            anchor_state = raw_state_from_obs(anchor_obs)
            roll = rollout_subgoal(
                env, sim, client, states, actions, sg,
                base_pos_ref=base_pos_ref, base_yaw_ref=base_yaw_ref,
                anchor_imgs=anchor_imgs, anchor_state=anchor_state,
                task_goal=ann.instruction, resize=args.resize_size,
                replan_steps=args.replan_steps, horizon_mult=args.horizon_mult,
                max_steps_cap=args.max_steps_cap, settle_steps=args.settle_steps,
                norm_stats=norm_stats, ep_len=len(actions),
                zero_arm_in_base=not args.no_zero_arm_in_base)

            clean = roll.pop("_clean_frames", [])
            roll.pop("_overlay_frames", None)   # overlay video no longer used by the GUI (info is in panels)
            step_records = roll.pop("_step_records", [])
            sub_dir = ep_out / f"child{sg.child_index:02d}_{sg.primitive}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            if clean:
                # Full-res 256x768 video (kept for clarity — the steps.npz slimming already made
                # the per-subtask payload tiny). GOP=1 (every frame a keyframe) so the GUI seeks to
                # any exact frame; +faststart for quick web load.
                imageio.mimwrite(sub_dir / "clean.mp4", clean, fps=fps, codec="libx264",
                                 macro_block_size=1, output_params=["-g", "1", "-movflags", "+faststart"])
            # save the anchor (subgoal-start) views for the GUI (full res)
            anchor_img_files = {}
            for cam_key, img in anchor_imgs.items():
                fn = f"anchor_{cam_key}.jpg"
                imageio.imwrite(sub_dir / fn, np.ascontiguousarray(img))
                anchor_img_files[cam_key] = fn
            # anchor state (raw16 / lean14 / normalized) at the subgoal-start frame
            anchor_lean = lean_state_from_raw16(anchor_state, base_pos_ref, base_yaw_ref)
            anchor_lean_norm = (quantile_norm(anchor_lean,
                                np.asarray(norm_stats["state"]["q01"]), np.asarray(norm_stats["state"]["q99"]))
                                if norm_stats else None)
            doc_meta = dict(
                method=method, episode_id=ann.episode_id, task_name=ann.task_name,
                child_index=sg.child_index, milestone_index=sg.milestone_index,
                is_terminal=sg.is_terminal, primitive=sg.primitive,
                subgoal=sg.subgoal, subgoal_detail=sg.subgoal_detail,
                milestone_subgoal=sg.milestone_subgoal, task_goal=ann.instruction,
                span=[sg.start, sg.end], fps=ann.fps,
                settle_steps=roll["settle_steps"], budget=roll["budget"],
                clean_video="clean.mp4", n_steps=len(step_records),
                anchor_images=anchor_img_files,
                anchor_state_raw16=np.round(np.asarray(anchor_state), 4).tolist(),
                anchor_state_lean14=np.round(anchor_lean, 4).tolist(),
                anchor_state_lean14_norm=(np.round(anchor_lean_norm, 4).tolist() if anchor_lean_norm is not None else None),
                base_pos_ref=np.round(np.asarray(base_pos_ref), 4).tolist(), base_yaw_ref=round(float(base_yaw_ref), 4),
                summary={k: roll[k] for k in ("steps", "budget", "span_len", "est_length",
                        "first_chunk_action_mse", "mean_step_action_mse",
                        "sim_success_final", "sim_success_any", "n_progress_reads")},
                timing=roll["timing"], progress_final=roll["progress_final"])
            # Compact per-step log: steps.npz (fp16 arrays + uint8 flags) + steps_meta.json sidecar.
            _write_steps_npz(sub_dir, doc_meta, step_records)

            rec = dict(child_index=sg.child_index, milestone_index=sg.milestone_index,
                       is_terminal=sg.is_terminal, span=[sg.start, sg.end],
                       primitive=sg.primitive, subgoal=sg.subgoal,
                       subgoal_detail=sg.subgoal_detail, milestone_subgoal=sg.milestone_subgoal,
                       out_dir=str(sub_dir.relative_to(out_root)), timing=roll["timing"],
                       **{k: roll[k] for k in ("steps", "budget", "span_len", "est_length",
                          "first_chunk_action_mse", "mean_step_action_mse",
                          "sim_success_final", "sim_success_any", "progress_final")})
            records.append(rec)
        ep_doc = dict(method=method, episode_id=ann.episode_id, task_name=ann.task_name,
                      instruction=ann.instruction, n_subgoals=len(ann.subgoals),
                      subgoals=records, error=None)
        (ep_out / "episode.json").write_text(json.dumps(ep_doc, indent=1))
        return ep_doc
    finally:
        try:
            env.close()
        except Exception:
            pass


def _discover(data_root: Path, method: str) -> list[Path]:
    return [p.parent for p in sorted(data_root.rglob("episode.json"))
            if (p.parent / "grounded_results" / method / "result.json").exists()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--episode-list", type=Path, default=None)
    p.add_argument("--subgoal-method", default=DEFAULT_SUBGOAL_METHOD)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--horizon-mult", type=float, default=2.0,
                   help="policy step budget = span_len * this (clamped to --max-steps-cap)")
    p.add_argument("--max-steps-cap", type=int, default=400)
    p.add_argument("--settle-steps", type=int, default=10,
                   help="zero-motion gripper-settle steps after reset before the policy acts")
    p.add_argument("--no-zero-arm-in-base", action="store_true",
                   help="disable the teleop convention of zeroing the arm eef delta when "
                        "control_mode>0 (mobile/base mode). Default: zero it (matches training).")
    p.add_argument("--norm-stats", type=Path, default=None,
                   help="checkpoint norm_stats.json (for normalized state/action logging). "
                        "e.g. <ckpt>/assets/robocasa_system1/norm_stats.json")
    p.add_argument("--out-root", type=Path, required=True,
                   help="structured output root: <out-root>/<method>/<episode>/child<NN>_<prim>/")
    p.add_argument("--method", required=True,
                   help="method label = subdir under out-root (e.g. classes / progact / reg / oracle)")
    p.add_argument("--oracle", action="store_true",
                   help="ORACLE mode: replay recorded actions instead of a served policy (no "
                        "server needed). Same structured logs/videos, as a ground-truth reference.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    norm_stats = None
    if args.norm_stats is not None:
        norm_stats = json.loads(args.norm_stats.read_text()).get("norm_stats")

    if args.episode_dir:
        episodes = [args.episode_dir]
    elif args.episode_list:
        episodes = [Path(l.strip()) for l in args.episode_list.read_text().splitlines() if l.strip()]
    elif args.data_root:
        episodes = _discover(args.data_root, args.subgoal_method)
        if args.limit:
            episodes = episodes[: args.limit]
    else:
        raise SystemExit("pass --episode-dir / --episode-list / --data-root")

    if args.oracle:
        client = None
        print(f"ORACLE replay; rolling out {len(episodes)} episode(s) as method "
              f"'{args.method}' into {args.out_root}")
    else:
        client = _wcp.WebsocketClientPolicy(args.host, args.port)
        print(f"Connected to policy at {args.host}:{args.port}; rolling out {len(episodes)} episode(s) "
              f"as method '{args.method}' into {args.out_root}")

    results = []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        ep_t0 = time.time()
        try:
            r = eval_episode(ep, client, args, out_root=args.out_root, method=args.method,
                             norm_stats=norm_stats)
        except Exception:
            r = dict(episode_id=str(ep), error=traceback.format_exc(), subgoals=[])
        secs = round(time.time() - ep_t0, 2)
        results.append(dict(episode_id=r.get("episode_id"), n_subgoals=len(r.get("subgoals", [])),
                            error=r.get("error"), seconds=secs))
        n = len(r.get("subgoals", []))
        # report the aux sim-success signal (NOT our verdict) just as a rough progress cue
        n_simsucc = sum(1 for s in r.get("subgoals", []) if s.get("sim_success_final"))
        tm = next((s.get("timing") for s in r.get("subgoals", []) if s.get("timing")), None)
        tstr = f" | render {tm['render_fps']}fps, sim {tm['sim_fps']}fps, infer {tm['infer_ms']}ms" if tm else ""
        tag = "ERR" if r.get("error") else f"{n} subgoals rolled out ({n_simsucc} sim_success_final)"
        print(f"[{i+1}/{len(episodes)}] {r.get('episode_id')} :: {tag} ({secs}s){tstr}", flush=True)

    index = dict(method=args.method, subgoal_method=args.subgoal_method,
                 host=args.host, port=args.port, replan_steps=args.replan_steps,
                 horizon_mult=args.horizon_mult, settle_steps=args.settle_steps,
                 n_episodes=len(results), wall_seconds=round(time.time() - t0, 1),
                 episodes=results)
    idx_path = args.out_root / args.method / "index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"\nWrote per-method index to {idx_path}")


if __name__ == "__main__":
    main()
