"""Interactive LIVE rollout of a served RoboCasa System1 policy — the /interactive GUI.

Unlike subtask_eval.py / episode_eval.py (batch runners that write a rollout tree), this holds
ONE open sim in memory and lets a human drive it live from a browser: pick an episode, reset to
the EPISODE start or any ground-truth SUBGOAL start, then edit the task goal / current subgoal /
decision-transformer conditioning and watch the policy act step-by-step. Nothing is saved — every
run is ephemeral; close the page and it's gone.

Runs in the ROBOCASA micromamba env (needs robosuite/mujoco + openpi_client) and talks to a policy
already served by openpi scripts/serve_policy.py — exactly like subtask_eval.py:

    # OPENPI env, free GPU. --policy.config defaults to "auto".
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=6 \
    .venv/bin/python scripts/serve_policy.py --port 8010 policy:checkpoint \
        --policy.dir checkpoints/m0717-50k-bs512-v1__progcls_granfine_verbsimp/49999

    # ROBOCASA env — point at ONE OR MORE served policies; the GUI switches between them (each
    # carries its own norm_stats, so the normalized display always matches the selected ckpt).
    PY=/home/yinpei.dai/micromamba/envs/robocasa/bin/python
    $PY examples/robocasa/interactive_eval.py \
        --data-root /home/yinpei.dai/RoboAnnotator/data_annotation \
        --policy port=8011,ckpt=checkpoints/m0717-50k-bs512-v1__progcls_granfine_verbsimp/49999 \
        --policy port=8012,ckpt=checkpoints/m0717-50k-bs512-v3__progreg_granfine_verbsimp/49999 \
        --port 8093
    # or, for many policies: --policies-json policies.json
    # open http://<host>:8093/interactive

Reuses subtask_eval.py's faithful helpers (obs assembly, per-frame fields, progress reader, the
train/test conditioning contract) so what the GUI shows matches the batch evals byte-for-byte.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import threading
from pathlib import Path

import imageio
import numpy as np
from flask import Flask, abort, jsonify, request
from openpi_client import websocket_client_policy as _wcp

from robocasa.scripts.eval.subtask_common import (
    DEFAULT_SUBGOAL_METHOD, EpisodeSim, load_subgoals, Subgoal, estimated_length,
)
from robocasa.scripts.eval.subtask_env import (
    make_camera_env, raw_state_from_obs, images_from_obs, base_reference,
    lerobot_action_to_sim, settle_action_from_first, lean_state_from_raw16,
    quantile_norm, build_prompt_text, sim_action_to_lean11,
)
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

# Reuse the subtask driver's helpers verbatim (single source of truth).
import subtask_eval as SE
from subtask_eval import HORIZON, SIM_GRIP_IDX, SIM_CTRL_IDX

app = Flask(__name__)

# ---- process-wide config (set in main) ----
DATA_ROOT: Path | None = None
SUBGOAL_METHOD = DEFAULT_SUBGOAL_METHOD
RESIZE = 224
EPISODE_LIST: Path | None = None   # optional: seed the picker from a file (one ep rel-path/line)

# Registry of served policies the GUI can switch between (host:port + that ckpt's own norm_stats).
# Each: {name, host, port, norm_stats(loaded dict|None), progress_hint}. Selecting a policy in the
# GUI changes BOTH which server we infer from AND which norm_stats normalize the display — they must
# match the ckpt. Switching mid-rollout is fine (sim state is policy-independent), so you can compare
# what different variants do from the exact same state.
POLICIES: list[dict] = []

# Episode-picker cache. rglob over the full data_annotation tree (55k episodes) takes >60s, so
# build the list ONCE (in a background thread at startup) and memoize. If --episode-list is given
# we skip the walk entirely and just read those paths.
_EP_CACHE: dict = {"list": None}
_EP_LOCK = threading.Lock()

# One live session at a time (single user). Guarded by a lock — sim + websocket are not reentrant.
LOCK = threading.Lock()
SESH: "Session | None" = None


# ----------------------------------------------------------------------------------
def _jpeg_b64(arr) -> str:
    """uint8 HxWx3 -> base64 data URI (JPEG)."""
    import base64
    buf = io.BytesIO()
    imageio.v2.imwrite(buf, np.ascontiguousarray(arr), format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _norm_qs(norm_stats):
    """(q01s,q99s,q01a,q99a) from a policy's norm_stats dict, or Nones."""
    if norm_stats is None:
        return None, None, None, None
    return (np.asarray(norm_stats["state"]["q01"], np.float32),
            np.asarray(norm_stats["state"]["q99"], np.float32),
            np.asarray(norm_stats["actions"]["q01"], np.float32),
            np.asarray(norm_stats["actions"]["q99"], np.float32))


class Session:
    """A loaded episode + open sim + live rollout state. Ephemeral; not persisted."""

    def __init__(self, episode_dir: Path):
        self.ann = load_subgoals(episode_dir, SUBGOAL_METHOD)
        self.env = make_camera_env(self.ann.lerobot_dir)
        ld = Path(self.ann.lerobot_dir)
        self.states = LU.get_episode_states(ld, self.ann.episode_index)
        self.actions = LU.get_episode_actions(ld, self.ann.episode_index)
        self.ep_len = len(self.actions)
        # frame-0 base reference (captured ONCE; relative base pose for the whole episode)
        reset_to(self.env, dict(
            states=self.states[0],
            model=LU.get_episode_model_xml(ld, self.ann.episode_index),
            ep_meta=json.dumps(LU.get_episode_meta(ld, self.ann.episode_index))))
        obs0 = self.env._get_observations(force_update=True)
        self.base_pos_ref, self.base_yaw_ref = base_reference(obs0)
        self.sim = EpisodeSim(env=self.env, lerobot_dir=self.ann.lerobot_dir,
                              episode_index=self.ann.episode_index)
        self.sim._model_loaded = True   # model already loaded via reset_to above
        # live rollout state (set by reset())
        self.reset_mode = None
        self.sg: Subgoal | None = None       # the current subgoal being driven (mutable copy)
        self.anchor_imgs = None
        self.anchor_state = None
        self.task_goal = self.ann.instruction
        self.executed = 0
        self.exec_override: int | None = None
        self.gflag_override: str | None = None
        self.est_override: int | None = None
        self.quality = "Success"
        self.action_plan = collections.deque()
        self.last_cmd_grip = 0.0
        self.last_prog = None
        self.last_query = None
        self.zero_arm_in_base = True
        self.replan_steps = 16
        self.active_policy = 0 if POLICIES else None   # index into POLICIES

    def policy(self) -> dict | None:
        if self.active_policy is None or not (0 <= self.active_policy < len(POLICIES)):
            return None
        return POLICIES[self.active_policy]

    def client(self):
        """A websocket client to the ACTIVE policy (fresh per call — cheap, avoids stale sockets)."""
        p = self.policy()
        if p is None:
            raise RuntimeError("no policy selected")
        return _wcp.WebsocketClientPolicy(p["host"], p["port"])

    def norm_stats(self):
        p = self.policy()
        return p["norm_stats"] if p else None

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    # -- subgoal list for the UI --
    def subgoal_list(self):
        return [dict(child_index=s.child_index, milestone_index=s.milestone_index,
                     is_terminal=s.is_terminal, start=s.start, end=s.end,
                     primitive=s.primitive, subgoal=s.subgoal,
                     subgoal_detail=s.subgoal_detail, milestone_subgoal=s.milestone_subgoal)
                for s in self.ann.subgoals]

    # -- reset to episode-start or a subgoal-start --
    def reset(self, mode: str, subgoal_index: int | None):
        if mode == "episode":
            frame = 0
            # drive the FIRST subgoal's prompt from the episode start (System1 gets one subgoal at a time)
            sg_src = self.ann.subgoals[0] if self.ann.subgoals else None
        else:  # "subgoal"
            if subgoal_index is None or not (0 <= subgoal_index < len(self.ann.subgoals)):
                raise ValueError("subgoal_index out of range")
            sg_src = self.ann.subgoals[subgoal_index]
            frame = sg_src.start
        self.sim.reset_to_frame(frame)
        obs = self.env._get_observations(force_update=True)
        self.anchor_imgs = images_from_obs(obs)
        self.anchor_state = raw_state_from_obs(obs)
        # mutable working copy of the subgoal (task_goal/subgoal editable below)
        self.sg = Subgoal(**{**sg_src.__dict__}) if sg_src is not None else Subgoal(
            child_index=0, milestone_index=0, is_terminal=False, start=frame, end=frame,
            primitive="other", subgoal="", subgoal_detail="", milestone_subgoal="")
        self.reset_mode = mode
        self.task_goal = self.ann.instruction
        self.executed = 0
        self.exec_override = None
        self.gflag_override = None
        self.est_override = None
        self.quality = "Success"
        self.action_plan.clear()
        # settle: prime the gripper to the subtask's first recorded action (open/close intent)
        first_action = self.actions[min(self.sg.start, self.ep_len - 1)]
        self.last_cmd_grip = float(settle_action_from_first(first_action)[SIM_GRIP_IDX])
        self.last_prog = None
        self.last_query = None
        return self.state_detail(frame_note=f"reset:{mode}")

    # -- conditioning that actually reaches the model this step --
    def _span_len(self):
        return (self.sg.end - self.sg.start + 1) if self.sg else 1

    def _est_length(self):
        return self.est_override if self.est_override is not None else estimated_length(self._span_len())

    def _exec_step(self):
        return self.exec_override if self.exec_override is not None else self.executed

    def _gripper_flag(self):
        if self.gflag_override is not None:
            return self.gflag_override
        return "Close" if self.last_cmd_grip > 0 else "Open"

    # -- current live detail (state + last query + progress + frame) --
    def state_detail(self, frame_note="") -> dict:
        q01s, q99s, q01a, q99a = _norm_qs(self.norm_stats())
        obs = self.env._get_observations(force_update=True)
        anchor_lean = lean_state_from_raw16(self.anchor_state, self.base_pos_ref, self.base_yaw_ref)
        anchor_lean_norm = quantile_norm(anchor_lean, q01s, q99s) if q01s is not None else None
        # a "current" per-frame field dict using a ZERO action placeholder (no action executed
        # this readout) so the state/gripper fields render; action fields are informational.
        zero_a = np.zeros(12)
        fld = SE._frame_fields(
            obs, self.task_goal, self.sg, self.executed, self._budget(), self._est_length(),
            self.base_pos_ref, self.base_yaw_ref, anchor_lean, anchor_lean_norm, zero_a,
            self.last_prog, q01s, q99s, q01a, q99a, "live")
        fld["eef_pos_world"] = np.round(self.sim.eef_pose()["pos"], 4).tolist()
        fld["gripper_width"] = round(float(fld["cur_raw16"][14] - fld["cur_raw16"][15]), 4)
        fld["sim_check_success"] = bool(self.sim.check_full_success())
        return dict(
            frame=_jpeg_b64(SE._stacked_from_obs(obs)),
            note=frame_note,
            executed=self.executed,
            reset_mode=self.reset_mode,
            task_goal=self.task_goal,
            subgoal=self.sg.subgoal if self.sg else "",
            subgoal_detail=self.sg.subgoal_detail if self.sg else "",
            primitive=self.sg.primitive if self.sg else "",
            span=[self.sg.start, self.sg.end] if self.sg else None,
            span_len=self._span_len(),
            est_length=self._est_length(),
            gripper_flag=self._gripper_flag(),
            quality=self.quality,
            exec_override=self.exec_override,
            active_policy=self.active_policy,
            policy_name=(self.policy() or {}).get("name"),
            anchor_state_lean14=np.round(anchor_lean, 4).tolist(),
            anchor_state_lean14_norm=(np.round(anchor_lean_norm, 4).tolist()
                                      if anchor_lean_norm is not None else None),
            anchor_state_raw16=np.round(np.asarray(self.anchor_state), 4).tolist(),
            anchor_images={k: _jpeg_b64(v) for k, v in self.anchor_imgs.items()},
            fields=fld,
            last_query=self.last_query,
        )

    def _budget(self):
        # informational cap only (interactive has no hard budget); mirror the batch formula.
        return int(min(400, max(HORIZON, round(self._span_len() * 2.0))))

    # -- INFER a fresh chunk from the CURRENT (possibly edited) conditioning --
    def infer(self, client) -> dict:
        gripper_flag = self._gripper_flag()
        element = SE._obs_dict(
            self.env, base_pos_ref=self.base_pos_ref, base_yaw_ref=self.base_yaw_ref,
            anchor_imgs=self.anchor_imgs, anchor_state=self.anchor_state,
            subgoal_text=self.sg.subgoal, task_goal=self.task_goal,
            est_length=self._est_length(), executed_step=self._exec_step(),
            gripper_flag=gripper_flag, resize=RESIZE)
        result = client.infer(element)
        chunk = np.asarray(result["actions"])                        # (H,12) LeRobot order
        chunk_sim = np.stack([lerobot_action_to_sim(a) for a in chunk], axis=0)
        prog = SE._read_progress(result)
        if prog:
            self.last_prog = prog
        self.action_plan.clear()
        self.action_plan.extend(chunk_sim[: self.replan_steps])
        # build the displayed query (real server prompt + lean/norm chunk + progress)
        q01s, q99s, q01a, q99a = _norm_qs(self.norm_stats())
        chunk_lean = np.stack([sim_action_to_lean11(a) for a in chunk_sim], axis=0)
        chunk_lean_norm = quantile_norm(chunk_lean, q01a, q99a) if q01a is not None else None
        real_prompt = (result.get("prompt_text") if isinstance(result, dict) else None) \
            or build_prompt_text(self.task_goal, self.sg.subgoal, self.quality,
                                 self._est_length(), self._exec_step(), gripper_flag)
        self.last_query = dict(
            prompt=real_prompt, gripper_flag=gripper_flag, executed_step=int(self._exec_step()),
            chunk_lean11=np.round(chunk_lean, 4).tolist(),
            chunk_lean11_norm=(np.round(chunk_lean_norm, 4).tolist() if chunk_lean_norm is not None else None),
            chunk_progress=(prog.get("progress_chunk") if prog else None),
            replan_steps=self.replan_steps, horizon=len(chunk), progress=prog)
        return self.last_query

    # -- STEP the sim n times, replanning when the plan empties --
    def step(self, client, n: int, run_to_stop=False, stop_progress=0.95) -> dict:
        trace = []
        replans = 0
        stopped = None
        for _ in range(n):
            if not self.action_plan:
                self.infer(client)
                replans += 1
            a = np.asarray(self.action_plan.popleft(), dtype=np.float64)
            self.last_cmd_grip = float(a[SIM_GRIP_IDX])
            a_step = a
            if self.zero_arm_in_base and a[SIM_CTRL_IDX] > 0.0:
                a_step = a.copy(); a_step[0:6] = 0.0
            self.env.step(a_step)
            self.executed += 1
            trace.append(dict(
                executed=self.executed,
                action_raw12=np.round(a, 4).tolist(),
                gripper=("Close" if a[SIM_GRIP_IDX] > 0 else "Open"),
                progress=SE._progress_str(self.last_prog)))
            if run_to_stop and self.last_prog is not None:
                pk = self.last_prog.get("progress_kind")
                pv = (self.last_prog.get("progress_expected_frac") if pk == "classes"
                      else self.last_prog.get("progress_now"))
                if pv is not None and pv >= stop_progress:
                    stopped = f"progress>={stop_progress}"
                    break
        out = self.state_detail(frame_note=f"stepped {len(trace)}")
        out["trace"] = trace
        out["replans_this_call"] = replans
        out["stopped"] = stopped
        return out


# ----------------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------------
def _scan_episodes() -> list[dict]:
    """Build the selectable-episode list (episodes with the subgoal annotation).

    From --episode-list if given (fast; one episode rel-path or data_annotation path per line),
    else an rglob over DATA_ROOT (slow — cached by the caller). Each entry: episode_id/task_name/rel.
    """
    out = []
    if EPISODE_LIST is not None and EPISODE_LIST.is_file():
        for line in EPISODE_LIST.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            ep_dir = Path(line)
            if not ep_dir.is_absolute():
                ep_dir = DATA_ROOT / line
            epj = ep_dir / "episode.json"
            if not epj.is_file():
                continue
            try:
                ep = json.loads(epj.read_text())
                out.append(dict(episode_id=ep.get("episode_id"), task_name=ep.get("task_name"),
                                rel=str(ep_dir.resolve().relative_to(DATA_ROOT))))
            except Exception:
                continue
        return out
    if DATA_ROOT is not None:
        for p in sorted(DATA_ROOT.rglob("episode.json")):
            if (p.parent / "grounded_results" / SUBGOAL_METHOD / "result.json").exists():
                try:
                    ep = json.loads(p.read_text())
                    out.append(dict(episode_id=ep.get("episode_id"), task_name=ep.get("task_name"),
                                    rel=str(p.parent.relative_to(DATA_ROOT))))
                except Exception:
                    continue
    return out


def _episodes_cached(refresh=False) -> list[dict]:
    with _EP_LOCK:
        if refresh or _EP_CACHE["list"] is None:
            _EP_CACHE["list"] = _scan_episodes()
        return _EP_CACHE["list"]


@app.route("/api/episodes")
def api_episodes():
    """Episodes with the subgoal annotation (cached; ?refresh=1 to rescan)."""
    return jsonify(_episodes_cached(refresh=request.args.get("refresh") == "1"))


@app.route("/api/load", methods=["POST"])
def api_load():
    global SESH
    body = request.json or {}
    rel = body.get("rel")
    if not rel:
        abort(400)
    ep_dir = (DATA_ROOT / rel).resolve()
    if DATA_ROOT.resolve() not in ep_dir.parents:
        abort(403)
    with LOCK:
        if SESH is not None:
            SESH.close()
            SESH = None
        SESH = Session(ep_dir)
        # honor a policy picked in the GUI before the first load
        pi = body.get("policy_index")
        if pi is not None and 0 <= int(pi) < len(POLICIES):
            SESH.active_policy = int(pi)
        return jsonify(dict(
            episode_id=SESH.ann.episode_id, task_name=SESH.ann.task_name,
            instruction=SESH.ann.instruction, num_frames=SESH.ann.num_frames,
            fps=SESH.ann.fps, subgoals=SESH.subgoal_list(),
            active_policy=SESH.active_policy, policy_name=(SESH.policy() or {}).get("name")))


@app.route("/api/reset", methods=["POST"])
def api_reset():
    body = request.json or {}
    mode = body.get("mode", "episode")
    si = body.get("subgoal_index")
    with LOCK:
        if SESH is None:
            abort(409)
        return jsonify(SESH.reset(mode, si))


@app.route("/api/config", methods=["POST"])
def api_config():
    """Edit the live conditioning. Any provided field overrides; null clears an override."""
    body = request.json or {}
    with LOCK:
        if SESH is None or SESH.sg is None:
            abort(409)
        if "task_goal" in body and body["task_goal"] is not None:
            SESH.task_goal = str(body["task_goal"])
        if "subgoal" in body and body["subgoal"] is not None:
            SESH.sg.subgoal = str(body["subgoal"])
        if "quality" in body and body["quality"]:
            SESH.quality = str(body["quality"])
        if "est_length" in body:
            SESH.est_override = (int(body["est_length"]) if body["est_length"] not in (None, "") else None)
        if "executed_step" in body:
            SESH.exec_override = (int(body["executed_step"]) if body["executed_step"] not in (None, "") else None)
        if "gripper_flag" in body:
            SESH.gflag_override = (body["gripper_flag"] if body["gripper_flag"] in ("Open", "Close") else None)
        if "replan_steps" in body and body["replan_steps"]:
            SESH.replan_steps = max(1, int(body["replan_steps"]))
        if "zero_arm_in_base" in body:
            SESH.zero_arm_in_base = bool(body["zero_arm_in_base"])
        # editing conditioning invalidates the queued plan -> next step re-queries
        SESH.action_plan.clear()
        return jsonify(SESH.state_detail(frame_note="config"))


@app.route("/api/policies")
def api_policies():
    """List the served policies the GUI can switch between (name/host/port/progress hint)."""
    out = [dict(index=i, name=p["name"], host=p["host"], port=p["port"],
                has_norm_stats=p["norm_stats"] is not None, progress_hint=p.get("progress_hint"))
           for i, p in enumerate(POLICIES)]
    active = SESH.active_policy if SESH is not None else (0 if POLICIES else None)
    return jsonify(dict(policies=out, active=active))


@app.route("/api/select_policy", methods=["POST"])
def api_select_policy():
    """Switch the ACTIVE policy (index into /api/policies). Clears the queued plan so the next
    Step/Replan uses the newly-selected model + its norm_stats. Sim state is untouched, so you can
    compare what different variants do from the exact same state."""
    idx = (request.json or {}).get("index")
    with LOCK:
        if SESH is None:
            abort(409)
        if idx is None or not (0 <= int(idx) < len(POLICIES)):
            abort(400)
        SESH.active_policy = int(idx)
        SESH.action_plan.clear()   # next infer hits the new server
        return jsonify(SESH.state_detail(frame_note=f"policy:{POLICIES[int(idx)]['name']}"))


@app.route("/api/replan", methods=["POST"])
def api_replan():
    """Force a fresh inference NOW from the current conditioning; return the query (no stepping)."""
    with LOCK:
        if SESH is None or SESH.sg is None:
            abort(409)
        q = SESH.infer(SESH.client())
        out = SESH.state_detail(frame_note="replan")
        out["last_query"] = q
        return jsonify(out)


@app.route("/api/step", methods=["POST"])
def api_step():
    body = request.json or {}
    n = max(1, int(body.get("n", 1)))
    run_to_stop = bool(body.get("run_to_stop", False))
    with LOCK:
        if SESH is None or SESH.sg is None:
            abort(409)
        return jsonify(SESH.step(SESH.client(), n, run_to_stop=run_to_stop,
                                 stop_progress=float(body.get("stop_progress", 0.95))))


@app.route("/api/state")
def api_state():
    with LOCK:
        if SESH is None:
            return jsonify(dict(loaded=False))
        return jsonify(SESH.state_detail(frame_note="state"))


@app.route("/interactive")
@app.route("/")
def page():
    return INDEX_HTML


@app.route("/gui.js")
def gui_js():
    return app.response_class(GUI_JS, mimetype="application/javascript")


INDEX_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Interactive System1 Eval</title>
<style>
  *{box-sizing:border-box}html,body{height:100%;margin:0}
  body{font-family:system-ui,sans-serif;color:#1a1a1a;background:#fafafb;display:flex;flex-direction:column;overflow:hidden}
  #top{display:flex;gap:10px;align-items:center;padding:6px 12px;border-bottom:1px solid #ddd;background:#fff;flex-wrap:wrap}
  #top h1{font-size:14px;margin:0 6px 0 0}#top h1 b{color:#b0431c}
  select,input[type=text],input[type=number]{font-family:ui-monospace,monospace;font-size:12px;padding:2px 5px;border:1px solid #bbb;border-radius:4px}
  input[type=text]{min-width:260px}
  button{font-size:13px;padding:3px 11px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer}
  button:disabled{opacity:.4;cursor:default}button:not(:disabled):hover{background:#f0f3fa}
  button.go{background:#b0431c;color:#fff;border-color:#b0431c}
  .lbl{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.04em;margin-right:3px}
  #main{flex:1;display:grid;grid-template-columns:520px 1fr 1fr;gap:10px;padding:10px;overflow:hidden;min-height:0}
  .col{display:flex;flex-direction:column;gap:8px;overflow:auto;min-height:0}
  .card{border:1px solid #e3e3e8;border-radius:8px;background:#fff;padding:8px 10px}
  .card h2{font-size:11px;margin:0 0 6px;text-transform:uppercase;letter-spacing:.05em;color:#666}
  #frame,.anchimg{width:100%;border-radius:6px;background:#000;display:block}
  .anchrow{display:flex;gap:4px}.anchrow div{flex:1}.anchrow .lbl{display:block;text-align:center}
  .field{display:flex;justify-content:space-between;gap:8px;font-size:12px;padding:1px 0;border-bottom:1px dashed #eee}
  .field b{font-weight:600;color:#333}.field span{font-family:ui-monospace,monospace;color:#0a6}
  .row{display:flex;gap:6px;align-items:center;margin:3px 0;flex-wrap:wrap}
  pre{font-family:ui-monospace,monospace;font-size:11px;white-space:pre-wrap;background:#f7f7fa;border-radius:5px;padding:6px;margin:4px 0;max-height:200px;overflow:auto}
  .mono{font-family:ui-monospace,monospace;font-size:11px;color:#444}
  .chip{font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;border:1px solid #ddd;background:#f0f0f4;color:#555}
  .ok{color:#0a6}.bad{color:#c0392b}
  #status{font-size:11px;color:#888;margin-left:auto}
  table{border-collapse:collapse;width:100%;font-size:11px;font-family:ui-monospace,monospace}
  td,th{border:1px solid #eee;padding:1px 4px;text-align:right}th{background:#f7f7fa}
</style></head><body>
<div id="top">
  <h1>Interactive <b>System1</b></h1>
  <span class="lbl">policy</span><select id="policy" title="served policy to drive"></select>
  <span class="lbl">episode</span><select id="ep"></select>
  <button id="load">Load</button>
  <span class="lbl">reset</span>
  <select id="resetmode"><option value="episode">episode start</option><option value="subgoal">subgoal start</option></select>
  <select id="sgpick" style="display:none"></select>
  <button id="reset">Reset</button>
  <span id="status"></span>
</div>
<div id="main">
  <div class="col">
    <div class="card"><h2>Live view (scene_left | scene_right | wrist)</h2>
      <img id="frame"/>
      <div class="row">
        <button id="s1">Step 1</button>
        <input type="number" id="nstep" value="16" style="width:64px"><button id="sn">Step N</button>
        <button id="run" class="go">Run→stop</button>
        <button id="replan">Replan</button>
      </div>
      <div class="row"><span class="lbl">exec</span><span id="execn" class="mono">0</span>
        <span class="lbl">sim_success</span><span id="simok" class="mono">-</span>
        <span class="lbl">replans</span><span id="repl" class="mono">0</span></div>
    </div>
    <div class="card"><h2>Anchor (subgoal-start)</h2>
      <div class="anchrow">
        <div><span class="lbl">scene_left</span><img class="anchimg" id="a_scene_left"/></div>
        <div><span class="lbl">scene_right</span><img class="anchimg" id="a_scene_right"/></div>
        <div><span class="lbl">wrist</span><img class="anchimg" id="a_wrist"/></div>
      </div>
    </div>
  </div>

  <div class="col">
    <div class="card"><h2>Edit conditioning (applies on next Step / Replan)</h2>
      <div class="row"><span class="lbl">task goal</span><input type="text" id="taskgoal"></div>
      <div class="row"><span class="lbl">subgoal</span><input type="text" id="subgoal"></div>
      <div class="row">
        <span class="lbl">quality</span><input type="text" id="quality" value="Success" style="min-width:90px">
        <span class="lbl">est_len</span><input type="number" id="estlen" style="width:74px" placeholder="auto">
        <span class="lbl">exec_step</span><input type="number" id="execstep" style="width:64px" placeholder="auto">
        <span class="lbl">gripper</span>
        <select id="gflag"><option value="">auto</option><option>Open</option><option>Close</option></select>
      </div>
      <div class="row">
        <span class="lbl">replan_steps</span><input type="number" id="replansteps" value="16" style="width:64px">
        <label class="lbl"><input type="checkbox" id="zab" checked> zero-arm-in-base</label>
        <button id="apply">Apply edits</button>
        <button id="revert">Revert to original</button>
      </div>
      <div class="mono" id="subhint"></div>
    </div>
    <div class="card"><h2>Model prompt (server-tokenized)</h2>
      <pre id="prompt">— reset, then Step/Replan —</pre>
    </div>
    <div class="card"><h2>Progress</h2>
      <div id="progblock" class="mono">—</div>
    </div>
  </div>

  <div class="col">
    <div class="card"><h2>Current state</h2><div id="stateblock"></div></div>
    <div class="card"><h2>Predicted action chunk (lean-11)
      <label class="lbl" style="float:right"><input type="checkbox" id="normtoggle"> normalized</label></h2>
      <div id="chunkblock" style="overflow:auto"></div>
    </div>
    <div class="card"><h2>Last step trace</h2><div id="traceblock" class="mono"></div></div>
  </div>
</div>
<script src="/gui.js"></script>
</body></html>"""

GUI_JS = r"""
const $=s=>document.querySelector(s);
const S={loaded:false, subgoals:[], orig:{}, last:null};
function status(t){ $('#status').textContent=t; }

async function j(url, body){
  const opt = body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {};
  const r = await fetch(url,opt);
  if(!r.ok){ status('ERR '+r.status+' '+url); throw new Error(r.status); }
  return r.json();
}

async function loadEpisodes(){
  const eps = await j('/api/episodes');
  $('#ep').innerHTML = eps.map(e=>`<option value="${e.rel}">${e.episode_id}</option>`).join('');
  status(eps.length+' episodes');
}

async function loadPolicies(){
  const d = await j('/api/policies');
  if(!d.policies.length){ $('#policy').innerHTML='<option>(none served)</option>'; return; }
  $('#policy').innerHTML = d.policies.map(p=>
    `<option value="${p.index}">${p.name} :${p.port}${p.progress_hint?' ['+p.progress_hint+']':''}${p.has_norm_stats?'':' (no norm)'}</option>`).join('');
  if(d.active!=null) $('#policy').value=d.active;
}

$('#policy').onchange = async ()=>{
  if(!S.loaded){ return; }   // selection is remembered; applied on load/reset otherwise
  const d = await j('/api/select_policy',{index:+$('#policy').value});
  render(d,false); status('policy → '+(d.policy_name||$('#policy').value)+' (plan cleared)');
};

$('#load').onclick = async ()=>{
  status('loading env… (builds mujoco scene, ~10s)');
  const d = await j('/api/load',{rel:$('#ep').value, policy_index:(+$('#policy').value||0)});
  S.loaded=true; S.subgoals=d.subgoals;
  $('#sgpick').innerHTML = d.subgoals.map(s=>
    `<option value="${s.child_index}">#${s.child_index} ${s.primitive}: ${s.subgoal} [${s.start}-${s.end}]</option>`).join('');
  status(`loaded ${d.episode_id} — ${d.subgoals.length} subgoals, ${d.num_frames} frames`);
};

$('#resetmode').onchange = ()=>{ $('#sgpick').style.display = $('#resetmode').value==='subgoal'?'':'none'; };

$('#reset').onclick = async ()=>{
  if(!S.loaded){ status('load an episode first'); return; }
  const body={mode:$('#resetmode').value};
  if(body.mode==='subgoal') body.subgoal_index=+$('#sgpick').value;
  status('resetting…');
  render(await j('/api/reset',body), true);
  status('reset ('+body.mode+')');
};

function fillEdits(d){
  $('#taskgoal').value=d.task_goal||''; $('#subgoal').value=d.subgoal||'';
  $('#quality').value=d.quality||'Success';
  $('#estlen').placeholder='auto ('+d.est_length+')'; $('#estlen').value='';
  $('#execstep').placeholder='auto'; $('#execstep').value='';
  $('#gflag').value='';
  S.orig={task_goal:d.task_goal, subgoal:d.subgoal};
  $('#subhint').textContent=`primitive=${d.primitive} · detail: ${d.subgoal_detail||'—'} · span ${d.span?d.span.join('-'):'?'} (len ${d.span_len})`;
}

$('#apply').onclick = async ()=>{
  const b={ task_goal:$('#taskgoal').value, subgoal:$('#subgoal').value, quality:$('#quality').value,
    est_length:$('#estlen').value, executed_step:$('#execstep').value, gflag:$('#gflag').value,
    gripper_flag:$('#gflag').value, replan_steps:$('#replansteps').value, zero_arm_in_base:$('#zab').checked };
  render(await j('/api/config',b), false); status('edits applied (plan cleared)');
};
$('#revert').onclick = async ()=>{
  render(await j('/api/config',{task_goal:S.orig.task_goal, subgoal:S.orig.subgoal,
    est_length:'', executed_step:'', gripper_flag:''}), false);
  status('reverted to original');
};

$('#replan').onclick = async ()=>{ status('inferring…'); render(await j('/api/replan',{}), false); status('replanned'); };
$('#s1').onclick = ()=>doStep(1,false);
$('#sn').onclick = ()=>doStep(+$('#nstep').value||16,false);
$('#run').onclick = ()=>doStep(400,true);
async function doStep(n,run){ if(!S.loaded){return;} status(run?'running→stop…':'stepping '+n+'…');
  const d=await j('/api/step',{n:n, run_to_stop:run, stop_progress:0.95}); render(d,false);
  status((d.stopped?('stopped: '+d.stopped):('stepped '+ (d.trace?d.trace.length:0))) ); }

$('#normtoggle').onchange = ()=>{ if(S.last) renderChunk(S.last); };

function fld(label,val){ return `<div class="field"><b>${label}</b><span>${val}</span></div>`; }
function vec(a,dp=3){ return a? '['+a.map(x=>(+x).toFixed(dp)).join(', ')+']' : '—'; }

function render(d, isReset){
  S.last=d;
  $('#frame').src=d.frame;
  $('#execn').textContent=d.executed;
  const ss=d.fields? d.fields.sim_check_success : false;
  $('#simok').innerHTML = ss? '<span class=ok>True</span>':'<span class=bad>False</span>';
  if(d.replans_this_call!=null) $('#repl').textContent=d.replans_this_call;
  if(isReset){
    fillEdits(d);
    for(const k of ['scene_left','scene_right','wrist']) if(d.anchor_images&&d.anchor_images[k]) $('#a_'+k).src=d.anchor_images[k];
  }
  // prompt
  if(d.last_query&&d.last_query.prompt) $('#prompt').textContent=d.last_query.prompt;
  // progress
  const f=d.fields||{};
  $('#progblock').innerHTML = fld('progress', f.progress||'—') +
    (d.last_query&&d.last_query.progress? renderProg(d.last_query.progress):'');
  // state
  renderState(d);
  // chunk
  renderChunk(d);
  // trace
  if(d.trace){ $('#traceblock').innerHTML = '<table><tr><th>exec</th><th>grip</th><th>progress</th></tr>'+
    d.trace.map(t=>`<tr><td>${t.executed}</td><td>${t.gripper}</td><td>${t.progress}</td></tr>`).join('')+'</table>'; }
}

function renderProg(p){
  if(!p) return '';
  let s=fld('kind', p.progress_kind||'—');
  if(p.progress_kind==='classes') s+=fld('argmax class', p.progress_argmax+'/'+((p.progress_num_classes||10)-1))+
     fld('conf',(p.progress_conf||0).toFixed(3))+fld('E[frac]',(p.progress_expected_frac||0).toFixed(3));
  else if(p.progress_kind==='continuous') s+=fld('now',(p.progress_now||0).toFixed(3));
  else if(p.progress_kind==='action') s+=fld('now',(p.progress_now||0).toFixed(3))+fld('end',(p.progress_end||0).toFixed(3));
  return s;
}

function renderState(d){
  const f=d.fields||{};
  $('#stateblock').innerHTML =
    fld('subgoal', d.subgoal) + fld('task goal', d.task_goal) +
    fld('primitive', d.primitive) + fld('gripper_flag', d.gripper_flag) +
    fld('est_length', d.est_length) + fld('gripper_width', f.gripper_width) +
    fld('eef_pos_world', vec(f.eef_pos_world)) +
    fld('cur_lean14', vec(f.cur_lean)) +
    fld('cur_lean14_norm', vec(f.cur_lean_norm)) +
    fld('anchor_lean14', vec(d.anchor_state_lean14)) +
    fld('anchor_lean14_norm', vec(d.anchor_state_lean14_norm));
}

function renderChunk(d){
  const q=d.last_query; if(!q){ $('#chunkblock').textContent='— no query yet —'; return; }
  const norm=$('#normtoggle').checked;
  const chunk = norm? q.chunk_lean11_norm : q.chunk_lean11;
  if(!chunk){ $('#chunkblock').textContent = norm?'(no norm_stats)':'—'; return; }
  const head=['bx','by','bz','ctl','px','py','pz','rx','ry','rz','grip'];
  const prog=q.chunk_progress;
  let h='<table><tr><th>h</th>'+head.map(x=>`<th>${x}</th>`).join('')+(prog?'<th>prog</th>':'')+'</tr>';
  chunk.forEach((row,i)=>{ h+=`<tr><td>${i}</td>`+row.map(x=>`<td>${(+x).toFixed(2)}</td>`).join('')+
    (prog?`<td>${prog[i]!=null?(+prog[i]).toFixed(2):'-'}</td>`:'')+'</tr>'; });
  $('#chunkblock').innerHTML=h+'</table>';
}

loadPolicies();
loadEpisodes();
"""


def _norm_stats_from_ckpt(ckpt_dir: str | None):
    """Load a ckpt's baked assets/robocasa_system1/norm_stats.json -> norm_stats dict (or None)."""
    if not ckpt_dir:
        return None
    f = Path(ckpt_dir) / "assets" / "robocasa_system1" / "norm_stats.json"
    if f.is_file():
        try:
            return json.loads(f.read_text()).get("norm_stats")
        except Exception:
            return None
    return None


def _progress_hint(name: str) -> str | None:
    """Cheap label from the exp-name tag so the dropdown shows the head type."""
    for tok, hint in (("progcls", "cls"), ("progact", "act"), ("progreg", "reg"), ("prognone", "none")):
        if tok in name:
            return hint
    return None


def _parse_policy_flag(spec: str) -> dict:
    """Parse a --policy spec: comma-separated key=value with keys host,port,name,ckpt,norm_stats.
    Minimal form: 'port=8011,ckpt=checkpoints/<exp>/49999'  (name+norm_stats auto-derived from ckpt).
    'host' defaults to 127.0.0.1; 'name' defaults to the ckpt's exp-name (parent-of-step dir)."""
    kv = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--policy: bad token '{part}' (expected key=value)")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    if "port" not in kv:
        raise SystemExit(f"--policy '{spec}': 'port' is required")
    ckpt = kv.get("ckpt")
    name = kv.get("name") or (Path(ckpt).parent.name if ckpt else f"policy:{kv['port']}")
    ns = kv["norm_stats"] if "norm_stats" in kv else None
    norm_stats = (json.loads(Path(ns).read_text()).get("norm_stats") if ns
                  else _norm_stats_from_ckpt(ckpt))
    return dict(name=name, host=kv.get("host", "127.0.0.1"), port=int(kv["port"]),
                norm_stats=norm_stats, progress_hint=_progress_hint(name))


def main():
    global DATA_ROOT, SUBGOAL_METHOD, RESIZE, EPISODE_LIST, POLICIES
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True,
                   help="data_annotation root; episodes with the subgoal annotation are selectable")
    p.add_argument("--episode-list", type=Path, default=None,
                   help="optional file of episode paths (one per line) to populate the picker "
                        "INSTEAD of walking the whole data_annotation tree (~55k eps, >60s). "
                        "Each line = a data_annotation-relative or absolute episode dir.")
    p.add_argument("--subgoal-method", default=DEFAULT_SUBGOAL_METHOD)
    # ---- policies the GUI can switch between (host multiple servers on different ports) ----
    p.add_argument("--policy", action="append", default=[],
                   help="a served policy, repeatable. Comma-separated key=value: "
                        "port=<n>[,host=<h>][,name=<label>][,ckpt=<dir>][,norm_stats=<file>]. "
                        "ckpt auto-fills name + norm_stats (assets/robocasa_system1/norm_stats.json). "
                        "e.g. --policy port=8011,ckpt=checkpoints/m0717-...-v1__progcls.../49999")
    p.add_argument("--policies-json", type=Path, default=None,
                   help="JSON file: a list of policy objects {name,host,port,ckpt|norm_stats}. "
                        "Convenient for many (e.g. 16) policies. Merged with any --policy flags.")
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8093)
    args = p.parse_args()
    DATA_ROOT = args.data_root.resolve()
    EPISODE_LIST = args.episode_list.resolve() if args.episode_list is not None else None
    SUBGOAL_METHOD = args.subgoal_method
    RESIZE = args.resize_size
    # build the policy registry (json file first, then --policy flags)
    if args.policies_json is not None:
        for o in json.loads(args.policies_json.read_text()):
            name = o.get("name") or (Path(o["ckpt"]).parent.name if o.get("ckpt") else f"policy:{o['port']}")
            ns = o.get("norm_stats")
            norm_stats = (json.loads(Path(ns).read_text()).get("norm_stats") if ns
                          else _norm_stats_from_ckpt(o.get("ckpt")))
            POLICIES.append(dict(name=name, host=o.get("host", "127.0.0.1"), port=int(o["port"]),
                                 norm_stats=norm_stats, progress_hint=_progress_hint(name)))
    for spec in args.policy:
        POLICIES.append(_parse_policy_flag(spec))
    if not POLICIES:
        raise SystemExit("no policies configured — pass at least one --policy port=<n>[,ckpt=<dir>] "
                         "(or --policies-json <file>)")
    print(f"Interactive eval GUI on :{args.port}  ->  /interactive")
    print(f"  data-root: {DATA_ROOT}")
    print(f"  episodes:  {'--episode-list ' + str(EPISODE_LIST) if EPISODE_LIST else 'rglob (cached after first scan)'}")
    print(f"  policies ({len(POLICIES)}):")
    for i, pol in enumerate(POLICIES):
        print(f"    [{i}] {pol['name']}  {pol['host']}:{pol['port']}  "
              f"norm_stats={'yes' if pol['norm_stats'] else 'NO'}  hint={pol['progress_hint']}")
    # Warm the episode-picker cache in the background so the page's first /api/episodes is fast.
    threading.Thread(target=_episodes_cached, daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
