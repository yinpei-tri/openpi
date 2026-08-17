"""Stateful human-in-the-loop controller for live RoboCasa System2 + System1 evaluation.

This module contains no Flask routes.  ``human_interactive_gui.py`` owns HTTP/UI concerns while
this file owns the single live simulator, model calls, rule application, durable attempts, and
rollback.  One process intentionally serves one operator/session: MuJoCo and the websocket policy
client are not re-entrant.

Rollback is soft-reset-and-replay, not a flattened-state jump. RoboCasa fixtures keep Python-side
latches and counters (microwave ``turned_on``, toaster timers, etc.) that MuJoCo's state vector does
not contain. We soft-reset controller/observable state, restore simple environment and fixture
attributes plus the official initial MuJoCo state, then replay the exact controls handed to
``env.step``. This reconstructs the complete turn history without reloading an EGL model.
"""

# RoboCasa's public evaluation contract itself uses env._get_observations(), env._check_success(),
# and subtask_eval._stacked_from_obs(); these are intentional parity calls, not accidental leakage.
# ruff: noqa: SLF001

from __future__ import annotations

import copy
from collections import deque
import dataclasses
import hashlib
import json
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from typing import Any
import uuid

import numpy as np

try:  # script execution (examples/robocasa is sys.path[0])
    from human_interactive_store import HumanInteractiveStore
    from human_interactive_store import TargetEpisode
except ImportError:  # package-style imports in tests
    from examples.robocasa.human_interactive_store import HumanInteractiveStore
    from examples.robocasa.human_interactive_store import TargetEpisode


@dataclasses.dataclass(frozen=True)
class InteractiveConfig:
    dataset_root: Path
    results_root: Path
    s1_host: str = "127.0.0.1"
    s1_port: int = 8060
    s1_checkpoint: str | None = None
    norm_stats: Path | None = None
    s2_host: str = "127.0.0.1"
    s2_port: int = 8100
    s2_model: str = "system2-full"
    s2_checkpoint: str | None = None
    s2_max_tokens: int = 512
    s2_file_uri: bool = False
    general_rules: bool = True
    task_rules: bool = True
    last_milestone_retry: bool = True
    prompt_source: str = "subgoal"
    horizon_mult: float = 2.0
    max_steps_cap: int = 400
    default_est_length: int = 50
    replan_steps: int = 16
    resize_size: int = 224
    stop_progress: float = 0.95
    stop_eps: float = 0.03
    stop_window: int = 5
    static_eps: float = 0.003
    zero_arm_in_base: bool = True

    def public(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        for key in ("dataset_root", "results_root", "norm_stats"):
            if out[key] is not None:
                out[key] = str(out[key])
        return out


def _runtime_modules():
    """Import simulator/model modules lazily so the read-only GUI still starts outside RoboCasa."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import combined_eval as ce
    from openpi_client import websocket_client_policy as wcp
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    import robocasa.utils.lerobot_utils as lu
    import subtask_eval as se
    import sys2_client as s2c
    import sys2_rules as sr

    return ce, se, s2c, sr, wcp, reset_to, lu


def official_target_catalog(config: InteractiveConfig) -> list[dict[str, Any]]:
    """The exact 50-task / 1500-episode target manifest used by ``combined_eval``."""
    ce, *_ = _runtime_modules()
    root = Path(config.dataset_root).expanduser().resolve()
    out: list[dict[str, Any]] = []
    for task, episodes in ce.TARGET_EVAL_EPISODES.items():
        hits = sorted(root.glob(f"*/{task}/*/lerobot"))
        out.append(
            {
                "task_name": task,
                "task_split": ce.TARGET_TASK_SPLIT[task],
                "lerobot_dir": str(hits[0]) if hits else None,
                "available": bool(hits),
                "episodes": [
                    {
                        "episode_index": int(ep),
                        "episode_id": f"{task}/target/episode_{int(ep):06d}",
                        "manifest_rank": rank,
                    }
                    for rank, ep in enumerate(episodes)
                ],
            }
        )
    return out


def _lerobot_dir(config: InteractiveConfig, task_name: str) -> Path:
    hits = sorted(Path(config.dataset_root).expanduser().resolve().glob(f"*/{task_name}/*/lerobot"))
    if not hits:
        raise FileNotFoundError(f"no target LeRobot directory for {task_name}")
    return hits[0]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _field_edits(before: dict[str, Any], after: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    fields = ("plan", "judge", "subgoal", "subgoal_detail", "estimated_step")
    return [(field, before.get(field), after.get(field)) for field in fields if before.get(field) != after.get(field)]


def _task_finish_is_terminal(judge: str | None, control: dict[str, Any]) -> bool:
    """A false System2 finish remains executable when the retry rule suppressed it."""
    return judge == "task_finish" and not bool(control.get("suppress_task_finish"))


_NOT_SIMPLE = object()


def _simple_copy(value: Any, *, depth: int = 0) -> Any:
    """Copy latch/counter-shaped values while rejecting simulator/model object graphs."""
    if depth > 5:
        return _NOT_SIMPLE
    if value is None or isinstance(value, bool | int | float | str | np.generic):
        return copy.deepcopy(value)
    if isinstance(value, np.ndarray):
        return value.copy() if value.size <= 10_000 else _NOT_SIMPLE
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, bool | int | float | str | tuple):
                return _NOT_SIMPLE
            copied = _simple_copy(item, depth=depth + 1)
            if copied is _NOT_SIMPLE:
                return _NOT_SIMPLE
            out[copy.deepcopy(key)] = copied
        return out
    if isinstance(value, list | tuple | set):
        items = []
        for item in value:
            copied = _simple_copy(item, depth=depth + 1)
            if copied is _NOT_SIMPLE:
                return _NOT_SIMPLE
            items.append(copied)
        if isinstance(value, tuple):
            return tuple(items)
        if isinstance(value, set):
            return set(items)
        return items
    return _NOT_SIMPLE


def _snapshot_simple_attrs(obj: Any) -> dict[str, Any]:
    out = {}
    for name, value in vars(obj).items():
        copied = _simple_copy(value)
        if copied is not _NOT_SIMPLE:
            out[name] = copied
    return out


def _restore_simple_attrs(obj: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        setattr(obj, name, copy.deepcopy(value))


class InteractiveSession:
    def __init__(
        self,
        config: InteractiveConfig,
        task_name: str,
        episode_index: int,
        *,
        operator: str | None = None,
    ):
        self.config = config
        self.task_name = task_name
        self.episode_index = int(episode_index)
        self.operator = operator
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.execution_dispatch = None
        self.stage = "loading"
        self.error: str | None = None
        self.proposal: dict[str, Any] | None = None
        # Retained after a proposal is accepted/executed so the GUI always has the exact most
        # recent System2 request and response available for human reference.
        self.last_s2: dict[str, Any] | None = None
        self.s2_stream: dict[str, Any] = {
            "request_id": 0,
            "kind": None,
            "status": "idle",
            "text": "",
            "started_at": None,
            "updated_at": None,
            "finished_at": None,
            "error": None,
        }
        self.timings: dict[str, Any] = {"startup": None}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.current_node_id: str | None = None
        self.latest_frame: np.ndarray | None = None
        self.live: dict[str, Any] = self._empty_live()
        self.frame_queue: deque[tuple[int, np.ndarray]] = deque(maxlen=32)
        self.frame_sequence = 0
        self._live_progress_chunk: list[float] | None = None
        self._live_progress_replan_step = 0
        self._live_progress_last: float | None = None

        ce, se, s2c, sr, wcp, reset_to, lu = _runtime_modules()
        self.CE, self.SE, self.S2C, self.SR = ce, se, s2c, sr
        self._reset_to, self.LU = reset_to, lu
        if task_name not in ce.TARGET_EVAL_EPISODES:
            raise ValueError(f"{task_name!r} is not in TARGET_EVAL_EPISODES")
        if self.episode_index not in ce.TARGET_EVAL_EPISODES[task_name]:
            raise ValueError(f"{task_name} episode {self.episode_index} is not in the official target manifest")

        self.lerobot_dir = _lerobot_dir(config, task_name)
        self.env = ce.make_camera_env(str(self.lerobot_dir))
        self.ep_meta = lu.get_episode_meta(self.lerobot_dir, self.episode_index)
        self.instruction = (self.ep_meta.get("lang") or "").strip()
        if not self.instruction:
            raise ValueError("target episode has no ep_meta['lang'] instruction")
        self.states = lu.get_episode_states(self.lerobot_dir, self.episode_index)
        self.model_xml = lu.get_episode_model_xml(self.lerobot_dir, self.episode_index)
        self.initial_reset = {
            "states": np.asarray(self.states[0]).copy(),
            "model": self.model_xml,
            "ep_meta": json.dumps(self.ep_meta),
        }
        self._reset_to(self.env, self.initial_reset)
        obs0 = self.env._get_observations(force_update=True)
        self.base_pos_ref, self.base_yaw_ref = ce.base_reference(obs0)
        self.latest_frame = se._stacked_from_obs(obs0)
        self._queue_frame(self.latest_frame)
        # Captured after the official reset so task-specific initial flags encoded by ep_meta are
        # retained. These are the Python-side values absent from MuJoCo's flattened state.
        self.initial_python_state = {
            "env": _snapshot_simple_attrs(self.env),
            "fixtures": {name: _snapshot_simple_attrs(fixture) for name, fixture in self.env.fixtures.items()},
        }

        target = TargetEpisode(
            episode_id=f"{task_name}/target/episode_{self.episode_index:06d}",
            task_name=task_name,
            episode_index=self.episode_index,
            task_split=ce.TARGET_TASK_SPLIT[task_name],
            lerobot_dir=str(self.lerobot_dir),
            instruction=self.instruction,
            initial_state_sha256=_sha256_array(np.asarray(self.states[0])),
            model_xml_sha256=hashlib.sha256(self.model_xml.encode()).hexdigest(),
        )
        runtime = {
            "controller": "human_interactive_controller",
            "controller_schema": 2,
            "s1_host": config.s1_host,
            "s1_port": config.s1_port,
            "s1_checkpoint": config.s1_checkpoint,
            "s2_host": config.s2_host,
            "s2_port": config.s2_port,
            "s2_model": config.s2_model,
            "s2_checkpoint": config.s2_checkpoint,
            "rule_config": sr.rule_config(
                general=config.general_rules or config.task_rules,
                task_tier=config.task_rules,
                last_milestone_retry=config.last_milestone_retry,
            ),
            "rollback": (
                "soft-reset controllers, restore initial Python latches and official MuJoCo "
                "state, then replay action_applied_raw12"
            ),
        }
        self.store = HumanInteractiveStore.create(
            target,
            results_root=config.results_root,
            runtime=runtime,
            operator=operator,
        )
        self.s1_client = wcp.WebsocketClientPolicy(host=config.s1_host, port=config.s1_port)
        self.s2_client = s2c.Sys2Client(
            config.s2_host,
            config.s2_port,
            config.s2_model,
            max_tokens=config.s2_max_tokens,
            inline_media=not config.s2_file_uri,
        )
        self.norm_stats = self._load_norm_stats(config.norm_stats)

        self._save_initial_reset()
        root = self._create_node(
            parent_node_id=None,
            via_attempt_id=None,
            turn_index=0,
            plan="",
            rule_state={},
            action_prefix=np.empty((0, 12), np.float64),
            last_cmd_grip=0.0,
            seg_grip_cmds=[],
            prev_clip_rel=None,
            prev_clip_frames=0,
            episode_steps=0,
            env_success=bool(self.env._check_success()),
        )
        self.current_node_id = root
        self.stage = "ready_for_plan"
        self.store.set_session_status("active")

    @staticmethod
    def _empty_live() -> dict[str, Any]:
        return {
            "executed": 0,
            "budget": 0,
            "motion": [],
            "motion_eef_pos": [],
            "motion_eef_rot": [],
            "motion_base": [],
            "progress": [],
            "gripper_width": [],
            "step_seconds": [],
            "env_render_seconds": [],
            "stream_render_seconds": [],
            "s1_infer_seconds": [],
            "s1_obs_build_seconds": [],
            "last_step": None,
            "last_chunk": None,
            "s1_language_prompt": None,
            "anchor_media": None,
            "chunk_replan_step": None,
            "chunk_offset": None,
            "started_at": None,
            "phase": None,
            "rollout_wall_seconds": None,
            "postprocess_seconds": None,
            "turn_wall_seconds": None,
        }

    def _begin_s2_stream(self, kind: str):
        """Start one observable vLLM stream and return its thread-safe text callback."""
        with self.lock:
            request_id = int(self.s2_stream.get("request_id") or 0) + 1
            now = time.time()
            self.s2_stream = {
                "request_id": request_id,
                "kind": kind,
                "status": "streaming",
                "text": "",
                "started_at": now,
                "updated_at": now,
                "finished_at": None,
                "error": None,
            }

        def update(text: str) -> None:
            with self.lock:
                if self.s2_stream.get("request_id") != request_id:
                    return
                self.s2_stream["text"] = text
                self.s2_stream["updated_at"] = time.time()

        return update

    def _finish_s2_stream(self, *, error: BaseException | None = None) -> None:
        with self.lock:
            self.s2_stream["status"] = "error" if error else "complete"
            self.s2_stream["finished_at"] = time.time()
            self.s2_stream["updated_at"] = self.s2_stream["finished_at"]
            self.s2_stream["error"] = f"{type(error).__name__}: {error}" if error else None

    def _queue_frame(self, frame: np.ndarray) -> None:
        self.latest_frame = np.asarray(frame).copy()
        self.frame_sequence += 1
        self.frame_queue.append((self.frame_sequence, self.latest_frame))

    def next_frame(self, after: int) -> tuple[int, np.ndarray] | None:
        """Pop one ordered post-action frame; an empty queue intentionally preserves the client image."""
        with self.lock:
            while self.frame_queue and self.frame_queue[0][0] <= after:
                self.frame_queue.popleft()
            if not self.frame_queue:
                return None
            sequence, frame = self.frame_queue.popleft()
            return sequence, frame.copy()

    @staticmethod
    def _load_norm_stats(path: Path | None) -> dict | None:
        if path is None or not Path(path).is_file():
            return None
        raw = json.loads(Path(path).read_text())
        return raw.get("norm_stats", raw)

    def _save_initial_reset(self) -> None:
        state_path = self.store.session_artifact_path("reset/initial_state.npz")
        np.savez_compressed(state_path, states=np.asarray(self.states[0]))
        self.store.label_session_artifact(
            "reset/initial_state.npz",
            artifact_type="official_target_initial_state",
            metadata={"sha256": self.store.episode.initial_state_sha256},
        )
        xml_path = self.store.session_artifact_path("reset/model.xml")
        xml_path.write_text(self.model_xml)
        self.store.label_session_artifact(
            "reset/model.xml",
            artifact_type="official_target_model_xml",
            metadata={"sha256": self.store.episode.model_xml_sha256},
        )
        self.store.record_event(
            "official_target_reset_saved",
            {
                "ep_meta": self.ep_meta,
                "n_recorded_frames": len(self.states),
                "manifest_rank": self.CE.TARGET_EVAL_EPISODES[self.task_name].index(self.episode_index),
            },
        )

    def _checkpoint_doc(self, node: dict[str, Any]) -> dict[str, Any]:
        return {
            "turn_index": node["turn_index"],
            "action_count": len(node["action_prefix"]),
            "action_prefix": f"nodes/{node['node_id']}/checkpoint/action_prefix.npz",
            "sim_state": f"nodes/{node['node_id']}/checkpoint/sim_state.npz",
            "scene": f"nodes/{node['node_id']}/checkpoint/scene_full.png",
            "plan": node["plan"],
            "rule_state": _jsonable(node["rule_state"]),
            "last_cmd_grip": node["last_cmd_grip"],
            "seg_grip_cmds": node["seg_grip_cmds"],
            "prev_clip_rel": node["prev_clip_rel"],
            "prev_clip_frames": node["prev_clip_frames"],
            "episode_steps": node["episode_steps"],
            "env_success": node["env_success"],
            "last_final": _jsonable(node.get("last_final")),
            "restore_strategy": "soft_reset_python_snapshot_then_replay_applied_actions",
        }

    def _persist_node(self, node: dict[str, Any], *, write_binary: bool) -> None:
        node_id = node["node_id"]
        if write_binary:
            ap = self.store.node_artifact_path(node_id, "checkpoint/action_prefix.npz")
            np.savez_compressed(ap, actions=np.asarray(node["action_prefix"], np.float64))
            self.store.label_node_artifact(
                node_id,
                "checkpoint/action_prefix.npz",
                artifact_type="replay_action_prefix",
                metadata={"steps": len(node["action_prefix"]), "post_transform": True},
            )
            sp = self.store.node_artifact_path(node_id, "checkpoint/sim_state.npz")
            np.savez_compressed(sp, state=np.asarray(self.env.sim.get_state().flatten()))
            self.store.label_node_artifact(
                node_id,
                "checkpoint/sim_state.npz",
                artifact_type="diagnostic_mujoco_state",
                metadata={
                    "authoritative_for_rollback": False,
                    "reason": "fixture Python latches are not represented",
                },
            )
            scene = self.SE._stacked_from_obs(self.env._get_observations(force_update=True))
            self.latest_frame = scene
            scene_path = self.store.node_artifact_path(node_id, "checkpoint/scene_full.png")
            self.S2C.write_image(scene, scene_path)
            self.store.label_node_artifact(
                node_id,
                "checkpoint/scene_full.png",
                artifact_type="turn_start_scene",
                metadata={"shape": list(scene.shape)},
            )
        self.store.update_node_checkpoint(node_id, self._checkpoint_doc(node))

    def _create_node(
        self,
        *,
        parent_node_id: str | None,
        via_attempt_id: str | None,
        turn_index: int,
        plan: str,
        rule_state: dict[str, Any],
        action_prefix: np.ndarray,
        last_cmd_grip: float,
        seg_grip_cmds: list[float],
        prev_clip_rel: str | None,
        prev_clip_frames: int,
        episode_steps: int,
        env_success: bool,
        last_final: dict[str, Any] | None = None,
    ) -> str:
        node_id = f"node-{uuid.uuid4().hex[:12]}"
        if last_final is None and parent_node_id in self.nodes:
            last_final = self.nodes[parent_node_id].get("last_final")
        node = {
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "via_attempt_id": via_attempt_id,
            "turn_index": int(turn_index),
            "plan": plan,
            "rule_state": copy.deepcopy(rule_state),
            "action_prefix": np.asarray(action_prefix, np.float64).copy(),
            "last_cmd_grip": float(last_cmd_grip),
            "seg_grip_cmds": [float(x) for x in seg_grip_cmds],
            "prev_clip_rel": prev_clip_rel,
            "prev_clip_frames": int(prev_clip_frames),
            "episode_steps": int(episode_steps),
            "env_success": bool(env_success),
            "last_final": copy.deepcopy(last_final),
        }
        self.store.create_node(
            node_id=node_id,
            parent_node_id=parent_node_id,
            via_attempt_id=via_attempt_id,
            turn_index=turn_index,
            action_count=len(action_prefix),
            checkpoint={"status": "writing"},
        )
        self.nodes[node_id] = node
        self._persist_node(node, write_binary=True)
        return node_id

    def _node(self) -> dict[str, Any]:
        if self.current_node_id is None:
            raise RuntimeError("session has no active node")
        return self.nodes[self.current_node_id]

    def _rule_config(self) -> dict[str, Any]:
        """Return the exact rule selection used by this interactive session."""
        return self.SR.rule_config(
            general=self.config.general_rules or self.config.task_rules,
            task_tier=self.config.task_rules,
            last_milestone_retry=self.config.last_milestone_retry,
        )

    def _apply_execution_rules(
        self,
        raw_values: dict[str, Any],
        rule_state: dict[str, Any],
        *,
        task_status: str,
    ) -> dict[str, Any]:
        """Apply the same rule inputs and finish-suppression contract as combined_eval."""
        # Rules that inspect the System2 judgement read it from the threaded episode state.
        # Assign even when None so a previous failure/finish cannot leak into this turn.
        rule_state["judge"] = raw_values["judge"]
        return self.SR.apply_rules(
            self.task_name,
            plan=raw_values["plan"],
            subgoal=raw_values["subgoal"],
            subgoal_detail=raw_values["subgoal_detail"],
            est=raw_values["estimated_step"],
            state=rule_state,
            general=self.config.general_rules or self.config.task_rules,
            task_tier=self.config.task_rules,
            task_status=task_status,
            last_milestone_retry=self.config.last_milestone_retry,
        )

    def _attempt_media(self, relative: str) -> Path:
        node = self._node()
        if node["prev_clip_rel"] is None:
            return self.store.node_artifact_path(node["node_id"], "checkpoint/scene_full.png")
        return self.store.session_dir / node["prev_clip_rel"]

    def propose_plan(self) -> dict[str, Any]:
        with self.lock:
            if self.stage not in ("ready_for_plan", "plan_review"):
                raise RuntimeError(f"cannot request a plan while stage={self.stage}")
            if self.stage == "plan_review" and self.proposal is not None:
                self.store.supersede_attempt(self.proposal["attempt_id"], reason="human_requested_plan_regeneration")
                self.proposal = None
            self.stage = "querying_s2_plan"
            self.error = None
            node = self._node()
        try:
            scene = self.store.node_artifact_path(node["node_id"], "checkpoint/scene_full.png")
            started = time.time()
            stream_update = self._begin_s2_stream("cold_plan")
            answer = self.s2_client.plan_cold(self.instruction, scene, on_text=stream_update)
            self._finish_s2_stream()
            raw_plan = (answer.get("plan") or "").strip()
            if not raw_plan:
                raise ValueError("System2 returned no <plan>")
            adjusted = raw_plan
            interventions: list[dict[str, Any]] = []
            if self.config.task_rules:
                pr = self.SR.apply_plan_rules(self.task_name, plan=raw_plan)
                adjusted = pr["plan"]
                interventions = pr["interventions"]
            model = {
                "kind": "plan",
                "system_prompt": self.S2C.SYS_PLAN_COLD,
                "user_prompt": self.S2C.user_plan_cold(self.instruction),
                "response_raw": answer.get("raw"),
                "thought": answer.get("thought"),
                "plan": raw_plan,
                "parsed_prediction": {"plan": raw_plan},
                "media": str(scene.relative_to(self.store.session_dir)),
                "media_kind": "image",
                "latency_s": answer.get("latency_s"),
                "usage": answer.get("usage"),
            }
            rules = {
                "kind": "plan",
                "enabled": bool(self.config.task_rules),
                "interventions": interventions,
                "effective": {"plan": adjusted},
            }
            attempt_id = self.store.create_attempt(node["node_id"], model=model, rules=rules)
            proposal = {
                "kind": "plan",
                "attempt_id": attempt_id,
                "model": model,
                "rules": rules,
                "prediction": {"plan": raw_plan},
                "effective": {"plan": adjusted},
                "query_seconds": round(time.time() - started, 3),
            }
            with self.lock:
                self.proposal = proposal
                self.last_s2 = proposal
                self.stage = "plan_review"
            return self.snapshot()
        except Exception as exc:
            self._finish_s2_stream(error=exc)
            self._set_error(exc)
            raise

    def accept_plan(
        self,
        plan: str,
        *,
        author: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if self.stage != "plan_review" or not self.proposal:
                raise RuntimeError("there is no plan proposal to accept")
            proposal = self.proposal
            final_plan = (plan or "").strip()
            if not final_plan:
                raise ValueError("plan cannot be empty")
            effective = proposal["effective"]["plan"]
            if final_plan != effective:
                self.store.record_edit(
                    proposal["attempt_id"],
                    field="plan",
                    before=effective,
                    after=final_plan,
                    author=author or self.operator,
                    rationale=rationale,
                )
            human = {
                "author": author or self.operator,
                "rationale": rationale,
                "edited": final_plan != effective,
            }
            self.store.commit_attempt(proposal["attempt_id"], human=human, final={"plan": final_plan})
            self.store.finish_attempt(proposal["attempt_id"], {"kind": "plan_accepted", "plan": final_plan})
            node = self._node()
            node["plan"] = final_plan
            self._persist_node(node, write_binary=False)
            self.proposal = None
            self.stage = "ready"
            return self.snapshot()

    def propose_turn(self) -> dict[str, Any]:
        with self.lock:
            if self.stage != "ready":
                raise RuntimeError(f"cannot ask System2 while stage={self.stage}")
            self.stage = "querying_s2_turn"
            self.error = None
            node = self._node()
            plan = node["plan"]
            rule_state = copy.deepcopy(node["rule_state"])
            turn = int(node["turn_index"])
            env_success = bool(self.env._check_success())
            task_status = "finished" if env_success else "ongoing"
            grip_status = self.CE._gripper_status(node["seg_grip_cmds"]) or (
                "close" if node["last_cmd_grip"] > 0 else "open"
            )
        try:
            started = time.time()
            held = self.SR.pending_resume(rule_state)
            media_path = self._attempt_media("unused")
            plan_in = plan
            first_answer = None
            if held:
                stream_update = self._begin_s2_stream("rule_held_turn")
                stream_update("System2 was not queried because a rule is resuming a held turn.")
                self._finish_s2_stream()
                answer = {
                    "held_by_rule": held["rule"],
                    "held_kind": held.get("held_kind", "s2_resume"),
                    "subgoal": held["subgoal"],
                    "subgoal_detail": held.get("subgoal_detail") or held["subgoal"],
                    "estimated_step": held.get("est"),
                    "judge": None,
                    "thought": "System2 was not queried because a rule is resuming a held turn.",
                    "plan_update": None,
                    "raw": None,
                }
                user_prompt = None
            elif turn == 0:
                stream_update = self._begin_s2_stream("execution_turn")
                answer = self.s2_client.exec_first(
                    self.instruction, plan, media_path, on_text=stream_update
                )
                self._finish_s2_stream()
                user_prompt = self.S2C.user_exec_first(self.instruction, plan)
            else:
                if node["prev_clip_rel"] is None or not media_path.is_file():
                    raise RuntimeError("the previous executed turn has no System2 input clip")
                stream_update = self._begin_s2_stream("execution_turn")
                answer = self.s2_client.exec_turn(
                    self.instruction,
                    plan,
                    media_path,
                    node["prev_clip_frames"],
                    task_status,
                    grip_status,
                    on_text=stream_update,
                )
                self._finish_s2_stream()
                user_prompt = self.S2C.user_exec_turn(self.instruction, plan, task_status, grip_status)

            if not held:
                plan = self.S2C.apply_plan_update(plan, answer.get("plan_update"))
            raw_values = {
                "plan": plan,
                "judge": answer.get("judge"),
                "subgoal": (answer.get("subgoal") or "").strip(),
                "subgoal_detail": (answer.get("subgoal_detail") or "").strip(),
                "estimated_step": answer.get("estimated_step"),
            }
            rr: dict[str, Any] = {}
            if held:
                effective = dict(raw_values)
                interventions: list[dict[str, Any]] = []
                control = {
                    "skip_s1": False,
                    "stop_episode": False,
                    "suppress_task_finish": False,
                    "force_steps": 0,
                    "tx_label": held.get("tx_label"),
                    "held_by_rule": held["rule"],
                }
            else:
                rr = self._apply_execution_rules(raw_values, rule_state, task_status=task_status)
                effective = {
                    "plan": rr["plan"],
                    "judge": raw_values["judge"],
                    "subgoal": rr["subgoal"],
                    "subgoal_detail": rr["subgoal_detail"],
                    "estimated_step": rr["est"],
                }
                interventions = list(rr["interventions"])
                control = {
                    "skip_s1": bool(rr["skip_s1"]),
                    "stop_episode": bool(rr.get("stop_episode")),
                    "suppress_task_finish": bool(rr.get("suppress_task_finish")),
                    "force_steps": int(rr.get("force_steps") or 0),
                    "tx_label": rr.get("tx_label"),
                    "requery_s2": bool(rr.get("requery_s2")),
                }

            if rr.get("requery_s2"):
                first_answer = answer
                plan_in = effective["plan"]
                stream_update = self._begin_s2_stream("rule_requery")
                if turn == 0:
                    answer = self.s2_client.exec_first(
                        self.instruction, plan_in, media_path, on_text=stream_update
                    )
                    user_prompt = self.S2C.user_exec_first(self.instruction, plan_in)
                else:
                    answer = self.s2_client.exec_turn(
                        self.instruction,
                        plan_in,
                        media_path,
                        node["prev_clip_frames"],
                        task_status,
                        grip_status,
                        on_text=stream_update,
                    )
                    user_prompt = self.S2C.user_exec_turn(self.instruction, plan_in, task_status, grip_status)
                self._finish_s2_stream()
                plan2 = self.S2C.apply_plan_update(plan_in, answer.get("plan_update"))
                raw_values = {
                    "plan": plan2,
                    "judge": answer.get("judge"),
                    "subgoal": (answer.get("subgoal") or "").strip(),
                    "subgoal_detail": (answer.get("subgoal_detail") or "").strip(),
                    "estimated_step": answer.get("estimated_step"),
                }
                rr2 = self._apply_execution_rules(raw_values, rule_state, task_status=task_status)
                interventions.extend(rr2["interventions"])
                effective = {
                    "plan": rr2["plan"],
                    "judge": raw_values["judge"],
                    "subgoal": rr2["subgoal"],
                    "subgoal_detail": rr2["subgoal_detail"],
                    "estimated_step": rr2["est"],
                }
                control = {
                    "skip_s1": bool(rr2["skip_s1"]),
                    "stop_episode": bool(rr2.get("stop_episode")),
                    "suppress_task_finish": bool(rr2.get("suppress_task_finish")),
                    "force_steps": int(rr2.get("force_steps") or 0),
                    "tx_label": rr2.get("tx_label"),
                    "requery_s2": True,
                }

            action_override = self.SR.action_overrides(
                self.task_name,
                effective["plan"],
                effective["subgoal"],
                task_tier=self.config.task_rules,
            )
            control["action_override"] = action_override
            model = {
                "kind": "execution_turn",
                "turn": turn,
                "system_prompt": self.S2C.SYS_EXEC,
                "user_prompt": user_prompt,
                "plan_in": plan_in,
                "response_raw": answer.get("raw"),
                "thought": answer.get("thought"),
                "judge": answer.get("judge"),
                "judge_raw": answer.get("judge_raw"),
                "plan_update": answer.get("plan_update"),
                "estimated_step": answer.get("estimated_step"),
                "subgoal": answer.get("subgoal"),
                "subgoal_detail": answer.get("subgoal_detail"),
                "parsed_prediction": raw_values,
                "held_by_rule": answer.get("held_by_rule"),
                "held_kind": answer.get("held_kind"),
                "superseded_response": first_answer,
                "latency_s": answer.get("latency_s"),
                "usage": answer.get("usage"),
                "privileged": {"task_status": task_status, "gripper_status": grip_status},
                "media": str(media_path.relative_to(self.store.session_dir)),
                "media_kind": "video" if media_path.suffix.lower() == ".mp4" else "image",
            }
            rules = {
                "kind": "execution_turn",
                "rule_config": self._rule_config(),
                "interventions": interventions,
                "effective": effective,
                "control": control,
            }
            attempt_id = self.store.create_attempt(node["node_id"], model=model, rules=rules)
            proposal = {
                "kind": "turn",
                "attempt_id": attempt_id,
                "turn": turn,
                "model": model,
                "rules": rules,
                "prediction": raw_values,
                "effective": effective,
                "control": control,
                "rule_state_after": rule_state,
                "query_seconds": round(time.time() - started, 3),
            }
            with self.lock:
                self.proposal = proposal
                self.last_s2 = proposal
                self.stage = "turn_review"
            return self.snapshot()
        except Exception as exc:
            self._finish_s2_stream(error=exc)
            self._set_error(exc)
            raise

    def requery_with_plan(
        self,
        plan: str,
        *,
        author: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        """Discard the current proposal after a human closes its milestone, then query S2 again."""
        with self.lock:
            if self.stage != "turn_review" or not self.proposal:
                raise RuntimeError("a reviewed System2 proposal is required before advancing its milestone")
            final_plan = (plan or "").strip()
            if not final_plan:
                raise ValueError("plan cannot be empty")
            proposal = self.proposal
            before = proposal["effective"]["plan"]
            if final_plan != before:
                self.store.record_edit(
                    proposal["attempt_id"],
                    field="plan",
                    before=before,
                    after=final_plan,
                    author=author or self.operator,
                    rationale=rationale or "human stepped beyond the last fine step in this milestone",
                )
            self.store.record_edit(
                proposal["attempt_id"],
                field="control.human_advance_milestone",
                before=False,
                after=True,
                author=author or self.operator,
                rationale=rationale,
            )
            self.store.supersede_attempt(proposal["attempt_id"], reason="human_advance_milestone")
            node = self._node()
            node["plan"] = final_plan
            self._persist_node(node, write_binary=False)
            self.store.record_event(
                "human_advance_milestone",
                {"plan_before": before, "plan_after": final_plan},
                node_id=node["node_id"],
                attempt_id=proposal["attempt_id"],
            )
            self.proposal = None
            self.stage = "ready"
        return self.propose_turn()

    def begin_intervention(
        self,
        *,
        author: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        """Open a human-only free-play turn without querying System2 or changing the plan.

        The resulting attempt uses the active node's checklist verbatim. The operator may still
        edit that checklist explicitly before Execute, but typing a free-form subgoal alone never
        inserts it into the plan. Keeping this as a normal durable attempt makes intervention data
        usable later while clearly distinguishing it from a System2 prediction.
        """
        with self.lock:
            if self.stage not in ("ready", "turn_review"):
                raise RuntimeError(f"cannot start an intervention while stage={self.stage}")
            if self.proposal is not None:
                self.store.supersede_attempt(
                    self.proposal["attempt_id"], reason="human_started_free_intervention"
                )
                self.proposal = None
            node = self._node()
            plan = node["plan"]
            if not plan:
                raise ValueError("accept a plan before starting a free intervention")
            turn = int(node["turn_index"])
            prediction = {
                "plan": plan,
                "judge": None,
                "subgoal": "",
                "subgoal_detail": "",
                "estimated_step": int(self.config.default_est_length),
            }
            model = {
                "kind": "human_intervention",
                "source": "human",
                "turn": turn,
                "system_prompt": None,
                "user_prompt": None,
                "response_raw": None,
                "parsed_prediction": prediction,
                "plan_in": plan,
                "note": "System2 was intentionally not queried for this free-play turn.",
            }
            control = {
                "skip_s1": False,
                "stop_episode": False,
                "suppress_task_finish": False,
                "force_steps": 0,
                "tx_label": None,
                "requery_s2": False,
                "action_override": {},
            }
            rules = {
                "kind": "human_intervention",
                "rule_config": self._rule_config(),
                "interventions": [],
                "effective": prediction,
                "control": control,
                "bypassed": True,
            }
            attempt_id = self.store.create_attempt(node["node_id"], model=model, rules=rules)
            rule_state = copy.deepcopy(node["rule_state"])
            rule_state["judge"] = None
            proposal = {
                "kind": "turn",
                "human_intervention": True,
                "attempt_id": attempt_id,
                "turn": turn,
                "model": model,
                "rules": rules,
                "prediction": prediction,
                "effective": prediction,
                "control": control,
                "rule_state_after": rule_state,
                "query_seconds": 0.0,
            }
            self.store.record_event(
                "human_intervention_started",
                {
                    "author": author or self.operator,
                    "rationale": rationale,
                    "plan_unchanged": True,
                    "system2_queried": False,
                },
                node_id=node["node_id"],
                attempt_id=attempt_id,
            )
            self.proposal = proposal
            self.stage = "turn_review"
            return self.snapshot()

    def execute_turn(
        self,
        final: dict[str, Any],
        *,
        author: str | None = None,
        rationale: str | None = None,
        override_rule_stop: bool = False,
        execute_despite_skip: bool = False,
        disable_action_override: bool = False,
        disable_force_steps: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            if self.stage != "turn_review" or not self.proposal:
                raise RuntimeError("there is no execution proposal to commit")
            proposal = self.proposal
            is_intervention = bool(proposal.get("human_intervention"))
            effective = proposal["effective"]
            committed = {
                "plan": (final.get("plan") or "").strip(),
                "judge": (final.get("judge") or "").strip() or None,
                "subgoal": (final.get("subgoal") or "").strip(),
                "subgoal_detail": (final.get("subgoal_detail") or "").strip(),
                "estimated_step": int(final["estimated_step"])
                if final.get("estimated_step") not in (None, "")
                else None,
            }
            if committed["judge"] is not None and committed["judge"] not in self.S2C.JUDGES:
                raise ValueError(f"unknown judge {committed['judge']!r}")
            if not committed["plan"]:
                raise ValueError("plan cannot be empty")
            for field, before, after in _field_edits(effective, committed):
                self.store.record_edit(
                    proposal["attempt_id"],
                    field=field,
                    before=before,
                    after=after,
                    author=author or self.operator,
                    rationale=rationale,
                )
            control = proposal["control"]
            # Record only overrides that actually defeated an active rule. The GUI always treats
            # an explicit Execute click as authoritative for stop/skip, without polluting ordinary
            # turns with meaningless override=true metadata.
            overrides = {
                "override_rule_stop": bool(override_rule_stop and control.get("stop_episode")),
                "execute_despite_skip": bool(execute_despite_skip and control.get("skip_s1")),
                "disable_action_override": bool(disable_action_override and control.get("action_override")),
                "disable_force_steps": bool(disable_force_steps and int(control.get("force_steps") or 0) > 0),
            }
            for name, enabled in overrides.items():
                if enabled:
                    self.store.record_edit(
                        proposal["attempt_id"],
                        field=f"control.{name}",
                        before=False,
                        after=True,
                        author=author or self.operator,
                        rationale=rationale,
                    )
            human = {
                "author": author or self.operator,
                "rationale": rationale,
                "mode": "intervention" if is_intervention else "system2_review",
                "edited_fields": [x[0] for x in _field_edits(effective, committed)],
                "control_overrides": overrides,
            }
            final_doc = {
                **committed,
                "human_intervention": is_intervention,
                "control_overrides": overrides,
                "task_finish_suppressed": bool(control.get("suppress_task_finish")),
            }
            self.store.commit_attempt(proposal["attempt_id"], human=human, final=final_doc)

            no_execution_reason = None
            if _task_finish_is_terminal(committed["judge"], control):
                no_execution_reason = "human_final_task_finish"
            elif control.get("stop_episode") and not overrides["override_rule_stop"]:
                no_execution_reason = "rule_stop_episode"
            elif control.get("skip_s1") and not overrides["execute_despite_skip"]:
                no_execution_reason = "rule_skip_s1"
            if no_execution_reason:
                return self._finish_without_execution(proposal, committed, no_execution_reason=no_execution_reason)
            if not committed["subgoal"]:
                raise ValueError("an executable turn requires a non-empty subgoal")
            if committed["estimated_step"] is not None and committed["estimated_step"] <= 0:
                raise ValueError("estimated_step must be positive")

            self.cancel_event.clear()
            self.live = self._empty_live()
            self.live["started_at"] = time.time()
            self.live["phase"] = "system1_rollout"
            self._live_progress_chunk = None
            self._live_progress_replan_step = 0
            self._live_progress_last = None
            self.stage = "executing"

            def job():
                return self._execute_worker(
                    proposal,
                    committed,
                    # Free-play means the typed language reaches System1 without task-rule action
                    # pinning. Ordinary reviewed turns retain the existing rule behavior.
                    disable_action_override=disable_action_override or is_intervention,
                    disable_force_steps=disable_force_steps,
                )

            if self.execution_dispatch is not None:
                # InteractiveManager queues this onto the SAME owner thread that constructed the
                # EGL context. HTTP returns immediately; /api/hitl/state reads copied telemetry.
                self.execution_dispatch(job)
            else:
                # Direct/non-GUI use remains thread-correct by running synchronously.
                job()
            return self.snapshot()

    def _finish_without_execution(
        self,
        proposal: dict[str, Any],
        final: dict[str, Any],
        *,
        no_execution_reason: str,
    ) -> dict[str, Any]:
        node = self._node()
        result = {
            "kind": "turn_not_executed",
            "reason": no_execution_reason,
            "turn": proposal["turn"],
            "steps": 0,
            "env_success": bool(self.env._check_success()),
        }
        self.store.finish_attempt(proposal["attempt_id"], result)
        child = self._create_node(
            parent_node_id=node["node_id"],
            via_attempt_id=proposal["attempt_id"],
            turn_index=node["turn_index"] + 1,
            plan=final["plan"],
            rule_state=proposal["rule_state_after"],
            action_prefix=node["action_prefix"],
            last_cmd_grip=node["last_cmd_grip"],
            seg_grip_cmds=[],
            prev_clip_rel=node["prev_clip_rel"],
            prev_clip_frames=node["prev_clip_frames"],
            episode_steps=node["episode_steps"],
            env_success=result["env_success"],
        )
        self.current_node_id = child
        self.store.set_active_node(child, reason=no_execution_reason)
        self.proposal = None
        # HITL sessions are deliberately non-terminal: task_finish, rule stops, and observed
        # success remain visible signals, but the operator may keep probing without a turn cap.
        self.stage = "ready"
        self.store.set_session_status("active", reason=f"observed_{no_execution_reason}")
        return self.snapshot()

    def _live_step(self, record: dict[str, Any], frame: np.ndarray) -> None:
        # The rollout's recorded frame is the observation used to choose the action (pre-step).
        # For a live operator, render once after env.step so the stream shows the RESULT of the
        # action that just completed. This runs on the simulator owner thread.
        stream_started = time.perf_counter()
        try:
            post_obs = self.env._get_observations(force_update=True)
            post_frame = self.SE._stacked_from_obs(post_obs)
        except Exception:
            post_frame = frame
        stream_render_seconds = time.perf_counter() - stream_started
        with self.lock:
            self._queue_frame(post_frame)
            self.live["executed"] = int(record["frame_step"]) + 1
            self.live["motion"].append(record.get("motion_norm"))
            self.live["motion_eef_pos"].append(record.get("action_eef_pos_norm"))
            self.live["motion_eef_rot"].append(record.get("action_eef_rot_norm"))
            self.live["motion_base"].append(record.get("action_base_norm"))
            query = record.get("query") or None
            frame_step = int(record["frame_step"])
            if query:
                chunk = query.get("chunk_progress")
                self._live_progress_chunk = list(chunk) if chunk is not None else None
                self._live_progress_replan_step = frame_step
                self.live["chunk_replan_step"] = frame_step
                self.live["last_chunk"] = _jsonable(query)
                self.live["s1_language_prompt"] = query.get("prompt")
            progress = record.get("progress_raw") or {}
            offset = max(0, frame_step - self._live_progress_replan_step)
            pv = None
            if self._live_progress_chunk is not None and offset < len(self._live_progress_chunk):
                pv = self._live_progress_chunk[offset]
            if pv is None:
                pv = progress.get("progress_expected_frac")
            if pv is None:
                pv = progress.get("progress_now")
            if pv is None:
                pv = self._live_progress_last
            elif np.isfinite(float(pv)):
                self._live_progress_last = float(pv)
            self.live["progress"].append(pv)
            self.live["gripper_width"].append(record.get("grip_width"))
            self.live["step_seconds"].append(record.get("t_env_step_s"))
            self.live["env_render_seconds"].append(record.get("t_env_render_s"))
            self.live["stream_render_seconds"].append(round(stream_render_seconds, 4))
            self.live["last_step"] = _jsonable(record)
            self.live["chunk_offset"] = offset
            if query:
                self.live["s1_infer_seconds"].append(query.get("s1_infer_s"))
                self.live["s1_obs_build_seconds"].append(query.get("s1_obs_build_s"))

    def _execute_worker(
        self,
        proposal: dict[str, Any],
        final: dict[str, Any],
        *,
        disable_action_override: bool,
        disable_force_steps: bool,
    ) -> None:
        turn_started = time.perf_counter()
        try:
            node = self._node()
            obs = self.env._get_observations(force_update=True)
            anchor_imgs = self.CE.images_from_obs(obs)
            anchor_state = self.CE.raw_state_from_obs(obs)
            attempt_id = proposal["attempt_id"]
            anchor_path = self.store.artifact_path(attempt_id, "execution/anchor.png")
            self.S2C.write_image(self.SE._stacked_from_obs(obs), anchor_path)
            with self.lock:
                self.live["anchor_media"] = str(anchor_path.relative_to(self.store.session_dir))
            self.store.label_artifact(
                attempt_id,
                "execution/anchor.png",
                artifact_type="s1_anchor",
                metadata={"turn": proposal["turn"]},
                node_id=node["node_id"],
            )
            est = final["estimated_step"] or self.config.default_est_length
            force_steps = 0 if disable_force_steps else int(proposal["control"].get("force_steps") or 0)
            budget = int(min(self.config.max_steps_cap, max(1, round(est * self.config.horizon_mult))))
            if force_steps > 0:
                budget = force_steps
            self.live["budget"] = budget
            action_override = (
                {}
                if disable_action_override
                else self.SR.action_overrides(
                    self.task_name,
                    final["plan"],
                    final["subgoal"],
                    task_tier=self.config.task_rules,
                )
            )
            s1_text = (
                final["subgoal_detail"]
                if self.config.prompt_source == "subgoal_detail" and final["subgoal_detail"]
                else final["subgoal"]
            )
            from stop_criterion import StopConfig

            rollout_started = time.perf_counter()
            roll = self.CE.run_s1_segment(
                self.env,
                self.s1_client,
                subgoal_text=s1_text,
                task_goal=self.instruction,
                est_length=est,
                base_pos_ref=self.base_pos_ref,
                base_yaw_ref=self.base_yaw_ref,
                anchor_imgs=anchor_imgs,
                anchor_state=anchor_state,
                resize=self.config.resize_size,
                replan_steps=self.config.replan_steps,
                budget=budget,
                stop_cfg=StopConfig(
                    progress_thresh=self.config.stop_progress,
                    eps=self.config.stop_eps,
                    window=self.config.stop_window,
                ),
                norm_stats=self.norm_stats,
                last_cmd_grip_init=node["last_cmd_grip"],
                zero_arm_in_base=self.config.zero_arm_in_base,
                act_override=action_override,
                force_steps=force_steps,
                step_callback=self._live_step,
                should_cancel=self.cancel_event.is_set,
            )
            rollout_wall = time.perf_counter() - rollout_started
            with self.lock:
                self.live["rollout_wall_seconds"] = round(rollout_wall, 3)
                self.live["phase"] = "saving_rollout_artifacts"
            postprocess_started = time.perf_counter()
            frames = roll.pop("_clean_frames")
            steps = roll.pop("_step_records")
            motion = roll.pop("_motion")
            applied = np.asarray(roll.pop("_applied_actions"), np.float64)
            actions_path = self.store.artifact_path(attempt_id, "execution/actions_applied.npz")
            np.savez_compressed(actions_path, actions=applied)
            self.store.label_artifact(
                attempt_id,
                "execution/actions_applied.npz",
                artifact_type="env_step_action_trace",
                metadata={
                    "steps": len(applied),
                    "post_action_override": True,
                    "post_zero_arm_in_base": True,
                },
                node_id=node["node_id"],
            )
            self.store.write_attempt_record(
                attempt_id,
                "execution/steps.json",
                record_type="s1_step_trace",
                payload={"steps": _jsonable(steps)},
                node_id=node["node_id"],
            )

            raw_rel = "execution/s1_rollout_raw.mp4"
            if frames:
                self.S2C.write_clip(frames, self.store.artifact_path(attempt_id, raw_rel), fps=self.S2C.SIM_FPS)
                self.store.label_artifact(
                    attempt_id,
                    raw_rel,
                    artifact_type="s1_raw_rollout_video",
                    metadata={"frames": len(frames), "fps": self.S2C.SIM_FPS},
                    node_id=node["node_id"],
                )
            clip_frames, clip_stats = self.S2C.build_clip_frames(
                frames, motion, final["subgoal"], eps=self.config.static_eps
            )
            clip_rel = None
            if clip_frames:
                clip_rel = f"attempts/{attempt_id}/execution/s2_input_clip_full.mp4"
                clip_file = self.store.artifact_path(attempt_id, "execution/s2_input_clip_full.mp4")
                self.S2C.write_clip(clip_frames, clip_file)
                self.store.label_artifact(
                    attempt_id,
                    "execution/s2_input_clip_full.mp4",
                    artifact_type="s2_next_turn_model_input",
                    metadata={"frames": len(clip_frames), "fps": self.S2C.CLIP_FPS},
                    node_id=node["node_id"],
                )
                display_file = self.store.artifact_path(attempt_id, "execution/s2_input_clip.mp4")
                self.S2C.write_clip(self.S2C.downscale(clip_frames), display_file)
                self.store.label_artifact(
                    attempt_id,
                    "execution/s2_input_clip.mp4",
                    artifact_type="s2_input_display_copy",
                    metadata={"model_input": clip_rel},
                    node_id=node["node_id"],
                )

            rule_state = copy.deepcopy(proposal["rule_state_after"])
            self.SR.record_executed_subgoal(
                rule_state,
                subgoal=final["subgoal"],
                subgoal_detail=final["subgoal_detail"],
                est=final["estimated_step"],
            )
            widths = [x.get("grip_width") for x in steps if x.get("grip_width") is not None]
            if widths:
                rule_state["grip_width"] = float(widths[-1])
                rule_state["grip_width_min"] = float(min(widths))
            prefix = np.concatenate((node["action_prefix"], applied), axis=0)
            env_success = bool(self.env._check_success())
            result = {
                "kind": "turn_executed",
                "human_intervention": bool(proposal.get("human_intervention")),
                "turn": proposal["turn"],
                "subgoal": final["subgoal"],
                "subgoal_detail": final["subgoal_detail"],
                "estimated_step": est,
                "budget": budget,
                "force_steps": force_steps or None,
                "n_steps": int(roll["n_steps"]),
                "stop_reason": roll["stop_reason"],
                "success_step": roll.get("success_step"),
                "progress_done": roll["progress_done"],
                "quiescent": roll["quiescent"],
                "env_success": env_success,
                "episode_steps_after": node["episode_steps"] + int(roll["n_steps"]),
                "action_override": action_override,
                "clip_stats": _jsonable(clip_stats),
                "timings": _jsonable(roll.get("timings")),
                "rollout_wall_seconds": round(rollout_wall, 3),
                "artifacts": {
                    "raw_video": f"attempts/{attempt_id}/{raw_rel}" if frames else None,
                    "s2_model_clip": clip_rel,
                    "s2_display_clip": (f"attempts/{attempt_id}/execution/s2_input_clip.mp4" if clip_frames else None),
                },
            }
            postprocess_seconds = time.perf_counter() - postprocess_started
            turn_wall_seconds = time.perf_counter() - turn_started
            result["postprocess_seconds"] = round(postprocess_seconds, 3)
            result["turn_wall_seconds"] = round(turn_wall_seconds, 3)
            s1_prompts = []
            for step in steps:
                prompt = (step.get("query") or {}).get("prompt")
                if prompt and (not s1_prompts or prompt != s1_prompts[-1]):
                    s1_prompts.append(prompt)
            self.store.write_attempt_record(
                attempt_id,
                "execution/demo_manifest.json",
                record_type="human_interactive_demo_turn",
                payload={
                    "schema_version": 1,
                    "turn": proposal["turn"],
                    "human_intervention": bool(proposal.get("human_intervention")),
                    "decision": {
                        "judge": final["judge"],
                        "subgoal": final["subgoal"],
                        "subgoal_detail": final["subgoal_detail"],
                        "estimated_step": final["estimated_step"],
                        "plan": final["plan"],
                    },
                    "execution": {
                        "episode_steps_before": int(node["episode_steps"]),
                        "episode_steps_after": result["episode_steps_after"],
                        "n_steps": result["n_steps"],
                        "budget": budget,
                        "stop_reason": result["stop_reason"],
                        "progress_done": result["progress_done"],
                        "env_success": env_success,
                        "s1_language_prompts": s1_prompts,
                    },
                    "artifacts": {
                        "anchor_image": f"attempts/{attempt_id}/execution/anchor.png",
                        "raw_video": result["artifacts"]["raw_video"],
                        "s2_model_clip": result["artifacts"]["s2_model_clip"],
                        "s2_display_clip": result["artifacts"]["s2_display_clip"],
                        "actions": f"attempts/{attempt_id}/execution/actions_applied.npz",
                        "steps": f"attempts/{attempt_id}/execution/steps.json",
                    },
                    "timings": result["timings"],
                },
                node_id=node["node_id"],
            )
            with self.lock:
                self.live["postprocess_seconds"] = round(postprocess_seconds, 3)
                self.live["turn_wall_seconds"] = round(turn_wall_seconds, 3)
                self.live["phase"] = "complete"
            self.store.finish_attempt(attempt_id, result)
            child = self._create_node(
                parent_node_id=node["node_id"],
                via_attempt_id=attempt_id,
                turn_index=node["turn_index"] + 1,
                plan=final["plan"],
                rule_state=rule_state,
                action_prefix=prefix,
                last_cmd_grip=roll["last_cmd_grip"],
                seg_grip_cmds=roll.get("grip_cmds") or [],
                prev_clip_rel=clip_rel or node["prev_clip_rel"],
                prev_clip_frames=(len(clip_frames) if clip_frames else node["prev_clip_frames"]),
                episode_steps=result["episode_steps_after"],
                env_success=env_success,
                last_final=final,
            )
            with self.lock:
                self.current_node_id = child
                self.store.set_active_node(child, reason="turn_executed")
                self.proposal = None
                # Success is a displayed observation, not a terminal state, in human-interactive
                # mode. There is intentionally no episode-level step or System2-turn cap.
                self.stage = "ready"
                self.store.set_session_status(
                    "active", reason="env_success_observed" if env_success else "turn_executed"
                )
        except Exception as exc:
            self._set_error(exc)

    def stop_execution(self) -> dict[str, Any]:
        with self.lock:
            if self.stage != "executing":
                raise RuntimeError("System1 is not executing")
            self.cancel_event.set()
            self.store.record_event(
                "human_stop_requested",
                {"turn": self.proposal.get("turn") if self.proposal else None},
                node_id=self.current_node_id,
                attempt_id=self.proposal.get("attempt_id") if self.proposal else None,
            )
            return self.snapshot()

    def revert_one_turn(self) -> dict[str, Any]:
        with self.lock:
            if self.stage in ("executing", "querying_s2_plan", "querying_s2_turn", "reverting"):
                raise RuntimeError(f"cannot revert while stage={self.stage}")
            self.stage = "reverting"
            if self.proposal is not None:
                self.store.supersede_attempt(self.proposal["attempt_id"], reason="discarded_before_one_turn_revert")
                self.proposal = None
            node = self._node()
            parent_id = node["parent_node_id"]
            if parent_id is None:
                self.stage = "ready" if node["plan"] else "ready_for_plan"
                raise RuntimeError("already at the beginning of the episode")
            if node["via_attempt_id"]:
                self.store.supersede_attempt(node["via_attempt_id"], reason="one_turn_revert")
            parent = self.nodes[parent_id]
        try:
            # Do not reload XML or destroy the EGL sim in-process: the eval host's binding may
            # terminate the long-lived Flask process while freeing that context. A soft reset
            # refreshes controllers/observables without replacing the model; then overwrite both
            # physics and the Python latch snapshot before replaying the accepted action prefix.
            hard_reset = self.env.hard_reset
            self.env.hard_reset = False
            try:
                self.env.reset()
            finally:
                self.env.hard_reset = hard_reset
            self.env.sim.set_state_from_flattened(self.initial_reset["states"])
            _restore_simple_attrs(self.env, self.initial_python_state["env"])
            for name, values in self.initial_python_state["fixtures"].items():
                if name in self.env.fixtures:
                    _restore_simple_attrs(self.env.fixtures[name], values)
            self.env.sim.forward()
            total = len(parent["action_prefix"])
            for i, action in enumerate(parent["action_prefix"]):
                self.env.step(np.asarray(action, np.float64))
                if i % 50 == 0:
                    with self.lock:
                        self.live["replay_progress"] = [i, total]
            expected_path = self.store.node_artifact_path(parent_id, "checkpoint/sim_state.npz")
            with np.load(expected_path) as checkpoint_data:
                expected = checkpoint_data["state"].copy()
            actual = np.asarray(self.env.sim.get_state().flatten())
            state_max_abs = (
                float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape and actual.size else None
            )
            self.store.record_event(
                "rollback_replay_completed",
                {
                    "replayed_actions": total,
                    "mujoco_state_max_abs_error": state_max_abs,
                    "mujoco_state_shape_match": actual.shape == expected.shape,
                    "note": "Python latch state is restored+replayed but not in this numeric check",
                },
                node_id=parent_id,
            )
            obs = self.env._get_observations(force_update=True)
            with self.lock:
                self.latest_frame = self.SE._stacked_from_obs(obs)
                self._queue_frame(self.latest_frame)
                self.current_node_id = parent_id
                self.store.set_active_node(parent_id, reason="one_turn_revert")
                self.stage = "ready" if parent["plan"] else "ready_for_plan"
                self.live = self._empty_live()
                self.live["replay_state_max_abs_error"] = state_max_abs
                self.store.set_session_status("active", reason="one_turn_revert")
                return self.snapshot()
        except Exception as exc:
            self._set_error(exc)
            raise

    def _set_error(self, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        with self.lock:
            self.error = detail
            self.stage = "error"
            self.store.record_event(
                "controller_error",
                {"error": detail, "traceback": traceback.format_exc()},
                node_id=self.current_node_id,
                attempt_id=self.proposal.get("attempt_id") if self.proposal else None,
            )
            self.store.set_session_status("error", error=detail)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            node = self._node()
            lineage = []
            cur = node
            while cur is not None:
                lineage.append(
                    {
                        "node_id": cur["node_id"],
                        "turn_index": cur["turn_index"],
                        "action_count": len(cur["action_prefix"]),
                        "via_attempt_id": cur["via_attempt_id"],
                    }
                )
                cur = self.nodes.get(cur["parent_node_id"])
            lineage.reverse()
            return {
                "session_id": self.store.session_id,
                "session_dir": str(self.store.session_dir),
                "episode": self.store.episode.as_dict(),
                "instruction": self.instruction,
                "stage": self.stage,
                "error": self.error,
                "active_node_id": node["node_id"],
                "turn_index": node["turn_index"],
                "episode_steps": node["episode_steps"],
                "env_success": node["env_success"],
                "plan": node["plan"],
                "previous_turn": _jsonable(node.get("last_final")),
                "proposal": _jsonable(self.proposal),
                "last_s2": _jsonable(self.last_s2),
                "s2_stream": _jsonable(self.s2_stream),
                "timings": _jsonable(self.timings),
                "live": _jsonable(self.live),
                "lineage": lineage,
                "can_revert": node["parent_node_id"] is not None,
                "frame_version": self.frame_sequence,
            }

    def close(self) -> None:
        self.cancel_event.set()
        worker = self.worker
        if worker and worker.is_alive():
            worker.join(timeout=5)
        # Do not call env.close() here. This host's EGL binding may terminate the Flask process
        # while freeing the offscreen context. InteractiveManager retains this object until process
        # exit, preventing __del__ from triggering the same teardown between annotation sessions.
        self.store.set_session_status("closed")
        self.store.close()


class InteractiveManager:
    """Process-wide single-session facade used by the Flask blueprint."""

    def __init__(self, config: InteractiveConfig):
        self.config = config
        self.lock = threading.RLock()
        self.session: InteractiveSession | None = None
        self.retired_sessions: list[InteractiveSession] = []
        self.commands: queue.Queue = queue.Queue()
        self.owner_thread = threading.Thread(
            target=self._owner_loop,
            name="hitl-simulator-owner",
            daemon=True,
        )
        self.owner_thread.start()

    def _owner_loop(self) -> None:
        while True:
            fn, done, holder = self.commands.get()
            try:
                holder["result"] = fn()
            except BaseException as exc:  # propagate command errors to the waiting HTTP request
                holder["error"] = exc
            finally:
                if done is not None:
                    done.set()

    def _call(self, fn):
        if threading.current_thread() is self.owner_thread:
            return fn()
        done = threading.Event()
        holder: dict[str, Any] = {}
        self.commands.put((fn, done, holder))
        done.wait()
        if "error" in holder:
            raise holder["error"]
        return holder.get("result")

    def _dispatch_async(self, fn) -> None:
        self.commands.put((fn, None, {}))

    def start(self, task_name: str, episode_index: int, *, operator: str | None = None) -> dict[str, Any]:
        def owned_start():
            total_started = time.perf_counter()
            close_seconds = 0.0
            with self.lock:
                if self.session is not None:
                    if self.session.stage == "executing":
                        raise RuntimeError("stop the current System1 execution before loading an episode")
                    close_started = time.perf_counter()
                    self.session.close()
                    self.retired_sessions.append(self.session)
                    close_seconds = time.perf_counter() - close_started
                self.session = None
            load_started = time.perf_counter()
            session = InteractiveSession(self.config, task_name, episode_index, operator=operator)
            episode_load_reset_seconds = time.perf_counter() - load_started
            session.execution_dispatch = self._dispatch_async
            # Publish after reset but before planning. Concurrent /state requests can now observe
            # the real vLLM token stream while this owner thread remains inside propose_plan().
            with self.lock:
                self.session = session
            plan_started = time.perf_counter()
            session.propose_plan()
            initial_plan_seconds = time.perf_counter() - plan_started
            startup = {
                "previous_session_close_s": round(close_seconds, 3),
                "episode_load_reset_s": round(episode_load_reset_seconds, 3),
                "initial_s2_plan_total_s": round(initial_plan_seconds, 3),
                "initial_s2_model_s": session.last_s2["model"].get("latency_s"),
                "total_s": round(time.perf_counter() - total_started, 3),
            }
            session.timings["startup"] = startup
            session.store.record_event("interactive_startup_timing", startup)
            return session.snapshot()

        return self._call(owned_start)

    def call_session(self, method: str, *args, **kwargs):
        def invoke():
            session = self.require()
            return getattr(session, method)(*args, **kwargs)

        return self._call(invoke)

    def stop_execution(self) -> dict[str, Any]:
        # The owner thread is busy inside run_s1_segment, so cancellation must be signalled directly
        # rather than queued behind that segment. stop_execution only touches thread-safe state,
        # an Event, and the store's own locked SQLite connection.
        return self.require().stop_execution()

    def require(self) -> InteractiveSession:
        with self.lock:
            if self.session is None:
                raise RuntimeError("no human-interactive session is loaded")
            return self.session

    def snapshot(self) -> dict[str, Any]:
        return self.require().snapshot()
