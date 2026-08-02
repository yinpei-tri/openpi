"""Shared subtask STOP / completion criterion for System1 open-loop eval.

A subgoal is considered "done" when BOTH signals fire (user-specified AND, not OR):

  1. PROGRESS at threshold — per progress method (see `_read_progress` in subtask_eval.py):
       - "action"     (progress-as-action): ``progress_now >= progress_thresh`` (chunk[0], the
                        current progress — NOT progress_end/chunk[-1], which is the model's forecast
                        for the chunk end and fires too early since a gripper open/close takes time).
       - "classes"    (10-way): argmax bucket == last class (K-1), i.e. "class 10".
       - "continuous" (regression scalar): ``progress_now >= progress_thresh``.
  2. QUIESCENCE — the last ``window`` executed steps all have a near-zero *commanded* motion:
     EEF (pos+rot) + base + |Δgripper| (change in the open/close command) below ``eps``. Including
     the gripper DELTA means a place/release where the arm is still but the fingers are actively
     opening/closing counts as MOVING (not quiescent); a steady "hold closed" (Δ=0) stays quiescent.

Both #2 (episode-level) and #3 (subtask-level) eval use this so their stop behavior is
identical and tunable from one place. Thresholds are conservative defaults; expose them as
CLI flags in the drivers.

The criterion is STATEFUL across a subgoal rollout (it tracks the recent action-norm
window), so construct one `StopTracker` per subgoal and feed it each executed step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class StopConfig:
    progress_thresh: float = 0.95   # proact progress_end / proreg scalar threshold
    eps: float = 0.02               # per-step EEF+base action-norm "near zero" bound
    window: int = 5                 # # consecutive quiescent steps required
    # classes head: which bucket counts as "complete". Default = last class (K-1),
    # resolved at runtime from the returned softmax size, so this stays None here.
    complete_class: int | None = None


def progress_complete(prog: dict | None, cfg: StopConfig) -> bool:
    """True if the PROGRESS signal alone says the subgoal is complete (method-aware)."""
    if not prog:
        return False
    kind = prog.get("progress_kind")
    if kind == "action":
        # Use progress_NOW (chunk[0]), NOT progress_end (chunk[-1]). progress_end is the model's
        # forecast for the END of the action chunk, which fires too EARLY: e.g. it predicts 1.0 while
        # progress_now is still ~0.86, but a gripper open/close (place/release) takes real time the
        # forecast skips. Gating on progress_now waits until the subgoal is ACTUALLY near-complete.
        return float(prog.get("progress_now", 0.0)) >= cfg.progress_thresh
    if kind == "classes":
        k = prog.get("progress_num_classes")
        last = cfg.complete_class if cfg.complete_class is not None else (int(k) - 1 if k else None)
        # If we don't know K, fall back to the expected-fraction threshold.
        if last is None:
            return float(prog.get("progress_expected_frac", 0.0)) >= cfg.progress_thresh
        return int(prog.get("progress_argmax", -1)) >= last
    if kind == "continuous":
        return float(prog.get("progress_now", 0.0)) >= cfg.progress_thresh
    return False


def action_eef_base_norm(action_sim: np.ndarray, prev_grip: float | None = None) -> float:
    """L2 norm of the motion components of a robosuite-native 12-d action, INCLUDING the gripper
    DELTA (change in the open/close command since the previous step).

    Robosuite-native 12-d layout (see subtask_eval.py SIM_*_IDX + lerobot_action_to_sim):
        [0:3] eef_pos, [3:6] eef_rot, [6] gripper_close, [7:11] base_motion, [11] control_mode.
    "Motion" = eef_pos+rot ([0:6]) + base ([7:11]) + |Δgripper| (|a[6] − prev_grip|). The gripper
    command is an ABSOLUTE open/close target, so we use its DELTA: a place/release where the arm is
    still but the gripper is actively opening/closing IS moving (the fingers take real time), and the
    old rule (gripper excluded) wrongly read it as quiescent. Once the gripper command HOLDS constant
    (Δ=0) the term vanishes, so a steady "hold closed" still counts as quiescent — as intended.
    control_mode ([11]) is excluded. prev_grip=None -> gripper term omitted (first step).
    """
    a = np.asarray(action_sim, dtype=np.float64).reshape(-1)
    if a.size >= 12:
        parts = [a[0:6], a[7:11]]
        if prev_grip is not None:
            parts.append(np.array([a[6] - float(prev_grip)]))
        motion = np.concatenate(parts)
    elif a.size >= 6:
        motion = a[0:6]  # at least the EEF part
    else:
        motion = a
    return float(np.linalg.norm(motion))


@dataclass
class StopTracker:
    """Per-subgoal stateful stop detector. Feed each EXECUTED step's commanded action +
    the most recent progress readout; ask `should_stop()`."""

    cfg: StopConfig
    _norms: deque = field(init=False)
    _last_progress: dict | None = field(default=None, init=False)
    _prev_grip: float | None = field(default=None, init=False)   # previous gripper command (for Δ)

    def __post_init__(self):
        self._norms = deque(maxlen=max(1, self.cfg.window))

    def update(self, action_sim: np.ndarray, prog: dict | None) -> None:
        a = np.asarray(action_sim, dtype=np.float64).reshape(-1)
        self._norms.append(action_eef_base_norm(a, prev_grip=self._prev_grip))
        if a.size >= 7:
            self._prev_grip = float(a[6])   # remember for next step's gripper delta
        if prog:  # progress is only produced on replan steps; keep the latest
            self._last_progress = prog

    def quiescent(self) -> bool:
        return len(self._norms) >= self.cfg.window and all(n < self.cfg.eps for n in self._norms)

    def progress_done(self) -> bool:
        return progress_complete(self._last_progress, self.cfg)

    def should_stop(self) -> bool:
        """BOTH progress-complete AND action-quiescent (the user-specified AND rule)."""
        return self.progress_done() and self.quiescent()

    def reason(self) -> dict:
        """Diagnostic snapshot for logging why (or why not) we stopped."""
        return dict(
            progress_done=self.progress_done(),
            quiescent=self.quiescent(),
            recent_norms=[round(n, 4) for n in self._norms],
            last_progress=self._last_progress,
        )
