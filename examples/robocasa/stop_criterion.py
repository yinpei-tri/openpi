"""Shared subtask STOP / completion criterion for System1 open-loop eval.

A subgoal is "done" when the progress signal is at threshold AND the motion has settled. Both the
episode-level (#2) and subtask-level (#3) evals import this so their stop behaviour is identical and
tunable from one place.

THREE signals, deliberately kept separate:

  1. PROGRESS at threshold — per progress method (see ``_read_progress`` in subtask_eval.py):
       - "action"     (progress-as-action): the model predicts progress for EVERY step of the chunk,
                        so we read ``progress_chunk[offset]`` for the step actually being executed.
                        chunk[0] would only refresh on a replan (a coarse staircase lagging by up to
                        replan_steps); chunk[-1] is the opposite error, an end-of-chunk forecast that
                        fires early because a gripper open/close takes real time.
       - "continuous" (regression scalar): ONE value per replan, held for the whole chunk -> it LAGS.
                        See ``progress_threshold_for``: the bar is relaxed by est_length to
                        compensate.
       - "classes"    (K-way): argmax bucket == last class (K-1).

  2. ARM QUIESCENCE — commanded EEF (pos+rot) + BASE motion below ``eps``. The gripper is NOT in
     this norm: its command is a saturated +/-1 target, so a |delta| term contributed a fixed 2.0
     spike on a flip and 0 otherwise -- it swamped real arm motion on flip steps and said nothing in
     between.

  3. GRIPPER SETTLED — only checked when the gripper command FLIPPED recently. We watch the gripper
     STATE (finger pad distance) not the command: the command flips instantly while the fingers take
     ~5-8 steps to travel. Measured on a real grasp (PickPlaceCounterToStove): width goes
     0.0799 -> 0.0623 over ~6 steps with |dwidth| peaking ~0.005, then settling to ~0.0006 -- hence
     ``grip_eps`` = 0.001. With no recent flip there is nothing to settle and this is skipped.

LOOKAHEAD: quiescence is evaluated on the FUTURE actions of the predicted chunk, not on executed
steps. Requiring N *executed* quiescent steps means every subgoal ends with N wasted sim steps (and
N MuJoCo renders); if the model has already decided to stop moving, that is visible in the chunk it
just predicted. ``should_stop_lookahead`` lets a driver stop BEFORE stepping.

The criterion is STATEFUL across a subgoal rollout, so construct one ``StopTracker`` per subgoal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field

import numpy as np

# progreg ("continuous") progress LAGS: the head emits one scalar per replan which is then held for
# the whole chunk, so a stale value crosses a high bar well after the subgoal is done. Relax the bar
# for the short/medium subgoals where one chunk covers a large fraction of the span. progact
# ("action") has a per-step value and needs no relaxation.
PROGREG_THRESH_BY_EST: dict[int, float] = {50: 0.88, 75: 0.88, 100: 0.92}
PROGREG_THRESH_DEFAULT = 0.95


@dataclass
class StopConfig:
    progress_thresh: float = 0.95   # progact/classes bar; progreg is relaxed per est_length
    eps: float = 0.03               # per-step EEF+base action-norm "near zero" bound
    window: int = 5                 # # consecutive quiescent steps (or future actions) required
    grip_eps: float = 0.001         # |d gripper WIDTH| (m) below which the fingers have settled
    grip_flip_window: int = 5       # a command flip within this many steps arms the gripper check
    # classes head: which bucket counts as "complete". Default = last class (K-1),
    # resolved at runtime from the returned softmax size, so this stays None here.
    complete_class: int | None = None


def progress_threshold_for(kind: str | None, est_length: int | None, cfg: StopConfig) -> float:
    """Progress threshold for this (method, est_length).

    progact / classes: ``cfg.progress_thresh`` (0.95) unchanged.
    progreg: 0.88 at est_length 50/75, 0.92 at 100, else 0.95 -- see PROGREG_THRESH_BY_EST.
    """
    if kind != "continuous" or est_length is None:
        return cfg.progress_thresh
    return PROGREG_THRESH_BY_EST.get(int(est_length), PROGREG_THRESH_DEFAULT)


def progress_complete(prog: dict | None, cfg: StopConfig, est_length: int | None = None) -> bool:
    """True if the PROGRESS signal alone says the subgoal is complete (method-aware)."""
    if not prog:
        return False
    kind = prog.get("progress_kind")
    thresh = progress_threshold_for(kind, est_length, cfg)
    if kind == "action":
        # Per-step value for the step being executed (see the module docstring).
        chunk = prog.get("progress_chunk")
        off = prog.get("chunk_offset")
        if chunk and off is not None and 0 <= int(off) < len(chunk):
            return float(chunk[int(off)]) >= thresh
        return float(prog.get("progress_now", 0.0)) >= thresh
    if kind == "classes":
        k = prog.get("progress_num_classes")
        last = cfg.complete_class if cfg.complete_class is not None else (int(k) - 1 if k else None)
        if last is None:
            return float(prog.get("progress_expected_frac", 0.0)) >= thresh
        return int(prog.get("progress_argmax", -1)) >= last
    if kind == "continuous":
        return float(prog.get("progress_now", 0.0)) >= thresh
    return False


def action_eef_base_norm(action_sim: np.ndarray, prev_grip: float | None = None) -> float:
    """L2 norm of the ARM+BASE motion of a robosuite-native 12-d action. NO gripper term.

    Robosuite-native 12-d layout (see subtask_eval.py SIM_*_IDX + lerobot_action_to_sim):
        [0:3] eef_pos, [3:6] eef_rot, [6] gripper_close, [7:11] base_motion, [11] control_mode.
    "Motion" = eef_pos+rot ([0:6]) + base ([7:11]). control_mode ([11]) is excluded.

    Gripper settling is tracked separately from the finger WIDTH (see ``StopTracker.gripper_settled``)
    because the two are different physical events on different timescales. ``prev_grip`` is accepted
    and ignored for call compatibility with the previous signature.
    """
    del prev_grip
    a = np.asarray(action_sim, dtype=np.float64).reshape(-1)
    if a.size >= 12:
        motion = np.concatenate([a[0:6], a[7:11]])
    elif a.size >= 6:
        motion = a[0:6]  # at least the EEF part
    else:
        motion = a
    return float(np.linalg.norm(motion))


def gripper_width(raw16) -> float | None:
    """Finger pad distance (m) from the RAW 16-d state: ``|q[14] - q[15]|``.

    The two pads sit at OPPOSITE signs (measured q14=+0.0400, q15=-0.0399), so the DIFFERENCE is the
    aperture: 0.0799 fully open -> 0.0623 closed on an object. Their sum is ~0.0001 and meaningless.
    ``abs`` makes the result independent of which pad is indexed first.
    """
    if raw16 is None:
        return None
    q = np.asarray(raw16, dtype=np.float64).reshape(-1)
    if q.size < 16:
        return None
    return float(abs(q[14] - q[15]))


def chunk_quiescent(chunk_sim, n: int, eps: float) -> bool:
    """True if the FIRST ``n`` actions of a predicted chunk are all arm/base-quiescent.

    Lets a driver stop WITHOUT stepping the sim: if the model's own next ``n`` actions command no
    motion, executing them only burns sim steps and renders. Fewer than ``n`` actions available ->
    False (never claim quiescence we cannot see).
    """
    if chunk_sim is None:
        return False
    arr = np.asarray(chunk_sim, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < n:
        return False
    return all(action_eef_base_norm(arr[i]) < eps for i in range(n))


@dataclass
class StopTracker:
    """Per-subgoal stateful stop detector.

    Feed each EXECUTED step's commanded action + the latest progress readout (+ the observed raw
    state, for gripper width); ask ``should_stop()``. To stop BEFORE stepping, hand the freshly
    predicted chunk to ``should_stop_lookahead()``.
    """

    cfg: StopConfig
    _norms: deque = field(init=False)                            # recent arm/base action norms
    _last_progress: dict | None = field(default=None, init=False)
    _grip_sign: int | None = field(default=None, init=False)      # sign of the gripper command
    _steps_since_flip: int = field(default=10**6, init=False)     # steps since the last flip
    _widths: deque = field(init=False)                            # recent gripper WIDTHS
    _chunk_off: int = field(default=0, init=False)                # steps consumed since the replan
    _est_length: int | None = field(default=None, init=False)     # selects the progreg threshold

    def __post_init__(self):
        self._norms = deque(maxlen=max(1, self.cfg.window))
        self._widths = deque(maxlen=max(2, self.cfg.window + 1))

    def set_est_length(self, est_length: int | None) -> None:
        """Record the subgoal's estimated length (selects the progreg progress threshold)."""
        self._est_length = int(est_length) if est_length is not None else None

    def update(self, action_sim: np.ndarray, prog: dict | None, raw16=None) -> None:
        a = np.asarray(action_sim, dtype=np.float64).reshape(-1)
        self._norms.append(action_eef_base_norm(a))

        # Gripper COMMAND flip detection on the SIGN: the served action is near +/-1 but not exact
        # (measured -1.0061 .. -0.9975), so equality tests would miss every flip.
        if a.size >= 7:
            sign = 1 if float(a[6]) > 0 else -1
            if self._grip_sign is not None and sign != self._grip_sign:
                self._steps_since_flip = 0
            else:
                self._steps_since_flip += 1
            self._grip_sign = sign

        w = gripper_width(raw16)
        if w is not None:
            self._widths.append(w)

        if prog:                    # a replan: new chunk, so the executed offset restarts at 0
            self._last_progress = prog
            self._chunk_off = 0
        else:                       # consuming the existing chunk: walk one step further into it
            self._chunk_off += 1
        if self._last_progress is not None:
            self._last_progress = {**self._last_progress, "chunk_offset": self._chunk_off}

    # -- individual signals -------------------------------------------------
    def quiescent(self) -> bool:
        """The last ``window`` EXECUTED steps were all arm/base-quiescent."""
        return len(self._norms) >= self.cfg.window and all(n < self.cfg.eps for n in self._norms)

    def gripper_flipped_recently(self) -> bool:
        """A gripper command flip within the last ``grip_flip_window`` steps."""
        return self._steps_since_flip < self.cfg.grip_flip_window

    def gripper_settled(self) -> bool:
        """Fingers stopped travelling: ``window`` consecutive small |d width| samples.

        Only meaningful after a flip. Without enough width samples we cannot verify settling, so a
        recent flip + no data returns False (conservative: keep executing rather than cut a grasp
        short mid-close).
        """
        if len(self._widths) < self.cfg.window + 1:
            return False
        d = [abs(self._widths[i] - self._widths[i - 1]) for i in range(1, len(self._widths))]
        return all(x < self.cfg.grip_eps for x in d[-self.cfg.window:])

    def progress_done(self) -> bool:
        return progress_complete(self._last_progress, self.cfg, self._est_length)

    def progress_now(self) -> float | None:
        """Progress at the step just executed (per-step for 'action', scalar otherwise)."""
        p = self._last_progress
        if not p:
            return None
        chunk, off = p.get("progress_chunk"), p.get("chunk_offset")
        if chunk and off is not None and 0 <= int(off) < len(chunk):
            return float(chunk[int(off)])
        return p.get("progress_now")

    def progress_threshold(self) -> float:
        """The threshold actually in force (method- and est_length-aware)."""
        kind = (self._last_progress or {}).get("progress_kind")
        return progress_threshold_for(kind, self._est_length, self.cfg)

    # -- combined decisions -------------------------------------------------
    def _gripper_ok(self) -> bool:
        """No recent flip -> nothing to settle. Recent flip -> require the fingers to have settled."""
        return (not self.gripper_flipped_recently()) or self.gripper_settled()

    def should_stop(self) -> bool:
        """Post-step rule: progress at threshold AND arm quiescent AND gripper settled-if-flipped."""
        return self.progress_done() and self.quiescent() and self._gripper_ok()

    def should_stop_lookahead(self, chunk_sim) -> bool:
        """Pre-step rule: stop WITHOUT executing, using the model's own next actions.

        Fires when ALL hold:
          * progress (for the step about to run) is at threshold,
          * the next ``window`` actions of the chunk are arm/base-quiescent (nothing left to do),
          * gripper: no flip in the last ``grip_flip_window`` steps, or the fingers have settled.
        Executing those quiescent actions would only add sim steps and renders.
        """
        return (self.progress_done()
                and chunk_quiescent(chunk_sim, self.cfg.window, self.cfg.eps)
                and self._gripper_ok())

    def debug(self) -> dict:
        """Snapshot of every signal (for logging / the GUI stop-reason panel)."""
        return {
            "progress_now": self.progress_now(),
            "progress_thresh": self.progress_threshold(),
            "progress_done": self.progress_done(),
            "quiescent": self.quiescent(),
            "grip_flipped_recently": self.gripper_flipped_recently(),
            "grip_settled": self.gripper_settled(),
            "steps_since_flip": (self._steps_since_flip
                                 if self._steps_since_flip < 10**6 else None),
            "width": (self._widths[-1] if self._widths else None),
        }
