"""RoboCasa System1 policy transforms (subgoal-conditioned pi0.5).

Converts the RoboCasa dataset/inference sample format into the model input format
and back. Used for BOTH training and inference, so it must be a pure
representation conversion (no training-only sample-construction logic — that lives
in ``training/robocasa_dataset.py`` and the WebDataset loader).

The PRODUCER bakes the lean state/action into the shards (see
``producers/preprocess_robocasa_to_tar.py``), so during TRAINING this transform is
mostly a passthrough. At INFERENCE the RoboCasa env supplies the RAW 16-dim state,
so ``lean_state_from_raw`` runs here too (dim-detected). These conversions are the
single source of truth; the producer vendors a byte-identical copy.

Key representation choices (see the System1 plan):
- 3 native cameras: ``scene_left``, ``scene_right`` (third-person) and ``wrist``
  (``robot0_eye_in_hand``). Anchor (before) views optionally added under
  ``anchor_*`` keys for the progress head; the anchor is the start frame of the
  conditioning subgoal span and gets the SAME image augmentation as the current
  views (anchor scene cams listed in ``geometric_aug_cameras``).
- Lean state (14-d): ``eef_pos_rel(3) + eef_rot_6D(6) + gripper_width(1) +
  base_pos_rel_xy(2) + base_yaw_sincos(2)``. The two RoboCasa state quaternions
  become continuous 6D / sin-cos (Zhou et al. 2019; no pi singularity, no yaw wrap).
  Base x,y + yaw are RELATIVE to the episode/rollout start (invariant to the kitchen
  world origin). DROPPED: base_position z (≈0.70 in 98.7% of all 44M teleop frames),
  base_rotation roll/pitch (base is pure yaw). ``control_mode`` is NOT in the state
  (the subgoal verb, e.g. "navigate to…", implies the regime).
- Lean action (11-d): the sim's 12-d delta interface MINUS the torso command
  (base_motion[3]), which is identically 0 across ALL 57,350 teleop episodes. The
  output transform re-inserts torso=0 to rebuild the sim's native 12-d command.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# RoboCasa raw state layout (lowdim.npz "state", 16-dim), per episode.json modality:
#   [0:3] base_position, [3:7] base_quat, [7:10] eef_pos_rel,
#   [10:14] eef_quat_rel, [14:16] gripper_qpos (width = q0 - q1, in [0, 0.08]).
STATE_BASE_POS = slice(0, 3)
STATE_BASE_QUAT = slice(3, 7)
STATE_EEF_POS = slice(7, 10)
STATE_EEF_QUAT = slice(10, 14)
STATE_GRIPPER = slice(14, 16)

# RoboCasa native camera keys (third-person scene + wrist).
SCENE_LEFT = "scene_left"
SCENE_RIGHT = "scene_right"
WRIST = "wrist"
CAMERA_KEYS = (SCENE_LEFT, SCENE_RIGHT, WRIST)
# Anchor (before) view keys: same cameras, prefixed.
ANCHOR_PREFIX = "anchor_"

# RoboCasa native (sim) dims, and the dead torso command we drop.
ROBOCASA_STATE_DIM = 16   # raw state width (env supplies this at inference)
ROBOCASA_ACTION_DIM = 12  # sim controller expects this many dims
ACTION_TORSO_IDX = 3      # base_motion[3]; identically 0 in all teleop data
LEAN_ACTION_DIM = 11      # ROBOCASA_ACTION_DIM minus torso


def _parse_image(image) -> np.ndarray:
    """Coerce an image to uint8 HWC (LeRobot may store float32 CHW)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 6D rotation (first two columns of R), flattened.

    The 6D representation is continuous over SO(3) and has no sign/double-cover
    ambiguity (Zhou et al. 2019). Accepts a single (4,) quaternion or a batch
    (..., 4); returns (..., 6).
    """
    quat = np.asarray(quat, dtype=np.float32)
    q = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # Columns 0 and 1 of the rotation matrix.
    c0 = np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=-1
    )
    c1 = np.stack(
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=-1
    )
    return np.concatenate([c0, c1], axis=-1)


def _yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Yaw (z-rotation) from a (...,4) quaternion (x,y,z,w). RoboCasa base is pure yaw."""
    q = np.asarray(q, dtype=np.float32)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def lean_state_from_raw(
    raw_state: np.ndarray,
    *,
    include_base_pose: bool,
    base_pos_ref: np.ndarray | None,
    base_yaw_ref: float | None = None,
) -> np.ndarray:
    """RoboCasa raw 16-dim state -> lean state.

    Layout: eef_pos_rel(3) + eef_rot_6D(6) + gripper_width(1)
            [+ base_pos_rel_xy(2) + rel_base_yaw_sincos(2)].
    Base x,y + heading are RELATIVE to the episode/rollout-start reference
    (``base_pos_ref`` (3,), ``base_yaw_ref`` scalar) so they're invariant to the
    kitchen's absolute world origin; relative yaw -> (sin, cos) (no +/-pi wrap).
    DROPPED: base z (near-constant 0.70) and base roll/pitch (base is pure yaw).
    EEF rotation stays 6D (continuous; RoboCasa's gripper sits near the axis-angle
    pi singularity).
    """
    raw_state = np.asarray(raw_state, dtype=np.float32)
    eef_pos = raw_state[..., STATE_EEF_POS]
    eef_rot6d = quat_xyzw_to_rot6d(raw_state[..., STATE_EEF_QUAT])
    grip = raw_state[..., STATE_GRIPPER]
    gripper_width = (grip[..., 0] - grip[..., 1])[..., None]
    parts = [eef_pos, eef_rot6d, gripper_width]
    if include_base_pose:
        base_xy = raw_state[..., 0:2]  # drop z (dim 2)
        if base_pos_ref is not None:
            base_xy = base_xy - np.asarray(base_pos_ref, dtype=np.float32)[..., 0:2]
        yaw = _yaw_from_quat_xyzw(raw_state[..., STATE_BASE_QUAT])
        if base_yaw_ref is not None:
            yaw = yaw - np.float32(base_yaw_ref)
        rel_yaw_sincos = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)
        parts += [base_xy, rel_yaw_sincos]
    return np.concatenate(parts, axis=-1)


# Lean state dim depends only on include_base_pose.
def lean_state_dim(*, include_base_pose: bool) -> int:
    # eef_pos(3) + rot6d(6) + grip(1) [+ base_xy(2) + rel_yaw_sincos(2)]
    return 10 + (4 if include_base_pose else 0)


def lean_action_from_raw(raw_action: np.ndarray) -> np.ndarray:
    """RoboCasa raw 12-dim action -> lean 11-dim (drop torso = base_motion[3])."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    return np.concatenate([raw_action[..., :ACTION_TORSO_IDX], raw_action[..., ACTION_TORSO_IDX + 1 :]], axis=-1)


def _as_str(x) -> str:
    """Coerce a prompt field (str / np.str_ / 0-d array / bytes) to a python str."""
    if isinstance(x, bytes):
        return x.decode()
    if hasattr(x, "item"):
        try:
            x = x.item()
        except (ValueError, AttributeError):
            x = str(x)
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


# Metadata fields rendered on their own line after the task/subgoal text, as
# "Label: value; Label: value" (pi0.7-style conditioning tags). Each entry maps a
# sample-dict key -> the human-facing label. ORDER is the render order and is FROZEN
# into the training data, so append-only. Today only `scope` (subgoal level) is
# populated; Quality / Mistake / etc. are reserved for future offline-RL data (curated
# failures + System1 rollouts) and are emitted only when the key is present + non-empty.
METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("scope", "Scope"),       # "milestone" (long span) | "step" (short, one fine action)
    # ("quality", "Quality"),   # FUTURE: 0-5 trajectory quality (needs a scorer)
    # ("mistake", "Mistake"),   # FUTURE: "true"/"false" (needs curated failure data)
)


def build_prompt(
    subgoal: str,
    task_goal: str | None = None,
    *,
    include_task_goal: bool = False,
    metadata: dict | None = None,
) -> str:
    """Assemble the language prompt fed to pi0.5 (the SINGLE prompt-assembly point).

    Used by BOTH the WebDataset loader (training) and the eval adapter (inference) so
    the conditioning text is identical. Both the whole-task goal and the subgoal are
    LOWERCASED for normalization (the PaliGemma tokenizer strips/_-cleans but does NOT
    lowercase, and RoboCasa subgoals come in as full sentences with varied casing).

    - ``include_task_goal=False`` (default): prompt = the subgoal alone. System1 is a
      clean, composable subgoal executor — leaning on the whole-task string couples the
      policy to the task and hurts recombined / OOD subgoal sequences.
    - ``include_task_goal=True``: prepend the whole-task goal as disambiguating context
      -> ``"<task>; Current Subgoal: <subgoal>"`` (an ablation; System2/the env always
      knows the task, so it's free at inference).

    ``metadata`` (optional): conditioning tags rendered on a SECOND line, e.g.
    ``Scope: step`` (subgoal level). At TRAINING set them to the OBSERVED value; at
    INFERENCE set them to the DESIRED value (decision-transformer style). The tokenizer
    wraps the whole thing as ``Task: <line1>\\n<line2>\\nState: <ints>;``.
    """
    # Strip any newlines from the raw text components so the ONLY newline in the prompt
    # is the structural one before the metadata line (the tokenizer preserves newlines
    # for RoboCasa, so a stray newline in subgoal text would inject a spurious line).
    subgoal = _as_str(subgoal).strip().lower().replace("\n", " ")
    if include_task_goal and task_goal:
        # Strip trailing sentence punctuation off the task goal so the join reads
        # "make coffee; Current Subgoal: …" not "make coffee.; Current Subgoal: …".
        task = _as_str(task_goal).strip().lower().replace("\n", " ").rstrip(".!; ")
        line1 = f"{task}; Current Subgoal: {subgoal}"
    else:
        line1 = subgoal

    if metadata:
        tags = []
        for key, label in METADATA_FIELDS:
            val = metadata.get(key)
            if val is not None and str(val) != "":
                tags.append(f"{label}: {_as_str(val).strip().lower()}")
        if tags:
            # Second line (pi0.7 puts metadata tags after the task text); the tokenizer
            # then appends "\nState: …\nAction: ".
            return line1 + "\n" + "; ".join(tags)
    return line1


def sim_action_from_lean(lean_action: np.ndarray) -> np.ndarray:
    """Lean 11-dim action -> sim native 12-dim (re-insert torso=0 at base_motion[3])."""
    lean_action = np.asarray(lean_action)
    torso = np.zeros((*lean_action.shape[:-1], 1), dtype=lean_action.dtype)
    return np.concatenate(
        [lean_action[..., :ACTION_TORSO_IDX], torso, lean_action[..., ACTION_TORSO_IDX:]], axis=-1
    )


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """Convert a RoboCasa sample to model inputs (training + inference)."""

    # Model action dim to pad to (e.g. 32 for pi0.5).
    action_dim: int
    # Determines image masking convention.
    model_type: _model.ModelType
    # Include relative base position in the state vector.
    include_base_pose: bool = True
    # Provide anchor (before) views to the progress head.
    use_anchor_images: bool = True
    # Prepend the whole-task goal to the subgoal prompt (disambiguating context).
    include_task_goal: bool = False
    # Concatenate the anchor (subgoal-start) lean state onto the current lean state, so
    # pi0.5's discretized state ints carry the proprioceptive before/after DELTA
    # (complements the anchor images). Doubles the state width (14 -> 28). Independent
    # of use_anchor_images (images vs proprioception), but typically paired with it.
    include_anchor_state: bool = False
    # When the loader sends RAW state + a baked-lean reference (training, recompute
    # mode), assert the recomputed lean matches the baked lean within this tolerance —
    # guarding against drift between the producer's lean math and this transform's. The
    # check is skipped when no baked-lean is present (e.g. inference).
    verify_lean_atol: float = 1e-4
    # Emit the metadata line (e.g. "Scope: milestone|step") in the prompt. Tells the VLA
    # the subgoal level so it executes a long (milestone) vs short (step) action span —
    # the conditioning hook that later carries offline-RL tags (Quality/Mistake/...).
    include_metadata: bool = True

    def _to_lean_state(self, raw_state: np.ndarray, data: dict, baked_lean: np.ndarray | None = None) -> np.ndarray:
        """Lean state from a stored-lean OR raw-16-dim vector (dim-detected).

        When ``raw_state`` is raw 16-d (training recompute mode or inference), convert
        via ``lean_state_from_raw`` (the single source of truth). If ``baked_lean`` is
        supplied (the producer's pre-baked lean), assert the recompute matches it —
        catching any divergence between the producer's vendored lean math and this one.
        """
        raw_state = np.asarray(raw_state, dtype=np.float32)
        if raw_state.shape[-1] == ROBOCASA_STATE_DIM:
            base_pos_ref = data.get("observation/base_pos_ref")
            base_yaw_ref = data.get("observation/base_yaw_ref")
            lean = lean_state_from_raw(
                raw_state,
                include_base_pose=self.include_base_pose,
                base_pos_ref=base_pos_ref,
                base_yaw_ref=None if base_yaw_ref is None else float(np.asarray(base_yaw_ref).reshape(-1)[0]),
            )
            if baked_lean is not None:
                baked = np.asarray(baked_lean, dtype=np.float32)
                if lean.shape == baked.shape and not np.allclose(lean, baked, atol=self.verify_lean_atol):
                    max_diff = float(np.max(np.abs(lean - baked)))
                    raise ValueError(
                        "RoboCasa lean-state drift: recomputed lean state differs from the producer's "
                        f"baked lean by {max_diff:.2e} (> atol {self.verify_lean_atol:.0e}). The vendored "
                        "lean math in producers/preprocess_robocasa_to_tar.py and robocasa_policy.py "
                        "have diverged — re-sync them (or re-shard)."
                    )
            return lean
        return raw_state  # already lean (from shards), no raw to recompute/verify

    def __call__(self, data: dict) -> dict:
        scene_left = _parse_image(data["observation/scene_left"])
        scene_right = _parse_image(data["observation/scene_right"])
        wrist = _parse_image(data["observation/wrist"])

        # State arrives RAW (16-d) from the env (inference) OR — in training recompute
        # mode — RAW + a baked-lean reference; it may also arrive already-lean (recompute
        # off). _to_lean_state dim-detects, converts raw->lean via the single source of
        # truth, and verifies against the baked lean when present.
        state = self._to_lean_state(
            data["observation/state"], data, baked_lean=data.get("observation/state_lean_baked")
        )

        if self.include_anchor_state:
            # Anchor (subgoal-start) lean state, appended so the discretized prompt ints
            # carry the proprioceptive before-state. Recompute from raw anchor (verifying
            # against the baked lean anchor) the same way as the current state. At
            # inference fall back to the current state (delta -> 0, a benign "no progress
            # yet" prior) when the env can't provide a before-state.
            anchor_raw = data.get("observation/anchor_state")
            anchor_state = (
                self._to_lean_state(anchor_raw, data, baked_lean=data.get("observation/anchor_state_lean_baked"))
                if anchor_raw is not None
                else state
            )
            state = np.concatenate([state, anchor_state], axis=-1)

        image = {SCENE_LEFT: scene_left, SCENE_RIGHT: scene_right, WRIST: wrist}
        image_mask = {SCENE_LEFT: np.True_, SCENE_RIGHT: np.True_, WRIST: np.True_}

        if self.use_anchor_images:
            # Anchor views are the same cameras at the subtask-start (before) frame.
            # Fall back to the current frame if not provided (degenerate before==now).
            for cam in CAMERA_KEYS:
                src = data.get(f"observation/{ANCHOR_PREFIX}{cam}")
                anchor = _parse_image(src) if src is not None else image[cam]
                image[f"{ANCHOR_PREFIX}{cam}"] = anchor
                image_mask[f"{ANCHOR_PREFIX}{cam}"] = np.bool_(src is not None)

        inputs = {"state": state, "image": image, "image_mask": image_mask}

        if "actions" in data:
            # Actions are LEAN (11-d) from shards; if RAW (12-d) ever arrives, drop torso.
            act = np.asarray(data["actions"], dtype=np.float32)
            if act.shape[-1] == ROBOCASA_ACTION_DIM:
                act = lean_action_from_raw(act)
            inputs["actions"] = act
        if "prompt" in data:
            # Single prompt-assembly point (train + inference): lowercase + optionally
            # prepend the whole-task goal + a metadata line. The loader/eval adapter pass
            # the raw subgoal phrasing in `prompt`, the episode task in `task_goal`, and
            # conditioning tags (e.g. `scope`) read here into the metadata dict.
            metadata = None
            if self.include_metadata:
                metadata = {key: data[key] for key, _ in METADATA_FIELDS if key in data}
            inputs["prompt"] = build_prompt(
                data["prompt"],
                data.get("task_goal"),
                include_task_goal=self.include_task_goal,
                metadata=metadata,
            )
        # Progress label + span metadata are training-only targets (pass through).
        for k in ("progress_frac", "subgoal_start", "subgoal_end", "frame_index"):
            if k in data:
                inputs[k] = data[k]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """Recover the sim's native 12-dim RoboCasa action from the model output.

    The model emits LEAN_ACTION_DIM (11) real dims (after un-padding from action_dim);
    re-insert torso=0 at base_motion[3] to rebuild the sim's 12-d command.
    """

    def __call__(self, data: dict) -> dict:
        lean = np.asarray(data["actions"][:, :LEAN_ACTION_DIM])
        return {"actions": sim_action_from_lean(lean)}
