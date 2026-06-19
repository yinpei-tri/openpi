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
    include_base_pos: bool,
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
    if include_base_pos:
        base_xy = raw_state[..., 0:2]  # drop z (dim 2)
        if base_pos_ref is not None:
            base_xy = base_xy - np.asarray(base_pos_ref, dtype=np.float32)[..., 0:2]
        yaw = _yaw_from_quat_xyzw(raw_state[..., STATE_BASE_QUAT])
        if base_yaw_ref is not None:
            yaw = yaw - np.float32(base_yaw_ref)
        rel_yaw_sincos = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1)
        parts += [base_xy, rel_yaw_sincos]
    return np.concatenate(parts, axis=-1)


# Lean state dim depends only on include_base_pos.
def lean_state_dim(*, include_base_pos: bool) -> int:
    # eef_pos(3) + rot6d(6) + grip(1) [+ base_xy(2) + rel_yaw_sincos(2)]
    return 10 + (4 if include_base_pos else 0)


def lean_action_from_raw(raw_action: np.ndarray) -> np.ndarray:
    """RoboCasa raw 12-dim action -> lean 11-dim (drop torso = base_motion[3])."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    return np.concatenate([raw_action[..., :ACTION_TORSO_IDX], raw_action[..., ACTION_TORSO_IDX + 1 :]], axis=-1)


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
    include_base_pos: bool = True
    # Provide anchor (before) views to the progress head.
    use_anchor_images: bool = True

    def __call__(self, data: dict) -> dict:
        scene_left = _parse_image(data["observation/scene_left"])
        scene_right = _parse_image(data["observation/scene_right"])
        wrist = _parse_image(data["observation/wrist"])

        # State is LEAN when it comes from the producer's shards (training) and RAW
        # when it comes from the RoboCasa env (inference). Detect by dim and convert
        # raw -> lean; lean passes through unchanged.
        raw_state = np.asarray(data["observation/state"], dtype=np.float32)
        target_dim = lean_state_dim(include_base_pos=self.include_base_pos)
        if raw_state.shape[-1] == ROBOCASA_STATE_DIM:
            base_pos_ref = data.get("observation/base_pos_ref")
            base_yaw_ref = data.get("observation/base_yaw_ref")
            state = lean_state_from_raw(
                raw_state,
                include_base_pos=self.include_base_pos,
                base_pos_ref=base_pos_ref,
                base_yaw_ref=None if base_yaw_ref is None else float(np.asarray(base_yaw_ref).reshape(-1)[0]),
            )
        else:
            state = raw_state  # already lean (from shards)

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
            inputs["prompt"] = data["prompt"]
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
