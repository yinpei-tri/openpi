"""Streaming WebDataset loader for System1 RoboCasa shards.

Reads the per-frame, globally-shuffled tar shards written by RoboAnnotator's
``producers/preprocess_robocasa_to_tar.py`` (one tar = many frames from many
episodes) and yields the SAME pre-transform sample contract that
``policies/robocasa_policy.py::RobocasaInputs`` consumes — so the model / transform
/ norm path is identical to any other openpi dataset.

Design (system1_full format):
- Shards are streamed sequentially (one big read; works for local files and S3 via
  boto3), which is why the producer packs each shard with frames from MANY episodes
  pre-shuffled — a small reservoir buffer then decorrelates cheaply.
- Each frame bakes ONE subgoal (level selection happened upstream); ``prompt_source``
  picks the phrasing (subgoal / subgoal_detail / milestone_text) — a fixed load-time
  knob, not a random axis. Offline-RL conditioning (quality / est_length /
  executed_step / gripper_flag) + the 10-way progress class ride along per frame.
- Action label: ``subgoal_action_pad`` selects the baked chunk (subgoal- vs
  episode-padded); ``repad_actions`` re-derives the settle-pad in-loader from the real
  chunk + subgoal mask (padding applied to UNNORMALIZED lean actions).
- Anchor = the subgoal's START frame, read by ``anchor_key`` from the anchor store.

This is an IterableDataset; it plugs into ``transform_iterable_dataset`` +
``create_robocasa_webdataset_data_loader`` in ``data_loader.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import io
import json
from pathlib import Path
import random
import tarfile
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

CAM_KEYS = ("scene_left", "scene_right", "wrist")

# Lean 11-d action layout: [base_vx,vy,yaw(0:3), control_mode(3), eef_pos(4:7),
# eef_rot(7:10), gripper_close(10)]. Settle-pad HOLDS control_mode + gripper at their
# last real value and zeros the rest (velocity/eef -> 0 == "stop moving").
LEAN_ACTION_HOLD_IDX = [3, 10]


@dataclasses.dataclass(frozen=True)
class WebDatasetConfig:
    """Load-time knobs for streaming the RoboCasa shards."""

    shards: str  # local dir/glob or s3://bucket/prefix; comma-separate to pool multiple dirs
    action_horizon: int = 20
    include_base_pose: bool = True
    use_anchor_images: bool = True
    # Append the anchor (subgoal-start) lean state to the per-frame sample so the
    # policy can concatenate it onto the current state (proprioceptive before/after).
    include_anchor_state: bool = False
    # If True (default), send the RAW state (current + anchor) + the episode base
    # reference instead of the pre-baked lean state, so the policy recomputes lean from
    # raw via the SAME robocasa_policy.lean_state_from_raw used at inference — and also
    # ship the baked lean so the policy can ASSERT recompute == baked (catches any drift
    # between the producer's lean math and the policy's). If False, pass the baked lean
    # straight through (cheaper; no verification).
    recompute_lean_from_raw: bool = True
    # Anchor store (path-addressable). If None, derived as the `anchors/` sibling of
    # `shards`. Anchors are NOT embedded in shards (~1M frames at full scale); the
    # loader reads them by key on demand. For S3, STAGE this dir to local disk first
    # (the OS page cache then serves hot anchors from RAM — no per-sample S3 GET).
    anchors_dir: str | None = None
    # --- prompt scope (system1_full: flat per-sample subgoal, no per-level lists) ---
    # Which text drives "Current Subgoal": "subgoal" (terse) | "subgoal_detail" (verbose)
    # | "milestone" (the milestone_text). Fixed per run (no random axis for now); the new
    # data bakes ONE subgoal per frame, so level selection happened upstream.
    prompt_source: str = "subgoal"
    # Emit the offline-RL conditioning line "Quality: …; Estimated Length: …; Executed
    # Step: …" in the prompt (the failure-augmented data makes Quality meaningful).
    include_conditioning: bool = True
    # Emit "Current Gripper: Open|Close;" after the state block.
    include_gripper_flag: bool = True
    # --- action padding ---
    # Which stored chunk to use as the action label: "subgoal" (subgoal-padded) |
    # "episode" (episode-padded). system1_full bakes both (lean_action_subgoal_pad /
    # lean_action_subtask).
    subgoal_action_pad: str = "subgoal"
    # If True, IGNORE the baked padded chunk and re-derive the settle-pad in the loader
    # from the real chunk (lean_action_subtask) + the subgoal mask: past the subgoal end,
    # zero the velocity/eef dims and HOLD control_mode + gripper at their last real value.
    repad_actions: bool = False
    # 10-way progress classification target (subgoal-level). The class label
    # (subgoal_progress_class 0..9) is passed through for the classification head.
    progress_num_classes: int = 10
    # Reservoir shuffle (on top of the producer's GLOBAL shard shuffle + per-epoch
    # shard-order shuffle — belt-and-suspenders for batch decorrelation).
    shuffle_buffer: int = 16000
    shuffle_initial: int = 4000
    # Opportunistic anchor LRU (small; global shuffle limits hit rate, so this is a
    # cheap belt-and-suspenders on top of OS page cache, not the primary mechanism).
    anchor_cache_size: int = 256
    seed: int = 0


# ---------------------------------------------------------------------------
# Shard IO (local + S3), dependency-light.
# ---------------------------------------------------------------------------
def _list_one_shard_dir(shards: str) -> list[str]:
    """Resolve ONE shards spec (local dir/glob or s3:// prefix) to sorted tar URLs."""
    if shards.startswith("s3://"):
        import boto3

        u = urlparse(shards)
        bucket, prefix = u.netloc, u.path.lstrip("/")
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".tar"):
                    keys.append(f"s3://{bucket}/{obj['Key']}")
        return sorted(keys)
    p = Path(shards)
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.tar"))
    # treat as a glob
    return sorted(str(x) for x in Path(p.parent).glob(p.name))


def list_shards(shards: str) -> list[str]:
    """Resolve a shards spec to a sorted list of tar URLs (local paths or s3://).

    ``shards`` may be a SINGLE dir/glob/s3-prefix, OR a comma-separated list of them to
    pool multiple shard directories into one dataset (e.g. "s3://a/shards,s3://b/shards").
    Each part is resolved independently and the results concatenated; within-part order
    is sorted, and the pooled list is sorted so shard assignment is deterministic across
    processes/workers.
    """
    parts = [s.strip() for s in shards.split(",") if s.strip()]
    if len(parts) == 1:
        return _list_one_shard_dir(parts[0])
    pooled: list[str] = []
    for part in parts:
        pooled.extend(_list_one_shard_dir(part))
    return sorted(pooled)


def open_shard(url: str) -> io.BufferedReader | io.BytesIO:
    """Open a shard URL as a binary stream (local file or full S3 object)."""
    if url.startswith("s3://"):
        import boto3

        u = urlparse(url)
        s3 = boto3.client("s3")
        body = s3.get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
        return io.BytesIO(body)
    return open(url, "rb")


def iter_shard_samples(url: str) -> Iterator[dict[str, bytes]]:
    """Yield per-sample dicts {member_suffix: bytes} from one tar shard.

    Members are named ``<key>.<suffix>``; we group consecutive members by ``<key>``.
    The producer writes all members of a sample contiguously.
    """
    stream = open_shard(url)
    with tarfile.open(fileobj=stream, mode="r|*") as tar:  # streaming mode
        cur_key: str | None = None
        group: dict[str, bytes] = {}
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            key, suffix = name.split(".", 1)
            if cur_key is not None and key != cur_key:
                yield group
                group = {}
            cur_key = key
            group[suffix] = tar.extractfile(member).read()
        if group:
            yield group


# ---------------------------------------------------------------------------
# Sample decode -> pre-transform contract dict.
# ---------------------------------------------------------------------------
def _decode_jpeg(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


class RoboCasaWebDataset:
    """IterableDataset over RoboCasa shards yielding the RobocasaInputs contract."""

    def __init__(self, cfg: WebDatasetConfig):
        self.cfg = cfg
        self._shards = list_shards(cfg.shards)
        if not self._shards:
            raise ValueError(f"No .tar shards found at {cfg.shards}")
        self._epoch = 0
        # shards may be a comma-separated list of dirs (pooled dataset); split for the
        # per-dir manifest count + anchor-root derivation.
        self._shard_dirs = [s.strip() for s in cfg.shards.split(",") if s.strip()]
        # Approximate length from the manifest(s) if present (for len()).
        self._len = self._read_manifest_count()
        # Anchor store roots (one per shard dir): explicit override, else the `anchors/`
        # sibling of each shards dir. _read_anchor searches these in order (a sample's
        # anchor lives in the anchor store of whichever dir its shard came from).
        if cfg.anchors_dir is not None:
            self._anchor_roots = [a.strip().rstrip("/") for a in cfg.anchors_dir.split(",") if a.strip()]
        else:
            self._anchor_roots = [self._derive_anchor_root(d) for d in self._shard_dirs]
        self._anchor_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _derive_anchor_root(shards_dir: str) -> str:
        """The `anchors/` sibling of one shards dir (local or s3://)."""
        if shards_dir.startswith("s3://"):
            return shards_dir.rstrip("/").rsplit("/", 1)[0] + "/anchors"
        sp = Path(shards_dir)
        base = sp if sp.is_dir() else sp.parent
        return str((base.parent / "anchors") if base.name == "shards" else (base / "anchors"))

    def _read_anchor(self, key: str, cam: str) -> np.ndarray:
        """Resolve an anchor key (<flat_id>/f<frame:06d>) + cam to a decoded image.

        Reads ``<anchor_root>/<key>.<cam>.jpg`` (local or s3://) with a small LRU. When
        multiple shard dirs are pooled, the anchor roots are searched in order (first hit
        wins) — a sample's anchor lives in the store paired with its shard's dir. For S3,
        stage the anchor store locally first so this is a local read.
        """
        cache_key = f"{key}.{cam}"
        cached = self._anchor_cache.get(cache_key)
        if cached is not None:
            return cached
        data = None
        last_err: Exception | None = None
        for root in self._anchor_roots:
            url = f"{root}/{key}.{cam}.jpg"
            try:
                if url.startswith("s3://"):
                    from urllib.parse import urlparse as _up

                    import boto3

                    u = _up(url)
                    data = boto3.client("s3").get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
                else:
                    data = Path(url).read_bytes()
                break
            except Exception as e:  # noqa: BLE001 - try the next root
                last_err = e
        if data is None:
            raise FileNotFoundError(
                f"anchor {key}.{cam}.jpg not found in any anchor root {self._anchor_roots}"
            ) from last_err
        img = _decode_jpeg(data)
        if len(self._anchor_cache) >= self.cfg.anchor_cache_size:
            self._anchor_cache.pop(next(iter(self._anchor_cache)))
        self._anchor_cache[cache_key] = img
        return img

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _read_manifest_count(self) -> int:
        # Sum the local manifest(s) across all shard dirs. S3 length is approximate /
        # optional (skipped). Returns 0 if no local manifest is found.
        total = 0
        for d_str in self._shard_dirs:
            if d_str.startswith("s3://"):
                continue
            d = Path(d_str)
            man = (d / "manifest.jsonl") if d.is_dir() else (d.parent / "manifest.jsonl")
            if man.exists():
                total += sum(
                    json.loads(line)["num_samples"] for line in man.read_text().splitlines() if line.strip()
                )
        return total

    def __len__(self) -> int:
        return self._len

    def _worker_shards(self) -> list[str]:
        """Split shards across torch DataLoader workers + shuffle shard order per epoch."""
        try:
            import torch.utils.data as tud

            info = tud.get_worker_info()
        except Exception:
            info = None
        shards = list(self._shards)
        random.Random(self.cfg.seed + self._epoch).shuffle(shards)
        if info is not None and info.num_workers > 1:
            shards = shards[info.id :: info.num_workers]
        return shards

    def _build_sample(self, raw: dict[str, bytes], rng: random.Random) -> dict[str, Any] | None:
        if "meta.json" not in raw:
            return None
        meta = json.loads(raw["meta.json"])
        arrays = np.load(io.BytesIO(raw["arrays.npz"]))

        # --- prompt text (system1_full bakes ONE subgoal per frame; pick the phrasing) ---
        if self.cfg.prompt_source == "subgoal_detail":
            prompt = meta.get("subgoal_detail") or meta.get("subgoal", "")
        elif self.cfg.prompt_source == "milestone":
            prompt = meta.get("milestone_text", "")
        else:  # "subgoal"
            prompt = meta.get("subgoal", "")

        # --- images + state ---
        sample: dict[str, Any] = {
            "observation/scene_left": _decode_jpeg(raw["scene_left.jpg"]),
            "observation/scene_right": _decode_jpeg(raw["scene_right.jpg"]),
            "observation/wrist": _decode_jpeg(raw["wrist.jpg"]),
        }
        # base reference lives in meta.json now (t=0 base pose).
        base_pos_ref = np.asarray(meta.get("base_pos_ref", [0.0, 0.0, 0.0]), dtype=np.float32)
        base_yaw_ref = np.float32(meta.get("base_yaw_ref", 0.0))
        if self.cfg.recompute_lean_from_raw and "state" in arrays:
            # RAW 16-d state + base ref -> RobocasaInputs recomputes lean via the SAME
            # path used at inference; ship the baked lean too so it asserts recompute==baked.
            sample["observation/state"] = arrays["state"].astype(np.float32)
            sample["observation/base_pos_ref"] = base_pos_ref
            sample["observation/base_yaw_ref"] = base_yaw_ref
            sample["observation/state_lean_baked"] = arrays["lean_state"].astype(np.float32)
        else:
            # Pass the pre-baked lean straight through (dim-detected as already-lean).
            sample["observation/state"] = arrays["lean_state"].astype(np.float32)

        if self.cfg.use_anchor_images:
            # Single anchor key per frame (subgoal-start), path-addressable.
            anchor_key = meta["anchor_key"]
            for ck in CAM_KEYS:
                sample[f"observation/anchor_{ck}"] = self._read_anchor(anchor_key, ck)

        if self.cfg.include_anchor_state:
            # Anchor (subgoal-start) state, concatenated onto the current by the policy.
            if self.cfg.recompute_lean_from_raw and "state_anchor" in arrays:
                sample["observation/anchor_state"] = arrays["state_anchor"].astype(np.float32)
                if "lean_state_anchor" in arrays:
                    sample["observation/anchor_state_lean_baked"] = arrays["lean_state_anchor"].astype(np.float32)
            elif "lean_state_anchor" in arrays:
                sample["observation/anchor_state"] = arrays["lean_state_anchor"].astype(np.float32)

        # --- actions (lean 11-d; chosen pad variant, optionally re-padded) ---
        sample["actions"] = self._build_actions(arrays)

        # --- targets / meta ---
        sample["prompt"] = prompt
        sample["task_goal"] = meta.get("task_goal", "")
        # Offline-RL conditioning tags (rendered as a line after the subgoal). At inference
        # the eval adapter sets these to the DESIRED values (decision-transformer style).
        sample["quality"] = meta.get("quality", "")
        sample["est_length"] = np.int32(meta.get("est_length", 0))
        sample["executed_step"] = np.int32(meta.get("executed_step", 0))
        sample["gripper_flag"] = meta.get("gripper_flag", "")
        # Progress targets: continuous frac + K-way class (subgoal level). The producer
        # emits subgoal_progress_class 1-INDEXED (1..progress_classes, e.g. 1..10); the
        # classifier head wants 0-indexed labels 0..K-1, so shift by -1 and clip into
        # range (defends against an out-of-range label silently becoming an all-zeros
        # one-hot row = dead gradient). progress_frac stays the raw [0,1] fraction.
        sample["progress_frac"] = np.float32(meta.get("subgoal_progress_frac", meta.get("progress_frac", 0.0)))
        k = self.cfg.progress_num_classes
        raw_cls = int(meta.get("subgoal_progress_class", 1))
        sample["progress_class"] = np.int32(np.clip(raw_cls - 1, 0, k - 1))
        sp = meta.get("span", [0, 0])
        sample["subgoal_start"] = np.int32(sp[0])
        sample["subgoal_end"] = np.int32(sp[1])
        sample["frame_index"] = np.int32(meta.get("frame_index", 0))
        return sample

    def _build_actions(self, arrays: np.lib.npyio.NpzFile) -> np.ndarray:
        """Lean 11-d action chunk for the chosen pad mode, optionally re-padded.

        Baked chunks: ``lean_action_subgoal_pad`` (subgoal settle-pad) and
        ``lean_action_subtask`` (real actions, episode-padded). ``repad_actions`` re-derives
        the settle-pad from the real chunk + the subgoal mask: past the subgoal end, zero
        the non-HOLD dims and hold control_mode + gripper at their last real value.
        """
        if self.cfg.repad_actions:
            real = arrays["lean_action_subtask"].astype(np.float32).copy()
            mask = arrays["action_pad_mask_subgoal"]
            if mask.all():
                return real  # subgoal spans the whole horizon; nothing to pad
            last = int(np.where(mask)[0][-1])
            hold = real[last, LEAN_ACTION_HOLD_IDX]
            real[last + 1 :] = 0.0
            real[last + 1 :, LEAN_ACTION_HOLD_IDX] = hold
            return real
        key = "lean_action_subtask" if self.cfg.subgoal_action_pad == "episode" else "lean_action_subgoal_pad"
        return arrays[key].astype(np.float32)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        cfg = self.cfg
        rng = random.Random(cfg.seed + self._epoch * 7919)
        buffer: list[dict[str, Any]] = []
        for shard in self._worker_shards():
            for raw in iter_shard_samples(shard):
                sample = self._build_sample(raw, rng)
                if sample is None:
                    continue
                buffer.append(sample)
                # Once the buffer is warm, emit a random element for every new one
                # added (reservoir-style streaming shuffle: buffer size stays ~constant).
                if len(buffer) >= cfg.shuffle_buffer:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    yield buffer.pop()
        # Drain the remaining buffer in random order.
        rng.shuffle(buffer)
        yield from buffer
