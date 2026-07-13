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
import functools
import io
import json
import logging
from pathlib import Path
import random
import tarfile
import time as _time
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

logger = logging.getLogger("openpi")

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
#
# S3 resilience (adopted from vla_foundry_internal's data path): reads go through a
# SHARED boto3 client configured with adaptive retries + a read timeout, and a small
# manual retry-with-backoff wraps each object fetch. This survives the transient S3 /
# FastFile blips (throttling, connection resets) that otherwise crash a multi-day run.
# We fetch objects DIRECTLY via boto3 (not through the FastFile FUSE mount) when given
# an s3:// URL — boto3 opens its own connection per call, so there is no FUSE transport
# to drop (the `ENOTCONN` failure class disappears). Local paths still read from disk.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _s3_client():
    """One retry-configured boto3 S3 client per process (matches vla_foundry's config)."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=120),
    )


def _s3_get_bytes(bucket: str, key: str, *, attempts: int = 4) -> bytes:
    """Fetch a full S3 object with retry-with-backoff on top of boto3's own retries.

    boto3's ``max_attempts`` covers establishing the GET; this outer loop additionally
    retries a failure that surfaces WHILE reading the streaming body (mid-stream
    connection reset / timeout), which boto3 does not retry. Backoff 0.5s, 1s, 2s, ...
    """
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                _time.sleep(0.5 * (2**i))
    raise OSError(f"S3 get failed after {attempts} attempts: s3://{bucket}/{key}") from last_err


def _list_one_shard_dir(shards: str) -> list[str]:
    """Resolve ONE shards spec (local dir/glob or s3:// prefix) to sorted tar URLs."""
    if shards.startswith("s3://"):
        u = urlparse(shards)
        bucket, prefix = u.netloc, u.path.lstrip("/")
        paginator = _s3_client().get_paginator("list_objects_v2")
        keys = [
            f"s3://{bucket}/{obj['Key']}"
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".tar")
        ]
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
    """Open a shard URL as a binary stream (local file or full S3 object).

    S3 shards are fetched via the shared retry client (direct boto3 GET, not FUSE), so a
    transient blip retries instead of killing the run.
    """
    if url.startswith("s3://"):
        u = urlparse(url)
        return io.BytesIO(_s3_get_bytes(u.netloc, u.path.lstrip("/")))
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
        # S3 fallback anchor roots: any anchor root already on s3:// PLUS the anchors/
        # sibling of any s3:// shard dir. Used when a local/FUSE anchor read fails (e.g.
        # FastFile mount drop -> ENOTCONN) so we can still pull the image directly from
        # S3 via the retry client instead of crashing. Deduped, order-preserving.
        s3_fallbacks = [r for r in self._anchor_roots if r.startswith("s3://")]
        for d in self._shard_dirs:
            if d.startswith("s3://"):
                fb = self._derive_anchor_root(d)
                if fb not in s3_fallbacks:
                    s3_fallbacks.append(fb)
        self._anchor_s3_fallbacks = s3_fallbacks
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

        Reads ``<anchor_root>/<key>.<cam>.jpg`` with a small LRU. Anchor roots are tried
        in order (first hit wins); s3:// roots use the shared retry client, local roots
        read from disk. RESILIENCE: if every configured root fails (e.g. a FastFile FUSE
        mount drops mid-run -> OSError/ENOTCONN on the local path), fall back to a direct
        S3 GET from ``_anchor_s3_fallbacks`` — the data lives in S3 regardless of the
        mount, so this recovers instead of crashing the whole run.
        """
        cache_key = f"{key}.{cam}"
        cached = self._anchor_cache.get(cache_key)
        if cached is not None:
            return cached
        rel = f"{key}.{cam}.jpg"
        data = None
        last_err: Exception | None = None
        # 1) Configured roots (as-is: s3:// via retry client, local via disk).
        for root in self._anchor_roots:
            try:
                if root.startswith("s3://"):
                    u = urlparse(f"{root}/{rel}")
                    data = _s3_get_bytes(u.netloc, u.path.lstrip("/"))
                else:
                    data = Path(f"{root}/{rel}").read_bytes()
                break
            except Exception as e:
                last_err = e
        # 2) S3 fallback (recovers a dead FUSE mount: the object is in S3 regardless).
        if data is None:
            for root in self._anchor_s3_fallbacks:
                try:
                    u = urlparse(f"{root}/{rel}")
                    data = _s3_get_bytes(u.netloc, u.path.lstrip("/"))
                    break
                except Exception as e:
                    last_err = e
        if data is None:
            raise FileNotFoundError(
                f"anchor {rel} not readable from roots {self._anchor_roots} "
                f"or S3 fallbacks {self._anchor_s3_fallbacks} (last error: {last_err!r})"
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
            # Anchor images (subgoal-start frame). Two dataset layouts are supported:
            #   (new, Option C) anchors BAKED INTO the shard as `anchor_<cam>.jpg` members
            #     alongside the current-cam images — read sequentially from the tar group,
            #     no random lookup, no separate anchor store. This is the robust layout.
            #   (old) anchors as LOOSE per-key files in an `anchors/` sibling dir — fetched
            #     on demand by `anchor_key` via _read_anchor (path/S3, with retry+fallback).
            if f"anchor_{CAM_KEYS[0]}.jpg" in raw:
                for ck in CAM_KEYS:
                    sample[f"observation/anchor_{ck}"] = _decode_jpeg(raw[f"anchor_{ck}.jpg"])
            else:
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
        # Skip-and-continue tolerance (vla_foundry pattern): a shard that fails to
        # open/stream, or a single sample that fails to decode/build, is LOGGED and
        # SKIPPED rather than crashing a multi-day run. A transient S3/FUSE blip on one
        # shard costs that shard's samples, not the whole job. Retries happen a layer
        # down (open_shard / _read_anchor via the retry client + S3 fallback); this is
        # the last-resort tolerance for anything that still gets through.
        for shard in self._worker_shards():
            try:
                shard_iter = iter_shard_samples(shard)
                while True:
                    try:
                        raw = next(shard_iter)
                    except StopIteration:
                        break
                    except Exception as e:
                        logger.warning("Skipping rest of shard %s after read error: %r", shard, e)
                        break
                    try:
                        sample = self._build_sample(raw, rng)
                    except Exception as e:
                        logger.warning("Skipping unbuildable sample in shard %s: %r", shard, e)
                        continue
                    if sample is None:
                        continue
                    buffer.append(sample)
                    # Once the buffer is warm, emit a random element for every new one
                    # added (reservoir-style streaming shuffle: buffer size stays ~constant).
                    if len(buffer) >= cfg.shuffle_buffer:
                        j = rng.randrange(len(buffer))
                        buffer[j], buffer[-1] = buffer[-1], buffer[j]
                        yield buffer.pop()
            except Exception as e:
                logger.warning("Skipping shard %s (failed to open): %r", shard, e)
                continue
        # Drain the remaining buffer in random order.
        rng.shuffle(buffer)
        yield from buffer
