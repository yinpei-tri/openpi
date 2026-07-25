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
import os
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


# ---------------------------------------------------------------------------
# SUBOPTIMAL (failure+recovery) non-linear progress overlay.
# A suboptimal merged span's true progress is NOT a single 0->1 ramp: it rises during the approach,
# DROPS to 0 at the failed grasp, then rises 0->1 over the recovery. This overlay supplies that
# curve without re-sharding — an in-tree JSON of markers keyed by span-prefix + a piecewise formula
# applied at load time. Only suboptimal spans are in the JSON; every other span falls through to the
# usual linear ramp (dict.get -> None), so non-suboptimal data is byte-identical to before.
#   JSON: src/openpi/training/assets/suboptimal_progress.json (baked into the docker image).
#   key : <flat_episode_id>__<vtag>__s<start:06d>-<end:06d>  (episode_id '/'->'__', variant ':'->'-')
# Built by RoboAnnotator producers/build_suboptimal_progress_json.py --markers-only (single source
# of truth for the formula: robo_annotator/system2/build_samples.py::suboptimal_progress_frac).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _suboptimal_overlay() -> dict:
    try:
        import importlib.resources

        p = importlib.resources.files("openpi.training") / "assets" / "suboptimal_progress.json"
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - missing/unreadable overlay -> everything stays linear
        return {}


def _suboptimal_frac(t: int, s: int, e: int, grip_onset: int, fail_end: int) -> float:
    """Piecewise suboptimal progress at frame ``t`` in span [s,e]: approach linear rise 0->onset_frac
    (onset_frac=(grip_onset-s)/(e-s)), then linear ramp DOWN to 0 through the failed grasp
    (grip_onset..fail_end], then linear rise 0->1 over the recovery (fail_end..e]."""
    span = max(1, e - s)
    onset_frac = (grip_onset - s) / span
    if t <= grip_onset:
        p = (t - s) / max(1, grip_onset - s) * onset_frac
    elif t <= fail_end:
        p = onset_frac * (1.0 - (t - grip_onset) / max(1, fail_end - grip_onset))
    else:
        p = (t - fail_end) / max(1, e - fail_end)
    return min(1.0, max(0.0, p))


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
    # Progress-as-action: emit a per-step subgoal-progress target vector `progress_action`
    # of shape [action_horizon], to be appended as an extra action dim (after Normalize)
    # so the flow-matching action head predicts progress instead of a separate head.
    #   step i (in-subgoal) = min(1, progress_frac + i/span_len)  [step 0 == the data's
    #   progress_frac exactly]; padding steps (past subgoal end) = 1.0 (subgoal complete).
    #   normalized to [-1,1] via 2*p - 1. Off by default.
    progress_as_action: bool = False
    # Reservoir shuffle (on top of the producer's GLOBAL shard shuffle + per-epoch
    # shard-order shuffle — belt-and-suspenders for batch decorrelation).
    shuffle_buffer: int = 16000
    shuffle_initial: int = 4000
    # Opportunistic anchor LRU (small; global shuffle limits hit rate, so this is a
    # cheap belt-and-suspenders on top of OS page cache, not the primary mechanism).
    anchor_cache_size: int = 256
    seed: int = 0
    # Data-resume: number of GLOBAL shards already consumed before the resumed step
    # (0 = fresh start). Set at loader build from the checkpoint step so a spot-preempted
    # run doesn't replay the first shards. The per-worker skip and the starting epoch are
    # derived from this in _worker_shards / by set_resume_state. Approximate at the sample
    # level (the reservoir buffer's in-flight samples are dropped), but shard-accurate.
    resume_shards_consumed: int = 0
    # MULTI-NODE debug: log a one-time [data-shard-split] line per (jax-process, torch-worker)
    # showing that feeder's disjoint shard slice. Off by default; only meaningful when
    # process_count > 1 (single-node never takes the multi-node split path).
    debug_shard_split: bool = False


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
        self._debug_shard_split_logged = False
        # MULTI-NODE: capture (process_index, process_count) HERE, in the MAIN process,
        # where jax.distributed has already been initialized. torch DataLoader workers are
        # spawned as FRESH interpreters that never call jax.distributed.initialize(), so
        # jax.process_index()/process_count() inside a worker return 0/1 -> every node would
        # read ALL shards in the same order and feed DUPLICATE data. Freezing the values now
        # and reading them via _process_info() in the worker guarantees each node gets its
        # disjoint shard slice. Single-node: (0, 1) -> the split below is an identity no-op.
        try:
            import jax

            self._proc_idx = jax.process_index()
            self._proc_cnt = jax.process_count()
        except Exception:
            self._proc_idx, self._proc_cnt = 0, 1
        # Data-resume state (derived from cfg.resume_shards_consumed in _init_resume_state):
        #   _resume_epoch      = the epoch the resumed step falls in (skip applies only here)
        #   _resume_epoch_skip = GLOBAL shards to skip within that epoch (before per-worker split)
        self._resume_epoch = 0
        self._resume_epoch_skip = 0
        self._resume_skip_logged = False
        # Multi-node resume is NOT supported: the per-process disjoint slice + the resume
        # shard-skip would need to be reconciled per-process, which is untested. Fail loudly
        # rather than silently skip the wrong shards on each node. (Single-node resume works.)
        if self.cfg.resume_shards_consumed > 0 and self._proc_cnt > 1:
            raise NotImplementedError(
                "Multi-node resuming is not supported (resume_shards_consumed>0 with "
                f"process_count={self._proc_cnt}). Resume is single-node only; for multi-node, "
                "start a fresh run (EMA-only checkpoints, no train_state)."
            )
        self._init_resume_state()
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

    def set_resume_shards_consumed(self, n: int) -> None:
        """Set the GLOBAL shards-consumed count for data resume, then recompute the
        starting epoch + intra-epoch skip. Called from the trainer once the resumed step
        is known (the loader is built before the checkpoint step is restored)."""
        self.cfg = dataclasses.replace(self.cfg, resume_shards_consumed=int(n))
        self._resume_epoch = 0
        self._resume_epoch_skip = 0
        self._resume_skip_logged = False
        self._epoch = 0
        self._init_resume_state()

    def _process_info(self) -> tuple[int, int]:
        """(process_index, process_count) for multi-node shard splitting.

        Returns the values FROZEN at construction time in the main process. Do NOT call
        jax.process_index() here: this may run inside a spawned torch worker where JAX is
        not distributed-initialized and would report 0/1 (see __init__).
        """
        return self._proc_idx, self._proc_cnt

    def _init_resume_state(self) -> None:
        """Translate cfg.resume_shards_consumed (GLOBAL shards consumed pre-checkpoint)
        into a starting epoch + an intra-epoch global skip, and START the loader at that
        epoch so the resumed pass uses the correct seed+epoch shard shuffle.

        n_total = len(self._shards). consumed shards wrap across epochs:
          resume_epoch      = consumed // n_total
          resume_epoch_skip = consumed %  n_total   (shards to drop within the resumed epoch)
        """
        consumed = int(self.cfg.resume_shards_consumed)
        if consumed <= 0:
            return
        n_total = len(self._shards)
        self._resume_epoch = consumed // n_total
        self._resume_epoch_skip = consumed % n_total
        # Start at the resumed epoch so _worker_shards' seed+epoch shuffle matches the
        # ordering that produced those consumed shards.
        self._epoch = self._resume_epoch

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
                total += sum(json.loads(line)["num_samples"] for line in man.read_text().splitlines() if line.strip())
        return total

    def __len__(self) -> int:
        return self._len

    def _worker_shards(self) -> list[str]:
        """Split shards across torch DataLoader workers + shuffle shard order per epoch.

        Two code paths, selected by process_count (frozen at construction):

        SINGLE-NODE (proc_cnt == 1): unchanged from before. Shuffle by seed+epoch, slice by
        torch worker, then apply the resume shard-skip on the resumed epoch. Data-resume:
        ownership is seed-deterministic, so we reconstruct exactly which shards each worker
        had and skip its consumed prefix — no persisted per-worker counter needed.

        MULTI-NODE (proc_cnt > 1): split shards by jax.process_index() FIRST, so each node
        owns a disjoint slice. Ordering is CRITICAL for correctness: process ownership must
        be assigned BEFORE the per-epoch reshuffle, using an epoch-INDEPENDENT seed. Epochs
        advance independently per host (__iter__ bumps self._epoch when a host finishes a
        pass), so if we shuffled by seed+epoch and THEN sliced by process, a host at epoch 1
        and a host at epoch 0 would shuffle differently and their shards[proc::cnt] slices
        would OVERLAP -> duplicate training data. So:
          1. shuffle by a FIXED seed (same on every process, every epoch), slice by process
             -> each process owns a PERMANENT disjoint set, invariant to epoch skew.
          2. reshuffle that fixed slice by seed+epoch for order diversity across passes.
          3. slice by torch worker within the node.
        Resume shard-skip is NOT applied under multi-node (multi-node resume is rejected in
        __init__), so there is no skew between the two paths' resume handling.
        """
        try:
            import torch.utils.data as tud

            info = tud.get_worker_info()
        except Exception:
            info = None
        num_workers = info.num_workers if (info is not None and info.num_workers > 1) else 1
        worker_id = info.id if (info is not None and info.num_workers > 1) else 0

        proc_idx, proc_cnt = self._process_info()

        if proc_cnt > 1:
            # --- MULTI-NODE path (see docstring). Single-node never enters here. ---
            shards = list(self._shards)  # list_shards() returns a deterministic sorted list
            random.Random(self.cfg.seed).shuffle(shards)  # epoch-INDEPENDENT ownership assignment
            shards = shards[proc_idx::proc_cnt]  # this process's PERMANENT disjoint slice
            random.Random(self.cfg.seed + self._epoch).shuffle(shards)  # per-epoch order within the slice
            if num_workers > 1:
                shards = shards[worker_id::num_workers]  # this worker's slice within the node
            if self.cfg.debug_shard_split and not self._debug_shard_split_logged:
                self._debug_shard_split_logged = True
                logger.info(
                    "[data-shard-split] jax_proc=%d/%d torch_worker=%d/%d epoch=%d -> %d/%d shards (first: %s)",
                    proc_idx,
                    proc_cnt,
                    worker_id,
                    num_workers,
                    self._epoch,
                    len(shards),
                    len(self._shards),
                    Path(shards[0]).name if shards else "NONE",
                )
            return shards

        # --- SINGLE-NODE path: byte-for-byte identical to the pre-multinode behavior. ---
        shards = list(self._shards)
        random.Random(self.cfg.seed + self._epoch).shuffle(shards)
        if num_workers > 1:
            shards = shards[worker_id::num_workers]

        # Per-worker resume skip, applied only on the resumed epoch.
        if self.cfg.resume_shards_consumed > 0 and self._epoch == self._resume_epoch:
            # GLOBAL shards consumed this epoch are split evenly across workers; each
            # worker skips its share of its own (already worker-strided) slice.
            per_worker_skip = self._resume_epoch_skip // max(num_workers, 1)
            if per_worker_skip > 0:
                skipped = min(per_worker_skip, len(shards))
                if worker_id == 0 and not self._resume_skip_logged:
                    self._resume_skip_logged = True
                    logger.info(
                        "Data-resume: epoch=%d, skipping ~%d shards/worker (global consumed=%d) "
                        "-> worker slice %d -> %d shards",
                        self._epoch,
                        skipped,
                        self.cfg.resume_shards_consumed,
                        len(shards),
                        len(shards) - skipped,
                    )
                shards = shards[skipped:]
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
        sp = meta.get("span", [0, 0])
        k = self.cfg.progress_num_classes
        # SUBOPTIMAL non-linear overlay for the CURRENT-frame scalar progress (drives the
        # classification / continuous progress head). Only matches suboptimal merged spans; all
        # other spans use the producer's linear meta values UNCHANGED.
        _sp_pref = (f'{meta["episode_id"].replace("/", "__")}'
                    f'__{meta.get("variant", "normal").replace(":", "-")}__s{int(sp[0]):06d}-{int(sp[1]):06d}')
        _sp_mk = _suboptimal_overlay().get(_sp_pref)
        if _sp_mk is not None:
            _frac = _suboptimal_frac(int(meta.get("frame_index", sp[0])), int(sp[0]), int(sp[1]),
                                     int(_sp_mk["grip_onset"]), int(_sp_mk["fail_end"]))
            sample["progress_frac"] = np.float32(_frac)
            sample["progress_class"] = np.int32(np.clip(int(_frac * k), 0, k - 1))
        else:
            sample["progress_frac"] = np.float32(meta.get("subgoal_progress_frac", meta.get("progress_frac", 0.0)))
            raw_cls = int(meta.get("subgoal_progress_class", 1))
            sample["progress_class"] = np.int32(np.clip(raw_cls - 1, 0, k - 1))
        sample["subgoal_start"] = np.int32(sp[0])
        sample["subgoal_end"] = np.int32(sp[1])
        sample["frame_index"] = np.int32(meta.get("frame_index", 0))
        # Progress-as-action target: per-step subgoal progress over the action horizon.
        # step 0 == this frame's progress_frac (exact); future step i advances by
        # i/span_len (clamped to 1); steps past the subgoal end (from the subgoal pad
        # mask) are 1.0 (complete). Normalized to [-1,1] (2*p-1). Appended as an extra
        # action dim AFTER Normalize (the real 11-d action normalizes with its own stats;
        # this dim is already in range, so it must not go through Normalize).
        if self.cfg.progress_as_action:
            horizon = self.cfg.action_horizon
            # SUBOPTIMAL non-linear overlay: if this sample's span-prefix is in the overlay JSON
            # (only failure+recovery merged spans are), the per-step progress follows the
            # approach->miss->recover curve; step i is the command at frame t0+i, clamped to 1.0
            # past the subgoal end (settle-pad, complete). Every non-suboptimal span -> mk is None
            # -> the linear branch below runs UNCHANGED.
            _s, _e = int(sp[0]), int(sp[1])
            _t0 = int(sample["frame_index"])
            _prefix = (f'{meta["episode_id"].replace("/", "__")}'
                       f'__{meta.get("variant", "normal").replace(":", "-")}__s{_s:06d}-{_e:06d}')
            _mk = _suboptimal_overlay().get(_prefix)
            if _mk is not None:
                _go, _fe = int(_mk["grip_onset"]), int(_mk["fail_end"])
                prog = np.array(
                    [_suboptimal_frac(_t0 + i, _s, _e, _go, _fe) if (_t0 + i) <= _e else 1.0
                     for i in range(horizon)], dtype=np.float32)
            else:
                frac0 = float(sample["progress_frac"])
                span_len = int(sp[1]) - int(sp[0])
                steps = np.arange(horizon, dtype=np.float32)
                prog = np.minimum(1.0, frac0 + steps / span_len) if span_len > 0 else np.full(horizon, frac0, np.float32)
            # Zero out (set to complete) past the last in-subgoal step, per the pad mask.
            mask = arrays.get("action_pad_mask_subgoal")
            if mask is not None and not mask.all():
                in_subgoal = np.where(mask)[0]
                if in_subgoal.size == 0:
                    # Intentional full-stop sample: the subgoal is already complete, so NO
                    # step lies within it (all-False mask). The whole chunk is "done" -> the
                    # per-step progress target is 1.0 everywhere. (Guard against the empty
                    # np.where(...)[0][-1] that would otherwise raise IndexError and silently
                    # drop these deliberately-added stop examples via skip-and-continue.)
                    prog[:] = 1.0
                else:
                    prog[int(in_subgoal[-1]) + 1 :] = 1.0
            sample["progress_action"] = (2.0 * prog - 1.0).astype(np.float32)  # [-1,1], shape [horizon]
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
            in_subgoal = np.where(mask)[0]
            if in_subgoal.size == 0:
                # Intentional full-stop sample: subgoal already complete, no in-subgoal step
                # to re-derive the hold from. The producer's baked subgoal-pad chunk already
                # IS the full stop pose -> use it directly (and avoid the empty-np.where crash).
                return arrays["lean_action_subgoal_pad"].astype(np.float32)
            last = int(in_subgoal[-1])
            hold = real[last, LEAN_ACTION_HOLD_IDX]
            real[last + 1 :] = 0.0
            real[last + 1 :, LEAN_ACTION_HOLD_IDX] = hold
            return real
        key = "lean_action_subtask" if self.cfg.subgoal_action_pad == "episode" else "lean_action_subgoal_pad"
        return arrays[key].astype(np.float32)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        cfg = self.cfg
        # Snapshot the epoch for THIS pass. _worker_shards + the reservoir rng both key off
        # it, so it must stay fixed for the whole pass; we bump self._epoch only after the
        # pass drains. This is what makes the resume shard-skip ONE-SHOT: pass 0 runs at
        # epoch == _resume_epoch (skip applies), pass 1 at _resume_epoch+1 (skip disabled),
        # etc. Without this bump the JAX trainer (which never calls set_epoch) would pin
        # _epoch forever and re-skip the same shards every pass.
        epoch = self._epoch
        # Per-process reservoir rng: fold in process_index so each node shuffles its
        # (already disjoint) buffer differently. Single-node: proc_idx==0, so this reduces
        # to the original seed (cfg.seed + epoch*7919) -> unchanged behavior.
        rng = random.Random(cfg.seed + epoch * 7919 + self._process_info()[0])
        buffer: list[dict[str, Any]] = []
        # --- data-loading progress logs (opt-in via ROBOCASA_WDS_LOG=1) ---------------------
        # The reservoir fills to cfg.shuffle_buffer BEFORE the first sample is yielded, so a
        # large buffer + JPEG-decode-bound shards can look like a multi-minute hang with the
        # GPU idle. These logs make the warmup visible (and are the knob to tune: set
        # shuffle_buffer small for eval / a metric pass where decorrelation doesn't matter).
        _wds_log = os.environ.get("ROBOCASA_WDS_LOG") == "1"
        _t_iter0 = _time.monotonic()
        _n_shards_done = 0
        _n_built = 0
        _n_yielded = 0
        _first_yielded = False
        if _wds_log:
            logger.info("[wds] __iter__ start: epoch=%d shuffle_buffer=%d n_shards(worker)=%d",
                        epoch, cfg.shuffle_buffer, len(self._worker_shards()))
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
                    _n_built += 1
                    if _wds_log and not _first_yielded and (_n_built % 1000 == 0):
                        logger.info("[wds] warming reservoir: %d/%d samples (%.1fs elapsed)",
                                    _n_built, cfg.shuffle_buffer, _time.monotonic() - _t_iter0)
                    # Once the buffer is warm, emit a random element for every new one
                    # added (reservoir-style streaming shuffle: buffer size stays ~constant).
                    if len(buffer) >= cfg.shuffle_buffer:
                        j = rng.randrange(len(buffer))
                        buffer[j], buffer[-1] = buffer[-1], buffer[j]
                        if _wds_log and not _first_yielded:
                            _first_yielded = True
                            logger.info("[wds] reservoir warm (%d samples) after %.1fs -> first yield",
                                        len(buffer), _time.monotonic() - _t_iter0)
                        _n_yielded += 1
                        yield buffer.pop()
                _n_shards_done += 1
                if _wds_log and _n_shards_done % 10 == 0:
                    logger.info("[wds] %d shards done, %d built, %d yielded (%.1fs)",
                                _n_shards_done, _n_built, _n_yielded, _time.monotonic() - _t_iter0)
            except Exception as e:
                logger.warning("Skipping shard %s (failed to open): %r", shard, e)
                continue
        # Drain the remaining buffer in random order.
        if _wds_log:
            logger.info("[wds] shards exhausted; draining %d buffered samples (%d built, %d yielded, %.1fs)",
                        len(buffer), _n_built, _n_yielded, _time.monotonic() - _t_iter0)
        rng.shuffle(buffer)
        yield from buffer
        # Advance the epoch so the NEXT pass reshuffles differently AND the resume
        # shard-skip becomes a no-op after the resumed epoch (see __iter__ snapshot note).
        self._epoch = epoch + 1
