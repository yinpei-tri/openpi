"""Streaming WebDataset loader for preprocessed LIBERO shards (multi-node ready).

Reads the LIBERO WebDataset tars at s3://.../preprocessed/libero/shards (or a local
copy). One tar packs many per-frame samples; each sample = these members, keyed by a
shared UUID:
  <key>.image_t0.jpg          third-person view (256x256 RGB)
  <key>.wrist_image_t0.jpg    wrist view
  <key>.lowdim.npz            state (T,8), actions (T,7), past_mask (T,), future_mask (T,)
  <key>.metadata.json         camera_names, frame/episode indices, ...
  <key>.language_instructions.json   {"original": "<task instruction>"}

Chunk indexing (producer config: past_lowdim_steps=1, future_lowdim_steps=10 -> T=12):
  index 0 is the CURRENT frame (past_mask[0]=True); indices 1.. are future.
  -> current state = state[0] (8-d); action horizon = actions[1:1+action_horizon].

Yields the pre-transform contract that ``policies/libero_policy.py::LiberoInputs``
consumes: observation/state, observation/image, observation/wrist_image, actions, prompt.

MULTI-NODE: shards are split by jax.process_index() (disjoint per node) AND by torch
DataLoader worker, so a 2-node x N-worker run reads every shard exactly once per epoch.
A reservoir buffer decorrelates the sequential within-shard reads.
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import io
import json
import logging
from pathlib import Path
import random
import tarfile
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class LiberoWebDatasetConfig:
    """Load-time knobs for streaming the LIBERO shards."""

    shards: str  # local dir/glob or s3://bucket/prefix; comma-separate to pool dirs
    action_horizon: int = 10
    image_key: str = "image"  # base cam member prefix -> <key>.image_t0.jpg
    wrist_key: str = "wrist_image"
    # Reservoir shuffle (belt-and-suspenders on top of the producer's shuffle).
    shuffle_buffer: int = 8000
    seed: int = 0
    # Emit a one-time debug line per (process, worker) showing the shard split — so a
    # multi-node run's logs prove each of the 16 GPUs' feeders got a disjoint shard set.
    debug_shard_split: bool = True


# ---------------------------------------------------------------------------
# Shard IO (local + S3).
# ---------------------------------------------------------------------------
def _list_one(shards: str) -> list[str]:
    if shards.startswith("s3://"):
        import boto3

        u = urlparse(shards)
        s3 = boto3.client("s3")
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=u.netloc, Prefix=u.path.lstrip("/")):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".tar"):
                    keys.append(f"s3://{u.netloc}/{obj['Key']}")
        return sorted(keys)
    p = Path(shards)
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.tar"))
    return sorted(str(x) for x in Path(p.parent).glob(p.name))


def list_shards(shards: str) -> list[str]:
    """Resolve a (possibly comma-separated) shards spec to sorted tar URLs."""
    parts = [s.strip() for s in shards.split(",") if s.strip()]
    if len(parts) == 1:
        return _list_one(parts[0])
    pooled: list[str] = []
    for part in parts:
        pooled.extend(_list_one(part))
    return sorted(pooled)


def open_shard(url: str) -> io.BufferedReader | io.BytesIO:
    if url.startswith("s3://"):
        import boto3

        u = urlparse(url)
        body = boto3.client("s3").get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
        return io.BytesIO(body)
    return open(url, "rb")


def iter_shard_samples(url: str) -> Iterator[dict[str, bytes]]:
    """Yield {suffix: bytes} per sample; members of one sample are contiguous by key."""
    with tarfile.open(fileobj=open_shard(url), mode="r|*") as tar:
        cur_key: str | None = None
        group: dict[str, bytes] = {}
        for member in tar:
            if not member.isfile():
                continue
            key, suffix = member.name.split(".", 1)
            if cur_key is not None and key != cur_key:
                yield group
                group = {}
            cur_key = key
            group[suffix] = tar.extractfile(member).read()
        if group:
            yield group


def _decode_jpeg(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


class LiberoWebDataset:
    """IterableDataset over LIBERO shards yielding the LiberoInputs contract.

    Multi-node: shards split by jax.process_index() first (disjoint per node), then by
    torch worker within the node.
    """

    def __init__(self, cfg: LiberoWebDatasetConfig):
        self.cfg = cfg
        self._shards = list_shards(cfg.shards)
        if not self._shards:
            raise ValueError(f"No .tar shards found at {cfg.shards}")
        self._epoch = 0
        self._debug_logged = False
        # CRITICAL for multi-node: capture (process_index, process_count) HERE, in the
        # main process, where jax.distributed has already been initialized. torch
        # DataLoader workers are spawned as FRESH interpreters that never call
        # jax.distributed.initialize(), so jax.process_index()/process_count() inside a
        # worker return 0/1 -> every node would read ALL shards in the same order and
        # feed DUPLICATE data. Freezing the values now and reading them in the worker
        # (via _process_info) guarantees each node gets its disjoint shard slice.
        try:
            import jax

            self._proc_idx = jax.process_index()
            self._proc_cnt = jax.process_count()
        except Exception:
            self._proc_idx, self._proc_cnt = 0, 1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        # Approximate: sum manifest num_sequences if a local manifest is present.
        total = 0
        for d in [s.strip() for s in self.cfg.shards.split(",") if s.strip()]:
            if d.startswith("s3://"):
                continue
            man = Path(d) / "manifest.jsonl" if Path(d).is_dir() else Path(d).parent / "manifest.jsonl"
            if man.exists():
                for line in man.read_text().splitlines():
                    if line.strip():
                        total += json.loads(line).get("num_sequences", json.loads(line).get("num_samples", 0))
        return total

    def _process_info(self) -> tuple[int, int]:
        """(process_index, process_count) for multi-node shard splitting.

        Returns the values FROZEN at construction time in the main process. Do NOT call
        jax.process_index() here: this runs inside a spawned torch worker where JAX is
        not distributed-initialized and would report 0/1 (see __init__).
        """
        return self._proc_idx, self._proc_cnt

    def _worker_shards(self) -> list[str]:
        """Shards for THIS (jax-process, torch-worker), shuffled per epoch.

        Split order: shuffle all shards deterministically (same seed on every process so
        the global order agrees), then take this process's stride, then this worker's
        stride within the process. Guarantees disjoint coverage across all feeders.
        """
        proc_idx, proc_cnt = self._process_info()
        try:
            import torch.utils.data as tud

            winfo = tud.get_worker_info()
        except Exception:
            winfo = None
        worker_id = winfo.id if winfo is not None else 0
        num_workers = winfo.num_workers if winfo is not None else 1

        shards = list(self._shards)
        random.Random(self.cfg.seed + self._epoch).shuffle(shards)  # SAME order on all procs
        shards = shards[proc_idx::proc_cnt]  # this node's disjoint slice
        if num_workers > 1:
            shards = shards[worker_id::num_workers]  # this worker's slice within the node

        if self.cfg.debug_shard_split and not self._debug_logged:
            self._debug_logged = True
            logger.info(
                f"[data-shard-split] jax_proc={proc_idx}/{proc_cnt} torch_worker={worker_id}/{num_workers} "
                f"epoch={self._epoch} -> {len(shards)}/{len(self._shards)} shards "
                f"(first: {Path(shards[0]).name if shards else 'NONE'})"
            )
        return shards

    def _build_sample(self, raw: dict[str, bytes]) -> dict[str, Any] | None:
        if "lowdim.npz" not in raw:
            return None
        arrays = np.load(io.BytesIO(raw["lowdim.npz"]))
        state = arrays["state"]  # (T, 8)
        actions = arrays["actions"]  # (T, 7)
        h = self.cfg.action_horizon
        # index 0 = current frame; actions[1:1+h] = the future action chunk to predict.
        cur_state = state[0].astype(np.float32)
        act = actions[1 : 1 + h].astype(np.float32)
        if act.shape[0] < h:  # pad short tails by repeating the last action (copy strategy)
            pad = np.repeat(act[-1:], h - act.shape[0], axis=0) if act.shape[0] else np.zeros((h, act.shape[1]), np.float32)
            act = np.concatenate([act, pad], axis=0)

        lang = json.loads(raw["language_instructions.json"]) if "language_instructions.json" in raw else {}
        prompt = lang.get("original", "") if isinstance(lang, dict) else ""

        return {
            "observation/state": cur_state,
            "observation/image": _decode_jpeg(raw[f"{self.cfg.image_key}_t0.jpg"]),
            "observation/wrist_image": _decode_jpeg(raw[f"{self.cfg.wrist_key}_t0.jpg"]),
            "actions": act,
            "prompt": prompt,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        cfg = self.cfg
        rng = random.Random(cfg.seed + self._epoch * 7919 + self._process_info()[0])  # per-proc reservoir rng
        buffer: list[dict[str, Any]] = []
        for shard in self._worker_shards():
            for raw in iter_shard_samples(shard):
                sample = self._build_sample(raw)
                if sample is None:
                    continue
                buffer.append(sample)
                if len(buffer) >= cfg.shuffle_buffer:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    yield buffer.pop()
        rng.shuffle(buffer)
        yield from buffer
        # Advance the epoch so the NEXT pass reshuffles shards + reservoir. TorchDataLoader
        # re-invokes __iter__ each time the dataset is exhausted (data_loader.py restarts
        # the iterator), and set_epoch() is never called + can't reach persistent workers.
        # All feeders start at epoch 0 and complete each pass in lockstep, so the seed
        # (seed+epoch) stays aligned across processes -> the shard stride split remains
        # disjoint while the order changes epoch-to-epoch.
        self._epoch += 1
