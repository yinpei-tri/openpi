"""Streaming WebDataset loader for System1 RoboCasa shards.

Reads the per-frame, globally-shuffled tar shards written by RoboAnnotator's
``producers/preprocess_robocasa_to_tar.py`` (one tar = many frames from many
episodes) and yields the SAME pre-transform sample contract that
``policies/robocasa_policy.py::RobocasaInputs`` consumes — so the model / transform
/ norm path is identical to any other openpi dataset.

Design (see the System1 plan):
- Shards are streamed sequentially (one big read; works for local files and S3 via
  boto3), which is why the producer packs each shard with frames from MANY episodes
  pre-shuffled — a small reservoir buffer then decorrelates cheaply.
- Two random axes at load (config probs): LEVEL (milestone vs child, p_milestone)
  picks the conditioning span → drives the prompt span, progress label, and which
  stored action chunk; PHRASING (subgoal vs subgoal_detail, p_detail) is prompt
  text only. Pad mode {subtask, episode} selects which stored chunk.
- Anchor = the chosen span's START frame, embedded in the sample by the producer
  (child anchor falls back to the milestone anchor when identical — dedup flag).

This is an IterableDataset; it plugs into ``transform_iterable_dataset`` +
``create_robocasa_webdataset_data_loader`` in ``data_loader.py``.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path
import random
import tarfile
from typing import Any, Iterator
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from openpi.policies import robocasa_policy as _rp

CAM_KEYS = ("scene_left", "scene_right", "wrist")


@dataclasses.dataclass(frozen=True)
class WebDatasetConfig:
    """Load-time knobs for streaming the RoboCasa shards."""

    shards: str  # glob-ish dir or brace pattern; local path or s3://bucket/prefix
    action_horizon: int = 20
    include_base_pos: bool = True
    use_anchor_images: bool = True
    # Anchor store (path-addressable). If None, derived as the `anchors/` sibling of
    # `shards`. Anchors are NOT embedded in shards (~1M frames at full scale); the
    # loader reads them by key on demand. For S3, STAGE this dir to local disk first
    # (the OS page cache then serves hot anchors from RAM — no per-sample S3 GET).
    anchors_dir: str | None = None
    # Random axes.
    subgoal_level: str = "child"  # "milestone" | "child" | "mixed"
    p_milestone: float = 0.5  # P(use milestone) when subgoal_level == "mixed"
    p_detail: float = 0.0  # P(prompt = subgoal_detail instead of subgoal)
    # Action padding: which stored chunk to use as the label.
    subgoal_action_pad: str = "subtask"  # "subtask" | "episode"
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
def list_shards(shards: str) -> list[str]:
    """Resolve a shards spec to a sorted list of tar URLs (local paths or s3://)."""
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


def open_shard(url: str) -> io.BufferedReader | io.BytesIO:
    """Open a shard URL as a binary stream (local file or full S3 object)."""
    if url.startswith("s3://"):
        import boto3

        u = urlparse(url)
        s3 = boto3.client("s3")
        body = s3.get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
        return io.BytesIO(body)
    return open(url, "rb")  # noqa: SIM115 (closed by tarfile context)


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
        # Approximate length from the manifest if present (for len()).
        self._len = self._read_manifest_count()
        # Anchor store root: explicit, else the `anchors/` sibling of `shards`.
        if cfg.anchors_dir is not None:
            self._anchors_root = cfg.anchors_dir.rstrip("/")
        elif cfg.shards.startswith("s3://"):
            self._anchors_root = cfg.shards.rstrip("/").rsplit("/", 1)[0] + "/anchors"
        else:
            sp = Path(cfg.shards)
            base = sp if sp.is_dir() else sp.parent
            self._anchors_root = str((base.parent / "anchors") if base.name == "shards" else (base / "anchors"))
        self._anchor_cache: dict[str, np.ndarray] = {}

    def _read_anchor(self, key: str, cam: str) -> np.ndarray:
        """Resolve an anchor key (<flat_id>/f<frame:06d>) + cam to a decoded image.

        Reads ``<anchors_root>/<key>.<cam>.jpg`` (local or s3://) with a small LRU.
        For S3, stage the anchor store locally first so this is a local read.
        """
        cache_key = f"{key}.{cam}"
        cached = self._anchor_cache.get(cache_key)
        if cached is not None:
            return cached
        url = f"{self._anchors_root}/{key}.{cam}.jpg"
        if url.startswith("s3://"):
            import boto3
            from urllib.parse import urlparse as _up

            u = _up(url)
            data = boto3.client("s3").get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
        else:
            data = Path(url).read_bytes()
        img = _decode_jpeg(data)
        if len(self._anchor_cache) >= self.cfg.anchor_cache_size:
            self._anchor_cache.pop(next(iter(self._anchor_cache)))
        self._anchor_cache[cache_key] = img
        return img

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _read_manifest_count(self) -> int:
        # Local manifest only; S3 length is approximate / optional.
        if self.cfg.shards.startswith("s3://"):
            return 0
        d = Path(self.cfg.shards)
        man = (d / "manifest.jsonl") if d.is_dir() else (d.parent / "manifest.jsonl")
        if man.exists():
            return sum(json.loads(line)["num_samples"] for line in man.read_text().splitlines() if line.strip())
        return 0

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

        # --- axis 1: level ---
        if self.cfg.subgoal_level == "milestone":
            use_milestone = True
        elif self.cfg.subgoal_level == "child":
            use_milestone = False
        else:  # mixed
            use_milestone = rng.random() < self.cfg.p_milestone
        level = meta["milestone"] if use_milestone else meta["child"]

        # --- axis 2: phrasing ---
        use_detail = rng.random() < self.cfg.p_detail
        phrasings = level["subgoal_detail"] if (use_detail and level.get("subgoal_detail")) else level["subgoal"]
        prompt = rng.choice(phrasings) if phrasings else ""

        # --- arrays (state + 3 action chunks + masks) ---
        arrays = np.load(io.BytesIO(raw["arrays.npz"]))

        # --- images + state ---
        # State is already PREPROCESSED (lean, base x/y/yaw pre-made relative) in the
        # shards; RobocasaInputs passes it through (dim-detects lean vs raw).
        sample: dict[str, Any] = {
            "observation/scene_left": _decode_jpeg(raw["scene_left.jpg"]),
            "observation/scene_right": _decode_jpeg(raw["scene_right.jpg"]),
            "observation/wrist": _decode_jpeg(raw["wrist.jpg"]),
            "observation/state": arrays["state"].astype(np.float32),
        }

        if self.cfg.use_anchor_images:
            # Anchor for the CHOSEN level, read by key from the (path-addressable)
            # anchor store. Child anchor falls back to the milestone anchor when the
            # producer deduped them (anchor_child_is_milestone / child key is None).
            if use_milestone or meta.get("anchor_child_is_milestone") or meta.get("anchor_child_key") is None:
                anchor_key = meta["anchor_milestone_key"]
            else:
                anchor_key = meta["anchor_child_key"]
            for ck in CAM_KEYS:
                sample[f"observation/anchor_{ck}"] = self._read_anchor(anchor_key, ck)

        # --- actions (chosen pad variant) ---
        if self.cfg.subgoal_action_pad == "episode":
            actions = arrays["action_episode"]
        else:
            actions = arrays["action_subtask_milestone" if use_milestone else "action_subtask_child"]
        sample["actions"] = actions.astype(np.float32)

        # --- targets / meta ---
        sample["prompt"] = prompt
        sample["progress_frac"] = np.float32(level["progress_frac"])
        sample["subgoal_start"] = np.int32(level["start"])
        sample["subgoal_end"] = np.int32(level["end"])
        sample["frame_index"] = np.int32(meta["frame_index"])
        return sample

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
