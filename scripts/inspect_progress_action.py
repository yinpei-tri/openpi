"""Inspect RoboCasa samples one-by-one: raw action (before) vs the fully-processed
32-d action (after), with the augmented progress-as-action dim broken out.

Reads SEQUENTIALLY (shuffle_buffer=1 -> natural shard order, no reservoir reorder),
so consecutive frames of an episode appear in order. Read-only.

Usage:
    ROBOCASA_SHARDS_DIR=/path/to/shards \
    uv run python scripts/inspect_progress_action.py --start 0 --num 5
    # or point at a local/S3 shards dir explicitly:
    uv run python scripts/inspect_progress_action.py --shards s3://.../system1_check/shards --num 3

Notes:
- Requires the config's norm_stats (assets/pi05_robocasa_system1/robocasa_system1/) to be
  installed for the AFTER view; --no-after skips normalization and just shows the raw + the
  computed progress vector (norm_stats-independent).
"""

import argparse
import dataclasses
import os

import numpy as np

from openpi.training import robocasa_webdataset as w
import openpi.training.config as _config
import openpi.transforms as _transforms

np.set_printoptions(precision=3, suppress=True, linewidth=200)

# 11-d lean action column names (dim 11 = appended progress).
LEAN_COLS = ["base_vx", "base_vy", "yaw_vel", "ctrl_mode", "eef_dx", "eef_dy", "eef_dz", "d_roll", "d_pitch", "d_yaw", "grip"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default=os.environ.get("ROBOCASA_SHARDS_DIR"), help="shards dir (local or s3://)")
    p.add_argument("--start", type=int, default=0, help="skip this many samples first")
    p.add_argument("--num", type=int, default=5, help="how many samples to print")
    p.add_argument("--config", default="pi05_robocasa_system1")
    p.add_argument("--no-after", action="store_true", help="skip the normalized 'after' view (no norm_stats needed)")
    p.add_argument("--full", action="store_true", help="print the full 32-d action (else dims 0-11)")
    args = p.parse_args()
    assert args.shards, "set --shards or ROBOCASA_SHARDS_DIR"

    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, progress_as_action=True),
        model=dataclasses.replace(cfg.model, use_progress_head=False),
    )
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    # shuffle_buffer=1 => samples emitted in natural shard order (no reservoir reorder).
    settings = dataclasses.replace(dc.robocasa_webdataset_settings, shards=args.shards, shuffle_buffer=1)
    ds = w.RoboCasaWebDataset(settings)

    chain = None
    if not args.no_after:
        chain = _transforms.compose(
            [
                *dc.repack_transforms.inputs,
                *dc.data_transforms.inputs,
                _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
                *dc.model_transforms.inputs,
            ]
        )

    print(f"shards={args.shards}  showing samples [{args.start}, {args.start + args.num})\n")
    for i, s in enumerate(ds):
        if i < args.start:
            continue
        if i >= args.start + args.num:
            break
        frac = float(s["progress_frac"])
        ss, se, fi = int(s["subgoal_start"]), int(s["subgoal_end"]), int(s["frame_index"])
        prompt = s.get("prompt", "")
        pa = np.asarray(s["progress_action"])  # normalized [-1,1], per step
        print("=" * 100)
        print(f"[sample {i}]  frame={fi}  span=[{ss},{se}] (len {se - ss})  progress_frac={frac:.4f}")
        print(f"  subgoal: {prompt!r}")

        print("  BEFORE — raw lean action from shard (11-d, subgoal settle-padded):")
        raw = np.asarray(s["actions"])
        print("   cols:", " ".join(f"{c:>8}" for c in LEAN_COLS))
        for t in range(raw.shape[0]):
            print(f"   t{t:02d} " + " ".join(f"{v:8.3f}" for v in raw[t]))

        print("  progress-as-action (un-normalized [0,1]):")
        print("   ", np.round((pa + 1) / 2, 3).tolist())
        print("  progress-as-action (normalized [-1,1], appended as action dim 11):")
        print("   ", np.round(pa, 3).tolist())

        if chain is not None:
            act = np.asarray(chain(dict(s))["actions"])  # [h, 32]
            cols = act.shape[1] if args.full else 12
            print(f"  AFTER — processed action[:, :{cols}] (0-10 normalized real, 11 progress, 12+ zero-pad):")
            for t in range(act.shape[0]):
                print(f"   t{t:02d} " + " ".join(f"{v:7.3f}" for v in act[t, :cols]))
        print()


if __name__ == "__main__":
    main()
