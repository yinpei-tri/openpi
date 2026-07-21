"""Eval #1 — VALIDATION action-MSE (+ progress metrics) for System1 checkpoints.

In-process (no policy server): load a checkpoint's params, stream the held-out validation
WebDataset (``system1_midset_0717_val``) through the SAME data pipeline used in training, and
compare the model's SAMPLED action chunk against the ground-truth chunk. The headline metric is
``action_mse`` (sampled vs GT, in NORMALIZED action space — the space both live in, comparable
across proact/procls/proreg). We also report ``flow_loss`` and the per-method progress metric
via ``compute_loss(return_metrics=True)``.

Determinism / parity: the checkpoint's architecture + prompt format is resolved from the ckpt
itself (config.json / dir-name tag) via ``resolve_robocasa_config``, and norm_stats come from the
ckpt's own baked ``assets/robocasa_system1/norm_stats.json`` — so the validation input matches how
THIS checkpoint was trained (nostate/notask/... ablations get their own prompt).

Usage (one run dir -> all step ckpts, or a single step dir):
    ROBOCASA_VAL_SHARDS=s3://.../system1_midset_0717_val/shards AWS_PROFILE=sagemaker \
    uv run python scripts/eval_val_mse.py \
        --run-dir checkpoints/m0717-50k-bs512-v1__progcls_granfine_verbsimp \
        --num-batches 16 --batch-size 8 \
        --out eval_out/val_mse/m0717-50k-bs512-v1__progcls_granfine_verbsimp.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

# Sibling import (this file runs as a script; scripts/ is sys.path[0], not the repo root).
from train import resolve_robocasa_config  # noqa: E402


def _val_shards() -> str:
    v = os.environ.get("ROBOCASA_VAL_SHARDS")
    if not v:
        raise SystemExit("Set ROBOCASA_VAL_SHARDS to the validation shards (local dir or s3://...).")
    return v


def _step_dirs(run_dir: pathlib.Path, only_step: int | None) -> list[pathlib.Path]:
    """Numeric step subdirs (each with params/) under a run dir, or the dir itself if it IS one."""
    if (run_dir / "params").is_dir():
        return [run_dir]
    steps = sorted(
        (d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit() and (d / "params").is_dir()),
        key=lambda d: int(d.name),
    )
    if only_step is not None:
        steps = [d for d in steps if int(d.name) == only_step]
    return steps


def _eval_one_ckpt(step_dir: pathlib.Path, args) -> dict:
    """Load one checkpoint step and compute val metrics over a fixed batch set."""
    # 1) Resolve the exact TrainConfig (arch + prompt format) from the checkpoint itself.
    config = resolve_robocasa_config(step_dir)
    if config is None:
        raise SystemExit(f"Could not resolve a RoboCasa config from {step_dir} (no config.json / tag).")

    # 2) Point the data pipeline at the VAL shards, and resolve norm_stats from THIS ckpt's own
    #    baked assets/ dir (via AssetsConfig.assets_dir), so normalization matches how the ckpt
    #    was trained. create_data_loader builds the DataConfig internally from config.data, so we
    #    steer it entirely through the config (no data_config kwarg exists).
    ckpt_norm = step_dir / "assets" / "robocasa_system1" / "norm_stats.json"
    if not ckpt_norm.is_file():
        raise SystemExit(f"Missing ckpt norm_stats: {ckpt_norm}")

    data = dataclasses.replace(
        config.data,
        shards=_val_shards(),
        # Resolve norm_stats from <step_dir>/assets/robocasa_system1/norm_stats.json.
        assets=_config.AssetsConfig(assets_dir=str(step_dir / "assets"), asset_id="robocasa_system1"),
    )
    config = dataclasses.replace(config, data=data, batch_size=args.batch_size, num_workers=args.num_workers)

    params = _model.restore_params(step_dir / "params", restore_type=jax.Array, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()

    loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=True,
        num_batches=args.num_batches,
        framework="jax",
    )

    rng = jax.random.key(args.seed)
    mse_vals, flow_vals, prog_vals, acc_vals, mae_vals = [], [], [], [], []
    for obs, act in loader:
        rng, s_rng, l_rng = jax.random.split(rng, 3)
        # action_mse: sampled vs GT, in normalized action space (both are post-Normalize here).
        pred = model.sample_actions(s_rng, obs, num_steps=args.flow_steps)
        # Only score the REAL action dims (exclude zero-pad to model action_dim, and the proact
        # 12th progress dim which is not a control action). Use the GT chunk's finite region.
        real_dim = getattr(config.model, "flow_loss_real_dim", None) or act.shape[-1]
        mse = float(jnp.mean((pred[..., :real_dim] - act[..., :real_dim]) ** 2))
        mse_vals.append(mse)

        _, metrics = model.compute_loss(l_rng, obs, act, train=False, return_metrics=True)
        flow_vals.append(float(metrics.get("flow_loss", jnp.nan)))
        prog_vals.append(float(metrics.get("progress_loss", jnp.nan)))
        acc_vals.append(float(metrics.get("progress_acc", jnp.nan)))
        mae_vals.append(float(metrics.get("progress_class_mae", jnp.nan)))

    def m(xs):
        a = np.array(xs, np.float64)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else None

    return dict(
        step=int(step_dir.name) if step_dir.name.isdigit() else None,
        action_mse=m(mse_vals),
        flow_loss=m(flow_vals),
        progress_loss=m(prog_vals),
        progress_acc=m(acc_vals),
        progress_class_mae=m(mae_vals),
        n_batches=len(mse_vals),
        batch_size=args.batch_size,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="ckpt run dir (all step subdirs) OR a single step dir")
    ap.add_argument("--only-step", type=int, default=None, help="restrict to one step")
    ap.add_argument("--num-batches", type=int, default=1024, help="1024 x batch-size = samples/step (default 8192)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="data-loader workers; the val WebDataset (6 JPEGs/sample) is decode-bound, "
                         "so 0 makes it GPU-starved. 8 keeps the GPU fed.")
    ap.add_argument("--flow-steps", type=int, default=10, help="flow-matching integration steps for sampling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    steps = _step_dirs(run_dir, args.only_step)
    if not steps:
        raise SystemExit(f"No step ckpts (with params/) found under {run_dir}")

    exp_name = run_dir.name
    print(f"jax devices: {jax.devices()}")
    print(f"exp={exp_name}  steps={[s.name for s in steps]}  val={_val_shards()}")

    results = []
    for sd in steps:
        print(f"--- {sd.name} ---")
        r = _eval_one_ckpt(sd, args)
        results.append(r)
        print(f"    action_mse={r['action_mse']}  flow_loss={r['flow_loss']}  "
              f"progress_loss={r['progress_loss']}  acc={r['progress_acc']}  mae={r['progress_class_mae']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(exp_name=exp_name, val_shards=_val_shards(), steps=results), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
