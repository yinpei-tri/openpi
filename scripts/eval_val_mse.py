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

    # Disable the reservoir shuffle for the val metric pass. RoboCasaDataConfig.create() bakes
    # `shuffle_buffer` into the WebDatasetConfig, and the iterable fills that reservoir (default
    # 16000) BEFORE yielding the first sample — a decode-bound warmup that leaves the GPU idle for
    # minutes (create_data_loader's shuffle=False does NOT touch it). A metric eval over a fixed
    # val set needs no decorrelation, so shuffle_buffer=1: samples stream straight through and the
    # first batch lands immediately.
    data = dataclasses.replace(
        config.data,
        shards=_val_shards(),
        shuffle_buffer=1,
        # Resolve norm_stats from <step_dir>/assets/robocasa_system1/norm_stats.json.
        assets=_config.AssetsConfig(assets_dir=str(step_dir / "assets"), asset_id="robocasa_system1"),
    )
    config = dataclasses.replace(config, data=data, batch_size=args.batch_size, num_workers=args.num_workers)

    params = _model.restore_params(step_dir / "params", restore_type=jax.Array, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()

    # JIT the model calls (freeze state), exactly like Policy.infer's fast serving path. WITHOUT
    # this the 3B PaliGemma + 10 flow-integration steps run EAGERLY — every op dispatched from
    # Python one at a time, no XLA fusion, GPU ~0% util, ~50s/batch. Jitted, it compiles once then
    # runs as one fused GPU program (~1s/batch after the first). num_steps is a Python constant ->
    # static_argnames so it doesn't trigger retracing.
    from openpi.shared import nnx_utils
    # Determine the ACTIVE progress mode from BOTH data + model config, not model.progress_mode
    # alone. For progact, progress_as_action=True + use_progress_head=False, but progress_mode is
    # left at its inherited "classes" — reading it alone would misroute progact to the (disabled)
    # head path and silently skip progress. Precedence: action (data flag) > head mode > none.
    has_head = bool(getattr(config.model, "use_progress_head", False))
    if bool(getattr(config.data, "progress_as_action", False)):
        prog_mode = "action"
    elif has_head:
        prog_mode = getattr(config.model, "progress_mode", None)
    else:
        prog_mode = "none"
    # Head variants (progcls/progreg): sample actions AND read the progress head in ONE prefix
    # forward pass (sample_actions_with_progress) — calling sample_actions + predict_progress
    # separately re-runs the expensive 3B prefix twice (~2x). progact/no-head: plain sample_actions
    # (progress is the action's extra dim, read from the chunk below).
    if has_head:
        sample_prog_fn = nnx_utils.module_jit(model.sample_actions_with_progress, static_argnames=("num_steps",))
        sample_fn = None
    else:
        sample_fn = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
        sample_prog_fn = None

    loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        # No shuffle for a metric eval: a fixed deterministic prefix of the val set (the shards
        # are already globally shuffled at build time). shuffle=True would force the reservoir
        # buffer (16k default) to fill from S3/disk before the FIRST batch — the GPU-starving
        # slowdown we measured. Streaming shards directly yields batches immediately.
        shuffle=False,
        num_batches=args.num_batches,
        framework="jax",
    )

    import logging, time
    log = logging.getLogger("val_mse")
    rng = jax.random.key(args.seed)
    # We compute ONLY inference outputs vs ground truth (no training losses):
    #   action_mse   = sampled action chunk vs GT chunk, in normalized action space
    #   progress_*   = predicted subgoal-progress vs GT (mode-dependent, see below)
    real_dim = getattr(config.model, "flow_loss_real_dim", None)
    mse_vals, prog_err_vals, prog_acc_vals = [], [], []
    _t = time.monotonic()
    for bi, (obs, act) in enumerate(loader):
        rng, s_rng = jax.random.split(rng)
        rd = real_dim or act.shape[-1]
        # Head variants: actions + progress from ONE prefix pass. Others: actions only.
        p = None
        if sample_prog_fn is not None:
            pred, p = sample_prog_fn(s_rng, obs, num_steps=args.flow_steps)
        else:
            pred = sample_fn(s_rng, obs, num_steps=args.flow_steps)
        # Score the REAL control dims only: exclude zero-pad to model action_dim AND the progact
        # 12th progress dim (handled separately as progress, not a control action).
        mse = float(jnp.mean((pred[..., :rd] - act[..., :rd]) ** 2))
        mse_vals.append(mse)

        # --- predicted progress vs GT ---
        # progress_mae is reported in a COMMON [0,1] fraction space across all variants so the
        # numbers are comparable: proreg already lives in [0,1]; progact lives in [-1,1] (2*frac-1)
        # and is rescaled to [0,1] via (x+1)/2 before the MAE.
        if prog_mode == "action":
            # progact (v2): NO independent head — progress IS the sampled action chunk's extra
            # (last) dim. Compare the FIRST-STEP progress (step 0 = this frame's progress), which
            # is the value System1 would act on. rd = real control dims, so index rd = progress dim.
            # Both pred & GT are in [-1,1]; rescale to [0,1] to match proreg's scale.
            pred_frac = (pred[:, 0, rd] + 1.0) / 2.0
            gt_frac = (act[:, 0, rd] + 1.0) / 2.0
            prog_err_vals.append(float(jnp.mean(jnp.abs(pred_frac - gt_frac))))
        elif p is not None:  # head variant: progress came from sample_actions_with_progress
            if prog_mode == "classes" and obs.progress_class is not None:
                pred_cls = jnp.argmax(p, axis=-1)
                gt_cls = obs.progress_class.reshape(-1)
                prog_acc_vals.append(float(jnp.mean((pred_cls == gt_cls).astype(jnp.float32))))
                # Bucket MAE rescaled to [0,1]: classes are deciles 0..K-1, so dividing the
                # bucket-distance by (K-1) puts it on the same [0,1] progress scale as reg/act.
                k = p.shape[-1]
                prog_err_vals.append(float(jnp.mean(jnp.abs(pred_cls - gt_cls).astype(jnp.float32)) / max(k - 1, 1)))
            elif obs.progress is not None:  # continuous (proreg) — already [0,1]
                prog_err_vals.append(float(jnp.mean(jnp.abs(p.reshape(-1) - obs.progress.reshape(-1)))))

        dt = time.monotonic() - _t; _t = time.monotonic()
        # Batch 1 includes one-time JIT compile of the jitted sample/progress fns (GPU ~0% during
        # graph build); steady-state batches are ~1s. Log both so the sweep is sized from steady.
        log.info("batch %d/%d: +%.1fs  action_mse=%.4f%s", bi + 1, args.num_batches, dt, mse,
                 "  (incl. JIT)" if bi == 0 else "")

    def m(xs):
        a = np.array(xs, np.float64)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else None

    return dict(
        step=int(step_dir.name) if step_dir.name.isdigit() else None,
        action_mse=m(mse_vals),
        # progress vs GT: for classes -> accuracy + bucket-MAE; continuous/action -> MAE. None if
        # the variant surfaces no comparable progress signal.
        progress_acc=m(prog_acc_vals) if prog_acc_vals else None,
        progress_mae=m(prog_err_vals) if prog_err_vals else None,
        progress_mode=prog_mode,
        n_batches=len(mse_vals),
        batch_size=args.batch_size,
    )


def main():
    import logging
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
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

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for sd in steps:
        print(f"--- {sd.name} ---")
        r = _eval_one_ckpt(sd, args)
        results.append(r)
        print(f"    action_mse={r['action_mse']}  progress_acc={r['progress_acc']}  "
              f"progress_mae={r['progress_mae']}  ({r['progress_mode']})")
        # Write incrementally after EACH checkpoint: a later ckpt failing (OOM, bad params)
        # then can't discard the completed ones. Overwrites with the growing list each time.
        out.write_text(json.dumps(dict(exp_name=exp_name, val_shards=_val_shards(), steps=results), indent=2))

    print(f"wrote {out}  ({len(results)} checkpoint(s))")


if __name__ == "__main__":
    main()
