"""Load a trained RoboCasa System1 checkpoint and sanity-check its loss on a few
training samples.

Usage:
    ROBOCASA_SHARDS_DIR=s3://.../system1_full_0711/shards \
    AWS_PROFILE=sagemaker \
    uv run python scripts/eval_ckpt_loss.py \
        --ckpt .local_ckpt/smoke3k_2999 \
        --config pi05_robocasa_system1 \
        --num-batches 8 --batch-size 4

Expectation for the 3k smoke ckpt: flow_loss ~0.02, progress_loss ~0.77.
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="local checkpoint step dir (contains params/ and assets/)")
    p.add_argument("--config", default="pi05_robocasa_system1")
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = pathlib.Path(args.ckpt).resolve()
    params_dir = ckpt / "params"
    assert params_dir.exists(), f"no params/ under {ckpt}"

    config = _config.get_config(args.config)
    # Shrink the batch for a quick pass. norm_stats come from the repo's baked
    # ./assets/<config>/robocasa_system1/norm_stats.json, which we verified is
    # byte-identical to the checkpoint's baked assets — so normalization matches
    # exactly what training used.
    import dataclasses

    config = dataclasses.replace(
        config,
        batch_size=args.batch_size,
        num_workers=0,
    )

    print(f"jax devices: {jax.devices()}")
    print(f"Loading params from {params_dir} ...")
    params = _model.restore_params(params_dir, restore_type=jax.Array, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    print("Checkpoint loaded into model OK.")

    # Data loader over the (S3) shards; norm stats come from the ckpt assets.
    loader = _data_loader.create_data_loader(
        config,
        sharding=None,
        shuffle=True,
        num_batches=args.num_batches,
        framework="jax",
    )

    rng = jax.random.key(args.seed)
    flow_vals, prog_vals, total_vals = [], [], []
    acc_vals, mae_vals = [], []
    for i, (obs, act) in enumerate(loader):
        rng, step_rng = jax.random.split(rng)
        loss, metrics = model.compute_loss(step_rng, obs, act, train=False, return_metrics=True)
        flow = float(metrics.get("flow_loss", jnp.nan))
        prog = float(metrics.get("progress_loss", jnp.nan))
        acc = float(metrics.get("progress_acc", jnp.nan))
        mae = float(metrics.get("progress_class_mae", jnp.nan))
        total = float(jnp.mean(loss))
        flow_vals.append(flow)
        prog_vals.append(prog)
        total_vals.append(total)
        acc_vals.append(acc)
        mae_vals.append(mae)
        print(
            f"batch {i:2d}: flow_loss={flow:.4f}  progress_loss={prog:.4f}  "
            f"progress_acc={acc:.3f}  progress_class_mae={mae:.3f}  total={total:.4f}"
        )

    def stat(name, xs):
        a = np.array(xs, dtype=np.float64)
        print(f"  {name:20s} mean={a.mean():.4f}  std={a.std():.4f}  n={len(a)}")

    print(f"\n=== averaged over {len(flow_vals)} batches (bs={args.batch_size}) ===")
    stat("flow_loss", flow_vals)
    stat("progress_loss", prog_vals)
    stat("progress_acc", acc_vals)
    stat("progress_class_mae", mae_vals)
    stat("total_loss", total_vals)


if __name__ == "__main__":
    main()
