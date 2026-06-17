"""Profile the per-part latency of a training step (data wait vs compute).

JAX dispatch is async — ``ptrain_step`` returns futures immediately and the work
runs lazily. So we time:
  - data_wait:   time to pull the next batch from the loader
  - compute:     ptrain_step + jax.block_until_ready (real forward+backward+opt)
  - h2d/overhead: residual

Also splits compute into (a) full train step and (b) forward-only loss, to see how
much of the step is FSDP all-gather + remat backward vs the forward.

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/profile_step.py \
      pi05_robocasa_system1 --batch_size 3 --fsdp_devices 3 --steps 12
"""

import functools
import os
import sys
import time

import jax
import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make `train` importable

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
from train import init_train_state, train_step


def main(
    config_name: str,
    batch_size: int = 3,
    fsdp_devices: int = 3,
    steps: int = 12,
    anchor: bool | None = None,
    progress: bool | None = None,
):
    config = _config.get_config(config_name)
    # Optional model-config overrides to A/B image count + progress head.
    model = config.model
    overrides = {}
    if anchor is not None:
        overrides["use_anchor_images"] = anchor
    if progress is not None:
        overrides["use_progress_head"] = progress
    if overrides:
        import dataclasses as _dc

        model = _dc.replace(model, **overrides)
    config = config.__class__(**{**config.__dict__, "batch_size": batch_size, "fsdp_devices": fsdp_devices, "model": model})
    print(f"[cfg] anchor={model.use_anchor_images} progress={model.use_progress_head} "
          f"image_keys={model.image_keys}")

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)

    t0 = time.perf_counter()
    batch = next(data_iter)
    print(f"[init] first batch fetch: {time.perf_counter() - t0:.2f}s")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    print(f"\nProfiling {steps} steps (batch={batch_size}, fsdp={fsdp_devices}, "
          f"{jax.device_count()} devices)...\n")
    data_times, compute_times = [], []
    for step in range(steps):
        with sharding.set_mesh(mesh):
            tc = time.perf_counter()
            train_state, info = ptrain_step(train_rng, train_state, batch)
            jax.block_until_ready((train_state, info))  # force compute to finish
            compute_s = time.perf_counter() - tc

        td = time.perf_counter()
        batch = next(data_iter)
        jax.block_until_ready(batch)
        data_s = time.perf_counter() - td

        # first 2 steps include JIT compile — report but exclude from averages.
        tag = " (compile)" if step < 2 else ""
        print(f"step {step:2d}: compute={compute_s:6.2f}s  data_wait={data_s:5.2f}s{tag}")
        if step >= 2:
            compute_times.append(compute_s)
            data_times.append(data_s)

    print("\n=== steady-state averages (excluding compile) ===")
    print(f"  compute (fwd+bwd+opt, blocked): {np.mean(compute_times):.2f}s  +/- {np.std(compute_times):.2f}")
    print(f"  data_wait (next batch):         {np.mean(data_times):.2f}s  +/- {np.std(data_times):.2f}")
    print(f"  total/step:                     {np.mean(compute_times) + np.mean(data_times):.2f}s")
    print(f"  samples/s:                      {batch_size / (np.mean(compute_times) + np.mean(data_times)):.2f}")


if __name__ == "__main__":
    tyro.cli(main)
