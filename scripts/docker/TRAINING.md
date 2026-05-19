# Containerized training for openpi

This directory contains a Dockerfile + compose file purpose-built for **fine-tuning**
(`scripts/train.py` and `scripts/train_pytorch.py`). The serve-policy image
(`serve_policy.Dockerfile`, `compose.yml`) is unchanged and remains the canonical
inference path.

## Files

- `train.Dockerfile` — CUDA 12.2 + uv. Installs the locked deps, bakes the
  openpi package into the venv, and applies the `models_pytorch/transformers_replace/`
  patch so the same image runs the JAX *or* the PyTorch trainer. Built to run
  under any UID, so the container can adopt your host UID and write
  bind-mounted files with the right ownership.
- `train.compose.yml` — One service (`trainer`) that bind-mounts the repo plus
  the host caches, exposes GPUs via `runtime: nvidia` (snap-Docker compatible),
  and launches `scripts/train.py` by default.
- `.env.example` — Copy to `.env` and fill in absolute paths for your caches.

## One-time host setup

1. Pre-fetch the base checkpoint and dataset to the host so the container
   doesn't redownload at every run start:

   ```bash
   uv run python -c "from openpi.shared import download; \
       print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"
   uv run python -c "from huggingface_hub import snapshot_download; \
       print(snapshot_download('physical-intelligence/libero', repo_type='dataset'))"
   ```

   Defaults land in `~/.cache/openpi` and `~/.cache/huggingface`.

2. (PyTorch only) Convert the JAX base model to PyTorch on the host:

   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
       --config_name pi05_libero \
       --output_path ./checkpoints/pi05_base_pytorch
   ```

   The `pi05_libero_debug` config already points `pytorch_weight_path` at this
   location.

3. Compute norm stats once per config. The debug config reuses
   `pi05_libero`'s assets, so you only need to run this once:

   ```bash
   # If your host has more GPUs than evenly divide batch_size (e.g. 3 GPUs and
   # batch_size=256), pin to a single GPU. JAX will otherwise try to shard the
   # batch and fail with `dim 0 should be divisible by N` from pjit.
   CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
       --config-name pi05_libero --max-frames 5000
   ```

   `--max-frames 5000` keeps the script fast for debugging; drop it for the
   real run on the DGX. The output lands in
   `assets/pi05_libero/physical-intelligence/libero/norm_stats.json` and is
   bind-mounted into the container along with the rest of the repo.

4. Copy the env template:

   ```bash
   cp scripts/docker/.env.example scripts/docker/.env
   # then edit scripts/docker/.env to point at your absolute cache paths
   ```

## Why the env file uses absolute paths

If you installed Docker via snap (the default on Ubuntu 22.04 if you ran
`apt install docker.io` from snap), the daemon is confined and resolves `~`
**and `$HOME`** to its private cache (`~/snap/docker/.../...`). That makes
shell-style `~/.cache/openpi` mounts silently point at empty directories.
The compose file therefore expects `OPENPI_DATA_HOME` and `HF_HOME` to be
absolute paths supplied via `.env`.

If you migrate to the official apt or rootless Docker, `~` works again — but
absolute paths still work everywhere.

## Local 2x A6000 smoke run (JAX)

Default `TRAIN_CONFIG=pi05_libero_debug` (defined in
`src/openpi/training/config.py`): `batch_size=4`, `fsdp_devices=2`,
`num_train_steps=200`, no EMA. Designed to fit on two 48 GB A6000s and finish
in minutes.

```bash
docker compose -f scripts/docker/train.compose.yml up --build
```

> **GPU pinning required on hosts with >2 GPUs.** JAX shards `batch_size=4`
> across every device it can see; if `NVIDIA_VISIBLE_DEVICES=all` exposes 3
> GPUs you'll get `ValueError: Batch size 4 must be divisible by the number of
> devices 3.` Set `NVIDIA_VISIBLE_DEVICES=0,1` in `scripts/docker/.env` (or
> inline before the compose call) before launching.

## Local 2x A6000 smoke run (PyTorch)

```bash
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_libero_debug --exp_name=pt_debug" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

The PyTorch trainer now mirrors the JAX trainer for the features that matter
on this image:

- **FSDP2** is the default for any multi-GPU PyTorch run — with
  `nproc_per_node=2` you automatically get pure parameter sharding across both
  ranks (no extra flags needed). To use a hybrid FSDP x DP layout, set
  `--fsdp_devices K` (with `K | world_size`, `K > 1`) — FSDP2 then shards
  across K ranks and replicates across the remaining `world_size // K` groups,
  matching the JAX mesh. The `pi05_libero_debug` config keeps
  `fsdp_devices=2` for explicitness, but the default value of 1 now also means
  "shard across all ranks" on the PyTorch side.
- **EMA** is on by default (`config.ema_decay = 0.99` in `TrainConfig`).
  Disable per-run with `--ema_decay None`. The debug config explicitly sets
  `ema_decay=None`, so EMA is off there; `pi05_libero` (the production config)
  inherits the default and has it on.
- **`torch.compile` on the training forward** is opt-in — append
  `--model.pytorch_compile_train` to the `TRAIN_CMD`.
- **Gradient checkpointing** is on by default; opt out with
  `--model.pytorch_gradient_checkpointing False`.

Still **not** supported on the PyTorch path: π₀-FAST, LoRA, mixed precision
(use full bf16 or full fp32 via `pytorch_training_precision`).

## Single-node DGX A100 run (JAX, full pi05_libero)

On the DGX, after re-running the host-setup steps above:

```bash
TRAIN_CONFIG=pi05_libero \
EXP_NAME=libero_a100 \
TRAIN_ARGS="--fsdp-devices=8" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

`pi05_libero` defaults to `batch_size=256`, `num_train_steps=30_000`, and EMA on
— it's tuned for an 8x A100 80 GB box. Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
in the env file for full memory.

## Single-node DGX A100 run (PyTorch, full pi05_libero)

Same image, swap the entrypoint via `TRAIN_CMD`. Note the flag style: the JAX
trainer uses `--fsdp-devices` (tyro), the PyTorch trainer uses `--fsdp_devices`
(argparse). On the PyTorch side, FSDP2 across all 8 GPUs is the default — no
extra flags needed:

```bash
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

To match the JAX mesh (FSDP across 4, DP across 2) on 8 GPUs:

```bash
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt_mesh --fsdp_devices=4" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

Add `--model.pytorch_compile_train` once you've validated the run; the
inference forward is compiled either way.

## SageMaker note

When you move to SageMaker, this image is a starting point, but SageMaker
expects:

- The training entrypoint to read `SM_CHANNEL_*` env vars instead of bind
  mounts (or you stage data from S3 to `/opt/ml/input/data/...`).
- `OPENPI_DATA_HOME` to point at a writable location under `/opt/ml/`.
- The container to write checkpoints to `/opt/ml/checkpoints/` for them to be
  uploaded back to S3.

Adapt `train.Dockerfile`'s default CMD / wrap it in a SageMaker-style
`train` shell script when you get there.
