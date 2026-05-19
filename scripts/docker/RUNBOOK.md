# pi05 + LIBERO finetuning runbook

End-to-end commands for the three phases of bringing up `pi05_libero`
finetuning: local host smoke test, local Docker smoke test, and a single-node
8x A100 production run via Docker. Working directory is `~/openpi` everywhere.

For the architecture and design notes (snap-Docker quirks, why the env file
uses absolute paths, the SageMaker note), see [TRAINING.md](TRAINING.md). For
the PyTorch-specific feature matrix (FSDP2/EMA/`pytorch_compile_train`), see
the PyTorch Support section in the top-level [README.md](../../README.md).

---

## Phase 1 — Local 2x A6000 smoke test (host uv, no Docker)

One-time host setup:

```bash
# 1. uv env + submodules.
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 2. Apply the PyTorch transformers patch (no-op for the JAX trainer).
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/

# 3. Pre-fetch base ckpt + LIBERO dataset (cached under ~/.cache).
uv run python -c "from openpi.shared import download; \
    print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"
uv run python -c "from huggingface_hub import snapshot_download; \
    print(snapshot_download('physical-intelligence/libero', repo_type='dataset'))"

# 4. Convert JAX base -> PyTorch base (only needed for the PyTorch run).
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --config_name pi05_libero \
    --output_path ./checkpoints/pi05_base_pytorch

# 5. Norm stats (once — `pi05_libero_debug` reuses pi05_libero's assets).
#    Pin to one GPU so JAX doesn't try to shard batch_size=256 across 3 GPUs.
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name pi05_libero --max-frames 5000
```

JAX smoke run (2 GPUs, batch_size=4 from the config):

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_libero_debug --exp-name=jax_local --overwrite
```

PyTorch smoke run (2 GPUs, FSDP2 across both — multi-GPU now defaults to full
FSDP sharding; the debug config also keeps `fsdp_devices=2` explicitly):

```bash
CUDA_VISIBLE_DEVICES=0,1 \
    uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        scripts/train_pytorch.py pi05_libero_debug --exp_name=pt_local
```

Expected: ~200 steps, checkpoints under
`./checkpoints/pi05_libero_debug/{jax_local,pt_local}/`.

---

## Phase 2 — Local 2x A6000 smoke test (Docker, same machine)

```bash
# 1. Wire up .env (absolute paths required for snap-Docker).
cp scripts/docker/.env.example scripts/docker/.env
# Then edit scripts/docker/.env:
#   OPENPI_DATA_HOME=/home/yinpeidai/.cache/openpi
#   HF_HOME=/home/yinpeidai/.cache/huggingface
#   HOST_UID=$(id -u)         # → 1003 on this box
#   HOST_GID=$(id -g)
#   NVIDIA_VISIBLE_DEVICES=0,1   # MUST be 2 GPUs — pi05_libero_debug has batch_size=4
#                                # and JAX shards across every visible device. With 3
#                                # GPUs visible JAX errors: "Batch size 4 must be
#                                # divisible by the number of devices 3."
#   TRAIN_CONFIG=pi05_libero_debug
#   EXP_NAME=jax_docker

# 2. JAX run (default CMD).
NVIDIA_VISIBLE_DEVICES=0,1 EXP_NAME=jax_docker \
    docker compose -f scripts/docker/train.compose.yml up --build

# 3. PyTorch run — override TRAIN_CMD inline. Reuses the cached image.
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_libero_debug --exp_name=pt_docker" \
EXP_NAME=pt_docker \
    docker compose -f scripts/docker/train.compose.yml up
```

Verify: `ls -l ./checkpoints/pi05_libero_debug/{jax_docker,pt_docker}/` — files
should be owned by `yinpeidai:yinpeidai`, not root. If they're root-owned,
`HOST_UID/HOST_GID` in `.env` is wrong.

---

## Phase 3 — Single-node 8x A100 (Docker only)

On the A100 box, after a fresh `git clone --recurse-submodules`:

```bash
# 1. Same host-side prefetch as Phase 1, steps 3–5. You don't need uv sync on
#    the host for training — only for these prefetch commands. If uv isn't on
#    the A100 box, skip the uv calls and either:
#      (a) bind-mount the prefetched caches from your local box, or
#      (b) rsync ~/.cache/openpi and ~/.cache/huggingface over.
#    The PyTorch base checkpoint (./checkpoints/pi05_base_pytorch) likewise needs
#    to be present on the A100 host; rsync it or rerun convert_jax_model_to_pytorch.py
#    in a one-shot container (see TRAIN_CMD example at the end).
#    Run full norm stats (no --max-frames) once.
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 2. .env for the A100 box (absolute paths to wherever the caches live there).
cp scripts/docker/.env.example scripts/docker/.env
# Edit:
#   OPENPI_DATA_HOME=/abs/path/to/.cache/openpi
#   HF_HOME=/abs/path/to/.cache/huggingface
#   HOST_UID=$(id -u); HOST_GID=$(id -g)
#   NVIDIA_VISIBLE_DEVICES=all
#   WANDB_MODE=online        # if you want online logging
#   WANDB_API_KEY=...        # set in your shell, or here
```

Full pi05_libero JAX run, FSDP across all 8 GPUs:

```bash
TRAIN_CONFIG=pi05_libero \
EXP_NAME=libero_a100_jax \
TRAIN_ARGS="--fsdp-devices=8 --overwrite" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

Full pi05_libero PyTorch run, pure FSDP2 across all 8 GPUs (default — no flag
needed):

```bash
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

PyTorch with the JAX-style mesh (FSDP across 4, DP across 2) — usually faster
than pure FSDP-8 because the all-gather is only 4-way:

```bash
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt_mesh --fsdp_devices=4" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

Optional: enable training-forward compile after you've confirmed the run is
stable:

```bash
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt_compile \
    --model.pytorch_compile_train" \
    docker compose -f scripts/docker/train.compose.yml up --build
```

### Side-by-side JAX vs PyTorch on one A100 node (4 GPUs each)

Run two parallel stacks on the same host, namespaced via `-p` so container
names don't collide. GPUs 0–3 → JAX, GPUs 4–7 → PyTorch. Different `EXP_NAME`
so the checkpoint dirs don't stomp on each other.

```bash
# Build once. Both stacks share the same image.
docker compose -f scripts/docker/train.compose.yml build
```

Terminal 1 — JAX on GPUs 0–3:

```bash
NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_CONFIG=pi05_libero \
EXP_NAME=libero_a100_jax \
TRAIN_ARGS="--fsdp-devices=4 --overwrite" \
    docker compose -p openpi_jax -f scripts/docker/train.compose.yml up
```

Terminal 2 — PyTorch on GPUs 4–7:

```bash
NVIDIA_VISIBLE_DEVICES=4,5,6,7 \
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt --fsdp_devices=4" \
    docker compose -p openpi_pt -f scripts/docker/train.compose.yml up
```

One-shell variant (background both, follow both log streams):

```bash
NVIDIA_VISIBLE_DEVICES=0,1,2,3 TRAIN_CONFIG=pi05_libero \
EXP_NAME=libero_a100_jax TRAIN_ARGS="--fsdp-devices=4 --overwrite" \
    docker compose -p openpi_jax -f scripts/docker/train.compose.yml up -d

NVIDIA_VISIBLE_DEVICES=4,5,6,7 \
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt --fsdp_devices=4" \
    docker compose -p openpi_pt -f scripts/docker/train.compose.yml up -d

docker compose -p openpi_jax logs -f trainer &
docker compose -p openpi_pt  logs -f trainer
```

Notes for side-by-side runs:

- **`-p` is required.** Without a project name, both stacks would try to create
  a container named `trainer` and the second `up` would recreate (kill) the
  first. `openpi_jax` / `openpi_pt` keeps them isolated.
- **GPU isolation.** `NVIDIA_VISIBLE_DEVICES` is the per-container var the
  NVIDIA runtime honors. Inside each container the visible GPUs renumber to
  `0..3`, so `nproc_per_node=4` / `--fsdp-devices=4` / `--fsdp_devices=4` are
  all correct as written. Don't add `CUDA_VISIBLE_DEVICES` on top.
- **Checkpoints don't collide.** Paths are
  `./checkpoints/pi05_libero/libero_a100_jax/...` vs
  `./checkpoints/pi05_libero/libero_a100_pt/...`.
- **Dataloader CPU.** `pi05_libero` defaults to `num_workers=2`, so two stacks
  use 4 workers total — fine on a DGX.
- **NCCL port.** torchrun defaults to master port 29500 inside its container;
  only one stack uses it. If you ever run *two* PyTorch stacks side-by-side,
  add `--rdzv_endpoint=localhost:29501` (or `--master_port=29501`) on the
  second.
- **Stop one without killing the other.**
  `docker compose -p openpi_jax down` (or `-p openpi_pt`).

---

One-shot helpers using the same image (no separate Dockerfile needed):

```bash
# Compute norm stats inside the container (handy if A100 host lacks uv).
TRAIN_CMD="python scripts/compute_norm_stats.py --config-name pi05_libero" \
    docker compose -f scripts/docker/train.compose.yml run --rm trainer

# Convert JAX -> PyTorch base inside the container.
TRAIN_CMD="python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /openpi_assets/openpi-assets/checkpoints/pi05_base \
    --config_name pi05_libero \
    --output_path /app/checkpoints/pi05_base_pytorch" \
    docker compose -f scripts/docker/train.compose.yml run --rm trainer
```

---

## Things to watch on the A100 box

- **GPU runtime**: `runtime: nvidia` requires `nvidia-container-toolkit` on the
  host (not snap-Docker). If `docker info | grep Runtimes` doesn't show
  `nvidia`, run `scripts/docker/install_nvidia_container_toolkit.sh`.
- **Disk**: full `pi05_libero` writes large checkpoints (~30 G each at the JAX
  path; PyTorch ~7–14 G). Make sure `./checkpoints` is on a big disk and
  consider lowering `keep_period` if you're tight.
- **Resume**: rerun the same command without `--overwrite` (JAX) or with
  `--resume` (PyTorch); both trainers restart from the latest checkpoint.

---

## Flag-style cheat sheet

| Concept              | JAX (`scripts/train.py`)        | PyTorch (`scripts/train_pytorch.py`)        |
| -------------------- | ------------------------------- | ------------------------------------------- |
| Experiment name      | `--exp-name=NAME`               | `--exp_name=NAME`                           |
| FSDP devices         | `--fsdp-devices=N` (default 1)  | `--fsdp_devices=N` (default 1; on PyTorch the multi-GPU default auto-promotes to N=world_size — full sharding) |
| Overwrite checkpoint | `--overwrite`                   | (delete `./checkpoints/<config>/<exp>/`)    |
| Resume               | (default if dir exists)         | `--resume`                                  |
| Compile training fwd | (n/a — JAX traces by default)   | `--model.pytorch_compile_train`             |
| Memory frac          | `XLA_PYTHON_CLIENT_MEM_FRACTION`| (n/a; PyTorch uses caching allocator)       |
