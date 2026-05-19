# pi05 + LIBERO finetuning runbook

End-to-end commands for fine-tuning `pi05_libero` in three settings: local
host (uv), local Docker, and a single-node 8x A100 box via Docker only.
Working directory is `~/openpi` everywhere.

## What's in this directory

- `train.Dockerfile` — CUDA 12.2 + uv. Installs the locked deps, bakes the
  openpi package into the venv, and applies the
  `models_pytorch/transformers_replace/` patch so the same image runs both
  trainers. Built to run under any UID, so the container can adopt the host
  UID for clean file ownership.
- `train.compose.yml` — One service (`trainer`) that bind-mounts the repo and
  host caches, exposes GPUs via `runtime: nvidia` (works under snap-Docker),
  and launches `scripts/train.py` by default.
- `.env.example` — Copy to `.env` and fill in absolute paths for your caches.

The serve-policy image (`serve_policy.Dockerfile`, `compose.yml`) is unchanged
and remains the canonical inference path.

## Why the env file uses absolute paths

If you installed Docker via snap (the default on Ubuntu 22.04 if you ran
`apt install docker.io` from snap), the daemon resolves `~` and `$HOME` to
its private cache (`~/snap/docker/.../...`). That makes shell-style
`~/.cache/openpi` mounts silently point at empty directories. So the compose
file expects `OPENPI_DATA_HOME` and `HF_HOME` to be absolute paths supplied
via `.env`. Absolute paths work everywhere; `~` only works on official
apt/rootless Docker.

## PyTorch trainer at a glance

The PyTorch trainer uses **DDP** for multi-GPU runs (full replica per GPU —
pi05 is ~4 B params and fits on a single 48 GB / 80 GB GPU). Mirrors the JAX
trainer for **EMA** (driven by `config.ema_decay`; shadow saved as
`ema.safetensors`). Still **not** supported: π₀-FAST, LoRA, FSDP, mixed
precision (use full bf16 or full fp32 via `pytorch_training_precision`).

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
#    Pin to one GPU so JAX doesn't try to shard batch_size=256.
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name pi05_libero
```

JAX smoke run (2 GPUs, batch_size=4 from the config):

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi05_libero_debug --exp-name=jax_local --overwrite
```

PyTorch smoke run (2 GPUs, DDP):

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
# Edit:
#   OPENPI_DATA_HOME=/home/yinpeidai/.cache/openpi
#   HF_HOME=/home/yinpeidai/.cache/huggingface
#   HOST_UID=$(id -u); HOST_GID=$(id -g)
#   NVIDIA_VISIBLE_DEVICES=0,1   # MUST be 2 GPUs — pi05_libero_debug has
#                                # batch_size=4 and JAX shards across every
#                                # visible device. With 3 GPUs visible JAX
#                                # errors: "Batch size 4 must be divisible
#                                # by the number of devices 3."
#   TRAIN_CONFIG=pi05_libero_debug
#   EXP_NAME=jax_docker

# 2. JAX run (default CMD).
docker compose -f scripts/docker/train.compose.yml up --build

# 3. PyTorch run — override TRAIN_CMD inline. Reuses the cached image.
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi05_libero_debug --exp_name=pt_docker" \
EXP_NAME=pt_docker \
    docker compose -f scripts/docker/train.compose.yml up
```

Verify: `ls -l ./checkpoints/pi05_libero_debug/{jax_docker,pt_docker}/` —
files should be owned by your user, not root. If they're root-owned,
`HOST_UID/HOST_GID` in `.env` is wrong.

---

## Phase 3 — Single-node 8x A100 (Docker only)

The A100 box doesn't need uv, Python, or any host setup beyond Docker +
nvidia-container-toolkit. Everything (downloads, conversion, norm stats,
training) runs inside the same image.

### 3a. Prep the host

```bash
git clone --recurse-submodules <your-repo-url> ~/openpi
cd ~/openpi

# Pre-create bind-mount targets as your user. Otherwise Docker creates them
# as root on first use and the container (running as your UID) can't write.
mkdir -p ~/.cache/openpi ~/.cache/huggingface ./checkpoints ./assets ./wandb

cp scripts/docker/.env.example scripts/docker/.env
# Edit:
#   OPENPI_DATA_HOME=/abs/path/to/your/.cache/openpi
#   HF_HOME=/abs/path/to/your/.cache/huggingface
#   HOST_UID=$(id -u); HOST_GID=$(id -g)
#   NVIDIA_VISIBLE_DEVICES=all
#   WANDB_MODE=online        # if you want online logging
#   WANDB_API_KEY=...        # set in your shell, or here

# Build the image once.
docker compose -f scripts/docker/train.compose.yml build
```

### 3b. Bootstrap inputs inside the container

`docker compose run --rm trainer <cmd>` overrides the default CMD, so anything
after `trainer` runs as a one-shot — same image, same bind-mounts, no
leftover container.

```bash
COMPOSE="docker compose -f scripts/docker/train.compose.yml"

# 1. Pre-fetch pi05_base into /openpi_assets (host: ~/.cache/openpi).
$COMPOSE run --rm trainer \
    python -c 'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))'

# 2. Pre-fetch LIBERO into /cache/huggingface (host: ~/.cache/huggingface).
$COMPOSE run --rm trainer \
    python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("physical-intelligence/libero", repo_type="dataset"))'

# 3. Convert JAX base -> PyTorch base. Output lands at
#    ./checkpoints/pi05_base_pytorch on the host (the repo is bind-mounted at
#    /app), which is exactly where pi05_libero_debug points pytorch_weight_path.
$COMPOSE run --rm trainer \
    python examples/convert_jax_model_to_pytorch.py \
        --checkpoint_dir /openpi_assets/openpi-assets/checkpoints/pi05_base \
        --config_name pi05_libero \
        --output_path /app/checkpoints/pi05_base_pytorch

# 4. Compute norm stats once. Pin to one GPU so JAX doesn't shard
#    batch_size=256 across all 8 A100s. Drop --max-frames for production.
$COMPOSE run --rm -e CUDA_VISIBLE_DEVICES=0 trainer \
    python scripts/compute_norm_stats.py --config-name pi05_libero
```

Verify on the host (no container needed):

```bash
ls ~/.cache/openpi/openpi-assets/checkpoints/pi05_base/
ls ~/.cache/huggingface/hub/datasets--physical-intelligence--libero/
ls ./checkpoints/pi05_base_pytorch/model.safetensors
ls ./assets/pi05_libero/physical-intelligence/libero/norm_stats.json
```

### 3c. Train

JAX run, FSDP across all 8 GPUs:

```bash
TRAIN_CONFIG=pi05_libero \
EXP_NAME=libero_a100_jax \
TRAIN_ARGS="--fsdp-devices=8 --overwrite" \
    docker compose -f scripts/docker/train.compose.yml up
```

PyTorch run, DDP across all 8 GPUs (one full replica per GPU):

```bash
TRAIN_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt" \
    docker compose -f scripts/docker/train.compose.yml up
```

`pi05_libero` defaults to `batch_size=256`, `num_train_steps=30_000`, EMA on.
Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` in `.env` for full memory on JAX.

### 3d. Side-by-side JAX vs PyTorch on one A100 node (4 GPUs each)

Run two parallel stacks on the same host, namespaced via `-p` so container
names don't collide. GPUs 0–3 → JAX, GPUs 4–7 → PyTorch.

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
    scripts/train_pytorch.py pi05_libero --exp_name=libero_a100_pt" \
    docker compose -p openpi_pt -f scripts/docker/train.compose.yml up
```

Notes:

- **`-p` is required.** Without a project name, both stacks would try to
  create a container named `trainer` and the second `up` would recreate
  (kill) the first. `openpi_jax` / `openpi_pt` keeps them isolated.
- **GPU isolation.** `NVIDIA_VISIBLE_DEVICES` is the per-container var the
  NVIDIA runtime honors. Inside each container the visible GPUs renumber to
  `0..3`, so `nproc_per_node=4` / `--fsdp-devices=4` are correct as written.
  Don't add `CUDA_VISIBLE_DEVICES` on top.
- **Checkpoints don't collide.** Different `EXP_NAME` →
  `./checkpoints/pi05_libero/libero_a100_{jax,pt}/`.
- **Stop one without the other.** `docker compose -p openpi_jax down` (or
  `-p openpi_pt`).

---

## Things to watch on the A100 box

- **GPU runtime.** `runtime: nvidia` requires `nvidia-container-toolkit` on
  the host. If `docker info | grep Runtimes` doesn't show `nvidia`, run
  `scripts/docker/install_nvidia_container_toolkit.sh`.
- **Disk.** `pi05_libero` writes ~30 G JAX checkpoints / ~7–14 G PyTorch
  checkpoints. Make sure `./checkpoints` is on a big disk; consider lowering
  `keep_period` if you're tight.
- **Resume.** Rerun the same command without `--overwrite` (JAX) or with
  `--resume` (PyTorch); both restart from the latest checkpoint.

## SageMaker note

When you move to SageMaker, this image is a starting point but SageMaker
expects:

- The training entrypoint to read `SM_CHANNEL_*` env vars instead of bind
  mounts (or you stage data from S3 to `/opt/ml/input/data/...`).
- `OPENPI_DATA_HOME` to point at a writable location under `/opt/ml/`.
- The container to write checkpoints to `/opt/ml/checkpoints/` for them to
  be uploaded back to S3.

Adapt `train.Dockerfile`'s default CMD or wrap it in a SageMaker-style
`train` shell script when you get there.

## Flag-style cheat sheet

| Concept              | JAX (`scripts/train.py`)        | PyTorch (`scripts/train_pytorch.py`)        |
| -------------------- | ------------------------------- | ------------------------------------------- |
| Experiment name      | `--exp-name=NAME`               | `--exp_name=NAME`                           |
| FSDP devices         | `--fsdp-devices=N` (default 1)  | (n/a — PyTorch trainer uses DDP only)       |
| Overwrite checkpoint | `--overwrite`                   | `--overwrite`                               |
| Resume               | (default if dir exists)         | `--resume`                                  |
| Memory frac          | `XLA_PYTHON_CLIENT_MEM_FRACTION`| (n/a; PyTorch uses caching allocator)       |
