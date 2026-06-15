# Local build & test

How to build the SageMaker image and validate it on your own box before
pushing to ECR / submitting to the queue. Useful for iterating on
`train.Dockerfile`, `sm_entrypoint.sh`, and the trainer args without
burning queue time.

There are two flavors:

- **(a) Plain `docker run`** — fastest. No SageMaker runtime, no S3. You
  bind-mount your local LeRobot cache + base ckpt directly onto the
  container's expected channel paths, then exercise `sm_entrypoint.sh`
  end-to-end. Good for "does the image work?" loops.
- **(b) `launch.py --local`** — uses SageMaker's `LocalSession` to run the
  same image with the SageMaker runtime emulated locally (channels really
  pulled via FastFile-style FUSE, etc.). Slower but exercises the
  SageMaker contract more faithfully.

Use (a) for almost everything. Reach for (b) only when you suspect a
SageMaker-specific quirk.

---

## (a) Plain `docker run`

### Build

```bash
make -C scripts/sagemaker docker-build-sm
```

This calls the same `docker build` that `launch.py` runs but skips the
ECR push. The image is tagged `yinpeidai-openpi-train:latest` locally.

### Smoke check (no training, just shell)

```bash
make -C scripts/sagemaker docker-shell
# inside the container:
python -c "import openpi, transformers; print(transformers.__file__)"
ls /opt/ml/code/assets/pi05_libero/physical-intelligence/libero/norm_stats.json
ls /opt/ml/code/scripts/train_pytorch.py
```

Confirms:
- venv is on `PATH`, openpi is importable.
- transformers patch landed (the `__file__` should be under `/.venv/`).
- The norm-stats asset is baked in.

### End-to-end smoke training run

Use the `pi05_libero_debug` config — 200 steps, batch_size=4, finishes in
a few minutes on one GPU. Bind-mount your local artifacts onto the
container's expected channel paths:

```bash
docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 --shm-size=16g \
  -e TRAIN_CONFIG=pi05_libero_debug \
  -e EXP_NAME=local_smoke \
  -e WANDB_MODE=disabled \
  -e SM_NUM_GPUS=2 \
  -v ~/.cache/huggingface/lerobot/physical-intelligence/libero:/opt/ml/input/data/libero:ro \
  -v ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch:/opt/ml/input/data/base_ckpt:ro \
  -v "$PWD/.local_ckpt:/opt/ml/checkpoints" \
  -v "$PWD/.local_output:/opt/ml/output" \
  yinpeidai-openpi-train
```

What this verifies:
- `sm_entrypoint.sh` finds `meta/info.json`, builds the
  `/tmp/lerobot_root/physical-intelligence/libero` symlink, and exports
  `HF_LEROBOT_HOME` correctly.
- `--pytorch-weight-path` resolves the bind-mounted base ckpt.
- `--assets-base-dir` resolves to the baked-in norm stats.
- `torchrun --standalone` comes up, FSDP/DDP initializes, the dataloader
  starts producing batches, and a few training steps land.

The bind-mounts (`-v`) are doing manually what SageMaker does for you in
production via FastFile FUSE. Without them, `/opt/ml/input/data/libero`
would be empty inside the container.

### Tweaks while iterating

- **Pass extra trainer args** via `-e TRAIN_ARGS="..."`, same syntax as
  `config.yaml: training.extra_args`. Example:
  `-e TRAIN_ARGS="--num_train_steps=10 --save_interval=5"`.
- **Multi-GPU smoke**: bump `-e SM_NUM_GPUS=2`. Single-node only — multi-
  node rdzv requires real DNS so it's only meaningful on SageMaker.
- **Wandb on**: drop `WANDB_MODE=disabled`, add `-e WANDB_API_KEY=...` and
  `-e WANDB_PROJECT=openpi-local-smoke`.

---

## (b) `launch.py --local`

```bash
uv run --group sagemaker scripts/sagemaker/launch.py --local --skip-build \
    --training.config=pi05_libero_debug --training.exp_name=local_smoke
```

The `--group sagemaker` brings in `boto3` / `sagemaker` / `pyyaml`, which
live in a workstation-only uv group (not baked into the trainer image).
Run `uv sync --group sagemaker` once after pulling new deps.

What's different from (a):
- Uses `sagemaker.local.LocalSession`, which actually does an `aws s3 sync`
  for each channel into a temp dir on your box, then mounts that into the
  container. (FastFile FUSE isn't supported in local mode.)
- Honors `keep_alive_period_in_seconds`, `tags`, etc., the same way the
  remote runtime does.
- Slower start (S3 download), but catches SageMaker-runtime issues.

`--skip-build` reuses the local image from (a). Drop it if you want
`launch.py` to (re)build before running.

---

## Common gotchas

- **`/opt/ml/input/data/libero/meta/info.json missing`** — your bind-mount
  points at the wrong directory. The container expects the LeRobot tree
  *root* (containing `data/` and `meta/`), not `~/.cache/huggingface/`.
- **`ImportError: ... transformers ...`** after editing the patch — rerun
  `docker-build-sm`. The patch is copied into the venv at image build
  time; in-container edits don't persist.
- **`No module named openpi`** — the image build failed mid-way and you
  ran an older layer. Force a rebuild: `docker-build-sm` (Docker reuses
  cache aggressively).
- **OOM with `pi05_libero` (batch_size=256)** on a single consumer GPU —
  expected. Use `pi05_libero_debug` for local runs.
