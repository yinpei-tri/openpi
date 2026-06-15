# SageMaker training for openpi

End-to-end recipe to train `pi05_libero` (PyTorch trainer) on AWS SageMaker
via `TrainingQueue`. Mirrors the layout of `~/sagemaker-example` but adapted
for openpi's data + checkpoint + transformers-patch needs.

The DGX/local Docker setup under `scripts/docker/` is unchanged; this
directory is its SageMaker counterpart.

## Files

| File | What it does |
| --- | --- |
| `config.yaml` | Single source of truth: region, role, queue, instance, image, S3 channels, training args, output. |
| `launch.py` | Reads `config.yaml` (CLI overrides allowed), builds + pushes the image, submits to the queue. |
| `train.Dockerfile` | Region-parametrized DLC base + uv-managed venv + openpi source baked in + transformers patch. |
| `sm_entrypoint.sh` | Runs inside the container. Translates `SM_*` env vars into `torchrun` + openpi flags. |
| `upload_data_to_s3.py` | One-shot uploader for the two input channels (libero / base_ckpt) plus opt-in mirrors of released checkpoints. |
| `Makefile` | Standalone `docker-build-sm` / `docker-push-sm` if you want to (re)build without `launch.py`. |
| `secrets.env.example` | Template for `WANDB_API_KEY` / `HF_TOKEN`. Real `secrets.env` is gitignored. |
| `LOCAL_TESTING.md` | How to build the image and smoke-test training on your laptop without submitting to SageMaker. |
| `FAQ.md` | Running list of design questions answered while wiring this up (data flow, env vars, why `repo_id` doesn't change, etc.). |

## Channels

Two S3 channels mounted FastFile (no download cost):

| Channel | Container path | S3 source |
| --- | --- | --- |
| `libero` | `/opt/ml/input/data/libero/{data,meta}/...` | `s3://tri-ml-datasets-uw2/yinpeidai/data/libero/` |
| `base_ckpt` | `/opt/ml/input/data/base_ckpt/model.safetensors` | `s3://tri-ml-datasets-uw2/yinpeidai/openpi/released_ckpt/openpi-assets/checkpoints/pi05_base_pytorch/` |

Norm-stats assets (~20 KB) are baked into the image at
`/opt/ml/code/assets/pi05_libero/`, so no S3 channel is needed.

How the data is loaded: `train_pytorch.py` calls
`LeRobotDataset(repo_id="physical-intelligence/libero")`, which resolves to
`$HF_LEROBOT_HOME/<repo_id>/{data,meta,...}`. The S3 channel holds the raw
LeRobot tree (`data/`, `meta/`) at the root, so `sm_entrypoint.sh` symlinks
`/opt/ml/input/data/libero` → `/tmp/lerobot_root/physical-intelligence/libero`
and exports `HF_LEROBOT_HOME=/tmp/lerobot_root`. LeRobot's "all parquet
files exist? skip download" check at `lerobot_dataset.py:498` is what makes
this offline-by-default once the files are in place.

The base ckpt is wired the same way: `--pytorch-weight-path` points
`train_pytorch.py` at the read-only FastFile mount. `--assets-base-dir`
points at the baked-in `/opt/ml/code/assets/`.

## One-time setup

```bash
# 0. Install workstation-only deps (boto3 / sagemaker SDK / pyyaml). These
#    are in the `sagemaker` uv group — not in the base env, and not baked
#    into the trainer image.
GIT_LFS_SKIP_SMUDGE=1 uv sync --group sagemaker

# 1. AWS SSO.
aws sso login --profile sagemaker

# 2. Make sure the local artifacts exist (these are what we'll upload):
#    - LeRobot LIBERO dataset (note: NOT the HF Hub cache — LeRobot uses its own)
#    - pi05_base_pytorch (PyTorch port of the JAX base ckpt)
#    - assets/pi05_libero/.../norm_stats.json (baked into the image, not uploaded)
ls ~/.cache/huggingface/lerobot/physical-intelligence/libero/meta/info.json
ls ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch/model.safetensors
ls ./assets/pi05_libero/physical-intelligence/libero/norm_stats.json

# If the LeRobot dataset is missing, fetch it once on the host:
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('physical-intelligence/libero', repo_type='dataset', \
        local_dir='~/.cache/huggingface/lerobot/physical-intelligence/libero')"

# If pi05_base_pytorch is missing:
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --config_name pi05_libero \
    --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch

# If norm stats are missing:
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 3. Upload to S3 (idempotent; uses `aws s3 sync`).
#    Defaults to the two training channels (libero / base_ckpt).
uv run --group sagemaker scripts/sagemaker/upload_data_to_s3.py

# Optional opt-in mirrors (NOT mounted into the training job):
#   base_jax        — JAX pi05_base orbax ckpt (~12 GB), useful for in-cloud
#                     JAX→PyTorch conversion or JAX SageMaker jobs.
#   libero_pytorch  — released pi05_libero PyTorch ckpt (eval / resume baseline).
uv run --group sagemaker scripts/sagemaker/upload_data_to_s3.py --channels base_jax libero_pytorch

# 4. Secrets (never commit).
cp scripts/sagemaker/secrets.env.example scripts/sagemaker/secrets.env
$EDITOR scripts/sagemaker/secrets.env
```

When `launch.py` submits the job it prints a "Resolved S3 URIs" block listing
every input channel URI, the output checkpoint URI, and a ready-to-paste
`aws s3 sync` command for pulling the trained checkpoints back to your
laptop. Save that output if you want to skip reconstructing paths from the
job name later.

## Submit a job

```bash
uv run --group sagemaker scripts/sagemaker/launch.py
```

This will:
1. Load `config.yaml`.
2. Build `train.Dockerfile` (DLC base for `us-west-2`) and push to
   `<account>.dkr.ecr.us-west-2.amazonaws.com/yinpeidai-openpi-train:latest`.
3. Compose the queue name `fss-vla-p5-48xlarge-us-west-2`.
4. Submit a `torch_distributed` Estimator with the two channels.

Useful overrides:

```bash
# Skip image rebuild (use the :latest already in ECR).
uv run --group sagemaker scripts/sagemaker/launch.py --skip-build

# Plan-only — print what would be submitted.
uv run --group sagemaker scripts/sagemaker/launch.py --dry-run

# Different exp name / extra trainer args.
uv run --group sagemaker scripts/sagemaker/launch.py \
    --training.exp_name=run3 \
    --training.extra_args="--num_train_steps=10000 --save_interval=2000"

# Resume from latest checkpoint in the same exp dir.
uv run --group sagemaker scripts/sagemaker/launch.py --training.resume=true

# Switch instance shape.
uv run --group sagemaker scripts/sagemaker/launch.py --instance.type=p4de
```

## Local mode

```bash
uv run --group sagemaker scripts/sagemaker/launch.py --local --instance.count=1
```

`local_gpu` runs the same image on the host via SageMaker's `LocalSession`.
Useful for smoke-testing the entrypoint without paying for a real instance.

## Notes

- **No source upload.** We bake `src/`, `scripts/`, `packages/` into the image
  and submit via the lower-level `Estimator` (not `PyTorch`), so SageMaker
  does *not* upload a `source_dir` to S3 on each submit. To pick up code
  changes, rebuild the image (`make -C scripts/sagemaker docker-push-sm` or
  just rerun `launch.py` without `--skip-build`).
- **Checkpoint sync.** SageMaker mirrors `/opt/ml/checkpoints/` to
  `checkpoint_s3_uri` continuously; spot interruptions don't lose progress.
  Final layout in S3:
  `s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints/{job_name}/{config}/{exp_name}/{step}/`.
  The job-name layer keeps concurrent runs from colliding even if they share
  an `exp_name`.
- **`pi05_libero` defaults.** `batch_size=256`, `num_train_steps=30_000`,
  EMA on. On p5.48xlarge (8x H100) DDP, expect ~0.5–1 day end-to-end.
- **Volume size.** `job.volume_size=200` is sized for `/opt/ml/checkpoints`
  plus uv/pip/HF caches — bump it if you change checkpoint cadence.
- **Multi-node.** `sm_entrypoint.sh` honors `SM_HOST_COUNT > 1` and switches
  torchrun to rdzv. Untested in this image; start single-node.
