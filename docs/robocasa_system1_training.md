# System1 (RoboCasa pi0.5) — training setup

Subgoal-conditioned **pi0.5** + a **progress head**, trained on RoboCasa WebDataset
shards. This doc covers the data, the configs, and how to run training in each
environment: bare-metal, local Docker, an 8×A100 box (S3 data), and SageMaker.

## TL;DR — run on the 8×A100 box, streaming data from S3

```bash
cd <openpi-repo>
# 1. AWS creds in env (so the WebDataset loader's boto3 can read s3://)
eval "$(aws --profile sagemaker configure export-credentials --format env)"
export AWS_DEFAULT_REGION=us-west-2
# 2. point the config at the S3 shards
export ROBOCASA_SHARDS_DIR=s3://tri-ml-datasets-uw2/yinpeidai/data/robocasa_system1/shards
# 3. norm stats must exist locally for this config (see "Norm stats" below):
#    assets/pi05_robocasa_system1_noanchor/robocasa_system1/norm_stats.json
# 4. train — 8 GPUs, FSDP across all 8, batch divisible by 8
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/train.py pi05_robocasa_system1_noanchor \
    --exp-name=a100_run1 \
    --batch-size=64 --fsdp-devices=8 --ema-decay=None \
    --num-train-steps=50000 --save-interval=5000 --overwrite
```

A100 80GB has far more memory than the dev A6000s, so you have headroom: larger
`--batch-size`, and you can re-enable EMA / anchors (see "Memory & knobs").

## The data

- **Format**: per-frame, globally-shuffled WebDataset tar shards + a deduped anchor
  store. Built by RoboAnnotator `producers/preprocess_robocasa_to_tar.py`. Full field
  schema is in the dataset's `SCHEMA.md`.
- **Local**: `/home/yinpeidai/RoboAnnotator/data/robocasa_system1/` (16GB, 594 shards,
  303,755 frames, 350 episodes).
- **S3**: `s3://tri-ml-datasets-uw2/yinpeidai/data/robocasa_system1/`
  (`shards/`, `anchors/`, `meta.json`, `SCHEMA.md`).
- The loader (`openpi/training/robocasa_webdataset.py`) reads `ROBOCASA_SHARDS_DIR`
  (local dir OR `s3://`); it derives the `anchors/` sibling automatically. For S3 it
  uses boto3 (a core dep) and reads creds from the standard AWS env vars.

## The configs (`src/openpi/training/config.py`)

| config | use | images | progress head | EMA |
| --- | --- | --- | --- | --- |
| `pi05_robocasa_system1` | full design | 3 current + 3 anchor (6) | yes (shallow_transformer) | 0.999 |
| `pi05_robocasa_system1_noanchor` | **fast baseline** (anchor=none) | 3 current | yes | None |
| `pi05_robocasa_system1_debug` | tiny smoke | 3 current | yes | None |

All: pi05, `action_horizon=20`, 15-dim lean state (rel base pos + rel-yaw sin/cos +
eef pos + eef 6D + gripper width), action padded to 32, quantile norm, fixed LR 5e-5,
weight-loader = released `pi05_base` (progress head + anchor role-emb kept fresh).

`ROBOCASA_SHARDS_DIR` env overrides the shards path for ALL three (default = the local
host path), so the same config runs bare-metal / Docker / A100 / SageMaker.

## Key training args

- `--fsdp-devices=N` — shard the ~3B-param model across N GPUs. **Must divide the
  visible GPU count**; with N == #GPUs there's 1 data-parallel group (pure FSDP).
- `--batch-size=B` — **must be divisible by the visible GPU count**. On A100/H100 with
  NVLink, a bigger batch ≈ proportionally more throughput — use a large batch.
- `--ema-decay=None` — disables EMA (frees a full param-copy of memory). Needed to fit
  the dev A6000s; on A100 80GB you can leave EMA on (drop this flag).
- `--num-train-steps`, `--save-interval`, `--overwrite`, `--resume`.
- `--exp-name=` — names the checkpoint dir: `checkpoints/<config>/<exp-name>/<step>/`.

### Epoch math (303,755 frames)
`steps/epoch = 303755 / batch_size`. batch 64 → ~4,747/epoch → 4 epochs ≈ 19k steps;
batch 128 → ~2,373/epoch → 4 epochs ≈ 9.5k steps.

## Memory & knobs (dev A6000 48GB vs A100 80GB)

- Dev box (3×A6000, no NVLink) needed `--fsdp-devices=2 --ema-decay=None` + a 3-image
  config to fit, and fsdp=2 was much faster than fsdp=3 (topology: fsdp=3 dragged the
  all-gather over a slow link). Profiling: step time was compute-bound (data_wait
  0.03s), batch- and image-count-independent — pure FSDP comms + remat double-forward.
- **8×A100 80GB**: ~1.7× the memory/GPU + NVLink. Run `--fsdp-devices=8`, large
  `--batch-size` (e.g. 128/256, divisible by 8), keep EMA, and you can use the full
  6-image `pi05_robocasa_system1` (anchor) config. Expect far better throughput.
- Gradient checkpointing (`nn.remat(nothing_saveable)`) is always on in gemma — roughly
  doubles the forward (recompute in backward). Intrinsic; fine.

## Norm stats

Recipe-dependent (15-dim lean state + episode-padded action quantiles), stored at
`assets/<config>/robocasa_system1/norm_stats.json`. Already present for the three
configs. To recompute (e.g. if you change the state recipe):
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compute_norm_stats.py \
    --config-name pi05_robocasa_system1_noanchor --max-frames 60000
```
(Works with `ROBOCASA_SHARDS_DIR` pointing at local OR s3.)

## Local Docker (matches the SageMaker container)

```bash
eval "$(aws --profile sagemaker configure export-credentials --format env)"
HOST_UID=$(id -u) HOST_GID=$(id -g) \
NVIDIA_VISIBLE_DEVICES=0,1 \
ROBOCASA_SHARDS_DIR=s3://tri-ml-datasets-uw2/yinpeidai/data/robocasa_system1/shards \
AWS_DEFAULT_REGION=us-west-2 \
TRAIN_CONFIG=pi05_robocasa_system1_noanchor EXP_NAME=docker_run \
TRAIN_ARGS="--batch-size=2 --fsdp-devices=2 --ema-decay=None --num-train-steps=200 --overwrite" \
WANDB_MODE=offline \
  docker compose -f scripts/docker/train.compose.yml up
```
The compose bind-mounts local data at `/data/robocasa_system1` (used when
`ROBOCASA_SHARDS_DIR` is left at its local default) and forwards `AWS_*` for s3://.
Code is bind-mounted at `/app` — source edits need no rebuild; only dep/Dockerfile
changes need `docker compose ... build`.

## SageMaker (8×H100 p5)

```bash
python scripts/sagemaker/launch.py --config scripts/sagemaker/config_robocasa_jax.yaml \
    --training.exp_name=run1 \
    [--training.extra_args="--fsdp-devices=8 --batch-size=64 --ema-decay=None --num-train-steps=50000 --save-interval=5000"] \
    [--dry-run]
```
Builds + pushes the JAX image (`image.sm_entrypoint=sm_entrypoint_jax.sh`), mounts two
FastFile channels (`robocasa` dataset, `base_ckpt` JAX orbax), runs the single-process
JAX trainer. Checkpoints sync continuously to
`s3://.../openpi/checkpoints/<job_name>/<config>/<exp-name>/<step>/`. Download with the
`aws s3 sync` command launch.py prints.

## Checkpoints

`checkpoints/<config>/<exp-name>/<step>/` (local); on SageMaker `/opt/ml/checkpoints`
syncs to the S3 path above. Each holds orbax `params/` + `train_state/` + `assets/`
(norm stats). Resume with `--resume`.

## Watch out

- **boto3** is required at train time for s3:// shards (a core dep now; rebuild any old
  image that predates this).
- **Docker GPU on snap-docker**: after a host driver upgrade, regenerate the CDI spec
  (`sudo nvidia-ctk cdi generate --output=/var/snap/docker/<rev>/etc/cdi/nvidia.yaml;
  sudo snap restart docker`).
- **`/tmp` is volatile** on the dev box — write logs under the repo (`.local_output/`).
- S3 first-batch latency: the reservoir buffer (`shuffle_buffer`/`shuffle_initial`)
  fills from S3 before step 0 (~1-2 min). Lower `shuffle_initial` for quick smokes.
