#!/usr/bin/env bash
# SageMaker entrypoint for the openpi PyTorch trainer.
#
# Translates the SageMaker contract (channels under /opt/ml/input/data/, env
# vars like SM_CHANNEL_*, SM_NUM_GPUS, master host info) into the flags the
# trainer expects, then launches via torchrun.
#
# Required env (set by launch.py via the estimator's `environment=`):
#   TRAIN_CONFIG    e.g. pi05_libero
#   EXP_NAME        e.g. sagemaker_p5_run1
#
# Optional:
#   TRAIN_ARGS      extra args appended to scripts/train_pytorch.py
#   RESUME          "1" to pass --resume
#   OVERWRITE       "1" to pass --overwrite
set -euo pipefail

cd /opt/ml/code

CONFIG="${TRAIN_CONFIG:?Set TRAIN_CONFIG (e.g. pi05_libero)}"
EXP="${EXP_NAME:?Set EXP_NAME}"

# SageMaker channels mount under /opt/ml/input/data/<channel>/.
# We expect two channels named libero / base_ckpt (see launch.py).
# Norm-stats assets are baked into the image at /opt/ml/code/assets/pi05_libero/.
LIBERO_DIR="/opt/ml/input/data/libero"
BASE_CKPT_DIR="/opt/ml/input/data/base_ckpt"
ASSETS_DIR="/opt/ml/code/assets"

# LeRobot dataset: train_pytorch.py loads via LeRobotDataset(repo_id="physical-intelligence/libero"),
# which reads parquet files from $HF_LEROBOT_HOME/<repo_id>/{data,meta,...}.
# Our S3 channel holds the raw LeRobot tree at the root (data/, meta/), so we
# stitch a writable view under /tmp that puts those dirs at the
# physical-intelligence/libero/ subpath HF_LEROBOT_HOME requires.
if [[ ! -f "$LIBERO_DIR/meta/info.json" ]]; then
    echo "ERROR: $LIBERO_DIR/meta/info.json missing." >&2
    echo "       Re-run: python scripts/sagemaker/upload_data_to_s3.py --channels libero" >&2
    exit 1
fi
LEROBOT_ROOT="/tmp/lerobot_root"
mkdir -p "$LEROBOT_ROOT/physical-intelligence"
ln -sfn "$LIBERO_DIR" "$LEROBOT_ROOT/physical-intelligence/libero"
export HF_LEROBOT_HOME="$LEROBOT_ROOT"

# Norm stats: baked into the image. TrainConfig.assets_dirs resolves to
# {assets_base_dir}/{config.name}/, e.g. /opt/ml/code/assets/pi05_libero/.
if [[ ! -f "$ASSETS_DIR/$CONFIG/physical-intelligence/libero/norm_stats.json" ]]; then
    echo "ERROR: norm_stats.json missing under $ASSETS_DIR/$CONFIG/. Was the image rebuilt?" >&2
    exit 1
fi

# pi05_base_pytorch lives under base_ckpt/. The config's pytorch_weight_path
# defaults to ./checkpoints/pi05_base_pytorch on disk, so we point it at the
# mount via CLI override.
if [[ ! -f "$BASE_CKPT_DIR/model.safetensors" ]]; then
    echo "ERROR: Expected $BASE_CKPT_DIR/model.safetensors (pi05 base ckpt) but it's missing." >&2
    exit 1
fi

# Output dirs that the trainer writes to. SageMaker syncs /opt/ml/checkpoints
# to checkpoint_s3_uri and /opt/ml/output to output_path on job exit.
mkdir -p /opt/ml/checkpoints /opt/ml/output/wandb /opt/ml/output/triton

EXTRA_ARGS=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA_ARGS+=("--overwrite")
[[ "${RESUME:-0}"    == "1" ]] && EXTRA_ARGS+=("--resume")
# shellcheck disable=SC2206
[[ -n "${TRAIN_ARGS:-}" ]] && EXTRA_ARGS+=( ${TRAIN_ARGS} )

# torch_distributed sets MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK.
# For single-node we use --standalone; for multi-node we honor MASTER_ADDR.
NPROC="${SM_NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
NNODES="${SM_HOST_COUNT:-1}"
NODE_RANK="${SM_CURRENT_HOST_RANK:-0}"

echo "=== SageMaker openpi launch ==="
echo "config=$CONFIG  exp=$EXP"
echo "nodes=$NNODES  node_rank=$NODE_RANK  nproc=$NPROC"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "checkpoint_base_dir=/opt/ml/checkpoints"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "==============================="

if [[ "$NNODES" -gt 1 ]]; then
    RDZV=( --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}" --rdzv_id="${TRAINING_JOB_NAME:-openpi}" )
else
    RDZV=( --standalone )
fi

exec torchrun \
    "${RDZV[@]}" \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" \
    scripts/train_pytorch.py \
    "$CONFIG" \
    --exp_name="$EXP" \
    --pytorch-weight-path="$BASE_CKPT_DIR" \
    --assets-base-dir="$ASSETS_DIR" \
    --checkpoint-base-dir=/opt/ml/checkpoints \
    "${EXTRA_ARGS[@]}"
