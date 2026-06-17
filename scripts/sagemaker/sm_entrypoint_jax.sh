#!/usr/bin/env bash
# SageMaker entrypoint for the openpi JAX trainer (scripts/train.py).
#
# Unlike the PyTorch entrypoint (torchrun, one process per GPU), the JAX trainer
# is a SINGLE process that sees all visible GPUs and shards the model itself via
# --fsdp-devices. So there is no torchrun / rdzv here.
#
# Tailored for the RoboCasa System1 config (pi05_robocasa_system1*): the dataset
# is WebDataset shards (+ anchors), mounted as a FastFile channel; we point the
# config at it via ROBOCASA_SHARDS_DIR. Norm stats are baked into the image at
# /opt/ml/code/assets/<config>/. The JAX pi05_base orbax checkpoint is mounted as
# the base_ckpt channel and loaded by the config's CheckpointWeightLoader (we
# override its params path to the mount via the weight-loader env, see below).
#
# Required env (set by launch.py via the estimator's environment=):
#   TRAIN_CONFIG    e.g. pi05_robocasa_system1_noanchor
#   EXP_NAME        e.g. sagemaker_p5_run1
# Optional:
#   TRAIN_ARGS      extra args appended to scripts/train.py (e.g. "--fsdp-devices=8 --batch-size=64")
#   RESUME / OVERWRITE  "1" to pass --resume / --overwrite
set -euo pipefail

cd /opt/ml/code

CONFIG="${TRAIN_CONFIG:?Set TRAIN_CONFIG (e.g. pi05_robocasa_system1_noanchor)}"
EXP="${EXP_NAME:?Set EXP_NAME}"

# SageMaker channels mount under /opt/ml/input/data/<channel>/.
#   robocasa/  -> the dataset root (shards/ + anchors/ + meta.json)
#   base_ckpt/ -> JAX pi05_base orbax checkpoint (the `params` dir)
ROBOCASA_DIR="/opt/ml/input/data/robocasa"
BASE_CKPT_DIR="/opt/ml/input/data/base_ckpt"
ASSETS_DIR="/opt/ml/code/assets"

if [[ ! -d "$ROBOCASA_DIR/shards" ]]; then
    echo "ERROR: $ROBOCASA_DIR/shards missing. Upload the dataset channel." >&2
    exit 1
fi
# Point the RoboCasa config at the mounted shards (loader derives anchors/ sibling).
export ROBOCASA_SHARDS_DIR="$ROBOCASA_DIR/shards"

if [[ ! -f "$ASSETS_DIR/$CONFIG/robocasa_system1/norm_stats.json" ]]; then
    echo "ERROR: norm_stats.json missing under $ASSETS_DIR/$CONFIG/robocasa_system1/. Rebuild the image." >&2
    exit 1
fi

# JAX base checkpoint: the config's CheckpointWeightLoader points at
# gs://openpi-assets/checkpoints/pi05_base/params by default. On SageMaker (no GCS
# creds; FastFile mount instead) override the params path to the mounted orbax dir.
# The base_ckpt channel root holds {params/, assets/}; the loader wants the params/
# subdir (fall back to the root if params/ isn't present).
if [[ -d "$BASE_CKPT_DIR/params" ]]; then
    export OPENPI_WEIGHT_LOADER_PARAMS_PATH="$BASE_CKPT_DIR/params"
elif [[ -d "$BASE_CKPT_DIR" ]]; then
    export OPENPI_WEIGHT_LOADER_PARAMS_PATH="$BASE_CKPT_DIR"
fi

# SageMaker syncs /opt/ml/checkpoints -> checkpoint_s3_uri and /opt/ml/output on exit.
mkdir -p /opt/ml/checkpoints /opt/ml/output/wandb

EXTRA_ARGS=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA_ARGS+=("--overwrite")
[[ "${RESUME:-0}"    == "1" ]] && EXTRA_ARGS+=("--resume")
# shellcheck disable=SC2206
[[ -n "${TRAIN_ARGS:-}" ]] && EXTRA_ARGS+=( ${TRAIN_ARGS} )

NGPU="$(nvidia-smi -L | wc -l)"
echo "=== SageMaker openpi JAX launch ==="
echo "config=$CONFIG  exp=$EXP  gpus=$NGPU"
echo "ROBOCASA_SHARDS_DIR=$ROBOCASA_SHARDS_DIR"
echo "base_ckpt=${OPENPI_WEIGHT_LOADER_PARAMS_PATH:-<config default>}"
echo "assets=$ASSETS_DIR/$CONFIG  checkpoints=/opt/ml/checkpoints"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "==================================="

exec python scripts/train.py \
    "$CONFIG" \
    --exp-name="$EXP" \
    --assets-base-dir="$ASSETS_DIR" \
    --checkpoint-base-dir=/opt/ml/checkpoints \
    "${EXTRA_ARGS[@]}"
