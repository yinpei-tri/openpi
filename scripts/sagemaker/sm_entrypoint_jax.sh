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

# Data source: two modes.
#   (download) If ROBOCASA_DATA_S3_URI is set, `aws s3 sync` the whole dataset to a local
#     EBS dir ONCE up front and read from local disk for the rest of the run. This fully
#     avoids the FastFile FUSE mount (no ENOTCONN mid-run mount-drop, fastest reads) — the
#     right choice for a long on-demand run. volume_size must fit the dataset + checkpoints.
#   (mount) Otherwise, read from the FastFile-mounted channel at /opt/ml/input/data/robocasa.
if [[ -n "${ROBOCASA_DATA_S3_URI:-}" ]]; then
    LOCAL_DATA="/opt/ml/local_data"
    mkdir -p "$LOCAL_DATA"
    # High-concurrency sync (bandwidth-bound on ~12k big shards, not latency-bound).
    aws configure set default.s3.max_concurrent_requests 100
    aws configure set default.s3.max_queue_size 10000
    echo "=== Downloading dataset to local EBS: ${ROBOCASA_DATA_S3_URI} -> $LOCAL_DATA ==="
    _dl_start=$(date +%s)
    aws s3 sync "${ROBOCASA_DATA_S3_URI}" "$LOCAL_DATA" --only-show-errors
    echo "=== Download finished in $(( ($(date +%s) - _dl_start) / 60 )) min ==="
    df -h "$LOCAL_DATA" | tail -1
    if [[ ! -d "$LOCAL_DATA/shards" ]]; then
        echo "ERROR: $LOCAL_DATA/shards missing after sync from ${ROBOCASA_DATA_S3_URI}." >&2
        exit 1
    fi
    n_tar=$(find "$LOCAL_DATA/shards" -name '*.tar' | wc -l)
    echo "Local shards: $n_tar"
    export ROBOCASA_SHARDS_DIR="$LOCAL_DATA/shards"
else
    if [[ ! -d "$ROBOCASA_DIR/shards" ]]; then
        echo "ERROR: $ROBOCASA_DIR/shards missing. Upload the dataset channel (or set ROBOCASA_DATA_S3_URI to download)." >&2
        exit 1
    fi
    # Point the RoboCasa config at the FastFile-mounted shards (loader derives anchors/ sibling).
    export ROBOCASA_SHARDS_DIR="$ROBOCASA_DIR/shards"
fi

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

# Checkpointing: do NOT use SageMaker's managed /opt/ml/checkpoints sync for the JAX
# trainer. That mount is eventually-consistent (a background sync sidecar), and orbax
# writes a tiny array-metadata file then immediately reads it back in `finalize` to
# validate it — the read-back can return an empty file there, crashing the save with
# `JSONDecodeError: Expecting value`. Instead checkpoint to a plain instance-EBS dir
# (strong read-after-write) and run our OWN `aws s3 sync` to CHECKPOINT_S3_URI:
# a periodic background upload (crash protection) + an authoritative final sync on exit.
# launch.py leaves SageMaker-managed checkpoint sync off and passes CHECKPOINT_S3_URI.
CKPT_DIR="/opt/ml/local_checkpoints"
mkdir -p "$CKPT_DIR" /opt/ml/output/wandb

# Periodic background upload (default every 30 min; override via CKPT_SYNC_INTERVAL).
# Crash insurance: if the instance dies mid-run, S3 still has checkpoints up to the
# last tick (local EBS dies with the instance). We EXCLUDE orbax's in-progress
# `*.orbax-checkpoint-tmp-*` dirs so a partially-written checkpoint never lands in S3;
# orbax renames the tmp dir to its final name only once the save is complete. No
# `--delete` here (a periodic delete could race orbax's pruning) — the final sync
# reconciles. Long interval keeps S3 traffic/overhead low.
SYNC_PID=""
if [[ -n "${CHECKPOINT_S3_URI:-}" ]]; then
    echo "Periodic checkpoint upload: $CKPT_DIR -> ${CHECKPOINT_S3_URI} (every ${CKPT_SYNC_INTERVAL:-1800}s)"
    (
        while true; do
            sleep "${CKPT_SYNC_INTERVAL:-1800}"
            aws s3 sync "$CKPT_DIR" "${CHECKPOINT_S3_URI}" \
                --exclude "*.orbax-checkpoint-tmp-*/*" --only-show-errors || true
        done
    ) &
    SYNC_PID=$!
fi

# Final sync on exit (success OR failure): stop the periodic loop, then one
# authoritative `--delete` sync so S3 mirrors orbax's settled local dir exactly
# (drops any tmp dirs / pruned steps the periodic uploads may have left behind).
final_sync() {
    [[ -n "$SYNC_PID" ]] && kill "$SYNC_PID" 2>/dev/null || true
    if [[ -n "${CHECKPOINT_S3_URI:-}" ]]; then
        echo "Final checkpoint upload: $CKPT_DIR -> ${CHECKPOINT_S3_URI}"
        aws s3 sync "$CKPT_DIR" "${CHECKPOINT_S3_URI}" --delete --only-show-errors || true
    fi
}
trap final_sync EXIT

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
echo "assets=$ASSETS_DIR/$CONFIG  checkpoints=$CKPT_DIR (synced to ${CHECKPOINT_S3_URI:-<none>} every ${CKPT_SYNC_INTERVAL:-1800}s + on exit)"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "==================================="

# NOTE: not `exec` — we need the EXIT trap (final_sync) to run after training, so the
# last checkpoint is flushed to S3. Propagate the trainer's exit code to SageMaker.
python scripts/train.py \
    "$CONFIG" \
    --exp-name="$EXP" \
    --assets-base-dir="$ASSETS_DIR" \
    --checkpoint-base-dir="$CKPT_DIR" \
    "${EXTRA_ARGS[@]}"
TRAIN_RC=$?
echo "Trainer exited with code $TRAIN_RC"
exit $TRAIN_RC
