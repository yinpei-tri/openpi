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

# Rank guard for multi-node: only the FIRST host (rank 0) uploads checkpoints, else the
# N nodes race the same S3 sync. SageMaker exposes hosts + current_host in
# resourceconfig.json; the sorted-first host is rank 0. Single-node -> always rank 0.
# (Only rank 0's orbax process writes the full checkpoint anyway — other ranks
# contribute shards via the collective but process_index 0 owns the save.)
IS_RANK0=1
RC="/opt/ml/input/config/resourceconfig.json"
if [[ -f "$RC" ]]; then
    FIRST_HOST=$(python3 -c "import json;print(sorted(json.load(open('$RC'))['hosts'])[0])" 2>/dev/null || echo "")
    CUR_HOST=$(python3 -c "import json;print(json.load(open('$RC'))['current_host'])" 2>/dev/null || echo "")
    if [[ -n "$FIRST_HOST" && "$CUR_HOST" != "$FIRST_HOST" ]]; then
        IS_RANK0=0
        echo "This host ($CUR_HOST) is NOT rank 0 ($FIRST_HOST): skipping S3 checkpoint sync."
    fi
fi

# Periodic background upload (default every 30 min; override via CKPT_SYNC_INTERVAL).
# Crash insurance: if the instance dies mid-run, S3 still has checkpoints up to the
# last tick (local EBS dies with the instance). We EXCLUDE orbax's in-progress
# `*.orbax-checkpoint-tmp-*` dirs so a partially-written checkpoint never lands in S3;
# orbax renames the tmp dir to its final name only once the save is complete. No
# `--delete` here (a periodic delete could race orbax's pruning) — the final sync
# reconciles. Long interval keeps S3 traffic/overhead low.
SYNC_PID=""
if [[ "$IS_RANK0" == "1" && -n "${CHECKPOINT_S3_URI:-}" ]]; then
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
    if [[ "$IS_RANK0" == "1" && -n "${CHECKPOINT_S3_URI:-}" ]]; then
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
