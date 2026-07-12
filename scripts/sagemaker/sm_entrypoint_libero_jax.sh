#!/usr/bin/env bash
# SageMaker entrypoint for the MULTI-NODE LIBERO JAX trainer (pi05_libero_wds).
#
# Differences from sm_entrypoint_jax.sh (robocasa):
#   - LIBERO WebDataset shards stream directly from S3 (boto3) via the path baked into
#     the config's --data.shards; NO `robocasa`/`libero` data channel mount is required.
#   - Multi-node: SageMaker launches ONE process per node (instance_count=2). The JAX
#     trainer calls jax.distributed.initialize (see src/openpi/training/distributed.py),
#     reading /opt/ml/input/config/resourceconfig.json to form one 16-device mesh.
#   - S3 checkpoint sync runs on RANK 0 only (guarded here) to avoid nodes racing.
#
# Required env (set by launch.py):
#   TRAIN_CONFIG  (e.g. pi05_libero_wds)   EXP_NAME
# Optional: TRAIN_ARGS, RESUME/OVERWRITE, CHECKPOINT_S3_URI, CKPT_SYNC_INTERVAL,
#   LIBERO_SHARDS (override --data.shards), SM_MASTER_PORT (coordinator port).
set -euo pipefail

cd /opt/ml/code

CONFIG="${TRAIN_CONFIG:?Set TRAIN_CONFIG (e.g. pi05_libero_wds)}"
EXP="${EXP_NAME:?Set EXP_NAME}"
BASE_CKPT_DIR="/opt/ml/input/data/base_ckpt"
ASSETS_DIR="/opt/ml/code/assets"

# JAX base checkpoint: override the config's gs:// path to the mounted orbax dir.
if [[ -d "$BASE_CKPT_DIR/params" ]]; then
    export OPENPI_WEIGHT_LOADER_PARAMS_PATH="$BASE_CKPT_DIR/params"
elif [[ -d "$BASE_CKPT_DIR" ]]; then
    export OPENPI_WEIGHT_LOADER_PARAMS_PATH="$BASE_CKPT_DIR"
fi

# --- multi-node coordinator port (jax.distributed reads hosts from resourceconfig.json) ---
export SM_MASTER_PORT="${SM_MASTER_PORT:-12355}"
RC="/opt/ml/input/config/resourceconfig.json"
if [[ -f "$RC" ]]; then
    NUM_HOSTS=$(python3 -c "import json;print(len(json.load(open('$RC'))['hosts']))" 2>/dev/null || echo 1)
    CUR_HOST=$(python3 -c "import json;print(json.load(open('$RC'))['current_host'])" 2>/dev/null || echo "")
    echo "resourceconfig: hosts=$NUM_HOSTS current=$CUR_HOST"
fi

# MULTI-NODE checkpointing: orbax multi-host saves assume a filesystem visible to ALL
# processes (primary host creates the checkpoint base dir, non-primary hosts wait for
# it to appear on the SAME path; each host writes its own param shards there). There is
# NO shared POSIX FS across SageMaker nodes, so we use SageMaker's MANAGED
# /opt/ml/checkpoints — both nodes' copies sync bidirectionally to the same S3 location,
# acting as the common store. (Per-node local EBS + rank-0-only sync does NOT work
# multi-node: process 1's shards live on node 2's disk and rank-0 can't upload them, and
# orbax's cross-host base-dir wait times out.) The array_metadata store is disabled in
# checkpoints.py so the eventually-consistent mount doesn't crash finalize / the
# base-dir coordination. launch.py leaves managed checkpoint sync ON for this entrypoint.
CKPT_DIR="/opt/ml/checkpoints"
mkdir -p "$CKPT_DIR" /opt/ml/output/wandb

EXTRA_ARGS=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA_ARGS+=("--overwrite")
[[ "${RESUME:-0}"    == "1" ]] && EXTRA_ARGS+=("--resume")
# Override the LIBERO shards path if provided (else the config default S3 path is used).
[[ -n "${LIBERO_SHARDS:-}" ]] && EXTRA_ARGS+=("--data.shards=${LIBERO_SHARDS}")
# shellcheck disable=SC2206
[[ -n "${TRAIN_ARGS:-}" ]] && EXTRA_ARGS+=( ${TRAIN_ARGS} )

NGPU="$(nvidia-smi -L | wc -l)"
echo "=== SageMaker openpi LIBERO multi-node JAX launch ==="
echo "config=$CONFIG  exp=$EXP  gpus/node=$NGPU"
echo "base_ckpt=${OPENPI_WEIGHT_LOADER_PARAMS_PATH:-<config default>}"
echo "checkpoints=$CKPT_DIR (SageMaker-managed sync to checkpoint_s3_uri)"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "===================================================="

exec python scripts/train.py \
    "$CONFIG" \
    --exp-name="$EXP" \
    --assets-base-dir="$ASSETS_DIR" \
    --checkpoint-base-dir="$CKPT_DIR" \
    "${EXTRA_ARGS[@]}"
