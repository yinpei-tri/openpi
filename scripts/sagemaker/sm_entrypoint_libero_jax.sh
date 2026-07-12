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
IS_RANK0=1
if [[ -f "$RC" ]]; then
    NUM_HOSTS=$(python3 -c "import json;print(len(json.load(open('$RC'))['hosts']))" 2>/dev/null || echo 1)
    FIRST_HOST=$(python3 -c "import json;print(sorted(json.load(open('$RC'))['hosts'])[0])" 2>/dev/null || echo "")
    CUR_HOST=$(python3 -c "import json;print(json.load(open('$RC'))['current_host'])" 2>/dev/null || echo "")
    echo "resourceconfig: hosts=$NUM_HOSTS current=$CUR_HOST rank0=$FIRST_HOST"
    [[ -n "$FIRST_HOST" && "$CUR_HOST" != "$FIRST_HOST" ]] && IS_RANK0=0
fi

# Checkpoint to instance-EBS (not the eventually-consistent /opt/ml/checkpoints), and
# S3-sync ourselves — RANK 0 ONLY (else nodes race the upload). See the robocasa
# entrypoint for the orbax-finalize rationale.
CKPT_DIR="/opt/ml/local_checkpoints"
mkdir -p "$CKPT_DIR" /opt/ml/output/wandb
[[ "$IS_RANK0" == "0" ]] && echo "Not rank 0: skipping S3 checkpoint sync."

SYNC_PID=""
if [[ "$IS_RANK0" == "1" && -n "${CHECKPOINT_S3_URI:-}" ]]; then
    echo "Periodic checkpoint upload: $CKPT_DIR -> ${CHECKPOINT_S3_URI} (every ${CKPT_SYNC_INTERVAL:-1800}s)"
    (
        while true; do
            sleep "${CKPT_SYNC_INTERVAL:-1800}"
            aws s3 sync "$CKPT_DIR" "${CHECKPOINT_S3_URI}" --exclude "*.orbax-checkpoint-tmp-*/*" --only-show-errors || true
        done
    ) &
    SYNC_PID=$!
fi
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
# Override the LIBERO shards path if provided (else the config default S3 path is used).
[[ -n "${LIBERO_SHARDS:-}" ]] && EXTRA_ARGS+=("--data.shards=${LIBERO_SHARDS}")
# shellcheck disable=SC2206
[[ -n "${TRAIN_ARGS:-}" ]] && EXTRA_ARGS+=( ${TRAIN_ARGS} )

NGPU="$(nvidia-smi -L | wc -l)"
echo "=== SageMaker openpi LIBERO multi-node JAX launch ==="
echo "config=$CONFIG  exp=$EXP  gpus/node=$NGPU  rank0=$IS_RANK0"
echo "base_ckpt=${OPENPI_WEIGHT_LOADER_PARAMS_PATH:-<config default>}"
echo "checkpoints=$CKPT_DIR (synced to ${CHECKPOINT_S3_URI:-<none>} by rank0)"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "===================================================="

# NOT exec — keep the shell alive for the EXIT-trap final sync. Propagate exit code.
python scripts/train.py \
    "$CONFIG" \
    --exp-name="$EXP" \
    --assets-base-dir="$ASSETS_DIR" \
    --checkpoint-base-dir="$CKPT_DIR" \
    "${EXTRA_ARGS[@]}"
TRAIN_RC=$?
echo "Trainer exited with code $TRAIN_RC"
exit $TRAIN_RC
