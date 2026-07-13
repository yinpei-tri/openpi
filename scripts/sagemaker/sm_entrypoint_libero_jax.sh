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

# MULTI-NODE checkpointing: the trainer scopes the orbax save to PROCESS 0 ONLY
# (active_processes={0}) and writes the FULL model (use_replica_parallel=False), which is
# valid because the model is REPLICATED across nodes (FSDP shards within a node; the node
# axis is a data-parallel replica). So process 0 (== rank-0 host) writes a COMPLETE
# checkpoint to its OWN local EBS — no shared filesystem, no cross-node coordination.
# We then S3-sync it OURSELVES from rank 0 only (managed /opt/ml/checkpoints is a
# per-node dir + racy dual sync that corrupted the checkpoint before). Non-rank-0 nodes
# have no checkpoint to upload.
CKPT_DIR="/opt/ml/local_checkpoints"
mkdir -p "$CKPT_DIR" /opt/ml/output/wandb
[[ "$IS_RANK0" == "0" ]] && echo "Not rank 0: skipping S3 checkpoint sync (process 0 owns the full checkpoint)."

# --resume: a NEW SageMaker job gets fresh, EMPTY local volumes, so we must first pull the
# previous job's checkpoint tree down from S3 or the trainer silently starts from scratch.
# Restore on EVERY host (each host's orbax reads its own local filesystem on restore).
if [[ "${RESUME:-0}" == "1" ]]; then
    : "${RESUME_S3_URI:?RESUME=1 but RESUME_S3_URI unset; set output.resume_s3_uri to the previous job checkpoint root}"
    echo "Restoring checkpoints on this host: ${RESUME_S3_URI} -> $CKPT_DIR"
    aws s3 sync "${RESUME_S3_URI}" "$CKPT_DIR" --only-show-errors
    if [[ ! -d "$CKPT_DIR/$CONFIG/$EXP" ]] || \
       [[ -z "$(find "$CKPT_DIR/$CONFIG/$EXP" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit)" ]]; then
        echo "ERROR: no checkpoint steps found at ${RESUME_S3_URI}${CONFIG}/${EXP}/ — cannot resume." >&2
        exit 1
    fi
fi

SYNC_INTERVAL="${CKPT_SYNC_INTERVAL:-1800}"
if [[ ! "$SYNC_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CKPT_SYNC_INTERVAL must be a positive integer, got: $SYNC_INTERVAL" >&2
    exit 1
fi

# Periodic + final S3 sync (rank 0 only). A file lock serializes the periodic loop against
# the final sync so two `aws s3 sync` never race the same prefix. tmp dirs are excluded so
# an in-progress orbax save never lands in S3. Failures are surfaced (no blanket `|| true`)
# so SageMaker cannot report success without durable checkpoints.
SYNC_PID=""
SYNC_LOCK="/tmp/openpi-checkpoint-s3-sync.lock"
if [[ "$IS_RANK0" == "1" && -n "${CHECKPOINT_S3_URI:-}" ]]; then
    echo "Periodic checkpoint upload: $CKPT_DIR -> ${CHECKPOINT_S3_URI} (every ${SYNC_INTERVAL}s)"
    (
        while true; do
            sleep "$SYNC_INTERVAL"
            if ! flock "$SYNC_LOCK" aws s3 sync "$CKPT_DIR" "${CHECKPOINT_S3_URI}" \
                --exclude "*.orbax-checkpoint-tmp-*/*" --only-show-errors; then
                echo "WARNING: periodic checkpoint upload failed; final upload will retry." >&2
            fi
        done
    ) &
    SYNC_PID=$!
fi
final_sync() {
    local trainer_rc=$?
    local sync_rc=0
    trap - EXIT
    set +e
    # Stop the periodic loop AND wait for any in-flight sync to release the lock before we
    # run the authoritative final sync (otherwise the two race the same prefix).
    [[ -n "$SYNC_PID" ]] && kill "$SYNC_PID" 2>/dev/null
    [[ -n "$SYNC_PID" ]] && wait "$SYNC_PID" 2>/dev/null
    if [[ "$IS_RANK0" == "1" && -n "${CHECKPOINT_S3_URI:-}" ]]; then
        echo "Final checkpoint upload: $CKPT_DIR -> ${CHECKPOINT_S3_URI}"
        flock "$SYNC_LOCK" aws s3 sync "$CKPT_DIR" "${CHECKPOINT_S3_URI}" --delete \
            --exclude "*.orbax-checkpoint-tmp-*/*" --only-show-errors
        sync_rc=$?
        if [[ "$sync_rc" == "0" ]]; then
            # --delete skips excluded keys, so purge any stale tmp objects a prior uploader
            # may have left once the authoritative sync succeeds.
            aws s3 rm "${CHECKPOINT_S3_URI}" --recursive --exclude "*" \
                --include "*.orbax-checkpoint-tmp-*/*" --only-show-errors
            sync_rc=$?
        fi
        [[ "$sync_rc" != "0" ]] && echo "ERROR: final checkpoint upload failed with code $sync_rc" >&2
    fi
    # If training succeeded but the checkpoint never reached S3, FAIL the job.
    [[ "$trainer_rc" == "0" && "$sync_rc" != "0" ]] && trainer_rc=$sync_rc
    exit "$trainer_rc"
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
echo "checkpoints=$CKPT_DIR (process-0 full write; rank-0 self-sync to ${CHECKPOINT_S3_URI:-<none>})"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "===================================================="

# NOT exec — keep the shell alive so the EXIT trap (final_sync) runs. Propagate exit code.
python scripts/train.py \
    "$CONFIG" \
    --exp-name="$EXP" \
    --assets-base-dir="$ASSETS_DIR" \
    --checkpoint-base-dir="$CKPT_DIR" \
    "${EXTRA_ARGS[@]}"
TRAIN_RC=$?
echo "Trainer exited with code $TRAIN_RC"
exit $TRAIN_RC
