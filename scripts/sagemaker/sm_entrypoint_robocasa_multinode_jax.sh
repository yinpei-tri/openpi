#!/usr/bin/env bash
# SageMaker entrypoint for the MULTI-NODE RoboCasa System1 JAX trainer (pi05_robocasa_system1*).
#
# This is the multi-node sibling of sm_entrypoint_jax.sh (single-node). It keeps that
# entrypoint's RoboCasa data handling verbatim — FastFile mount OR download-to-local-EBS,
# anchors, base_ckpt override, norm_stats check — and ADDS the multi-node pieces:
#   - Multi-node: SageMaker launches ONE process per node (instance_count=N). The JAX
#     trainer calls jax.distributed.initialize (src/openpi/training/distributed.py), which
#     reads /opt/ml/input/config/resourceconfig.json to form one N*8-device mesh. There is
#     NO torchrun (launch.py sets distribution={} for *_jax.sh entrypoints).
#   - S3 checkpoint sync runs on RANK 0 (the sorted-first host) ONLY. The trainer gathers
#     the full (replicated) model to process 0 via process_allgather and writes ONE
#     self-contained orbax checkpoint to rank 0's local EBS; the other nodes have nothing
#     to upload. Managed /opt/ml/checkpoints is deliberately OFF (per-node dir; dual sync
#     corrupted the checkpoint before).
#
# NOTE: multi-node RESUME is not supported (the trainer + robocasa loader reject it). This
# entrypoint is EMA-only, fresh-start. Do not pass RESUME=1 with instance_count>1.
#
# Required env (set by launch.py): TRAIN_CONFIG (e.g. pi05_robocasa_system1), EXP_NAME.
# Optional: TRAIN_ARGS, OVERWRITE, CHECKPOINT_S3_URI, CKPT_SYNC_INTERVAL,
#   ROBOCASA_DATA_S3_URI (download-to-EBS mode), SM_MASTER_PORT (coordinator port).
set -euo pipefail

cd /opt/ml/code

CONFIG="${TRAIN_CONFIG:?Set TRAIN_CONFIG (e.g. pi05_robocasa_system1)}"
EXP="${EXP_NAME:?Set EXP_NAME}"

# SageMaker channels mount under /opt/ml/input/data/<channel>/.
#   robocasa/  -> the dataset root (shards/ + anchors/ + meta.json)
#   base_ckpt/ -> JAX pi05_base orbax checkpoint (the `params` dir)
ROBOCASA_DIR="/opt/ml/input/data/robocasa"
BASE_CKPT_DIR="/opt/ml/input/data/base_ckpt"
ASSETS_DIR="/opt/ml/code/assets"

# --- multi-node coordinator port + rank resolution ------------------------------------
# jax.distributed reads hosts from resourceconfig.json; SM_MASTER_PORT is the coordinator
# port. Rank 0 = the sorted-first host. Single-node (no resourceconfig / one host) -> rank 0.
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

# --- data source: FastFile mount OR download-to-local-EBS (same as single-node) --------
#   (download) If ROBOCASA_DATA_S3_URI is set, `aws s3 sync` the whole dataset to a local
#     EBS dir ONCE up front and read from local disk for the rest of the run. Avoids the
#     FastFile FUSE mount entirely. EACH node downloads its own copy (EBS is per-node).
#   (mount) Otherwise, read from the FastFile-mounted channel at /opt/ml/input/data/robocasa.
if [[ -n "${ROBOCASA_DATA_S3_URI:-}" ]]; then
    LOCAL_DATA="/opt/ml/local_data"
    mkdir -p "$LOCAL_DATA"
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
    export ROBOCASA_SHARDS_DIR="$ROBOCASA_DIR/shards"
fi

if [[ ! -f "$ASSETS_DIR/$CONFIG/robocasa_system1/norm_stats.json" ]]; then
    echo "ERROR: norm_stats.json missing under $ASSETS_DIR/$CONFIG/robocasa_system1/. Rebuild the image." >&2
    exit 1
fi

# JAX base checkpoint: override the config's gs:// path to the mounted orbax dir.
if [[ -d "$BASE_CKPT_DIR/params" ]]; then
    export OPENPI_WEIGHT_LOADER_PARAMS_PATH="$BASE_CKPT_DIR/params"
elif [[ -d "$BASE_CKPT_DIR" ]]; then
    export OPENPI_WEIGHT_LOADER_PARAMS_PATH="$BASE_CKPT_DIR"
fi

# --- checkpointing (rank-0 self-managed S3 sync) --------------------------------------
# The trainer gathers the full (replicated across nodes) model to PROCESS 0 via
# process_allgather and writes ONE self-contained orbax checkpoint to local EBS. We then
# S3-sync it OURSELVES from rank 0 only (managed /opt/ml/checkpoints is a per-node dir +
# racy dual sync that corrupted the checkpoint before). Non-rank-0 nodes skip the sync.
CKPT_DIR="/opt/ml/local_checkpoints"
mkdir -p "$CKPT_DIR" /opt/ml/output/wandb
[[ "$IS_RANK0" == "0" ]] && echo "Not rank 0: skipping S3 checkpoint sync (process 0 owns the full checkpoint)."

# Periodic + final S3 sync (rank 0 only). A file lock serializes the periodic loop against
# the final sync so two `aws s3 sync` never race the same prefix. tmp dirs are excluded so
# an in-progress orbax save never lands in S3. Failures are surfaced (no blanket `|| true`)
# so SageMaker cannot report success without durable checkpoints.
SYNC_PID=""
SYNC_LOCK="/tmp/openpi-checkpoint-s3-sync.lock"
if [[ "$IS_RANK0" == "1" && -n "${CHECKPOINT_S3_URI:-}" ]]; then
    echo "Periodic checkpoint upload: $CKPT_DIR -> ${CHECKPOINT_S3_URI} (every ${CKPT_SYNC_INTERVAL:-1800}s)"
    (
        while true; do
            sleep "${CKPT_SYNC_INTERVAL:-1800}"
            if ! flock "$SYNC_LOCK" aws s3 sync "$CKPT_DIR" "${CHECKPOINT_S3_URI}" \
                --exclude "*.orbax-checkpoint-tmp-*/*" --only-show-errors; then
                echo "WARNING: periodic checkpoint upload failed; final upload will retry." >&2
            fi
        done
    ) &
    SYNC_PID=$!
fi

# Final sync on exit (success OR failure): stop the periodic loop, wait for any in-flight
# sync to release the lock, then one authoritative `--delete` sync so S3 mirrors orbax's
# settled local dir exactly (rank 0 only). CRITICAL: if training SUCCEEDED but the
# checkpoint never reached S3, FAIL the job — else SageMaker reports success and destroys
# the local EBS with no durable checkpoint.
final_sync() {
    local trainer_rc=$?
    local sync_rc=0
    trap - EXIT
    set +e
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
    [[ "$trainer_rc" == "0" && "$sync_rc" != "0" ]] && trainer_rc=$sync_rc
    exit "$trainer_rc"
}
trap final_sync EXIT

EXTRA_ARGS=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA_ARGS+=("--overwrite")
# RESUME intentionally NOT wired: multi-node resume is unsupported (trainer + loader reject it).
# shellcheck disable=SC2206
[[ -n "${TRAIN_ARGS:-}" ]] && EXTRA_ARGS+=( ${TRAIN_ARGS} )

# AWS Batch / SageMaker inject WANDB_RUN_ID / WANDB_NAME / WANDB_RUN_GROUP (the ugly
# "AWSBatch...-ip-..." string) into the container env. wandb env vars take precedence over
# some wandb.init args, so unset them here and let the trainer set its OWN id/name
# (exp_name). Keep WANDB_PROJECT / WANDB_MODE / WANDB_API_KEY, which launch.py sets.
unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_RESUME

NGPU="$(nvidia-smi -L | wc -l)"
echo "=== SageMaker openpi RoboCasa multi-node JAX launch ==="
echo "config=$CONFIG  exp=$EXP  gpus/node=$NGPU  rank0=$IS_RANK0"
echo "ROBOCASA_SHARDS_DIR=$ROBOCASA_SHARDS_DIR"
echo "base_ckpt=${OPENPI_WEIGHT_LOADER_PARAMS_PATH:-<config default>}"
echo "checkpoints=$CKPT_DIR (process-0 full write; rank-0 self-sync to ${CHECKPOINT_S3_URI:-<none>})"
echo "extra_args=${EXTRA_ARGS[*]:-}"
echo "======================================================="

# NOT `exec` — keep the shell alive so the EXIT trap (final_sync) runs after training.
# Unbuffered output preserves the first Python traceback if one distributed process fails.
python -u scripts/train.py \
    "$CONFIG" \
    --exp-name="$EXP" \
    --assets-base-dir="$ASSETS_DIR" \
    --checkpoint-base-dir="$CKPT_DIR" \
    "${EXTRA_ARGS[@]}"
TRAIN_RC=$?
echo "Trainer exited with code $TRAIN_RC"
exit $TRAIN_RC
