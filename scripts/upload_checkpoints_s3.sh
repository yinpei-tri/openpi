#!/bin/bash
# Upload openpi checkpoints to S3.
# Usage:
#   ./scripts/upload_checkpoints_s3.sh                  # upload both
#   ./scripts/upload_checkpoints_s3.sh pytorch          # pytorch only
#   ./scripts/upload_checkpoints_s3.sh jax              # jax only

set -euo pipefail

S3_BASE="s3://tri-ml-datasets-uw2/yinpeidai/checkpoints/pi05_libero"

CKPT_ROOT="/home/yinpei.dai/openpi/checkpoints/pi05_libero"
PT_DIR="$CKPT_ROOT/libero_a100_pt/29999"
JAX_DIR="$CKPT_ROOT/libero_a100_jax/29999"

upload_pytorch() {
    echo "Uploading PyTorch checkpoint (step 29999)..."
    aws s3 sync "$PT_DIR" "$S3_BASE/pytorch/29999/"
    echo "Done: $S3_BASE/pytorch/29999/"
}

upload_jax() {
    echo "Uploading JAX checkpoint (step 29999)..."
    aws s3 sync "$JAX_DIR" "$S3_BASE/jax/29999/"
    echo "Done: $S3_BASE/jax/29999/"
}

TARGET="${1:-all}"

case "$TARGET" in
    pytorch|pt)
        upload_pytorch
        ;;
    jax)
        upload_jax
        ;;
    all)
        upload_pytorch
        upload_jax
        ;;
    *)
        echo "Usage: $0 [pytorch|jax|all]"
        exit 1
        ;;
esac

echo "Upload complete."
