#!/bin/bash
# Wait for the val-MSE sweep to finish, then run the IDENTICAL sweep on the TRAIN shards
# (same 40 ckpts, same 32768-sample deterministic eval) into a parallel results dir. This gives
# train-seen vs val-unseen action_mse for the generalization-gap plot.
set -u
OPENPI=/home/yinpei.dai/openpi
MASTER=$OPENPI/_evallogs/sweep_master.log
TRAIN_SHARDS=$OPENPI/robocasa_training_data/system1_midset_0717/shards
TRAIN_OUT=$OPENPI/_evallogs/trainmse_results

echo "[chain] waiting for val sweep to complete ..."
while ! grep -q "SWEEP COMPLETE" "$MASTER" 2>/dev/null; do
  sleep 60
done
echo "[chain] val sweep complete. Launching TRAIN-set sweep."
grep "SWEEP COMPLETE" "$MASTER" | tail -1

cd "$OPENPI"
export SWEEP_SHARDS="$TRAIN_SHARDS"
export SWEEP_OUT_DIR="$TRAIN_OUT"
# fresh master log for the train sweep (separate from val's)
exec "$OPENPI/.venv/bin/python" "$OPENPI/scripts/run_val_sweep.py"
