#!/usr/bin/env bash
set -euo pipefail

cd /shared/openpi

reset_root=/home/ec2-user/data/new_tasks_v2
tasks=$(jq -r 'keys | join(",")' "$reset_root/MAX_OFFICIAL_STEPS.json")

exec /home/ec2-user/micromamba/envs/robocasa/bin/python \
  scripts/generate_unseen_reset_bank.py \
  --root "$reset_root" \
  --episodes 0-19 \
  --seed-base 1000000 \
  --workers 8 \
  --gpus 0,1,2,3,4,5,6,7 \
  --tasks "$tasks" \
  --reuse-existing-limits
