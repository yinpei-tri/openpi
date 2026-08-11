#!/usr/bin/env bash
# Release every GPU held by a COMBINED System2+System1 eval stack.
#
# A stack leaves THREE kinds of process holding GPU memory, and stopping only the obvious one
# leaves the card occupied:
#   1. the rollout clients          (combined_eval.py, MuJoCo/EGL render context)
#   2. the System1 JAX policy servers (serve_policy.py -- XLA preallocates its whole share)
#   3. the System2 vLLM servers     (docker containers named sys2-vllm-*, NOT host processes,
#                                    so pkill can never reach them)
# The fleet launcher itself is killed first so it cannot respawn a client mid-teardown.
#
# Usage:
#   bash examples/robocasa/release_gpus.sh              # release now
#   bash examples/robocasa/release_gpus.sh --wait       # wait for the running sweep, then release
set -uo pipefail

if [[ "${1:-}" == "--wait" ]]; then
  # Wait for the rollout clients to drain. The fleet script's own `wait` is not a reliable signal
  # (it dies with its parent shell when launched via nohup/setsid), so poll for live clients
  # instead. Require several consecutive empty samples: between two episodes of one shard there is
  # a gap with no client running, and exiting on the first empty sample would kill a live sweep.
  echo "[release] waiting for rollout clients to drain..."
  empty=0
  while :; do
    # Match EVERY rollout client, not just combined_eval.py: the fleet's EVAL_SCRIPT knob also
    # runs combine_memory_eval.py, and a pattern that missed it would report "no clients" while a
    # memory sweep was mid-episode and then tear down its servers.
    n=$(pgrep -cf "[c]ombine.*_eval\.py" || true)
    if [[ "${n:-0}" -gt 0 ]]; then
      empty=0
    else
      empty=$((empty + 1))
      # 6 samples x 20s = 2 min of quiet. Comfortably longer than the inter-episode gap
      # (process start + env make), well short of any real episode.
      [[ $empty -ge 6 ]] && break
    fi
    sleep 20
  done
  echo "[release] no clients for 2 min -- sweep is done, releasing."
fi

echo "[release] before: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

# Order matters: launcher -> clients -> servers, so nothing is respawned behind us.
pkill -f "[r]un_combine_fleet.sh" 2>/dev/null && echo "[release] killed fleet launcher" || true
pkill -f "[c]ombine.*_eval\.py"   2>/dev/null && echo "[release] killed rollout clients" || true
pkill -f "[s]erve_policy.py"      2>/dev/null && echo "[release] killed System1 servers" || true

# vLLM runs in containers; pkill cannot see them.
mapfile -t cids < <(docker ps -aq --filter "name=sys2-vllm-" 2>/dev/null)
if [[ ${#cids[@]} -gt 0 ]]; then
  docker rm -f "${cids[@]}" >/dev/null 2>&1 && echo "[release] removed ${#cids[@]} vLLM container(s)"
fi

sleep 8
echo "[release] after : $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
leftover=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | wc -l)
if [[ "$leftover" -gt 0 ]]; then
  echo "[release] WARNING: $leftover process(es) still hold GPU memory:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
else
  echo "[release] all GPUs free."
fi
