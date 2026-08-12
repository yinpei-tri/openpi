#!/usr/bin/env bash
# Serialize fleet runs behind one lock, for PARALLEL rule-experiment agents.
#
# Why: run_combine_fleet.sh claims all 8 GPUs (one S1 + one vLLM + one rollout client each). Two
# fleets launched at once do not run twice as fast -- they contend for the same servers, and (before
# RUN_LABEL was part of the log dir) they silently overwrote each other's shard files: a 68-episode
# run was truncated to 20 that way. GPU throughput is the ceiling either way, so serializing costs
# nothing: 5 tasks x 20 episodes as five locked 20-episode runs and as one batched 100-episode run
# take the same wall clock. Serializing keeps each experiment's results in its own labelled dir.
#
# Usage -- identical to run_combine_fleet.sh, which this execs once the lock is held:
#   UNITS_FILE=... RUN_LABEL=debug-ArrangeTea-v1 TASK_RULES=1 METHOD=progact STEP=269999 \
#     bash examples/robocasa/run_fleet_locked.sh
#
# Blocks until the lock is free (default 4h). Servers are NEVER started or stopped here: the shared
# 8 stacks are the session's, and a run that started its own would fight the ones already listening.
set -euo pipefail
_SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_OPENPI_REPO=$(cd "$_SELF_DIR/../.." && pwd)
OPENPI_REPO=${OPENPI_REPO:-$_OPENPI_REPO}

LOCK=${FLEET_LOCK:-$OPENPI_REPO/_evallogs/.fleet.lock}
LOCK_WAIT=${LOCK_WAIT:-14400}
mkdir -p "$(dirname "$LOCK")"

# Re-exec under flock, holding fd 9 for the whole run. -w rather than -n: an experiment agent should
# QUEUE behind the run in progress, not fail and lose its work.
if [[ -z "${_FLEET_LOCK_HELD:-}" ]]; then
  echo "[lock] waiting for the fleet lock ($LOCK)..."
  # "bash $0", not "$0": flock execs its command directly, so a checkout without the +x bit would
  # otherwise fail with "Permission denied" after having waited for the lock.
  exec env _FLEET_LOCK_HELD=1 flock -w "$LOCK_WAIT" "$LOCK" bash "$0" "$@"
fi
echo "[lock] acquired by pid $$ ${RUN_LABEL:+for $RUN_LABEL}"

# The 8 warm stacks must already be listening. Without this a locked run against dead servers
# records every episode as a ~20s "Connection refused" failure -- junk that looks like a bad result.
S1_BASE=${S1_BASE:-8060}
S2_BASE=${S2_BASE:-8100}
NGPU=${NGPU:-8}
missing=()
for i in $(seq 0 $((NGPU - 1))); do
  s1=$(ss -lnt 2>/dev/null | grep -c ":$((S1_BASE + i)) " || true)
  s2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$((S2_BASE + i))/v1/models" 2>/dev/null || echo 000)
  [[ "$s1" == "1" && "$s2" == "200" ]] || missing+=("stack$i(S1=$s1 S2=$s2)")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[lock] FATAL: ${#missing[@]} stack(s) not serving: ${missing[*]}" >&2
  echo "[lock] Ask the session owner to bring the fleet up; do not start servers from here." >&2
  exit 1
fi
echo "[lock] all $NGPU stacks serving"

SKIP_SERVERS=1 exec bash "$OPENPI_REPO/examples/robocasa/run_combine_fleet.sh" "$@"
