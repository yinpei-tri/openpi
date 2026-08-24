#!/usr/bin/env bash
# Run System1+System2 rollouts from a frozen procedural reset bank across warm server stacks.
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_REPO=${OPENPI_REPO:-$(cd "$SELF_DIR/../.." && pwd)}
DATA_DIR=${DATA_DIR:-$(dirname "$OPENPI_REPO")/data}
RESET_ROOT=${RESET_ROOT:-/home/ec2-user/data/new_tasks}
BENCHMARK_JSON=${BENCHMARK_JSON:-$RESET_ROOT/BENCHMARK.json}
TASK_LIMITS_JSON=${TASK_LIMITS_JSON:-$RESET_ROOT/MAX_STEPS.json}
RUN_LABEL=${RUN_LABEL:-debug-newtask}
EPISODES=${EPISODES:-0}
TASKS=${TASKS:-}
SEED_BASE=${SEED_BASE:-1000000}
NGPU=${NGPU:-8}
S1_BASE=${S1_BASE:-8060}
S2_BASE=${S2_BASE:-8100}
METHOD=${METHOD:-progact}
STEP=${STEP:-269999}
GENERAL_RULES=${GENERAL_RULES:-1}
TASK_RULES=${TASK_RULES:-1}
LAST_MILESTONE_RETRY=${LAST_MILESTONE_RETRY:-$GENERAL_RULES}
ROLLOUT_LIMIT_MODE=${ROLLOUT_LIMIT_MODE:-max_official_steps}
MAX_TURNS=${MAX_TURNS:-20}
MAX_STEPS_CAP=${MAX_STEPS_CAP:-400}
HIGHRES_VIDEO=${HIGHRES_VIDEO:-0}
MAX_S2_CALLS_SAFETY=${MAX_S2_CALLS_SAFETY:-100}
RESUME=${RESUME:-0}
ROBOCASA_PY=${ROBOCASA_PY:-/home/ec2-user/micromamba/envs/robocasa/bin/python}
SYS1_CKPT=${SYS1_CKPT:-$DATA_DIR/sys1_ckpts/f0717-270k-bs512-progact_granfine_verbsimp_noexec/$STEP}
S2_CKPT=${S2_CKPT:-$DATA_DIR/sys2_ckpts/system2-full-0804-qwen3vl-4b-gb192-full-vitfull-lr1e5-vitlr2e6-alignerlr1e5-zero2-2n-ep3/checkpoint-17124}
OUT_ROOT=${OUT_ROOT:-$DATA_DIR/sys1_eval_results/combine}
LOG_ROOT=${LOG_ROOT:-$OPENPI_REPO/_evallogs/seeded_${RUN_LABEL}}
FLEET_LOCK=${FLEET_LOCK:-$OPENPI_REPO/_evallogs/.fleet.lock}
LOCK_WAIT=${LOCK_WAIT:-14400}

[[ "$METHOD" == "progact" ]] || { echo "FATAL: this runner currently expects METHOD=progact" >&2; exit 1; }
[[ -f "$TASK_LIMITS_JSON" ]] || { echo "FATAL: missing $TASK_LIMITS_JSON" >&2; exit 1; }
[[ -d "$SYS1_CKPT" ]] || { echo "FATAL: missing $SYS1_CKPT" >&2; exit 1; }
[[ -d "$S2_CKPT" ]] || { echo "FATAL: missing $S2_CKPT" >&2; exit 1; }

mkdir -p "$LOG_ROOT" "$(dirname "$FLEET_LOCK")"
if [[ -z "${SEEDED_FLEET_LOCK_HELD:-}" ]]; then
  echo "[seeded-fleet] waiting for $FLEET_LOCK"
  exec env SEEDED_FLEET_LOCK_HELD=1 flock -w "$LOCK_WAIT" "$FLEET_LOCK" bash "$0" "$@"
fi

for i in $(seq 0 $((NGPU - 1))); do
  s1=$(ss -lnt 2>/dev/null | grep -c ":$((S1_BASE + i)) " || true)
  s2=$(curl -s -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:$((S2_BASE + i))/v1/models" 2>/dev/null || true)
  [[ "$s1" == 1 && "$s2" == 200 ]] || {
    echo "FATAL: stack$i is not ready (S1=$s1 S2=${s2:-000})" >&2
    exit 1
  }
done

WORK="$LOG_ROOT/work.tsv"
"$ROBOCASA_PY" - "$BENCHMARK_JSON" "$TASK_LIMITS_JSON" "$RESET_ROOT" \
  "$EPISODES" "$TASKS" >"$WORK" <<'PY'
import json
import sys
from pathlib import Path

benchmark, limits_json, reset_root, spec, task_spec = sys.argv[1:]
benchmark_path = Path(benchmark)
limits = json.loads(Path(limits_json).read_text())
if not isinstance(limits, dict) or not limits:
    raise SystemExit(f"task limits must be a non-empty object: {limits_json}")
if benchmark_path.is_file():
    tasks = [row["task"] for row in json.loads(benchmark_path.read_text())["tasks"]]
else:
    # The v1/v2 reset banks were consolidated into one directory after generation, so their old
    # BENCHMARK.json sidecars no longer exist. MAX_STEPS is now the authoritative task manifest;
    # require a matching reset-bank manifest for every key so a typo cannot silently become work.
    tasks = list(limits)
    missing = [task for task in tasks
               if not (Path(reset_root) / task / "manifest.json").is_file()]
    if missing:
        raise SystemExit(f"MAX_STEPS tasks missing reset manifests: {missing}")
if task_spec.strip():
    requested = [task.strip() for task in task_spec.split(",") if task.strip()]
    unknown = sorted(set(requested) - set(tasks))
    if unknown:
        raise SystemExit(f"TASKS contains tasks absent from benchmark: {unknown}")
    requested_set = set(requested)
    tasks = [task for task in tasks if task in requested_set]
episodes = []
for part in spec.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part.lstrip("-"):
        lo, hi = part.split("-", 1)
        episodes.extend(range(int(lo), int(hi) + 1))
    else:
        episodes.append(int(part))
for task in tasks:
    for episode in sorted(set(episodes)):
        print(f"{task}\t{episode}")
PY

NUNIT=$(wc -l <"$WORK")
[[ "$NUNIT" -gt 0 ]] || { echo "FATAL: no work units" >&2; exit 1; }
echo "[seeded-fleet] $NUNIT units, method=$RUN_LABEL, rules general=$GENERAL_RULES task=$TASK_RULES"

mkdir -p "$LOG_ROOT/failures"
for i in $(seq 0 $((NGPU - 1))); do
  awk -v n="$NGPU" -v k="$i" '((NR-1) % n) == k' "$WORK" >"$LOG_ROOT/shard$i.tsv"
  (
    while IFS=$'\t' read -r task episode; do
      echo "[stack$i] START $task episode=$episode"
      args=(
        examples/robocasa/seeded_combined_eval.py
        --task "$task" --episodes "$episode" --seed-base "$SEED_BASE"
        --load-reset-root "$RESET_ROOT"
        --s1-dir "$SYS1_CKPT" --s2-dir "$S2_CKPT"
        --s1-port "$((S1_BASE + i))" --s2-port "$((S2_BASE + i))"
        --s2-model system2-full
        --norm-stats "$SYS1_CKPT/assets/robocasa_system1/norm_stats.json"
        --out-root "$OUT_ROOT" --method "$RUN_LABEL"
        --max-turns "$MAX_TURNS" --rollout-limit-mode "$ROLLOUT_LIMIT_MODE"
        --task-limits-json "$TASK_LIMITS_JSON"
        --max-s2-calls-safety "$MAX_S2_CALLS_SAFETY"
        --max-steps-cap "$MAX_STEPS_CAP"
      )
      [[ "$HIGHRES_VIDEO" == 1 ]] && args+=(--highres-video) || args+=(--no-highres-video)
      [[ "$GENERAL_RULES" == 1 ]] && args+=(--general-rules) || args+=(--no-general-rules)
      [[ "$TASK_RULES" == 1 ]] && args+=(--task-rules) || args+=(--no-task-rules)
      [[ "$LAST_MILESTONE_RETRY" == 1 ]] \
        && args+=(--last-milestone-retry) || args+=(--no-last-milestone-retry)
      [[ "$RESUME" == 1 ]] && args+=(--resume)
      if ! MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$i" \
          "$ROBOCASA_PY" "${args[@]}"; then
        echo "$task $episode" >>"$LOG_ROOT/failures/stack$i.txt"
        echo "[stack$i] COMMAND FAILED $task episode=$episode" >&2
      fi
    done <"$LOG_ROOT/shard$i.tsv"
    echo "[stack$i] SHARD COMPLETE"
  ) >"$LOG_ROOT/client-stack$i.log" 2>&1 &
done

wait
failures=$(find "$LOG_ROOT/failures" -type f -size +0c -print | wc -l)
if [[ "$failures" -gt 0 ]]; then
  echo "[seeded-fleet] command failures recorded under $LOG_ROOT/failures" >&2
  exit 1
fi
echo "[seeded-fleet] ALL SHARDS COMPLETE"
