#!/usr/bin/env bash
# Launch a COMBINED System2+System1 benchmark sweep across all 8 GPUs.
#
# Per GPU we start a full independent stack — one System1 JAX policy server, one System2 vLLM
# server, and one RoboCasa rollout client — then shard the (task, episode) work list across the 8
# stacks. Nothing is shared between GPUs, so there is no cross-talk and a dead GPU only loses its
# own shard.
#
#   GPU g : S1 policy on port (S1_BASE+g)   |  S2 vLLM on port (S2_BASE+g)  |  1 rollout client
#
# Memory: the S1 server takes XLA_FRAC of the GPU and vLLM takes GPU_FRAC; keep the sum <= ~0.9 so
# MuJoCo/EGL still has room to render (it shares the same device).
#
# Usage:
#   METHOD=progreg TASKS=CloseFridge EPISODES=0-9  bash examples/robocasa/run_combine_fleet.sh
#   METHOD=progact TASK_SET=atomic_seen EPISODES=0-4 bash examples/robocasa/run_combine_fleet.sh
#
# Monitor:  tail -f _evallogs/fleet_<method>/client-gpu*.log
# Results:  eval_results/combine/<derived-method-name>/
set -euo pipefail
cd /home/ec2-user/openpi

METHOD=${METHOD:-progreg}                  # progreg | progact  (which System1 head)
EPISODES=${EPISODES:-0}                    # per-task episode spec, e.g. "0", "0-9", "0,5,10"
TASKS=${TASKS:-}                           # explicit comma list; overrides TASK_SET
USE_EVAL_SET=${USE_EVAL_SET:-0}            # 1 = use the 1000-episode manifest inlined in
                                           # combined_eval.TARGET_EVAL_EPISODES
TASK_SET=${TASK_SET:-all}                  # all (the 50-task benchmark) | atomic_seen |
                                           # composite_seen | composite_unseen
NGPU=${NGPU:-8}
GPUS=${GPUS:-$(seq -s, 0 $((NGPU-1)))}
S1_BASE=${S1_BASE:-8060}
S2_BASE=${S2_BASE:-8100}
XLA_FRAC=${XLA_FRAC:-0.32}                 # System1 JAX share
GPU_FRAC=${GPU_FRAC:-0.42}                 # System2 vLLM share
MAX_TURNS=${MAX_TURNS:-20}
DATA_ROOT=${DATA_ROOT:-/home/ec2-user/data/robocasa_dataset/v1.0/target}
S2_CKPT=${S2_CKPT:-/home/ec2-user/sys2_train_eval/data/sys2_ckpts/system2-full-0804-qwen35-4b-gb192-full-vitfull-lr1e5-vitlr2e6-alignerlr1e5-zero2-2n-ep3/checkpoint-11416}
SKIP_SERVERS=${SKIP_SERVERS:-0}            # 1 = reuse servers already listening
RESUME=${RESUME:-0}                        # 1 = skip episodes already finished (restartable)
ROBOCASA_PY=${ROBOCASA_PY:-/home/ec2-user/micromamba/envs/robocasa/bin/python}

case "$METHOD" in
  progreg) S1_CKPT=/home/ec2-user/data/sys1_ckpts/f0717-270k-bs512-progreg_granfine_verbsimp_noexec/269999 ;;
  progact) S1_CKPT=/home/ec2-user/data/sys1_ckpts/f0717-270k-bs512-progact_granfine_verbsimp_noexec/269999 ;;
  *) echo "METHOD must be progreg or progact" >&2; exit 1 ;;
esac
[[ -d "$S1_CKPT" ]] || { echo "FATAL: missing S1 ckpt $S1_CKPT" >&2; exit 1; }
[[ -d "$S2_CKPT" ]] || { echo "FATAL: missing S2 ckpt $S2_CKPT" >&2; exit 1; }

LOG=_evallogs/fleet_$METHOD
mkdir -p "$LOG"
IFS=',' read -ra GPULIST <<< "$GPUS"
NG=${#GPULIST[@]}

# ---- work list --------------------------------------------------------------------------------
# Either an explicit (task, episode) manifest (EPISODE_JSON) or TASK_SET x EPISODES.
WORK="$LOG/worklist.txt"
: > "$WORK"
# robocasa env: the registry import needs robosuite, absent from system python3.
"$ROBOCASA_PY" - "$DATA_ROOT" "$TASK_SET" "$TASKS" "$EPISODES" "$USE_EVAL_SET" \
  >> "$WORK" 2>"$LOG/worklist.err" <<'PY'
import sys, glob, os
root, task_set, tasks_csv, eps, use_eval_set = sys.argv[1:6]
sys.path.insert(0, "/home/ec2-user/openpi/examples/robocasa")
missing = []

def lerobot_dir(task):
    hits = sorted(glob.glob(os.path.join(root, "*", task, "*", "lerobot")))
    return hits[0] if hits else None

if use_eval_set == "1":
    # The 1000-episode manifest, inlined in combined_eval (no sidecar JSON). ONE line per
    # (task, episode) so nothing is inferred from a range.
    from combined_eval import TARGET_EVAL_EPISODES
    for task, episodes in TARGET_EVAL_EPISODES.items():
        ld = lerobot_dir(task)
        if not ld:
            missing.append(task)
            continue
        for e in episodes:
            print(f"{ld} {int(e)}")
else:
    if tasks_csv.strip():
        names = [t.strip() for t in tasks_csv.split(",") if t.strip()]
    else:
        from robocasa.utils.dataset_registry import TARGET_TASKS
        sets = (["atomic_seen", "composite_seen", "composite_unseen"]
                if task_set == "all" else [task_set])
        names = [t for s in sets for t in TARGET_TASKS[s]]
    for n in names:
        ld = lerobot_dir(n)
        print(f"{ld} {eps}") if ld else missing.append(n)
if missing:
    # Never silently skip: a task absent from the local dataset must be visible in the log.
    print("# MISSING (no local lerobot dir): " + ",".join(missing), file=sys.stderr)
PY
# Keep ONLY well-formed "<path>/lerobot <episode-spec>" lines, so a stray warning or a
# partial write can never be mistaken for a task.
grep -E '^/.*/lerobot [0-9]' "$WORK" > "$WORK.clean" || true
mv "$WORK.clean" "$WORK"
NTASK=$(wc -l < "$WORK")
echo "[fleet] method=$METHOD  src=$([[ $USE_EVAL_SET == 1 ]] && echo TARGET_EVAL_EPISODES || echo taskset:$TASK_SET)  lines=$NTASK  gpus=${GPULIST[*]}"
[[ -s "$LOG/worklist.err" ]] && grep -i "MISSING" "$LOG/worklist.err" || true
[[ "$NTASK" -gt 0 ]] || { echo "FATAL: empty work list" >&2; exit 1; }

# ---- servers ----------------------------------------------------------------------------------
if [[ "$SKIP_SERVERS" != "1" ]]; then
  for i in "${!GPULIST[@]}"; do
    g=${GPULIST[$i]}
    XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRAC CUDA_VISIBLE_DEVICES=$g \
      nohup .venv/bin/python scripts/serve_policy.py --port $((S1_BASE+g)) \
        policy:checkpoint --policy.config=auto --policy.dir "$S1_CKPT" \
        > "$LOG/s1-gpu$g.log" 2>&1 &
    MODEL_DIR="$S2_CKPT" GPU=$g GPU_FRAC=$GPU_FRAC PORT=$((S2_BASE+g)) \
      NAME=sys2-vllm-fleet-$g MEDIA_DIR=/tmp/sys2_media \
      nohup bash /home/ec2-user/sys2_train_eval/scripts/serve_system2_vllm.sh \
        > "$LOG/s2-gpu$g.log" 2>&1 &
    echo "[fleet] gpu$g: S1 :$((S1_BASE+g))  S2 :$((S2_BASE+g))"
  done

  echo "[fleet] waiting for all servers (vLLM takes ~4 min to load)..."
  for i in "${!GPULIST[@]}"; do
    g=${GPULIST[$i]}
    for _ in $(seq 1 90); do
      s1=$(ss -lnt 2>/dev/null | grep -c ":$((S1_BASE+g)) " || true)
      s2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$((S2_BASE+g))/v1/models" 2>/dev/null || echo 000)
      [[ "$s1" == "1" && "$s2" == "200" ]] && break
      sleep 10
    done
    [[ "${s1:-0}" == "1" && "${s2:-000}" == "200" ]] \
      && echo "[fleet] gpu$g READY" \
      || { echo "[fleet] gpu$g FAILED (S1=$s1 S2=$s2); see $LOG/{s1,s2}-gpu$g.log" >&2; }
  done
fi

# ---- shard by (task, EPISODE) so the load balances -------------------------------------------
# Sharding whole TASKS would give 50/8 = 6 or 7 tasks per GPU, and a task's cost scales with its
# episode count -- so one GPU can finish long before another. Expanding to one line per (task,
# episode) makes every unit the same size (one episode) and round-robin then splits 150 units into
# 18/19 per GPU. Each unit is still one combined_eval invocation, so a crash loses one episode.
UNITS="$LOG/units.txt"
"$ROBOCASA_PY" - "$WORK" > "$UNITS" <<'PY'
import sys
spec_cache = {}
def expand(spec):
    if spec in spec_cache:
        return spec_cache[spec]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    spec_cache[spec] = sorted(dict.fromkeys(out))
    return spec_cache[spec]
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    ld, spec = line.rsplit(" ", 1)
    for e in expand(spec):
        print(f"{ld} {e}")
PY
NUNIT=$(wc -l < "$UNITS")
echo "[fleet] work units (task,episode pairs): $NUNIT across $NG gpus"

for i in "${!GPULIST[@]}"; do
  g=${GPULIST[$i]}
  awk -v n="$NG" -v k="$i" '{ if (((NR-1) % n) == k) print }' "$UNITS" > "$LOG/shard-gpu$g.txt"
  echo "[fleet] gpu$g -> $(wc -l < "$LOG/shard-gpu$g.txt") episodes"
done

for i in "${!GPULIST[@]}"; do
  g=${GPULIST[$i]}
  (
    while read -r ld ep; do
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$g \
        /home/ec2-user/micromamba/envs/robocasa/bin/python examples/robocasa/combined_eval.py \
          --lerobot-dir "$ld" --episodes "$ep" \
          --s1-dir "$S1_CKPT" --s2-dir "$S2_CKPT" \
          --s1-port $((S1_BASE+g)) --s2-port $((S2_BASE+g)) --s2-model system2-full \
          --norm-stats "$S1_CKPT/assets/robocasa_system1/norm_stats.json" \
          --out-root eval_results/combine --max-turns "$MAX_TURNS" \
          $([[ "$RESUME" == 1 ]] && echo --resume) \
        || echo "[fleet] gpu$g FAILED $ld ep$ep" >&2
    done < "$LOG/shard-gpu$g.txt"
    echo "[fleet] gpu$g SHARD COMPLETE"
  ) > "$LOG/client-gpu$g.log" 2>&1 &
done

echo "[fleet] $NG clients launched. Monitor: tail -f $LOG/client-gpu*.log"
wait
echo "[fleet] ALL SHARDS COMPLETE for method=$METHOD"
