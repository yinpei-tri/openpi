#!/usr/bin/env bash
# Launch a COMBINED System2+System1 benchmark sweep across all 8 GPUs.
#
# Per GPU we start a full independent stack — one System1 JAX policy server, one System2 vLLM
# server, and one RoboCasa rollout client — then shard the (task, episode) work list across the 8
# stacks. Nothing is shared between GPUs, so there is no cross-talk and a dead GPU only loses its
# own shard.
#
#   stack i : S1 policy on port (S1_BASE+i)  |  S2 vLLM on port (S2_BASE+i)  |  1 rollout client
#
# Memory: the S1 server takes XLA_FRAC of the GPU and vLLM takes GPU_FRAC; keep the sum <= ~0.9 so
# MuJoCo/EGL still has room to render (it shares the S1 device). To give each server a whole GPU
# instead, split them with S1_GPUS/S2_GPUS -- useful when something else already holds memory on the
# card, or to run one stack at full size.
#
# Usage:
#   METHOD=progreg TASKS=CloseFridge EPISODES=0-9  bash examples/robocasa/run_combine_fleet.sh
#   METHOD=progact TASK_SET=atomic_seen EPISODES=0-4 bash examples/robocasa/run_combine_fleet.sh
#   # one stack, S1+sim on GPU0 and S2 on GPU1:
#   METHOD=progreg-noanchor STEP=240000 S1_GPUS=0 S2_GPUS=1 XLA_FRAC=0.9 GPU_FRAC=0.9 \
#     TASKS=CloseFridge EPISODES=0 bash examples/robocasa/run_combine_fleet.sh
#   # MEMORY variant (narrate the demo -> recipe -> warm plan -> execute); own results dir + log dir:
#   EVAL_SCRIPT=combine_memory_eval.py METHOD=progact TASK_SET=composite_unseen EPISODES=0-4 \
#     bash examples/robocasa/run_combine_fleet.sh
#
# Monitor:  tail -f _evallogs/fleet_<method>_<step>[_<variant>]/client-stack*.log
# Results:  eval_results/combine/<derived-method-name>/
set -euo pipefail

# ---- PATHS ------------------------------------------------------------------------------------
# Nothing is hardcoded to one machine's layout. Resolution order for every root:
#   1. the env contract (REPO_ROOT / DATA_DIR / SYS1_CKPT_DIR / ... , exported by shared_bashrc on
#      the shared-filesystem setup); a non-interactive shell never sources that file, hence 2-3.
#   2. derived from THIS SCRIPT's location: it lives at <openpi>/examples/robocasa/, so the openpi
#      checkout and its parent are known without naming a mount. Works for /shared/openpi,
#      ~/openpi, or any other path.
#   3. DATA_DIR: the first EXISTING candidate of <repo_root>/data, <openpi>/../data, ~/data, so
#      the classic "repo + sibling data/" workstation layout keeps working untouched.
# Venvs are deliberately NOT derived from these: they are node-local (not relocatable, and
# small-file imports over a shared mount are slow) -- see ROBOCASA_PY / OPENPI_PY below.
_SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_OPENPI_REPO=$(cd "$_SELF_DIR/../.." && pwd)          # <openpi>/examples/robocasa -> <openpi>
OPENPI_REPO=${OPENPI_REPO:-$_OPENPI_REPO}
REPO_ROOT=${REPO_ROOT:-$(dirname "$OPENPI_REPO")}
if [[ -z "${DATA_DIR:-}" ]]; then
  for _c in "$REPO_ROOT/data" "$(dirname "$OPENPI_REPO")/data" "$HOME/data"; do
    [[ -d "$_c" ]] && { DATA_DIR=$_c; break; }
  done
  DATA_DIR=${DATA_DIR:-$REPO_ROOT/data}
fi
# The rollout client imports sys2.data.video_policy from here (frame sampling must match training
# exactly); it falls back to a vendored copy and warns if this path is wrong. EXPORTED below.
SYS2_REPO=${SYS2_REPO:-$REPO_ROOT/sys2_train_eval}
SYS1_CKPT_DIR=${SYS1_CKPT_DIR:-$DATA_DIR/sys1_ckpts}
CKPT_DIR=${CKPT_DIR:-$DATA_DIR/sys2_ckpts}
SYS1_RESULTS_DIR=${SYS1_RESULTS_DIR:-$DATA_DIR/sys1_eval_results}
# Export so the rollout clients and the worklist generator resolve the SAME roots as this script,
# rather than each re-deriving them (or, for SYS2_REPO, silently falling back to a vendored copy of
# the frame-sampling formula). A shell variable alone is invisible to the child processes.
export OPENPI_REPO REPO_ROOT DATA_DIR SYS2_REPO SYS1_CKPT_DIR SYS1_RESULTS_DIR
cd "$OPENPI_REPO"

METHOD=${METHOD:-progreg}                  # progreg | progact  (which System1 head)
EPISODES=${EPISODES:-0}                    # per-task episode spec, e.g. "0", "0-9", "0,5,10"
TASKS=${TASKS:-}                           # explicit comma list; overrides TASK_SET
USE_EVAL_SET=${USE_EVAL_SET:-0}            # 1 = use the 1000-episode manifest inlined in
                                           # combined_eval.TARGET_EVAL_EPISODES
TASK_SET=${TASK_SET:-all}                  # all (the 50-task benchmark) | atomic_seen |
                                           # composite_seen | composite_unseen
NGPU=${NGPU:-8}
GPUS=${GPUS:-$(seq -s, 0 $((NGPU-1)))}
# One stack normally packs S1+S2+sim onto a single GPU. Set S1_GPUS/S2_GPUS to split them across
# two devices instead -- e.g. S1_GPUS=0 S2_GPUS=1 gives one stack whose policy server owns GPU0 and
# whose vLLM owns GPU1, so neither has to fit in a fraction of one card. The two lists are matched
# element-wise and must be the same length; that length (not NGPU) is the number of stacks.
S1_GPUS=${S1_GPUS:-$GPUS}
S2_GPUS=${S2_GPUS:-$GPUS}
S1_BASE=${S1_BASE:-8060}
S2_BASE=${S2_BASE:-8100}
XLA_FRAC=${XLA_FRAC:-0.32}                 # System1 JAX share
GPU_FRAC=${GPU_FRAC:-0.42}                 # System2 vLLM share
MAX_TURNS=${MAX_TURNS:-20}
# Hard ceiling on ONE subgoal segment (budget = min(this, est_length*horizon_mult)).
# Back to 400: the 800 round existed only because long WAIT subgoals were being truncated, and the
# wait rule in sys2_rules now returns force_steps, which bypasses this cap entirely. Measured on the
# 800 run, only 0.8% of segments had a budget above 400 and 28 actually ran past it. NOTE a run at 400
# is comparable with the 1000-episode no-rules baseline and NOT with the -estbump sweep (which used 800).
MAX_STEPS_CAP=${MAX_STEPS_CAP:-400}
DATA_ROOT=${DATA_ROOT:-${ROBOCASA_LEROBOT_ROOT:-$DATA_DIR/robocasa_dataset}/v1.0/target}
S2_CKPT=${S2_CKPT:-$CKPT_DIR/system2-full-0804-qwen35-4b-gb192-full-vitfull-lr1e5-vitlr2e6-alignerlr1e5-zero2-2n-ep3/checkpoint-11416}
SKIP_SERVERS=${SKIP_SERVERS:-0}            # 1 = reuse servers already listening
RESUME=${RESUME:-0}                        # 1 = skip episodes already finished (restartable)
# RUN_LABEL overrides the derived <method> results dir name. Use it to write a REPLICATION run
# beside the real one (same checkpoints, separate dir) instead of appending into it.
RUN_LABEL=${RUN_LABEL:-}
# UNITS_FILE supplies an explicit "<lerobot-dir> <episode>" manifest, one line per episode, and
# bypasses worklist generation entirely -- for re-running a hand-picked set of episodes.
UNITS_FILE=${UNITS_FILE:-}
# TASK_RULES=1 applies the hardcoded per-task System2 revisions in sys2_rules.py. Off by default:
# the baseline path must stay identical. Every override is recorded in the results.
TASK_RULES=${TASK_RULES:-0}
# Which rollout client each stack runs. The default is the cold-plan loop; combine_memory_eval.py is
# the same loop with the narrate->recipe->warm-plan memory pass in front of it (it derives its own
# "<...>-memory" results dir, so a memory sweep never mixes into the cold run's numbers).
# EVAL_ARGS passes variant-specific flags through, e.g. EVAL_ARGS="--memory-episode 200".
EVAL_SCRIPT=${EVAL_SCRIPT:-combined_eval.py}
EVAL_ARGS=${EVAL_ARGS:-}
[[ -f "$OPENPI_REPO/examples/robocasa/$EVAL_SCRIPT" ]] \
  || { echo "FATAL: no such eval script examples/robocasa/$EVAL_SCRIPT" >&2; exit 1; }
ROBOCASA_PY=${ROBOCASA_PY:-/home/ec2-user/micromamba/envs/robocasa/bin/python}   # instance-local venv
OPENPI_PY=${OPENPI_PY:-/home/ec2-user/venvs/openpi_venv/bin/python}                  # serves System1

# STEP selects which checkpoint of a run to serve (210000 / 240000 / 269999).
STEP=${STEP:-269999}
case "$METHOD" in
  progreg)          S1_CKPT=$SYS1_CKPT_DIR/f0717-270k-bs512-progreg_granfine_verbsimp_noexec/$STEP ;;
  progact)          S1_CKPT=$SYS1_CKPT_DIR/f0717-270k-bs512-progact_granfine_verbsimp_noexec/$STEP ;;
  # The noanchor runs were downloaded from s3 as "...-2node-progreg-noexec-noanchor-noanchorstate"
  # but were normalised to the underscore form of their siblings when data/ moved to the shared
  # mount; these are the on-disk names.
  progreg-noanchor) S1_CKPT=$SYS1_CKPT_DIR/f0717-270k-bs512-progreg_granfine_verbsimp_noexec_noanchor_noanchorstate/$STEP ;;
  progact-noanchor) S1_CKPT=$SYS1_CKPT_DIR/f0717-270k-bs512-progact_granfine_verbsimp_noexec_noanchor_noanchorstate/$STEP ;;
  *) echo "METHOD must be progreg|progact|progreg-noanchor|progact-noanchor" >&2; exit 1 ;;
esac
[[ -d "$S1_CKPT" ]] || { echo "FATAL: missing S1 ckpt $S1_CKPT" >&2; exit 1; }
[[ -d "$S2_CKPT" ]] || { echo "FATAL: missing S2 ckpt $S2_CKPT" >&2; exit 1; }

# STEP is in the log dir too: it holds the per-run worklist/units/shard files, so two steps of the
# same method running at once would otherwise clobber each other's work list. Scratch only --
# --resume reads the results dir, so renaming this strands no state.
# The variant is in the name for the same reason: a memory sweep and a cold sweep of the same
# method+step are different runs and must not share shard files (or each other's client logs).
# The default (cold) client keeps the historic unsuffixed path so existing logs stay put.
if [[ "$EVAL_SCRIPT" == "combined_eval.py" ]]; then _VARIANT=""; else
  _VARIANT=$(basename "$EVAL_SCRIPT" .py); _VARIANT=${_VARIANT#combine_}; _VARIANT=${_VARIANT%_eval}
fi
# RUN_LABEL is part of the log dir, not just METHOD/STEP. Two runs of the SAME checkpoint with
# different labels (e.g. a rules-on run and its rules-off control) otherwise share this directory
# and clobber each other's worklist/units/shard files WHILE the first run's clients are still
# reading them -- observed live: a 68-episode run had its shards overwritten by a concurrent
# 20-episode run and silently stopped at 20, with per-stack counts (4/3, 5/3) exceeding the shard
# files they came from.
LOG=_evallogs/fleet_${METHOD}_$STEP${_VARIANT:+_$_VARIANT}${RUN_LABEL:+_$(echo "$RUN_LABEL" | tr -c 'A-Za-z0-9._-' '_')}
mkdir -p "$LOG"
IFS=',' read -ra S1LIST <<< "$S1_GPUS"
IFS=',' read -ra S2LIST <<< "$S2_GPUS"
[[ ${#S1LIST[@]} -eq ${#S2LIST[@]} ]] || {
  echo "FATAL: S1_GPUS ($S1_GPUS) and S2_GPUS ($S2_GPUS) must list the same number of GPUs" >&2
  exit 1; }
# The rollout client renders MuJoCo/EGL on the same GPU as its policy server: observations go
# straight to S1 every step, whereas S2 is consulted only once per subgoal.
GPULIST=("${S1LIST[@]}")
NG=${#GPULIST[@]}
# Ports are keyed by stack index, not by GPU id -- with a split, two stacks can share an S1 GPU (or
# an S2 GPU) and GPU-derived ports would collide.
s1_port() { echo $((S1_BASE+$1)); }
s2_port() { echo $((S2_BASE+$1)); }

# ---- work list --------------------------------------------------------------------------------
# Either an explicit (task, episode) manifest (EPISODE_JSON) or TASK_SET x EPISODES.
WORK="$LOG/worklist.txt"
: > "$WORK"
if [[ -n "$UNITS_FILE" ]]; then
  # Explicit manifest: every line is already one (task, episode) unit, so generation and the
  # episode-spec expansion below are both skipped. Validated the same way as a generated list.
  [[ -s "$UNITS_FILE" ]] || { echo "FATAL: empty/missing UNITS_FILE $UNITS_FILE" >&2; exit 1; }
  grep -E '^/.*/lerobot [0-9]+$' "$UNITS_FILE" > "$WORK" || true
  [[ -s "$WORK" ]] || { echo "FATAL: no valid '<lerobot-dir> <ep>' lines in $UNITS_FILE" >&2; exit 1; }
  echo "[fleet] explicit manifest: $(wc -l < "$WORK") units from $UNITS_FILE"
fi
if [[ -z "$UNITS_FILE" ]]; then
# robocasa env: the registry import needs robosuite, absent from system python3.
OPENPI_REPO="$OPENPI_REPO" "$ROBOCASA_PY" - "$DATA_ROOT" "$TASK_SET" "$TASKS" "$EPISODES" "$USE_EVAL_SET" \
  >> "$WORK" 2>"$LOG/worklist.err" <<'PY'
import sys, glob, os
root, task_set, tasks_csv, eps, use_eval_set = sys.argv[1:6]
sys.path.insert(0, os.path.join(os.environ["OPENPI_REPO"], "examples", "robocasa"))
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
fi   # end of generated-worklist branch
NTASK=$(wc -l < "$WORK")
echo "[fleet] method=$METHOD  src=$([[ $USE_EVAL_SET == 1 ]] && echo TARGET_EVAL_EPISODES || echo taskset:$TASK_SET)  lines=$NTASK  gpus=${GPULIST[*]}"
[[ -s "$LOG/worklist.err" ]] && grep -i "MISSING" "$LOG/worklist.err" || true
[[ "$NTASK" -gt 0 ]] || { echo "FATAL: empty work list" >&2; exit 1; }

# ---- servers ----------------------------------------------------------------------------------
if [[ "$SKIP_SERVERS" != "1" ]]; then
  for i in "${!S1LIST[@]}"; do
    g1=${S1LIST[$i]}; g2=${S2LIST[$i]}
    XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRAC CUDA_VISIBLE_DEVICES=$g1 \
      nohup "$OPENPI_PY" scripts/serve_policy.py --port "$(s1_port "$i")" \
        policy:checkpoint --policy.config=auto --policy.dir "$S1_CKPT" \
        > "$LOG/s1-stack$i.log" 2>&1 &
    MODEL_DIR="$S2_CKPT" GPU=$g2 GPU_FRAC=$GPU_FRAC PORT="$(s2_port "$i")" \
      NAME=sys2-vllm-fleet-$i MEDIA_DIR=/tmp/sys2_media \
      nohup bash "$SYS2_REPO/scripts/serve_system2_vllm.sh" \
        > "$LOG/s2-stack$i.log" 2>&1 &
    echo "[fleet] stack$i: S1 gpu$g1 :$(s1_port "$i")  S2 gpu$g2 :$(s2_port "$i")"
  done

  # READY_TRIES x 10s per stack. A cold-cache vLLM start is far slower than the warm ~4 min: on a
  # fresh instance the container spends ~9.5 min merely importing its own site-packages (a 40 GB
  # image of small files, latency-bound) before vLLM prints anything, then ~2 min of config, 48 s
  # of torch.compile and ~4 min of KV-cache + CUDA-graph capture -- measured 19m20s end to end.
  # The old 15 min budget expired 4 min short of that, so default to 40 min; warm starts still
  # break out of the loop as soon as they are ready and cost nothing.
  READY_TRIES=${READY_TRIES:-240}
  echo "[fleet] waiting for all servers (up to $((READY_TRIES * 10 / 60)) min; a cold vLLM start is ~20 min)..."
  failed=()
  for i in "${!S1LIST[@]}"; do
    for _ in $(seq 1 "$READY_TRIES"); do
      s1=$(ss -lnt 2>/dev/null | grep -c ":$(s1_port "$i") " || true)
      s2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$(s2_port "$i")/v1/models" 2>/dev/null || echo 000)
      [[ "$s1" == "1" && "$s2" == "200" ]] && break
      sleep 10
    done
    [[ "${s1:-0}" == "1" && "${s2:-000}" == "200" ]] \
      && echo "[fleet] stack$i READY" \
      || { echo "[fleet] stack$i FAILED (S1=$s1 S2=$s2); see $LOG/{s1,s2}-stack$i.log" >&2
           failed+=("$i"); }
  done
  # A stack whose servers never came up must NOT be given work. This used to fall through and
  # launch the clients anyway: every episode of that shard then died on "Connection refused" in
  # ~20s and was recorded as a failure, so a whole shard could be burnt while looking like a
  # legitimately bad result. Refuse to start rather than produce junk.
  if [[ ${#failed[@]} -gt 0 ]]; then
    echo "[fleet] FATAL: ${#failed[@]} stack(s) not ready: ${failed[*]}." >&2
    echo "[fleet] Not launching any clients (episodes would fail instantly against dead servers)." >&2
    echo "[fleet] Inspect $LOG/{s1,s2}-stack*.log, then release GPUs with" >&2
    echo "[fleet]   bash examples/robocasa/release_gpus.sh" >&2
    exit 1
  fi
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
  awk -v n="$NG" -v k="$i" '{ if (((NR-1) % n) == k) print }' "$UNITS" > "$LOG/shard-stack$i.txt"
  echo "[fleet] stack$i (sim on gpu${GPULIST[$i]}) -> $(wc -l < "$LOG/shard-stack$i.txt") episodes"
done

for i in "${!GPULIST[@]}"; do
  g=${GPULIST[$i]}
  (
    while read -r ld ep; do
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$g \
        "$ROBOCASA_PY" "examples/robocasa/$EVAL_SCRIPT" \
          --lerobot-dir "$ld" --episodes "$ep" \
          --s1-dir "$S1_CKPT" --s2-dir "$S2_CKPT" \
          --s1-port "$(s1_port "$i")" --s2-port "$(s2_port "$i")" --s2-model system2-full \
          --norm-stats "$S1_CKPT/assets/robocasa_system1/norm_stats.json" \
          --out-root "$SYS1_RESULTS_DIR/combine" --max-turns "$MAX_TURNS" \
          --max-steps-cap "$MAX_STEPS_CAP" \
          ${RUN_LABEL:+--method "$RUN_LABEL"} \
          $([[ "$TASK_RULES" == 1 ]] && echo --task-rules) \
          $([[ "$RESUME" == 1 ]] && echo --resume) \
          ${EVAL_ARGS:-} \
        || echo "[fleet] stack$i FAILED $ld ep$ep" >&2
    done < "$LOG/shard-stack$i.txt"
    echo "[fleet] stack$i SHARD COMPLETE"
  ) > "$LOG/client-stack$i.log" 2>&1 &
done

echo "[fleet] $NG clients launched. Monitor: tail -f $LOG/client-stack*.log"
wait
echo "[fleet] ALL SHARDS COMPLETE for method=$METHOD"
