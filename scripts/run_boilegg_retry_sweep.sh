#!/usr/bin/env bash
set -euo pipefail

cd /shared/openpi

readonly reset_root=/home/ec2-user/data/new_tasks
readonly benchmark_json=$reset_root/BENCHMARK.json
readonly task_limits_json=$reset_root/MAX_OFFICIAL_STEPS_SUCCESS_X2.json
readonly out_root=/home/ec2-user/data/sys1_eval_results/combine
readonly start_iter=${START_ITER:-1}
readonly max_iters=${MAX_ITERS:-10}
readonly -a target_episodes=(4 10 16 19)

episode_doc() {
  local iter=$1
  local episode=$2
  printf '%s/debug-boilegg-iter%s/BoilEggs__procedural_target__episode_%06d/episode.json' \
    "$out_root" "$iter" "$episode"
}

has_prior_success() {
  local before_iter=$1
  local episode=$2
  local iter doc
  for ((iter = 1; iter < before_iter; iter++)); do
    doc=$(episode_doc "$iter" "$episode")
    if [[ -f "$doc" ]] && jq -e '.episode_success == true' "$doc" >/dev/null; then
      return 0
    fi
  done
  return 1
}

for ((iter = start_iter; iter <= max_iters; iter++)); do
  pending=()
  for episode in "${target_episodes[@]}"; do
    if ! has_prior_success "$iter" "$episode"; then
      pending+=("$episode")
    fi
  done

  if ((${#pending[@]} == 0)); then
    echo "[boilegg-sweep] all target episodes have a successful rollout before iter$iter"
    exit 0
  fi

  episode_spec=$(IFS=,; echo "${pending[*]}")
  run_label="debug-boilegg-iter$iter"
  resume=0
  [[ -d "$out_root/$run_label" ]] && resume=1
  echo "[boilegg-sweep] iter$iter pending=$episode_spec resume=$resume"

  env \
    RESET_ROOT="$reset_root" \
    BENCHMARK_JSON="$benchmark_json" \
    TASK_LIMITS_JSON="$task_limits_json" \
    RUN_LABEL="$run_label" \
    TASKS=BoilEggs \
    EPISODES="$episode_spec" \
    NGPU=4 \
    GENERAL_RULES=1 \
    TASK_RULES=1 \
    RESUME="$resume" \
    bash examples/robocasa/run_seeded_fleet.sh
done

missing=()
for episode in "${target_episodes[@]}"; do
  if ! has_prior_success "$((max_iters + 1))" "$episode"; then
    missing+=("$episode")
  fi
done

if ((${#missing[@]})); then
  echo "[boilegg-sweep] reached iter$max_iters without success for episodes: ${missing[*]}" >&2
  exit 2
fi

echo "[boilegg-sweep] all target episodes succeeded by iter$max_iters"
