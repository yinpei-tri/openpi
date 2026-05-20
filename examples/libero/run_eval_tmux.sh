#!/usr/bin/env bash
# Launch a LIBERO eval in a tmux session: pane 0 runs the openpi policy server,
# pane 1 runs examples/libero/main.py once the server is reachable.
#
# Defaults to evaluating the released JAX pi05_libero checkpoint on libero_spatial.
# Override anything via env vars or flags; run with `-h` for usage.

set -euo pipefail

# ---- defaults ----------------------------------------------------------------
RUN_NAME="${RUN_NAME:-pi05_libero_jax_release}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
# Session defaults to libero_<run-name>_<task-suite> so multiple suites for the
# same run-name don't collide. Override with --session if you want a custom id.
SESSION="${SESSION:-}"
REPLACE_SESSION="${REPLACE_SESSION:-0}"
CONFIG="${CONFIG:-pi05_libero}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero}"
SERVER_GPU="${SERVER_GPU:-0}"           # CUDA device id used by the policy server
CLIENT_GPU="${CLIENT_GPU:-$SERVER_GPU}" # MuJoCo can also use a GPU; usually fine to share
PORT="${PORT:-8010}"
HOST="${HOST:-127.0.0.1}"
RESULTS_ROOT="${RESULTS_ROOT:-data/libero/results}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
XLA_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

usage() {
    cat <<EOF
Usage: $0 [options]

Spawns a tmux session with two panes:
  pane 0: serve_policy.py (JAX or PyTorch, auto-detected by checkpoint contents)
  pane 1: examples/libero/main.py against that server

Defaults reproduce the released JAX pi05_libero on libero_spatial.

Options (override with env vars of the same name):
  --run-name        NAME      results dir suffix              [$RUN_NAME]
  --session         NAME      tmux session name               [libero_<run>_<suite>]
  --replace                   kill existing session with the same name first
  --task-suite      NAME      libero_{spatial,object,goal,10} [$TASK_SUITE]
  --config          NAME      training config name            [$CONFIG]
  --checkpoint-dir  PATH      path passed to --policy.dir     [$CHECKPOINT_DIR]
  --server-gpu      ID        CUDA_VISIBLE_DEVICES for server [$SERVER_GPU]
  --client-gpu      ID        CUDA_VISIBLE_DEVICES for client [$CLIENT_GPU]
  --port            N         server port                     [$PORT]
  --host            HOST      client connects to this host    [$HOST]
  --results-root    PATH      eval output root                [$RESULTS_ROOT]
  --num-trials      N         rollouts per task               [$NUM_TRIALS]
  --seed            N         eval seed                       [$SEED]
  --mujoco-gl       BACKEND   egl|glx|osmesa                  [$MUJOCO_GL]
  -h, --help                  show this help
EOF
}

# ---- arg parsing -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)       RUN_NAME="$2"; shift 2 ;;
        --session)        SESSION="$2"; shift 2 ;;
        --task-suite)     TASK_SUITE="$2"; shift 2 ;;
        --config)         CONFIG="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --server-gpu)     SERVER_GPU="$2"; shift 2 ;;
        --client-gpu)     CLIENT_GPU="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --host)           HOST="$2"; shift 2 ;;
        --results-root)   RESULTS_ROOT="$2"; shift 2 ;;
        --num-trials)     NUM_TRIALS="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --mujoco-gl)      MUJOCO_GL="$2"; shift 2 ;;
        --replace)        REPLACE_SESSION=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

# Default session name includes both run-name and suite to avoid collisions.
SESSION="${SESSION:-libero_${RUN_NAME}_${TASK_SUITE}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBERO_VENV="$REPO_ROOT/examples/libero/.venv"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint dir does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi
if [[ ! -d "$LIBERO_VENV" ]]; then
    echo "LIBERO client venv missing at $LIBERO_VENV." >&2
    echo "Set it up first per examples/libero/README.md." >&2
    exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed." >&2
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ "$REPLACE_SESSION" == "1" ]]; then
        echo "Killing existing tmux session '$SESSION' before starting (--replace)."
        tmux kill-session -t "$SESSION"
    else
        echo "tmux session '$SESSION' already exists." >&2
        echo "Re-run with --replace to kill it, or pass --session NAME to pick a different id." >&2
        echo "Manual kill:  tmux kill-session -t $SESSION" >&2
        exit 1
    fi
fi

SERVER_CMD=(
    env
    "CUDA_VISIBLE_DEVICES=$SERVER_GPU"
    "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION"
    uv run scripts/serve_policy.py
    --port "$PORT"
    policy:checkpoint
    --policy.config "$CONFIG"
    --policy.dir "$CHECKPOINT_DIR"
)

CLIENT_CMD=(
    env
    "CUDA_VISIBLE_DEVICES=$CLIENT_GPU"
    "MUJOCO_GL=$MUJOCO_GL"
    "PYTHONPATH=$REPO_ROOT/third_party/libero:${PYTHONPATH:-}"
    "$LIBERO_VENV/bin/python" examples/libero/main.py
    --args.host "$HOST"
    --args.port "$PORT"
    --args.task-suite-name "$TASK_SUITE"
    --args.num-trials-per-task "$NUM_TRIALS"
    --args.seed "$SEED"
    --args.run-name "$RUN_NAME"
    --args.results-root "$RESULTS_ROOT"
)

quote_cmd() {
    local out=""
    for tok in "$@"; do
        out+=" $(printf '%q' "$tok")"
    done
    printf '%s' "${out# }"
}

SERVER_CMD_STR="$(quote_cmd "${SERVER_CMD[@]}")"
CLIENT_CMD_STR="$(quote_cmd "${CLIENT_CMD[@]}")"

# Keep panes alive after the process exits so logs stay visible.
SERVER_PANE_CMD="cd $(printf '%q' "$REPO_ROOT") && echo '[server] gpu=$SERVER_GPU port=$PORT' && $SERVER_CMD_STR; echo; echo '[server exited, press enter to close]'; read"
CLIENT_PANE_CMD="cd $(printf '%q' "$REPO_ROOT") && echo '[client] gpu=$CLIENT_GPU host=$HOST:$PORT run=$RUN_NAME suite=$TASK_SUITE' && $CLIENT_CMD_STR; echo; echo '[client exited, press enter to close]'; read"

tmux new-session -d -s "$SESSION" -n eval "bash -lc $(printf '%q' "$SERVER_PANE_CMD")"
tmux split-window -h -t "$SESSION:eval" "bash -lc $(printf '%q' "$CLIENT_PANE_CMD")"
tmux select-pane -t "$SESSION:eval.0"
tmux set-option -t "$SESSION" mouse on >/dev/null

echo "Started tmux session '$SESSION'."
echo "  pane 0 (left):  policy server   [GPU $SERVER_GPU]"
echo "  pane 1 (right): libero client   [GPU $CLIENT_GPU]"
echo
echo "Attach:           tmux attach -t $SESSION"
echo "Kill everything:  tmux kill-session -t $SESSION"
echo "Results:          $RESULTS_ROOT/$RUN_NAME/$TASK_SUITE/results.json"
