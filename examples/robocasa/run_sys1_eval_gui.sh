#!/usr/bin/env bash
# Launch the System1 evaluation GUI, including /combine and /human-interactive.
# The S1 websocket and S2 vLLM servers must already be running.
#
#   bash examples/robocasa/run_sys1_eval_gui.sh \
#     --port 8092 --s1-port 8060 --s2-port 8100
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_REPO=${OPENPI_REPO:-$(cd "$SELF_DIR/../.." && pwd)}
REPO_ROOT=${REPO_ROOT:-$(dirname "$OPENPI_REPO")}
if [[ -z "${DATA_DIR:-}" ]]; then
  for candidate in "$REPO_ROOT/data" "$(dirname "$OPENPI_REPO")/data" "$HOME/data"; do
    if [[ -d "$candidate" ]]; then
      DATA_DIR=$candidate
      break
    fi
  done
  DATA_DIR=${DATA_DIR:-$REPO_ROOT/data}
fi

ROBOCASA_PY=${ROBOCASA_PY:-/home/ec2-user/micromamba/envs/robocasa/bin/python}
ROBOCASA_REPO=${ROBOCASA_REPO:-$REPO_ROOT/robocasa}
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-$OPENPI_REPO/packages/openpi-client/src}
SYS2_REPO=${SYS2_REPO:-$REPO_ROOT/sys2_train_eval}

export OPENPI_REPO REPO_ROOT DATA_DIR SYS2_REPO
export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$OPENPI_CLIENT_SRC:$ROBOCASA_REPO${PYTHONPATH:+:$PYTHONPATH}"

cd "$OPENPI_REPO"
exec "$ROBOCASA_PY" examples/robocasa/subtask_eval_gui.py "$@"
