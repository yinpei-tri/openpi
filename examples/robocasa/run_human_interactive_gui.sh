#!/usr/bin/env bash
# Backward-compatible name for run_sys1_eval_gui.sh.
#
# System1 and System2 are expected to be served already. Override their addresses/checkpoint labels
# with HITL_S1_* / HITL_S2_* or pass the corresponding --hitl-* CLI flags after this script.
#
#   bash examples/robocasa/run_human_interactive_gui.sh --port 8092
#   HITL_S1_PORT=8060 HITL_S2_PORT=8100 \
#     bash examples/robocasa/run_human_interactive_gui.sh --port 8092
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SELF_DIR/run_sys1_eval_gui.sh" "$@"
