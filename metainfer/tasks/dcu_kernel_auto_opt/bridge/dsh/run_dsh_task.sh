#!/bin/bash
# Launch a dcu-kernel-auto-opt task with DSH agents instead of Claude Code.
#
# Usage:
#   TENCENT_API_KEY=<key> bash run_dsh_task.sh <requirements.json> <task_id>
#
# The requirements.json must contain task_type=dcu-kernel-auto-opt and the
# standard answers; agent_framework should be "dsh" (see the WebUI new-task
# form). This script only sets the claude_bin override — the orchestrator CLI
# resolves the DSH wrapper automatically when agent_framework=dsh, so this
# script is a convenience for headless runs.
#
# Optional env:
#   DSH_AGENT_MAX_TOKENS   per-request output cap (default 65536)
#   DSH_AGENT_MODEL        model override (default deepseek/deepseek-v4-flash-0731)
#   METAINFER_GPU_IDS      restrict GPUs, e.g. 4,5,6,7

set -euo pipefail

REQ="${1:?requirements.json path required}"
TASK_ID="${2:?task id required}"

# Resolve the MetaInfer root from this script's location:
#   <root>/metainfer/tasks/dcu_kernel_auto_opt/bridge/dsh/run_dsh_task.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
META_ROOT="$(cd "$HERE/../../../../.." && pwd)"
DSH_DIR="$HERE"
NODE_ROOT="${METAINFER_NODE_ROOT:-$META_ROOT/nodes/worker29}"

export METAINFER_CLAUDE_BIN="${METAINFER_CLAUDE_BIN:-$DSH_DIR/dsh_agent.py}"
export DSH_AGENT_MAX_TOKENS="${DSH_AGENT_MAX_TOKENS:-65536}"

STATE_DIR="$NODE_ROOT/.metainfer/tasks/$TASK_ID"
WORKSPACE_DIR="$NODE_ROOT/workspaces/$TASK_ID"
mkdir -p "$STATE_DIR" "$WORKSPACE_DIR"

echo "==> DSH agent: $METAINFER_CLAUDE_BIN"
echo "==> state:     $STATE_DIR"
echo "==> workspace: $WORKSPACE_DIR"

cd "$META_ROOT"
exec python3 -m metainfer.tasks.dcu_kernel_auto_opt.orchestrator.cli run \
  "$REQ" \
  --state-dir "$STATE_DIR" \
  --workspace-dir "$WORKSPACE_DIR"
