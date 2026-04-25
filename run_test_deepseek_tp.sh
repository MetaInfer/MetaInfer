#!/usr/bin/env bash
# 在 meta-infer 根目录下生成可观察的完整日志（勿写 /tmp）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
LOG="$ROOT/torchrun_test_deepseek_tp.log"
echo "Logging to: $LOG"
exec torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -v -s --tb=short 2>&1 | tee "$LOG"
