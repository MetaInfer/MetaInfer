#!/usr/bin/env bash
# DeepSeek TP 正确性用例（pytest），与压测的「5 条默认 prompt」一致见 tests/test_deepseek_tp_real.py。
# 与 bench 的对应关系: scripts/run_myengine_benchmark.sh 在 DATASET=zh-default 时使用相同 5 句循环。
# 在 meta-infer 根目录下生成完整日志
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
TP_SIZE="${TP_SIZE:-4}"
TEST_MODEL_DIR="${TEST_MODEL_DIR:-/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat}"

build_cuda_visible_devices() {
  local n="$1"
  local out=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ -z "$out" ]]; then
      out="$i"
    else
      out="${out},${i}"
    fi
  done
  echo "$out"
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(build_cuda_visible_devices "$TP_SIZE")"
fi
export TEST_MODEL_DIR
LOG="$ROOT/torchrun_test_deepseek_tp.log"
echo "Logging to: $LOG"
echo "TP_SIZE=$TP_SIZE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TEST_MODEL_DIR=$TEST_MODEL_DIR"
exec torchrun --nproc_per_node="$TP_SIZE" -m pytest tests/test_deepseek_tp_real.py -v -s --tb=short 2>&1 | tee "$LOG"
