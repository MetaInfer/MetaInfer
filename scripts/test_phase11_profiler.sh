#!/usr/bin/env bash
# Phase 11 profiler 检查——采集 profiler 数据，验证 profiler 可正常运行。
# model_dir: 由 MODEL_DIR 环境变量或命令行参数指定，未提供则交互询问。
# 不做阈值断言（aten::item / cudaMalloc 阈值为参考值，不阻塞）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PATH="${PYTHON_PATH}:${PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export META_INFER_LOG_RANK0_ONLY=1; export META_INFER_CUDA_GRAPH=0

# model_dir 获取优先级: 命令行参数 > 环境变量 > 交互输入
if [ -n "${1:-}" ]; then
    MODEL_DIR="$1"
elif [ -n "${MODEL_DIR:-}" ]; then
    :  # use env var as-is
else
    read -rp "Enter model directory path: " MODEL_DIR
fi

echo "=== Phase 11: Profiler Check ==="
echo "model_dir: ${MODEL_DIR}"

# Quick profiler snapshot of a single generate call
python -c "
import os; os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
import torch
import sys
from llm_engine import LLMEngine; from pathlib import Path

model_dir = Path('${MODEL_DIR}')
if not model_dir.exists():
    print(f'ERROR: model_dir not found: {model_dir}')
    sys.exit(1)

engine = LLMEngine(model_dir=model_dir, inference_backend='qwen_tp', max_num_seqs=4)

# Warmup
_ = engine.generate('你好', max_new_tokens=4, temperature=0.0)
torch.cuda.synchronize()

# Profiled run
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
    profile_memory=True,
) as prof:
    _ = engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)
    torch.cuda.synchronize()

events = prof.key_averages()
if len(events) == 0:
    print('PROFILER FAIL: no profiler events collected')
    sys.exit(1)

total_cuda_malloc = sum(getattr(e, 'cuda_memory_usage', 0) for e in events if 'cudaMalloc' in e.key or 'malloc' in e.key.lower())
total_item = sum(e.cpu_time_total for e in events if 'item' in e.key.lower()) / 1000  # us -> ms

print(f'PROFILER-001: cudaMalloc total = {total_cuda_malloc / 1024**2:.1f} MB (ref: 0 MB in decode)')
print(f'PROFILER-002: aten::item total = {total_item:.1f} ms (ref: <10ms)')
print(f'PROFILER-003: total profiler events collected = {len(events)}')

print('PROFILER CHECK: PASS (profiler data collected successfully)')
print('PHASE11_PROFILER: PASS')
" 2>&1

echo "Source: Phase 11 performance optimization — profiler data collection"
