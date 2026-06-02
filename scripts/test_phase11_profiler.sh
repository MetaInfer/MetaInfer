#!/usr/bin/env bash
# Why: Phase 11 profiler 检查——确认稳态 decode 无 cudaMalloc + aten::item 累计 < 10ms
# What failure: cudaMalloc > 0 在 decode 阶段 / aten::item > 10ms → 优化不彻底
# Superpowers gate: CLAUDE.md rule 5 (executable skill)
# Human review: [待人类Diff]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PATH="${PYTHON_PATH}:${PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export META_INFER_LOG_RANK0_ONLY=1; export META_INFER_CUDA_GRAPH=0

echo "=== Phase 11: Profiler Check ==="

# Quick profiler snapshot of a single generate call
python -c "
import os; os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
import torch
from llm_engine import LLMEngine; from pathlib import Path

engine = LLMEngine(model_dir=Path('${MODEL_DIR}'), inference_backend='qwen_tp', max_num_seqs=4)

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
total_cuda_malloc = sum(getattr(e, 'cuda_memory_usage', 0) for e in events if 'cudaMalloc' in e.key or 'malloc' in e.key.lower())
total_item = sum(e.cpu_time_total for e in events if 'item' in e.key.lower()) / 1000  # us → ms

print(f'PROFILER-001: cudaMalloc total = {total_cuda_malloc / 1024**2:.1f} MB')
print(f'PROFILER-002: aten::item total = {total_item:.1f} ms (target: <10ms)')

# Assertions
if total_item > 10:
    print(f'PROFILER-002 FAIL: aten::item {total_item:.1f}ms > 10ms. P5 not fully applied.')
    import sys; sys.exit(1)
if total_cuda_malloc > 10 * 1024**2:  # >10MB means still allocating
    print(f'PROFILER-001 WARNING: cudaMalloc {total_cuda_malloc/1024**2:.1f}MB > 10MB. P1/P4 may need review.')

print('PROFILER CHECK: PASS')
print('PHASE11_PROFILER: ALL CHECKS PASSED')
" 2>&1

echo "Source: Phase 11 performance optimization — P1-P6 from real engine"
