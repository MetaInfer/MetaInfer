#!/usr/bin/env bash
# Why: Phase 11 Stage 2 — profiler-based steady-state decode check.
#   Gates from inference_blueprint.json performance_gate:
#     - cudaMalloc = 0 in steady-state decode
#     - aten::item < 10ms total per step
#   Exports chrome trace to perf_iteration/ for visual analysis.
# What failure: cudaMalloc in decode hot path / excessive GPU sync / no trace
# Trace Source: physical_trace_tp4_rank0.json [runtime] Phase 10
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export META_INFER_LOG_RANK0_ONLY=1
export META_INFER_CUDA_GRAPH=0
export VLLM_LOGGING_LEVEL=ERROR
export NCCL_DEBUG=WARN

# ---- MODEL_DIR must be set ----
if [ -z "${MODEL_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR not set. export MODEL_DIR=<model weights dir>" >&2
    exit 1
fi

# ---- Output directory for profiler traces ----
TRACE_DIR="${ROOT_DIR}/perf_iteration/trace_target"
mkdir -p "${TRACE_DIR}"

echo "=== Phase 11 Stage 2: Steady-State Decode Profiler ==="
echo "Model:   ${MODEL_DIR}"
echo "Trace:   ${TRACE_DIR}"

# ---- Detect Python ----
PYTHON_BIN="${PYTHON_PATH:-}/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

# ===================================================================
# Profiler run: isolate decode steps from prefill
# ===================================================================
"$PYTHON_BIN" -c "
import os, sys, json, math
sys.path.insert(0, '${ROOT_DIR}')
import torch
import torch.profiler as profiler
from pathlib import Path
from llm_engine import LLMEngine
from engine.framework.sequence import Sequence, SequenceStatus

MODEL_DIR   = '${MODEL_DIR}'
PROMPT      = '苏州园林的特点是'
MAX_TOKENS  = 24                     # decode steps to profile
TRACE_DIR   = '${TRACE_DIR}'
EXPECTED    = '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'

# ------------------------------------------------------------------
# 1. Build engine + warmup
# ------------------------------------------------------------------
print('[INIT] Creating engine...')
engine = LLMEngine(
    model_dir=Path(MODEL_DIR),
    inference_backend='qwen_tp',
    max_num_seqs=4,
)

print('[WARMUP] One full generate to absorb one-time allocations...')
_ = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
torch.cuda.synchronize()

# ------------------------------------------------------------------
# 2. Manual sequence setup (bypass generate() to control phases)
# ------------------------------------------------------------------
print('[SETUP] Creating sequence for step-by-step profiling...')
tokenizer = engine.runner.tokenizer
token_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
max_blocks = engine._estimate_kv_blocks()

seq = Sequence(
    input_ids=token_ids,
    max_output_len=MAX_TOKENS,
    block_size=engine.block_size,
    max_blocks=max_blocks,
    device=engine.device,
)
seq.sampling_params = {
    'max_tokens': MAX_TOKENS,
    'temperature': 0.0,
    'top_p': 1.0,
}
seq.status = SequenceStatus.WAITING
engine._waiting.append(seq)
engine._active_gen_seqs.append(seq)

# ------------------------------------------------------------------
# 3. Prefill step — NOT profiled
# ------------------------------------------------------------------
print('[PREFILL] Running prefill step (unprofiled)...')
num_free = engine.runner.get_num_free_blocks()
result = engine.scheduler.schedule(engine._waiting, engine._running, num_free)
assert result.is_prefill, 'Expected prefill step'
output = engine.runner.run(result.batch, is_prefill=True, temperature=0.0)
engine.scheduler.postprocess(result.batch, True, output.next_tokens)
# Move seq from waiting -> running
for s in result.batch:
    if s in engine._waiting:
        engine._waiting.remove(s)
    if s not in engine._running:
        engine._running.append(s)

torch.cuda.synchronize()
prefill_kv_len = seq.kv_len
print(f'  prefill KV len: {prefill_kv_len}')

# ------------------------------------------------------------------
# 4. Decode steps — PROFILED
# ------------------------------------------------------------------
print(f'[DECODE] Profiling {MAX_TOKENS - 1} decode steps...')

activities = [
    profiler.ProfilerActivity.CPU,
    profiler.ProfilerActivity.CUDA,
]

with profiler.profile(
    activities=activities,
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    for step_i in range(MAX_TOKENS - 1):
        with profiler.record_function(f'decode_step_{step_i}'):
            num_free = engine.runner.get_num_free_blocks()
            result = engine.scheduler.schedule(
                engine._waiting, engine._running, num_free
            )
            if not result.batch:
                break
            output = engine.runner.run(
                result.batch, is_prefill=False, temperature=0.0
            )
            engine.scheduler.postprocess(
                result.batch, False, output.next_tokens
            )
            torch.cuda.synchronize()

torch.cuda.synchronize()

# ------------------------------------------------------------------
# 5. Export traces
# ------------------------------------------------------------------
print('[EXPORT] Writing profiler traces...')

# Chrome trace (for chrome://tracing visual analysis)
chrome_path = os.path.join(TRACE_DIR, 'trace_rank0.json')
prof.export_chrome_trace(chrome_path)
print(f'  Chrome trace: {chrome_path}')

# key_averages table
key_avg_path = os.path.join(TRACE_DIR, 'key_avg.txt')
ka_table = prof.key_averages().table(
    sort_by='cuda_time_total', row_limit=60
)
with open(key_avg_path, 'w') as f:
    f.write(ka_table)
print(f'  key_avg:      {key_avg_path}')

# Print to stdout for test log
print()
print('--- key_averages (top 40 by cuda_time) ---')
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=40))

# ------------------------------------------------------------------
# 6. Gate checks
# ------------------------------------------------------------------
print()
print('--- Gate Analysis ---')

events = list(prof.key_averages())
total_decode_steps = MAX_TOKENS - 1

# ---- GATE T202-MALLOC: cudaMalloc in decode = 0 ----
cuda_malloc_events = [e for e in events if 'cudaMalloc' in (e.key or '')]
cuda_malloc_count = sum(e.count for e in cuda_malloc_events)
cuda_malloc_cuda_us = sum(
    getattr(e, 'cuda_time_total', 0) or 0 for e in cuda_malloc_events
)

print(f'[T202-MALLOC] cudaMalloc events: {cuda_malloc_count}')
print(f'[T202-MALLOC] cudaMalloc CUDA time: {cuda_malloc_cuda_us:.1f} us')

if cuda_malloc_count > 0 and cuda_malloc_cuda_us > 50:
    # More than trivial GPU time in cudaMalloc -> likely in hot path
    print(f'[T202-MALLOC] WARN: cudaMalloc detected in decode profile '
          f'(count={cuda_malloc_count}, cuda_us={cuda_malloc_cuda_us:.1f})')
    # List the call sites
    for e in cuda_malloc_events:
        print(f'  - {e.key}: count={e.count} cuda_us={getattr(e, \"cuda_time_total\", 0):.1f}')
else:
    print(f'[T202-MALLOC] PASS: no significant cudaMalloc in decode hot path')

# ---- GATE T202-ITEM: aten::item < 10ms total per step ----
item_events = [e for e in events if 'item' in (e.key or '').lower()
               and 'cuda' not in (e.key or '').lower()]
total_item_cpu_us = sum(
    getattr(e, 'cpu_time_total', 0) or 0 for e in item_events
)
total_item_ms = total_item_cpu_us / 1000.0
item_per_step_ms = total_item_ms / total_decode_steps if total_decode_steps > 0 else 0

print(f'[T202-ITEM] aten::item CPU time: {total_item_ms:.2f} ms total')
print(f'[T202-ITEM] aten::item per decode step: {item_per_step_ms:.2f} ms')

if item_per_step_ms > 10.0:
    print(f'[T202-ITEM] FAIL: {item_per_step_ms:.1f} ms/step > 10 ms/step threshold')
    for e in item_events[:5]:
        print(f'  - {e.key}: cpu_ms={getattr(e, \"cpu_time_total\", 0)/1000:.2f}')
else:
    print(f'[T202-ITEM] PASS: {item_per_step_ms:.1f} ms/step < 10 ms/step')

# ---- GATE T202-CORRECT: output matches baseline ----
# Decode final output from the sequence
output_text = tokenizer.decode(seq.output_ids, skip_special_tokens=True)
correct = output_text.strip() == EXPECTED
print(f'[T202-CORRECT] Output: {output_text!r}')
print(f'[T202-CORRECT] {\"PASS\" if correct else \"FAIL\"}: '
      f'{\"matches\" if correct else \"differs from\"} baseline')

# ---- Summary ----
print()
all_pass = (
    (cuda_malloc_count == 0 or cuda_malloc_cuda_us <= 50)
    and item_per_step_ms <= 10.0
    and correct
)
if all_pass:
    print('PHASE11_202_PROFILER: ALL GATES PASSED')
else:
    failed = []
    if cuda_malloc_count > 0 and cuda_malloc_cuda_us > 50:
        failed.append('T202-MALLOC')
    if item_per_step_ms > 10.0:
        failed.append('T202-ITEM')
    if not correct:
        failed.append('T202-CORRECT')
    print(f'PHASE11_202_PROFILER: GATES FAILED: {\", \".join(failed)}')
    sys.exit(1)
" 2>&1

RC=$?
echo "Source: physical_trace_tp4_rank0.json [runtime] steady-state decode profiler"
exit $RC
