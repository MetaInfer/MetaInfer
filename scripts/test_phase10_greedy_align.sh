#!/usr/bin/env bash
# Why: E2E 验收最低底线——temperature=0.0 贪婪解码必须字字对齐基线。
#   Trace (2026-05-27): output='（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
#   发现于 P0 增量 KV Cache 解码验证阶段。
# What failure: 输出与基线不一致 → exit 1 报错 GREEDY-ALIGN-001。
# Superpowers gate: CLAUDE.md rule 5 (Skill = executable code)
# Human review: [待人类Diff]
# T11 source: physical_trace_tp4_rank0.json [runtime] output + greedy_match=True
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export META_INFER_LOG_RANK0_ONLY=1
export META_INFER_CUDA_GRAPH=0

EXPECTED="（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑"
PROMPT="苏州园林的特点是"
MAX_TOKENS=24
TP_SIZE="${TP_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRACE_SRC="Source: physical_trace_tp4_rank0.json [runtime] greedy_match=True"

echo "=== Phase 10: Greedy Decode Alignment Test ==="
echo "TP_SIZE=${TP_SIZE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Expected: ${EXPECTED}"

# Step 1: Single GPU quick test (fast path)
echo "[GREEDY-ALIGN-001] Single GPU test..."
OUTPUT=$(python -c "
import os; os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
from llm_engine import LLMEngine; from pathlib import Path
engine = LLMEngine(model_dir=Path('${MODEL_DIR}'), inference_backend='qwen_tp', max_num_seqs=4)
out = engine.generate('${PROMPT}', max_new_tokens=${MAX_TOKENS}, temperature=0.0)
import sys; sys.stdout.write('RESULT:' + out + '\n')
" 2>&1 | grep '^RESULT:' | sed 's/^RESULT://')

echo "Output: ${OUTPUT}"
echo "Expected: ${EXPECTED}"

if [ "${OUTPUT}" = "${EXPECTED}" ]; then
    echo "[GREEDY-ALIGN-001] PASS: single GPU greedy decode matches baseline exactly"
else
    echo "[GREEDY-ALIGN-001] FAIL: output differs from baseline"
    echo "  Got:      ${OUTPUT}"
    echo "  Expected: ${EXPECTED}"
    echo "  ${TRACE_SRC}"
    exit 1
fi

# Step 2: TP=4 torchrun test (optional, skip if only 1 GPU)
if [ "${TP_SIZE}" -gt 1 ] && command -v torchrun &>/dev/null; then
    echo "[GREEDY-ALIGN-002] TP=4 torchrun test..."
    torchrun --nproc_per_node="${TP_SIZE}" --master_port=$((29500 + RANDOM % 1000)) python -c "
import os, sys; os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
import torch.distributed as dist
from llm_engine import LLMEngine; from pathlib import Path
engine = LLMEngine(model_dir=Path('${MODEL_DIR}'), inference_backend='qwen_tp', max_num_seqs=4)
out = engine.generate('${PROMPT}', max_new_tokens=${MAX_TOKENS}, temperature=0.0)
if int(os.environ.get('RANK','0')) == 0:
    assert out == '${EXPECTED}', f\"GREEDY-ALIGN-002: TP=4 output differs. Got={out!r}\"
    sys.stdout.write('GREEDY-ALIGN-002: TP=4 greedy decode PASS\n')
" 2>/dev/null
    echo "[GREEDY-ALIGN-002] PASS: TP=4 greedy decode matches baseline"
fi

echo "PHASE10_GREEDY_ALIGN: ALL TESTS PASSED"
echo "${TRACE_SRC}"
