#!/usr/bin/env bash
# Why: E2E 验收最低底线——temperature=0.0 贪婪解码必须字字对齐基线。
#   Trace (2026-05-27): output='（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
#   发现于 P0 增量 KV Cache 解码验证阶段。
# What failure: 输出与基线不一致 → exit 1 报错 GREEDY-ALIGN-001/002。
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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRACE_SRC="Source: physical_trace_tp4_rank0.json [runtime] greedy_match=True"

# ---- MODEL_DIR 必须由用户设置 ----
if [ -z "${MODEL_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR 未设置。请在运行前 export MODEL_DIR=<模型权重目录>" >&2
    echo "示例: export MODEL_DIR=/path/to/qwen3_8b" >&2
    exit 1
fi

# ---- 临时文件管理 ----
TEMP_FILES=()
_cleanup() { rm -f "${TEMP_FILES[@]:-}"; }
trap _cleanup EXIT

# ---- Python / torchrun 路径检测 ----
PYTHON_BIN=""
TORCHRUN_BIN=""

_detect_python() {
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="$(command -v python3)"
        TORCHRUN_BIN="$(command -v torchrun 2>/dev/null || echo '')"
        return 0
    fi
    if command -v python &>/dev/null; then
        PYTHON_BIN="$(command -v python)"
        TORCHRUN_BIN="$(command -v torchrun 2>/dev/null || echo '')"
        return 0
    fi
    echo "ERROR: 找不到可用的 Python 解释器" >&2
    exit 1
}

_detect_python

# ---- GPU 数量检测 ----
_detect_gpu_count() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l
    elif command -v rocm-smi &>/dev/null; then
        # AMD ROCm / Hygon DCU: count unique HCU device indices
        rocm-smi --showmeminfo vram 2>/dev/null | grep -o 'HCU\[[0-9]*\]' | sort -u | wc -l
    else
        # Fallback: try torch.cuda
        python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0"
    fi
}
GPU_COUNT=$(_detect_gpu_count)
if [ "$GPU_COUNT" -eq 0 ] 2>/dev/null || [ -z "$GPU_COUNT" ]; then
    echo "ERROR: 没有可用的 GPU" >&2
    exit 1
fi

echo "=== Phase 10: Greedy Decode Alignment Test ==="
echo "Python:   ${PYTHON_BIN}"
echo "torchrun: ${TORCHRUN_BIN:-<not found>}"
echo "GPUs:     ${GPU_COUNT}"
echo "Model:    ${MODEL_DIR}"
echo "Expected: ${EXPECTED}"

# 共用的 Python 脚本模板函数
_write_single_gpu_script() {
    local script="$1"
    cat > "$script" << 'PYEOF'
import os, sys
sys.path.insert(0, "__ROOT_DIR__")
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
os.environ['META_INFER_CUDA_GRAPH'] = '0'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['NCCL_DEBUG'] = 'WARN'

from llm_engine import LLMEngine
from pathlib import Path

engine = LLMEngine(
    model_dir=Path("__MODEL_DIR__"),
    inference_backend='qwen_tp',
    max_num_seqs=4,
)
out = engine.generate("__PROMPT__", max_new_tokens=__MAX_TOKENS__, temperature=0.0)
sys.stdout.write(out)
PYEOF
    # Replace placeholders with actual values (avoids heredoc quoting issues)
    sed -i \
        -e "s|__ROOT_DIR__|${ROOT_DIR}|g" \
        -e "s|__MODEL_DIR__|${MODEL_DIR}|g" \
        -e "s|__PROMPT__|${PROMPT}|g" \
        -e "s|__MAX_TOKENS__|${MAX_TOKENS}|g" \
        "$script"
}

# ================================================================
# Step 1: Single GPU test
# ================================================================
echo ""
echo "[GREEDY-ALIGN-001] Single GPU test..."

SINGLE_GPU_SCRIPT=$(mktemp /tmp/greedy_align_single.XXXXXX.py)
SINGLE_STDERR=$(mktemp /tmp/greedy_align_single.XXXXXX.stderr)
TEMP_FILES+=("$SINGLE_GPU_SCRIPT" "$SINGLE_STDERR")
_write_single_gpu_script "$SINGLE_GPU_SCRIPT"

SINGLE_OUTPUT=$("$PYTHON_BIN" "$SINGLE_GPU_SCRIPT" 2>"$SINGLE_STDERR") || {
    rc=$?
    echo "[GREEDY-ALIGN-001] FAIL: Python process exited with code ${rc}" >&2
    echo "--- stderr ---" >&2
    cat "$SINGLE_STDERR" >&2
    echo "--------------" >&2
    exit 1
}

echo "Output:   ${SINGLE_OUTPUT}"
echo "Expected: ${EXPECTED}"

if [ "${SINGLE_OUTPUT}" = "${EXPECTED}" ]; then
    echo "[GREEDY-ALIGN-001] PASS: single GPU greedy decode matches baseline exactly"
else
    echo "[GREEDY-ALIGN-001] FAIL: output differs from baseline"
    echo "  Got:      ${SINGLE_OUTPUT}"
    echo "  Expected: ${EXPECTED}"
    echo "  ${TRACE_SRC}"
    exit 1
fi

# ================================================================
# Step 2: TP=4 torchrun test
# ================================================================
TP_SIZE=4

echo ""
echo "[GREEDY-ALIGN-002] TP=${TP_SIZE} torchrun test..."

if [ "$GPU_COUNT" -lt "$TP_SIZE" ]; then
    echo "[GREEDY-ALIGN-002] SKIP: need ${TP_SIZE} GPUs but only ${GPU_COUNT} available"
    echo "PHASE10_GREEDY_ALIGN: SINGLE GPU PASS (TP=${TP_SIZE} skipped — insufficient GPUs)"
    exit 0
fi

if [ -z "${TORCHRUN_BIN:-}" ] || [ ! -x "$TORCHRUN_BIN" ]; then
    echo "[GREEDY-ALIGN-002] SKIP: torchrun not found (${TORCHRUN_BIN:-N/A})"
    echo "PHASE10_GREEDY_ALIGN: SINGLE GPU PASS (TP=${TP_SIZE} skipped — no torchrun)"
    exit 0
fi

TP_SCRIPT=$(mktemp /tmp/greedy_align_tp4.XXXXXX.py)
TP_STDERR=$(mktemp /tmp/greedy_align_tp4.XXXXXX.stderr)
TEMP_FILES+=("$TP_SCRIPT" "$TP_STDERR")

# Write TP script with expected value embedded via Python code generation
# Use Python repr() to safely inject strings with special characters
"$PYTHON_BIN" - "$ROOT_DIR" "$MODEL_DIR" "$PROMPT" "$MAX_TOKENS" "$EXPECTED" "$TP_SIZE" > "$TP_SCRIPT" << 'PYEOF'
import sys
root_dir, model_dir, prompt, max_tokens, expected, tp_size = sys.argv[1:7]
script = f'''
import os, sys
sys.path.insert(0, {root_dir!r})
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
os.environ['META_INFER_CUDA_GRAPH'] = '0'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['NCCL_DEBUG'] = 'WARN'

from llm_engine import LLMEngine
from pathlib import Path

engine = LLMEngine(
    model_dir=Path({model_dir!r}),
    inference_backend='qwen_tp',
    max_num_seqs=4,
)
out = engine.generate({prompt!r}, max_new_tokens={max_tokens}, temperature=0.0)

rank = int(os.environ.get('RANK', '0'))
if rank == 0:
    expected = {expected!r}
    if out != expected:
        sys.stderr.write(f"TP={tp_size} MISMATCH: Got={{out!r}}\\nExpected={{expected!r}}\\n")
        sys.exit(1)
    sys.stdout.write(out)
'''
sys.stdout.write(script)
PYEOF

MASTER_PORT=$((29500 + RANDOM % 1000))
set +e
TP_OUTPUT=$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "$TORCHRUN_BIN" \
    --nproc_per_node="${TP_SIZE}" \
    --master_port="${MASTER_PORT}" \
    "$TP_SCRIPT" 2>"$TP_STDERR")
TP_RC=$?
set -e

if [ $TP_RC -ne 0 ]; then
    echo "[GREEDY-ALIGN-002] FAIL: torchrun exited with code ${TP_RC}" >&2
    echo "--- stderr (last 60 lines) ---" >&2
    tail -60 "$TP_STDERR" >&2
    echo "-------------------------------" >&2
    exit 1
fi

# The Python script only writes to stdout on rank 0, and only when
# the output matches expected (otherwise it exits 1). So any stdout
# is the correct output.
# Filter out RCCL/NCCL diagnostic lines, torchrun banners, and empty lines
TP_RESULT=$(echo "$TP_OUTPUT" | grep -vE 'worker|NCCL|RCCL|HIP version|ROCm version|Hostname|Librccl|^W[0-9]|^\[rank' | grep -v '^\s*$' | head -1)

if [ "${TP_RESULT}" = "${EXPECTED}" ]; then
    echo "Output:   ${TP_RESULT}"
    echo "Expected: ${EXPECTED}"
    echo "[GREEDY-ALIGN-002] PASS: TP=${TP_SIZE} greedy decode matches baseline exactly"
else
    echo "[GREEDY-ALIGN-002] FAIL: output differs from baseline"
    echo "  Got:      ${TP_RESULT}"
    echo "  Expected: ${EXPECTED}"
    echo "  ${TRACE_SRC}"
    echo "--- raw stdout ---" >&2
    echo "$TP_OUTPUT" >&2
    echo "------------------" >&2
    exit 1
fi

echo ""
echo "PHASE10_GREEDY_ALIGN: ALL TESTS PASSED"
echo "${TRACE_SRC}"
