#!/usr/bin/env bash
# Why: E2E 验证 LLMEngine.generate() 在单 GPU 上正确输出可读文本。
#   Trace: output='（ ） A：建筑与园林结合...' greedy_match=True
# What failure: generate() 报错/输出空/非中文→exit 1 "GEN-001"
# Superpowers gate: CLAUDE.md rule 5 (executable skill)
# Trace Source: physical_trace_tp4_rank0.json [runtime] output + greedy_match=True
# V3: Added model size check — skip single GPU for models >12GB (need TP≥2)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export META_INFER_LOG_RANK0_ONLY=1; export META_INFER_CUDA_GRAPH=0
TRACE_SRC="Source: physical_trace_tp4_rank0.json [runtime] greedy_match=True"

echo "=== Phase 9: Single GPU Generate E2E ==="

# Check model size — skip single GPU for large models (>12GB estimated)
MODEL_EST_GB=$(python -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg=json.load(f)
tc=cfg.get('text_config',cfg)
h=tc['hidden_size']; i=tc['intermediate_size']; L=tc['num_hidden_layers']; V=tc['vocab_size']
p=V*h + L*(4*h*h+3*h*i)
print(f'{p*2/1024**3:.1f}')
" 2>/dev/null)
MODEL_EST_GB_NUM=$(echo "$MODEL_EST_GB" | grep -o '[0-9.]*' | head -1)
if [ -n "${MODEL_EST_GB_NUM}" ] && python -c "exit(0 if ${MODEL_EST_GB_NUM} > 12 else 1)" 2>/dev/null; then
    echo "  Model ~${MODEL_EST_GB_NUM}GB > 12GB — skip single GPU test (requires TP≥2)"
    echo "[GEN-001] SKIP: model too large for single GPU"
    echo "PHASE9_GENERATE_SINGLE_GPU: SKIP (model requires TP>=2)"
    exit 0
fi

OUTPUT=$(python -c "
import os,sys; os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
from llm_engine import LLMEngine; from pathlib import Path
engine=LLMEngine(model_dir=Path('${MODEL_DIR}'),inference_backend='qwen_tp',max_num_seqs=4)
out=engine.generate('苏州园林的特点是',max_new_tokens=24,temperature=0.0)
sys.stdout.write('RESULT:'+out+'\n')
" 2>&1 | grep '^RESULT:' | sed 's/^RESULT://')

echo "Output: ${OUTPUT}"
if [ -n "${OUTPUT}" ] && echo "${OUTPUT}" | grep -q '园林'; then
    echo "[GEN-001] PASS: generate() returned readable Chinese text"
else
    echo "[GEN-001] FAIL: output empty or not Chinese. ${TRACE_SRC}"
    exit 1
fi
echo "PHASE9_GENERATE_SINGLE_GPU: PASS"
echo "${TRACE_SRC}"
