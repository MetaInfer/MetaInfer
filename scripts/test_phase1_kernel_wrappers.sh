#!/usr/bin/env bash
# Why: 防止 conda 环境缺少 vllm/flash_attn 依赖，导致 kernel import 失败。
#   Trace: flash_attn_varlen_func=available, flash_attn_with_kvcache=available, vllm._custom_ops=available
# What failure: import 失败 → exit 1 "KERNEL-SH-001"
# Superpowers gate: CLAUDE.md rule 5 (executable skill)
# Trace Source: physical_trace_tp4_rank0.json [env] dependencies
# Human review: [待人类Diff]
set -euo pipefail
echo "=== Phase 1: Kernel Wrapper Environment Check ==="

python -c "
import sys; errs=[]
try:
    from flash_attn import flash_attn_varlen_func; print('[KERNEL-SH-001] flash_attn_varlen_func OK')
except ImportError as e: errs.append(f'flash_attn_varlen_func: {e}')
try:
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache; print('[KERNEL-SH-001] flash_attn_with_kvcache OK')
except ImportError as e: errs.append(f'flash_attn_with_kvcache: {e}')
try:
    import vllm; import vllm._C; print('[KERNEL-SH-001] vllm._C OK (triggers torch.ops._C.silu_and_mul)')
except ImportError as e: errs.append(f'vllm._C: {e}')
try:
    from vllm import _custom_ops as ops; print('[KERNEL-SH-001] vllm._custom_ops OK')
except ImportError as e: errs.append(f'vllm._custom_ops: {e}')
if errs:
    for e in errs: print(f'  FAIL: {e}')
    print(f'KERNEL-SH-001: {len(errs)} dependency(s) missing. Source: physical_trace_tp4_rank0.json [env]')
    sys.exit(1)
print('PHASE1_KERNEL_WRAPPERS_SH: ALL DEPENDENCIES AVAILABLE')
" 2>&1
echo "Source: physical_trace_tp4_rank0.json [env] all dependencies available"
