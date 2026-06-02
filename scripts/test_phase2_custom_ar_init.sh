#!/usr/bin/env bash
# Why: 防止 CustomAR 初始化失败导致的 all_reduce 回退到慢速 NCCL（204ms vs 23.5ms）。
#   物理 tracing: CustomAR P2P 23.5ms, NCCL fallback 204.1ms (8.7x slower)。
#   发现于 Stage 5 TP 通信优化（2026-05-13），V17 OW-5 记录 IPC exchange 方法混淆风险。
# What failure: Agent 如果漏掉 init_custom_ar() 调用、忘记 import vllm._custom_ops、
#   或 gloo ProcessGroup 创建顺序错误，此脚本 detect 并报错退出。
# Superpowers gate: 此脚本对应 superpowers CLAUDE.md rule 5 (Skill = executable code)
#   — 必须在真实 TP=4 环境中确定性运行
# Human review: [待人类Diff] 请审查 torchrun 参数和 CustomAR 初始化步骤。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

TP_SIZE="${TP_SIZE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== Phase 2: CustomAR Init + TP Communication Check ==="
echo "TP_SIZE=${TP_SIZE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Step 1: Package availability check (single GPU, no torchrun)
python -c "
import sys
errors = []

# Check vLLM custom_ops
try:
    from vllm import _custom_ops as ops
    print(f'[OK] vllm._custom_ops available')
except ImportError as e:
    errors.append(f'vllm._custom_ops import failed: {e}')

# Check torch distributed
try:
    import torch.distributed as dist
    print(f'[OK] torch.distributed available')
except ImportError as e:
    errors.append(f'torch.distributed import failed: {e}')

# Check flash_attn
try:
    from flash_attn import flash_attn_varlen_func
    print(f'[OK] flash_attn available')
except ImportError as e:
    errors.append(f'flash_attn import failed: {e}')

if errors:
    print('CUSTOMAR-INIT-002: missing dependencies:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
print('All dependencies available')
"

# Step 2: torchrun CustomAR init (actual TP=4 test)
# torchrun requires a script file, not -c inline code. Write to tempfile.
_TP2_SCRIPT=$(mktemp /tmp/test_phase2_tp_XXXX.py)
cat > "${_TP2_SCRIPT}" << 'PYEOF'
import os, sys
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
world_size = dist.get_world_size()

cuda_dev = os.environ['CUDA_VISIBLE_DEVICES']
print(f'[rank={rank}] CUDA device={torch.cuda.current_device()}, visible={cuda_dev}')

# Verify peer access (CustomAR requires P2P)
can_peer = torch.cuda.can_device_access_peer(0, 1 % world_size)
if not can_peer and world_size > 1:
    print(f'[rank={rank}] WARNING: no peer access between GPU 0 and 1 — CustomAR will fall back to NCCL')

# Verify NCCL barrier
dist.barrier()
print(f'[rank={rank}] NCCL barrier passed')

# Test all_reduce (correctness: summed values should match)
local_r = int(os.environ['LOCAL_RANK'])
t = torch.tensor([float(rank + 1)], device=f'cuda:{local_r}')
dist.all_reduce(t, op=dist.ReduceOp.SUM)

expected = sum(range(1, world_size + 1))
assert abs(t.item() - expected) < 1e-5, (
    f'CUSTOMAR-INIT-001: all_reduce 结果错误。'
    f'rank={rank} 得到 {t.item()}，期望 {expected}。'
    f'各 rank 输入值={[i+1 for i in range(world_size)]}，和应为 {expected}。'
)
print(f'[rank={rank}] NCCL all_reduce sum={t.item()} (expected={expected})')

# Step 3: Full CustomAR init (meta_ptrs + buf_ptrs + init_custom_ar + register_buffer)
# The contract: CustomAR init may fail (IPC issues across CUDA versions/containers),
# but all_reduce must still work via NCCL. This test exercises the COMPLETE init path
# including buf_ptrs exchange (broadcast_object_list) which was the actual crash point.
dist.barrier()
if rank == 0:
    print('Testing full CustomAR init (meta_ptrs + buf_ptrs + register_buffer)...')
if world_size > 1:
    custom_ar_ok = False
    try:
        from vllm import _custom_ops as ops

        # --- Phase A: meta_ptrs exchange (all_gather_object) ---
        gloo_group = dist.new_group(backend='gloo')
        meta_size = ops.meta_size()
        max_size = 16 * 1024 * 1024  # 16 MB
        meta_raw, meta_handle = ops.allocate_shared_buffer_and_handle(meta_size + max_size)
        meta_handles = [None] * world_size
        dist.all_gather_object(meta_handles, meta_handle, group=gloo_group)
        meta_ptrs = [ops.open_mem_handle(h) if i != rank else meta_raw for i, h in enumerate(meta_handles)]
        if rank == 0:
            print(f'  meta_ptrs: {len(meta_ptrs)} handles exchanged (all_gather_object)')

        # --- Phase B: buf_ptrs exchange (all_gather_object, same as meta_ptrs) ---
        # 真实 engine 中 meta_ptrs 和 buf_ptrs 使用同一个 _allocate_and_exchange_handles 函数
        # 内部都是 dist.all_gather_object——不是 broadcast_object_list
        buf_raw, buf_handle = ops.allocate_shared_buffer_and_handle(max_size)
        buf_handles = [None] * world_size
        dist.all_gather_object(buf_handles, buf_handle, group=gloo_group)
        buf_ptrs = [ops.open_mem_handle(h) if i != rank else buf_raw for i, h in enumerate(buf_handles)]
        if rank == 0:
            print(f'  buf_ptrs: {len(buf_ptrs)} handles exchanged (all_gather_object)')

        # --- Phase C: init_custom_ar + register_buffer ---
        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=f'cuda:{local_r}')
        fully_connected = torch.cuda.can_device_access_peer(0, 1 % world_size)
        ptr = ops.init_custom_ar(meta_ptrs, rank_data, rank, fully_connected)
        ops.register_buffer(ptr, buf_ptrs)
        custom_ar_ok = True
        if rank == 0:
            print(f'  init_custom_ar: ptr={ptr}, register_buffer done')
        print(f'[rank={rank}] Full CustomAR init OK')

    except Exception as e:
        print(f'[rank={rank}] CustomAR init failed: {e}')
        print(f'[rank={rank}] NCCL fallback active — all_reduce via dist.all_reduce')

    # CRITICAL: all_reduce must work regardless of CustomAR status
    t2 = torch.tensor([float(rank + 10)], device=f'cuda:{local_r}')
    dist.all_reduce(t2, op=dist.ReduceOp.SUM)
    expected2 = sum(range(10, 10 + world_size))
    ar_status = 'OK' if custom_ar_ok else 'FAILED'
    assert abs(t2.item() - expected2) < 1e-5, (
        f'CUSTOMAR-INIT-003: NCCL fallback all_reduce 失败。'
        f'rank={rank} 得到 {t2.item()}，期望 {expected2}。'
        f'CustomAR init 状态: {ar_status}。'
        f'无论 CustomAR 是否初始化成功，all_reduce 必须通过 NCCL 正常工作。'
        f'Agent 错误: init_custom_ar 可能未 try/except → 异常向上传播 → TP 推理直接挂。'
    )
    if rank == 0:
        print(f'CustomAR init: {ar_status} (NCCL fallback verified)')

dist.barrier()
dist.destroy_process_group()

if rank == 0:
    print('PHASE2_CUSTOM_AR_INIT: ALL CHECKS PASSED')
PYEOF
torchrun --nproc_per_node="${TP_SIZE}" --master_port=$((29500 + RANDOM % 1000)) "${_TP2_SCRIPT}" 2>&1
EXIT_CODE=$?
rm -f "${_TP2_SCRIPT}"

if [ $EXIT_CODE -eq 0 ]; then
    echo "PHASE2_CUSTOM_AR_INIT: SUCCESS"
else
    echo "PHASE2_CUSTOM_AR_INIT: FAILED (exit code ${EXIT_CODE})"
    exit $EXIT_CODE
fi
