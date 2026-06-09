"""
Phase 2 — TP 分布式通信原语 (init + all_reduce + all_gather).

All signatures must match inference_blueprint.json
  > tp_distributed_runtime.init_sequence
  > tp_distributed_runtime.collectives
"""

import os
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Module-level globals — populated by init_custom_ar(), consumed by all_reduce_sum()
# ---------------------------------------------------------------------------
_custom_ar_handle = None
_buf_ptrs: list[int] | None = None
_max_size: int = 16 * 1024 * 1024  # 16 MB


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_world_size() -> int:
    """Safe world_size access: returns 1 if dist is not initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_tp_enabled() -> bool:
    """True when TP is active (world_size > 1 and dist is initialized)."""
    return _get_world_size() > 1


# ===========================================================================
# Task 1: init_tp_distributed()
#   Contract: inference_blueprint.json > tp_distributed_runtime.init_sequence
#   WORLD_SIZE <= 1 guard is MANDATORY — single-process hangs forever on
#   dist.init_process_group('nccl', 'env://') waiting for non-existent master.
# ===========================================================================

def init_tp_distributed() -> None:
    """Initialize torch.distributed with NCCL backend for TP.

    MUST be called after torchrun sets env vars:
        LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

    Built-in guard: WORLD_SIZE <= 1 → immediate return (single GPU, debug mode).
    Idempotent: if dist is already initialized, skip re-init (safe for multi-caller).
    Caller-side must also guard: if _world_size > 1: init_tp_distributed()
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        # Single-process: nothing to initialize.
        # dist.init_process_group would hang waiting for NCCL master.
        return

    if dist.is_available() and dist.is_initialized():
        # Already initialized (e.g., by torchrun or a parent caller).
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()


# ===========================================================================
# Task 2: init_custom_ar(device)
#   Contract: AGENT_SKILL.md §2.1
#     gloo ProcessGroup + IPC exchange + ops.init_custom_ar + ops.register_buffer
#     world_size==1 → immediate return
#     Entire function MUST be wrapped in try/except — open_mem_handle can
#     throw "Cannot access data pointer of Tensor that doesn't have storage"
#     on some CUDA versions. On failure → _custom_ar_handle=None so
#     all_reduce_sum falls back to NCCL automatically.
# ===========================================================================

def init_custom_ar(device: torch.device | int | str | None = None) -> None:
    """Initialize Custom AllReduce (P2P staging-buffer kernel).

    Performs the full 3-phase init machine state:
      Phase A — meta_ptrs: IPC exchange of metadata + staging region
                (allocate_shared_buffer_and_handle + all_gather_object over gloo)
      Phase B — buf_ptrs: IPC exchange of pure staging buffers
                (same all_gather_object pattern)
      Phase C — init_custom_ar + register_buffer

    On failure (any exception), sets _custom_ar_handle=None so
    all_reduce_sum automatically falls back to dist.all_reduce (NCCL).

    Args:
        device: target CUDA device. If None, reads LOCAL_RANK from env.
    """
    global _custom_ar_handle, _buf_ptrs, _max_size

    if not is_tp_enabled():
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    try:
        from vllm import _custom_ops as ops

        # Resolve device
        if device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            cuda_device = torch.device(f"cuda:{local_rank}")
        elif isinstance(device, int):
            cuda_device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            cuda_device = torch.device(device)
        else:
            cuda_device = device

        rank = dist.get_rank()
        max_size = _max_size  # 16 MB

        # ---- Phase A: meta_ptrs exchange (all_gather_object via gloo) ----
        gloo_group = dist.new_group(backend="gloo")
        meta_size_bytes = ops.meta_size()
        meta_raw, meta_handle = ops.allocate_shared_buffer_and_handle(
            meta_size_bytes + max_size
        )
        meta_handles = [None] * world_size
        dist.all_gather_object(meta_handles, meta_handle, group=gloo_group)
        meta_ptrs = [
            ops.open_mem_handle(h) if i != rank else meta_raw
            for i, h in enumerate(meta_handles)
        ]

        # ---- Phase B: buf_ptrs exchange (same as meta_ptrs — all_gather_object) ----
        # NOT broadcast_object_list — that is only used in register_graph_buffers()
        # (CUDA Graph path, nocompile irrelevant).
        buf_raw, buf_handle = ops.allocate_shared_buffer_and_handle(max_size)
        buf_handles = [None] * world_size
        dist.all_gather_object(buf_handles, buf_handle, group=gloo_group)
        buf_ptrs = [
            ops.open_mem_handle(h) if i != rank else buf_raw
            for i, h in enumerate(buf_handles)
        ]

        # ---- Phase C: init_custom_ar + register_buffer ----
        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=cuda_device)
        fully_connected = torch.cuda.can_device_access_peer(0, 1 % world_size)
        handle = ops.init_custom_ar(meta_ptrs, rank_data, rank, fully_connected)
        ops.register_buffer(handle, buf_ptrs)

        _custom_ar_handle = handle
        _buf_ptrs = buf_ptrs

    except Exception:
        # Hard survival requirement: init_custom_ar failure must NOT crash TP inference.
        # all_reduce_sum automatically detects _custom_ar_handle is None and falls
        # back to dist.all_reduce (NCCL).
        _custom_ar_handle = None
        _buf_ptrs = None


# ===========================================================================
# Task 3: all_reduce_sum(x)
#   Contract: inference_blueprint.json > tp_distributed_runtime.collectives.all_reduce_sum
#   @torch.library.custom_op registration + CustomAR P2P → NCCL fallback + register_fake
#   CRITICAL: reg_buf MUST use buf_ptrs[dist.get_rank()], NOT buf_ptrs[0]
#     — rank 0 is correct only by coincidence; rank 1/2/3 would get
#       "buffer address not registered" RuntimeError.
# ===========================================================================

@torch.library.custom_op("meta_infer::all_reduce_sum", mutates_args=())
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """In-place-style out-of-place all-reduce (SUM).

    Paths (in priority order):
      1. TP=1 → x.clone()  (custom_op forbids alias of input)
      2. CustomAR P2P → fast staging-buffer kernel (if init_custom_ar succeeded)
      3. NCCL → dist.all_reduce  (reliable fallback)
    """
    if not is_tp_enabled():
        # custom_op MUST return a new tensor, not an alias of the input
        return x.clone()

    if _custom_ar_handle is not None and _buf_ptrs is not None:
        # CustomAR P2P path
        from vllm import _custom_ops as ops  # noqa: F811

        out = torch.empty_like(x)
        rank = dist.get_rank()
        # ⚠️ MUST use buf_ptrs[dist.get_rank()] — the CURRENT rank's staging buffer.
        # buf_ptrs[0] works for rank 0 by coincidence; rank 1/2/3 would crash with
        # "buffer address not registered" because register_buffer registered
        # buf_ptrs[3] not buf_ptrs[0].
        ops.all_reduce(
            _custom_ar_handle,
            x,
            out,
            _buf_ptrs[rank],
            _max_size,
        )
        return out

    # NCCL fallback
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


@all_reduce_sum.register_fake
def _all_reduce_sum_fake(x: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.compile Dynamo tracing."""
    return torch.empty_like(x)


# ===========================================================================
# Task 4: all_gather_last_dim(x)
#   Contract: inference_blueprint.json > tp_distributed_runtime.collectives.all_gather_last_dim
#   dist.all_gather(outs, x) + torch.cat(outs, dim=-1)
#   Input [..., local_dim] → Output [..., local_dim * tp_size]
# ===========================================================================

def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """All-gather along the last dimension.

    Each rank holds a local shard of the last dimension.
    Output is the full concatenated tensor across all ranks along dim=-1.

    Uses dist.all_gather (NOT all_gather_into_tensor).
    """
    if not is_tp_enabled():
        return x

    tp_size = dist.get_world_size()
    outs = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(outs, x)
    return torch.cat(outs, dim=-1)
