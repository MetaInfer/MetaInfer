# engine/tp_layers/distributed.py
# Phase 2: TP 通信 — all_reduce_sum, all_gather_last_dim, CustomAR init.
#
# Blueprint contract:
#   framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime
#   framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce
#
# Ref:
#   inference_blueprint.json > tp_distributed_runtime.collectives
#   inference_blueprint.json > custom_ar_all_reduce.init_state_machine
#   inference_blueprint.json > custom_ar_all_reduce.register_buffer_detail

import os
import torch
import torch.distributed as dist
from typing import Any

# ================================================================
# Module-level state
# ================================================================

_custom_ar_handle: Any = None
_tp_initialized: bool = False


# ================================================================
# Helper functions
# ================================================================

def is_tp_enabled() -> bool:
    """Return True when distributed is initialized and world_size > 1."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_tp_size() -> int:
    """Return world_size if dist is initialized, else 1."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_tp_rank() -> int:
    """Return rank if dist is initialized, else 0."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_custom_ar_handle() -> Any:
    """Return the module-level CustomAR handle (None if not initialized)."""
    return _custom_ar_handle


# ================================================================
# _allocate_and_exchange_handles — shared IPC handle exchange helper
# ================================================================

def _allocate_and_exchange_handles(size: int, gloo_group, rank: int, world_size: int):
    """
    Allocate a shared buffer and exchange IPC handles across all ranks.

    Blueprint contract (register_buffer_detail):
      - Each rank allocates exactly ONE buffer via allocate_shared_buffer_and_handle(size).
      - IPC handles are exchanged via dist.all_gather_object (NOT broadcast_object_list).
      - Remote handles are opened via ops.open_mem_handle.

    Returns:
        pointers: list[int] — raw pointer for each rank (index i = rank i's buffer).
                  pointers[rank] == the local raw_ptr (not opened via open_mem_handle).
    """
    from vllm import _custom_ops as ops

    # Allocate exactly one buffer (Bug 2 fix: no per-rank loop).
    raw_ptr, ipc_handle = ops.allocate_shared_buffer_and_handle(size)

    # Exchange IPC handles via all_gather_object (Bug 1 fix: no broadcast_object_list).
    handles = [None] * world_size
    dist.all_gather_object(handles, ipc_handle, group=gloo_group)

    # Open remote handles; keep local raw_ptr as-is.
    pointers = [
        ops.open_mem_handle(h) if i != rank else raw_ptr
        for i, h in enumerate(handles)
    ]
    return pointers


# ================================================================
# CustomAllReduceHandle
# ================================================================

class CustomAllReduceHandle:
    """
    Encapsulates CustomAR P2P state.

    Attributes:
        _ptr:       int — opaque pointer returned by ops.init_custom_ar
        rank:       int — current process rank
        world_size: int — total TP processes
        rank_data:  Tensor[uint8] — staging buffer for all_reduce workspace
        buf_ptrs:   list[int] — registered staging buffer raw pointers
        max_size:   int — staging buffer size in bytes (16 MB)
    """

    def __init__(
        self,
        ptr: int,
        rank: int,
        world_size: int,
        rank_data: torch.Tensor,
        buf_ptrs: list,
        max_size: int,
    ):
        self._ptr = ptr
        self.rank = rank
        self.world_size = world_size
        self.rank_data = rank_data
        self.buf_ptrs = buf_ptrs
        self.max_size = max_size

    def all_reduce(self, x: torch.Tensor, registered: bool = False) -> torch.Tensor:
        """
        Out-of-place all-reduce via CustomAR P2P kernel.

        Args:
            x:          input tensor (bf16/fp16)
            registered: whether x is already registered as a P2P buffer

        Returns:
            out: new tensor = sum of x across all ranks
        """
        from vllm import _custom_ops as ops

        out = torch.empty_like(x)
        # reg_buf: use first staging buffer pointer (0 = no registered buffer)
        reg_buf = self.buf_ptrs[dist.get_rank()]  # must be THIS rank's buffer, not rank 0
        reg_buf_sz_bytes = self.max_size
        ops.all_reduce(self._ptr, x, out, reg_buf, reg_buf_sz_bytes)
        return out


# ================================================================
# all_reduce_sum — custom_op registered for torch.compile compatibility
# ================================================================

@torch.library.custom_op("meta_infer::all_reduce_sum", mutates_args=())
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Sum-reduce tensor across all TP ranks.

    Priority: CustomAR P2P > NCCL fallback.
    TP=1 returns x.clone() (custom_op forbids output aliasing input).

    Contract:
        _custom_ar_handle is not None → P2P staging buffer all_reduce
        _custom_ar_handle is None     → dist.all_reduce(NCCL)
        tp_size == 1                  → x.clone() (no-op)
    """
    if not is_tp_enabled():
        # custom_op forbids returning input alias → must clone
        return x.clone()

    if _custom_ar_handle is not None:
        return _custom_ar_handle.all_reduce(x, registered=False)

    # NCCL fallback
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


@all_reduce_sum.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    """Fake kernel for torch.compile tracing: returns empty_like(x)."""
    return torch.empty_like(x)


# ================================================================
# all_gather_last_dim
# ================================================================

def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    All-gather tensor along last dimension across TP ranks.

    Uses dist.all_gather(outs, x) + torch.cat(outs, dim=-1).
    Forbidden: dist.all_gather_into_tensor (blueprint explicit prohibition).

    Input:  [..., local_dim]
    Output: [..., local_dim * tp_size]

    TP=1: return x unchanged.
    """
    if not is_tp_enabled():
        return x

    tp_size = get_tp_size()
    outs = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(outs, x)
    return torch.cat(outs, dim=-1)


# ================================================================
# init_tp_distributed — NCCL bootstrap
# ================================================================

def init_tp_distributed():
    """
    Initialize NCCL distributed process group.

    Reads env vars: LOCAL_RANK, RANK, WORLD_SIZE (set by torchrun).
    Call once per process before any collective communication.
    """
    global _tp_initialized

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    _tp_initialized = True


# ================================================================
# init_custom_ar — P2P Custom AllReduce bootstrap
# ================================================================

def init_custom_ar(device=None):
    """
    Initialize CustomAR P2P all-reduce.

    CRITICAL: entire function is wrapped in try/except.
    On failure _custom_ar_handle stays None → all_reduce_sum auto-falls-back to NCCL.

    Sets the module-level _custom_ar_handle variable.

    Corrected flow (P0-3 bugs fixed):
        Phase A: meta_ptrs = _allocate_and_exchange_handles(meta_size + max_size, ...)
        Phase B: buf_ptrs  = _allocate_and_exchange_handles(max_size, ...)
        Phase C: ops.init_custom_ar(meta_ptrs, rank_data, ...) + ops.register_buffer(_ptr, buf_ptrs)

    Both exchanges use all_gather_object (NOT broadcast_object_list).
    Each rank allocates exactly 1 buffer (NOT world_size buffers).

    world_size == 1: immediate return (no-op).
    """
    global _custom_ar_handle

    if not dist.is_initialized():
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return

    try:
        from vllm import _custom_ops as ops

        # Step 1: all ranks synced after load_weights()
        dist.barrier()
        if rank == 0:
            print("weights loaded, initializing CustomAR...")

        # Step 2: create gloo ProcessGroup for IPC handle exchange
        gloo_group = dist.new_group(backend="gloo")

        max_size = 16 * 1024 * 1024  # 16 MB staging buffer
        meta_size = ops.meta_size()

        # Phase A: allocate and exchange meta_ptrs (metadata + staging buffer)
        meta_ptrs = _allocate_and_exchange_handles(
            meta_size + max_size, gloo_group, rank, world_size
        )

        # Phase B: allocate and exchange buf_ptrs (staging buffer only)
        buf_ptrs = _allocate_and_exchange_handles(
            max_size, gloo_group, rank, world_size
        )

        # Phase C: init_custom_ar + register_buffer
        target_device = device if device is not None else torch.device(f"cuda:{rank}")
        rank_data = torch.empty(max_size, dtype=torch.uint8, device=target_device)
        fully_connected = _check_p2p(rank, world_size, target_device)
        _ptr = ops.init_custom_ar(meta_ptrs, rank_data, rank, fully_connected)
        ops.register_buffer(_ptr, buf_ptrs)

        # Sync completion
        dist.barrier()
        if rank == 0:
            print("CustomAR initialized")

        _custom_ar_handle = CustomAllReduceHandle(
            ptr=_ptr,
            rank=rank,
            world_size=world_size,
            rank_data=rank_data,
            buf_ptrs=buf_ptrs,
            max_size=max_size,
        )

    except Exception as e:
        if rank == 0:
            print(f"CustomAR init failed: {e}. Falling back to NCCL.")
        _custom_ar_handle = None
        # Do not raise — NCCL fallback is fully functional.


# ================================================================
# _check_p2p — peer-to-peer access verification
# ================================================================

def _check_p2p(rank: int, world_size: int, device) -> bool:
    """
    Verify P2P access between all GPU pairs in this TP group.

    Uses torch.cuda.can_device_access_peer to test each pair.
    In single-node deployment LOCAL_RANK maps directly to GPU index.

    Ref: vllm/_custom_ops.py ~640-680 (CustomAR init flow)
    """
    if world_size <= 1:
        return True

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        for i in range(world_size):
            if i != rank:
                if not torch.cuda.can_device_access_peer(local_rank, i):
                    return False
        return True
    except Exception:
        return False
