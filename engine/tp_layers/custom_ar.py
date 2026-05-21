"""CustomAllReduce — vLLM P2P all_reduce kernel, replacing NCCL for small tensors.

Extracted from vLLM's custom_all_reduce.py. All kernel calls are black-box.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from vllm import _custom_ops as ops


# ---- public API ----


class CustomAllReduceHandle:
    """Per-rank state for vLLM P2P CustomAllReduce."""

    def __init__(self, group: dist.ProcessGroup, device: torch.device, max_size: int):
        self._ptr: int = 0
        self._disposed = False
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        if world_size == 1:
            return  # single GPU — no-op

        assert world_size in (2, 4, 6, 8), f"CustomAR supports 2/4/6/8 GPUs, got {world_size}"

        # Allocate metadata + staging buffer (one per rank, exchanged via IPC)
        meta_size = ops.meta_size()
        ptrs = _allocate_and_exchange_handles(meta_size + max_size, group, rank, world_size)
        self._meta_ptrs = ptrs

        # Allocate pre-registered IPC buffer for eager-mode copies
        buf_ptrs = _allocate_and_exchange_handles(max_size, group, rank, world_size)
        self._buf_ptrs = buf_ptrs
        self._max_size = max_size

        # rank_data: buffer for pointers from all ranks
        self._rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)

        # P2P connectivity check
        fully_connected = _check_p2p(rank, world_size)

        self._ptr = ops.init_custom_ar(self._meta_ptrs, self._rank_data, rank, fully_connected)
        ops.register_buffer(self._ptr, self._buf_ptrs)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Out-of-place all_reduce via P2P kernel.

        If the handle is not initialized (single GPU), returns inp unchanged.
        """
        if self._ptr == 0:
            return inp
        out = torch.empty_like(inp)
        ops.all_reduce(
            self._ptr, inp, out,
            self._buf_ptrs[dist.get_rank()], self._max_size,
        )
        return out

    def close(self):
        if self._ptr and not self._disposed:
            ops.dispose(self._ptr)
            self._ptr = 0
            self._disposed = True


# ---- internal helpers (extracted + simplified from vLLM) ----


def _allocate_and_exchange_handles(
    size: int, group: dist.ProcessGroup, rank: int, world_size: int,
) -> list[int]:
    """Allocate IPC shared buffer and exchange handles across ranks."""
    raw_pointer, ipc_handle = ops.allocate_shared_buffer_and_handle(size)
    # Exchange IPC handles via gloo all_gather (CPU operation)
    handles = [None] * world_size
    dist.all_gather_object(handles, ipc_handle, group=group)
    # Open remote handles to build the full pointer list
    pointers = [0] * world_size
    for i, h in enumerate(handles):
        if i == rank:
            pointers[i] = raw_pointer
        else:
            pointers[i] = ops.open_mem_handle(h)
    return pointers


def _check_p2p(rank: int, world_size: int) -> bool:
    """Check if all GPU pairs have P2P access."""
    for i in range(world_size):
        if i != rank and not torch.cuda.can_device_access_peer(rank, i):
            return False
    return True
