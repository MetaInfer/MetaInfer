from __future__ import annotations

import os

import torch
import torch.distributed as dist

# Lazy-loaded CustomAllReduce handle (init once after dist init + model on GPU)
_custom_ar_handle = None


def get_tp_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_tp_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_tp_enabled() -> bool:
    return get_tp_size() > 1


def init_tp_distributed() -> str:
    if dist.is_initialized():
        return dist.get_backend()
    if get_tp_size() <= 1:
        return "none"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=backend, init_method="env://")
    return backend


def init_custom_ar(device: torch.device | None = None, max_size: int = 16 * 1024 * 1024) -> None:
    """Initialize vLLM CustomAllReduce (P2P) for TP all_reduce.

    Must be called after init_tp_distributed() and after model is on GPU.
    Creates a secondary gloo group for IPC handle exchange (required by vLLM kernel).
    """
    global _custom_ar_handle
    if _custom_ar_handle is not None:
        return  # already initialized
    if not is_tp_enabled():
        return
    world_size = get_tp_size()
    if world_size == 1 or world_size not in (2, 4, 6, 8):
        return

    # Create a secondary gloo group for IPC handle exchange
    # (vLLM requires non-NCCL group for all_gather_object)
    gloo_group = dist.new_group(backend="gloo")

    if device is None:
        device = torch.device(f"cuda:{get_tp_rank()}")

    from engine.tp_layers.custom_ar import CustomAllReduceHandle
    _custom_ar_handle = CustomAllReduceHandle(gloo_group, device, max_size)


def ensure_divisible(value: int, divisor: int, *, name: str) -> None:
    if value % divisor != 0:
        raise ValueError(f"{name}={value} is not divisible by tp_size={divisor}")


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not is_tp_enabled():
        return x
    if _custom_ar_handle is not None:
        return _custom_ar_handle.all_reduce(x)
    # Fallback to NCCL
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    if not is_tp_enabled():
        return x
    outs = [torch.empty_like(x) for _ in range(get_tp_size())]
    dist.all_gather(outs, x)
    return torch.cat(outs, dim=-1)
