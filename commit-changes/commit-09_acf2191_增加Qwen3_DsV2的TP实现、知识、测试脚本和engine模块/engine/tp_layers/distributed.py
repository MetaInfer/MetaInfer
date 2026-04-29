from __future__ import annotations

import os

import torch
import torch.distributed as dist


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


def ensure_divisible(value: int, divisor: int, *, name: str) -> None:
    if value % divisor != 0:
        raise ValueError(f"{name}={value} is not divisible by tp_size={divisor}")


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not is_tp_enabled():
        return x
    if x.dtype in (torch.float16, torch.bfloat16):
        tmp = x.float()
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
        return tmp.to(dtype=x.dtype)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    if not is_tp_enabled():
        return x
    outs = [torch.empty_like(x) for _ in range(get_tp_size())]
    dist.all_gather(outs, x)
    return torch.cat(outs, dim=-1)
