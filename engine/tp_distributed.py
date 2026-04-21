"""
Tensor-parallel 辅助：在 SPMD 下初始化进程组并解析 TP rank / size。
不调用 init 时，视为 TP_SIZE=1 的单机路径（与未设置 RANK 一致）。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_ENV_RANK = "RANK"
_ENV_WORLD = "WORLD_SIZE"
_ENV_LOCAL_RANK = "LOCAL_RANK"


def get_tp_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get(_ENV_RANK, "0"))


def get_tp_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get(_ENV_WORLD, "1"))


def is_distributed() -> bool:
    return get_tp_size() > 1


def init_distributed() -> str:
    """
    多进程时初始化进程组；单进程时直接返回 ``gloo`` 占位。
    返回 backend 名称。cuda 用 nccl，否则 gloo（便于 CPU/单卡双进程 TDD）。
    """
    if dist.is_initialized():
        return dist.get_backend() or "gloo"
    if int(os.environ.get(_ENV_WORLD, "1")) <= 1:
        return "gloo"

    force_gloo = os.environ.get("META_INFER_TP_BACKEND", "").lower() == "gloo"
    if not force_gloo and torch.cuda.is_available() and _ENV_LOCAL_RANK in os.environ:
        torch.cuda.set_device(int(os.environ[_ENV_LOCAL_RANK]))
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return backend
