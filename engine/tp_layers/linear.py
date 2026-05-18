from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.tp_layers.distributed import (
    all_gather_last_dim,
    all_reduce_sum,
    ensure_divisible,
    get_tp_rank,
    get_tp_size,
)


class ColumnParallelLinear(nn.Module):
    """Shard weight on output dimension (dim=0)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False, gather_output: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.tp_size = get_tp_size()
        ensure_divisible(output_size, self.tp_size, name="output_size")
        self.local_output_size = output_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.local_output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(self.local_output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return all_gather_last_dim(y) if self.gather_output else y

    def load_weight_shard(self, full_weight: torch.Tensor) -> None:
        # 若已为本 rank 的列分片（例如 safetensors get_slice 惰性切片），与 self.weight 同形则直拷，避免双重复切片。
        if full_weight.shape == self.weight.shape:
            self.weight.data.copy_(
                full_weight.to(device=self.weight.device, dtype=self.weight.dtype)
            )
            return
        rank = get_tp_rank()
        start = rank * self.local_output_size
        end = start + self.local_output_size
        self.weight.data.copy_(full_weight[start:end, :].to(device=self.weight.device, dtype=self.weight.dtype))


class MergedColumnParallelLinear(nn.Module):
    """Merge gate_proj+up_proj into one GEMM: weight=[2*local_out, Hin]."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False, gather_output: bool = False):
        super().__init__()
        self.gather_output = gather_output
        self.tp_size = get_tp_size()
        ensure_divisible(output_size, self.tp_size, name="output_size")
        self.local_output_size = output_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(2 * self.local_output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(2 * self.local_output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return all_gather_last_dim(y) if self.gather_output else y

    def load_weight_shard(self, gate_weight: torch.Tensor, up_weight: torch.Tensor) -> None:
        if gate_weight.shape[0] != self.local_output_size:
            r = get_tp_rank()
            s = r * self.local_output_size
            e = s + self.local_output_size
            gate_weight = gate_weight[s:e, :]
            up_weight = up_weight[s:e, :]
        self.weight.data[:self.local_output_size].copy_(gate_weight.to(device=self.weight.device, dtype=self.weight.dtype))
        self.weight.data[self.local_output_size:].copy_(up_weight.to(device=self.weight.device, dtype=self.weight.dtype))


class RowParallelLinear(nn.Module):
    """Shard weight on input dimension (dim=1), then all-reduce partial outputs."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = get_tp_size()
        ensure_divisible(input_size, self.tp_size, name="input_size")
        self.local_input_size = input_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(output_size, self.local_input_size))
        self.bias = nn.Parameter(torch.zeros(output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, None)
        y = all_reduce_sum(y)
        if self.bias is not None:
            y = y + self.bias
        return y

    def load_weight_shard(self, full_weight: torch.Tensor) -> None:
        if full_weight.shape == self.weight.shape:
            self.weight.data.copy_(
                full_weight.to(device=self.weight.device, dtype=self.weight.dtype)
            )
            return
        rank = get_tp_rank()
        start = rank * self.local_input_size
        end = start + self.local_input_size
        self.weight.data.copy_(full_weight[:, start:end].to(device=self.weight.device, dtype=self.weight.dtype))
