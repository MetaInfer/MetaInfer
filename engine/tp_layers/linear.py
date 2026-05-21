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


class QKVColumnParallelLinear(nn.Module):
    """Merge q_proj, k_proj, v_proj into one GEMM (matching vLLM QKVParallelLinear)."""

    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads,
                 bias=False, gather_output=False):
        super().__init__()
        self.tp_size = get_tp_size()
        self.head_size = head_size
        self.num_heads = total_num_heads // self.tp_size
        if self.tp_size > total_num_kv_heads:
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = total_num_kv_heads // self.tp_size
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size
        # Keep total sizes for weight slicing
        self.total_q_size = total_num_heads * head_size
        self.total_kv_size = total_num_kv_heads * head_size
        total_local = self.q_size + self.kv_size * 2
        self.weight = nn.Parameter(torch.empty(total_local, hidden_size))
        self.gather_output = gather_output

    def forward(self, x):
        y = F.linear(x, self.weight)
        if self.gather_output and self.tp_size > 1:
            y = all_gather_last_dim(y)
        q, k, v = y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def load_weight_shard(self, q_weight, k_weight, v_weight):
        """Load QKV weights. Weights are already TP-sliced by _load_tensor."""
        w_q = q_weight.to(device=self.weight.device, dtype=self.weight.dtype)
        w_k = k_weight.to(device=self.weight.device, dtype=self.weight.dtype)
        w_v = v_weight.to(device=self.weight.device, dtype=self.weight.dtype)
        self.weight.data[:self.q_size].copy_(w_q)
        self.weight.data[self.q_size:self.q_size + self.kv_size].copy_(w_k)
        self.weight.data[self.q_size + self.kv_size:].copy_(w_v)
