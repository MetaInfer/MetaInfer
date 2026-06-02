# engine/tp_layers/linear.py
# Phase 3: TP 线性层 — Column/Row/Merged/QKV Parallel Linear.
#
# Blueprint contract:
#   framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers
#
# 4 种 TP Linear:
#   1. ColumnParallelLinear      — [out/tp, in], 可选的 all_gather 输出
#   2. RowParallelLinear         — [out, in/tp], 自动 all_reduce_sum
#   3. QKVColumnParallelLinear   — merged QKV: [q_size+2*kv_size, hidden], split → (q,k,v)
#   4. MergedColumnParallelLinear — merged gate+up: [2*intermediate/tp, in]
#
# All classes: double_shard_guard in load_weight_shard()
#
# Ref:
#   notebooks-cn/04_parallel_strategies/02_qwen_dense_tp_implementation_guide.md
#   notebooks-cn/06_experience/01_task10_tp_qwen_debug_experience.md

import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.tp_layers.distributed import all_reduce_sum, all_gather_last_dim, get_tp_size, get_tp_rank


# ================================================================
# ColumnParallelLinear
# ================================================================

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer: weight is split along output dimension (dim 0).

    Weight shape: [out_features / tp_size, in_features]
    Forward output: [B, T, out_features / tp_size] (or full [B, T, out_features] if gather_output=True)

    Args:
        in_features:   input feature dimension (full, not per-rank)
        out_features:  output feature dimension (full, not per-rank)
        tp_size:       tensor-parallel world size (default: from distributed runtime)
        gather_output: if True, all_gather outputs along last dim before returning
        bias:          if True, add bias after linear (bias shape matches per-rank output)
    """

    def __init__(self, in_features, out_features, tp_size=None, gather_output=False, bias=False):
        super().__init__()
        self.tp_size = tp_size if tp_size is not None else get_tp_size()
        self.tp_rank = get_tp_rank()
        self.in_features = in_features
        self.out_features = out_features  # full output dimension
        self.gather_output = gather_output

        local_out = out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(local_out, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(local_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_features] → y: [B, T, out_features/tp] (or [B, T, out_features] with gather)"""
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            y = all_gather_last_dim(y)
        return y

    def load_weight_shard(self, shard: torch.Tensor) -> None:
        """
        Load a weight shard with double_shard_guard.

        If shard.shape == self.weight.shape: direct copy_ (already pre-sliced).
        Otherwise: slice from full weight by tp_rank along dim 0.
        """
        if shard.shape == self.weight.shape:
            self.weight.data.copy_(shard)
        else:
            local_out = self.out_features // self.tp_size
            start = self.tp_rank * local_out
            end = start + local_out
            self.weight.data.copy_(shard[start:end, :])


# ================================================================
# RowParallelLinear
# ================================================================

class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer: weight is split along input dimension (dim 1).

    Weight shape: [out_features, in_features / tp_size]
    Forward: F.linear → all_reduce_sum (always called, handles tp=1 as no-op) → +bias

    Args:
        in_features:   input feature dimension (full, caller passes e.g. intermediate_size=12288)
        out_features:  output feature dimension (full)
        tp_size:       tensor-parallel world size (default: from distributed runtime)
        bias:          if True, add bias after all_reduce_sum
    """

    def __init__(self, in_features, out_features, tp_size=None, bias=False):
        super().__init__()
        self.tp_size = tp_size if tp_size is not None else get_tp_size()
        self.tp_rank = get_tp_rank()
        self.in_features = in_features  # full input dimension
        self.out_features = out_features

        local_in = in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(out_features, local_in))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_features/tp] → y: [B, T, out_features]"""
        y = F.linear(x, self.weight)
        y = all_reduce_sum(y)
        if self.bias is not None:
            y = y + self.bias
        return y

    def load_weight_shard(self, shard: torch.Tensor) -> None:
        """
        Load a weight shard with double_shard_guard.

        If shard.shape == self.weight.shape: direct copy_ (already pre-sliced).
        Otherwise: slice from full weight by tp_rank along dim 1 (input dimension).
        """
        if shard.shape == self.weight.shape:
            self.weight.data.copy_(shard)
        else:
            local_in = self.in_features // self.tp_size
            start = self.tp_rank * local_in
            end = start + local_in
            self.weight.data.copy_(shard[:, start:end])


# ================================================================
# QKVColumnParallelLinear
# ================================================================

class QKVColumnParallelLinear(nn.Module):
    """
    Merged QKV column-parallel linear layer for attention.

    Computes per-rank head counts internally.
    Weight shape: [q_size + 2*kv_size, hidden_size] per rank.
    Forward: F.linear → optionally all_gather → split([q_size, kv_size, kv_size]) → (q, k, v)

    Q/K/V order: Q first (front), K middle, V last.

    Args:
        hidden_size:       model hidden size (e.g. 4096)
        head_dim:          attention head dimension (e.g. 128)
        total_num_heads:   total Q heads across all ranks (e.g. 32)
        total_num_kv_heads: total KV heads across all ranks (e.g. 8), GQA compatible
        tp_size:           tensor-parallel world size (default: from distributed runtime)
        gather_output:     if True, all_gather before split (returns full Q/K/V per rank)
        bias:              if True, add bias after linear
    """

    def __init__(self, hidden_size, head_dim, total_num_heads, total_num_kv_heads,
                 tp_size=None, gather_output=False, bias=False):
        super().__init__()
        self.tp_size = tp_size if tp_size is not None else get_tp_size()
        self.tp_rank = get_tp_rank()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.gather_output = gather_output

        # Per-rank head counts
        self.num_heads = total_num_heads // self.tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // self.tp_size)

        # Per-rank output sizes
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        # Merged QKV weight: [q_size + 2*kv_size, hidden_size]
        local_out = self.q_size + 2 * self.kv_size
        self.weight = nn.Parameter(torch.empty(local_out, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(local_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: [B, T, hidden_size]
        Returns: (q, k, v) each [B, T, q_size] / [B, T, kv_size] / [B, T, kv_size]
        """
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            y = all_gather_last_dim(y)
        q, k, v = y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def load_weight_shard(self, shard: torch.Tensor) -> None:
        """
        Load a merged QKV weight shard with double_shard_guard.

        If shard.shape == self.weight.shape: direct copy_ (already pre-sliced).
        Otherwise: slice Q, K, V sections from the full merged weight by tp_rank.

        Full weight layout: [Q_section | K_section | V_section]
          Q:  total_num_heads * head_dim rows, each rank takes num_heads * head_dim
          K:  total_num_kv_heads * head_dim rows, each rank takes num_kv_heads * head_dim
          V:  total_num_kv_heads * head_dim rows, each rank takes num_kv_heads * head_dim
        """
        if shard.shape == self.weight.shape:
            self.weight.data.copy_(shard)
        else:
            # Full dimensions
            total_q = self.total_num_heads * self.head_dim
            total_kv = self.total_num_kv_heads * self.head_dim

            # Per-rank slice sizes
            q_slice = self.q_size  # = num_heads * head_dim
            kv_slice = self.kv_size  # = num_kv_heads * head_dim

            # Q section: rank * q_slice : (rank+1) * q_slice
            q_start = self.tp_rank * q_slice
            q_end = q_start + q_slice
            q_shard = shard[q_start:q_end, :]

            # K section: total_q + rank * kv_slice : total_q + (rank+1) * kv_slice
            k_start = total_q + self.tp_rank * kv_slice
            k_end = k_start + kv_slice
            k_shard = shard[k_start:k_end, :]

            # V section: total_q + total_kv + rank * kv_slice
            v_start = total_q + total_kv + self.tp_rank * kv_slice
            v_end = v_start + kv_slice
            v_shard = shard[v_start:v_end, :]

            sliced = torch.cat([q_shard, k_shard, v_shard], dim=0)
            self.weight.data.copy_(sliced)


# ================================================================
# MergedColumnParallelLinear
# ================================================================

class MergedColumnParallelLinear(nn.Module):
    """
    Merged column-parallel linear for gate+up projection (MLP).

    Merges gate_proj and up_proj into a single GEMM.
    Weight shape: [2 * out_features / tp_size, in_features]
    Forward output: [B, T, 2 * out_features / tp_size] (first half=gate, second half=up)

    Args:
        in_features:   input feature dimension (e.g. hidden_size=4096)
        out_features:  intermediate_size (full, e.g. 12288). Weight internally doubles this for gate+up.
        tp_size:       tensor-parallel world size (default: from distributed runtime)
        gather_output: if True, all_gather outputs along last dim before returning
        bias:          if True, add bias after linear
    """

    def __init__(self, in_features, out_features, tp_size=None, gather_output=False, bias=False):
        super().__init__()
        self.tp_size = tp_size if tp_size is not None else get_tp_size()
        self.tp_rank = get_tp_rank()
        self.in_features = in_features
        self.out_features = out_features  # = intermediate_size (full), merged output = 2*out_features
        self.gather_output = gather_output

        # Merged gate+up: [2 * out_features / tp, in_features]
        # e.g. out_features=12288, tp=4 → local_out=6144 (3072 gate + 3072 up)
        local_out = 2 * out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(local_out, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(local_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, in_features]
        Returns: [B, T, 2 * out_features / tp_size] (first half=gate, second half=up)
        """
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            y = all_gather_last_dim(y)
        return y

    def load_weight_shard(self, shard: torch.Tensor) -> None:
        """
        Load a merged gate+up weight shard with double_shard_guard.

        If shard.shape == self.weight.shape: direct copy_ (already pre-sliced).
        Otherwise: slice gate and up sections from the full merged weight by tp_rank.

        Full weight layout: [gate_section | up_section]
          gate: out_features rows, each rank takes out_features/tp rows
          up:   out_features rows, each rank takes out_features/tp rows
        """
        if shard.shape == self.weight.shape:
            self.weight.data.copy_(shard)
        else:
            local_out_per = self.out_features // self.tp_size

            # Gate section: rank * local_out_per : (rank+1) * local_out_per
            gate_start = self.tp_rank * local_out_per
            gate_end = gate_start + local_out_per
            gate_shard = shard[gate_start:gate_end, :]

            # Up section: out_features + rank * local_out_per
            up_start = self.out_features + self.tp_rank * local_out_per
            up_end = up_start + local_out_per
            up_shard = shard[up_start:up_end, :]

            sliced = torch.cat([gate_shard, up_shard], dim=0)
            self.weight.data.copy_(sliced)
