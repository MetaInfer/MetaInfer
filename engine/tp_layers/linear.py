"""
Phase 3 — TP Linear Layers.

Implements 4 TP linear layer types + load_weight_shard with double-shard guard.
All signatures must match inference_blueprint.json
  > tp_linear_layers (4 Linear shapes + forward pseudocode)
  > qwen3_8b_model_dims (per-rank dimensions)
  > qwen3_tp_model_interfaces.class_hierarchy (constructor signatures)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.tp_layers.distributed import all_reduce_sum, all_gather_last_dim, _get_world_size


# ===========================================================================
# ColumnParallelLinear — weight [out/tp, in], optional all_gather_last_dim
# ===========================================================================

class ColumnParallelLinear(nn.Module):
    """Linear layer with column-wise weight sharding.

    Weight shape: [out_features // tp_size, in_features]
    Forward: y = F.linear(x, self.weight)
    Optional: all_gather_last_dim (when gather_output=True and tp_size > 1)

    Constructor args:
        in_features:    full input dimension (e.g. 4096)
        out_features:   full output dimension BEFORE TP split (e.g. 12288)
        bias:           whether to include bias
        gather_output:  if True, all_gather along last dim after linear
        tp_size:        tensor parallel size (auto-detect from env if None)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        if tp_size is None:
            tp_size = _get_world_size()
        self.tp_size = tp_size
        self.tp_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.in_features = in_features
        self.out_features = out_features  # full (pre-TP)
        self.out_features_per_rank = out_features // tp_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: F.linear + optional all_gather_last_dim.

        x: [B, T, in_features]
        returns: [B, T, out_features_per_rank] or [B, T, out_features] if gather_output
        """
        y = F.linear(x, self.weight, self.bias)  # [B, T, out_features_per_rank]
        if self.gather_output and self.tp_size > 1:
            y = all_gather_last_dim(y)  # [B, T, out_features]
        return y

    def load_weight_shard(self, weight: torch.Tensor) -> None:
        """Load a weight shard with double-shard guard.

        If weight.shape == self.weight.shape, the weight is already pre-sliced →
        direct copy_. Otherwise, slice along dim=0 by tp_rank.

        Args:
            weight: full [out_features, in_features] or pre-sliced [out_features_per_rank, in_features]
        """
        if weight.shape == self.weight.shape:
            # Already pre-sliced per-rank shard → direct copy
            self.weight.data.copy_(weight)
        else:
            # Full weight → slice along dim=0 by tp_rank
            r = self.tp_rank
            per_rank = self.out_features_per_rank
            shard = weight[r * per_rank : (r + 1) * per_rank, :].contiguous()
            self.weight.data.copy_(shard)


# ===========================================================================
# RowParallelLinear — weight [out, in/tp], forward=F.linear + all_reduce_sum
# ===========================================================================

class RowParallelLinear(nn.Module):
    """Linear layer with row-wise weight sharding.

    Weight shape: [out_features, in_features // tp_size]
    Forward: y = F.linear(x, self.weight) + all_reduce_sum (always called)

    Constructor args:
        in_features:    full input dimension BEFORE TP split (e.g. 12288)
        out_features:   full output dimension (e.g. 4096)
        bias:           whether to include bias
        tp_size:        tensor parallel size (auto-detect from env if None)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        if tp_size is None:
            tp_size = _get_world_size()
        self.tp_size = tp_size
        self.tp_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.in_features = in_features  # full (pre-TP)
        self.out_features = out_features  # full
        self.in_features_per_rank = in_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: F.linear + all_reduce_sum.

        x: [B, T, in_features_per_rank] (input already sharded along last dim)
        returns: [B, T, out_features] (after all_reduce_sum)
        """
        y = F.linear(x, self.weight, None)  # [B, T, out_features]
        y = all_reduce_sum(y)  # single invocation, always called (tp_size=1 → no-op)
        if self.bias is not None:
            y = y + self.bias
        return y

    def load_weight_shard(self, weight: torch.Tensor) -> None:
        """Load a weight shard with double-shard guard.

        If weight.shape == self.weight.shape, the weight is already pre-sliced →
        direct copy_. Otherwise, slice along dim=1 by tp_rank (RowParallel: dim=-1 shard).

        Args:
            weight: full [out_features, in_features] or pre-sliced [out_features, in_features_per_rank]
        """
        if weight.shape == self.weight.shape:
            # Already pre-sliced per-rank shard → direct copy
            self.weight.data.copy_(weight)
        else:
            # Full weight → slice along dim=1 by tp_rank
            r = self.tp_rank
            per_rank = self.in_features_per_rank
            shard = weight[:, r * per_rank : (r + 1) * per_rank].contiguous()
            self.weight.data.copy_(shard)


# ===========================================================================
# MergedColumnParallelLinear — gate+up merged [2*inter/tp, hidden], no gather
# ===========================================================================

class MergedColumnParallelLinear(nn.Module):
    """Merged column-parallel linear for gate_proj + up_proj.

    Single GEMM replaces two separate projections.
    Weight shape: [2 * intermediate_size // tp_size, hidden_size]
    No all_gather — output fed directly to silu_and_mul.

    Constructor args:
        hidden_size:        e.g. 4096
        intermediate_size:  e.g. 12288 (full intermediate)
        bias:               always False for Qwen3
        gather_output:      always False for gate+up (silu_and_mul consumes per-rank)
        tp_size:            tensor parallel size (auto-detect from env if None)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        gather_output: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        if tp_size is None:
            tp_size = _get_world_size()
        self.tp_size = tp_size
        self.tp_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_per_rank = intermediate_size // tp_size  # e.g. 3072
        self.gate_up_out_dim = 2 * self.intermediate_per_rank  # e.g. 6144, NOT 6400
        self.gather_output = gather_output

        self.weight = nn.Parameter(
            torch.empty(self.gate_up_out_dim, hidden_size, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.gate_up_out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: single F.linear for gate+up merged.

        x: [B, T, hidden_size]
        returns: [B, T, 2 * intermediate_per_rank]  — first half=gate, second half=up
        """
        y = F.linear(x, self.weight, self.bias)  # [B, T, 2*intermediate_per_rank]
        # gather_output=False for gate_up → no all_gather (silu_and_mul operates on per-rank)
        return y

    def load_weight_shard(self, weight: torch.Tensor) -> None:
        """Load weight shard with double-shard guard.

        Full weight layout (HF): [gate_full, up_full] concatenated along dim=0.
            gate_full: [intermediate_size, hidden_size] = [12288, 4096]
            up_full:   [intermediate_size, hidden_size] = [12288, 4096]
            full:      [24576, 4096]

        Per-rank shard:
            gate_shard: gate_full[r*per_rank : (r+1)*per_rank, :]  → [3072, 4096]
            up_shard:   up_full[r*per_rank : (r+1)*per_rank, :]    → [3072, 4096]
            merged:     cat([gate_shard, up_shard], dim=0)          → [6144, 4096]

        Args:
            weight: full [2*intermediate_size, hidden_size] or pre-sliced [gate_up_out_dim, hidden_size]
        """
        if weight.shape == self.weight.shape:
            # Already pre-sliced per-rank shard → direct copy
            self.weight.data.copy_(weight)
        else:
            # Full weight → slice gate and up portions, then concatenate
            r = self.tp_rank
            per_rank = self.intermediate_per_rank  # 3072
            inter = self.intermediate_size  # 12288

            gate_shard = weight[r * per_rank : (r + 1) * per_rank, :]  # [3072, 4096]
            up_shard = weight[inter + r * per_rank : inter + (r + 1) * per_rank, :]  # [3072, 4096]
            merged = torch.cat([gate_shard, up_shard], dim=0).contiguous()  # [6144, 4096]
            self.weight.data.copy_(merged)


# ===========================================================================
# QKVColumnParallelLinear — QKV merged [q_size+2*kv_size, hidden], split Q-K-V
# ===========================================================================

class QKVColumnParallelLinear(nn.Module):
    """Merged column-parallel linear for q_proj + k_proj + v_proj.

    Single GEMM replaces three separate projections.
    Weight shape: [q_size + 2*kv_size, hidden_size]
    Output split into (q, k, v) in Q-K-V order (NOT K-Q-V).

    Constructor args:
        hidden_size:        e.g. 4096
        head_dim:           e.g. 128
        total_num_heads:    e.g. 32 (full, pre-TP)
        total_num_kv_heads: e.g. 8 (full, pre-TP)
        gather_output:      if True, all_gather before split (default False)
        tp_size:            tensor parallel size (auto-detect from env if None)
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        gather_output: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        if tp_size is None:
            tp_size = _get_world_size()
        self.tp_size = tp_size
        self.tp_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        # Per-rank head counts
        self.num_heads = total_num_heads // tp_size  # 8
        if total_num_kv_heads >= tp_size:
            self.num_kv_heads = total_num_kv_heads // tp_size  # 2
        else:
            self.num_kv_heads = 1  # replicated, each rank gets 1 head

        # Per-rank output dimensions along dim=-1
        self.q_size = self.num_heads * head_dim        # 1024
        self.kv_size = self.num_kv_heads * head_dim    # 256
        self.out_features_per_rank = self.q_size + 2 * self.kv_size  # 1536

        self.gather_output = gather_output

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, hidden_size, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward: single F.linear + split into Q, K, V.

        x: [B, T, hidden_size] e.g. [1, 1, 4096]

        Returns:
            q: [B, T, q_size]      e.g. [1, 1, 1024]
            k: [B, T, kv_size]     e.g. [1, 1, 256]
            v: [B, T, kv_size]     e.g. [1, 1, 256]
        """
        y = F.linear(x, self.weight)  # [B, T, out_features_per_rank] = [B, T, 1536]

        # all_gather before split (when gather_output=True and tp_size > 1)
        if self.gather_output and self.tp_size > 1:
            y = all_gather_last_dim(y)  # [B, T, total_qkv]

        # Ensure contiguous so split() produces contiguous views.
        # rocBLAS may return non-contiguous output, and split() inherits that.
        y = y.contiguous()

        # Q-K-V split order: Q first, then K, then V (STRICTLY Q-K-V, NOT K-Q-V)
        q, k, v = y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def load_weight_shard(self, weight: torch.Tensor) -> None:
        """Load weight shard with double-shard guard.

        Full weight layout (HF): Q-K-V concatenated along dim=0 in Q-K-V ORDER.
            q_full: [total_num_heads * head_dim, hidden_size] = [4096, 4096]
            k_full: [total_num_kv_heads * head_dim, hidden_size] = [1024, 4096]
            v_full: [total_num_kv_heads * head_dim, hidden_size] = [1024, 4096]
            full: [6144, 4096]

        Per-rank shard (Q-K-V cat order MUST be Q-K-V, never K-Q-V):
            q_shard: q_full[r*q_per_rank : (r+1)*q_per_rank, :]
            k_shard: k_full[r*kv_per_rank : (r+1)*kv_per_rank, :]
            v_shard: v_full[r*kv_per_rank : (r+1)*kv_per_rank, :]
            merged: cat([q_shard, k_shard, v_shard], dim=0)  ← Q-K-V order

        Args:
            weight: full [total_dim, hidden_size] or pre-sliced [out_features_per_rank, hidden_size]
        """
        if weight.shape == self.weight.shape:
            # Already pre-sliced per-rank shard → direct copy
            self.weight.data.copy_(weight)
        else:
            # Full weight → extract Q/K/V shards, concatenate in Q-K-V order
            r = self.tp_rank
            q_per_rank = self.num_heads * self.head_dim
            kv_per_rank = self.num_kv_heads * self.head_dim

            # Q portion: rows [0, total_num_heads*head_dim)
            q_start = r * q_per_rank
            q_shard = weight[q_start : q_start + q_per_rank, :]  # [1024, 4096]

            # K portion: rows [total_num_heads*head_dim, total_num_heads*head_dim + total_num_kv_heads*head_dim)
            k_offset = self.total_num_heads * self.head_dim  # 4096
            k_start = k_offset + r * kv_per_rank
            k_shard = weight[k_start : k_start + kv_per_rank, :]  # [256, 4096]

            # V portion: rows [k_offset + total_num_kv_heads*head_dim, total)
            v_offset = k_offset + self.total_num_kv_heads * self.head_dim  # 4096 + 1024 = 5120
            v_start = v_offset + r * kv_per_rank
            v_shard = weight[v_start : v_start + kv_per_rank, :]  # [256, 4096]

            # STRICT Q-K-V cat order (never K-Q-V or V-K-Q)
            merged = torch.cat([q_shard, k_shard, v_shard], dim=0).contiguous()  # [1536, 4096]
            self.weight.data.copy_(merged)
