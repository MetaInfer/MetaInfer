"""Diagnostic: Isolate FullAttention TP=4 forward correctness.

Compares the FullAttention layer output using TP=4 (per-rank heads) vs
a reference TP=1 computation (all heads on one rank).

This isolates the attention computation from the residual chain and weight loading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os, math

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29500')

# Import engine modules
from engine.tp_layers.distributed import init_tp_distributed, get_tp_size, get_tp_rank, all_reduce_sum, get_tp_group
from engine.tp_layers.linear import ColumnParallelLinear, RowParallelLinear
from engine.kernels.rms_norm import rms_norm
from engine.kernels.rotary_embedding import rotary_embedding, make_cos_sin_cache

# Import flash_attn (SDPA fallback)
from engine.kernels.attention import flash_attn_varlen_func, flash_attn_with_kvcache

# ============================================================================
# Config matching Qwen3.6-27B FullAttention
# ============================================================================
HIDDEN_SIZE = 5120
NUM_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
TP_SIZE = 4

# Per-rank heads
HEADS_PER_RANK = NUM_HEADS // TP_SIZE  # 6
KV_HEADS_PER_RANK = NUM_KV_HEADS // TP_SIZE  # 1

print(f"=== FullAttention TP={TP_SIZE} Diagnostic ===")
print(f"HIDDEN_SIZE={HIDDEN_SIZE}, NUM_HEADS={NUM_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, HEAD_DIM={HEAD_DIM}")
print(f"Per-rank: q_heads={HEADS_PER_RANK}, kv_heads={KV_HEADS_PER_RANK}")

# ============================================================================
# Step 1: Create Qwen3_5RMSNorm (same as FullAttention uses)
# ============================================================================
class DiagRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def _effective_weight(self):
        return 1.0 + self.weight

    def forward(self, x):
        out = torch.empty_like(x)
        rms_norm(out, x.contiguous(), self._effective_weight(), self.eps)
        return out

# ============================================================================
# Step 2: Create reference weights (random, same for all "ranks" in reference)
# ============================================================================
torch.manual_seed(42)

# Full weights
q_weight_full = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)  # [6144, 5120]
k_weight_full = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)  # [1024, 5120]
v_weight_full = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)  # [1024, 5120]
o_weight_full = torch.randn(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16)  # [5120, 6144]

# Gate weight (same shape as q for attn_output_gate)
gate_weight_full = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, dtype=torch.bfloat16)  # [6144, 5120]

# Norm weights
q_norm_weight = torch.randn(HEAD_DIM)  # [256]
k_norm_weight = torch.randn(HEAD_DIM)  # [256]

# RoPE cos/sin cache
cos_sin_cache = make_cos_sin_cache(4096, HEAD_DIM, 1000000.0, dtype=torch.bfloat16)

# ============================================================================
# Step 3: Random input
# ============================================================================
B, S = 1, 8  # 8-token sequence
hidden_states_ref = torch.randn(B, S, HIDDEN_SIZE, dtype=torch.bfloat16)
positions_ref = torch.arange(S, dtype=torch.int64)

# ============================================================================
# Step 4: Reference computation (TP=1, all heads together, no communication)
# ============================================================================
print("\n--- Step 4: Reference TP=1 computation ---")

def ref_forward(hs, positions, cos_sin):
    """Reference: all heads, no TP partitioning."""
    B, S, _ = hs.shape
    num_tokens = B * S

    # Q projection
    q = F.linear(hs, q_weight_full)  # [B,S,6144]
    k = F.linear(hs, k_weight_full)  # [B,S,1024]
    v = F.linear(hs, v_weight_full)  # [B,S,1024]
    gate = F.linear(hs, gate_weight_full)  # [B,S,6144]

    print(f"  q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    print(f"  gate shape: {gate.shape}")

    # Reshape to [B,S,H,D]
    q = q.view(B, S, NUM_HEADS, HEAD_DIM)
    k = k.view(B, S, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(B, S, NUM_KV_HEADS, HEAD_DIM)
    gate = gate.view(B, S, NUM_HEADS, HEAD_DIM)

    # Q/K norms (per-head, applied to last dim)
    # Use manual RMS norm for reference
    def manual_rms_norm(x, weight, eps=1e-6):
        effective_w = 1.0 + weight
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        return (x.float() / rms * effective_w).to(x.dtype)

    q = q.view(B*S, NUM_HEADS, HEAD_DIM)
    k = k.view(B*S, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(B*S, NUM_KV_HEADS, HEAD_DIM)

    q = manual_rms_norm(q, q_norm_weight)
    k = manual_rms_norm(k, k_norm_weight)

    # RoPE
    rotary_embedding(positions_ref, q, k, HEAD_DIM, cos_sin, is_neox=True)

    # Flash attention / SDPA fallback
    cu = torch.tensor([0, num_tokens], dtype=torch.int32)
    attn_out = flash_attn_varlen_func(
        q, k, v, cu, cu, num_tokens, num_tokens,
        causal=True, softmax_scale=HEAD_DIM**-0.5)
    # attn_out: [num_tokens, NUM_HEADS, HEAD_DIM] = [8, 24, 256]

    print(f"  attn_out shape: {attn_out.shape}")

    attn_out = attn_out.view(B, S, NUM_HEADS * HEAD_DIM)  # [B,S,6144]

    # Output gate
    gate_flat = gate.view(B, S, NUM_HEADS * HEAD_DIM)
    attn_out = attn_out * torch.sigmoid(gate_flat)

    # o_proj
    out = F.linear(attn_out, o_weight_full)  # [B,S,5120]

    print(f"  ref output shape: {out.shape}")
    print(f"  ref output norm: {out.float().norm():.4f}")

    return out

ref_out = ref_forward(hidden_states_ref, positions_ref, cos_sin_cache)

# ============================================================================
# Step 5: TP=4 per-rank computation (simulated)
# ============================================================================
print("\n--- Step 5: TP=4 per-rank computation ---")

# Split weights per rank
per_rank_outs = []
for rank in range(TP_SIZE):
    print(f"\n  Rank {rank}:")

    # ColumnParallel: split along dim=0 (rows for ColumnParallel)
    q_start = rank * HEADS_PER_RANK * HEAD_DIM
    q_end = q_start + HEADS_PER_RANK * HEAD_DIM
    q_weight_local = q_weight_full[q_start:q_end, :].clone()  # [1536, 5120]

    k_start = rank * KV_HEADS_PER_RANK * HEAD_DIM
    k_end = k_start + KV_HEADS_PER_RANK * HEAD_DIM
    k_weight_local = k_weight_full[k_start:k_end, :].clone()  # [256, 5120]
    v_weight_local = v_weight_full[k_start:k_end, :].clone()  # [256, 5120]

    gate_weight_local = gate_weight_full[q_start:q_end, :].clone()  # [1536, 5120]

    # RowParallel: split along dim=1 (columns for RowParallel)
    o_start = rank * HEADS_PER_RANK * HEAD_DIM
    o_end = o_start + HEADS_PER_RANK * HEAD_DIM
    o_weight_local = o_weight_full[:, o_start:o_end].clone()  # [5120, 1536]

    print(f"    q_weight_local: {list(q_weight_local.shape)}")
    print(f"    k_weight_local: {list(k_weight_local.shape)}")
    print(f"    o_weight_local: {list(o_weight_local.shape)}")

    # Forward (same computation as QwenFullAttentionTP.forward, sans all_reduce)
    hs = hidden_states_ref.clone()
    B, S, _ = hs.shape
    num_tokens = B * S

    q = F.linear(hs, q_weight_local)  # [B,S,1536]
    k = F.linear(hs, k_weight_local)  # [B,S,256]
    v = F.linear(hs, v_weight_local)  # [B,S,256]
    gate = F.linear(hs, gate_weight_local)  # [B,S,1536]

    print(f"    q local: {list(q.shape)}, k local: {list(k.shape)}")

    q = q.view(B, S, HEADS_PER_RANK, HEAD_DIM)
    k = k.view(B, S, KV_HEADS_PER_RANK, HEAD_DIM)
    v = v.view(B, S, KV_HEADS_PER_RANK, HEAD_DIM)
    gate = gate.view(B, S, HEADS_PER_RANK, HEAD_DIM)

    # Q/K norms
    q = q.reshape(num_tokens, HEADS_PER_RANK, HEAD_DIM)
    k = k.reshape(num_tokens, KV_HEADS_PER_RANK, HEAD_DIM)

    def manual_rms_norm_local(x, weight, eps=1e-6):
        effective_w = 1.0 + weight
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        return (x.float() / rms * effective_w).to(x.dtype)

    q = manual_rms_norm_local(q, q_norm_weight)
    k = manual_rms_norm_local(k, k_norm_weight)

    # RoPE
    rotary_embedding(positions_ref, q, k, HEAD_DIM, cos_sin, is_neox=True)

    # Flash attention (SPDA fallback with GQA)
    # q: [num_tokens, 6, 256], k: [num_tokens, 1, 256], v: [num_tokens, 1, 256]
    cu = torch.tensor([0, num_tokens], dtype=torch.int32)
    attn_out = flash_attn_varlen_func(
        q, k, v, cu, cu, num_tokens, num_tokens,
        causal=True, softmax_scale=HEAD_DIM**-0.5)
    # attn_out: [num_tokens, HEADS_PER_RANK, HEAD_DIM] = [8, 6, 256]

    print(f"    attn_out local: {list(attn_out.shape)}")

    attn_out = attn_out.view(B, S, HEADS_PER_RANK * HEAD_DIM)  # [B,S,1536]

    # Output gate
    gate_flat = gate.view(B, S, HEADS_PER_RANK * HEAD_DIM)
    attn_out = attn_out * torch.sigmoid(gate_flat)

    # o_proj (without all_reduce)
    partial_out = F.linear(attn_out, o_weight_local)  # [B,S,5120]

    print(f"    partial_out shape: {list(partial_out.shape)}")
    print(f"    partial_out norm: {partial_out.float().norm():.4f}")

    per_rank_outs.append(partial_out)

# Sum across ranks (simulating all_reduce_sum)
tp_out = sum(per_rank_outs)

print(f"\n--- Step 6: Comparison ---")
print(f"  TP=4 output norm: {tp_out.float().norm():.4f}")
print(f"  TP=1 ref  norm:   {ref_out.float().norm():.4f}")
print(f"  Ratio:            {tp_out.float().norm() / ref_out.float().norm():.4f}")

max_diff = (tp_out - ref_out).abs().max().item()
print(f"  Max abs diff:     {max_diff:.6f}")

# Check if close
if max_diff < 0.1:
    print("\nRESULT: PASS - TP=4 matches TP=1 reference")
else:
    ratio = tp_out.float().norm() / ref_out.float().norm()
    print(f"\nRESULT: MISMATCH - TP=4 output is {ratio:.2f}x vs reference")

    # Check if the mismatch is from attention or o_proj
    print("\n--- Step 7: Isolating the mismatch source ---")

    # Check per-rank attention output (before o_proj)
    for rank in range(TP_SIZE):
        hs = hidden_states_ref.clone()
        q = F.linear(hs, q_weight_full[rank*HEADS_PER_RANK*HEAD_DIM:(rank+1)*HEADS_PER_RANK*HEAD_DIM, :])
        k = F.linear(hs, k_weight_full[rank*KV_HEADS_PER_RANK*HEAD_DIM:(rank+1)*KV_HEADS_PER_RANK*HEAD_DIM, :])
        v = F.linear(hs, v_weight_full[rank*KV_HEADS_PER_RANK*HEAD_DIM:(rank+1)*KV_HEADS_PER_RANK*HEAD_DIM, :])
        gate = F.linear(hs, gate_weight_full[rank*HEADS_PER_RANK*HEAD_DIM:(rank+1)*HEADS_PER_RANK*HEAD_DIM, :])

        q = q.view(B, S, HEADS_PER_RANK, HEAD_DIM).reshape(-1, HEADS_PER_RANK, HEAD_DIM)
        k = k.view(B, S, KV_HEADS_PER_RANK, HEAD_DIM).reshape(-1, KV_HEADS_PER_RANK, HEAD_DIM)
        v = v.view(B, S, KV_HEADS_PER_RANK, HEAD_DIM).reshape(-1, KV_HEADS_PER_RANK, HEAD_DIM)

        q = manual_rms_norm_local(q, q_norm_weight)
        k = manual_rms_norm_local(k, k_norm_weight)

        rotary_embedding(positions_ref, q, k, HEAD_DIM, cos_sin, is_neox=True)

        cu = torch.tensor([0, num_tokens], dtype=torch.int32)
        rank_attn = flash_attn_varlen_func(
            q, k, v, cu, cu, num_tokens, num_tokens,
            causal=True, softmax_scale=HEAD_DIM**-0.5)

        print(f"  Rank {rank} attention output norm: {rank_attn.float().norm():.4f}")
        print(f"    shape: {list(rank_attn.shape)}")

    # Compare o_proj with reference
    attn_all_ranks = []
    for rank in range(TP_SIZE):
        hs = hidden_states_ref.clone()
        q = F.linear(hs, q_weight_full[rank*1536:(rank+1)*1536, :])
        k = F.linear(hs, k_weight_full[rank*256:(rank+1)*256, :])
        v = F.linear(hs, v_weight_full[rank*256:(rank+1)*256, :])
        gate = F.linear(hs, gate_weight_full[rank*1536:(rank+1)*1536, :])

        q = q.view(B, S, 6, 256).reshape(-1, 6, 256)
        k = k.view(B, S, 1, 256).reshape(-1, 1, 256)
        v = v.view(B, S, 1, 256).reshape(-1, 1, 256)

        q = manual_rms_norm_local(q, q_norm_weight)
        k = manual_rms_norm_local(k, k_norm_weight)
        rotary_embedding(positions_ref, q, k, HEAD_DIM, cos_sin, is_neox=True)

        cu = torch.tensor([0, num_tokens], dtype=torch.int32)
        attn = flash_attn_varlen_func(q, k, v, cu, cu, num_tokens, num_tokens, causal=True, softmax_scale=HEAD_DIM**-0.5)
        attn = attn.view(B, S, 1536)
        gate_f = gate.view(B, S, 1536)
        attn = attn * torch.sigmoid(gate_f)
        attn_all_ranks.append(attn)

    # Concatenate all rank attention outputs
    attn_combined = torch.cat(attn_all_ranks, dim=-1)  # [B,S,6144]
    o_proj_ref = F.linear(attn_combined, o_weight_full)  # [B,S,5120]
    print(f"\n  o_proj(ref attn combined): norm = {o_proj_ref.float().norm():.4f}")

    # o_proj per rank (without all_reduce)
    o_proj_per_rank_sum = sum(
        F.linear(attn_all_ranks[r], o_weight_full[:, r*1536:(r+1)*1536])
        for r in range(TP_SIZE)
    )
    print(f"  o_proj(per-rank sum):       norm = {o_proj_per_rank_sum.float().norm():.4f}")
    print(f"  Difference: {(o_proj_ref - o_proj_per_rank_sum).abs().max():.6f}")
