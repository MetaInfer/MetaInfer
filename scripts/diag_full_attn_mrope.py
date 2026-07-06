"""Diagnostic: Test FullAttention with MRoPE and partial rotary — the exact
code path from QwenFullAttentionTP but in a single-process TP=4 simulation.

The isolated test (diag_full_attention_tp4.py) passes because it uses standard
full-RoPE. Real Qwen3.6 uses MRoPE with partial_rotary_factor and mrope_section.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.kernels.rotary_embedding import (
    rotary_embedding, make_cos_sin_cache, get_cos_sin_cache,
)
from engine.kernels.attention import flash_attn_varlen_func

# ============================================================================
# Config: Qwen3.6-27B FullAttention with MRoPE
# ============================================================================
HIDDEN = 5120
NUM_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
TP = 4
# MRoPE settings (typical Qwen3.6 values)
# partial_rotary_factor gives rotary_dim
ROTARY_FACTOR = 0.5      # partial rotary: only first 128 of 256 dims
ROTARY_DIM = int(HEAD_DIM * ROTARY_FACTOR)  # 128
# mrope_section: divides rotary_dim/2 into sections for 3D position
# total = rotary_dim/2 = 64
MROPE_SECTION = [16, 24, 24]  # sum = 64
MROPE_INTERLEAVED = True
MAX_POS = 4096
ROPE_THETA = 1000000.0

H_PER_RANK = NUM_HEADS // TP      # 6
KV_PER_RANK = NUM_KV_HEADS // TP  # 1

print(f"=== MRoPE FullAttention Diagnostic ===")
print(f"HEAD_DIM={HEAD_DIM}, ROTARY_DIM={ROTARY_DIM}, rotary_factor={ROTARY_FACTOR}")
print(f"mrope_section={MROPE_SECTION}, sum={sum(MROPE_SECTION)}, rotary_dim/2={ROTARY_DIM//2}")
print(f"Per-rank: q_heads={H_PER_RANK}, kv_heads={KV_PER_RANK}")

torch.manual_seed(42)

# ============================================================================
# Create cos_sin cache with MRoPE
# ============================================================================
cos_sin = make_cos_sin_cache(
    MAX_POS, ROTARY_DIM, ROPE_THETA,
    dtype=torch.bfloat16,
    mrope_section=MROPE_SECTION, mrope_interleaved=MROPE_INTERLEAVED,
)
print(f"cos_sin_cache shape: {list(cos_sin.shape)}")  # [max_pos, rotary_dim]

# ============================================================================
# Create weights (same as isolated test)
# ============================================================================
q_w = torch.randn(NUM_HEADS*HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
k_w = torch.randn(NUM_KV_HEADS*HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
v_w = torch.randn(NUM_KV_HEADS*HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
o_w = torch.randn(HIDDEN, NUM_HEADS*HEAD_DIM, dtype=torch.bfloat16)
gate_w = torch.randn(NUM_HEADS*HEAD_DIM, HIDDEN, dtype=torch.bfloat16)

# ============================================================================
# Reference: TP=1 with all heads, MRoPE
# ============================================================================
B, S = 1, 8
hs = torch.randn(B, S, HIDDEN, dtype=torch.bfloat16)
pos = torch.arange(S, dtype=torch.int64)
N = B * S

print("\n--- Reference (TP=1, all heads, MRoPE) ---")

def manual_qwen35_rms_norm(x, weight):
    effective_w = 1.0 + weight
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (x.float() / rms * effective_w).to(x.dtype)

qn_w = torch.randn(HEAD_DIM)
kn_w = torch.randn(HEAD_DIM)

q = F.linear(hs, q_w)  # [B,S,6144]
k = F.linear(hs, k_w)  # [B,S,1024]
v = F.linear(hs, v_w)  # [B,S,1024]
gate = F.linear(hs, gate_w)

print(f"  q_full: {list(q.shape)}, k_full: {list(k.shape)}")

q = q.view(N, NUM_HEADS, HEAD_DIM)  # [8, 24, 256]
k = k.view(N, NUM_KV_HEADS, HEAD_DIM)  # [8, 4, 256]
v = v.view(N, NUM_KV_HEADS, HEAD_DIM)
gate = gate.view(B, S, NUM_HEADS, HEAD_DIM)

# Q/K norms
q = manual_qwen35_rms_norm(q, qn_w)
k = manual_qwen35_rms_norm(k, kn_w)

# MRoPE: only apply to first ROTARY_DIM
q_rot = q[..., :ROTARY_DIM].contiguous()  # [8, 24, 128]
k_rot = k[..., :ROTARY_DIM].contiguous()  # [8, 4, 128]
rotary_embedding(pos, q_rot, k_rot, ROTARY_DIM, cos_sin, is_neox=True)
q[..., :ROTARY_DIM] = q_rot
k[..., :ROTARY_DIM] = k_rot

# Flash attention (SDPA fallback)
cu = torch.tensor([0, N], dtype=torch.int32)
attn_out = flash_attn_varlen_func(q, k, v, cu, cu, N, N, causal=True, softmax_scale=HEAD_DIM**-0.5)
print(f"  attn_out: {list(attn_out.shape)}")

attn_out = attn_out.view(B, S, NUM_HEADS*HEAD_DIM)
gate_f = gate.view(B, S, NUM_HEADS*HEAD_DIM)
attn_out = attn_out * torch.sigmoid(gate_f)

ref_out = F.linear(attn_out, o_w)
print(f"  ref output norm: {ref_out.float().norm():.4f}")

# ============================================================================
# TP=4 per-rank computation, MRoPE
# ============================================================================
print("\n--- TP=4 per-rank, MRoPE ---")

per_rank_outs = []
for rank in range(TP):
    q_w_local = q_w[rank*H_PER_RANK*HEAD_DIM:(rank+1)*H_PER_RANK*HEAD_DIM, :]
    k_w_local = k_w[rank*KV_PER_RANK*HEAD_DIM:(rank+1)*KV_PER_RANK*HEAD_DIM, :]
    v_w_local = v_w[rank*KV_PER_RANK*HEAD_DIM:(rank+1)*KV_PER_RANK*HEAD_DIM, :]
    o_w_local = o_w[:, rank*H_PER_RANK*HEAD_DIM:(rank+1)*H_PER_RANK*HEAD_DIM]
    gate_w_local = gate_w[rank*H_PER_RANK*HEAD_DIM:(rank+1)*H_PER_RANK*HEAD_DIM, :]

    q = F.linear(hs, q_w_local)  # [B,S,1536]
    k = F.linear(hs, k_w_local)  # [B,S,256]
    v = F.linear(hs, v_w_local)
    gate_local = F.linear(hs, gate_w_local)

    q = q.view(N, H_PER_RANK, HEAD_DIM)  # [8, 6, 256]
    k = k.view(N, KV_PER_RANK, HEAD_DIM)  # [8, 1, 256]
    v = v.view(N, KV_PER_RANK, HEAD_DIM)

    q = manual_qwen35_rms_norm(q, qn_w)
    k = manual_qwen35_rms_norm(k, kn_w)

    # MRoPE: partial rotary
    q_rot = q[..., :ROTARY_DIM].contiguous()
    k_rot = k[..., :ROTARY_DIM].contiguous()
    rotary_embedding(pos, q_rot, k_rot, ROTARY_DIM, cos_sin, is_neox=True)
    q[..., :ROTARY_DIM] = q_rot
    k[..., :ROTARY_DIM] = k_rot

    cu = torch.tensor([0, N], dtype=torch.int32)
    rank_attn = flash_attn_varlen_func(
        q, k, v, cu, cu, N, N, causal=True, softmax_scale=HEAD_DIM**-0.5)

    attn_flat = rank_attn.view(B, S, H_PER_RANK*HEAD_DIM)
    gate_f = gate_local.view(B, S, H_PER_RANK*HEAD_DIM)
    attn_flat = attn_flat * torch.sigmoid(gate_f)

    partial = F.linear(attn_flat, o_w_local)
    per_rank_outs.append(partial)

tp_out = sum(per_rank_outs)

ratio = tp_out.float().norm() / ref_out.float().norm()
max_diff = (tp_out - ref_out).abs().max().item()
print(f"\n--- Comparison ---")
print(f"  TP=4 output norm: {tp_out.float().norm():.4f}")
print(f"  TP=1 ref norm:    {ref_out.float().norm():.4f}")
print(f"  Ratio:            {ratio:.4f}")
print(f"  Max abs diff:     {max_diff:.6f}")

# Also test WITHOUT MRoPE (standard RoPE) to confirm
print("\n--- Standard RoPE (no MRoPE) for comparison ---")
cos_sin_std = make_cos_sin_cache(MAX_POS, HEAD_DIM, ROPE_THETA, dtype=torch.bfloat16)

# TP=1 ref with standard RoPE
q2 = F.linear(hs, q_w).view(N, NUM_HEADS, HEAD_DIM)
k2 = F.linear(hs, k_w).view(N, NUM_KV_HEADS, HEAD_DIM)
v2 = F.linear(hs, v_w).view(N, NUM_KV_HEADS, HEAD_DIM)
q2 = manual_qwen35_rms_norm(q2, qn_w)
k2 = manual_qwen35_rms_norm(k2, kn_w)
rotary_embedding(pos, q2, k2, HEAD_DIM, cos_sin_std, is_neox=True)
attn2 = flash_attn_varlen_func(q2, k2, v2, cu, cu, N, N, causal=True, softmax_scale=HEAD_DIM**-0.5)
attn2 = attn2.view(B, S, NUM_HEADS*HEAD_DIM)
attn2 = attn2 * torch.sigmoid(gate.view(B, S, NUM_HEADS*HEAD_DIM))
ref_std = F.linear(attn2, o_w)

# TP=4 with standard RoPE
per_rank_std = []
for rank in range(TP):
    q_l = F.linear(hs, q_w[rank*1536:(rank+1)*1536, :]).view(N, 6, HEAD_DIM)
    k_l = F.linear(hs, k_w[rank*256:(rank+1)*256, :]).view(N, 1, HEAD_DIM)
    v_l = F.linear(hs, v_w[rank*256:(rank+1)*256, :]).view(N, 1, HEAD_DIM)
    g_l = F.linear(hs, gate_w[rank*1536:(rank+1)*1536, :])
    q_l = manual_qwen35_rms_norm(q_l, qn_w)
    k_l = manual_qwen35_rms_norm(k_l, kn_w)
    rotary_embedding(pos, q_l, k_l, HEAD_DIM, cos_sin_std, is_neox=True)
    attn_l = flash_attn_varlen_func(q_l, k_l, v_l, cu, cu, N, N, causal=True, softmax_scale=HEAD_DIM**-0.5)
    attn_f = attn_l.view(B, S, 1536) * torch.sigmoid(g_l.view(B, S, 1536))
    partial = F.linear(attn_f, o_w[:, rank*1536:(rank+1)*1536])
    per_rank_std.append(partial)

tp_std = sum(per_rank_std)
print(f"  Standard RoPE TP=4/TP=1 ratio: {tp_std.float().norm() / ref_std.float().norm():.4f}")

# Test MRoPE cache vs standard RoPE cache for per-rank = all heads
print("\n--- Per-rank MRoPE value comparison ---")
print(f"  Rank 0 q_rot (first 3 values): {q_rot[0, 0, :3]}")
print(f"  Rank 0 k_rot (first 3 values): {k_rot[0, 0, :3]}")
