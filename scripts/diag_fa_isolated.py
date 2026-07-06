#!/usr/bin/env python3
"""Isolate FullAttention TP computation — compare tp=1 full vs tp=4 sharded."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We simulate the FullAttention forward manually, testing each stage
head_dim = 256
num_heads = 24
num_kv_heads = 4
hidden_size = 5120
tp_size = 4
heads_per_rank = num_heads // tp_size  # 6
kv_heads_per_rank = max(1, num_kv_heads // tp_size)  # 1
B, S = 1, 8  # batch=1, seq=8

device = 'cuda:0'
dtype = torch.bfloat16

# Create random weights
torch.manual_seed(42)

# Full weights (tp=1)
q_weight_full = torch.randn(num_heads * head_dim, hidden_size, dtype=dtype, device=device)
k_weight_full = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype, device=device)
v_weight_full = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype, device=device)
o_weight_full = torch.randn(hidden_size, num_heads * head_dim, dtype=dtype, device=device)

# Sharded weights (tp=4, rank 0)
q_weight_shard = q_weight_full[:heads_per_rank * head_dim]  # [1536, 5120]
k_weight_shard = k_weight_full[:kv_heads_per_rank * head_dim]  # [256, 5120]
v_weight_shard = v_weight_full[:kv_heads_per_rank * head_dim]  # [256, 5120]
o_weight_shard = o_weight_full[:, :heads_per_rank * head_dim]  # [5120, 1536]

# Input
x = torch.randn(B, S, hidden_size, dtype=dtype, device=device)

# === TP=1 full computation ===
with torch.inference_mode():
    q_full = F.linear(x, q_weight_full)  # [B,S,6144]
    k_full = F.linear(x, k_weight_full)  # [B,S,1024]
    v_full = F.linear(x, v_weight_full)  # [B,S,1024]

    q_full = q_full.view(B, S, num_heads, head_dim)  # [B,S,24,256]
    k_full = k_full.view(B, S, num_kv_heads, head_dim)  # [B,S,4,256]
    v_full = v_full.view(B, S, num_kv_heads, head_dim)  # [B,S,4,256]

    q_flat = q_full.reshape(B*S, num_heads, head_dim)  # [S,24,256]
    k_flat = k_full.reshape(B*S, num_kv_heads, head_dim)  # [S,4,256]
    v_flat = v_full.reshape(B*S, num_kv_heads, head_dim)  # [S,4,256]

    # SDPA
    attn_out = F.scaled_dot_product_attention(
        q_flat.transpose(0,1).unsqueeze(0),  # [1,24,S,256]
        k_flat.transpose(0,1).unsqueeze(0),  # [1,4,S,256]
        v_flat.transpose(0,1).unsqueeze(0),  # [1,4,S,256]
        is_causal=True,
    )
    attn_out = attn_out.squeeze(0).transpose(0,1)  # [S,24,256]
    attn_out = attn_out.reshape(B, S, num_heads * head_dim)  # [B,S,6144]

    # o_proj
    output_full = F.linear(attn_out, o_weight_full)  # [B,S,5120]

print(f"TP=1 full output: shape={output_full.shape}, norm={output_full.norm().item():.4f}")

# === TP=4 sharded computation (single-rank simulation) ===
import torch.nn.functional as F

with torch.inference_mode():
    q_shard = F.linear(x, q_weight_shard)  # [B,S,1536]
    k_shard = F.linear(x, k_weight_shard)  # [B,S,256]
    v_shard = F.linear(x, v_weight_shard)  # [B,S,256]

    print(f"q_shard shape: {q_shard.shape}, expected [B,S,{heads_per_rank * head_dim}]")
    print(f"k_shard shape: {k_shard.shape}, expected [B,S,{kv_heads_per_rank * head_dim}]")

    q_shard = q_shard.view(B, S, heads_per_rank, head_dim)  # [B,S,6,256]
    k_shard = k_shard.view(B, S, kv_heads_per_rank, head_dim)  # [B,S,1,256]
    v_shard = v_shard.view(B, S, kv_heads_per_rank, head_dim)  # [B,S,1,256]

    q_flat_s = q_shard.reshape(B*S, heads_per_rank, head_dim)  # [S,6,256]
    k_flat_s = k_shard.reshape(B*S, kv_heads_per_rank, head_dim)  # [S,1,256]
    v_flat_s = v_shard.reshape(B*S, kv_heads_per_rank, head_dim)  # [S,1,256]

    # SDPA with GQA (shard-level)
    attn_out_s = F.scaled_dot_product_attention(
        q_flat_s.transpose(0,1).unsqueeze(0),  # [1,6,S,256]
        k_flat_s.transpose(0,1).unsqueeze(0),  # [1,1,S,256]
        v_flat_s.transpose(0,1).unsqueeze(0),  # [1,1,S,256]
        is_causal=True,
    )
    attn_out_s = attn_out_s.squeeze(0).transpose(0,1)  # [S,6,256]
    attn_out_s = attn_out_s.reshape(B, S, heads_per_rank * head_dim)  # [B,S,1536]

    # o_proj per-rank
    partial_output = F.linear(attn_out_s, o_weight_shard)  # [B,S,5120]

    # Simulate all_reduce_sum by computing all 4 ranks' outputs
    all_partial_outputs = []
    for r in range(tp_size):
        q_w_r = q_weight_full[r*heads_per_rank*head_dim:(r+1)*heads_per_rank*head_dim]
        k_w_r = k_weight_full[r*kv_heads_per_rank*head_dim:(r+1)*kv_heads_per_rank*head_dim] if r < num_kv_heads else k_weight_full[:kv_heads_per_rank*head_dim]
        v_w_r = v_weight_full[r*kv_heads_per_rank*head_dim:(r+1)*kv_heads_per_rank*head_dim] if r < num_kv_heads else v_weight_full[:kv_heads_per_rank*head_dim]
        o_w_r = o_weight_full[:, r*heads_per_rank*head_dim:(r+1)*heads_per_rank*head_dim]

        q_r = F.linear(x, q_w_r).view(B, S, heads_per_rank, head_dim).reshape(B*S, heads_per_rank, head_dim)
        k_r = F.linear(x, k_w_r).view(B, S, kv_heads_per_rank, head_dim).reshape(B*S, kv_heads_per_rank, head_dim)
        v_r = F.linear(x, v_w_r).view(B, S, kv_heads_per_rank, head_dim).reshape(B*S, kv_heads_per_rank, head_dim)

        attn_r = F.scaled_dot_product_attention(
            q_r.transpose(0,1).unsqueeze(0),
            k_r.transpose(0,1).unsqueeze(0),
            v_r.transpose(0,1).unsqueeze(0),
            is_causal=True,
        ).squeeze(0).transpose(0,1).reshape(B, S, heads_per_rank * head_dim)

        partial_r = F.linear(attn_r, o_w_r)
        all_partial_outputs.append(partial_r)

    output_sharded = torch.stack(all_partial_outputs).sum(dim=0)  # all_reduce_sum

print(f"TP=4 sharded output: shape={output_sharded.shape}, norm={output_sharded.norm().item():.4f}")
print(f"Difference norm: {(output_full - output_sharded).norm().item():.6f}")
print(f"Relative diff: {((output_full - output_sharded).norm() / output_full.norm()).item():.6f}")

# Also test: what if we DON'T all_reduce the sharded output?
print(f"\nWithout all_reduce (partial rank 0): norm={all_partial_outputs[0].norm().item():.4f}")
print(f"Ratio full/partial: {output_full.norm().item() / all_partial_outputs[0].norm().item():.4f}")
print(f"Ratio full/summed: {output_full.norm().item() / output_sharded.norm().item():.4f}")
