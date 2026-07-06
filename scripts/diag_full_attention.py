#!/usr/bin/env python3
"""Diagnostic: check FullAttention layer values in detail. TP=4.

Focus: Layer 3 (first full_attention) where residual_norm jumps from ~56 to ~217.
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F, math
sys.path.insert(0, os.environ.get('AGENT_INFER_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.tp_layers.linear import ColumnParallelLinear, RowParallelLinear
from engine.kernels.rms_norm import rms_norm
from engine.kernels.rotary_embedding import rotary_embedding, get_cos_sin_cache
from engine.kernels.attention import flash_attn_varlen_func
from engine.models.qwen import QwenTPConfig, Qwen3_5RMSNorm

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']

cfg = QwenTPConfig(model_dir)
print(f"[Rank {rank}] TP={cfg.tp_size}, hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}, head_dim={cfg.head_dim}")
print(f"[Rank {rank}] heads_per_rank={cfg.heads_per_rank}, kv_heads_per_rank={cfg.kv_heads_per_rank}")
print(f"[Rank {rank}] rotary_dim={cfg.rotary_dim}, mrope_section={cfg.mrope_section}, mrope_interleaved={cfg.mrope_interleaved}")
print(f"[Rank {rank}] attn_output_gate={cfg.attn_output_gate}")

# Load a single FullAttention layer's weights (layer 3)
import json, struct
from safetensors import safe_open

index_path = os.path.join(model_dir, 'model.safetensors.index.json')
with open(index_path) as f:
    index = json.load(f)

weight_map = index['weight_map']
layer_prefix = 'model.layers.3.self_attn.'

def load_weight(key):
    """Load a single weight tensor from safetensors."""
    fname = weight_map.get(key)
    if fname is None:
        print(f"[Rank {rank}] WARNING: key not found: {key}")
        return None
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

# === Load FullAttention weights for layer 3 ===
print(f"\n[Rank {rank}] === Loading FullAttention layer 3 weights ===")

q_proj_w = load_weight(layer_prefix + 'q_proj.weight')  # [2*num_heads*head_dim, hidden] = [12288, 5120]
k_proj_w = load_weight(layer_prefix + 'k_proj.weight')  # [1024, 5120]
v_proj_w = load_weight(layer_prefix + 'v_proj.weight')  # [1024, 5120]
o_proj_w = load_weight(layer_prefix + 'o_proj.weight')  # [hidden, num_heads*head_dim] = [5120, 6144]
q_norm_w = load_weight(layer_prefix + 'q_norm.weight')  # [head_dim] = [256]
k_norm_w = load_weight(layer_prefix + 'k_norm.weight')  # [head_dim] = [256]

# Shard weights per TP rank
tp_size = cfg.tp_size
tp_rank = rank

# ColumnParallel: shard output dim (dim 0)
total_q_out = q_proj_w.shape[0]  # 12288
total_kv_out = k_proj_w.shape[0]  # 1024
q_per_rank = total_q_out // tp_size
kv_per_rank = total_kv_out // tp_size
q_start = tp_rank * q_per_rank
kv_start = tp_rank * kv_per_rank

q_proj_w_local = q_proj_w[q_start:q_start+q_per_rank, :].clone()
k_proj_w_local = k_proj_w[kv_start:kv_start+kv_per_rank, :].clone()
v_proj_w_local = v_proj_w[kv_start:kv_start+kv_per_rank, :].clone()

# RowParallel: shard input dim (dim 1)
total_o_in = o_proj_w.shape[1]  # 6144
o_per_rank = total_o_in // tp_size
o_start = tp_rank * o_per_rank
o_proj_w_local = o_proj_w[:, o_start:o_start+o_per_rank].clone()

print(f"[Rank {rank}] q_proj weight local: {q_proj_w_local.shape} (norm={q_proj_w_local.norm().item():.4f})")
print(f"[Rank {rank}] k_proj weight local: {k_proj_w_local.shape} (norm={k_proj_w_local.norm().item():.4f})")
print(f"[Rank {rank}] v_proj weight local: {v_proj_w_local.shape} (norm={v_proj_w_local.norm().item():.4f})")
print(f"[Rank {rank}] o_proj weight local: {o_proj_w_local.shape} (norm={o_proj_w_local.norm().item():.4f})")
print(f"[Rank {rank}] q_norm weight: shape={q_norm_w.shape}, values={q_norm_w[:5].tolist()}...")
print(f"[Rank {rank}] k_norm weight: shape={k_norm_w.shape}, values={k_norm_w[:5].tolist()}...")

# Check if RMSNorm weights are zeros (Qwen3_5RMSNorm: weight=zeros, effective=(1+w))
print(f"[Rank {rank}] q_norm zeros: {torch.allclose(q_norm_w, torch.zeros_like(q_norm_w))}")
print(f"[Rank {rank}] k_norm zeros: {torch.allclose(k_norm_w, torch.zeros_like(k_norm_w))}")

# === Test with a simple input ===
test_input = torch.randn(1, 4, cfg.hidden_size, dtype=torch.bfloat16)  # [1, 4, 5120]
print(f"\n[Rank {rank}] === Running layer 3 with test input [1, 4, 5120] ===")
print(f"[Rank {rank}] Input norm: {test_input.norm().item():.4f}")

# Step 1: QKV projections (manual, using loaded weights)
q_full = F.linear(test_input, q_proj_w_local)  # [1, 4, 3072]
k = F.linear(test_input, k_proj_w_local)  # [1, 4, 256]
v = F.linear(test_input, v_proj_w_local)  # [1, 4, 256]

num_heads = cfg.heads_per_rank  # 6
num_kv_heads = cfg.kv_heads_per_rank  # 1
head_dim = cfg.head_dim  # 256

# Split q and gate (attn_output_gate doubles q output)
q_size = num_heads * head_dim  # 1536
q, gate = torch.chunk(q_full, 2, dim=-1)  # q=[1,4,1536], gate=[1,4,1536]

q = q.view(1, 4, num_heads, head_dim)
k = k.view(1, 4, num_kv_heads, head_dim)
v = v.view(1, 4, num_kv_heads, head_dim)

print(f"[Rank {rank}] After projections: q={q.shape}, k={k.shape}, v={v.shape}")
print(f"[Rank {rank}] q norm={q.norm().item():.4f}, k norm={k.norm().item():.4f}, v norm={v.norm().item():.4f}")
print(f"[Rank {rank}] gate norm={gate.norm().item():.4f}, sigmoid(gate) range=[{torch.sigmoid(gate).min().item():.4f}, {torch.sigmoid(gate).max().item():.4f}]")

# Step 2: Q/K norms (Qwen3_5RMSNorm: weight=zeros, effective=(1+w))
q_norm_eff = 1.0 + q_norm_w  # zeros → ones → weight=1.0 essentially
k_norm_eff = 1.0 + k_norm_w

# Apply RMSNorm manually per head
q_out = torch.empty_like(q)
for h in range(num_heads):
    w_h = q_norm_eff[h*head_dim:(h+1)*head_dim] if q_norm_eff.shape[0] >= (h+1)*head_dim else q_norm_eff
    rms_norm(q_out[:,:,h,:], q[:,:,h,:].contiguous(), w_h, 1e-6)

k_out = torch.empty_like(k)
for h in range(num_kv_heads):
    w_h = k_norm_eff[h*head_dim:(h+1)*head_dim] if k_norm_eff.shape[0] >= (h+1)*head_dim else k_norm_eff
    rms_norm(k_out[:,:,h,:], k[:,:,h,:].contiguous(), w_h, 1e-6)

print(f"[Rank {rank}] After Q/K norms: q norm={q_out.norm().item():.4f}, k norm={k_out.norm().item():.4f}")

# Step 3: MRoPE
num_tokens = 4
q_flat = q_out.reshape(num_tokens, num_heads, head_dim)
k_flat = k_out.reshape(num_tokens, num_kv_heads, head_dim)

# Get cos/sin cache
cos_sin = get_cos_sin_cache(
    cfg.max_position_embeddings, cfg.rotary_dim, cfg.rope_theta,
    dtype=torch.bfloat16,
    mrope_section=cfg.mrope_section, mrope_interleaved=cfg.mrope_interleaved)

rotary_dim = cfg.rotary_dim
positions = torch.arange(4, dtype=torch.int64)

q_rot = q_flat[..., :rotary_dim].contiguous()
k_rot = k_flat[..., :rotary_dim].contiguous()
rotary_embedding(positions, q_rot, k_rot, rotary_dim, cos_sin, is_neox=True)
q_flat[..., :rotary_dim] = q_rot
k_flat[..., :rotary_dim] = k_rot

print(f"[Rank {rank}] After MRoPE: q norm={q_flat.norm().item():.4f}, k norm={k_flat.norm().item():.4f}")

# Step 4: SDPA Attention
# Prepare inputs for flash_attn_varlen_func
cu_q = torch.tensor([0, 4], dtype=torch.int32)
cu_k = torch.tensor([0, 4], dtype=torch.int32)
scaling = head_dim ** -0.5

try:
    attn_out = flash_attn_varlen_func(
        q_flat, k_flat, v.view(num_tokens, num_kv_heads, head_dim),
        cu_q, cu_k, num_tokens, num_tokens,
        causal=True, softmax_scale=scaling)
    print(f"[Rank {rank}] FlashAttention output norm: {attn_out.norm().item():.4f}")
except Exception as e:
    print(f"[Rank {rank}] FlashAttention FAILED: {e}")
    print(f"[Rank {rank}] Falling back to manual SDPA...")
    # Manual SDPA fallback
    q_sdpa = q_flat.transpose(0, 1).unsqueeze(0)  # [1, 6, 4, 256]
    k_sdpa = k_flat.transpose(0, 1).unsqueeze(0)  # [1, 1, 4, 256]
    v_sdpa = v.view(num_tokens, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)  # [1, 1, 4, 256]

    # Expand K/V for GQA
    k_sdpa = k_sdpa.expand(-1, num_heads, -1, -1)
    v_sdpa = v_sdpa.expand(-1, num_heads, -1, -1)

    attn_weights = torch.matmul(q_sdpa, k_sdpa.transpose(-2, -1)) * scaling  # [1, 6, 4, 4]
    causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_out_sdpa = torch.matmul(attn_weights, v_sdpa)  # [1, 6, 4, 256]
    attn_out = attn_out_sdpa.squeeze(0).transpose(0, 1)  # [4, 6, 256]
    print(f"[Rank {rank}] Manual SDPA output norm: {attn_out.norm().item():.4f}")
    print(f"[Rank {rank}] Attention weights max: {attn_weights.max().item():.6f}, min: {attn_weights.min().item():.6f}")

# Step 5: Output gate
attn_out_2d = attn_out.reshape(1, 4, num_heads * head_dim)  # [1, 4, 1536]
gated = attn_out_2d * torch.sigmoid(gate)
print(f"[Rank {rank}] After output gate: norm={gated.norm().item():.4f}")

# Step 6: o_proj
o_proj_out = F.linear(gated, o_proj_w_local)  # [1, 4, 5120]
print(f"[Rank {rank}] o_proj output norm: {o_proj_out.norm().item():.4f}")

# All-reduce across TP ranks
import torch.distributed as dist
o_proj_all = o_proj_out.clone()
dist.all_reduce(o_proj_all, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"\n[Rank {rank}] === Final o_proj (all-reduced) output norm: {o_proj_all.norm().item():.4f} ===")

# Check the intermediate values more: what is the attention weight distribution?
print(f"\n[Rank {rank}] === Summary of FullAttention layer 3 diagnostics ===")
print(f"[Rank {rank}] Input → q_proj norm: {q.norm().item():.4f}")
print(f"[Rank {rank}] Q after norm: {q_flat.norm().item():.4f}")
print(f"[Rank {rank}] Attention output (raw): {attn_out.norm().item():.4f}")
print(f"[Rank {rank}] Gated attn output: {gated.norm().item():.4f}")
print(f"[Rank {rank}] o_proj output (pre-reduce): {o_proj_out.norm().item():.4f}")
if rank == 0:
    print(f"[Rank {rank}] o_proj output (post-reduce): {o_proj_all.norm().item():.4f}")

# Also check: what do the q_norm/k_norm weights actually look like?
print(f"\n[Rank {rank}] === Q/K Norm weight details ===")
print(f"[Rank {rank}] q_norm weight min={q_norm_w.min().item():.6f}, max={q_norm_w.max().item():.6f}, mean={q_norm_w.mean().item():.6f}")
print(f"[Rank {rank}] q_norm effective weight min={q_norm_eff.min().item():.6f}, max={q_norm_eff.max().item():.6f}")
print(f"[Rank {rank}] k_norm effective weight min={k_norm_eff.min().item():.6f}, max={k_norm_eff.max().item():.6f}")

# Check q_norm weight distribution more carefully
q_eff = q_norm_eff.reshape(-1, head_dim)
for h in range(min(6, q_eff.shape[0])):
    print(f"[Rank {rank}] q_norm effective head {h}: min={q_eff[h].min().item():.6f}, max={q_eff[h].max().item():.6f}, mean={q_eff[h].mean().item():.6f}")

dist.barrier()
if rank == 0:
    print("\n[DONE] FullAttention layer 3 diagnostic complete")
