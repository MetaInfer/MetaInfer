#!/usr/bin/env python3
"""Isolated FullAttention layer test — single GPU, no TP.

Compare our FullAttention output with a pure-PyTorch reference.
"""
import os, sys, torch, json, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

with open(os.path.join(model_dir, 'config.json')) as f:
    cfg_raw = json.load(f)
tc = cfg_raw.get('text_config', cfg_raw)

hidden_size = tc['hidden_size']
num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']
head_dim = tc['head_dim']
eps = tc['rms_norm_eps']
rotary_dim = int(head_dim * tc['rope_parameters']['partial_rotary_factor'])
rope_theta = tc['rope_parameters']['rope_theta']
mrope_section = tc['rope_parameters']['mrope_section']
mrope_interleaved = tc['rope_parameters']['mrope_interleaved']

# Load weights
index_path = os.path.join(model_dir, 'model.safetensors.index.json')
with open(index_path) as f:
    index = json.load(f)
weight_map = index['weight_map']

def load_raw(key):
    fname = weight_map[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

prefix = 'model.language_model.layers.3.self_attn.'
q_proj_w = load_raw(prefix + 'q_proj.weight')
k_proj_w = load_raw(prefix + 'k_proj.weight')
v_proj_w = load_raw(prefix + 'v_proj.weight')
o_proj_w = load_raw(prefix + 'o_proj.weight')
q_norm_w = load_raw(prefix + 'q_norm.weight')
k_norm_w = load_raw(prefix + 'k_norm.weight')
q_norm_eff = (1.0 + q_norm_w).to(torch.float32)
k_norm_eff = (1.0 + k_norm_w).to(torch.float32)

# Test input
torch.manual_seed(42)
dtype = torch.bfloat16
S = 4
test_input = torch.randn(1, S, hidden_size, dtype=dtype)
positions = torch.arange(S, dtype=torch.int64)

print(f"Input norm: {test_input.norm():.4f}")

# ==== REFERENCE: Pure PyTorch implementation ====

# 1. Projections
q_full = F.linear(test_input.float(), q_proj_w.float())
k = F.linear(test_input.float(), k_proj_w.float())
v = F.linear(test_input.float(), v_proj_w.float())

q_all, gate_all = torch.chunk(q_full, 2, dim=-1)
q_all = q_all.view(1, S, num_heads, head_dim)
k = k.view(1, S, num_kv_heads, head_dim)
v = v.view(1, S, num_kv_heads, head_dim)

gate_all = gate_all.view(1, S, num_heads, head_dim)

# 2. Q/K RMSNorm
q_rstd = 1.0 / torch.sqrt(q_all.float().pow(2).mean(-1, keepdim=True) + eps)
k_rstd = 1.0 / torch.sqrt(k.float().pow(2).mean(-1, keepdim=True) + eps)
q_normed = (q_all.float() * q_rstd * q_norm_eff)
k_normed = (k.float() * k_rstd * k_norm_eff)

# 3. MRoPE using existing rotary_embedding
from engine.kernels.rotary_embedding import make_cos_sin_cache, rotary_embedding

cos_sin = make_cos_sin_cache(
    262144, rotary_dim, rope_theta, dtype=torch.float32,
    mrope_section=mrope_section, mrope_interleaved=mrope_interleaved)

q_flat = q_normed.reshape(S, num_heads, head_dim).contiguous()
k_flat = k_normed.reshape(S, num_kv_heads, head_dim).contiguous()

q_rot = q_flat[..., :rotary_dim].contiguous()
k_rot = k_flat[..., :rotary_dim].contiguous()
rotary_embedding(positions, q_rot, k_rot, rotary_dim, cos_sin, is_neox=True)
q_flat[..., :rotary_dim] = q_rot
k_flat[..., :rotary_dim] = k_rot

q_roped = q_flat.reshape(1, S, num_heads, head_dim)
k_roped = k_flat.reshape(1, S, num_kv_heads, head_dim)

print(f"After MRoPE: q norm={q_roped.norm():.4f}, k norm={k_roped.norm():.4f}")

# 4. SDPA attention
import torch.nn.functional as F
scaling = head_dim ** -0.5
gqa_factor = num_heads // num_kv_heads

q_sdpa = q_roped.transpose(1, 2)  # [1, 24, S, 256]
k_sdpa = k_roped.transpose(1, 2)  # [1, 4, S, 256]
v_sdpa = v.transpose(1, 2)         # [1, 4, S, 256]

k_sdpa_expanded = k_sdpa.repeat_interleave(gqa_factor, dim=1)  # [1, 24, S, 256]
v_sdpa_expanded = v_sdpa.repeat_interleave(gqa_factor, dim=1)  # [1, 24, S, 256]

attn_scores = torch.matmul(q_sdpa, k_sdpa_expanded.transpose(-2, -1)) * scaling  # [1, 24, 4, 4]
causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
attn_probs = F.softmax(attn_scores, dim=-1)

attn_out = torch.matmul(attn_probs, v_sdpa_expanded)  # [1, 24, S, 256]
attn_out = attn_out.transpose(1, 2)  # [1, S, 24, 256]

print(f"Attention probs: max={attn_probs.max():.4f}, entropy={(-attn_probs*(attn_probs+1e-10).log()).sum(-1).mean():.4f}")

# 5. Output gate
gated = attn_out.float() * torch.sigmoid(gate_all.float())

# 6. o_proj
gated_flat = gated.reshape(1, S, num_heads * head_dim)
output_ref = F.linear(gated_flat.float(), o_proj_w.float())

print(f"REFERENCE FullAttention output norm: {output_ref.norm():.4f}")
print(f"  Per-token output norms: {output_ref.squeeze(0).norm(dim=-1)}")

# ==== COMPARISON: Use our model's FullAttention layer ====
print("\n=== Now running our FullAttentionTP code ===")
from engine.models.qwen import QwenTPConfig, QwenFullAttentionTP
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size

# We need to init TP to create the model
init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
print(f"TP rank {rank}/{tp_size}")

cfg = QwenTPConfig(model_dir)
layer = QwenFullAttentionTP(cfg, 3)

# Load weights into the model
with torch.no_grad():
    # q_proj (ColumnParallel, shard dim 0)
    q_per = q_proj_w.shape[0] // tp_size
    q_start = rank * q_per
    layer.q_proj.weight.data.copy_(q_proj_w[q_start:q_start+q_per, :].to(layer.q_proj.weight.dtype))

    # k_proj
    kv_per = k_proj_w.shape[0] // tp_size
    kv_start = rank * kv_per
    layer.k_proj.weight.data.copy_(k_proj_w[kv_start:kv_start+kv_per, :].to(layer.k_proj.weight.dtype))

    # v_proj
    layer.v_proj.weight.data.copy_(v_proj_w[kv_start:kv_start+kv_per, :].to(layer.v_proj.weight.dtype))

    # o_proj (RowParallel, shard dim 1)
    o_per = o_proj_w.shape[1] // tp_size
    o_start = rank * o_per
    layer.o_proj.weight.data.copy_(o_proj_w[:, o_start:o_start+o_per].to(layer.o_proj.weight.dtype))

    # Q/K norms
    layer.q_norm.weight.data.copy_(q_norm_w.to(layer.q_norm.weight.dtype))
    layer.k_norm.weight.data.copy_(k_norm_w.to(layer.k_norm.weight.dtype))

# Run our Forward
B = 1
x = test_input.clone()
with torch.no_grad():
    our_output = layer(x.to(f'cuda:{rank}'), positions.to(f'cuda:{rank}'), S)

# Our output is per-rank sharded, so it's [1, S, 5120/tp=1280] for the RowParallel
# After all_reduce on o_proj, it should be [1, S, 5120]. But we're running single-gpu here,
# so we need to check if RowParallel works with tp_size=1.

print(f"Our model output shape: {our_output.shape}")
print(f"Our model output norm: {our_output.cpu().norm():.4f}")

# Since TP=1, we should compare directly
if tp_size == 1:
    # Compare with reference
    diff = (our_output.cpu().float() - output_ref).norm()
    print(f"\nDifference norm: {diff:.4f}")
    print(f"Cosine similarity: {F.cosine_similarity(our_output.cpu().float().reshape(-1), output_ref.reshape(-1), dim=0):.6f}")
else:
    print(f"\nTP>1, our output is sharded. Full comparison not possible here.")

import torch.distributed as dist
if dist.is_initialized():
    dist.barrier()
    if rank == 0:
        dist.destroy_process_group()
