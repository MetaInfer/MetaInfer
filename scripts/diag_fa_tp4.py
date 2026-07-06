#!/usr/bin/env python3
"""Test TP=4 FullAttention: compare o_proj all_reduce with expected full computation."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank, get_tp_size
from engine.kernels.rotary_embedding import rotary_embedding, make_cos_sin_cache

init_tp_distributed()
rank = get_tp_rank()
tp_size = get_tp_size()
model_dir = os.environ['MODEL_DIR']
device = f'cuda:{rank}'

# Skip if not TP=4
if tp_size != 4:
    print(f"SKIP: expected TP_SIZE=4, got {tp_size}")
    import torch.distributed as dist
    if rank == 0:
        dist.destroy_process_group()
    sys.exit(0)

# Config
with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)
eps = tc['rms_norm_eps']
head_dim = tc['head_dim']
num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']
hidden_size = tc['hidden_size']
rp = tc.get('rope_parameters', tc.get('rope_scaling', {})) or {}
rotary_dim = int(head_dim * rp.get('partial_rotary_factor', 1.0))
mrope_section = rp.get('mrope_section')
rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)
max_pos = tc['max_position_embeddings']

hpr = num_heads // tp_size  # 6
kvpr = num_kv_heads // tp_size  # 1
full_q_size = num_heads * head_dim  # 6144

# Load weights
from safetensors import safe_open
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

prefix = 'model.language_model.layers.3.'
w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()
w_q = load_cpu(prefix + 'self_attn.q_proj.weight').float()       # [12288, 5120]
w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()       # [1024, 5120]
w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()       # [1024, 5120]
w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()       # [5120, 6144]
w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()      # [256]
w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()      # [256]

cos_sin_gpu = make_cos_sin_cache(
    max_pos, rotary_dim, rope_theta, dtype=torch.bfloat16,
    mrope_section=mrope_section, mrope_interleaved=True, device=device)

def rms_norm_apply(x, w, eps=eps):
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() / rms * (1.0 + w.float())).to(x.dtype)

torch.manual_seed(42)
B, S = 1, 7
num_tokens = B * S
hs = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
positions = torch.arange(S, dtype=torch.int64, device=device)

# Norm
residual = hs.clone()
hs_normed = rms_norm_apply(residual, w_input_ln.to(device))

# Q/K/V sharding (same as ColumnParallelLinear)
q_start = rank * hpr * head_dim
q_end = q_start + hpr * head_dim
q_w = w_q[:full_q_size, :][q_start:q_end, :].to(device=device, dtype=torch.bfloat16)
gate_start = full_q_size + rank * hpr * head_dim
gate_end = gate_start + hpr * head_dim
gate_w = w_q[gate_start:gate_end, :].to(device=device, dtype=torch.bfloat16)
k_start = rank * kvpr * head_dim
k_end = k_start + kvpr * head_dim
k_w = w_k[k_start:k_end, :].to(device=device, dtype=torch.bfloat16)
v_w = w_v[k_start:k_end, :].to(device=device, dtype=torch.bfloat16)

q = F.linear(hs_normed, q_w)
k = F.linear(hs_normed, k_w)
v = F.linear(hs_normed, v_w)
gate = F.linear(hs_normed, gate_w)

q = q.view(B, S, hpr, head_dim)
k = k.view(B, S, kvpr, head_dim)
v = v.view(B, S, kvpr, head_dim)
gate = gate.view(B, S, hpr, head_dim)

# Q/K norms
q = rms_norm_apply(q, w_qn.to(device))
k = rms_norm_apply(k, w_kn.to(device))

# Reshape for RoPE
q_flat = q.reshape(num_tokens, hpr, head_dim).clone()
k_flat = k.reshape(num_tokens, kvpr, head_dim).clone()
v_flat = v.reshape(num_tokens, kvpr, head_dim).clone()

# RoPE
q_rot = q_flat[..., :rotary_dim].contiguous().clone()
k_rot = k_flat[..., :rotary_dim].contiguous().clone()
rotary_embedding(positions, q_rot, k_rot, rotary_dim, cos_sin_gpu, is_neox=True)
q_flat[..., :rotary_dim] = q_rot
k_flat[..., :rotary_dim] = k_rot

# SDPA (per-rank, with GQA)
q_sdpa = q_flat.reshape(1, S, hpr, head_dim).transpose(1, 2)
k_sdpa = k_flat.reshape(1, S, kvpr, head_dim).transpose(1, 2)
v_sdpa = v_flat.reshape(1, S, kvpr, head_dim).transpose(1, 2)

gqa_factor = hpr // kvpr
if gqa_factor > 1:
    k_sdpa = k_sdpa.repeat_interleave(gqa_factor, dim=1)
    v_sdpa = v_sdpa.repeat_interleave(gqa_factor, dim=1)

attn = F.scaled_dot_product_attention(
    q_sdpa, k_sdpa, v_sdpa, is_causal=True, scale=head_dim ** -0.5)
attn = attn.transpose(1, 2).reshape(B, S, hpr * head_dim)

# Gate
gate_f = gate.reshape(B, S, hpr * head_dim)
attn_gated = attn * torch.sigmoid(gate_f)

# o_proj (RowParallel: split columns, each rank computes partial)
o_start = rank * hpr * head_dim
o_end = o_start + hpr * head_dim
o_w = w_o[:, o_start:o_end].to(device=device, dtype=torch.bfloat16)

partial_out = F.linear(attn_gated, o_w)

# all_reduce_sum (simulating what RowParallelLinear does)
from engine.tp_layers.distributed import all_reduce_sum
tp_out = all_reduce_sum(partial_out)

print(f"Rank {rank}: partial_out norm={partial_out.float().norm():.4f}, tp_out norm={tp_out.float().norm():.4f}")

# ---- CPU reference (full model, not TP) ----
# Compute full attention on CPU for comparison
hs_cpu = hs.cpu().float()
hs_normed_cpu = rms_norm_apply(hs_cpu.clone(), w_input_ln)

q_full_cpu = F.linear(hs_normed_cpu, w_q)
q_cpu = q_full_cpu[:, :, :full_q_size].view(B, S, num_heads, head_dim).float()
gate_cpu = q_full_cpu[:, :, full_q_size:].view(B, S, num_heads, head_dim).float()
k_cpu = F.linear(hs_normed_cpu, w_k).view(B, S, num_kv_heads, head_dim).float()
v_cpu = F.linear(hs_normed_cpu, w_v).view(B, S, num_kv_heads, head_dim).float()

q_cpu = rms_norm_apply(q_cpu, w_qn)
k_cpu = rms_norm_apply(k_cpu, w_kn)

q_flat_c = q_cpu.reshape(num_tokens, num_heads, head_dim).clone()
k_flat_c = k_cpu.reshape(num_tokens, num_kv_heads, head_dim).clone()

cos_sin_cpu = cos_sin_gpu.cpu().float()
positions_cpu = torch.arange(S, dtype=torch.int64)
q_rot_c = q_flat_c[..., :rotary_dim].contiguous().clone()
k_rot_c = k_flat_c[..., :rotary_dim].contiguous().clone()
rotary_embedding(positions_cpu, q_rot_c, k_rot_c, rotary_dim, cos_sin_cpu, is_neox=True)
q_flat_c[..., :rotary_dim] = q_rot_c
k_flat_c[..., :rotary_dim] = k_rot_c

q_sdpa_c = q_flat_c.reshape(1, S, num_heads, head_dim).transpose(1, 2).bfloat16()
k_sdpa_c = k_flat_c.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()
v_sdpa_c = v_cpu.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()

gqa_full = num_heads // num_kv_heads
k_sdpa_c = k_sdpa_c.repeat_interleave(gqa_full, dim=1)
v_sdpa_c = v_sdpa_c.repeat_interleave(gqa_full, dim=1)
attn_cpu = F.scaled_dot_product_attention(
    q_sdpa_c, k_sdpa_c, v_sdpa_c, is_causal=True, scale=head_dim ** -0.5)
attn_cpu = attn_cpu.transpose(1, 2).reshape(B, S, num_heads * head_dim).float()

gate_cpu_f = gate_cpu.reshape(B, S, num_heads * head_dim).float()
attn_gated_cpu = attn_cpu * torch.sigmoid(gate_cpu_f)

out_cpu = F.linear(attn_gated_cpu, w_o)

if rank == 0:
    diff = (tp_out.cpu().float() - out_cpu).abs()
    ratio = tp_out.float().norm() / (out_cpu.float().norm() + 1e-8)
    print(f"\nCPU full output norm: {out_cpu.float().norm():.4f}")
    print(f"TP=4 output norm: {tp_out.float().norm():.4f}")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Ratio GPU/CPU: {ratio:.4f}")

    if diff.max() < 1.0:
        print("PASS: TP=4 FullAttention matches CPU reference")
    else:
        print("FAIL: TP=4 FullAttention diverges from CPU reference")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
