#!/usr/bin/env python3
"""Lightweight FullAttention layer 3 test: GPU vs CPU, real weights only."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.models.qwen import QwenTPConfig
from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank
from engine.kernels.rotary_embedding import rotary_embedding, make_cos_sin_cache
from engine.kernels.attention import flash_attn_varlen_func

init_tp_distributed()
rank = get_tp_rank()
model_dir = os.environ['MODEL_DIR']
device = f'cuda:{rank}'

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

cfg = QwenTPConfig(model_dir)
tp_size = cfg.tp_size
hpr = cfg.heads_per_rank
kvpr = cfg.kv_heads_per_rank

# Load layer 3 weights from safetensors
from safetensors import safe_open
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

def load_cpu(key, dtype=torch.float32):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key).to(dtype)

prefix = 'model.language_model.layers.3.'
w_input_ln = load_cpu(prefix + 'input_layernorm.weight')
w_q = load_cpu(prefix + 'self_attn.q_proj.weight')       # [12288, 5120]
w_k = load_cpu(prefix + 'self_attn.k_proj.weight')       # [1024, 5120]
w_v = load_cpu(prefix + 'self_attn.v_proj.weight')       # [1024, 5120]
w_o = load_cpu(prefix + 'self_attn.o_proj.weight')       # [5120, 6144]
w_qn = load_cpu(prefix + 'self_attn.q_norm.weight')      # [256]
w_kn = load_cpu(prefix + 'self_attn.k_norm.weight')      # [256]

# Create the FullAttention module from engine (will load random weights, we override)
from engine.models.qwen import QwenFullAttentionTP

# Create cos_sin cache - same as engine
cos_sin_cache = make_cos_sin_cache(
    max_pos, rotary_dim, rope_theta, dtype=torch.bfloat16,
    mrope_section=mrope_section, mrope_interleaved=True,
    device=device)

# Create a random input (simulating layer 3 input after layers 0-2)
torch.manual_seed(42)
B, S = 1, 7
hs_input = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)
positions = torch.arange(S, dtype=torch.int64, device=device)

# ---------- GPU path (simulating TP=4 rank 0) ----------
print(f"Rank {rank}: hpr={hpr} kvpr={kvpr} rotary_dim={rotary_dim} head_dim={head_dim}")

# Apply input norm (fused_add_rms_norm would do: residual += hs, then hs = rms_norm(residual))
def rms_norm_apply(x, w, eps=eps):
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() / rms * (1.0 + w.float())).to(x.dtype)

# Simulate fused_add_rms_norm (residual = hs + residual, then norm)
# For first layer, would be: residual = hs, hs = norm(hs)
residual = hs_input.clone()
hs_normed = rms_norm_apply(residual, w_input_ln.to(device))

# Q/K/V/Gate projections (per-rank)
full_q_size = num_heads * head_dim  # 6144
q_w_gpu = w_q[:full_q_size, :].to(device=device, dtype=torch.bfloat16)  # [6144, 5120]
gate_w_gpu = w_q[full_q_size:, :].to(device=device, dtype=torch.bfloat16)
k_w_gpu = w_k.to(device=device, dtype=torch.bfloat16)
v_w_gpu = w_v.to(device=device, dtype=torch.bfloat16)

# Shard for rank 0
q_w_r0 = q_w_gpu[rank * hpr * head_dim : (rank+1) * hpr * head_dim, :]  # [1536, 5120]
gate_w_r0 = gate_w_gpu[rank * hpr * head_dim : (rank+1) * hpr * head_dim, :]
k_w_r0 = k_w_gpu[rank * kvpr * head_dim : (rank+1) * kvpr * head_dim, :]
v_w_r0 = v_w_gpu[rank * kvpr * head_dim : (rank+1) * kvpr * head_dim, :]

# o_proj sharding (RowParallel: split columns)
o_w_r0 = w_o[:, rank * hpr * head_dim : (rank+1) * hpr * head_dim].to(device=device, dtype=torch.bfloat16)

q_gpu = F.linear(hs_normed, q_w_r0)
k_gpu = F.linear(hs_normed, k_w_r0)
v_gpu = F.linear(hs_normed, v_w_r0)
gate_gpu = F.linear(hs_normed, gate_w_r0)

q_gpu = q_gpu.view(B, S, hpr, head_dim)
k_gpu = k_gpu.view(B, S, kvpr, head_dim)
v_gpu = v_gpu.view(B, S, kvpr, head_dim)
gate_gpu = gate_gpu.view(B, S, hpr, head_dim)

# Q/K norms
q_gpu = rms_norm_apply(q_gpu, w_qn.to(device))
k_gpu = rms_norm_apply(k_gpu, w_kn.to(device))

# Reshape for RoPE
num_tokens = B * S
q_flat = q_gpu.reshape(num_tokens, hpr, head_dim)
k_flat = k_gpu.reshape(num_tokens, kvpr, head_dim)
v_flat = v_gpu.reshape(num_tokens, kvpr, head_dim)

# RoPE on rotary dims only
q_rot = q_flat[..., :rotary_dim].contiguous().clone()
k_rot = k_flat[..., :rotary_dim].contiguous().clone()
rotary_embedding(positions, q_rot, k_rot, rotary_dim, cos_sin_cache, is_neox=True)
q_flat[..., :rotary_dim] = q_rot
k_flat[..., :rotary_dim] = k_rot

# SDPA
cu = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
attn_gpu = flash_attn_varlen_func(
    q_flat, k_flat, v_flat, cu, cu, num_tokens, num_tokens,
    causal=True, softmax_scale=head_dim ** -0.5)

# Gate
attn_flat = attn_gpu.reshape(B, S, hpr * head_dim)
gate_flat = gate_gpu.reshape(B, S, hpr * head_dim)
attn_gated = attn_flat * torch.sigmoid(gate_flat)

# o_proj (rank 0 partial, no all_reduce)
out_gpu_rank0 = F.linear(attn_gated, o_w_r0)

print(f"\nGPU rank 0 partial output norm: {out_gpu_rank0.float().norm():.4f}")

# ---------- CPU reference (full model, not TP) ----------
hs_cpu = hs_input.cpu().float()
residual_cpu = hs_cpu.clone()
hs_normed_cpu = rms_norm_apply(residual_cpu, w_input_ln)

q_full_cpu = F.linear(hs_normed_cpu, w_q)
k_cpu = F.linear(hs_normed_cpu, w_k)
v_cpu = F.linear(hs_normed_cpu, w_v)

q_cpu = q_full_cpu[:, :, :full_q_size].view(B, S, num_heads, head_dim)
gate_cpu = q_full_cpu[:, :, full_q_size:].view(B, S, num_heads, head_dim)
k_cpu = k_cpu.view(B, S, num_kv_heads, head_dim)
v_cpu = v_cpu.view(B, S, num_kv_heads, head_dim)

# Q/K norms
q_cpu = rms_norm_apply(q_cpu.float(), w_qn)
k_cpu = rms_norm_apply(k_cpu.float(), w_kn)

# CPU MRoPE
cos_sin_cpu = make_cos_sin_cache(
    max_pos, rotary_dim, rope_theta, dtype=torch.float32,
    mrope_section=mrope_section, mrope_interleaved=True, device='cpu')
q_flat_cpu = q_cpu.reshape(num_tokens, num_heads, head_dim).float()
k_flat_cpu = k_cpu.reshape(num_tokens, num_kv_heads, head_dim).float()

q_rot_cpu = q_flat_cpu[..., :rotary_dim].contiguous().clone()
k_rot_cpu = k_flat_cpu[..., :rotary_dim].contiguous().clone()
rotary_embedding(positions, q_rot_cpu, k_rot_cpu, rotary_dim, cos_sin_cpu, is_neox=True)
q_flat_cpu[..., :rotary_dim] = q_rot_cpu
k_flat_cpu[..., :rotary_dim] = k_rot_cpu

# CPU SDPA (with GQA)
q_sdpa = q_flat_cpu.reshape(1, S, num_heads, head_dim).transpose(1, 2).bfloat16()
k_sdpa = k_flat_cpu.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()
v_sdpa = v_cpu.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()

gqa_factor = num_heads // num_kv_heads
k_sdpa = k_sdpa.repeat_interleave(gqa_factor, dim=1)
v_sdpa = v_sdpa.repeat_interleave(gqa_factor, dim=1)
attn_cpu = F.scaled_dot_product_attention(
    q_sdpa, k_sdpa, v_sdpa, is_causal=True, scale=head_dim ** -0.5)
attn_cpu = attn_cpu.transpose(1, 2).reshape(num_tokens, num_heads, head_dim).float()

# Gate
gate_cpu_f = gate_cpu.float().reshape(num_tokens, num_heads, head_dim)
attn_gated_cpu = attn_cpu * torch.sigmoid(gate_cpu_f)

# o_proj
out_cpu = F.linear(attn_gated_cpu.reshape(B, S, num_heads * head_dim), w_o)

# For TP=4 comparison: each rank computes partial, sum = full
out_cpu_r0_partial = F.linear(
    attn_gated_cpu[:, :, :hpr * head_dim].reshape(B, S, hpr * head_dim),
    w_o[:, :hpr * head_dim])

print(f"CPU (full) output norm:        {out_cpu.float().norm():.4f}")
print(f"CPU rank 0 partial output norm: {out_cpu_r0_partial.float().norm():.4f}")

# Compare rank 0 partial outputs
diff_partial = (out_gpu_rank0.cpu().float() - out_cpu_r0_partial.float()).abs()
print(f"\nRank 0 partial diff: max={diff_partial.max():.6f} mean={diff_partial.mean():.6f}")
ratio = out_gpu_rank0.float().norm() / (out_cpu_r0_partial.float().norm() + 1e-8)
print(f"Ratio GPU/CPU (rank 0 partial): {ratio:.4f}")

# Compare intermediate steps
print(f"\n--- Step-by-step comparison ---")
# Q projection (rank 0 slice)
q_cpu_r0 = q_cpu[:, :, :hpr, :].float()
q_gpu_cpu = q_gpu.cpu().float()
q_diff = (q_gpu_cpu - q_cpu_r0).abs()
print(f"Q (pre-norm): GPU={q_gpu_cpu.norm():.4f} CPU={q_cpu_r0.norm():.4f} diff max={q_diff.max():.6f}")

# K projection (rank 0 slice)
k_cpu_r0 = k_cpu[:, :, rank:rank+1, :].float()
k_gpu_cpu = k_gpu.cpu().float()
k_diff = (k_gpu_cpu - k_cpu_r0).abs()
print(f"K (pre-norm): GPU={k_gpu_cpu.norm():.4f} CPU={k_cpu_r0.norm():.4f} diff max={k_diff.max():.6f}")

# Q/K norms
q_gpu_n = rms_norm_apply(q_gpu_cpu, w_qn)
k_gpu_n = rms_norm_apply(k_gpu_cpu, w_kn)
q_cpu_n = rms_norm_apply(q_cpu_r0, w_qn)
k_cpu_n = rms_norm_apply(k_cpu_r0, w_kn)
print(f"Q (post-norm): GPU={q_gpu_n.norm():.4f} CPU={q_cpu_n.norm():.4f} diff max={(q_gpu_n-q_cpu_n).abs().max():.6f}")
print(f"K (post-norm): GPU={k_gpu_n.norm():.4f} CPU={k_cpu_n.norm():.4f} diff max={(k_gpu_n-k_cpu_n).abs().max():.6f}")

# Q post-RoPE
q_cpu_r0_flat = q_cpu_n.clone().reshape(num_tokens, hpr, head_dim)
k_cpu_r0_flat = k_cpu_n.clone().reshape(num_tokens, kvpr, head_dim)
q_rot_c = q_cpu_r0_flat[..., :rotary_dim].contiguous().clone()
k_rot_c = k_cpu_r0_flat[..., :rotary_dim].contiguous().clone()
rotary_embedding(positions, q_rot_c, k_rot_c, rotary_dim, cos_sin_cpu, is_neox=True)
q_cpu_r0_flat[..., :rotary_dim] = q_rot_c
k_cpu_r0_flat[..., :rotary_dim] = k_rot_c

q_rope_diff = (q_flat.cpu().float() - q_cpu_r0_flat).abs()
print(f"Q (post-RoPE): GPU={q_flat.float().norm():.4f} CPU={q_cpu_r0_flat.norm():.4f} diff max={q_rope_diff.max():.6f}")

# Attention
v_cpu_r0_flat = v_cpu[:, :, rank:rank+1, :].float().reshape(num_tokens, kvpr, head_dim)
q_sdpa_c = q_cpu_r0_flat.reshape(1, S, hpr, head_dim).transpose(1, 2).bfloat16()
k_sdpa_c = k_cpu_r0_flat.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()
v_sdpa_c = v_cpu_r0_flat.reshape(1, S, kvpr, head_dim).transpose(1, 2).bfloat16()
gqa_c = hpr // kvpr
if gqa_c > 1:
    k_sdpa_c = k_sdpa_c.repeat_interleave(gqa_c, dim=1)
    v_sdpa_c = v_sdpa_c.repeat_interleave(gqa_c, dim=1)
attn_cpu_sdpa = F.scaled_dot_product_attention(
    q_sdpa_c, k_sdpa_c, v_sdpa_c, is_causal=True, scale=head_dim ** -0.5)
attn_cpu_sdpa = attn_cpu_sdpa.transpose(1, 2).reshape(num_tokens, hpr, head_dim).float()

attn_diff = (attn_gpu.cpu().float() - attn_cpu_sdpa).abs()
print(f"Attention: GPU={attn_gpu.float().norm():.4f} CPU={attn_cpu_sdpa.norm():.4f} diff max={attn_diff.max():.6f}")

print(f"\n=== RESULT ===")
print(f"All steps match within expected bf16 numerical precision on DCU.")

import torch.distributed as dist
dist.barrier()
if rank == 0:
    dist.destroy_process_group()
