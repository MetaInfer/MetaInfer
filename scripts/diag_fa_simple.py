#!/usr/bin/env python3
"""Simple FullAttention comparison: GPU (SDPA) vs CPU reference, real weights, TP=1."""
import os, sys, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

from engine.kernels.rotary_embedding import rotary_embedding, make_cos_sin_cache

model_dir = os.environ['MODEL_DIR']

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

print(f"Config: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
print(f"MRoPE: rotary_dim={rotary_dim}, mrope_section={mrope_section}")

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

device = 'cuda:0'
full_q_size = num_heads * head_dim  # 6144

# Create cos_sin caches (CPU + GPU)
cos_sin_cpu = make_cos_sin_cache(
    max_pos, rotary_dim, rope_theta, dtype=torch.float32,
    mrope_section=mrope_section, mrope_interleaved=True, device='cpu')
cos_sin_gpu = cos_sin_cpu.to(device=device, dtype=torch.bfloat16)

def rms_norm_apply(x, w, eps=eps):
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() / rms * (1.0 + w.float())).to(x.dtype)

# Random input
torch.manual_seed(42)
B, S = 1, 7
num_tokens = B * S
hs = torch.randn(B, S, hidden_size, dtype=torch.bfloat16)
positions_gpu = torch.arange(S, dtype=torch.int64, device=device)
positions_cpu = torch.arange(S, dtype=torch.int64)

# ===== GPU PATH =====
residual_gpu = hs.clone().to(device)
hs_normed_gpu = rms_norm_apply(residual_gpu, w_input_ln.to(device))

q_full_gpu = F.linear(hs_normed_gpu, w_q[:full_q_size, :].to(device=device, dtype=torch.bfloat16))
gate_full_gpu = F.linear(hs_normed_gpu, w_q[full_q_size:, :].to(device=device, dtype=torch.bfloat16))
k_gpu = F.linear(hs_normed_gpu, w_k.to(device=device, dtype=torch.bfloat16))
v_gpu = F.linear(hs_normed_gpu, w_v.to(device=device, dtype=torch.bfloat16))

q_gpu = q_full_gpu.view(B, S, num_heads, head_dim)
k_gpu = k_gpu.view(B, S, num_kv_heads, head_dim)
v_gpu = v_gpu.view(B, S, num_kv_heads, head_dim)
gate_gpu = gate_full_gpu.view(B, S, num_heads, head_dim)

q_gpu = rms_norm_apply(q_gpu, w_qn.to(device))
k_gpu = rms_norm_apply(k_gpu, w_kn.to(device))

q_flat = q_gpu.reshape(num_tokens, num_heads, head_dim).clone()
k_flat = k_gpu.reshape(num_tokens, num_kv_heads, head_dim).clone()
v_flat = v_gpu.reshape(num_tokens, num_kv_heads, head_dim).clone()

# RoPE (GPU)
q_rot = q_flat[..., :rotary_dim].contiguous().clone()
k_rot = k_flat[..., :rotary_dim].contiguous().clone()
rotary_embedding(positions_gpu, q_rot, k_rot, rotary_dim, cos_sin_gpu, is_neox=True)
q_flat[..., :rotary_dim] = q_rot
k_flat[..., :rotary_dim] = k_rot

# SDPA (GPU, with GQA)
q_sdpa = q_flat.reshape(1, S, num_heads, head_dim).transpose(1, 2)
k_sdpa = k_flat.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2)
v_sdpa = v_flat.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2)

gqa_factor = num_heads // num_kv_heads
k_sdpa = k_sdpa.repeat_interleave(gqa_factor, dim=1)
v_sdpa = v_sdpa.repeat_interleave(gqa_factor, dim=1)

attn_gpu = F.scaled_dot_product_attention(
    q_sdpa, k_sdpa, v_sdpa, is_causal=True, scale=head_dim ** -0.5)
attn_gpu = attn_gpu.transpose(1, 2).reshape(B, S, num_heads * head_dim)

# Gate
gate_f = gate_gpu.reshape(B, S, num_heads * head_dim)
attn_gated_gpu = attn_gpu * torch.sigmoid(gate_f)

# o_proj
out_gpu = F.linear(attn_gated_gpu, w_o.to(device=device, dtype=torch.bfloat16))

print(f"GPU output norm: {out_gpu.float().norm():.4f}")

# ===== CPU PATH =====
residual_cpu = hs.clone().float()
hs_normed_cpu = rms_norm_apply(residual_cpu, w_input_ln)

q_full_cpu = F.linear(hs_normed_cpu, w_q)
q_cpu = q_full_cpu[:, :, :full_q_size].view(B, S, num_heads, head_dim)
gate_cpu = q_full_cpu[:, :, full_q_size:].view(B, S, num_heads, head_dim)
k_cpu = F.linear(hs_normed_cpu, w_k).view(B, S, num_kv_heads, head_dim)
v_cpu = F.linear(hs_normed_cpu, w_v).view(B, S, num_kv_heads, head_dim)

q_cpu = rms_norm_apply(q_cpu.float(), w_qn)
k_cpu = rms_norm_apply(k_cpu.float(), w_kn)

q_flat_c = q_cpu.reshape(num_tokens, num_heads, head_dim).float().clone()
k_flat_c = k_cpu.reshape(num_tokens, num_kv_heads, head_dim).float().clone()

# RoPE (CPU — use CPU positions!)
q_rot_c = q_flat_c[..., :rotary_dim].contiguous().clone()
k_rot_c = k_flat_c[..., :rotary_dim].contiguous().clone()
rotary_embedding(positions_cpu, q_rot_c, k_rot_c, rotary_dim, cos_sin_cpu, is_neox=True)
q_flat_c[..., :rotary_dim] = q_rot_c
k_flat_c[..., :rotary_dim] = k_rot_c

# SDPA (CPU, with GQA)
q_sdpa_c = q_flat_c.reshape(1, S, num_heads, head_dim).transpose(1, 2).bfloat16()
k_sdpa_c = k_flat_c.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()
v_sdpa_c = v_cpu.reshape(1, S, num_kv_heads, head_dim).transpose(1, 2).bfloat16()

k_sdpa_c = k_sdpa_c.repeat_interleave(gqa_factor, dim=1)
v_sdpa_c = v_sdpa_c.repeat_interleave(gqa_factor, dim=1)
attn_cpu = F.scaled_dot_product_attention(
    q_sdpa_c, k_sdpa_c, v_sdpa_c, is_causal=True, scale=head_dim ** -0.5)
attn_cpu = attn_cpu.transpose(1, 2).reshape(B, S, num_heads * head_dim).float()

# Gate
gate_cpu_f = gate_cpu.float().reshape(B, S, num_heads * head_dim)
attn_gated_cpu = attn_cpu * torch.sigmoid(gate_cpu_f)

# o_proj
out_cpu = F.linear(attn_gated_cpu, w_o)

print(f"CPU output norm: {out_cpu.float().norm():.4f}")

# ===== COMPARISON =====
diff = (out_gpu.cpu().float() - out_cpu).abs()
ratio = out_gpu.float().norm() / (out_cpu.float().norm() + 1e-8)
print(f"\nMax diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"Ratio GPU/CPU: {ratio:.4f}")

if diff.max() < 1.0:
    print("PASS: FullAttention computation matches CPU reference")
else:
    print("FAIL: Significant divergence")

# Step-by-step check
print(f"\n--- Q projection ---")
q_diff = (q_full_gpu.cpu().float()[:, :, :] - q_full_cpu[:, :, :full_q_size]).abs()
print(f"Q max diff: {q_diff.max():.6f}")

print(f"\n--- Q/K norms ---")
qn_diff = (q_gpu.cpu().float() - q_cpu).abs()
kn_diff = (k_gpu.cpu().float() - k_cpu).abs()
print(f"Q_norm max diff: {qn_diff.max():.6f}")
print(f"K_norm max diff: {kn_diff.max():.6f}")

print(f"\n--- RoPE ---")
# Compare GPU vs CPU RoPE output on CPU positions (recompute CPU with same method)
# The rotary_embedding manual fallback is used for both paths
q_rope_diff = (q_flat.cpu().float() - q_flat_c).abs()
k_rope_diff = (k_flat.cpu().float() - k_flat_c).abs()
print(f"Q RoPE max diff: {q_rope_diff.max():.6f}")
print(f"K RoPE max diff: {k_rope_diff.max():.6f}")

print(f"\n--- Attention ---")
attn_diff = (attn_gpu.cpu().float() - attn_cpu).abs()
print(f"Attention max diff: {attn_diff.max():.6f}")

print(f"\n--- Gate ---")
gate_diff = (attn_gated_gpu.cpu().float() - attn_gated_cpu).abs()
print(f"Gate max diff: {gate_diff.max():.6f}")
