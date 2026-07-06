#!/usr/bin/env python3
"""Step 2: CPU FullAttention computation using GPU's layer 3 input.

Loads the exact normed hidden state from GPU, computes FullAttention on CPU
using safetensors weights, and compares with GPU layer 3 output.
Saves detailed intermediate values for substep comparison.
"""
import os, sys, torch, json, math
import torch.nn.functional as F

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

# Load GPU input
gpu_data = torch.load('/tmp/diag_gpu_layer3_input.pt', map_location='cpu', weights_only=True)
hs_work = gpu_data['hs_work'].float()  # Normed input to FullAttention: [1, 7, 5120]
hs_out_gpu = gpu_data['hs_out_gpu'].float()  # GPU layer 3 output: [1, 7, 5120]
tokens = gpu_data['tokens']
S = gpu_data['S']
B, T = 1, S

print(f"GPU layer 3 mlp_out norm: {hs_out_gpu.norm():.4f}")

# Load config
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
print(f"MRoPE: rotary_dim={rotary_dim}, mrope_section={mrope_section}, rope_theta={rope_theta}")

# Load weights
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

def qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * (1.0 + w.float())).float()

# Load all layer 3 weights
prefix = 'model.language_model.layers.3.'
w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()
w_post_ln = load_cpu(prefix + 'post_attention_layernorm.weight').float()

w_q_fused = load_cpu(prefix + 'self_attn.q_proj.weight').float()  # [12288, 5120]
full_q_size = num_heads * head_dim  # 6144
w_q = w_q_fused[:full_q_size, :]    # [6144, 5120]
w_gate = w_q_fused[full_q_size:, :] # [6144, 5120]
w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()  # [1024, 5120]
w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()  # [1024, 5120]
w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()  # [5120, 6144]
w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()  # [256]
w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()  # [256]

w_gate_proj = load_cpu(prefix + 'mlp.gate_proj.weight').float()
w_up_proj = load_cpu(prefix + 'mlp.up_proj.weight').float()
w_down_proj = load_cpu(prefix + 'mlp.down_proj.weight').float()

# Check Q w_norm weight (should match engine's)
# Engine uses Qwen3_5RMSNorm with weight initialized to zeros
print(f"\nq_norm weight: min={w_qn.min():.6f}, max={w_qn.max():.6f}, mean={w_qn.mean():.6f}")
print(f"k_norm weight: min={w_kn.min():.6f}, max={w_kn.max():.6f}, mean={w_kn.mean():.6f}")

# Build MRoPE cache using the simple interleaving (matches diag_cpu_layers05.py)
num_sections = len(mrope_section)
half_dim = rotary_dim // 2
section_freqs = []
t = torch.arange(max_pos, dtype=torch.float32)
for sec_size in mrope_section:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, sec_size, dtype=torch.float32) * 2 / rotary_dim))
    section_freqs.append(torch.einsum("i,j->ij", t, inv_freq))
pairs = []
for i in range(half_dim):
    s = i % num_sections
    idx = i // num_sections
    pairs.append(section_freqs[s][:, idx])
freqs = torch.stack(pairs, dim=-1)
cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

# ============================================================
# CPU FullAttention computation
# ============================================================

# Q, Gate, K, V projections (FULL, no TP sharding)
q_cpu = F.linear(hs_work, w_q)       # [1, 7, 6144]
gate_cpu = F.linear(hs_work, w_gate)  # [1, 7, 6144]
k_cpu = F.linear(hs_work, w_k)       # [1, 7, 1024]
v_cpu = F.linear(hs_work, w_v)       # [1, 7, 1024]

q_cpu = q_cpu.reshape(B, T, num_heads, head_dim)       # [1, 7, 24, 256]
k_cpu = k_cpu.reshape(B, T, num_kv_heads, head_dim)    # [1, 7, 4, 256]
v_cpu = v_cpu.reshape(B, T, num_kv_heads, head_dim)    # [1, 7, 4, 256]
gate_cpu_h = gate_cpu.reshape(B, T, num_heads, head_dim)  # [1, 7, 24, 256]

print(f"\n--- Q proj ---")
print(f"  Q norm: {q_cpu.norm():.4f}")
print(f"  K norm: {k_cpu.norm():.4f}")
print(f"  V norm: {v_cpu.norm():.4f}")
print(f"  Gate norm: {gate_cpu_h.norm():.4f}")

# Q/K norms (Qwen3_5RMSNorm)
q_cpu_normed = qwen35_rms_norm(q_cpu.float(), w_qn.unsqueeze(0).unsqueeze(0))
k_cpu_normed = qwen35_rms_norm(k_cpu.float(), w_kn.unsqueeze(0).unsqueeze(0))

print(f"\n--- After Q/K norms ---")
print(f"  Q_normed norm: {q_cpu_normed.norm():.4f}")
print(f"  K_normed norm: {k_cpu_normed.norm():.4f}")

# MRoPE (CLONE to preserve pre-MRoPE values)
q_cpu_rope = q_cpu_normed.clone()
k_cpu_rope = k_cpu_normed.clone()
pos_c = torch.arange(S, dtype=torch.int64)
cos = cos_sin_cache[pos_c, :rotary_dim // 2].view(1, S, 1, rotary_dim // 2)
sin = cos_sin_cache[pos_c, rotary_dim // 2:].view(1, S, 1, rotary_dim // 2)
cos_dup = torch.cat([cos, cos], dim=-1)
sin_dup = torch.cat([sin, sin], dim=-1)

q_rot = q_cpu_rope[..., :rotary_dim].float()
half_r = rotary_dim // 2
q1, q2 = q_rot[..., :half_r], q_rot[..., half_r:]
q_cpu_rope[..., :rotary_dim] = (q_rot * cos_dup + torch.cat([-q2, q1], dim=-1) * sin_dup)

k_rot = k_cpu_rope[..., :rotary_dim].float()
k1, k2 = k_rot[..., :half_r], k_rot[..., half_r:]
k_cpu_rope[..., :rotary_dim] = (k_rot * cos_dup + torch.cat([-k2, k1], dim=-1) * sin_dup)

print(f"\n--- After MRoPE ---")
print(f"  Q_rope norm: {q_cpu_rope.norm():.4f}")
print(f"  K_rope norm: {k_cpu_rope.norm():.4f}")

# SDPA with GQA
n_groups = num_heads // num_kv_heads  # 6
q_attn = q_cpu_rope.transpose(1, 2)  # [1, 24, 7, 256]
k_attn = k_cpu_rope.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads, T, head_dim)
v_attn = v_cpu.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads, T, head_dim)

scale = head_dim ** -0.5
attn_out_cpu = F.scaled_dot_product_attention(
    q_attn.float(), k_attn.float(), v_attn.float(),
    is_causal=True, scale=scale
).transpose(1, 2).reshape(B, T, num_heads * head_dim)  # [1, 7, 6144]

print(f"\n--- After SDPA ---")
print(f"  Attn out norm: {attn_out_cpu.norm():.4f}")

# Output gate
gate_flat = gate_cpu_h.reshape(B, T, num_heads * head_dim)
attn_gated_cpu = attn_out_cpu.float() * torch.sigmoid(gate_flat.float())

print(f"  Gated attn out norm: {attn_gated_cpu.norm():.4f}")

# o_proj
o_proj_cpu = F.linear(attn_gated_cpu, w_o)  # [1, 7, 5120]

print(f"  o_proj out norm: {o_proj_cpu.norm():.4f}")

# post_ln + MLP
# The GPU decoder layer computes: fused_add_rms_norm(attn_out, residual, post_ln_weight, eps)
# which does: residual += attn_out, attn_out = rms_norm(residual)
# Then mlp_out = mlp(attn_out)
# So we need to start with the residual from before layer 3
resid_pre = gpu_data['resid_pre_ln'].float() + gpu_data['hs_pre_ln'].float()  # residual after layers 0-2
resid_post_attn = resid_pre + o_proj_cpu.float()  # accumulated residual after attention
hs_mlp_input = qwen35_rms_norm(resid_post_attn, w_post_ln)

print(f"\n--- Post-attention ---")
print(f"  resid_pre_ln (after layer 2) norm: {resid_pre.norm():.4f}")
print(f"  resid_post_attn norm: {resid_post_attn.norm():.4f}")
print(f"  hs_mlp_input (normed) norm: {hs_mlp_input.norm():.4f}")

# MLP
gate_h = F.linear(hs_mlp_input, w_gate_proj)
up_h = F.linear(hs_mlp_input, w_up_proj)
mlp_out_cpu = F.linear(F.silu(gate_h) * up_h.float(), w_down_proj).float()

print(f"\n--- Final layer 3 output ---")
print(f"  CPU mlp_out norm: {mlp_out_cpu.norm():.4f}")
print(f"  GPU mlp_out norm: {hs_out_gpu.norm():.4f}")

# Compare
diff = (mlp_out_cpu - hs_out_gpu).abs()
print(f"  Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
norm_pct = abs(mlp_out_cpu.norm() - hs_out_gpu.norm()) / hs_out_gpu.norm() * 100
print(f"  Norm diff: {norm_pct:.2f}%")

# Also compare with diag_cpu_layers05 reference
ref = torch.load('/tmp/diag_cpu_layers05_ref.pt', map_location='cpu', weights_only=True)
cp_ref = ref['checkpoints']
if 3 in cp_ref:
    ref_norm = cp_ref[3]['norm']
    print(f"  CPU reference (diag_cpu_layers05) norm: {ref_norm:.4f}")
    ref_diff = (mlp_out_cpu - cp_ref[3]['hs'].float()).abs()
    print(f"  Diff vs CPU ref: max={ref_diff.max():.6f}, mean={ref_diff.mean():.6f}")

# Save intermediates
torch.save({
    'hs_work': hs_work,
    'q_proj': q_cpu,
    'k_proj': k_cpu,
    'v_proj': v_cpu,
    'gate_proj': gate_cpu_h,
    'q_normed': q_cpu_normed,
    'k_normed': k_cpu_normed,
    'q_rope': q_cpu_rope,
    'k_rope': k_cpu_rope,
    'attn_out': attn_out_cpu,
    'attn_gated': attn_gated_cpu,
    'o_proj': o_proj_cpu,
    'mlp_out': mlp_out_cpu,
    'resid_pre': resid_pre,
    'resid_post_attn': resid_post_attn,
}, '/tmp/diag_cpu_fullattn_detail.pt')
print(f"\nSaved detailed intermediates to /tmp/diag_cpu_fullattn_detail.pt")
