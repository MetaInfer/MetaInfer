#!/usr/bin/env python3
"""Lightweight GPU trace: load only layers 0-3, save intermediates for CPU comparison.

Loads only the first 4 layers (0-3) from safetensors directly, avoiding the full
64-layer model to stay within 16GB VRAM.
"""
import os, sys, json, math, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

# Single GPU mode (no TP) — for CPU reference comparison
rank = 0
tp_size = 1
device = 'cuda:0'
torch.cuda.set_device(0)

model_dir = os.environ['MODEL_DIR']
print(f"[GPU]Loading config and weights...", flush=True)

with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)

eps = tc['rms_norm_eps']
head_dim = tc['head_dim']
num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']
hidden_size = tc['hidden_size']
rotary_dim = int(head_dim * tc.get('rope_parameters', {}).get('partial_rotary_factor', 1.0))
max_pos = tc['max_position_embeddings']
rope_theta = tc.get('rope_theta') or tc.get('rope_parameters', {}).get('rope_theta', 1000000.0)
mrope_section = tc.get('rope_parameters', {}).get('mrope_section')
mrope_interleaved = tc.get('rope_parameters', {}).get('mrope_interleaved', False)

linear_k_heads = tc['linear_num_key_heads']
linear_v_heads = tc['linear_num_value_heads']
linear_k_dim = tc['linear_key_head_dim']
linear_v_dim = tc['linear_value_head_dim']

heads_per_rank = num_heads  # Full model (no TP)
kv_heads_per_rank = num_kv_heads
k_heads_per_rank = linear_k_heads
v_heads_per_rank = linear_v_heads

# Load only layer 0-3 weights from safetensors
from safetensors import safe_open
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

needed = set()
needed.add('model.language_model.embed_tokens.weight')
for i in range(4):  # Only layers 0-3
    pfx = f'model.language_model.layers.{i}.'
    needed.add(pfx + 'input_layernorm.weight')
    needed.add(pfx + 'post_attention_layernorm.weight')
    needed.add(pfx + 'mlp.gate_proj.weight')
    needed.add(pfx + 'mlp.up_proj.weight')
    needed.add(pfx + 'mlp.down_proj.weight')
    if i < 3:
        needed.add(pfx + 'linear_attn.in_proj_qkv.weight')
        needed.add(pfx + 'linear_attn.conv1d.weight')
        needed.add(pfx + 'linear_attn.in_proj_a.weight')
        needed.add(pfx + 'linear_attn.in_proj_b.weight')
        needed.add(pfx + 'linear_attn.A_log')
        needed.add(pfx + 'linear_attn.dt_bias')
        needed.add(pfx + 'linear_attn.in_proj_z.weight')
        needed.add(pfx + 'linear_attn.norm.weight')
        needed.add(pfx + 'linear_attn.out_proj.weight')
    else:
        needed.add(pfx + 'self_attn.q_proj.weight')
        needed.add(pfx + 'self_attn.k_proj.weight')
        needed.add(pfx + 'self_attn.v_proj.weight')
        needed.add(pfx + 'self_attn.o_proj.weight')
        needed.add(pfx + 'self_attn.q_norm.weight')
        needed.add(pfx + 'self_attn.k_norm.weight')

files_needed = set(os.path.join(model_dir, wm[k]) for k in needed)
print(f"[GPU]Loading {len(files_needed)} safetensor files with {len(needed)} tensors...", flush=True)

loaded = {}
for fpath in sorted(files_needed):
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            if k in needed:
                loaded[k] = sf.get_tensor(k)
print(f"[GPU]Loaded {len(loaded)} tensors", flush=True)

def shard_col(tensor, rank, size):
    """Column-parallel shard: split output dim (dim 0)."""
    n = tensor.shape[0]
    per = n // size
    return tensor[rank*per:(rank+1)*per, :].contiguous().to(device, dtype=torch.bfloat16)

def shard_row(tensor, rank, size):
    """Row-parallel shard: split input dim (dim 1)."""
    n = tensor.shape[1]
    per = n // size
    return tensor[:, rank*per:(rank+1)*per].contiguous().to(device, dtype=torch.bfloat16)

def full_tensor(tensor):
    return tensor.to(device, dtype=torch.bfloat16)

# ============================================================
# Build MRoPE cache on GPU
# ============================================================
print(f"[GPU]Building MRoPE cache...", flush=True)
num_sections = len(mrope_section)
half_dim = rotary_dim // 2
section_freqs = []
t_pos = torch.arange(max_pos, dtype=torch.float32)
for sec_size in mrope_section:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, sec_size, dtype=torch.float32) * 2 / rotary_dim))
    section_freqs.append(torch.einsum("i,j->ij", t_pos, inv_freq))
indices = [0] * num_sections
interleaved_pairs = []
for i in range(half_dim):
    s = i % num_sections
    found = False
    for offset in range(num_sections):
        s2 = (s + offset) % num_sections
        if indices[s2] < section_freqs[s2].shape[1]:
            interleaved_pairs.append(section_freqs[s2][:, indices[s2]])
            indices[s2] += 1
            found = True
            break
    assert found
freqs = torch.stack(interleaved_pairs, dim=-1)
cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device, dtype=torch.bfloat16)
del section_freqs, interleaved_pairs, freqs, t_pos

def apply_rope_gpu(x, pos):
    """In-place MRoPE using NeoX style — only on first rotary_dim dims."""
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim]  # view of the rotary part only
    cos = cos_sin[pos, :half].view(1, x.shape[1], 1, half)
    sin = cos_sin[pos, half:].view(1, x.shape[1], 1, half)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    x_f = x_rot.float()
    x1, x2 = x_f[..., :half], x_f[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    x_rot.copy_((x_f * cos + rotated * sin).to(x.dtype))

# ============================================================
# Helper: Qwen3_5RMSNorm
# ============================================================
def qwen35_rms(w):
    return 1.0 + w.to(device, dtype=torch.bfloat16)

def fused_add_rms_gpu(inp, res, w_eff, eps_val):
    """Manual fused_add_rms_norm without vLLM kernel."""
    res.add_(inp)
    x_fp32 = res.float()
    rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps_val)
    inp.copy_((w_eff.float() * (x_fp32 / rms)).to(inp.dtype))

# ============================================================
# Forward pass: layers 0-2 (GatedDeltaNet)
# ============================================================
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
S = len(tokens)
emb_w = full_tensor(loaded['model.language_model.embed_tokens.weight'])
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
hs = F.embedding(input_ids, emb_w).to(torch.bfloat16)
del emb_w, input_ids

positions = torch.arange(S, dtype=torch.int64, device=device)
residual = None
B = 1

print(f"[GPU]Forward: layers 0-2...", flush=True)
for layer_idx in range(3):
    pfx = f'model.language_model.layers.{layer_idx}.'
    w_iln_eff = qwen35_rms(loaded[pfx + 'input_layernorm.weight'])
    w_pln_eff = qwen35_rms(loaded[pfx + 'post_attention_layernorm.weight'])

    # Input norm
    if residual is None:
        residual = hs.clone()
        x_fp32 = hs.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
        hs_normed = (w_iln_eff.float() * (x_fp32 / rms)).to(torch.bfloat16)
    else:
        fused_add_rms_gpu(hs, residual, w_iln_eff, eps)
        hs_normed = hs

    T = S
    # GatedDeltaNet
    w_iqkv = full_tensor(loaded[pfx + 'linear_attn.in_proj_qkv.weight'])
    w_c1d = full_tensor(loaded[pfx + 'linear_attn.conv1d.weight'])  # [out, 1, 4]
    w_ia = full_tensor(loaded[pfx + 'linear_attn.in_proj_a.weight'])
    w_ib = full_tensor(loaded[pfx + 'linear_attn.in_proj_b.weight'])
    Alog = full_tensor(loaded[pfx + 'linear_attn.A_log'])
    dbias = full_tensor(loaded[pfx + 'linear_attn.dt_bias'])
    w_iz = full_tensor(loaded[pfx + 'linear_attn.in_proj_z.weight'])
    w_nm = full_tensor(loaded[pfx + 'linear_attn.norm.weight']).float()
    w_ot = full_tensor(loaded[pfx + 'linear_attn.out_proj.weight'])

    mixed_qkv = F.linear(hs_normed, w_iqkv)
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))
    conv_out = F.conv1d(mixed_qkv_pad, w_c1d, groups=mixed_qkv_t.shape[1])
    conv_out = F.silu(conv_out).transpose(1, 2)

    q = conv_out[:, :, :linear_k_heads*linear_k_dim].reshape(B, T, linear_k_heads, linear_k_dim)
    k = conv_out[:, :, linear_k_heads*linear_k_dim:2*linear_k_heads*linear_k_dim].reshape(B, T, linear_k_heads, linear_k_dim)
    v = conv_out[:, :, 2*linear_k_heads*linear_k_dim:].reshape(B, T, linear_v_heads, linear_v_dim)

    a = F.linear(hs_normed, w_ia)
    b = F.linear(hs_normed, w_ib)
    g = -torch.exp(Alog) * F.softplus(a + dbias)
    beta = torch.sigmoid(b)

    q_n = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
    k_n = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    rpt = linear_v_heads // linear_k_heads
    if rpt > 1:
        q_n = q_n.repeat_interleave(rpt, dim=2)
        k_n = k_n.repeat_interleave(rpt, dim=2)
    q_n = q_n * (1.0 / math.sqrt(linear_k_dim))

    state = torch.zeros(B, linear_v_heads, linear_k_dim, linear_v_dim, dtype=torch.float32, device=device)
    core_out = torch.zeros(B, T, linear_v_heads, linear_v_dim, dtype=torch.float32, device=device)
    for t_idx in range(T):
        g_t = g[:, t_idx, :]
        k_t = k_n[:, t_idx, :, :]
        v_t = v[:, t_idx, :, :]
        q_t = q_n[:, t_idx, :, :]
        beta_t = beta[:, t_idx, :]
        state = state * torch.exp(g_t.float())[:, :, None, None]
        kv_mem = torch.sum(state * k_t.float()[:, :, :, None], dim=-2)
        delta = (v_t.float() - kv_mem) * beta_t.float()[:, :, None]
        state = state + k_t.float()[:, :, :, None] * delta[:, :, None, :]
        o_t = torch.sum(state * q_t.float()[:, :, :, None], dim=-2)
        core_out[:, t_idx, :, :] = o_t

    z = F.linear(hs_normed, w_iz).reshape(B, T, linear_v_heads, linear_v_dim)
    # Gated RMS norm
    gated_out = core_out.reshape(-1, linear_v_dim).float()
    z_flat = z.reshape(-1, linear_v_dim)
    rstd_g = 1.0 / torch.sqrt(gated_out.pow(2).mean(-1, keepdim=True) + eps)
    gated_out = (gated_out * rstd_g * w_nm * F.silu(z_flat.float())).reshape(B, T, -1)

    attn_out = F.linear(gated_out.to(torch.bfloat16), w_ot)
    residual = residual + attn_out.float()

    # Post-attn norm + MLP
    fused_add_rms_gpu(attn_out, residual, w_pln_eff, eps)
    w_gate = full_tensor(loaded[pfx + 'mlp.gate_proj.weight'])
    w_up = full_tensor(loaded[pfx + 'mlp.up_proj.weight'])
    w_down = full_tensor(loaded[pfx + 'mlp.down_proj.weight'])
    gate_h = F.linear(attn_out, w_gate)
    up_h = F.linear(attn_out, w_up)
    mlp_out = F.linear(F.silu(gate_h) * up_h, w_down).float().to(torch.bfloat16)
    hs = mlp_out

    # Cleanup
    del w_iqkv, w_c1d, w_ia, w_ib, Alog, dbias, w_iz, w_nm, w_ot
    del w_gate, w_up, w_down, mixed_qkv, conv_out, q, k, v, a, b, g, beta
    del q_n, k_n, state, core_out, z, gated_out, attn_out, gate_h, up_h, mlp_out

print(f"[GPU]Layer 2 hs norm: {hs.norm():.4f}  residual norm: {residual.norm():.4f}", flush=True)

# Layer 3: FullAttention
pfx3 = 'model.language_model.layers.3.'
w_iln3_eff = qwen35_rms(loaded[pfx3 + 'input_layernorm.weight'])
w_pln3_eff = qwen35_rms(loaded[pfx3 + 'post_attention_layernorm.weight'])

hs_pre_ln = hs.clone()
resid_pre_ln = residual.clone()

# Apply layer 3 input layernorm
hs_in = hs.clone()
fused_add_rms_gpu(hs_in, residual, w_iln3_eff, eps)
hs_work = hs_in  # This is the normed input to FullAttention

# Now run the actual FullAttention — engine-equivalent (using expand+reshape for GQA!)
full_q_size = num_heads * head_dim
W_q_fused = full_tensor(loaded[pfx3 + 'self_attn.q_proj.weight'])
W_q = shard_col(W_q_fused[:full_q_size, :], rank, tp_size)  # [1536, 5120]
W_g = shard_col(W_q_fused[full_q_size:, :], rank, tp_size)  # [1536, 5120]
W_k = shard_col(full_tensor(loaded[pfx3 + 'self_attn.k_proj.weight']), rank, tp_size)  # [256, 5120]
W_v = shard_col(full_tensor(loaded[pfx3 + 'self_attn.v_proj.weight']), rank, tp_size)  # [256, 5120]
W_o = shard_row(full_tensor(loaded[pfx3 + 'self_attn.o_proj.weight']), rank, tp_size)  # [5120, 1536]
w_qn_w = loaded[pfx3 + 'self_attn.q_norm.weight'].to(device, dtype=torch.bfloat16)
w_kn_w = loaded[pfx3 + 'self_attn.k_norm.weight'].to(device, dtype=torch.bfloat16)
del W_q_fused

# Compute FullAttention using EXACT engine method but with expand+reshape GQA
q_r = F.linear(hs_work, W_q).reshape(B, S, heads_per_rank, head_dim)
gate_r = F.linear(hs_work, W_g).reshape(B, S, heads_per_rank, head_dim)
k_r = F.linear(hs_work, W_k).reshape(B, S, kv_heads_per_rank, head_dim)
v_r = F.linear(hs_work, W_v).reshape(B, S, kv_heads_per_rank, head_dim)

print(f"[GPU]Q norm: {q_r.norm():.4f}  K norm: {k_r.norm():.4f}  V norm: {v_r.norm():.4f}", flush=True)

# Q/K norms (Qwen3_5RMSNorm)
eff_qn = 1.0 + w_qn_w
eff_kn = 1.0 + w_kn_w
q_fp32 = q_r.float()
rms_q = torch.sqrt(q_fp32.pow(2).mean(-1, keepdim=True) + eps)
q_n = (eff_qn.float() * (q_fp32 / rms_q)).to(torch.bfloat16)
k_fp32 = k_r.float()
rms_k = torch.sqrt(k_fp32.pow(2).mean(-1, keepdim=True) + eps)
k_n = (eff_kn.float() * (k_fp32 / rms_k)).to(torch.bfloat16)

# MRoPE
q_r2 = q_n.clone()
k_r2 = k_n.clone()
apply_rope_gpu(q_r2, positions)
apply_rope_gpu(k_r2, positions)

print(f"[GPU]Q rope norm: {q_r2.norm():.4f}  K rope norm: {k_r2.norm():.4f}", flush=True)

# GQA: expand+reshape (ROUND-ROBIN pattern: Q0→K0, Q1→K1, Q2→K2, Q3→K3, Q4→K0, ...)
gqa_factor = heads_per_rank // kv_heads_per_rank
q_flat = q_r2.reshape(S, heads_per_rank, head_dim)
k_flat = k_r2.reshape(S, kv_heads_per_rank, head_dim)
v_flat = v_r.reshape(S, kv_heads_per_rank, head_dim)

# SDPA with expand+reshape GQA (ROUND-ROBIN pattern)
q_4d = q_flat.reshape(1, S, heads_per_rank, head_dim).transpose(1, 2)  # [1, H, S, D]
if gqa_factor > 1:
    # k: [S, KV, D] → [1, S, 1, KV, D] → expand gqa → [1, S, gqa, KV, D] → reshape [1, S, H, D]
    # This produces ROUND-ROBIN: Q0→K0, Q1→K1, Q2→K2, Q3→K3, Q4→K0, ...
    k_4d = k_flat.unsqueeze(1).unsqueeze(0)  # [1, S, 1, KV, D]
    k_4d = k_4d.expand(-1, -1, gqa_factor, -1, -1)  # [1, S, gqa, KV, D]  (dim 2 stride=0)
    k_4d = k_4d.reshape(1, S, heads_per_rank, head_dim).transpose(1, 2)  # [1, H, S, D]
    v_4d = v_flat.unsqueeze(1).unsqueeze(0)
    v_4d = v_4d.expand(-1, -1, gqa_factor, -1, -1)
    v_4d = v_4d.reshape(1, S, heads_per_rank, head_dim).transpose(1, 2)
else:
    k_4d = k_flat.reshape(1, S, heads_per_rank, head_dim).transpose(1, 2)
    v_4d = v_flat.reshape(1, S, heads_per_rank, head_dim).transpose(1, 2)

attn_out_4d = F.scaled_dot_product_attention(
    q_4d.float(), k_4d.float(), v_4d.float(), is_causal=True)
attn_out = attn_out_4d.transpose(1, 2).reshape(S, heads_per_rank * head_dim).to(torch.bfloat16)
attn_out = attn_out.reshape(B, S, heads_per_rank * head_dim)

# Output gate
gate_flat = gate_r.reshape(B, S, heads_per_rank * head_dim)
gated = attn_out * torch.sigmoid(gate_flat)
o_out = F.linear(gated, W_o)

print(f"[GPU]gated norm: {gated.norm():.4f}  o_out norm: {o_out.norm():.4f}", flush=True)

# MLP
resid_attn = resid_pre_ln + o_out.float()
hs_mlp_in = hs_in  # reuse tensor
hs_mlp_in.zero_()
x_fp32 = resid_attn.float()
rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
hs_mlp_in.copy_((w_pln3_eff.float() * (x_fp32 / rms)).to(torch.bfloat16))

W_gate3 = full_tensor(loaded[pfx3 + 'mlp.gate_proj.weight'])
W_up3 = full_tensor(loaded[pfx3 + 'mlp.up_proj.weight'])
W_down3 = full_tensor(loaded[pfx3 + 'mlp.down_proj.weight'])
gate_h = F.linear(hs_mlp_in, W_gate3)
up_h = F.linear(hs_mlp_in, W_up3)
mlp_out = F.linear(F.silu(gate_h) * up_h, W_down3).float()

# Save all intermediates
torch.cuda.synchronize()
torch.save({
    'hs_pre_ln': hs_pre_ln.cpu().float(),
    'resid_pre_ln': resid_pre_ln.cpu().float(),
    'hs_work': hs_work.cpu().float(),
    'q_proj': q_r.cpu().float(),
    'k_proj': k_r.cpu().float(),
    'v_proj': v_r.cpu().float(),
    'gate_proj': gate_r.cpu().float(),
    'q_normed': q_n.cpu().float(),
    'k_normed': k_n.cpu().float(),
    'q_rope': q_r2.cpu().float(),
    'k_rope': k_r2.cpu().float(),
    'attn_gated': gated.cpu().float(),
    'o_proj': o_out.cpu().float(),
    'mlp_out': mlp_out.cpu().float(),
    'resid_attn': resid_attn.cpu().float(),
    'tokens': tokens, 'S': S,
}, '/tmp/diag_gpu_layers03.pt')
print(f"\nSaved to /tmp/diag_gpu_layers03.pt")
print(f"  hs_work norm: {hs_work.norm():.4f}")
print(f"  mlp_out norm: {mlp_out.norm():.4f}")
print(f"  resid_pre_ln norm: {resid_pre_ln.norm():.4f}")

