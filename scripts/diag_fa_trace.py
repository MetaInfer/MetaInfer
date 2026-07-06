#!/usr/bin/env python3
"""Trace FullAttention layer 3 norms at every substep, comparing our impl
against the reference diag_cpu_layers05.py approach. CPU-only.
"""
import os, sys, torch, json, math
import torch.nn.functional as F

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

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

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

# Exact copy from diag_cpu_layers05.py
def ref_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + w)

def ref_rms_norm_gated(x, g, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    x_norm = x.float() * rstd
    return (x_norm * w.float() * F.silu(g.float())).float()

def build_mrope_cache(max_pos, head_size, rope_theta, mrope_section):
    num_sections = len(mrope_section)
    half_dim = head_size // 2
    section_freqs = []
    t = torch.arange(max_pos, dtype=torch.float32)
    for section_size in mrope_section:
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, section_size, dtype=torch.float32) * 2 / head_size))
        freqs_s = torch.einsum("i,j->ij", t, inv_freq)
        section_freqs.append(freqs_s)
    interleaved_pairs = []
    for i in range(half_dim):
        s = i % num_sections
        idx = i // num_sections
        interleaved_pairs.append(section_freqs[s][:, idx])
    freqs = torch.stack(interleaved_pairs, dim=-1)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)

cos_sin = build_mrope_cache(max_pos, rotary_dim, rope_theta, mrope_section)

def apply_mrope(q, k, positions):
    B, S = q.shape[0], q.shape[1]
    cos = cos_sin[positions, :rotary_dim//2]
    sin = cos_sin[positions, rotary_dim//2:]
    cos = cos.view(1, S, 1, rotary_dim//2)
    sin = sin.view(1, S, 1, rotary_dim//2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    q_rot = q[..., :rotary_dim].float()
    half_rot = rotary_dim // 2
    q1, q2 = q_rot[..., :half_rot], q_rot[..., half_rot:]
    q_rotated = torch.cat([-q2, q1], dim=-1)
    q[..., :rotary_dim] = (q_rot * cos + q_rotated * sin).to(q.dtype)
    k_rot = k[..., :rotary_dim].float()
    k1, k2 = k_rot[..., :half_rot], k_rot[..., half_rot:]
    k_rotated = torch.cat([-k2, k1], dim=-1)
    k[..., :rotary_dim] = (k_rot * cos + k_rotated * sin).to(k.dtype)

# Preload all layer 0-3 weights
needed = set()
for i in range(4):
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
needed.add('model.language_model.embed_tokens.weight')

files_needed = set()
for k in needed:
    files_needed.add(os.path.join(model_dir, wm[k]))

loaded = {}
for fpath in sorted(files_needed):
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            if k in needed:
                loaded[k] = sf.get_tensor(k).float()
print(f"Loaded {len(loaded)} tensors")

def w(k):
    return loaded[k]

# Forward pass: layers 0-2 (GatedDeltaNet) — EXACT copy from diag_cpu_layers05.py
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long)
positions = torch.arange(len(tokens), dtype=torch.int64)

linear_k_heads = tc['linear_num_key_heads']
linear_v_heads = tc['linear_num_value_heads']
linear_k_dim = tc['linear_key_head_dim']
linear_v_dim = tc['linear_value_head_dim']

emb = F.embedding(input_ids, w('model.language_model.embed_tokens.weight'))
hs = emb.clone()
residual = None

for layer_idx in range(3):
    pfx = f'model.language_model.layers.{layer_idx}.'
    w_iln = w(pfx + 'input_layernorm.weight')
    w_pln = w(pfx + 'post_attention_layernorm.weight')

    if residual is None:
        residual = hs.clone()
        hs_normed = ref_rms_norm(hs, w_iln)
    else:
        residual = residual + hs
        hs_normed = ref_rms_norm(residual, w_iln)
        hs.copy_(residual)

    B, T, H = hs_normed.shape

    w_iqkv = w(pfx + 'linear_attn.in_proj_qkv.weight')
    w_c1d = w(pfx + 'linear_attn.conv1d.weight')
    w_ia = w(pfx + 'linear_attn.in_proj_a.weight')
    w_ib = w(pfx + 'linear_attn.in_proj_b.weight')
    Alog = w(pfx + 'linear_attn.A_log')
    dbias = w(pfx + 'linear_attn.dt_bias')
    w_iz = w(pfx + 'linear_attn.in_proj_z.weight')
    w_nm = w(pfx + 'linear_attn.norm.weight')
    w_ot = w(pfx + 'linear_attn.out_proj.weight')

    mixed_qkv = F.linear(hs_normed, w_iqkv)
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))
    conv_out = F.conv1d(mixed_qkv_pad, w_c1d, bias=None, groups=mixed_qkv_t.shape[1])
    conv_out = F.silu(conv_out).transpose(1, 2)

    q = conv_out[:, :, :linear_k_heads*linear_k_dim].view(B, T, linear_k_heads, linear_k_dim)
    k = conv_out[:, :, linear_k_heads*linear_k_dim:2*linear_k_heads*linear_k_dim].view(B, T, linear_k_heads, linear_k_dim)
    v = conv_out[:, :, 2*linear_k_heads*linear_k_dim:].view(B, T, linear_v_heads, linear_v_dim)

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

    state = torch.zeros(B, linear_v_heads, linear_k_dim, linear_v_dim, dtype=torch.float32)
    core_out = torch.zeros(B, T, linear_v_heads, linear_v_dim, dtype=torch.float32)
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

    z = F.linear(hs_normed, w_iz).view(B, T, linear_v_heads, linear_v_dim)
    gated_out = ref_rms_norm_gated(
        core_out.reshape(-1, linear_v_dim),
        z.reshape(-1, linear_v_dim),
        w_nm.float()
    ).view(B, T, linear_v_heads * linear_v_dim)
    attn_out = F.linear(gated_out.float(), w_ot)
    residual = residual + attn_out.float()
    hs_normed_mlp = ref_rms_norm(residual, w_pln)

    w_gate = w(pfx + 'mlp.gate_proj.weight')
    w_up = w(pfx + 'mlp.up_proj.weight')
    w_down = w(pfx + 'mlp.down_proj.weight')
    gate_h = F.linear(hs_normed_mlp, w_gate)
    up_h = F.linear(hs_normed_mlp, w_up)
    mlp_out = F.linear(F.silu(gate_h) * up_h.float(), w_down).float()
    hs = mlp_out

print(f"Layer 2 output norm: {hs.norm():.4f}")

# ================================================
# Layer 3 FullAttention: Reference method
# ================================================
pfx3 = 'model.language_model.layers.3.'
w_iln3 = w(pfx3 + 'input_layernorm.weight')
w_pln3 = w(pfx3 + 'post_attention_layernorm.weight')
w_q3 = w(pfx3 + 'self_attn.q_proj.weight')  # [12288, 5120]
w_k3 = w(pfx3 + 'self_attn.k_proj.weight')  # [1024, 5120]
w_v3 = w(pfx3 + 'self_attn.v_proj.weight')  # [1024, 5120]
w_o3 = w(pfx3 + 'self_attn.o_proj.weight')  # [5120, 6144]
w_qn3 = w(pfx3 + 'self_attn.q_norm.weight')
w_kn3 = w(pfx3 + 'self_attn.k_norm.weight')
w_gate3 = w(pfx3 + 'mlp.gate_proj.weight')
w_up3 = w(pfx3 + 'mlp.up_proj.weight')
w_down3 = w(pfx3 + 'mlp.down_proj.weight')

residual = residual + hs
hs_normed = ref_rms_norm(residual, w_iln3)
hs.copy_(residual)
B, T = 1, len(tokens)
print(f"\nLayer 3 normed input norm: {hs_normed.norm():.4f}")

# Q, K, V, Gate — using REFERENCE method (chunk)
q_full = F.linear(hs_normed, w_q3)
q_ref, gate_ref = torch.chunk(q_full, 2, dim=-1)
k_ref = F.linear(hs_normed, w_k3)
v_ref = F.linear(hs_normed, w_v3)

q_ref = q_ref.view(B, T, num_heads, head_dim)
k_ref = k_ref.view(B, T, num_kv_heads, head_dim)
v_ref = v_ref.view(B, T, num_kv_heads, head_dim)
gate_ref = gate_ref.view(B, T, num_heads, head_dim)

print(f"Q proj norm: {q_ref.norm():.4f}")
print(f"K proj norm: {k_ref.norm():.4f}")
print(f"V proj norm: {v_ref.norm():.4f}")
print(f"Gate proj norm: {gate_ref.norm():.4f}")

# Q/K norms
q_ref_n = ref_rms_norm(q_ref.float(), w_qn3.unsqueeze(0).unsqueeze(0)).to(q_ref.dtype)
k_ref_n = ref_rms_norm(k_ref.float(), w_kn3.unsqueeze(0).unsqueeze(0)).to(k_ref.dtype)
print(f"\nAfter Q/K norms: Q={q_ref_n.norm():.4f}, K={k_ref_n.norm():.4f}")

# MRoPE
apply_mrope(q_ref_n, k_ref_n, positions)
print(f"After MRoPE: Q={q_ref_n.norm():.4f}, K={k_ref_n.norm():.4f}")

# GQA + SDPA
n_groups = num_heads // num_kv_heads
k_exp = k_ref_n.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, T, num_heads, head_dim)
v_exp = v_ref.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, T, num_heads, head_dim)

q_attn = q_ref_n.transpose(1, 2)
k_attn = k_exp.transpose(1, 2)
v_attn = v_exp.transpose(1, 2)

attn_out = F.scaled_dot_product_attention(
    q_attn.float(), k_attn.float(), v_attn.float(),
    attn_mask=None, dropout_p=0.0, is_causal=True
).transpose(1, 2).to(q_ref.dtype)

attn_out = attn_out.reshape(B, T, num_heads * head_dim)
print(f"After SDPA: norm={attn_out.norm():.4f}")

# Output gate
gate_out = torch.sigmoid(gate_ref.reshape(B, T, num_heads * head_dim))
attn_gated = attn_out * gate_out
print(f"After gate: norm={attn_gated.norm():.4f}")

# o_proj
o_out = F.linear(attn_gated, w_o3)
print(f"After o_proj: norm={o_out.norm():.4f}")

# Post-LN + MLP
resid_attn = residual + o_out.float()
hs_mlp_in = ref_rms_norm(resid_attn, w_pln3)
print(f"Post-LN input norm: {hs_mlp_in.norm():.4f}")

gate_h = F.linear(hs_mlp_in, w_gate3)
up_h = F.linear(hs_mlp_in, w_up3)
mlp_out = F.linear(F.silu(gate_h) * up_h.float(), w_down3).float()
print(f"Layer 3 mlp_out norm: {mlp_out.norm():.4f}")

# Compare with CPU reference
ref = torch.load('/tmp/diag_cpu_layers05_ref.pt', map_location='cpu', weights_only=True)
cp_ref = ref['checkpoints']
ref_norm = cp_ref[3]['norm']
ref_hs = cp_ref[3]['hs'].float()
diff = (mlp_out - ref_hs).abs()
print(f"\nCPU ref norm: {ref_norm:.4f}")
print(f"Diff vs ref: max={diff.max():.6f}, mean={diff.mean():.6f}")
print(f"Norm diff: {abs(mlp_out.norm()-ref_norm)/ref_norm*100:.2f}%")

# Also compute using the "separate projections" method (diag_fullattn_cpu_verify.py approach)
full_q_size = num_heads * head_dim
w_q_only = w_q3[:full_q_size, :]     # [6144, 5120]
w_gate_only = w_q3[full_q_size:, :]  # [6144, 5120]

q_ours = F.linear(hs_normed, w_q_only).view(B, T, num_heads, head_dim)
gate_ours = F.linear(hs_normed, w_gate_only).view(B, T, num_heads, head_dim)
k_ours = F.linear(hs_normed, w_k3).view(B, T, num_kv_heads, head_dim)
v_ours = F.linear(hs_normed, w_v3).view(B, T, num_kv_heads, head_dim)

print(f"\n=== Separate vs Fused projections ===")
qd = (q_ours - q_ref).abs()
gd = (gate_ours - gate_ref).abs()
kd = (k_ours - k_ref).abs()
vd = (v_ours - v_ref).abs()
print(f"  Q diff: max={qd.max():.12f}, mean={qd.mean():.12f}")
print(f"  Gate diff: max={gd.max():.12f}, mean={gd.mean():.12f}")
print(f"  K diff: max={kd.max():.12f}, mean={kd.mean():.12f}")
print(f"  V diff: max={vd.max():.12f}, mean={vd.mean():.12f}")

# Also compare qwen35_rms_norm variants
def our_rms_norm_verbose(x, w):
    """From diag_fullattn_cpu_verify.py"""
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * (1.0 + w.float())).float()

# Apply norms using both methods
q_ours_n_ref = ref_rms_norm(q_ours.float(), w_qn3.unsqueeze(0).unsqueeze(0)).to(q_ours.dtype)
q_ours_n_our = our_rms_norm_verbose(q_ours, w_qn3.unsqueeze(0).unsqueeze(0))
qn_diff = (q_ours_n_our - q_ours_n_ref).abs()
print(f"\n=== Q norm methods ===")
print(f"  ref norm: {q_ours_n_ref.norm():.4f}, our norm: {q_ours_n_our.norm():.4f}")
print(f"  Diff: max={qn_diff.max():.12f}, mean={qn_diff.mean():.12f}")

# Full pipeline using "our" method (separate projections, verbose norm)
q_ours_n = q_ours_n_our  # already computed above
k_ours_n = our_rms_norm_verbose(k_ours, w_kn3.unsqueeze(0).unsqueeze(0))

# MRoPE
q_ours_r = q_ours_n.clone()
k_ours_r = k_ours_n.clone()
apply_mrope(q_ours_r, k_ours_r, positions)

n_g = num_heads // num_kv_heads
k_exp2 = k_ours_r.unsqueeze(2).expand(-1, -1, n_g, -1, -1).reshape(B, T, num_heads, head_dim)
v_exp2 = v_ours.unsqueeze(2).expand(-1, -1, n_g, -1, -1).reshape(B, T, num_heads, head_dim)
q_attn2 = q_ours_r.transpose(1, 2)
k_attn2 = k_exp2.transpose(1, 2)
v_attn2 = v_exp2.transpose(1, 2)
attn_out2 = F.scaled_dot_product_attention(
    q_attn2.float(), k_attn2.float(), v_attn2.float(),
    attn_mask=None, dropout_p=0.0, is_causal=True
).transpose(1, 2).to(q_ours.dtype)
attn_out2 = attn_out2.reshape(B, T, num_heads * head_dim)
gate_out2 = torch.sigmoid(gate_ours.reshape(B, T, num_heads * head_dim))
attn_gated2 = attn_out2 * gate_out2
o_out2 = F.linear(attn_gated2, w_o3)

resid_a2 = residual + o_out2.float()
hs_mlp_in2 = ref_rms_norm(resid_a2, w_pln3)
gate_h2 = F.linear(hs_mlp_in2, w_gate3)
up_h2 = F.linear(hs_mlp_in2, w_up3)
mlp_out2 = F.linear(F.silu(gate_h2) * up_h2.float(), w_down3).float()

print(f"\n=== Separate projections pipeline ===")
print(f"  After Q norm: {q_ours_n.norm():.4f} (ref: {q_ref_n.norm():.4f})")
print(f"  After MRoPE: {q_ours_r.norm():.4f} (ref: {q_ref_n.norm():.4f})")
print(f"  After SDPA: {attn_out2.norm():.4f} (ref: {attn_out.norm():.4f})")
print(f"  After gate: {attn_gated2.norm():.4f} (ref: {attn_gated.norm():.4f})")
print(f"  After o_proj: {o_out2.norm():.4f} (ref: {o_out.norm():.4f})")
print(f"  mlp_out: {mlp_out2.norm():.4f} (ref: {mlp_out.norm():.4f})")

# Detailed substep comparison
print(f"\n=== Substep norms ===")
print(f"  {'Step':<25} {'Ref':<12} {'Separate':<12} {'Diff%':<10}")
steps = [
    ('Q proj', q_ref, q_ours),
    ('Gate proj', gate_ref, gate_ours),
    ('K proj', k_ref, k_ours),
    ('V proj', v_ref, v_ours),
    ('Q after norm', q_ref_n, q_ours_n),
    ('K after norm', k_ref_n, k_ours_n),
    ('SDPA output', attn_out, attn_out2),
    ('Gated output', attn_gated, attn_gated2),
    ('o_proj output', o_out, o_out2),
    ('MLP output', mlp_out, mlp_out2),
]
for name, ref_val, sep_val in steps:
    r_n = ref_val.norm().item()
    s_n = sep_val.norm().item()
    d = abs(r_n - s_n) / r_n * 100 if r_n > 0 else 0
    print(f"  {name:<25} {r_n:<12.4f} {s_n:<12.4f} {d:<10.4f}")

