#!/usr/bin/env python3
"""Save the correct FullAttention intermediates for element-wise comparison
against the old (wrong) diag_fullattn_cpu_ref.pt.
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

def ref_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + w)

def ref_rms_norm_gated(x, g, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    x_norm = x.float() * rstd
    return (x_norm * w.float() * F.silu(g.float())).float()

# Load all weights
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
def w(k): return loaded[k]

# MRoPE cache
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
cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

def apply_mrope(q_n, k_n, positions):
    B_n, S_n = q_n.shape[0], q_n.shape[1]
    cos = cos_sin[positions, :rotary_dim//2].view(1, S_n, 1, rotary_dim//2)
    sin = cos_sin[positions, rotary_dim//2:].view(1, S_n, 1, rotary_dim//2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    q_rot = q_n[..., :rotary_dim].float()
    half_r = rotary_dim // 2
    q1, q2 = q_rot[..., :half_r], q_rot[..., half_r:]
    q_n[..., :rotary_dim] = (q_rot * cos + torch.cat([-q2, q1], dim=-1) * sin).to(q_n.dtype)
    k_rot = k_n[..., :rotary_dim].float()
    k1, k2 = k_rot[..., :half_r], k_rot[..., half_r:]
    k_n[..., :rotary_dim] = (k_rot * cos + torch.cat([-k2, k1], dim=-1) * sin).to(k_n.dtype)

# Forward pass
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
    gated_out = ref_rms_norm_gated(core_out.reshape(-1, linear_v_dim), z.reshape(-1, linear_v_dim), w_nm.float()).view(B, T, linear_v_heads * linear_v_dim)
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

# Layer 3
pfx3 = 'model.language_model.layers.3.'
w_iln3 = w(pfx3 + 'input_layernorm.weight')
w_pln3 = w(pfx3 + 'post_attention_layernorm.weight')
w_q3 = w(pfx3 + 'self_attn.q_proj.weight')
w_k3 = w(pfx3 + 'self_attn.k_proj.weight')
w_v3 = w(pfx3 + 'self_attn.v_proj.weight')
w_o3 = w(pfx3 + 'self_attn.o_proj.weight')
w_qn3 = w(pfx3 + 'self_attn.q_norm.weight')
w_kn3 = w(pfx3 + 'self_attn.k_norm.weight')

residual = residual + hs
hs_normed = ref_rms_norm(residual, w_iln3)
B, T = 1, len(tokens)

q_full = F.linear(hs_normed, w_q3)
q_ref, gate_ref = torch.chunk(q_full, 2, dim=-1)
k_ref = F.linear(hs_normed, w_k3)
v_ref = F.linear(hs_normed, w_v3)
q_ref = q_ref.view(B, T, num_heads, head_dim)
k_ref = k_ref.view(B, T, num_kv_heads, head_dim)
v_ref = v_ref.view(B, T, num_kv_heads, head_dim)
gate_ref = gate_ref.view(B, T, num_heads, head_dim)
q_ref_n = ref_rms_norm(q_ref.float(), w_qn3.unsqueeze(0).unsqueeze(0)).to(q_ref.dtype)
k_ref_n = ref_rms_norm(k_ref.float(), w_kn3.unsqueeze(0).unsqueeze(0)).to(k_ref.dtype)
q_ref_r = q_ref_n.clone()
k_ref_r = k_ref_n.clone()
apply_mrope(q_ref_r, k_ref_r, positions)

n_groups = num_heads // num_kv_heads
k_exp = k_ref_r.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, T, num_heads, head_dim)
v_exp = v_ref.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, T, num_heads, head_dim)
q_attn = q_ref_r.transpose(1, 2)
k_attn = k_exp.transpose(1, 2)
v_attn = v_exp.transpose(1, 2)
attn_out = F.scaled_dot_product_attention(
    q_attn.float(), k_attn.float(), v_attn.float(),
    attn_mask=None, dropout_p=0.0, is_causal=True
).transpose(1, 2).to(q_ref.dtype)
attn_out = attn_out.reshape(B, T, num_heads * head_dim)
gate_out = torch.sigmoid(gate_ref.reshape(B, T, num_heads * head_dim))
attn_gated = attn_out * gate_out
o_out = F.linear(attn_gated, w_o3)

# Save everything
torch.save({
    'hs_input': hs_normed.detach(),
    'q_proj': q_ref.detach(),
    'k_proj': k_ref.detach(),
    'v_proj': v_ref.detach(),
    'gate_proj': gate_ref.detach(),
    'q_normed': q_ref_n.detach(),
    'k_normed': k_ref_n.detach(),
    'q_rope': q_ref_r.detach(),
    'k_rope': k_ref_r.detach(),
    'k_exp': k_exp.detach(),
    'v_exp': v_exp.detach(),
    'attn_out': attn_gated.detach(),
    'layer3_out': o_out.detach(),
    'residual_before_ln': residual.detach(),
}, '/tmp/diag_fa_correct_ref.pt')
print(f"Correct reference saved to /tmp/diag_fa_correct_ref.pt")
print(f"  q_rope norm: {q_ref_r.norm():.4f}")
print(f"  attn_out norm: {attn_gated.norm():.4f}")
