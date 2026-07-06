#!/usr/bin/env python3
"""Verify FullAttention layer 3 using GPU weights loaded on CPU and CPU-referenced input.

Avoids OOM by not loading the full model. Only loads layer 3 weights from safetensors
and the layer 2 output (hidden states) from the CPU reference checkpoint.
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

hidden_size = tc['hidden_size']
eps = tc['rms_norm_eps']
head_dim = tc['head_dim']
num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']
rp = tc.get('rope_parameters', tc.get('rope_scaling', {})) or {}
rotary_dim = int(head_dim * rp.get('partial_rotary_factor', 1.0))
mrope_section = rp.get('mrope_section')
rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)
max_pos = tc['max_position_embeddings']

def load_cpu(key):
    fname = wm[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

def qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + w)

# Build MRoPE cache (same as our implementation)
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

# Load layer 3 weights
prefix = 'model.language_model.layers.3.'
w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()
w_post_ln = load_cpu(prefix + 'post_attention_layernorm.weight').float()

# Split fused q_proj into Q and Gate
w_q_fused = load_cpu(prefix + 'self_attn.q_proj.weight').float()  # [12288, 5120]
full_q_size = num_heads * head_dim  # 6144
w_q = w_q_fused[:full_q_size, :]    # [6144, 5120]
w_gate = w_q_fused[full_q_size:, :] # [6144, 5120]
w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()  # [1024, 5120]
w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()  # [1024, 5120]
w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()  # [5120, 6144]
w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()  # [256]
w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()  # [256]

# MLP weights
w_gate_proj = load_cpu(prefix + 'mlp.gate_proj.weight').float()
w_up_proj = load_cpu(prefix + 'mlp.up_proj.weight').float()
w_down_proj = load_cpu(prefix + 'mlp.down_proj.weight').float()

# Load CPU reference: we need the input to layer 3
# From diag_cpu_layers05.py, the checkpoints store the hidden states AFTER each layer (mlp output)
ref = torch.load('/tmp/diag_cpu_layers05_ref.pt', map_location='cpu', weights_only=True)
cp_ref = ref['checkpoints']

# To get the input to layer 3, we need to reconstruct from layers 0-2 computation
# Or use a simpler approach: compute layer 3 from scratch using the layer 2 output
tokens = ref['tokens']
input_ids = torch.tensor([tokens], dtype=torch.long)
S = len(tokens)

# Option 1: Use the saved layer 2 hidden states + residual
# We need to reconstruct the full state at layer 3 input
# Let's compute layers 0-2 on CPU to get the exact input

# Load embedding
embed_key = 'model.language_model.embed_tokens.weight'
embed_w = load_cpu(embed_key).float()
emb = F.embedding(input_ids, embed_w)

positions = torch.arange(S, dtype=torch.int64)
hs = emb.clone()
residual = None

for layer_idx in range(3):
    pfx = f'model.language_model.layers.{layer_idx}.'
    w_iln = load_cpu(pfx + 'input_layernorm.weight').float()
    w_pln = load_cpu(pfx + 'post_attention_layernorm.weight').float()

    if residual is None:
        residual = hs.clone()
        hs_normed = qwen35_rms_norm(hs, w_iln)
    else:
        residual = residual + hs
        hs_normed = qwen35_rms_norm(residual, w_iln)
        hs.copy_(residual)

    B, T, H = hs_normed.shape

    # GatedDeltaNet layers 0-2
    w_in_qkv = load_cpu(pfx + 'linear_attn.in_proj_qkv.weight').float()
    w_conv1d = load_cpu(pfx + 'linear_attn.conv1d.weight').float()
    w_in_a = load_cpu(pfx + 'linear_attn.in_proj_a.weight').float()
    w_in_b = load_cpu(pfx + 'linear_attn.in_proj_b.weight').float()
    A_log = load_cpu(pfx + 'linear_attn.A_log').float()
    dt_bias = load_cpu(pfx + 'linear_attn.dt_bias').float()
    w_in_z = load_cpu(pfx + 'linear_attn.in_proj_z.weight').float()
    w_norm_gdn = load_cpu(pfx + 'linear_attn.norm.weight').float()
    w_out_gdn = load_cpu(pfx + 'linear_attn.out_proj.weight').float()

    lk_heads = tc['linear_num_key_heads']
    lv_heads = tc['linear_num_value_heads']
    lk_dim = tc['linear_key_head_dim']
    lv_dim = tc['linear_value_head_dim']

    mixed_qkv = F.linear(hs_normed, w_in_qkv)
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))
    conv_out = F.conv1d(mixed_qkv_pad, w_conv1d, bias=None, groups=mixed_qkv_t.shape[1])
    conv_out = F.silu(conv_out).transpose(1, 2)

    q_lin = conv_out[:, :, :lk_heads*lk_dim].view(B, T, lk_heads, lk_dim)
    k_lin = conv_out[:, :, lk_heads*lk_dim:2*lk_heads*lk_dim].view(B, T, lk_heads, lk_dim)
    v_lin = conv_out[:, :, 2*lk_heads*lk_dim:].view(B, T, lv_heads, lv_dim)

    a = F.linear(hs_normed, w_in_a)
    b = F.linear(hs_normed, w_in_b)
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    beta = torch.sigmoid(b)

    q_norm_gdn = F.normalize(q_lin.float(), p=2, dim=-1).to(q_lin.dtype)
    k_norm_gdn = F.normalize(k_lin.float(), p=2, dim=-1).to(k_lin.dtype)
    rpt = lv_heads // lk_heads
    if rpt > 1:
        q_norm_gdn = q_norm_gdn.repeat_interleave(rpt, dim=2)
        k_norm_gdn = k_norm_gdn.repeat_interleave(rpt, dim=2)
    q_norm_gdn = q_norm_gdn * (1.0 / math.sqrt(lk_dim))

    state = torch.zeros(B, lv_heads, lk_dim, lv_dim, dtype=torch.float32)
    core_out_gdn = torch.zeros(B, T, lv_heads, lv_dim, dtype=torch.float32)
    for t_idx in range(T):
        g_t = g[:, t_idx, :]
        k_t = k_norm_gdn[:, t_idx, :, :]
        v_t = v_lin[:, t_idx, :, :]
        q_t = q_norm_gdn[:, t_idx, :, :]
        beta_t = beta[:, t_idx, :]
        state = state * torch.exp(g_t.float())[:, :, None, None]
        kv_mem = torch.sum(state * k_t.float()[:, :, :, None], dim=-2)
        delta = (v_t.float() - kv_mem) * beta_t.float()[:, :, None]
        state = state + k_t.float()[:, :, :, None] * delta[:, :, None, :]
        o_t = torch.sum(state * q_t.float()[:, :, :, None], dim=-2)
        core_out_gdn[:, t_idx, :, :] = o_t

    z = F.linear(hs_normed, w_in_z).view(B, T, lv_heads, lv_dim)

    # Qwen3_5RMSNormGated
    core_flat = core_out_gdn.reshape(-1, lv_dim)
    z_flat = z.reshape(-1, lv_dim)
    rstd = 1.0 / torch.sqrt(core_flat.float().pow(2).mean(-1, keepdim=True) + eps)
    x_norm = core_flat.float() * rstd
    gated = (x_norm * w_norm_gdn.float() * F.silu(z_flat)).view(B, T, lv_heads * lv_dim)

    attn_out = F.linear(gated.float(), w_out_gdn)

    residual = residual + attn_out.float()
    hs_normed_mlp = qwen35_rms_norm(residual, w_pln)

    w_gate_mlp = load_cpu(pfx + 'mlp.gate_proj.weight').float()
    w_up_mlp = load_cpu(pfx + 'mlp.up_proj.weight').float()
    w_down_mlp = load_cpu(pfx + 'mlp.down_proj.weight').float()
    gate_h = F.linear(hs_normed_mlp, w_gate_mlp)
    up_h = F.linear(hs_normed_mlp, w_up_mlp)
    mlp_out = F.linear(F.silu(gate_h) * up_h.float(), w_down_mlp).float()
    hs = mlp_out

print(f"Layer 2 output (CPU): norm={hs.norm():.4f}")
print(f"Residual after layer 2 (CPU): norm={residual.norm():.4f}")

# Now compute layer 3 FullAttention
residual = residual + hs  # Add mlp_2 to residual
hs_input = qwen35_rms_norm(residual, w_input_ln)  # Input to layer 3 attention

print(f"Layer 3 input norm (CPU): {hs_input.norm():.4f}")

# FullAttention forward
B, T, H = 1, S, hidden_size

# Q, Gate, K, V projections (FULL, no TP sharding)
q_cpu = F.linear(hs_input, w_q)  # [1, 7, 6144]
gate_cpu = F.linear(hs_input, w_gate)  # [1, 7, 6144]
k_cpu = F.linear(hs_input, w_k)  # [1, 7, 1024]
v_cpu = F.linear(hs_input, w_v)  # [1, 7, 1024]

q_cpu = q_cpu.reshape(B, T, num_heads, head_dim)  # [1, 7, 24, 256]
k_cpu = k_cpu.reshape(B, T, num_kv_heads, head_dim)  # [1, 7, 4, 256]
v_cpu = v_cpu.reshape(B, T, num_kv_heads, head_dim)
gate_cpu = gate_cpu.reshape(B, T, num_heads, head_dim)

# Q/K norms
q_cpu_normed = qwen35_rms_norm(q_cpu.float(), w_qn.unsqueeze(0).unsqueeze(0))
k_cpu_normed = qwen35_rms_norm(k_cpu.float(), w_kn.unsqueeze(0).unsqueeze(0))

# MRoPE
q_cpu_rope = q_cpu_normed.clone()
k_cpu_rope = k_cpu_normed.clone()
pos_c = torch.arange(S, dtype=torch.int64)
cos = cos_sin_cache[pos_c, :rotary_dim//2]
sin = cos_sin_cache[pos_c, rotary_dim//2:]
cos = cos.view(1, S, 1, rotary_dim//2)
sin = sin.view(1, S, 1, rotary_dim//2)
cos_dup = torch.cat([cos, cos], dim=-1)
sin_dup = torch.cat([sin, sin], dim=-1)

q_rot = q_cpu_rope[..., :rotary_dim].float()
k_rot = k_cpu_rope[..., :rotary_dim].float()
half_r = rotary_dim // 2
q1, q2 = q_rot[..., :half_r], q_rot[..., half_r:]
q_cpu_rope[..., :rotary_dim] = (q_rot * cos_dup + torch.cat([-q2, q1], dim=-1) * sin_dup)
k1, k2 = k_rot[..., :half_r], k_rot[..., half_r:]
k_cpu_rope[..., :rotary_dim] = (k_rot * cos_dup + torch.cat([-k2, k1], dim=-1) * sin_dup)

# SDPA (GQA: expand KV to match Q)
n_groups = num_heads // num_kv_heads  # 6
q_attn = q_cpu_rope.transpose(1, 2)  # [1, 24, 7, 256]
k_attn = k_cpu_rope.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads, T, head_dim)  # [1, 24, 7, 256]
v_attn = v_cpu.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, num_heads, T, head_dim)

scale = head_dim ** -0.5
attn_out_cpu = F.scaled_dot_product_attention(
    q_attn.float(), k_attn.float(), v_attn.float(),
    is_causal=True, scale=scale
).transpose(1, 2).reshape(B, T, num_heads * head_dim)  # [1, 7, 6144]

# Output gate
gate_flat = gate_cpu.reshape(B, T, num_heads * head_dim)
attn_gated_cpu = attn_out_cpu.float() * torch.sigmoid(gate_flat.float())

# o_proj
layer3_out_cpu = F.linear(attn_gated_cpu, w_o)  # [1, 7, 5120]

print(f"\nLayer 3 FullAttention output (CPU, TP=1): norm={layer3_out_cpu.norm():.4f}")
print(f"  min={layer3_out_cpu.min():.4f}, max={layer3_out_cpu.max():.4f}, mean={layer3_out_cpu.mean():.4f}")

# Continue with post_ln and MLP
residual_full = residual + layer3_out_cpu.float()
hs_mlp_input = qwen35_rms_norm(residual_full, w_post_ln)

gate_h = F.linear(hs_mlp_input, w_gate_proj)
up_h = F.linear(hs_mlp_input, w_up_proj)
mlp_out_cpu = F.linear(F.silu(gate_h) * up_h.float(), w_down_proj).float()

print(f"\nLayer 3 MLP output (CPU, TP=1): norm={mlp_out_cpu.norm():.4f}")
print(f"  (This should match the GPU output norm ~15.27 from norms_compare)")
print(f"  (CPU reference norm was: {cp_ref[3]['norm']:.4f})")

# Save for GPU comparison
torch.save({
    'hs_input': hs_input,  # Normed input to layer 3 attention
    'q_proj': q_cpu,
    'k_proj': k_cpu,
    'v_proj': v_cpu,
    'gate_proj': gate_cpu,
    'q_normed': q_cpu_normed,
    'k_normed': k_cpu_normed,
    'q_rope': q_cpu_rope,
    'k_rope': k_cpu_rope,
    'attn_out': attn_gated_cpu,
    'layer3_out': layer3_out_cpu,
    'mlp_out': mlp_out_cpu,
    'residual_before_ln': residual,
}, '/tmp/diag_fullattn_cpu_ref.pt')
print(f"\nSaved detailed reference to /tmp/diag_fullattn_cpu_ref.pt")
