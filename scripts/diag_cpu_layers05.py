#!/usr/bin/env python3
"""CPU reference for layers 0-5; save checkpoints for GPU comparison."""
import os, sys, torch, json, math
import torch.nn.functional as F

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

print("=== Loading weights ===")

with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
weight_map = idx['weight_map']

# Pre-load all needed tensors to avoid repeated file opens
needed_keys = set()
for layer_idx in range(6):  # layers 0-5
    prefix = f'model.language_model.layers.{layer_idx}.'
    is_full = layer_idx in {3}  # Only layer 3 is FullAttention in layers 0-5

    needed_keys.add(prefix + 'input_layernorm.weight')
    needed_keys.add(prefix + 'post_attention_layernorm.weight')
    needed_keys.add(prefix + 'mlp.gate_proj.weight')
    needed_keys.add(prefix + 'mlp.up_proj.weight')
    needed_keys.add(prefix + 'mlp.down_proj.weight')

    if is_full:
        needed_keys.add(prefix + 'self_attn.q_proj.weight')
        needed_keys.add(prefix + 'self_attn.k_proj.weight')
        needed_keys.add(prefix + 'self_attn.v_proj.weight')
        needed_keys.add(prefix + 'self_attn.o_proj.weight')
        needed_keys.add(prefix + 'self_attn.q_norm.weight')
        needed_keys.add(prefix + 'self_attn.k_norm.weight')
    else:
        needed_keys.add(prefix + 'linear_attn.in_proj_qkv.weight')
        needed_keys.add(prefix + 'linear_attn.conv1d.weight')
        needed_keys.add(prefix + 'linear_attn.in_proj_a.weight')
        needed_keys.add(prefix + 'linear_attn.in_proj_b.weight')
        needed_keys.add(prefix + 'linear_attn.A_log')
        needed_keys.add(prefix + 'linear_attn.dt_bias')
        needed_keys.add(prefix + 'linear_attn.in_proj_z.weight')
        needed_keys.add(prefix + 'linear_attn.norm.weight')
        needed_keys.add(prefix + 'linear_attn.out_proj.weight')

needed_keys.add('model.language_model.embed_tokens.weight')

# Group by safetensors file for efficient loading
files_needed = set()
for k in needed_keys:
    files_needed.add(os.path.join(model_dir, weight_map[k]))

# Load all files at once
loaded_tensors = {}
for fpath in sorted(files_needed):
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            if k in needed_keys:
                loaded_tensors[k] = sf.get_tensor(k)

print(f"Loaded {len(loaded_tensors)} tensors from {len(files_needed)} files")

def get_tensor(key):
    return loaded_tensors[key].float()

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
mrope_section = rp.get('mrope_section', None)
mrope_interleaved = rp.get('mrope_interleaved', False)
rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)
linear_k_heads = tc['linear_num_key_heads']
linear_v_heads = tc['linear_num_value_heads']
linear_k_dim = tc['linear_key_head_dim']
linear_v_dim = tc['linear_value_head_dim']
conv_kernel = tc['linear_conv_kernel_dim']
layer_types = tc['layer_types']

full_attn = {i for i, lt in enumerate(layer_types) if lt == 'full_attention'}
print(f"FullAttention layers: {sorted(full_attn)}")

def qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + w)

def qwen35_rms_norm_gated(x, g, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    x_norm = x.float() * rstd
    return (x_norm * w.float() * F.silu(g.float())).float()

# MRoPE cache
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

cos_sin = build_mrope_cache(tc['max_position_embeddings'], rotary_dim, rope_theta, mrope_section)

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

# Forward pass
embed_w = get_tensor('model.language_model.embed_tokens.weight')
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long)
positions = torch.arange(len(tokens), dtype=torch.int64)

emb = F.embedding(input_ids, embed_w)
print(f"\nembedding: norm={emb.norm():.4f}")

hs = emb.clone()
residual = None
checkpoints = {}

for layer_idx in range(6):
    prefix = f'model.language_model.layers.{layer_idx}.'
    is_full = layer_idx in full_attn

    w_input_ln = get_tensor(prefix + 'input_layernorm.weight')
    w_post_ln = get_tensor(prefix + 'post_attention_layernorm.weight')

    # input_layernorm
    if residual is None:
        residual = hs.clone()
        hs_normed = qwen35_rms_norm(hs, w_input_ln)
    else:
        residual = residual + hs
        hs_normed = qwen35_rms_norm(residual, w_input_ln)
        hs.copy_(residual)

    B, T, H = hs_normed.shape

    if is_full:
        w_q = get_tensor(prefix + 'self_attn.q_proj.weight')
        w_k = get_tensor(prefix + 'self_attn.k_proj.weight')
        w_v = get_tensor(prefix + 'self_attn.v_proj.weight')
        w_o = get_tensor(prefix + 'self_attn.o_proj.weight')
        w_qn = get_tensor(prefix + 'self_attn.q_norm.weight')
        w_kn = get_tensor(prefix + 'self_attn.k_norm.weight')

        q_full = F.linear(hs_normed, w_q)
        q, gate = torch.chunk(q_full, 2, dim=-1)
        k = F.linear(hs_normed, w_k)
        v = F.linear(hs_normed, w_v)

        q = q.view(B, T, num_heads, head_dim)
        k = k.view(B, T, num_kv_heads, head_dim)
        v = v.view(B, T, num_kv_heads, head_dim)
        gate = gate.view(B, T, num_heads, head_dim)

        q = qwen35_rms_norm(q.float(), w_qn.unsqueeze(0).unsqueeze(0)).to(q.dtype)
        k = qwen35_rms_norm(k.float(), w_kn.unsqueeze(0).unsqueeze(0)).to(k.dtype)

        apply_mrope(q, k, positions)

        # GQA: repeat KV to match Q heads (24 Q heads, 4 KV heads)
        n_groups = num_heads // num_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, T, num_heads, head_dim)
        v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(B, T, num_heads, head_dim)

        q_attn = q.transpose(1, 2)
        k_attn = k.transpose(1, 2)
        v_attn = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q_attn.float(), k_attn.float(), v_attn.float(),
            attn_mask=None, dropout_p=0.0, is_causal=True
        ).transpose(1, 2).to(q.dtype)

        attn_out = attn_out.reshape(B, T, num_heads * head_dim)
        gate_out = torch.sigmoid(gate.reshape(B, T, num_heads * head_dim))
        attn_out = attn_out * gate_out
        attn_out = F.linear(attn_out, w_o)
    else:
        w_in_qkv = get_tensor(prefix + 'linear_attn.in_proj_qkv.weight')
        w_conv1d = get_tensor(prefix + 'linear_attn.conv1d.weight')
        w_in_a = get_tensor(prefix + 'linear_attn.in_proj_a.weight')
        w_in_b = get_tensor(prefix + 'linear_attn.in_proj_b.weight')
        A_log = get_tensor(prefix + 'linear_attn.A_log')
        dt_bias = get_tensor(prefix + 'linear_attn.dt_bias')
        w_in_z = get_tensor(prefix + 'linear_attn.in_proj_z.weight')
        w_norm = get_tensor(prefix + 'linear_attn.norm.weight')
        w_out = get_tensor(prefix + 'linear_attn.out_proj.weight')

        mixed_qkv = F.linear(hs_normed, w_in_qkv)
        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))
        conv_out = F.conv1d(mixed_qkv_pad, w_conv1d, bias=None, groups=mixed_qkv_t.shape[1])
        conv_out = F.silu(conv_out).transpose(1, 2)

        q = conv_out[:, :, :linear_k_heads*linear_k_dim].view(B, T, linear_k_heads, linear_k_dim)
        k = conv_out[:, :, linear_k_heads*linear_k_dim:2*linear_k_heads*linear_k_dim].view(B, T, linear_k_heads, linear_k_dim)
        v = conv_out[:, :, 2*linear_k_heads*linear_k_dim:].view(B, T, linear_v_heads, linear_v_dim)

        a = F.linear(hs_normed, w_in_a)
        b = F.linear(hs_normed, w_in_b)

        g = -torch.exp(A_log) * F.softplus(a + dt_bias)
        beta = torch.sigmoid(b)

        q_norm = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k_norm = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)

        rpt = linear_v_heads // linear_k_heads
        if rpt > 1:
            q_norm = q_norm.repeat_interleave(rpt, dim=2)
            k_norm = k_norm.repeat_interleave(rpt, dim=2)

        q_norm = q_norm * (1.0 / math.sqrt(linear_k_dim))

        state = torch.zeros(B, linear_v_heads, linear_k_dim, linear_v_dim, dtype=torch.float32)
        core_out = torch.zeros(B, T, linear_v_heads, linear_v_dim, dtype=torch.float32)

        for t_idx in range(T):
            g_t = g[:, t_idx, :]
            k_t = k_norm[:, t_idx, :, :]
            v_t = v[:, t_idx, :, :]
            q_t = q_norm[:, t_idx, :, :]
            beta_t = beta[:, t_idx, :]

            state = state * torch.exp(g_t.float())[:, :, None, None]
            kv_mem = torch.sum(state * k_t.float()[:, :, :, None], dim=-2)
            delta = (v_t.float() - kv_mem) * beta_t.float()[:, :, None]
            state = state + k_t.float()[:, :, :, None] * delta[:, :, None, :]
            o_t = torch.sum(state * q_t.float()[:, :, :, None], dim=-2)
            core_out[:, t_idx, :, :] = o_t

        z = F.linear(hs_normed, w_in_z).view(B, T, linear_v_heads, linear_v_dim)
        gated_out = qwen35_rms_norm_gated(
            core_out.reshape(-1, linear_v_dim),
            z.reshape(-1, linear_v_dim),
            w_norm.float()
        ).view(B, T, linear_v_heads * linear_v_dim)

        attn_out = F.linear(gated_out.float(), w_out)

    # post_attention_layernorm
    residual = residual + attn_out.float()
    hs_normed_mlp = qwen35_rms_norm(residual, w_post_ln)

    # MLP
    w_gate = get_tensor(prefix + 'mlp.gate_proj.weight')
    w_up = get_tensor(prefix + 'mlp.up_proj.weight')
    w_down = get_tensor(prefix + 'mlp.down_proj.weight')

    gate_h = F.linear(hs_normed_mlp, w_gate)
    up_h = F.linear(hs_normed_mlp, w_up)
    mlp_out = F.linear(F.silu(gate_h) * up_h.float(), w_down).float()

    hs = mlp_out

    # Save checkpoint
    checkpoints[layer_idx] = {
        'hs': hs.clone().detach(),
        'norm': float(hs.norm()),
        'is_full': is_full,
    }

    print(f"Layer {layer_idx} ({'FULL' if is_full else 'LINE'}): "
          f"attn_out_norm={attn_out.norm():.2f}, mlp_out_norm={mlp_out.norm():.2f}")

# Save
torch.save({
    'checkpoints': checkpoints,
    'tokens': tokens,
    'embedding': emb.clone().detach(),
}, '/tmp/diag_cpu_layers05_ref.pt')
print(f"\nSaved reference to /tmp/diag_cpu_layers05_ref.pt")
