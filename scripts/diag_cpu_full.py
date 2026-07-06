#!/usr/bin/env python3
"""Trace model through first few layers; compare with CPU reference at each FullAttention layer.

Computes sequential CPU reference for all layers (TP=1) and compares with GPU (TP=4) at
key checkpoints: layers 0, 2, 3 (first FullAttention), and final output.
"""
import os, sys, torch, json, math
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

# ============================================================
# CPU Reference: full model forward (TP=1)
# ============================================================
print("=== CPU Reference: Loading weights ===")

with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
weight_map = idx['weight_map']

def load_cpu(key):
    fname = weight_map[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)

hidden_size = tc['hidden_size']
num_layers = tc['num_hidden_layers']
eps = tc['rms_norm_eps']
head_dim = tc['head_dim']
num_heads = tc['num_attention_heads']
num_kv_heads = tc['num_key_value_heads']
rotary_dim = int(head_dim * tc.get('rope_parameters', tc.get('rope_scaling', {})).get('partial_rotary_factor', 1.0))
mrope_section = tc.get('rope_parameters', tc.get('rope_scaling', {})).get('mrope_section', None)
mrope_interleaved = tc.get('rope_parameters', tc.get('rope_scaling', {})).get('mrope_interleaved', False)
rope_theta = tc.get('rope_theta') or tc.get('rope_parameters', {}).get('rope_theta', 1000000.0)

linear_k_heads = tc['linear_num_key_heads']
linear_v_heads = tc['linear_num_value_heads']
linear_k_dim = tc['linear_key_head_dim']
linear_v_dim = tc['linear_value_head_dim']
conv_kernel = tc['linear_conv_kernel_dim']
layer_types = tc['layer_types']

# Build layer type map
full_attention_layers = set()
for i, lt in enumerate(layer_types):
    if lt == 'full_attention':
        full_attention_layers.add(i)

print(f"Hidden: {hidden_size}, Layers: {num_layers}")
print(f"FullAttention layers: {sorted(full_attention_layers)}")
print(f"GatedDeltaNet: k_heads={linear_k_heads}, v_heads={linear_v_heads}, k_dim={linear_k_dim}, v_dim={linear_v_dim}")

# Qwen3_5RMSNorm
def qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + w)

# Standard RMSNorm
def std_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * w

# Qwen3_5RMSNormGated
def qwen35_rms_norm_gated(x, g, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    x_norm = x.float() * rstd
    return (x_norm * w.float() * F.silu(g.float())).float()

# Build MRoPE cos/sin cache (same as our implementation)
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
    return torch.cat((cos, sin), dim=-1)  # [max_pos, head_size]

cos_sin_cache = build_mrope_cache(tc['max_position_embeddings'], rotary_dim, rope_theta, mrope_section)

def apply_mrope(q, k, positions, head_dim_full, rotary_dim):
    """Apply MRoPE to q and k in-place. Only first rotary_dim dimensions are rotated."""
    B, S = q.shape[0], q.shape[1]

    cos = cos_sin_cache[positions, :rotary_dim//2]  # [S, rotary_dim//2]
    sin = cos_sin_cache[positions, rotary_dim//2:]  # [S, rotary_dim//2]

    # Expand for broadcast
    cos = cos.view(1, S, 1, rotary_dim//2)
    sin = sin.view(1, S, 1, rotary_dim//2)

    # NeoX style
    cos = torch.cat([cos, cos], dim=-1)  # [1, S, 1, rotary_dim]
    sin = torch.cat([sin, sin], dim=-1)

    # Apply to q
    q_rot = q[..., :rotary_dim].float()
    half_rot = rotary_dim // 2
    q1, q2 = q_rot[..., :half_rot], q_rot[..., half_rot:]
    q_rotated = torch.cat([-q2, q1], dim=-1)
    q_out = q_rot * cos + q_rotated * sin
    q[..., :rotary_dim] = q_out.to(q.dtype)

    # Apply to k
    k_rot = k[..., :rotary_dim].float()
    k1, k2 = k_rot[..., :half_rot], k_rot[..., half_rot:]
    k_rotated = torch.cat([-k2, k1], dim=-1)
    k_out = k_rot * cos + k_rotated * sin
    k[..., :rotary_dim] = k_out.to(k.dtype)

embed_w = load_cpu('model.language_model.embed_tokens.weight').float()

# Input
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]  # 苏州园林的特点是讲究
input_ids = torch.tensor([tokens], dtype=torch.long)
S = len(tokens)
positions = torch.arange(S, dtype=torch.int64)

emb = F.embedding(input_ids, embed_w)  # [1, 7, 5120]
print(f"embedding: norm={emb.norm():.4f}")

hs = emb.clone()
residual = None
checkpoints = {0: 'after_layer0', 2: 'after_layer2', 3: 'after_layer3'}
cp_results = {}

for layer_idx in range(num_layers):
    prefix = f'model.language_model.layers.{layer_idx}.'
    is_full = layer_idx in full_attention_layers

    # Load weights for this layer
    w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()
    w_post_ln = load_cpu(prefix + 'post_attention_layernorm.weight').float()

    # --- input_layernorm ---
    if residual is None:
        residual = hs.clone()
        hs_normed = qwen35_rms_norm(hs, w_input_ln)
    else:
        residual = residual + hs
        hs_normed = qwen35_rms_norm(residual, w_input_ln)
        hs.copy_(residual)  # hs now holds the accumulated residual

    B, T, H = hs_normed.shape

    if is_full:
        # === FullAttention forward ===
        w_q = load_cpu(prefix + 'self_attn.q_proj.weight').float()
        w_k = load_cpu(prefix + 'self_attn.k_proj.weight').float()
        w_v = load_cpu(prefix + 'self_attn.v_proj.weight').float()
        w_o = load_cpu(prefix + 'self_attn.o_proj.weight').float()
        w_qn = load_cpu(prefix + 'self_attn.q_norm.weight').float()
        w_kn = load_cpu(prefix + 'self_attn.k_norm.weight').float()

        # Q/K/V proj
        q_full = F.linear(hs_normed, w_q)  # [1, 7, num_heads*head_dim*2]
        q, gate = torch.chunk(q_full, 2, dim=-1)  # q: [1,7,num_heads*head_dim], gate: same
        k = F.linear(hs_normed, w_k)  # [1,7,num_kv_heads*head_dim]
        v = F.linear(hs_normed, w_v)

        q = q.view(B, T, num_heads, head_dim)
        k = k.view(B, T, num_kv_heads, head_dim)
        v = v.view(B, T, num_kv_heads, head_dim)
        gate = gate.view(B, T, num_heads, head_dim)

        # Q/K norms (Qwen3_5RMSNorm per-head, last dim)
        q = qwen35_rms_norm(q.float(), w_qn.unsqueeze(0).unsqueeze(0)).to(q.dtype)
        k = qwen35_rms_norm(k.float(), w_kn.unsqueeze(0).unsqueeze(0)).to(k.dtype)

        # MRoPE
        apply_mrope(q, k, positions, head_dim, rotary_dim)

        # SDPA attention
        scale = head_dim ** -0.5
        q_attn = q.transpose(1, 2)  # [1, num_heads, 7, head_dim]
        k_attn = k.transpose(1, 2)  # [1, num_kv_heads, 7, head_dim]
        v_attn = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q_attn.float(), k_attn.float(), v_attn.float(),
            attn_mask=None, dropout_p=0.0, is_causal=True
        ).transpose(1, 2).to(q.dtype)  # [1, 7, num_heads, head_dim]

        attn_out = attn_out.reshape(B, T, num_heads * head_dim)

        # Output gate
        gate_out = torch.sigmoid(gate.reshape(B, T, num_heads * head_dim))
        attn_out = attn_out * gate_out

        # O proj
        attn_out = F.linear(attn_out, w_o)

    else:
        # === GatedDeltaNet forward ===
        w_in_qkv = load_cpu(prefix + 'linear_attn.in_proj_qkv.weight').float()
        w_conv1d = load_cpu(prefix + 'linear_attn.conv1d.weight').float()
        w_in_a = load_cpu(prefix + 'linear_attn.in_proj_a.weight').float()
        w_in_b = load_cpu(prefix + 'linear_attn.in_proj_b.weight').float()
        A_log = load_cpu(prefix + 'linear_attn.A_log').float()
        dt_bias = load_cpu(prefix + 'linear_attn.dt_bias').float()
        w_in_z = load_cpu(prefix + 'linear_attn.in_proj_z.weight').float()
        w_norm = load_cpu(prefix + 'linear_attn.norm.weight').float()
        w_out = load_cpu(prefix + 'linear_attn.out_proj.weight').float()

        # 1. in_proj_qkv
        mixed_qkv = F.linear(hs_normed, w_in_qkv)

        # 2. Causal conv1d
        mixed_qkv_t = mixed_qkv.transpose(1, 2)
        mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))
        conv_out = F.conv1d(mixed_qkv_pad, w_conv1d, bias=None, groups=mixed_qkv_t.shape[1])
        conv_out = F.silu(conv_out).transpose(1, 2)

        # 3. Split Q, K, V
        q = conv_out[:, :, :linear_k_heads*linear_k_dim].view(B, T, linear_k_heads, linear_k_dim)
        k = conv_out[:, :, linear_k_heads*linear_k_dim:2*linear_k_heads*linear_k_dim].view(B, T, linear_k_heads, linear_k_dim)
        v = conv_out[:, :, 2*linear_k_heads*linear_k_dim:].view(B, T, linear_v_heads, linear_v_dim)

        # 4. in_proj_a, in_proj_b
        a = F.linear(hs_normed, w_in_a)
        b = F.linear(hs_normed, w_in_b)

        # 5. Gate and beta
        g = -torch.exp(A_log) * F.softplus(a + dt_bias)
        beta = torch.sigmoid(b)

        # 6. Q/K L2 norm
        q_norm = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k_norm = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)

        # Repeat to match v heads
        repeat_factor = linear_v_heads // linear_k_heads
        if repeat_factor > 1:
            q_norm = q_norm.repeat_interleave(repeat_factor, dim=2)
            k_norm = k_norm.repeat_interleave(repeat_factor, dim=2)

        q_scale = 1.0 / math.sqrt(linear_k_dim)
        q_norm = q_norm * q_scale

        # 7. Recurrence
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

        # 8. in_proj_z + gated norm
        z = F.linear(hs_normed, w_in_z).view(B, T, linear_v_heads, linear_v_dim)
        gated_out = qwen35_rms_norm_gated(
            core_out.reshape(-1, linear_v_dim),
            z.reshape(-1, linear_v_dim),
            w_norm.float()
        ).view(B, T, linear_v_heads * linear_v_dim)

        # 9. out_proj
        attn_out = F.linear(gated_out.float(), w_out)

    # --- post_attention_layernorm ---
    residual = residual + attn_out.float()
    hs_normed_mlp = qwen35_rms_norm(residual, w_post_ln)

    # --- MLP ---
    w_gate = load_cpu(prefix + 'mlp.gate_proj.weight').float()
    w_up = load_cpu(prefix + 'mlp.up_proj.weight').float()
    w_down = load_cpu(prefix + 'mlp.down_proj.weight').float()

    gate_h = F.linear(hs_normed_mlp, w_gate)
    up_h = F.linear(hs_normed_mlp, w_up)
    mlp_out = F.linear(F.silu(gate_h) * up_h.float(), w_down).float()

    # Update hs for next layer
    hs = mlp_out

    # Checkpoint
    if layer_idx in checkpoints:
        cp_results[layer_idx] = {
            'hs': hs.clone().detach(),  # mlp output (to be used as input to next layer)
            'norm': float(hs.norm()),
            'label': checkpoints[layer_idx],
        }

    if layer_idx < 5:
        print(f"Layer {layer_idx} ({'full' if is_full else 'line'}): "
              f"attn_norm={attn_out.norm():.2f} -> mlp_norm={mlp_out.norm():.2f}")

# Final: add residual and norm
if residual is not None:
    hs = hs + residual

w_final_norm = load_cpu('model.language_model.norm.weight').float()
hs_final = qwen35_rms_norm(hs, w_final_norm)

w_lm_head = load_cpu('model.language_model.lm_head.weight').float()
logits_cpu = F.linear(hs_final, w_lm_head)
last_logits = logits_cpu[0, -1, :]

print(f"\n=== CPU Final Output ===")
print(f"Final norm: {hs_final.norm():.4f}")
print(f"Logits: min={last_logits.min():.4f}, max={last_logits.max():.4f}, mean={last_logits.mean():.4f}")
topk = torch.topk(last_logits, 10)
print("Top-10:")
for i in range(10):
    print(f"  #{i+1}: id={topk.indices[i].item()}, logit={topk.values[i].item():.4f}")

# Save checkpoints for GPU comparison
torch.save({
    'checkpoints': cp_results,
    'final_logits': last_logits.cpu(),
    'final_norm': float(hs_final.norm()),
    'tokens': tokens,
    'embedding': emb.clone().detach(),
}, '/tmp/diag_cpu_full_ref.pt')
print(f"\nSaved CPU reference to /tmp/diag_cpu_full_ref.pt")
print("Now run TP model and compare.")
