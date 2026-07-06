#!/usr/bin/env python3
"""CPU replica of the engine's FullAttention (layer 3), comparing with correct reference.

This script mimics EXACTLY what the engine does:
  - Separate Q/Gate/K/V projections (from split HF weights)
  - Qwen3_5RMSNorm on Q/K (weight=zeros, output=(1+w)*rms_norm(x))
  - MRoPE using cos_sin_cache with mrope_interleaved (on rotary_dim=64 only)
  - SDPA with repeat_interleave GQA
  - Output gate (sigmoid)
  - o_proj

Then compares norms at every substep against the correct reference from
/tmp/diag_fa_correct_ref.pt (which was computed by diag_save_correct_ref.py).
"""
import os, sys, torch, json, math
import torch.nn.functional as F

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

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
mrope_interleaved = rp.get('mrope_interleaved', False)
rope_theta = tc.get('rope_theta') or rp.get('rope_theta', 1000000.0)
max_pos = tc['max_position_embeddings']

print(f"Config: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")
print(f"MRoPE: rotary_dim={rotary_dim}, mrope_section={mrope_section}, interleaved={mrope_interleaved}")
print(f"rope_theta={rope_theta}")

# Load correct reference
ref = torch.load('/tmp/diag_fa_correct_ref.pt', map_location='cpu', weights_only=True)
hs_input = ref['hs_input']  # [1, 7, 5120] — normed input to layer 3 FullAttention
print(f"\nhs_input norm: {hs_input.norm():.4f}")

# =====================================================================
# Build MRoPE cache — EXACTLY as engine does via make_cos_sin_cache
# =====================================================================
def build_mrope_engine(max_pos, head_size, rope_theta, mrope_section, mrope_interleaved, dtype):
    """Matches engine's make_cos_sin_cache with mrope_interleaved=True."""
    num_sections = len(mrope_section)
    half_dim = head_size // 2
    total_section_sum = sum(mrope_section)
    assert total_section_sum == half_dim, f"{total_section_sum} != {half_dim}"

    section_freqs = []
    t = torch.arange(max_pos, dtype=torch.float32)
    for section_size in mrope_section:
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, section_size, dtype=torch.float32) * 2 / head_size))
        freqs_s = torch.einsum("i,j->ij", t, inv_freq)
        section_freqs.append(freqs_s)

    # Interleaved: dimension-pairs from different sections are round-robin
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
        assert found, f"No available section at interleave pos {i}"
    freqs = torch.stack(interleaved_pairs, dim=-1)  # [max_pos, half_dim]

    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return torch.cat((cos, sin), dim=-1)  # [max_pos, head_size]

cos_sin_engine = build_mrope_engine(max_pos, rotary_dim, rope_theta, mrope_section, mrope_interleaved, torch.float32)

def apply_rope_engine(x, positions, cos_sin_cache, rotary_dim):
    """Apply RoPE in-place — matches engine's rotary_embedding (_rope_neox fallback)."""
    half_head = rotary_dim // 2
    pos = positions.long()
    cos = cos_sin_cache[pos, :half_head].unsqueeze(1)  # [S, 1, half_head]
    sin = cos_sin_cache[pos, half_head:].unsqueeze(1)  # [S, 1, half_head]
    # NeoX: duplicate cos/sin
    cos = torch.cat([cos, cos], dim=-1).float()
    sin = torch.cat([sin, sin], dim=-1).float()

    x_f = x.float()
    x1, x2 = x_f[..., :half_head], x_f[..., half_head:]
    rotated = torch.cat([-x2, x1], dim=-1)
    result = (x_f * cos + rotated * sin).to(x.dtype)
    x.copy_(result)

# =====================================================================
# Load FullAttention weights for layer 3 (full scale — no TP)
# =====================================================================
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
wm = idx['weight_map']

pfx3 = 'model.language_model.layers.3.'
needed = [
    pfx3 + 'self_attn.q_proj.weight',   # [12288, 5120] — fused Q + Gate
    pfx3 + 'self_attn.k_proj.weight',   # [1024, 5120]
    pfx3 + 'self_attn.v_proj.weight',   # [1024, 5120]
    pfx3 + 'self_attn.o_proj.weight',   # [5120, 6144]
    pfx3 + 'self_attn.q_norm.weight',   # [256]
    pfx3 + 'self_attn.k_norm.weight',   # [256]
]

files_needed = set(os.path.join(model_dir, wm[k]) for k in needed)
loaded = {}
for fpath in sorted(files_needed):
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            if k in needed:
                loaded[k] = sf.get_tensor(k).float()

def w(k): return loaded[k]

full_q_size = num_heads * head_dim  # 6144
W_q_fused = w(pfx3 + 'self_attn.q_proj.weight')  # [12288, 5120]
W_q = W_q_fused[:full_q_size, :]     # [6144, 5120]
W_gate = W_q_fused[full_q_size:, :]  # [6144, 5120]
W_k = w(pfx3 + 'self_attn.k_proj.weight')    # [1024, 5120]
W_v = w(pfx3 + 'self_attn.v_proj.weight')    # [1024, 5120]
W_o = w(pfx3 + 'self_attn.o_proj.weight')    # [5120, 6144]
w_qn = w(pfx3 + 'self_attn.q_norm.weight')  # [256]
w_kn = w(pfx3 + 'self_attn.k_norm.weight')  # [256]

print(f"\nq_norm weight: min={w_qn.min():.6f}, max={w_qn.max():.6f}, mean={w_qn.mean():.6f}")
print(f"k_norm weight: min={w_kn.min():.6f}, max={w_kn.max():.6f}, mean={w_kn.mean():.6f}")

# =====================================================================
# Engine-equivalent FullAttention computation (full scale)
# =====================================================================

B, T = 1, 7
positions = torch.arange(T, dtype=torch.int64)

# Step 1: Separate projections (EXACTLY as engine does)
q = F.linear(hs_input, W_q).view(B, T, num_heads, head_dim)       # [1, 7, 24, 256]
gate = F.linear(hs_input, W_gate).view(B, T, num_heads, head_dim) # [1, 7, 24, 256]
k = F.linear(hs_input, W_k).view(B, T, num_kv_heads, head_dim)    # [1, 7, 4, 256]
v = F.linear(hs_input, W_v).view(B, T, num_kv_heads, head_dim)    # [1, 7, 4, 256]

print(f"\n--- Step 1: Projections ---")
print(f"  Q norm: {q.norm():.4f} (ref: {ref['q_proj'].norm():.4f})")
print(f"  K norm: {k.norm():.4f} (ref: {ref['k_proj'].norm():.4f})")
print(f"  V norm: {v.norm():.4f} (ref: {ref['v_proj'].norm():.4f})")
print(f"  Gate norm: {gate.norm():.4f} (ref: {ref['gate_proj'].norm():.4f})")

# Step 2: Q/K norms — Qwen3_5RMSNorm: (1+w) * rms_norm(x)
def qwen35_rms_norm_engine(x, w):
    """Matches engine's Qwen3_5RMSNorm (via rms_norm fallback)."""
    x_fp32 = x.float()
    rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    eff_w = (1.0 + w.float()).view(*([1]*(x.ndim-1)), -1)  # broadcast
    return (eff_w * (x_fp32 / rms)).to(x.dtype)

q_normed = qwen35_rms_norm_engine(q, w_qn)  # [1, 7, 24, 256]
k_normed = qwen35_rms_norm_engine(k, w_kn)  # [1, 7, 4, 256]

print(f"\n--- Step 2: Q/K Norms (Qwen3_5RMSNorm) ---")
print(f"  Q_normed norm: {q_normed.norm():.4f} (ref: {ref['q_normed'].norm():.4f})")
print(f"  K_normed norm: {k_normed.norm():.4f} (ref: {ref['k_normed'].norm():.4f})")

# Step 3: MRoPE (in-place on rotary_dim only)
q_rope = q_normed.clone()
k_rope = k_normed.clone()
q_rot = q_rope[..., :rotary_dim].contiguous()  # [1, 7, 24, 64]
k_rot = k_rope[..., :rotary_dim].contiguous()  # [1, 7, 4, 64]
apply_rope_engine(q_rot, positions, cos_sin_engine, rotary_dim)
apply_rope_engine(k_rot, positions, cos_sin_engine, rotary_dim)
q_rope[..., :rotary_dim] = q_rot
k_rope[..., :rotary_dim] = k_rot

print(f"\n--- Step 3: MRoPE ---")
print(f"  Q_rope norm: {q_rope.norm():.4f} (ref: {ref['q_rope'].norm():.4f})")
print(f"  K_rope norm: {k_rope.norm():.4f} (ref: {ref['k_rope'].norm():.4f})")

# Step 4: GQA expansion + SDPA (using repeat_interleave — as engine does)
q_flat = q_rope.reshape(T, num_heads, head_dim)       # [7, 24, 256]
k_flat = k_rope.reshape(T, num_kv_heads, head_dim)    # [7, 4, 256]
v_flat = v.reshape(T, num_kv_heads, head_dim)         # [7, 4, 256]

# Engine method: repeat_interleave
gqa_factor = num_heads // num_kv_heads  # 6
q_4d = q_flat.reshape(1, T, num_heads, head_dim).transpose(1, 2)  # [1, 24, 7, 256]
k_4d = (k_flat.reshape(1, T, num_kv_heads, head_dim)
        .transpose(1, 2)
        .repeat_interleave(gqa_factor, dim=1))  # [1, 24, 7, 256]
v_4d = (v_flat.reshape(1, T, num_kv_heads, head_dim)
        .transpose(1, 2)
        .repeat_interleave(gqa_factor, dim=1))  # [1, 24, 7, 256]

attn_out = F.scaled_dot_product_attention(
    q_4d.float(), k_4d.float(), v_4d.float(),
    is_causal=True).transpose(1, 2).reshape(T, num_heads * head_dim)  # [7, 6144]

print(f"\n--- Step 4: SDPA (repeat_interleave GQA) ---")
print(f"  attn_out norm: {attn_out.norm():.4f}")

# Step 5: Output gate
gate_flat = gate.reshape(B, T, num_heads * head_dim)
attn_gated = attn_out * torch.sigmoid(gate_flat)

print(f"\n--- Step 5: Output Gate ---")
print(f"  attn_gated norm: {attn_gated.norm():.4f} (ref: {ref['attn_out'].norm():.4f})")

# Step 6: o_proj
o_out = F.linear(attn_gated, W_o)  # [1, 7, 5120]

print(f"\n--- Step 6: o_proj ---")
print(f"  o_out norm: {o_out.norm():.4f} (ref: {ref['layer3_out'].norm():.4f})")

# =====================================================================
# Comparison with reference
# =====================================================================
print(f"\n{'='*60}")
print(f"DIFFERENCES vs reference (should all be ~0):")
print(f"{'='*60}")

checks = [
    ('q_proj', q, ref['q_proj']),
    ('k_proj', k, ref['k_proj']),
    ('v_proj', v, ref['v_proj']),
    ('gate_proj', gate, ref['gate_proj']),
    ('q_normed', q_normed, ref['q_normed']),
    ('k_normed', k_normed, ref['k_normed']),
    ('q_rope', q_rope, ref['q_rope']),
    ('k_rope', k_rope, ref['k_rope']),
    ('attn_gated', attn_gated, ref['attn_out']),
    ('o_out', o_out, ref['layer3_out']),
]

for name, ours, rv in checks:
    diff = (ours - rv).abs()
    norm_pct = abs(ours.norm() - rv.norm()) / rv.norm() * 100 if rv.norm() > 0 else 0
    status = "MATCH" if diff.max() < 1e-4 else "⚠️ DIVERGE"
    print(f"  {name:15s}: max_diff={diff.max():.8f}  mean_diff={diff.mean():.8f}  norm_diff={norm_pct:.4f}%  [{status}]")

# Also save for later use
torch.save({
    'hs_input': hs_input,
    'q_proj': q, 'k_proj': k, 'v_proj': v, 'gate_proj': gate,
    'q_normed': q_normed, 'k_normed': k_normed,
    'q_rope': q_rope, 'k_rope': k_rope,
    'attn_out': attn_gated, 'o_out': o_out,
}, '/tmp/diag_engine_fa_replica.pt')
print(f"\nSaved to /tmp/diag_engine_fa_replica.pt")
