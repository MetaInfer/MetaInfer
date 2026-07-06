#!/usr/bin/env python3
"""Pure CPU test: one FullAttention layer, no TP, no GPU. Compare numeric values."""
import os, sys, torch, json, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F

model_dir = os.environ['MODEL_DIR']
from safetensors import safe_open

with open(os.path.join(model_dir, 'config.json')) as f:
    cfg_raw = json.load(f)
tc = cfg_raw.get('text_config', cfg_raw)
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    index = json.load(f)

weight_map = index['weight_map']
def load_raw(key):
    fname = weight_map[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

hidden_size = tc['hidden_size']          # 5120
num_heads = tc['num_attention_heads']    # 24
num_kv_heads = tc['num_key_value_heads']  # 4
head_dim = tc['head_dim']               # 256
eps = tc['rms_norm_eps']
rotary_dim = int(head_dim * tc['rope_parameters']['partial_rotary_factor'])  # 64
rope_theta = tc['rope_parameters']['rope_theta']
mrope_section = tc['rope_parameters']['mrope_section']
mrope_interleaved = tc['rope_parameters']['mrope_interleaved']
scaling = head_dim ** -0.5

# Load FullAttention layer 3 weights
prefix = 'model.language_model.layers.3.self_attn.'
q_proj = load_raw(prefix + 'q_proj.weight').float()
k_proj = load_raw(prefix + 'k_proj.weight').float()
v_proj = load_raw(prefix + 'v_proj.weight').float()
o_proj = load_raw(prefix + 'o_proj.weight').float()
q_norm_w = load_raw(prefix + 'q_norm.weight').float()
k_norm_w = load_raw(prefix + 'k_norm.weight').float()

# Load input_layernorm for layer 3
input_ln_w = load_raw('model.language_model.layers.3.input_layernorm.weight').float()

# Create test input that mimics real model behavior
# Simulate: embedding → 3 GatedDeltaNet layers → this FullAttention
torch.manual_seed(42)
embed = torch.randn(1, 4, hidden_size) * 0.3  # small embedding

# Simulate residual = embed (after embedding)
residual = embed.clone()
# Simulate 3 GatedDeltaNet layers with approximate residual growth
for i in range(3):
    # Each GatedDeltaNet adds ~8 to residual norm
    delta = torch.randn(1, 4, hidden_size) * 0.3
    residual = residual + delta

print(f"Simulated residual norm (after 3 GatedDeltaNet layers): {residual.norm():.4f}")

# Step 1: input_layernorm
# Qwen3_5RMSNorm: x * rsqrt(var) * (1+w)
hs = residual.float()
rstd = 1.0 / torch.sqrt(hs.pow(2).mean(-1, keepdim=True) + eps)
eff_w = 1.0 + input_ln_w
hs = hs * rstd * eff_w
print(f"After input_layernorm: norm={hs.norm():.4f}")

# Step 2: QKV projections
q_full = F.linear(hs, q_proj)           # [1,4, 12288]
k = F.linear(hs, k_proj)                # [1,4, 1024]
v = F.linear(hs, v_proj)                # [1,4, 1024]
q_all, gate = torch.chunk(q_full, 2, dim=-1)

q_all = q_all.view(1, 4, num_heads, head_dim)
k = k.view(1, 4, num_kv_heads, head_dim)
v = v.view(1, 4, num_kv_heads, head_dim)
gate = gate.view(1, 4, num_heads, head_dim)

print(f"After QKV: q={q_all.shape} norm={q_all.norm():.4f}, k norm={k.norm():.4f}, v norm={v.norm():.4f}")

# Step 3: Q/K RMSNorm
q_rstd = 1.0 / torch.sqrt(q_all.pow(2).mean(-1, keepdim=True) + eps)
k_rstd = 1.0 / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + eps)
q = q_all * q_rstd * (1.0 + q_norm_w)
k = k * k_rstd * (1.0 + k_norm_w)

print(f"After Q/K norm: q norm={q.norm():.4f}, k norm={k.norm():.4f}")

# Step 4: MRoPE
from engine.kernels.rotary_embedding import make_cos_sin_cache, rotary_embedding
positions = torch.arange(4, dtype=torch.int64)
# Use small max_position for speed (only need positions 0-3)
cos_sin = make_cos_sin_cache(
    128, rotary_dim, rope_theta, dtype=torch.float32,
    mrope_section=mrope_section, mrope_interleaved=mrope_interleaved)

q_flat = q.reshape(4, num_heads, head_dim).contiguous()
k_flat = k.reshape(4, num_kv_heads, head_dim).contiguous()

q_rot = q_flat[..., :rotary_dim].contiguous()
k_rot = k_flat[..., :rotary_dim].contiguous()
rotary_embedding(positions, q_rot, k_rot, rotary_dim, cos_sin, is_neox=True)
q_flat[..., :rotary_dim] = q_rot
k_flat[..., :rotary_dim] = k_rot

q = q_flat.reshape(1, 4, num_heads, head_dim)
k = k_flat.reshape(1, 4, num_kv_heads, head_dim)

print(f"After MRoPE: q norm={q.norm():.4f}, k norm={k.norm():.4f}")

# Step 5: SDPA
gqa_factor = num_heads // num_kv_heads
q_sdpa = q.transpose(1, 2)       # [1, 24, 4, 256]
k_sdpa = k.transpose(1, 2)       # [1, 4, 4, 256]
v_sdpa = v.transpose(1, 2)       # [1, 4, 4, 256]
k_sdpa = k_sdpa.repeat_interleave(gqa_factor, dim=1)
v_sdpa = v_sdpa.repeat_interleave(gqa_factor, dim=1)

attn_scores = torch.matmul(q_sdpa, k_sdpa.transpose(-2, -1)) * scaling
causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
attn_probs = F.softmax(attn_scores, dim=-1)

print(f"\nAttention diagnostics:")
print(f"  Score range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")
print(f"  Probs max avg: {attn_probs.max(-1).values.mean():.4f}")
print(f"  Probs entropy: {(-attn_probs*(attn_probs+1e-10).log()).sum(-1).mean():.4f}")

attn_out = torch.matmul(attn_probs, v_sdpa)
attn_out = attn_out.transpose(1, 2)  # [1, 4, 24, 256]

print(f"  Attention output norm: {attn_out.norm():.4f}")

# Step 6: Output gate
gated = attn_out.float() * torch.sigmoid(gate.float())
print(f"  Gated output norm: {gated.norm():.4f}")
print(f"  sigmoid(gate) range: [{torch.sigmoid(gate).min():.4f}, {torch.sigmoid(gate).max():.4f}]")

# Step 7: o_proj
gated_flat = gated.reshape(1, 4, num_heads * head_dim)
output = F.linear(gated_flat.float(), o_proj)
print(f"  Final o_proj norm: {output.norm():.4f}")

print(f"\n=== SUMMARY ===")
print(f"Simulated residual before FullAttention: {residual.norm():.4f}")
print(f"FullAttention output norm: {output.norm():.4f}")
print(f"Ratio: {output.norm()/residual.norm():.4f}")

# If this ratio is ~0.1-0.5, the attention is well-behaved
# If it's >2, something is wrong
print(f"\nExpected ratio (healthy model): < 1.0")
print(f"Actual ratio: {output.norm() / residual.norm():.4f}")

# Check: what does the o_proj weight look like?
print(f"\no_proj weight norm: {o_proj.norm():.4f}")
print(f"o_proj weight singular values:")
s = torch.linalg.svdvals(o_proj)
print(f"  Top 5: {s[:5].tolist()}")
print(f"  Bottom 5: {s[-5:].tolist()}")
