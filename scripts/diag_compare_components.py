#!/usr/bin/env python3
"""Compare GatedDeltaNet and FullAttention output magnitudes on the SAME input scale.

Key question: is the FullAttention output reasonably scaled relative to its input?
"""
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
k_heads = tc['linear_num_key_heads']     # 16
v_heads = tc['linear_num_value_heads']   # 48
k_head_dim = tc['linear_key_head_dim']   # 128
v_head_dim = tc['linear_value_head_dim'] # 128
conv_kernel = tc['linear_conv_kernel_dim'] # 4
eps = tc['rms_norm_eps']
num_attn_heads = tc['num_attention_heads']  # 24
num_kv_heads = tc['num_key_value_heads']   # 4
head_dim = tc['head_dim']                  # 256
rotary_dim = int(head_dim * tc['rope_parameters']['partial_rotary_factor'])  # 64
rope_theta = tc['rope_parameters']['rope_theta']
mrope_section = tc['rope_parameters']['mrope_section']
mrope_interleaved = tc['rope_parameters']['mrope_interleaved']

# Create a COMMON input that simulates what an intermediate layer would see
# Use a residual norm of ~50 (matching what we see before layer 3)
torch.manual_seed(42)
B, T = 1, 4
residual = torch.randn(B, T, hidden_size) * (50.0 / math.sqrt(hidden_size * T))
print(f"Common residual: shape={residual.shape}, norm={residual.norm():.4f}")
print(f"  Per-element std: {residual.std():.4f}")

# Load input_layernorm weights for layer 0 (GatedDeltaNet) and layer 3 (FullAttention)
input_ln0 = load_raw('model.language_model.layers.0.input_layernorm.weight').float()
input_ln3 = load_raw('model.language_model.layers.3.input_layernorm.weight').float()

# Apply input_layernorm to residual (Qwen3_5RMSNorm: x * rsqrt * (1+w))
def apply_qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return x.float() * rstd * (1.0 + w)

hs0 = apply_qwen35_rms_norm(residual, input_ln0)
hs3 = apply_qwen35_rms_norm(residual, input_ln3)
print(f"After input_layernorm (layer 0 w): norm={hs0.norm():.4f}")
print(f"  input_ln0 effective (1+w): min={1.0+input_ln0.min():.4f}, max={1.0+input_ln0.max():.4f}, mean={1.0+input_ln0.mean():.4f}")
print(f"After input_layernorm (layer 3 w): norm={hs3.norm():.4f}")
print(f"  input_ln3 effective (1+w): min={1.0+input_ln3.min():.4f}, max={1.0+input_ln3.max():.4f}, mean={1.0+input_ln3.mean():.4f}")

# =============================================
# COMPUTE GatedDeltaNet layer 0
# =============================================
print("\n=== GatedDeltaNet Layer 0 ===")
in_qkv = load_raw('model.language_model.layers.0.linear_attn.in_proj_qkv.weight').float()
in_a = load_raw('model.language_model.layers.0.linear_attn.in_proj_a.weight').float()
in_b = load_raw('model.language_model.layers.0.linear_attn.in_proj_b.weight').float()
in_z = load_raw('model.language_model.layers.0.linear_attn.in_proj_z.weight').float()
A_log = load_raw('model.language_model.layers.0.linear_attn.A_log').float()
dt_bias = load_raw('model.language_model.layers.0.linear_attn.dt_bias').float()
conv_w = load_raw('model.language_model.layers.0.linear_attn.conv1d.weight').float()
norm_w = load_raw('model.language_model.layers.0.linear_attn.norm.weight').float()
out_proj = load_raw('model.language_model.layers.0.linear_attn.out_proj.weight').float()

# GatedDeltaNet forward
mixed_qkv = F.linear(hs0, in_qkv)
mixed_t = mixed_qkv.transpose(1, 2)
mixed_pad = F.pad(mixed_t, (conv_kernel - 1, 0))
conv_out = F.conv1d(mixed_pad, conv_w, groups=10240)
conv_out = F.silu(conv_out)
conv_out = conv_out.transpose(1, 2)  # [B, T, 10240]

q = conv_out[:, :, :2048].view(B, T, k_heads, k_head_dim)
k = conv_out[:, :, 2048:4096].view(B, T, k_heads, k_head_dim)
v = conv_out[:, :, 4096:].view(B, T, v_heads, v_head_dim)

a = F.linear(hs0, in_a)
b = F.linear(hs0, in_b)

g = -torch.exp(A_log) * F.softplus(a + dt_bias)
beta = torch.sigmoid(b)

q_norm = F.normalize(q.float(), p=2, dim=-1) * (1.0 / math.sqrt(k_head_dim))
k_norm = F.normalize(k.float(), p=2, dim=-1)

rpt = v_heads // k_heads
q_norm = q_norm.repeat_interleave(rpt, dim=2)
k_norm = k_norm.repeat_interleave(rpt, dim=2)

state = torch.zeros(B, v_heads, k_head_dim, v_head_dim, dtype=torch.float32)
for t in range(T):
    g_t = g[:, t, :]
    k_t = k_norm[:, t, :, :]
    v_t = v[:, t, :, :]
    q_t = q_norm[:, t, :, :]
    beta_t = beta[:, t, :]
    state = state * torch.exp(g_t.float())[:, :, None, None]
    kv_mem = torch.sum(state * k_t[:, :, :, None], dim=-2)
    delta = (v_t - kv_mem) * beta_t[:, :, None]
    state = state + k_t[:, :, :, None] * delta[:, :, None, :]
    o_t = torch.sum(state * q_t[:, :, :, None], dim=-2)
    if t == T - 1:
        core_last = o_t  # [B, Vh, Dv]

z = F.linear(hs0, in_z)
z = z.view(B, T, v_heads, v_head_dim)
z_last = z[:, -1, :, :]

core_flat = core_last.reshape(-1, v_head_dim)
z_flat = z_last.reshape(-1, v_head_dim)
rstd_core = 1.0 / torch.sqrt(core_flat.pow(2).mean(-1, keepdim=True) + eps)
x_norm = core_flat * rstd_core
gdn_out_last = (x_norm * norm_w * F.silu(z_flat)).view(B, 1, v_heads * v_head_dim)
gdn_output = F.linear(gdn_out_last.float(), out_proj)  # [B, 1, hidden]

print(f"GatedDeltaNet output (last position): norm={gdn_output.norm():.4f}")

# =============================================
# COMPUTE FullAttention layer 3
# =============================================
print("\n=== FullAttention Layer 3 ===")
prefix = 'model.language_model.layers.3.self_attn.'
q_proj = load_raw(prefix + 'q_proj.weight').float()
k_proj = load_raw(prefix + 'k_proj.weight').float()
v_proj = load_raw(prefix + 'v_proj.weight').float()
o_proj = load_raw(prefix + 'o_proj.weight').float()
q_norm_w = load_raw(prefix + 'q_norm.weight').float()
k_norm_w = load_raw(prefix + 'k_norm.weight').float()

# FullAttention forward
q_full = F.linear(hs3, q_proj)
k = F.linear(hs3, k_proj)
v = F.linear(hs3, v_proj)

q_all, gate = torch.chunk(q_full, 2, dim=-1)
q_all = q_all.view(B, T, num_attn_heads, head_dim)
k = k.view(B, T, num_kv_heads, head_dim)
v = v.view(B, T, num_kv_heads, head_dim)
gate = gate.view(B, T, num_attn_heads, head_dim)

q_rstd = 1.0 / torch.sqrt(q_all.float().pow(2).mean(-1, keepdim=True) + eps)
k_rstd = 1.0 / torch.sqrt(k.float().pow(2).mean(-1, keepdim=True) + eps)
q = q_all.float() * q_rstd * (1.0 + q_norm_w)
k = k.float() * k_rstd * (1.0 + k_norm_w)

# MRoPE
from engine.kernels.rotary_embedding import make_cos_sin_cache, rotary_embedding
positions = torch.arange(T, dtype=torch.int64)
cos_sin = make_cos_sin_cache(128, rotary_dim, rope_theta, dtype=torch.float32,
                              mrope_section=mrope_section, mrope_interleaved=mrope_interleaved)

q_flat = q.reshape(T, num_attn_heads, head_dim).contiguous()
k_flat = k.reshape(T, num_kv_heads, head_dim).contiguous()
q_rot = q_flat[..., :rotary_dim].contiguous()
k_rot = k_flat[..., :rotary_dim].contiguous()
rotary_embedding(positions, q_rot, k_rot, rotary_dim, cos_sin, is_neox=True)
q_flat[..., :rotary_dim] = q_rot
k_flat[..., :rotary_dim] = k_rot
q = q_flat.reshape(B, T, num_attn_heads, head_dim)
k = k_flat.reshape(B, T, num_kv_heads, head_dim)

# SDPA
scaling = head_dim ** -0.5
gqa = num_attn_heads // num_kv_heads
q_sdpa = q.transpose(1, 2)
k_sdpa = k.transpose(1, 2).repeat_interleave(gqa, dim=1)
v_sdpa = v.transpose(1, 2).repeat_interleave(gqa, dim=1)

attn_scores = torch.matmul(q_sdpa, k_sdpa.transpose(-2, -1)) * scaling
attn_scores = attn_scores.masked_fill(
    torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), float('-inf'))
attn_probs = F.softmax(attn_scores, dim=-1)
attn_out = torch.matmul(attn_probs, v_sdpa).transpose(1, 2)  # [B, T, H, D]

attn_last = attn_out[:, -1:, :, :]  # [B, 1, H, D]
gate_last = gate[:, -1:, :, :]

gated_last = attn_last.float() * torch.sigmoid(gate_last.float())
gated_flat = gated_last.reshape(B, 1, num_attn_heads * head_dim)
fa_output = F.linear(gated_flat.float(), o_proj)  # [B, 1, hidden]

print(f"FullAttention output (last position): norm={fa_output.norm():.4f}")

# =============================================
# COMPARISON
# =============================================
print("\n=== COMPARISON (last position output) ===")
print(f"Both layers see the SAME input residual (norm ~{residual.norm():.0f})")
print(f"GatedDeltaNet (layer 0) output: {gdn_output.norm():.4f}")
print(f"FullAttention (layer 3) output:  {fa_output.norm():.4f}")
print(f"Ratio FullAttention/GatedDeltaNet: {fa_output.norm()/gdn_output.norm():.1f}x")

# Check: what if GatedDeltaNet should produce output in the SAME ballpark as FullAttention?
# Maybe the GatedDeltaNet output is TOO SMALL.
print(f"\nHypothesis: GatedDeltaNet output should be comparable to FullAttention output")
print(f"If the model expects both layer types to contribute similarly, but GatedDeltaNet")
print(f"produces 20x smaller output, the residual would be dominated by FullAttention layers,")
print(f"and the GatedDeltaNet signal would be lost.")

# Let's check if the `core_out` for GatedDeltaNet is reasonable
print(f"\nGatedDeltaNet core_out (recurrent output) norm: {core_last.norm():.4f}")
print(f"This gets amplified by GatedRMSNorm: x_norm norm={x_norm.norm():.4f}")
print(f"Times norm_w ({norm_w.mean():.4f}): scaled norm={x_norm.norm()*norm_w.mean():.4f}")
print(f"Times silu(z) (avg ~{F.silu(z_last).mean():.4f}): final norm={x_norm.norm()*norm_w.mean()*F.silu(z_last).mean():.4f}")
print(f"Then out_proj amplifies by factor ~{gdn_out_last.norm()/x_norm.norm()/norm_w.mean()/F.silu(z_last).mean():.1f}")
