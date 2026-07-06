#!/usr/bin/env python3
"""Pure CPU test: one GatedDeltaNet layer. Compare to reference implementation."""
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

hidden_size = tc['hidden_size']
k_heads = tc['linear_num_key_heads']      # 16
v_heads = tc['linear_num_value_heads']     # 48
k_head_dim = tc['linear_key_head_dim']     # 128
v_head_dim = tc['linear_value_head_dim']   # 128
conv_kernel = tc['linear_conv_kernel_dim'] # 4
eps = tc['rms_norm_eps']

print(f"GDN: k_heads={k_heads}, v_heads={v_heads}, k_dim={k_head_dim}, v_dim={v_head_dim}, conv_k={conv_kernel}")

# Load GatedDeltaNet layer 0 weights
prefix = 'model.language_model.layers.0.linear_attn.'
in_qkv = load_raw(prefix + 'in_proj_qkv.weight').float()  # [10240, 5120] = [2048+2048+6144, 5120]
in_a = load_raw(prefix + 'in_proj_a.weight').float()      # [48, 5120]
in_b = load_raw(prefix + 'in_proj_b.weight').float()      # [48, 5120]
in_z = load_raw(prefix + 'in_proj_z.weight').float()      # [6144, 5120]
A_log = load_raw(prefix + 'A_log').float()                # [48]
dt_bias = load_raw(prefix + 'dt_bias').float()            # [48]
conv_w = load_raw(prefix + 'conv1d.weight').float()       # [10240, 1, 4]
norm_w = load_raw(prefix + 'norm.weight').float()         # [128]
out_proj = load_raw(prefix + 'out_proj.weight').float()   # [5120, 6144]

input_ln_w = load_raw('model.language_model.layers.0.input_layernorm.weight').float()

# Test input — simulate embedding output
torch.manual_seed(42)
B, T = 1, 4
x = torch.randn(B, T, hidden_size) * 0.5
print(f"Input norm: {x.norm():.4f}")

# ============ REFERENCE IMPLEMENTATION ============

# Step 0: Input layernorm (Pre-LN)
hs = x.float()
eff_w = 1.0 + input_ln_w
rstd = 1.0 / torch.sqrt(hs.pow(2).mean(-1, keepdim=True) + eps)
hs = hs * rstd * eff_w
print(f"After input_layernorm: norm={hs.norm():.4f}")

# Step 1: in_proj_qkv
q_dim = k_heads * k_head_dim  # 2048
k_dim = k_heads * k_head_dim  # 2048
v_dim = v_heads * v_head_dim  # 6144
mixed_qkv = F.linear(hs, in_qkv)  # [1, 4, 10240]
print(f"in_proj_qkv output: {mixed_qkv.shape}, norm={mixed_qkv.norm():.4f}")

# Step 2: Causal conv1d
mixed_t = mixed_qkv.transpose(1, 2)  # [1, 10240, 4]
mixed_pad = F.pad(mixed_t, (conv_kernel - 1, 0))  # [1, 10240, 7]
conv_out = F.conv1d(mixed_pad, conv_w, groups=10240)  # [1, 10240, 4]
conv_out = F.silu(conv_out)
conv_out = conv_out.transpose(1, 2)  # [1, 4, 10240]
print(f"Conv1d output: {conv_out.shape}, norm={conv_out.norm():.4f}")

# Step 3: Split Q, K, V
q = conv_out[:, :, :2048].view(B, T, k_heads, k_head_dim)   # [1, 4, 16, 128]
k = conv_out[:, :, 2048:4096].view(B, T, k_heads, k_head_dim)  # [1, 4, 16, 128]
v = conv_out[:, :, 4096:].view(B, T, v_heads, v_head_dim)   # [1, 4, 48, 128]

print(f"After split: Q={q.shape} norm={q.norm():.4f}, K norm={k.norm():.4f}, V norm={v.norm():.4f}")

# Step 4: in_proj_a, in_proj_b
a = F.linear(hs, in_a)  # [1, 4, 48]
b = F.linear(hs, in_b)  # [1, 4, 48]

# Step 5: Gate and beta
g = -torch.exp(A_log) * F.softplus(a + dt_bias)  # [1, 4, 48]
beta = torch.sigmoid(b)  # [1, 4, 48]

print(f"g: min={g.min():.4f}, max={g.max():.4f}, mean={g.mean():.4f}")
print(f"beta: min={beta.min():.4f}, max={beta.max():.4f}, mean={beta.mean():.4f}")
print(f"exp(g): min={torch.exp(g).min():.6f}, max={torch.exp(g).max():.6f}, mean={torch.exp(g).mean():.6f}")

# Step 6: Q/K L2 normalize
q_norm = F.normalize(q.float(), p=2, dim=-1)
k_norm = F.normalize(k.float(), p=2, dim=-1)

# Repeat k to match v heads
repeat_factor = v_heads // k_heads  # 3
k_norm = k_norm.repeat_interleave(repeat_factor, dim=2)  # [1, 4, 48, 128]
q_norm = q_norm.repeat_interleave(repeat_factor, dim=2)  # [1, 4, 48, 128]

# Query scaling
q_scale = 1.0 / math.sqrt(k_head_dim)  # 1/sqrt(128) ≈ 0.0884
q_norm = q_norm * q_scale

print(f"After L2 norm + repeat: Q norm={q_norm.norm():.4f}, K norm={k_norm.norm():.4f}")
print(f"After q_scale: Q norm={q_norm.norm():.4f}")

# Step 7: Recurrent Gated Delta Rule
state = torch.zeros(B, v_heads, k_head_dim, v_head_dim, dtype=torch.float32)
core_out = torch.zeros(B, T, v_heads, v_head_dim, dtype=torch.float32)

for t in range(T):
    g_t = g[:, t, :]       # [1, 48]
    k_t = k_norm[:, t, :, :]  # [1, 48, 128]
    v_t = v[:, t, :, :]    # [1, 48, 128]
    q_t = q_norm[:, t, :, :]  # [1, 48, 128]
    beta_t = beta[:, t, :]  # [1, 48]

    # State decay
    state = state * torch.exp(g_t.float())[:, :, None, None]

    # Key-value memory
    kv_mem = torch.sum(state * k_t[:, :, :, None], dim=-2)  # [1, 48, 128]

    # Delta update
    delta = (v_t - kv_mem) * beta_t[:, :, None]

    # State update (outer product)
    state = state + k_t[:, :, :, None] * delta[:, :, None, :]

    # Output (POST-update state)
    o_t = torch.sum(state * q_t[:, :, :, None], dim=-2)  # [1, 48, 128]
    core_out[:, t, :, :] = o_t

print(f"Recurrent output: {core_out.shape}, norm={core_out.norm():.4f}")
print(f"  State norm (final): {state.norm():.4f}")

# Step 8: in_proj_z (output gate)
z = F.linear(hs, in_z)  # [1, 4, 6144]
z = z.view(B, T, v_heads, v_head_dim)  # [1, 4, 48, 128]

print(f"in_proj_z: {z.shape}, norm={z.norm():.4f}")

# Step 9: Gated RMSNorm
core_flat = core_out.reshape(-1, v_head_dim)  # [4*48, 128]
z_flat = z.reshape(-1, v_head_dim)              # [4*48, 128]

# Qwen3_5RMSNormGated: x * rsqrt(var) * w * silu(gate)
rstd_core = 1.0 / torch.sqrt(core_flat.pow(2).mean(-1, keepdim=True) + eps)
x_norm = core_flat * rstd_core
gated_out = x_norm * norm_w * F.silu(z_flat)
gated_out = gated_out.view(B, T, v_heads * v_head_dim)  # [1, 4, 6144]

print(f"Gated norm output: {gated_out.shape}, norm={gated_out.norm():.4f}")
print(f"  norm_w: min={norm_w.min():.4f}, max={norm_w.max():.4f}, mean={norm_w.mean():.4f}")
print(f"  silu(z) range: [{F.silu(z).min():.4f}, {F.silu(z).max():.4f}]")

# Step 10: out_proj
output = F.linear(gated_out.float(), out_proj)  # [1, 4, 5120]
print(f"GatedDeltaNet output norm: {output.norm():.4f}")

print(f"\n=== SUMMARY ===")
print(f"Input norm: {x.norm():.4f}")
print(f"GatedDeltaNet output norm: {output.norm():.4f}")
print(f"Ratio output/input: {output.norm()/x.norm():.4f}")

# Check what norm_w (Gated RMSNorm weight) looks like
print(f"\nnorm_w (expected ones): {norm_w[:5].tolist()}... mean={norm_w.mean():.4f}")
print(f"norm_w all ones? {torch.allclose(norm_w, torch.ones_like(norm_w))}")
print(f"If not ones, the formula 'x*rsqrt*w*silu(z)' uses these actual values")

# Check out_proj singular values
s = torch.linalg.svdvals(out_proj)
print(f"\nout_proj sv top 5: {s[:5].tolist()}")
print(f"out_proj sv bottom 5: {s[-5:].tolist()}")
