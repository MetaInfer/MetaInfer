#!/usr/bin/env python3
"""CPU verification: load layer 0 weights from safetensors and compute a reference output.

Compare with the output our model produces for the same layer, same input.
"""
import os, sys, torch, json, math
import torch.nn.functional as F

model_dir = os.environ['MODEL_DIR']

from safetensors import safe_open

# Load index
with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
weight_map = idx['weight_map']

def load_cpu(key):
    fname = weight_map[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

# Load config
with open(os.path.join(model_dir, 'config.json')) as f:
    raw = json.load(f)
tc = raw.get('text_config', raw)

hidden_size = tc['hidden_size']           # 5120
k_heads = tc['linear_num_key_heads']      # 16
v_heads = tc['linear_num_value_heads']    # 48
k_head_dim = tc['linear_key_head_dim']    # 128
v_head_dim = tc['linear_value_head_dim']  # 128
conv_kernel = tc['linear_conv_kernel_dim']# 4
eps = tc['rms_norm_eps']                  # 1e-6

# Load layer 0 weights
prefix = 'model.language_model.layers.0.'
w_in_qkv = load_cpu(prefix + 'linear_attn.in_proj_qkv.weight').float()
w_conv1d = load_cpu(prefix + 'linear_attn.conv1d.weight').float()  # [10240, 1, 4]
w_conv1d_bias = torch.zeros(w_conv1d.shape[0])  # No bias in HF
w_in_a = load_cpu(prefix + 'linear_attn.in_proj_a.weight').float()
w_in_b = load_cpu(prefix + 'linear_attn.in_proj_b.weight').float()
A_log = load_cpu(prefix + 'linear_attn.A_log').float()
dt_bias = load_cpu(prefix + 'linear_attn.dt_bias').float()
w_in_z = load_cpu(prefix + 'linear_attn.in_proj_z.weight').float()
w_norm = load_cpu(prefix + 'linear_attn.norm.weight').float()
w_out = load_cpu(prefix + 'linear_attn.out_proj.weight').float()
w_input_ln = load_cpu(prefix + 'input_layernorm.weight').float()

# Load embedding
embed_w = load_cpu('model.language_model.embed_tokens.weight').float()

# Input: 苏州园林的特点是
tokens = [108618, 102066, 137351, 105017, 100462, 106808, 103105]
input_ids = torch.tensor([tokens], dtype=torch.long)  # [1, 7]

# Get embedding
emb = F.embedding(input_ids, embed_w)  # [1, 7, 5120]
print(f"Embedding output norm: {emb.norm():.4f}, std: {emb.std():.4f}")

# Apply input_layernorm
def qwen35_rms_norm(x, w):
    rstd = 1.0 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * (1.0 + w)

hs = qwen35_rms_norm(emb, w_input_ln)
print(f"After input_layernorm: norm={hs.norm():.4f}, std={hs.std():.4f}")

B, T, H = hs.shape  # 1, 7, 5120

# ---- GatedDeltaNet forward (TP=1, all heads) ----
# 1. in_proj_qkv
mixed_qkv = F.linear(hs, w_in_qkv)  # [1, 7, 10240]
print(f"in_proj_qkv output: norm={mixed_qkv.norm():.4f}")

# 2. Causal conv1d
mixed_qkv_t = mixed_qkv.transpose(1, 2)  # [1, 10240, 7]
mixed_qkv_pad = F.pad(mixed_qkv_t, (3, 0))  # left pad 3
conv_out = F.conv1d(mixed_qkv_pad, w_conv1d, w_conv1d_bias, groups=10240)
conv_out = F.silu(conv_out).transpose(1, 2)  # [1, 7, 10240]
print(f"Conv1d output: norm={conv_out.norm():.4f}")

# 3. Split Q, K, V
q = conv_out[:, :, :2048].view(B, T, k_heads, k_head_dim)
k = conv_out[:, :, 2048:4096].view(B, T, k_heads, k_head_dim)
v = conv_out[:, :, 4096:].view(B, T, v_heads, v_head_dim)

# 4. in_proj_a, in_proj_b
a = F.linear(hs, w_in_a)  # [1, 7, 48]
b = F.linear(hs, w_in_b)  # [1, 7, 48]

# 5. Gate and beta
g = -torch.exp(A_log) * F.softplus(a + dt_bias)  # [1, 7, 48]
beta = torch.sigmoid(b)  # [1, 7, 48]

# 6. Q/K L2 norm
q_norm = F.normalize(q.float(), p=2, dim=-1)
k_norm = F.normalize(k.float(), p=2, dim=-1)

# Repeat k/q to match v heads (48/16=3)
q_norm = q_norm.repeat_interleave(3, dim=2)
k_norm = k_norm.repeat_interleave(3, dim=2)

# Query scaling
q_scale = 1.0 / math.sqrt(k_head_dim)
q_norm = q_norm * q_scale

# 7. Recurrence (fp32)
state = torch.zeros(B, v_heads, k_head_dim, v_head_dim, dtype=torch.float32)
core_out = torch.zeros(B, T, v_heads, v_head_dim, dtype=torch.float32)

for t in range(T):
    g_t = g[:, t, :]
    k_t = k_norm[:, t, :, :]
    v_t = v[:, t, :, :]
    q_t = q_norm[:, t, :, :]
    beta_t = beta[:, t, :]

    state = state * torch.exp(g_t)[:, :, None, None]
    kv_mem = torch.sum(state * k_t[:, :, :, None], dim=-2)
    delta = (v_t - kv_mem) * beta_t[:, :, None]
    state = state + k_t[:, :, :, None] * delta[:, :, None, :]
    o_t = torch.sum(state * q_t[:, :, :, None], dim=-2)
    core_out[:, t, :, :] = o_t

# 8. in_proj_z + gated norm
z = F.linear(hs, w_in_z).view(B, T, v_heads, v_head_dim)
core_flat = core_out.reshape(-1, v_head_dim)
z_flat = z.reshape(-1, v_head_dim)

# Qwen3_5RMSNormGated
rstd = 1.0 / torch.sqrt(core_flat.pow(2).mean(-1, keepdim=True) + eps)
x_norm = core_flat * rstd
gated_out = (x_norm * w_norm * F.silu(z_flat)).view(B, T, v_heads * v_head_dim)

# 9. out_proj
out = F.linear(gated_out.float(), w_out)  # [1, 7, 5120]

print(f"\nLayer 0 CPU output (TP=1, all heads):")
print(f"  norm={out.norm():.4f}")
print(f"  min={out.min():.4f}, max={out.max():.4f}")
print(f"  mean={out.mean():.4f}, std={out.std():.4f}")
print(f"\nLast position (position 6):")
print(f"  norm={out[0, -1, :].norm():.4f}")
print(f"  values: min={out[0,-1].min():.4f}, max={out[0,-1].max():.4f}")

# Save the output for comparison with model
torch.save({'hs': hs, 'out': out, 'tokens': tokens}, '/tmp/diag_layer0_cpu_ref.pt')
print("\nSaved reference to /tmp/diag_layer0_cpu_ref.pt")
print("Now run the TP model and compare.")
