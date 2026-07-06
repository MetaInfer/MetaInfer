#!/usr/bin/env python3
"""Diagnostic: Direct weight comparison — model loaded weights vs raw safetensors.

Verify that the weight loading dispatch is correct for key layers.
Runs on CPU — just loads and compares tensors.
"""
import os, sys, torch, json, math
sys.path.insert(0, os.environ.get('AGENT_INFER_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safetensors import safe_open

model_dir = os.environ['MODEL_DIR']
index_path = os.path.join(model_dir, 'model.safetensors.index.json')
with open(index_path) as f:
    index = json.load(f)
weight_map = index['weight_map']

def load_raw(key):
    """Load a single weight tensor from safetensors."""
    fname = weight_map.get(key)
    if fname is None:
        print(f"KEY NOT FOUND: {key}")
        return None
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

# ========== Check FullAttention layer 3 weights ==========
print("=" * 60)
print("FullAttention layer 3 weight check")
print("=" * 60)

prefix = 'model.language_model.layers.3.self_attn.'

# Load raw weights
q_proj_raw = load_raw(prefix + 'q_proj.weight')     # [12288, 5120]
k_proj_raw = load_raw(prefix + 'k_proj.weight')     # [1024, 5120]
v_proj_raw = load_raw(prefix + 'v_proj.weight')     # [1024, 5120]
o_proj_raw = load_raw(prefix + 'o_proj.weight')     # [5120, 6144]
q_norm_raw = load_raw(prefix + 'q_norm.weight')     # [256]
k_norm_raw = load_raw(prefix + 'k_norm.weight')     # [256]

print(f"q_proj raw: {q_proj_raw.shape}, norm={q_proj_raw.norm().item():.4f}")
print(f"k_proj raw: {k_proj_raw.shape}, norm={k_proj_raw.norm().item():.4f}")
print(f"v_proj raw: {v_proj_raw.shape}, norm={v_proj_raw.norm().item():.4f}")
print(f"o_proj raw: {o_proj_raw.shape}, norm={o_proj_raw.norm().item():.4f}")
print(f"q_norm raw: {q_norm_raw.shape}, min={q_norm_raw.min().item():.6f}, max={q_norm_raw.max().item():.6f}")
print(f"k_norm raw: {k_norm_raw.shape}, min={k_norm_raw.min().item():.6f}, max={k_norm_raw.max().item():.6f}")

# Check q_norm and k_norm values
print(f"\nq_norm first 10 values: {q_norm_raw[:10].tolist()}")
print(f"q_norm last 10 values: {q_norm_raw[-10:].tolist()}")
print(f"q_norm abs max: {q_norm_raw.abs().max().item():.6f}")
print(f"q_norm mean: {q_norm_raw.mean().item():.6f}")
print(f"Are all zero? {torch.allclose(q_norm_raw, torch.zeros_like(q_norm_raw))}")

# Actually look at the distribution
print(f"\nq_norm percentiles: p0={q_norm_raw.min():.6f}, p25={q_norm_raw.quantile(0.25):.6f}, p50={q_norm_raw.median():.6f}, p75={q_norm_raw.quantile(0.75):.6f}, p100={q_norm_raw.max():.6f}")

# Check if q_norm is all zeros or not
zero_count = (q_norm_raw.abs() < 1e-8).sum().item()
print(f"q_norm near-zero entries: {zero_count}/{q_norm_raw.numel()}")

# ========== Check GatedDeltaNet layer 0 weights ==========
print("\n" + "=" * 60)
print("GatedDeltaNet layer 0 weight check")
print("=" * 60)

ld0 = 'model.language_model.layers.0.linear_attn.'

in_qkv_raw = load_raw(ld0 + 'in_proj_qkv.weight')   # [10240, 5120]
in_a_raw = load_raw(ld0 + 'in_proj_a.weight')        # [48, 5120]
in_b_raw = load_raw(ld0 + 'in_proj_b.weight')        # [48, 5120]
in_z_raw = load_raw(ld0 + 'in_proj_z.weight')        # [6144, 5120]
A_log_raw = load_raw(ld0 + 'A_log')                  # [48]
dt_bias_raw = load_raw(ld0 + 'dt_bias')              # [48]
conv1d_raw = load_raw(ld0 + 'conv1d.weight')         # [10240, 1, 4]
norm_raw = load_raw(ld0 + 'norm.weight')             # [128]
out_proj_raw = load_raw(ld0 + 'out_proj.weight')     # [5120, 6144]

print(f"in_proj_qkv: {in_qkv_raw.shape}, norm={in_qkv_raw.norm().item():.4f}")
print(f"in_proj_a: {in_a_raw.shape}, norm={in_a_raw.norm().item():.4f}")
print(f"in_proj_b: {in_b_raw.shape}, norm={in_b_raw.norm().item():.4f}")
print(f"in_proj_z: {in_z_raw.shape}, norm={in_z_raw.norm().item():.4f}")
print(f"A_log: {A_log_raw.shape}, values={A_log_raw[:10].tolist()}...")
print(f"dt_bias: {dt_bias_raw.shape}, values={dt_bias_raw[:10].tolist()}...")
print(f"conv1d: {conv1d_raw.shape}, norm={conv1d_raw.norm().item():.4f}")
print(f"norm.weight: {norm_raw.shape}, values={norm_raw[:10].tolist()}...")
print(f"out_proj: {out_proj_raw.shape}, norm={out_proj_raw.norm().item():.4f}")

# Check in_proj_qkv: verify Q (2048), K (2048), V (6144) split dimensions
# Total = 2048 + 2048 + 6144 = 10240
q_raw = in_qkv_raw[:2048, :]
k_raw = in_qkv_raw[2048:4096, :]
v_raw = in_qkv_raw[4096:10240, :]
print(f"\nQ section: {q_raw.shape}, norm={q_raw.norm().item():.4f}")
print(f"K section: {k_raw.shape}, norm={k_raw.norm().item():.4f}")
print(f"V section: {v_raw.shape}, norm={v_raw.norm().item():.4f}")

# Check conv1d weight: same Q/K/V split
conv_q_raw = conv1d_raw[:2048, :, :]
conv_k_raw = conv1d_raw[2048:4096, :, :]
conv_v_raw = conv1d_raw[4096:10240, :, :]
print(f"\nConv1d Q section: {conv_q_raw.shape}, norm={conv_q_raw.norm().item():.4f}")
print(f"Conv1d K section: {conv_k_raw.shape}, norm={conv_k_raw.norm().item():.4f}")
print(f"Conv1d V section: {conv_v_raw.shape}, norm={conv_v_raw.norm().item():.4f}")

# Verify A_log distribution
print(f"\nA_log min={A_log_raw.min():.6f}, max={A_log_raw.max():.6f}, mean={A_log_raw.mean():.6f}")
print(f"exp(A_log) min={torch.exp(A_log_raw).min():.6f}, max={torch.exp(A_log_raw).max():.6f}, mean={torch.exp(A_log_raw).mean():.6f}")

# Check if in_proj_a output dim matches num_v_heads (48)
print(f"\nin_proj_a output dim: {in_a_raw.shape[0]} (expected: 48)")
print(f"in_proj_b output dim: {in_b_raw.shape[0]} (expected: 48)")
print(f"in_proj_z output dim: {in_z_raw.shape[0]} (expected: v_heads_local * v_head_dim * tp_size = ?)")
# in_proj_z: ColumnParallel sharded, full = num_v_heads * v_head_dim = 48 * 128 = 6144
print(f"num_v_heads(48) * v_head_dim(128) = {48*128}, raw shape[0] = {in_z_raw.shape[0]}")
print(f"MATCH: {in_z_raw.shape[0] == 48 * 128}")

# ========== Check MLP weights (layer 0) ==========
print("\n" + "=" * 60)
print("MLP layer 0 weight check")
print("=" * 60)

mlp0 = 'model.language_model.layers.0.mlp.'
gate_raw = load_raw(mlp0 + 'gate_proj.weight')   # [17408, 5120]
up_raw = load_raw(mlp0 + 'up_proj.weight')        # [17408, 5120]
down_raw = load_raw(mlp0 + 'down_proj.weight')    # [5120, 17408]

print(f"gate_proj: {gate_raw.shape}, norm={gate_raw.norm().item():.4f}")
print(f"up_proj: {up_raw.shape}, norm={up_raw.norm().item():.4f}")
print(f"down_proj: {down_raw.shape}, norm={down_raw.norm().item():.4f}")

# Check TP=4 sharding: intermediate=17408, per_rank=4352
print(f"\nintermediate_size=17408, per_rank(TP=4)=4352")
print(f"Gate per rank rows: {17408 // 4}")

# ========== Check layer norms ==========
print("\n" + "=" * 60)
print("Layer norm weight check")
print("=" * 60)

ln0_input = load_raw('model.language_model.layers.0.input_layernorm.weight')  # [5120]
ln0_post = load_raw('model.language_model.layers.0.post_attention_layernorm.weight')  # [5120]
final_norm = load_raw('model.language_model.norm.weight')  # [5120]

print(f"input_layernorm (layer 0): shape={ln0_input.shape}, min={ln0_input.min():.6f}, max={ln0_input.max():.6f}")
print(f"  Are all zeros? {torch.allclose(ln0_input, torch.zeros_like(ln0_input))}")
print(f"post_attn_norm (layer 0): shape={ln0_post.shape}, min={ln0_post.min():.6f}, max={ln0_post.max():.6f}")
print(f"  Are all zeros? {torch.allclose(ln0_post, torch.zeros_like(ln0_post))}")
print(f"final_norm: shape={final_norm.shape}, min={final_norm.min():.6f}, max={final_norm.max():.6f}")
print(f"  Are all zeros? {torch.allclose(final_norm, torch.zeros_like(final_norm))}")

# If not all zeros, print some values
if not torch.allclose(ln0_input, torch.zeros_like(ln0_input)):
    print(f"  input_layernorm sample: {ln0_input[:10].tolist()}")
if not torch.allclose(final_norm, torch.zeros_like(final_norm)):
    print(f"  final_norm sample: {final_norm[:10].tolist()}")

# ========== Summary ==========
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If all layer norms are zeros → Qwen3_5RMSNorm (1+w) formula is correct")
print("If q_norm/k_norm are NOT zeros → Qwen3_5RMSNorm might be wrong formula")
print(f"q_norm is zero: {torch.allclose(q_norm_raw, torch.zeros_like(q_norm_raw))}")
print(f"k_norm is zero: {torch.allclose(k_norm_raw, torch.zeros_like(k_norm_raw))}")
print(f"input_layernorm is zero: {torch.allclose(ln0_input, torch.zeros_like(ln0_input))}")
print(f"final_norm is zero: {torch.allclose(final_norm, torch.zeros_like(final_norm))}")
print(f"gateddelta_norm is ones: {torch.allclose(norm_raw, torch.ones_like(norm_raw))}")
