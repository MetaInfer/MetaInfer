#!/usr/bin/env python3
"""Check if q_norm/k_norm are regular RMSNorm or Qwen3_5RMSNorm in HF Qwen3.5."""
import os, sys, json, torch

model_dir = os.environ['MODEL_DIR']

# Read config
with open(os.path.join(model_dir, 'config.json')) as f:
    cfg = json.load(f)

tc = cfg.get('text_config', cfg)
print(f"model_type: {cfg.get('model_type')}")
print(f"text_config model_type: {tc.get('model_type')}")

# Check if there's any hint about RMSNorm type
print(f"\nFull config text_config keys:")
for k, v in tc.items():
    if 'norm' in k.lower() or 'rms' in k.lower():
        print(f"  {k}: {v}")

# Check q_norm/k_norm weight shapes
from safetensors import safe_open
index_path = os.path.join(model_dir, 'model.safetensors.index.json')
with open(index_path) as f:
    index = json.load(f)

weight_map = index['weight_map']

def load_raw(key):
    fname = weight_map[key]
    fpath = os.path.join(model_dir, fname)
    with safe_open(fpath, framework='pt', device='cpu') as sf:
        return sf.get_tensor(key)

# Check q_norm for full_attention layers (indices 3, 7, 11, ...)
print("\n=== Q/K Norm weights for FullAttention layers ===")
for layer_idx in [3, 7, 11, 15, 19, 23, 27, 31]:
    prefix = f'model.language_model.layers.{layer_idx}.self_attn.'
    qn = load_raw(prefix + 'q_norm.weight')
    kn = load_raw(prefix + 'k_norm.weight')
    print(f"Layer {layer_idx}: q_norm [min={qn.min():.4f}, max={qn.max():.4f}, mean={qn.mean():.4f}, std={qn.std():.4f}]")
    print(f"           k_norm [min={kn.min():.4f}, max={kn.max():.4f}, mean={kn.mean():.4f}, std={kn.std():.4f}]")

# Check input_layernorm, post_attention_layernorm, and final norm
print("\n=== Decoder norms ===")
for name in ['model.language_model.layers.0.input_layernorm.weight',
             'model.language_model.layers.0.post_attention_layernorm.weight',
             'model.language_model.layers.3.input_layernorm.weight',
             'model.language_model.layers.3.post_attention_layernorm.weight',
             'model.language_model.norm.weight']:
    w = load_raw(name)
    is_zero = torch.allclose(w, torch.zeros_like(w))
    print(f"{name}: min={w.min():.6f}, max={w.max():.6f}, mean={w.mean():.6f}, is_zero={is_zero}")

# Check GatedDeltaNet norm.weight
print("\n=== GatedDeltaNet norm.weight ===")
for layer_idx in [0, 1, 2, 4]:
    key = f'model.language_model.layers.{layer_idx}.linear_attn.norm.weight'
    w = load_raw(key)
    is_one = torch.allclose(w, torch.ones_like(w))
    print(f"Layer {layer_idx} norm.weight: shape={w.shape}, min={w.min():.4f}, max={w.max():.4f}, is_one={is_one}")

# THE KEY QUESTION: Are q_norm/k_norm using Qwen3_5RMSNorm (1+w) or standard RMSNorm?
# If they're standard RMSNorm, weight is the gain parameter.
# If Qwen3_5RMSNorm, weight was initialized to zeros during training and (1+w) is the gain.
# To determine: look at weights — if they're small and centered, they're probably regular RMSNorm gains.
# If they're -1.0 (i.e., 1+w ≈ 0 when w=-1), they're Qwen3_5RMSNorm.

print("\n=== CRITICAL: Determining q_norm/k_norm formula ===")
qn0 = load_raw('model.language_model.layers.3.self_attn.q_norm.weight')
kn0 = load_raw('model.language_model.layers.3.self_attn.k_norm.weight')
ln0 = load_raw('model.language_model.layers.0.input_layernorm.weight')

print(f"q_norm weights: NOT zeros, range [{qn0.min():.4f}, {qn0.max():.4f}]")
print(f"k_norm weights: NOT zeros, range [{kn0.min():.4f}, {kn0.max():.4f}]")
print(f"input_layernorm weights: range [{ln0.min():.6f}, {ln0.max():.6f}], is_zero={torch.allclose(ln0, torch.zeros_like(ln0))}")

# If input_layernorm is zeros → Qwen3_5RMSNorm (1+w)
# If q_norm is NOT zeros → standard RMSNorm (w)
# These are INCONSISTENT if both use Qwen3_5RMSNorm.

# Let's check: if q_norm uses standard RMSNorm, effective weight = weight value (e.g. 0.22)
# If q_norm uses Qwen3_5RMSNorm, effective weight = 1.0 + weight
# The effective weight should be positive for the norm to work.
# With Qwen3_5RMSNorm: eff = 1.0 + 0.22 = 1.22 → amplification
# With standard RMSNorm: eff = 0.22 → attenuation
# The actual behavior depends on what HF uses.

print("\nHypothesis: q_norm/k_norm use STANDARD RMSNorm (not Qwen3_5RMSNorm)")
print(f"  Standard RMSNorm: output = x * rstd * weight  (weight ≈ 0.22)")
print(f"  Qwen3_5RMSNorm:   output = x * rstd * (1+weight)  (effective ≈ 1.22)")
print(f"  Ratio: {1.22/0.22:.1f}x difference!")

# Let me check HF source for this
print("\nChecking HF Qwen3.5 model source...")
# We need to look at how HF handles q_norm/k_norm
