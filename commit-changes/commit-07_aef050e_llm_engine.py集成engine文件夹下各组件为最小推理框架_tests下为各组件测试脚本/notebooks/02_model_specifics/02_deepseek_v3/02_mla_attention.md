# Multi-Latent Attention (MLA)

## Core Idea

MLA compresses the Key and Value projections through a low-rank bottleneck, dramatically reducing the KV cache size while maintaining expressiveness close to full MHA.

## Comparison with Standard Attention

### Standard MHA/GQA
```
hidden_state [B, H]
    ↓
Q_proj: [H → N_q × D_h]     → Q [B, N_q, D_h]
K_proj: [H → N_kv × D_h]    → K [B, N_kv, D_h]
V_proj: [H → N_kv × D_h]    → V [B, N_kv, D_h]
    ↓
KV Cache stores: K [N_kv × D_h] + V [N_kv × D_h] per token per layer
```

### MLA
```
hidden_state [B, H]
    ↓
kv_a_proj: [H → kv_lora_rank + qk_rope_head_dim]  → compressed_kv [B, R+D_rope]
    ↓
Split:  c_kv [B, R]  (latent vector)
        k_rope [B, D_rope]  (RoPE portion of key)
    ↓
kv_b_proj: [R → N_kv × (D_nope + D_v)]  → K_nope, V  (up-projected)
    ↓
K = concat(K_nope, RoPE(k_rope))
    ↓
KV Cache stores: c_kv [R] + k_rope [D_rope] per token per layer
```

## Key Dimensions

```python
kv_lora_rank = 512         # Bottleneck dimension (R)
qk_nope_head_dim = 128     # Non-positional part of Q/K
qk_rope_head_dim = 64      # Positional (RoPE) part of Q/K
v_head_dim = 128            # Value head dimension
num_heads = 128             # Number of attention heads
```

## KV Cache Savings

| Method | Cache per token per layer | Example (128 heads × 128 dim) |
|--------|--------------------------|-------------------------------|
| MHA | `2 × N_h × D_h` | 2 × 128 × 128 = 32,768 |
| GQA (8 groups) | `2 × N_kv × D_h` | 2 × 8 × 128 = 2,048 |
| MLA | `R + D_rope` | 512 + 64 = 576 |

MLA achieves ~56x compression vs MHA, ~3.6x vs GQA-8, with comparable quality.

## Inference Modes

### "Absorb" Mode (Optimized for Inference)
The key insight: the up-projection `kv_b_proj` can be mathematically absorbed into the query projection and output projection, so attention operates directly on the compressed latent vectors.

```
Standard: Q × K^T = Q × (W_UK × c_kv)^T = (Q × W_UK^T) × c_kv^T
Absorbed: Q_absorbed = Q × W_UK^T, then Q_absorbed × c_kv^T
```

This means:
1. During prefill: Store only `c_kv` and `k_rope` in cache (not the full K, V)
2. During decode: Compute attention using the compressed cache directly
3. The `kv_b_proj` weight is folded into Q and O projections

### Decoupled RoPE
Only a small portion of Q and K gets RoPE applied:
```python
q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
k_nope, k_rope = k.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

# Only apply RoPE to the rope portions
q_rope = apply_rope(q_rope, positions)
k_rope = apply_rope(k_rope, positions)

# Recombine
q = concat(q_nope, q_rope)
k = concat(k_nope, k_rope)
```

## Impact on Inference Framework

### KV Cache Structure
```python
# Instead of standard [2, layers, blocks, block_size, kv_heads, head_dim]
# MLA stores:
kv_cache_shape = [layers, blocks, block_size, kv_lora_rank + qk_rope_head_dim]
# Much smaller per-token footprint
```

### Attention Kernel
Standard attention kernels (FlashAttention) cannot be used directly with MLA because the Q/K dot product has different dimensionality. Specialized kernels are needed:
- **FlashMLA**: Custom kernel for MLA attention
- **Absorbed attention**: Q is pre-multiplied by `W_UK^T` before calling standard attention

### Weight Loading
MLA models have different projection weight names:
```python
# Instead of: q_proj, k_proj, v_proj, o_proj
# MLA has: q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj
# Plus potential q_a_layernorm, kv_a_layernorm
```
