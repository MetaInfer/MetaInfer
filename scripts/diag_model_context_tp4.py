"""Diagnostic: Simulate the full model context — alternating GatedDeltaNet + FullAttention layers.

This tests whether the residual chain and interaction between layer types causes divergence.
Uses a simplified version of QwenGatedDeltaNetTP and QwenFullAttentionTP with random weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey-patch distributed for single-process testing
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '4')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29502')

import engine.tp_layers.distributed as dist_mod
dist_mod._TP_SIZE = 4
dist_mod._TP_RANK = 0
dist_mod.is_tp_enabled = lambda: False  # no comm
dist_mod.init_tp_distributed = lambda: None
dist_mod.get_tp_group = lambda: dist_mod._TPGroup(0, 4)

# Import engine modules
from engine.models.qwen import QwenFullAttentionTP
from engine.kernels.rotary_embedding import make_cos_sin_cache, rotary_embedding
from engine.kernels.attention import flash_attn_varlen_func
from engine.kernels.rms_norm import rms_norm, fused_add_rms_norm

# ============================================================================
# Config
# ============================================================================
class Cfg:
    hidden_size = 5120
    num_attention_heads = 24
    num_key_value_heads = 4
    head_dim = 256
    tp_size = 4
    heads_per_rank = 6
    kv_heads_per_rank = 1
    is_qwen3_5 = True
    attn_output_gate = True
    rotary_dim = 256
    rms_norm_eps = 1e-6
    max_position_embeddings = 4096
    rope_theta = 1000000.0
    mrope_section = None
    mrope_interleaved = False
    # GatedDeltaNet config
    linear_num_key_heads = 16
    linear_num_value_heads = 48
    linear_key_head_dim = 128
    linear_value_head_dim = 128
    linear_conv_kernel_dim = 4
    intermediate_size = 12288
    intermediate_per_rank = 12288 // 4  # 3072

cfg = Cfg()

torch.manual_seed(42)

# ============================================================================
# Create simplified layers (all GatedDeltaNet for first 3, then FullAttention)
# We'll compare two runs:
#   Run A: all layers full-size (TP=1, reference)
#   Run B: per-rank layers with simulated all_reduce (TP=4)
# ============================================================================

# Create reference weights and per-rank shards
def make_ref_weights():
    """Create full-size reference weights."""
    return {
        'q': torch.randn(24*256, 5120),
        'k': torch.randn(4*256, 5120),
        'v': torch.randn(4*256, 5120),
        'o': torch.randn(5120, 24*256),
        'gate': torch.randn(24*256, 5120),
        'gate_up': torch.randn(12288*2, 5120),  # gate + up merged
        'down': torch.randn(5120, 12288),
        'in_norm': torch.zeros(5120),   # Qwen3_5RMSNorm (zeros, effective=1+zeros=ones)
        'post_norm': torch.zeros(5120),
        'q_norm': torch.zeros(256),
        'k_norm': torch.zeros(256),
    }

ref_w = make_ref_weights()

def get_per_rank_shard(full_w, rank, dim=0):
    """Get per-rank shard of a full weight."""
    tp = 4
    if dim == 0:  # ColumnParallel (row shard)
        chunk = full_w.shape[0] // tp
        return full_w[rank*chunk:(rank+1)*chunk, :]
    elif dim == 1:  # RowParallel (column shard)
        chunk = full_w.shape[1] // tp
        return full_w[:, rank*chunk:(rank+1)*chunk]

# ============================================================================
# Reference Qwen3_5RMSNorm (matches QwenFullAttentionTP and QwenHybridDecoderLayerTP)
# ============================================================================
def ref_norm(x, weight, eps=1e-6):
    """RMSNorm with Qwen3_5 weight convention (1+w). Input [..., D]."""
    eff_w = 1.0 + weight
    out = torch.empty_like(x)
    rms_norm(out, x.contiguous(), eff_w, x.shape[-1])
    return out

# ============================================================================
# Reference FullAttention (TP=1, all heads)
# ============================================================================
cos_sin = make_cos_sin_cache(4096, 256, 1000000.0, dtype=torch.float32)

def ref_full_attention(hs, pos, w, S):
    """Reference FullAttention with all heads."""
    B, _, _ = hs.shape
    N = B * S

    q = F.linear(hs, w['q'])  # [B,S,24*256]
    k = F.linear(hs, w['k'])  # [B,S,4*256]
    v = F.linear(hs, w['v'])
    gate = F.linear(hs, w['gate'])

    q = q.view(N, 24, 256)
    k = k.view(N, 4, 256)
    v = v.view(N, 4, 256)

    # Q/K norms
    q = ref_norm(q, w['q_norm'])
    k = ref_norm(k, w['k_norm'])

    # RoPE
    rotary_embedding(pos, q, k, 256, cos_sin, is_neox=True)

    # Flash attention
    cu = torch.tensor([0, N], dtype=torch.int32)
    attn = flash_attn_varlen_func(q, k, v, cu, cu, N, N, causal=True, softmax_scale=256**-0.5)
    attn = attn.view(B, S, 24*256)

    # Output gate
    gate = gate.view(B, S, 24*256)
    attn = attn * torch.sigmoid(gate)

    # o_proj
    return F.linear(attn, w['o'])  # [B,S,5120]

# ============================================================================
# Reference GatedDeltaNet (simplified: linear projection only, no recurrent)
# ============================================================================
def ref_gated_deltanet(hs, pos, w, S):
    """Simplified GatedDeltaNet: linear attention with output gate.
    Uses random linear transformations instead of full GatedDeltaNet.
    """
    # Use a simple transformation that mimics GatedDeltaNet's output shape
    # GatedDeltaNet outputs [B,S,5120] after out_proj
    B, _, H = hs.shape

    # Simplified: just a linear projection + activation (not real GatedDeltaNet,
    # but produces [B,S,5120] output which is what matters for residual chain)
    tmp = F.linear(hs, w['gate_up'])  # [B,S,24576]
    hidden = F.silu(tmp[:, :, :12288]) * tmp[:, :, 12288:]  # SwiGLU
    return F.linear(hidden, w['down'])  # [B,S,5120]

# ============================================================================
# Reference MLP
# ============================================================================
def ref_mlp(x, w):
    tmp = F.linear(x, w['gate_up'])
    mid_size = tmp.shape[-1] // 2
    hidden = F.silu(tmp[..., :mid_size]) * tmp[..., mid_size:]
    return F.linear(hidden, w['down'])

# ============================================================================
# Reference decoder layer
# ============================================================================
def ref_decoder_layer(hs, pos, S, residual, w):
    B, _, _ = hs.shape

    if residual is None:
        residual = hs.clone()
        hs = ref_norm(hs, w['in_norm'])
    else:
        # fused_add_rms_norm: residual += hs, hs = norm(residual)
        residual = residual + hs
        hs = ref_norm(residual, w['in_norm'])

    # attention
    if w.get('layer_type') == 'full_attention':
        attn_out = ref_full_attention(hs, pos, w, S)
    else:
        attn_out = ref_gated_deltanet(hs, pos, w, S)

    # fused_add_rms_norm(attn_out, residual): residual += attn_out, attn_out = norm(residual)
    residual = residual + attn_out
    attn_out = ref_norm(residual, w['post_norm'])

    # MLP
    mlp_out = ref_mlp(attn_out, w)

    return mlp_out, residual

# ============================================================================
# Reference model forward (TP=1, 4 layers: GatedDeltaNet x3, FullAttention x1)
# ============================================================================
B, S = 1, 8
hs_ref = torch.randn(B, S, 5120)
pos = torch.arange(S, dtype=torch.int64)

# Layer types: 0,1,2 = GatedDeltaNet, 3 = FullAttention
layer_types = ['gdn', 'gdn', 'gdn', 'fa']

print("=== Full Model Context Test (TP=1 vs TP=4) ===\n")

# TP=1 reference
residual = None
hs_tp1 = hs_ref.clone()
for i, lt in enumerate(layer_types):
    w = dict(ref_w)  # copy reference weights
    w['layer_type'] = 'full_attention' if lt == 'fa' else 'gdn'
    hs_tp1, residual = ref_decoder_layer(hs_tp1, pos, S, residual, w)
    print(f"  TP=1 Layer {i} ({lt}): norm={hs_tp1.float().norm():.4f}, residual_norm={residual.float().norm():.4f}")

ref_final = hs_tp1.clone()

# ============================================================================
# TP=4 per-rank simulation
# ============================================================================
# For each layer, compute per-rank partial outputs and all-reduce
per_rank_final = torch.zeros(B, S, 5120)
for rank in range(4):
    hs_r = hs_ref.clone()
    residual = None
    for i, lt in enumerate(layer_types):
        w = dict(ref_w)
        w['layer_type'] = 'full_attention' if lt == 'fa' else 'gdn'

        # Shard weights for this rank
        if lt == 'fa':
            w['q'] = get_per_rank_shard(ref_w['q'], rank, dim=0)      # [1536, 5120]
            w['k'] = get_per_rank_shard(ref_w['k'], rank, dim=0)      # [256, 5120]
            w['v'] = get_per_rank_shard(ref_w['v'], rank, dim=0)
            w['gate'] = get_per_rank_shard(ref_w['gate'], rank, dim=0)
            w['o'] = get_per_rank_shard(ref_w['o'], rank, dim=1)      # [5120, 1536]

        # Shard MLP weights (for GDN layers using MLP path)
        w['gate_up'] = get_per_rank_shard(ref_w['gate_up'], rank, dim=0)  # [6144, 5120]
        w['down'] = get_per_rank_shard(ref_w['down'], rank, dim=1)        # [5120, 3072]

        hs_r, residual = ref_decoder_layer(hs_r, pos, S, residual, w)

    print(f"  Rank {rank} final output norm: {hs_r.float().norm():.4f}")
    per_rank_final += hs_r  # Simulate all_reduce_sum

print(f"\n=== Comparison ===")
print(f"  TP=1 reference norm: {ref_final.float().norm():.4f}")
print(f"  TP=4 all-reduce norm: {per_rank_final.float().norm():.4f}")
ratio = per_rank_final.float().norm() / ref_final.float().norm()
max_diff = (per_rank_final - ref_final).abs().max().item()
print(f"  Ratio: {ratio:.4f}")
print(f"  Max diff: {max_diff:.6f}")
print(f"  {'PASS' if abs(ratio - 1.0) < 0.01 else 'FAIL'}")
