"""Diagnostic: Simple residual chain test — 4 identical layers (all FullAttention style)
to verify that weight sharding + residual chain = correct for all layers.
"""
import torch
import torch.nn.functional as F
import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '4')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29503')

import engine.tp_layers.distributed as dist_mod
dist_mod._TP_SIZE = 4
dist_mod._TP_RANK = 0
dist_mod.is_tp_enabled = lambda: False
dist_mod.init_tp_distributed = lambda: None
dist_mod.get_tp_group = lambda: dist_mod._TPGroup(0, 4)

from engine.kernels.rotary_embedding import make_cos_sin_cache, rotary_embedding
from engine.kernels.attention import flash_attn_varlen_func
from engine.kernels.rms_norm import rms_norm

def manual_rms_norm(x, effective_weight, eps=1e-6):
    """Manual RMS norm matching Qwen3_5RMSNorm behavior."""
    out = torch.empty_like(x)
    rms_norm(out, x.contiguous(), effective_weight, x.shape[-1])
    return out

cos_sin = make_cos_sin_cache(4096, 256, 1000000.0, dtype=torch.float32)

torch.manual_seed(42)
B, S = 1, 8
N = B * S
TP = 4
HIDDEN = 5120
NH = 24
NKV = 4
HD = 256

# Full weights for all layers
W = {
    'q': [torch.randn(NH*HD, HIDDEN) for _ in range(4)],
    'k': [torch.randn(NKV*HD, HIDDEN) for _ in range(4)],
    'v': [torch.randn(NKV*HD, HIDDEN) for _ in range(4)],
    'o': [torch.randn(HIDDEN, NH*HD) for _ in range(4)],
    'gate': [torch.randn(NH*HD, HIDDEN) for _ in range(4)],
    'g_up': [torch.randn(24576, HIDDEN) for _ in range(4)],  # 12288*2 gate_up
    'down': [torch.randn(HIDDEN, 12288) for _ in range(4)],  # down
    'in_norm': [torch.zeros(HIDDEN) for _ in range(4)],
    'post_norm': [torch.zeros(HIDDEN) for _ in range(4)],
    'q_norm': [torch.zeros(HD) for _ in range(4)],
    'k_norm': [torch.zeros(HD) for _ in range(4)],
}

def fa_layer(hs, pos, S, w, residual):
    """FullAttention-style decoder layer."""
    if residual is None:
        residual = hs.clone()
        hs = manual_rms_norm(hs, 1.0 + w['in_norm'])
    else:
        residual = residual + hs
        hs = manual_rms_norm(residual, 1.0 + w['in_norm'])

    # Attention
    q = F.linear(hs, w['q']).view(N, w['q'].shape[0]//HD, HD)
    k = F.linear(hs, w['k']).view(N, w['k'].shape[0]//HD, HD)
    v = F.linear(hs, w['v']).view(N, w['v'].shape[0]//HD, HD)
    g = F.linear(hs, w['gate'])

    q = manual_rms_norm(q, 1.0 + w['q_norm'])
    k = manual_rms_norm(k, 1.0 + w['k_norm'])
    rotary_embedding(pos, q, k, HD, cos_sin, is_neox=True)

    cu = torch.tensor([0, N], dtype=torch.int32)
    attn = flash_attn_varlen_func(q, k, v, cu, cu, N, N, causal=True, softmax_scale=HD**-0.5)
    attn = attn.view(B, S, w['q'].shape[0])
    attn = attn * torch.sigmoid(g)
    attn_out = F.linear(attn, w['o'])

    residual = residual + attn_out
    attn_out = manual_rms_norm(residual, 1.0 + w['post_norm'])

    # MLP
    tmp = F.linear(attn_out, w['g_up'])
    mid = tmp.shape[-1] // 2
    hidden = F.silu(tmp[..., :mid]) * tmp[..., mid:]
    mlp_out = F.linear(hidden, w['down'])

    return mlp_out, residual

# TP=1 reference
hs_ref = torch.randn(B, S, HIDDEN)
pos = torch.arange(S, dtype=torch.int64)
residual = None
hs_tp1 = hs_ref.clone()
for i in range(4):
    w = {k: v[i] for k, v in W.items()}
    hs_tp1, residual = fa_layer(hs_tp1, pos, S, w, residual)
    print(f"  TP=1 L{i}: out_norm={hs_tp1.float().norm():.4f}")

ref_out = hs_tp1.clone()

# TP=4 per-rank simulation
final = torch.zeros(B, S, HIDDEN)
for rank in range(4):
    hs_r = hs_ref.clone()
    residual = None
    for i in range(4):
        w = {}
        w['q'] = W['q'][i][rank*1536:(rank+1)*1536, :]
        w['k'] = W['k'][i][rank*256:(rank+1)*256, :]
        w['v'] = W['v'][i][rank*256:(rank+1)*256, :]
        w['o'] = W['o'][i][:, rank*1536:(rank+1)*1536]
        w['gate'] = W['gate'][i][rank*1536:(rank+1)*1536, :]
        w['g_up'] = W['g_up'][i][rank*6144:(rank+1)*6144, :]   # 24576/4 = 6144
        w['down'] = W['down'][i][:, rank*3072:(rank+1)*3072]     # 12288/4 = 3072
        w['in_norm'] = W['in_norm'][i]
        w['post_norm'] = W['post_norm'][i]
        w['q_norm'] = W['q_norm'][i]
        w['k_norm'] = W['k_norm'][i]
        hs_r, residual = fa_layer(hs_r, pos, S, w, residual)
    final += hs_r  # simulate all_reduce_sum

print(f"\n=== Comparison ===")
print(f"TP=1 ref: {ref_out.float().norm():.4f}")
print(f"TP=4:     {final.float().norm():.4f}")
ratio = final.float().norm() / ref_out.float().norm()
md = (final - ref_out).abs().max().item()
print(f"Ratio:    {ratio:.6f}  MaxDiff: {md:.6f}")
print(f"{'PASS' if abs(ratio-1.0) < 0.01 else 'FAIL'}")
