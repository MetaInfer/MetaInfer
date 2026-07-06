"""Diagnostic: Use actual engine QwenFullAttentionTP to verify the forward
computation with ColumnParallel/RowParallel and SDPA fallback.

Uses float32 throughout. Tests TP=4 per-rank vs TP=1 reference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LOCAL_RANK', '0')
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '4')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29501')

# Monkey-patch distributed before any imports
import engine.tp_layers.distributed as dist_mod
dist_mod._TP_SIZE = 4
dist_mod._TP_RANK = 0
def _noop_init():
    dist_mod._TP_SIZE = int(os.environ.get('WORLD_SIZE', '4'))
    dist_mod._TP_RANK = int(os.environ.get('RANK', '0'))
dist_mod.init_tp_distributed = _noop_init
dist_mod.is_tp_enabled = lambda: False  # Make all_reduce_sum return x.clone()
dist_mod.get_tp_group = lambda: dist_mod._TPGroup(dist_mod._TP_RANK, dist_mod._TP_SIZE)

# The custom_op all_reduce_sum checks is_tp_enabled() first;
# when False, it returns x.clone() (no communication needed)

from engine.models.qwen import QwenFullAttentionTP
from engine.kernels.rotary_embedding import make_cos_sin_cache, rotary_embedding
from engine.kernels.attention import flash_attn_varlen_func

class MockCfg:
    hidden_size = 5120
    num_hidden_layers = 64
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
    max_position_embeddings = 40960
    rope_theta = 1000000.0
    mrope_section = None
    mrope_interleaved = False

torch.manual_seed(42)
print("=== Engine FullAttention TP=4 test (float32) ===")

layer = QwenFullAttentionTP(MockCfg(), layer_idx=3)
print(f"num_heads={layer.num_heads}, num_kv_heads={layer.num_kv_heads}")
print(f"q_proj.weight: {list(layer.q_proj.weight.shape)}")
print(f"o_proj.weight: {list(layer.o_proj.weight.shape)}")

# Full reference weights (float32 for nn.Linear compatibility)
q_full = torch.randn(24*256, 5120)
k_full = torch.randn(4*256, 5120)
v_full = torch.randn(4*256, 5120)
o_full = torch.randn(5120, 24*256)
g_full = torch.randn(24*256, 5120)

# Prepare cos/sin cache
layer._cos_sin_cache_cpu = make_cos_sin_cache(
    4096, 256, 1000000.0, dtype=torch.float32)
layer._ensure_cos_sin_gpu = lambda device: setattr(
    layer, '_cos_sin_cache_gpu', layer._cos_sin_cache_cpu)

# Override KV cache
def _lazy_kv_cache_passthrough(self, num_tokens, device, dtype):
    max_blocks = (num_tokens + 255) // 256
    self._key_cache = torch.zeros(max_blocks, 256, self.num_kv_heads, self.head_dim, dtype=dtype)
    self._value_cache = torch.zeros_like(self._key_cache)
    self._block_table = torch.zeros(1, max_blocks, dtype=torch.int32)
layer._lazy_kv_cache = lambda num_tokens, device, dtype: _lazy_kv_cache_passthrough(layer, num_tokens, device, dtype)

B, S = 1, 8
hs = torch.randn(B, S, 5120)
positions = torch.arange(S, dtype=torch.int64)

# --- Per-rank simulation ---
per_rank_outs = []
for rank in range(4):
    layer.q_proj.weight.data.copy_(q_full[rank*1536:(rank+1)*1536, :])
    layer.q_gate_proj.weight.data.copy_(g_full[rank*1536:(rank+1)*1536, :])
    layer.k_proj.weight.data.copy_(k_full[rank*256:(rank+1)*256, :])
    layer.v_proj.weight.data.copy_(v_full[rank*256:(rank+1)*256, :])
    layer.o_proj.weight.data.copy_(o_full[:, rank*1536:(rank+1)*1536])
    out_r = layer(hs, positions, S)
    per_rank_outs.append(out_r)

tp_out = sum(per_rank_outs)
print(f"TP=4 output norm: {tp_out.float().norm():.4f}")

# --- TP=1 reference ---
q_ref = F.linear(hs, q_full).view(B*S, 24, 256)
k_ref = F.linear(hs, k_full).view(B*S, 4, 256)
v_ref = F.linear(hs, v_full).view(B*S, 4, 256)
g_ref = F.linear(hs, g_full).view(B, S, 24*256)

def rms_norm_ref(x, eps=1e-6):
    rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype)

q_ref = rms_norm_ref(q_ref)
k_ref = rms_norm_ref(k_ref)
rotary_embedding(positions, q_ref, k_ref, 256, layer._cos_sin_cache_gpu, is_neox=True)

cu = torch.tensor([0, B*S], dtype=torch.int32)
attn_ref = flash_attn_varlen_func(
    q_ref, k_ref, v_ref, cu, cu, B*S, B*S, causal=True, softmax_scale=256**-0.5)
attn_ref = attn_ref.view(B, S, 24*256)
attn_ref = attn_ref * torch.sigmoid(g_ref)
ref_out = F.linear(attn_ref, o_full)

print(f"TP=1 ref  norm:   {ref_out.float().norm():.4f}")

ratio = tp_out.float().norm() / ref_out.float().norm()
max_diff = (tp_out - ref_out).abs().max().item()
print(f"Ratio: {ratio:.4f}")
print(f"Max diff: {max_diff:.6f}")
print(f"{'PASS' if max_diff < 10.0 else 'FAIL'}: TP=4 {'matches' if max_diff < 10.0 else 'differs from'} reference")
