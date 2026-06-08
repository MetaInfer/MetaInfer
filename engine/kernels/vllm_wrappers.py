"""
Phase 1 — vLLM/FlashAttn kernel wrappers (7 numerical primitives).

All signatures must match inference_blueprint.json > qwen3_kernel_contracts
and kernel_replacement_plan.md §九 exactly.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Kernel 1: rms_norm
#   Source: vllm/_custom_ops.py:420-423
#   Contract: out pre-allocated, input contiguous, all same dtype
# ---------------------------------------------------------------------------
from vllm._custom_ops import rms_norm as _vllm_rms_norm


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Black-box wrapper for vLLM rms_norm CUDA kernel.

    Contract:
        out:     [*, H]  bf16/fp16/fp32, contiguous, pre-allocated (empty_like)
        input:   [*, H]  bf16/fp16/fp32, MUST be contiguous
        weight:  [H]     bf16/fp16/fp32, contiguous
        epsilon: float   (typical: 1e-6)

    Operation: out = rms_norm(input) * weight  (internal fp32 compute)
    Do NOT modify this function's internal logic.
    """
    _vllm_rms_norm(out, input, weight, epsilon)


# ---------------------------------------------------------------------------
# Kernel 2: fused_add_rms_norm
#   Source: vllm/_custom_ops.py:420-423
#   Contract: dual in-place (residual += input; input = rms_norm(residual))
# ---------------------------------------------------------------------------
from vllm._custom_ops import fused_add_rms_norm as _vllm_fused_add_rms_norm


def fused_add_rms_norm(
    input: torch.Tensor,       #! in-place: becomes rms_norm(residual)
    residual: torch.Tensor,    #! in-place: becomes residual + input
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Black-box wrapper for vLLM fused_add_rms_norm CUDA kernel.

    Contract:
        input:    [*, H]  bf16, contiguous  (sub-layer output, e.g. attention output)
        residual: [*, H]  bf16, contiguous  (residual state)
        weight:   [H]     bf16, contiguous  (RMSNorm weight)
        epsilon:  float

    Two in-place operations:
        1. residual = residual + input          (fused residual)
        2. input    = rms_norm(residual) * weight  (normalised, for next sub-layer)

    Post-mlp call uses THIS layer's post_attention_layernorm.weight.
    All 4 calls in decode use self-layer weight (physically traced).
    Do NOT modify this function's internal logic.
    """
    _vllm_fused_add_rms_norm(input, residual, weight, epsilon)


# ---------------------------------------------------------------------------
# Kernel 3: silu_and_mul
#   Source: vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda
#   Contract: out pre-allocated; input = [*, 2*d] (front gate, rear up)
#   Must import vllm._C FIRST to trigger torch.ops._C registration.
# ---------------------------------------------------------------------------
import vllm._C  # noqa: F401 — triggers torch.ops._C.silu_and_mul registration


def silu_and_mul(
    out: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """Black-box wrapper for vLLM silu_and_mul CUDA kernel.

    Contract:
        input: [*, 2*d]  bf16, contiguous  (gate+up merged projection output)
        out:   [*, d]    bf16, contiguous, pre-allocated

    Operation: out = SiLU(input[..., :d]) * input[..., d:]

    Pre-requisite: gate_proj and up_proj must be merged into single GEMM.
    The output [*, 2*d] has front-half = gate, rear-half = up.
    Do NOT modify this function's internal logic.
    """
    torch.ops._C.silu_and_mul(out, input)


# ---------------------------------------------------------------------------
# Kernel 4: rotary_embedding
#   Source: vllm/_custom_ops.py:400-410
#   Contract: q/k in-place, 2D input [num_tokens, heads, dim] (NOT 4D)
#   Qwen3: is_neox=True
# ---------------------------------------------------------------------------
from vllm._custom_ops import rotary_embedding as _vllm_rotary_embedding


def rotary_embedding(
    positions: torch.Tensor,       # [num_tokens] int64
    query: torch.Tensor,           #! [num_tokens, num_heads, head_dim] bf16, in-place
    key: torch.Tensor | None,      #! [num_tokens, num_kv_heads, head_dim] bf16, in-place
    head_size: int,
    cos_sin_cache: torch.Tensor,   # [max_position, head_size]
    is_neox: bool,                 # True for GPT-NeoX style (Qwen3)
) -> None:
    """Black-box wrapper for vLLM rotary_embedding CUDA kernel.

    Contract:
        positions:      [num_tokens]         int64, position indices
        query:          [num_tokens, N, D]   bf16, in-place modified
        key:            [num_tokens, Nkv, D] bf16, in-place modified (may be None)
        head_size:      int                  per-head dimension (Qwen3=128)
        cos_sin_cache:  [max_pos, head_size] pre-computed cache
        is_neox:        bool                 Qwen3 uses True (GPT-NeoX style)

    IMPORTANT:
        - cos_sin_cache format is [max_position, head_size] NOT [max_pos, 2*head_size].
          vLLM kernel internally decodes: cos = cache[pos, :head_size//2],
          sin = cache[pos, head_size//2:].
        - query and key MUST be contiguous 2D tensors [num_tokens, heads, dim].
          4D [B,S,H,D] inputs must be reshaped to 2D BEFORE calling.
    Do NOT modify this function's internal logic.
    """
    _vllm_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


# ---------------------------------------------------------------------------
# Kernel 5: _get_cos_sin_cache  (module-level registry + lazy GPU transfer)
#   Source: vllm model_executor/layers/rotary_embedding/base.py:76-84
#   Contract: [max_pos, head_size] format, fp32 compute then .to(input_dtype)
# ---------------------------------------------------------------------------

_COS_SIN_CACHE_REGISTRY: dict[tuple, torch.Tensor] = {}


def make_cos_sin_cache(
    max_position: int,
    head_size: int,
    rope_theta: float = 1000000.0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct cos_sin_cache for vLLM rotary_embedding kernel.

    Format: [max_position, head_size]  (NOT 2*head_size!)
      cache[pos, :head_size//2] = cos values
      cache[pos, head_size//2:] = sin values
    vLLM kernel internally handles NeoX-style cos/sin repetition.

    Qwen3-8B parameters:
        max_position = 32768
        head_size = 128
        rope_theta = 1000000.0

    Numerically verified: matches vLLM RotaryEmbeddingBase._compute_cos_sin_cache.
    """
    # fp32 compute, then cast to target dtype (RoPE precision law)
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, head_size, 2, dtype=torch.float32, device=device) / head_size
    ))
    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)   # [max_pos, head_size//2]
    cos = freqs.cos().to(dtype=dtype)                 # [max_pos, head_size//2]
    sin = freqs.sin().to(dtype=dtype)                 # [max_pos, head_size//2]
    return torch.cat((cos, sin), dim=-1)              # [max_pos, head_size]


def _get_cos_sin_cache(
    max_pos: int,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Module-level registry-backed cos_sin_cache factory.

    Caches created caches by key=(max_pos, head_dim, rope_theta) to avoid
    redundant allocation across 36 DecoderLayers. Saves ~36 * 8MB = 288MB.

    Calling convention for layer __init__:
        # CPU creation in __init__
        self._cos_sin_cache_cpu = _get_cos_sin_cache(max_pos, head_dim, rope_theta)
        self._cos_sin_cache_gpu = None  # lazy GPU transfer on first forward

    Lazy GPU transfer in forward():
        if self._cos_sin_cache_gpu is None:
            self._cos_sin_cache_gpu = self._cos_sin_cache_cpu.to(x.device)
        # Use self._cos_sin_cache_gpu for rotary_embedding calls
    """
    key = (max_pos, head_dim, rope_theta)
    if key not in _COS_SIN_CACHE_REGISTRY:
        _COS_SIN_CACHE_REGISTRY[key] = make_cos_sin_cache(
            max_position=max_pos,
            head_size=head_dim,
            rope_theta=rope_theta,
            dtype=dtype,
            device=device,
        )
    return _COS_SIN_CACHE_REGISTRY[key]


# ---------------------------------------------------------------------------
# Kernel 6: flash_attn_varlen_func
#   Source: flash_attn.flash_attn_interface (flash_attn installed package)
#   Contract: nocompile — direct import, no custom_op registration needed
#   Usage: prefill attention, Q/K/V ragged 3D [num_tokens, heads, dim]
# ---------------------------------------------------------------------------
from flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: F401 — re-exported below
# flash_attn_varlen_func is imported directly for use in prefill attention:
#   flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
#                          max_seqlen_q, max_seqlen_k, causal=True)
# q/k/v format: [num_tokens, num_heads, head_dim] 3D ragged (no permute needed)


# ---------------------------------------------------------------------------
# Kernel 7: flash_attn_with_kvcache
#   Source: flash_attn.flash_attn_interface
#   Contract: nocompile — direct import, no custom_op registration needed
#   Usage: decode attention with paged KV cache
# ---------------------------------------------------------------------------
from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # noqa: F401
# flash_attn_with_kvcache is imported directly for use in decode attention:
#   flash_attn_with_kvcache(q, k_cache, v_cache,
#                           cache_seqlens=kv_len_gpu,
#                           block_table=block_table,
#                           softmax_scale=1.0/sqrt(head_dim),
#                           causal=False)
# q format: [batch, num_heads, head_dim] (reshape from [1,1,H,D]→[1,H,D])
# block_size must be >= 256; block_table must be int32
