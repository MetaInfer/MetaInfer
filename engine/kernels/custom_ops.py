"""Register external C++/CUDA kernels as opaque custom ops for torch.compile.

torch.compile cannot trace into pybind11 C++ extensions (FakeTensor access violation).
Wrapping them as torch.library.custom_op makes them opaque black-box ops in the FX graph.
"""
from __future__ import annotations

import torch
from flash_attn import flash_attn_with_kvcache as _fa_kvcache
from flash_attn import flash_attn_varlen_func as _fa_varlen


# ---- flash_attn_with_kvcache ----

@torch.library.custom_op("meta_infer::flash_attn_with_kvcache", mutates_args=())
def flash_attn_with_kvcache_op(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    return _fa_kvcache(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
    )

@flash_attn_with_kvcache_op.register_fake
def _(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal):
    return torch.empty_like(q)


# ---- flash_attn_varlen_func (for prefill, not used in graph but register for completeness) ----

@torch.library.custom_op("meta_infer::flash_attn_varlen_func", mutates_args=())
def flash_attn_varlen_func_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    return _fa_varlen(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        softmax_scale=softmax_scale,
    )

@flash_attn_varlen_func_op.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, softmax_scale):
    return torch.empty_like(q)
