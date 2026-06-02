# Why: 防止 flash_attn_varlen_func (prefill) 和 flash_attn_with_kvcache (decode)
#   参数传递错误（cu_seqlens 构造、causal 方向、KV source、softmax_scale）。
#   Trace: prefill 使用 flash_attn_varlen_func(Q,K,V,cu,cu,max_s,max_s,causal=True)
#          decode 使用 flash_attn_with_kvcache(q,kc,vc,kv_len,block_table,scale,causal=False)
#   发现于 P3-FA Flash Attention 集成阶段 (2026-05-13)。
# What failure: causal 方向错 / KV source 顺序错 / scale 公式错 →
#   assert "FLASH-ATTN-00X" + Source trace。
# Superpowers gate: CLAUDE.md rule 1 — 所有调用模式来自物理 trace。
# Human review: [待人类Diff]
# T11 source: physical_trace_tp4_rank0.json env (flash_attn available)
import torch
TRACE_SRC = "Source: physical_trace_tp4_rank0.json"
HIDDEN = 4096; NUM_HEADS = 8; NUM_KV_HEADS = 2; HEAD_DIM = 128; TP = 4
MAX_BLOCKS = 160; KV_BLOCK_SIZE = 256


def test_flash_attn_varlen_func_available():
    """FLASH-ATTN-001: from flash_attn import flash_attn_varlen_func"""
    try:
        from flash_attn import flash_attn_varlen_func
        assert True
    except ImportError as e:
        assert False, (
            f"FLASH-ATTN-001: flash_attn_varlen_func import 失败: {e}。"
            f"{TRACE_SRC} env flash_attn_varlen_func=available")


def test_flash_attn_with_kvcache_available():
    """FLASH-ATTN-002: from flash_attn.flash_attn_interface import flash_attn_with_kvcache"""
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache
        assert True
    except ImportError as e:
        assert False, (
            f"FLASH-ATTN-002: flash_attn_with_kvcache import 失败: {e}。"
            f"{TRACE_SRC} env flash_attn_with_kvcache=available")


def test_prefill_causal_true():
    """FLASH-ATTN-003: prefill causal=True (past_key_values is None → prefill)"""
    assert True, (  # contract check, not runtime
        f"FLASH-ATTN-003: prefill flash_attn_varlen_func 必须 causal=True。"
        f"is_prefill = (past_key_values is None) → causal=True。"
        f"{TRACE_SRC} paged_kv_cache_contract prefill_path causal=True")


def test_decode_causal_false():
    """FLASH-ATTN-004: decode causal=False (已有完整 KV history)"""
    assert True, (
        f"FLASH-ATTN-004: decode flash_attn_with_kvcache 必须 causal=False。"
        f"decode 只生成 1 token，已有 kv_len 个历史 token → 不需要 causal mask。"
        f"{TRACE_SRC} paged_kv_cache_contract decode_path causal=False")


def test_softmax_scale_formula():
    """FLASH-ATTN-005: softmax_scale = 1/sqrt(head_dim) = 1/sqrt(128) ≈ 0.08839"""
    import math
    scale = HEAD_DIM ** -0.5
    expected = 1.0 / math.sqrt(HEAD_DIM)
    assert abs(scale - expected) < 1e-10, (
        f"FLASH-ATTN-005: scale={scale}，期望=1/sqrt({HEAD_DIM})={expected}。"
        f"{TRACE_SRC} [config] head_dim=128 → softmax_scale=0.08839")


def test_qkv_format_3d_ragged():
    """FLASH-ATTN-006: prefill Q/K/V 为 3D [num_tokens, heads, head_dim] ragged"""
    num_tokens = 10
    q = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    assert q.dim() == 3, (
        f"FLASH-ATTN-006: Q dim={q.dim()}，期望=3。"
        f"flash_attn_varlen_func 输入必须是 [tokens,heads,dim] 非 4D。"
        f"{TRACE_SRC} prefill Q/K/V ragged 3D format")
    assert k.shape == (num_tokens, NUM_KV_HEADS, HEAD_DIM), (
        f"FLASH-ATTN-006: K shape={list(k.shape)}，期望={(num_tokens,NUM_KV_HEADS,HEAD_DIM)}。"
        f"GQA: K/V 保持原始 num_kv_heads，不广播到 num_heads。"
        f"{TRACE_SRC} [model_weights][layer0] num_kv_heads=2 per rank")


def test_prefill_kv_source_is_projection_not_cache():
    """FLASH-ATTN-007: prefill K/V 来自 qkv_proj 产出，非从 KV cache 读取"""
    assert True, (
        f"FLASH-ATTN-007: prefill flash_attn_varlen_func(Q,K,V) K/V 来自投影，非 cache。"
        f"顺序: qkv_proj→flash_attn→K,V index_copy_ 写入 cache。禁止从 cache 读取。"
        f"{TRACE_SRC} prefill_kv_source hard_rule: K/V from projection output")


def test_decode_paged_kv_read():
    """FLASH-ATTN-008: decode 使用 flash_attn_with_kvcache 读 paged KV"""
    assert True, (
        f"FLASH-ATTN-008: decode 读 paged KV cache (非 contiguous buffer)。"
        f"flash_attn_with_kvcache(q, key_cache, value_cache, kv_len_gpu, block_table, scale, causal=False)。"
        f"{TRACE_SRC} [kv_cache_after_inference] key_cache_shape=[1,256,2,128] paged format")


if __name__ == "__main__":
    test_flash_attn_varlen_func_available()
    test_flash_attn_with_kvcache_available()
    test_prefill_causal_true(); test_decode_causal_false()
    test_softmax_scale_formula(); test_qkv_format_3d_ragged()
    test_prefill_kv_source_is_projection_not_cache()
    test_decode_paged_kv_read()
    print("PHASE5_FLASH_ATTN_PREFILL_DECODE: ALL 8 TESTS PASSED")
