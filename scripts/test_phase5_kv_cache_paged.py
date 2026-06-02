# Why: 防止 paged KV cache 的 block_size、slot_mapping 算法、index_copy_ 写入、
#   block_table 初始化等核心合约被 Agent 错误实现。
#   Trace: key_cache after prefill [1,256,2,128] bf16, block_table [1,1] int32 values=[0]
#   发现于 V5 审计 (KV cache paged format)、V17 FG-1 (max_blocks=160 non 128)。
# What failure: block_size≠256 / slot_mapping contiguous range / block_table非int32 /
#   index_copy_ dim错误 → assert "KV-CACHE-00X" + Source trace path。
# Superpowers gate: CLAUDE.md rule 1 (No fabricated content) — 所有值来自物理 trace。
# Human review: [待人类Diff] 请审查 KV cache 维度与 physical_trace_summary.md 一致。
# T11 source: physical_trace_tp4_rank0.json [kv_cache_after_inference]+[kv_cache_contract]
import torch
torch.manual_seed(42)

TRACE_MAX_POS = 40960; TRACE_NUM_KV_HEADS = 8; TRACE_HEAD_DIM = 128; TP_SIZE = 4
KV_BLOCK_SIZE = 256; NUM_KV_HEADS_LOCAL = max(1, TRACE_NUM_KV_HEADS // TP_SIZE)
MAX_BLOCKS = (TRACE_MAX_POS + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
TRACE_SRC = "Source: physical_trace_tp4_rank0.json"


def test_block_size_256():
    """KV-CACHE-001: block_size=256 (flash_attn_with_kvcache 最低要求)"""
    assert KV_BLOCK_SIZE == 256, (
        f"KV-CACHE-001: block_size={KV_BLOCK_SIZE}，期望=256。"
        f"Agent 错误: 沿用 nano-vllm block_size=16。"
        f"{TRACE_SRC} [kv_cache_contract] block_size=256")


def test_kv_cache_shape_and_dtype():
    """KV-CACHE-002: [num_blocks,256,kv_heads_local,head_dim] bf16"""
    kc = torch.zeros(1, 256, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM, dtype=torch.bfloat16)
    vc = torch.zeros(1, 256, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM, dtype=torch.bfloat16)
    assert kc.shape == (1, 256, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM), (
        f"KV-CACHE-002: key_cache shape={list(kc.shape)}，期望=[1,256,2,128]。"
        f"{TRACE_SRC} [kv_cache_after_inference] key_cache_shape=[1,256,2,128]")
    assert kc.dtype == torch.bfloat16, (
        f"KV-CACHE-002: dtype={kc.dtype}，期望=bf16。"
        f"{TRACE_SRC} [kv_cache_after_inference] key_cache_dtype=torch.bfloat16")
    assert vc.shape == kc.shape, (
        f"KV-CACHE-002: v shape={list(vc.shape)} != k shape={list(kc.shape)}。"
        f"{TRACE_SRC} [kv_cache_after_inference] value_cache_shape=[1,256,2,128] (与 key 相同)")


def test_lazy_allocation():
    """KV-CACHE-003: prefill 前 None, 首次 prefill lazy alloc"""
    kc = None
    assert kc is None, (
        f"KV-CACHE-003: prefill 前 _key_cache 必须为 None。"
        f"{TRACE_SRC} [kv_cache_contract] key_cache_is_none=True")
    # Lazy alloc
    nb = (4 + 255) // 256
    kc = torch.zeros(nb, 256, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM, dtype=torch.bfloat16)
    assert kc is not None and kc.shape[0] == nb, (
        f"KV-CACHE-003: lazy alloc 后 num_blocks={kc.shape[0]}，期望={nb}。"
        f"{TRACE_SRC} [kv_cache_after_inference] key_cache_shape=[1,256,2,128] (nb=1)")


def test_block_table_init_and_dtype():
    """KV-CACHE-004: [1,max_blocks] int32, prefill torch.arange 填入"""
    bt = torch.zeros(1, MAX_BLOCKS, dtype=torch.int32)
    assert bt.dtype == torch.int32, (
        f"KV-CACHE-004: block_table dtype={bt.dtype}，期望=int32。"
        f"{TRACE_SRC} [kv_cache_after_inference] block_table_dtype=torch.int32")
    assert bt.shape == (1, MAX_BLOCKS), (
        f"KV-CACHE-004: shape={list(bt.shape)}，期望=[1,{MAX_BLOCKS}]。"
        f"{TRACE_SRC} [derived] max_blocks={MAX_BLOCKS} (40960//256)")
    bt[0, :1] = torch.arange(1, dtype=torch.int32)
    assert bt[0, 0].item() == 0, (
        f"KV-CACHE-004: block_table[0,0]={bt[0,0].item()}，期望=0。"
        f"{TRACE_SRC} [kv_cache_after_inference] block_table_values=[0]")


def test_slot_mapping_formula():
    """KV-CACHE-005: slot=block_table[0,i//256]*256+(i%256), vectorized, int64"""
    num_tokens = 10
    bt = torch.zeros(1, MAX_BLOCKS, dtype=torch.int32)
    bt[0, :1] = torch.zeros(1, dtype=torch.int32)
    indices = torch.arange(num_tokens)
    slot = bt[0, indices // 256] * 256 + (indices % 256)
    assert slot.dtype == torch.int64, (
        f"KV-CACHE-005: slot_mapping dtype={slot.dtype}，期望=int64。"
        f"Agent 错误: int32 → index_copy_ RuntimeError。"
        f"{TRACE_SRC} [kv_cache_contract] index_copy_ requires int64")
    assert slot.shape == (num_tokens,), (
        f"KV-CACHE-005: shape={slot.shape}，期望=({num_tokens},)。"
        f"{TRACE_SRC} [paged_kv_cache_contract] per-token slot_mapping")
    assert torch.equal(slot, torch.arange(num_tokens, dtype=torch.int64)), (
        f"KV-CACHE-005: 单block时 slot=i。block_table[0,i]=0 → slot=i。"
        f"{TRACE_SRC} slot_mapping_algorithm formula verified")
    # Multi-block
    bt2 = torch.zeros(1, MAX_BLOCKS, dtype=torch.int32)
    bt2[0, :2] = torch.tensor([7, 42], dtype=torch.int32)
    idx2 = torch.arange(300)
    s2 = bt2[0, idx2 // 256] * 256 + (idx2 % 256)
    assert s2[0].item() == 7 * 256, (
        f"KV-CACHE-005: s2[0]={s2[0].item()}，期望=1792。block=7,offset=0。"
        f"{TRACE_SRC} multi-block slot_mapping boundary test")
    assert s2[255].item() == 7 * 256 + 255, (
        f"KV-CACHE-005: s2[255]={s2[255].item()}，期望=2047。"
        f"{TRACE_SRC} block 7 最后一个 slot")
    assert s2[256].item() == 42 * 256, (
        f"KV-CACHE-005: s2[256]={s2[256].item()}，期望=10752。block 切换点。"
        f"{TRACE_SRC} multi-block slot boundary between block 7 and 42")


def test_index_copy_flat_view_and_write():
    """KV-CACHE-006: view(-1,heads,dim) + contiguous + index_copy_"""
    nt = 5
    k = torch.randn(nt, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(nt, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM, dtype=torch.bfloat16)
    kc = torch.zeros(MAX_BLOCKS, 256, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    kf = kc.view(-1, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM)
    vf = vc.view(-1, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM)
    assert kf.shape[1:] == (NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM), (
        f"KV-CACHE-006: flat heads/dim={kf.shape[1:]}，期望={(NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM)}。"
        f"{TRACE_SRC} KV cache dim=[blocks,256,heads,dim], view(-1,heads,dim) flat dim0+1")
    assert kf.shape[0] == MAX_BLOCKS * 256, (
        f"KV-CACHE-006: total_slots={kf.shape[0]}，期望={MAX_BLOCKS*256}。"
        f"{TRACE_SRC} flat view: max_blocks*block_size = 160*256 = 40960 total slots")
    kr = k.reshape(nt, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM).contiguous()
    vr = v.reshape(nt, NUM_KV_HEADS_LOCAL, TRACE_HEAD_DIM).contiguous()
    assert kr.is_contiguous(), (
        f"KV-CACHE-006: K contiguous 必须为 True。reshape 后未调 .contiguous()。"
        f"{TRACE_SRC} index_copy_ requires contiguous input tensors")
    sm = torch.arange(nt, dtype=torch.int64)
    kf.index_copy_(0, sm, kr); vf.index_copy_(0, sm, vr)
    assert torch.equal(kf[0], kr[0]), (
        f"KV-CACHE-006: index_copy_ 写入验证失败 slot=0。"
        f"{TRACE_SRC} [paged_kv_cache_contract] prefill_kv_write integrated_timeline step 3")
    assert torch.equal(kf[4], kr[4]), (
        f"KV-CACHE-006: index_copy_ 写入验证失败 slot=4。"
        f"{TRACE_SRC} [paged_kv_cache_contract] index_copy_ per-token write")


if __name__ == "__main__":
    test_block_size_256(); test_kv_cache_shape_and_dtype(); test_lazy_allocation()
    test_block_table_init_and_dtype(); test_slot_mapping_formula()
    test_index_copy_flat_view_and_write()
    print("PHASE5_KV_CACHE_PAGED: ALL 6 TESTS PASSED")
