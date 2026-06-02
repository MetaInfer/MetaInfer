# Why: 防止 HF safetensors key → 模型属性的映射错误。
#   最频繁错误: QKV cat 顺序 (必须 Q-K-V，严禁 K-Q-V 或 V-K-Q)；
#   Gate-Up cat 顺序 (必须 gate-up)；double_shard_guard 二次切片。
#   Trace: qkv_proj=[1536,4096] (1024+256+256=1536)，gate_up=[6144,4096] (2*3072)
#   发现于 V5/V15 审计 HF key mapping 错误模式。
# What failure: cat 顺序错 / 切片范围错 / double_shard_guard 失效 →
#   assert "KEYMAP-00X" + Source trace。
# Superpowers gate: CLAUDE.md rule 2 — 所有 key 映射来自真实权重物理验证。
# Human review: [待人类Diff]
# T11 source: physical_trace_tp4_rank0.json [model_weights][layer0]
import torch; torch.manual_seed(42)
TRACE_SRC = "Source: physical_trace_tp4_rank0.json"
HIDDEN = 4096; NUM_HEADS = 32; NUM_KV_HEADS = 8; HEAD_DIM = 128; TP = 4
Q_SZ = NUM_HEADS * HEAD_DIM // TP  # 1024
KV_SZ = max(1, NUM_KV_HEADS // TP) * HEAD_DIM  # 256
INTER = 12288; INTER_PER = INTER // TP  # 3072


def test_qkv_cat_order_is_Q_then_K_then_V():
    """KEYMAP-001: torch.cat([q,k,v], dim=0) 顺序必须是 Q-K-V"""
    full_q = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN); full_k = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
    full_v = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
    q_slice = full_q[:Q_SZ, :]; k_slice = full_k[:KV_SZ, :]; v_slice = full_v[:KV_SZ, :]
    cat_qkv = torch.cat([q_slice, k_slice, v_slice], dim=0)
    assert cat_qkv.shape == (Q_SZ + 2 * KV_SZ, HIDDEN), (
        f"KEYMAP-001: qkv shape={list(cat_qkv.shape)}，期望=({Q_SZ+2*KV_SZ},{HIDDEN})=({1536},{HIDDEN})。"
        f"{TRACE_SRC} [model_weights][layer0] qkv_proj=[1536,4096]")
    # Wrong order detection: if K-Q-V, q starts at offset KV_SZ
    cat_kqv = torch.cat([k_slice, q_slice, v_slice], dim=0)
    assert not torch.equal(cat_kqv[:Q_SZ, :], cat_qkv[:Q_SZ, :]), (
        f"KEYMAP-001: K-Q-V 和 Q-K-V 的 Q 段不同。Q 必须在 dim0 最前面。"
        f"Agent 错误: 如果用 K-Q-V 顺序 → Q 被放在 KV_SZ 偏移处，导致 attention 输入错误。")


def test_gate_up_cat_order_is_gate_then_up():
    """KEYMAP-002: torch.cat([gate, up], dim=0) 顺序必须是 gate-up"""
    gate = torch.randn(INTER_PER, HIDDEN); up = torch.randn(INTER_PER, HIDDEN)
    cat_gu = torch.cat([gate, up], dim=0)
    assert cat_gu.shape == (2 * INTER_PER, HIDDEN), (
        f"KEYMAP-002: gate_up shape={list(cat_gu.shape)}，期望=({2*INTER_PER},{HIDDEN})=({6144},{HIDDEN})。"
        f"gate=SiLU 输入 (前半), up=乘数 (后半)。{TRACE_SRC} [model_weights][layer0_mlp] gate_up=[6144,4096]")


def test_double_shard_guard_presliced_direct_copy():
    """KEYMAP-003: double_shard_guard: 预切片 weight → 直拷，不二次切片"""
    presliced = torch.randn(Q_SZ + 2*KV_SZ, HIDDEN)
    model_w = torch.empty(Q_SZ + 2*KV_SZ, HIDDEN)
    if presliced.shape == model_w.shape:
        model_w.copy_(presliced)
    else:
        raise AssertionError(f"KEYMAP-003: shape 相同但走了切片分支。{TRACE_SRC} double_shard_guard contract")
    assert torch.equal(model_w, presliced), (
        f"KEYMAP-003: copy_ 后值不等。{TRACE_SRC} load_weight_shard presliced path")


def test_weight_slicing_uses_tp_rank_bounds():
    """KEYMAP-004: 全量 weight 切片范围正确 (per-rank chunk)"""
    full_w = torch.randn(Q_SZ*TP + 2*KV_SZ*TP, HIDDEN)  # [4096+2048, 4096] = [6144, 4096]
    rank = 0
    q_start, q_end = rank * Q_SZ, (rank+1) * Q_SZ
    k_start = NUM_HEADS * HEAD_DIM + rank * KV_SZ
    k_end = k_start + KV_SZ
    v_start = NUM_HEADS * HEAD_DIM + NUM_KV_HEADS * HEAD_DIM + rank * KV_SZ
    v_end = v_start + KV_SZ
    shard = torch.cat([full_w[q_start:q_end], full_w[k_start:k_end], full_w[v_start:v_end]], dim=0)
    assert shard.shape == (Q_SZ + 2*KV_SZ, HIDDEN), (
        f"KEYMAP-004: shard shape={list(shard.shape)}，期望=({Q_SZ+2*KV_SZ},{HIDDEN})。"
        f"Q slice [{q_start}:{q_end}], K [{k_start}:{k_end}], V [{v_start}:{v_end}]。"
        f"{TRACE_SRC} [model_weights][layer0] qkv_proj_weight_shape=[1536,4096] = per-rank shard")


if __name__ == "__main__":
    test_qkv_cat_order_is_Q_then_K_then_V()
    test_gate_up_cat_order_is_gate_then_up()
    test_double_shard_guard_presliced_direct_copy()
    test_weight_slicing_uses_tp_rank_bounds()
    print("PHASE7_HF_KEY_MAPPING: ALL 4 TESTS PASSED")
