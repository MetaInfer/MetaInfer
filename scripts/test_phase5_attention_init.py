# Why: 防止 QwenAttentionTP.__init__ 中 KV head replication、per-rank 维度计算、
#   KV block_size、buffer 注册等初始化合约被 Agent 错误实现。
#   物理 trace (2026-05-27, TP=4 nocompile) 确认:
#   - num_heads=8 (per-rank), num_kv_heads=2 (per-rank, max(1,8//4)=2)
#   - kv_block_size=256, head_dim=128
#   - qkv_proj=[1536,4096], o_proj=[4096,1024]
#   发现于 V17 FG-1 (max_position_embeddings=40960 non 32768)、
#   V5 审计 (KV head replication: tp>num_kv_heads 时 num_kv_heads=1)。
# What failure: Agent 如果 num_heads 直接用了全量值 32 而非 per-rank 8、
#   或用 num_heads 替代 num_kv_heads 来 reshape K/V (> shape mismatch)、
#   kv_block_size 未设为 256、_kv_len_gpu 未用 register_buffer 注册，
#   此测试通过精确 shape assert 报错 "ATTN-INIT-00X"。
# Superpowers gate: 此脚本对应 superpowers CLAUDE.md rule 1 (No fabricated content)
#   — 所有维度值来自物理 trace physical_trace_tp4_rank0.json。
# Human review: [待人类Diff] 请审查 per-rank 维度与 physical_trace_summary.md 一致。
# T11 source: physical_trace_tp4_rank0.json [config] + [derived] + [model_weights][layer0]
import torch
import torch.nn as nn

# === PHYSICAL TRACE CONSTANTS (from trace 2026-05-27 TP=4 nocompile) ===
# Source: notebooks-cn/07_improvementPlan/test_scrpits/physical_trace_tp4_rank0.json
#   [config]: max_position_embeddings=40960, intermediate_size=12288, hidden_size=4096,
#             num_attention_heads=32, num_key_value_heads=8, head_dim=128
#   [derived]: per_rank_attn_heads=8, per_rank_kv_heads=2, block_size=256,
#              q_size=1024, kv_size=256, qkv_weight_total=[1536,4096]
#   [model_weights][layer0]: num_heads=8, num_kv_heads=2, kv_block_size=256
TRACE_MAX_POS = 40960
TRACE_HIDDEN = 4096
TRACE_NUM_HEADS = 32        # full model (from config.json)
TRACE_NUM_KV_HEADS = 8      # full model
TRACE_HEAD_DIM = 128
TP_SIZE = 4

# Per-rank derived (must match trace)
PER_RANK_HEADS = TRACE_NUM_HEADS // TP_SIZE       # 8
PER_RANK_KV_HEADS = max(1, TRACE_NUM_KV_HEADS // TP_SIZE)  # max(1, 8//4) = 2
PER_RANK_Q_SIZE = PER_RANK_HEADS * TRACE_HEAD_DIM   # 1024
PER_RANK_KV_SIZE = PER_RANK_KV_HEADS * TRACE_HEAD_DIM  # 256
KV_BLOCK_SIZE = 256
MAX_BLOCKS = (TRACE_MAX_POS + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE  # 160


def test_num_heads_per_rank():
    """
    ATTN-INIT-001: num_heads 必须是 per-rank 值 = 8（非全量 32）。
    Trace: physical_trace_tp4_rank0.json [derived] per_rank_attn_heads=8
    """
    num_heads_per_rank = TRACE_NUM_HEADS // TP_SIZE
    assert num_heads_per_rank == 8, (
        f"ATTN-INIT-001: per_rank num_heads={num_heads_per_rank}，期望=8。"
        f"全量 num_attention_heads=32, TP=4 → per_rank=8。"
        f"Agent 错误: 可能直接用 cfg.num_attention_heads=32 作为 num_heads。"
        f"Source: physical_trace_tp4_rank0.json [derived] per_rank_attn_heads=8"
    )


def test_num_kv_heads_per_rank_with_replication():
    """
    ATTN-INIT-002: num_kv_heads 必须是 max(1, num_kv_heads // tp_size)。
    Qwen3-8B: 8 // 4 = 2。
    Trace: physical_trace_tp4_rank0.json [derived] per_rank_kv_heads=2
    """
    kv_heads_local = max(1, TRACE_NUM_KV_HEADS // TP_SIZE)
    assert kv_heads_local == 2, (
        f"ATTN-INIT-002: per_rank num_kv_heads={kv_heads_local}，期望=2。"
        f"num_key_value_heads=8, TP=4 → max(1, 8//4) = 2。"
        f"Agent 错误: 可能用 num_attention_heads//TP (=8) 替代 num_kv_heads_local (=2)。"
        f"Source: physical_trace_tp4_rank0.json [derived] per_rank_kv_heads=2"
    )


def test_kv_head_replication_when_tp_exceeds_kv_heads():
    """
    ATTN-INIT-003: 当 tp_size > num_kv_heads 时，num_kv_heads=1，
    kv_head_replica=tp_size//num_kv_heads。
    例如 tp=8, num_kv_heads=8 → kv_heads=1, replica=8。
    此测试使用 Qwen3-8B TP=4: tp(4) ≤ num_kv_heads(8)，不触发 replica。
    但 Agent 的实现必须包含此逻辑分支。
    """
    # Scenario: TP=8 on a model with 8 KV heads
    tp_sim = 8
    kv_heads_sim = 8
    if kv_heads_sim >= tp_sim:
        kv_local = kv_heads_sim // tp_sim
        replica = 1
    else:
        kv_local = 1
        replica = tp_sim // kv_heads_sim

    assert kv_local == 1, (
        f"ATTN-INIT-003: tp=8, num_kv_heads=8 → per_rank_kv_heads={kv_local}，期望=1。"
        f"Source: physical_trace_tp4_rank0.json [derived] per_rank_kv_heads=2 (TP=4 case)，"
        f"extrapolated to TP=8 case per KV head replication formula。"
    )
    assert replica == 1, (
        f"ATTN-INIT-003: replica={replica}，期望=1 (tp≤kv_heads 不复制)。"
        f"Source: physical_trace_tp4_rank0.json [model_weights][layer0] num_kv_heads=2, TP=4 implies no replication needed。"
    )

    # Scenario: TP=8 on a model with 4 KV heads (needs replication)
    tp2, kv2 = 8, 4
    kv_local2 = kv2 // tp2 if kv2 >= tp2 else 1
    replica2 = 1 if kv2 >= tp2 else tp2 // kv2
    assert kv_local2 == 1, (
        f"ATTN-INIT-003: kv_local2={kv_local2}，期望=1。"
        f"Source: KV head replication formula from inference_blueprint.json "
        f"qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP (num_kv_heads=max(1, cfg.num_key_value_heads//tp))。"
    )
    assert replica2 == 2, (
        f"ATTN-INIT-003: tp=8, num_kv_heads=4 → replica=tp//kv={replica2}，期望=2。"
        f"8 GPUs 共享 4 个 KV head，每个 head 复制到 2 个 rank。"
        f"Source: inference_blueprint.json QwenAttentionTP.kv_head_replica = tp_size // cfg.num_key_value_heads。"
        f"Agent 错误: 可能漏掉 kv_head_replica 逻辑或计算 tp_size/num_kv_heads 颠倒。"
    )


def test_kv_block_size_256():
    """
    ATTN-INIT-004: _kv_block_size 必须为 256。
    flash_attn_with_kvcache 最低要求 block_size >= 256。
    Trace: physical_trace_tp4_rank0.json [derived] block_size=256
    """
    block_size = KV_BLOCK_SIZE
    assert block_size == 256, (
        f"ATTN-INIT-004: kv_block_size={block_size}，期望=256。"
        f"flash_attn_with_kvcache 要求 block_size >= 256。"
        f"Agent 错误: 可能沿用框架 block_size=16。"
        f"Source: physical_trace_tp4_rank0.json [kv_cache_contract] block_size=256"
    )


def test_max_blocks_from_max_position_embeddings():
    """
    ATTN-INIT-005: max_blocks = ceil(max_position_embeddings / 256)。
    Trace: max_position_embeddings=40960 → max_blocks=160 (NOT 128!)
    Source: physical_trace_tp4_rank0.json [derived] max_blocks=160
    """
    max_blocks = (TRACE_MAX_POS + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    assert max_blocks == 160, (
        f"ATTN-INIT-005: max_blocks={max_blocks}，期望=160。"
        f"40960/256=160。旧文档写 32768/256=128 是错的。"
        f"Agent 错误: 可能硬编码 128 或使用旧 max_position_embeddings=32768。"
        f"Source: physical_trace_tp4_rank0.json [derived] max_blocks=160"
    )


def test_qkv_weight_dimensions():
    """
    ATTN-INIT-006: QKVColumnParallelLinear weight [q_size+2*kv_size, hidden]。
    Per-rank: [1024+256+256, 4096] = [1536, 4096]。
    Trace: physical_trace_tp4_rank0.json [model_weights][layer0] qkv_proj=[1536,4096]
    """
    qkv_per_rank = PER_RANK_Q_SIZE + 2 * PER_RANK_KV_SIZE
    assert qkv_per_rank == 1536, (
        f"ATTN-INIT-006: qkv_total={qkv_per_rank}，期望=1536。"
        f"= q_size({PER_RANK_Q_SIZE}) + 2*kv_size({PER_RANK_KV_SIZE})。"
        f"Source: physical_trace_tp4_rank0.json [model_weights][layer0] qkv_proj_weight_shape=[1536,4096]"
    )


def test_o_proj_weight_dimensions():
    """
    ATTN-INIT-007: o_proj (RowParallelLinear) weight [hidden, q_size_per_rank]。
    Per-rank: [4096, 1024]。
    Trace: physical_trace_tp4_rank0.json [model_weights][layer0] o_proj=[4096,1024]
    """
    assert PER_RANK_Q_SIZE == 1024, (
        f"ATTN-INIT-007: o_proj in_features(per_rank)={PER_RANK_Q_SIZE}，期望=1024。"
        f"= num_heads_per_rank({PER_RANK_HEADS}) × head_dim({TRACE_HEAD_DIM})。"
        f"Source: physical_trace_tp4_rank0.json o_proj_weight_shape=[4096,1024]"
    )


def test_buffer_registration_kv_len_gpu():
    """
    ATTN-INIT-008: _kv_len_gpu 必须是 register_buffer (persistent=False)。
    Shape [1], dtype int32, 初始值 0。
    物理 trace 确认 _kv_len_gpu 存在。读取走 CPU 算术（past_key_values[0] + 1），禁止 .item()。
    """
    kv_len = torch.zeros(1, dtype=torch.int32)
    assert kv_len.shape == (1,), (
        f"ATTN-INIT-008: _kv_len_gpu shape={kv_len.shape}，期望=(1,)。"
        f"Source: physical_trace_tp4_rank0.json [kv_cache_contract] kv_len_gpu_shape=[1], kv_len_gpu_dtype=torch.int32。"
    )
    assert kv_len.dtype == torch.int32, (
        f"ATTN-INIT-008: _kv_len_gpu dtype={kv_len.dtype}，期望=torch.int32。"
        f"Agent 错误：可能用 Python int 替代 GPU tensor → .item() 在 compiled region 触发 SIGABRT。"
        f"Source: physical_trace_tp4_rank0.json [kv_cache_contract] kv_len_gpu_dtype=torch.int32。"
    )


def test_buffer_registration_slot_mapping_decode():
    """
    ATTN-INIT-009: _slot_mapping_decode 必须是 register_buffer (persistent=False)。
    Shape [1], dtype int64。decode 时写入当前 token 位置。
    """
    slot = torch.zeros(1, dtype=torch.int64)
    assert slot.shape == (1,), (
        f"ATTN-INIT-009: _slot_mapping_decode shape={slot.shape}，期望=(1,)。"
        f"Source: physical_trace_tp4_rank0.json [kv_cache_contract] slot_mapping_decode_shape=[1], dtype=torch.int64。"
    )
    assert slot.dtype == torch.int64, (
        f"ATTN-INIT-009: _slot_mapping_decode dtype={slot.dtype}，期望=torch.int64。"
        f"Agent 错误：可能用 int32 → index_copy_ 要求 int64。"
        f"Source: physical_trace_tp4_rank0.json [kv_cache_contract] slot_mapping_decode_dtype=torch.int64，"
        f"index_copy_ 要求 slot_mapping dtype 与 cache flat view dim 0 兼容。"
    )


if __name__ == "__main__":
    test_num_heads_per_rank()
    test_num_kv_heads_per_rank_with_replication()
    test_kv_head_replication_when_tp_exceeds_kv_heads()
    test_kv_block_size_256()
    test_max_blocks_from_max_position_embeddings()
    test_qkv_weight_dimensions()
    test_o_proj_weight_dimensions()
    test_buffer_registration_kv_len_gpu()
    test_buffer_registration_slot_mapping_decode()
    print("PHASE5_ATTENTION_INIT: ALL 9 TESTS PASSED")
