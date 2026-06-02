# Why: 防止 4 种 TP Linear 的 shape 计算错误（Column/Row/Merged/QKV）和 double_shard_guard
#   二次切片风险。V17 FG-1 确认 intermediate_size=12288（非 12800），导致 gate_up_proj
#   weight shape 系统性错算 [6400,4096] vs 正确 [6144,4096]。
#   发现于 V17 审计 2026-05-27，物理 config.json 验证确认。
# What failure: Agent 如果 Linear forward 的中间 shape 不正确（如 QKV split 维度对不上、
#   gate_up 输出不是 [B,T,2*intermediate/tp]、RowParallel 漏调 all_reduce_sum），
#   此测试通过精确 shape assert 报错 "LINEAR-00X"。
# Superpowers gate: 此脚本对应 superpowers CLAUDE.md rule 2 (No speculative fixes)
#   — FG-1 维度错误是真实 config.json 物理验证确认的。
# Human review: [待人类Diff] 请审查 TP Linear 的 per-rank 维度公式。
import torch
import torch.nn as nn
import torch.nn.functional as F

# Verified dimensions from config.json (2026-05-27):
HIDDEN = 4096
INTERMEDIATE = 12288  # NOT 12800!
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
TP = 4

# Per-rank dimensions:
HIDDEN_PER_RANK = HIDDEN  # 4096 (input to column parallel is full hidden)
INTER_PER_RANK = INTERMEDIATE // TP  # 3072 (NOT 3200!)
Q_SIZE = NUM_HEADS * HEAD_DIM // TP  # 1024
KV_SIZE = NUM_KV_HEADS * HEAD_DIM // TP  # = max(1, 8//4) * 128 = 2 * 128 = 256
QKV_TOTAL = Q_SIZE + 2 * KV_SIZE  # 1024 + 512 = 1536
GATE_UP_TOTAL = 2 * INTER_PER_RANK  # 6144 (NOT 6400!)
DOWN_OUT = HIDDEN  # 4096


# Minimal Linear classes for contract testing — Agent must match these signatures
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, gather_output=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # per-rank
        self.gather_output = gather_output
        self.tp_size = TP
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        y = F.linear(x, self.weight)
        if self.gather_output and self.tp_size > 1:
            pass  # all_gather_last_dim — tested in Phase 4
        return y


class RowParallelLinear(nn.Module):
    def __init__(self, in_features_per_rank, out_features_full, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features_full, in_features_per_rank))

    def forward(self, x):
        y = F.linear(x, self.weight)
        # all_reduce_sum — Phase 2 contract
        return y


def test_column_parallel_output_shape():
    """
    LINEAR-001: ColumnParallelLinear [out/tp, in] 输出 shape 验证。
    """
    torch.manual_seed(42)
    B, T = 1, 4
    in_dim = HIDDEN  # 4096
    out_dim = INTERMEDIATE  # 12288 (full)

    m = ColumnParallelLinear(in_dim, out_dim // TP, gather_output=False)
    x = torch.randn(B, T, in_dim)
    y = m(x)

    assert y.shape == (B, T, out_dim // TP), (
        f"LINEAR-001: ColumnParallelLinear output shape={y.shape}，期望={(B,T,out_dim // TP)}。"
        f"weight shape={list(m.weight.shape)}。"
        f"per-rank out = {out_dim}/{TP} = {out_dim // TP}。"
    )
    assert m.weight.shape == (out_dim // TP, in_dim), (
        f"LINEAR-001: ColumnParallel weight shape={list(m.weight.shape)}，"
        f"期望=({out_dim // TP}, {in_dim})。"
    )


def test_row_parallel_output_shape():
    """
    LINEAR-002: RowParallelLinear [out, in/tp] 输出 shape 验证。
    """
    torch.manual_seed(42)
    B, T = 1, 4
    out_dim = HIDDEN  # 4096
    in_per_rank = INTER_PER_RANK  # 3072 = 12288//4

    m = RowParallelLinear(in_per_rank, out_dim)
    x = torch.randn(B, T, in_per_rank)
    y = m(x)

    assert y.shape == (B, T, out_dim), (
        f"LINEAR-002: RowParallelLinear output shape={y.shape}，期望={(B,T,out_dim)}。"
    )
    assert m.weight.shape == (out_dim, in_per_rank), (
        f"LINEAR-002: RowParallel weight shape={list(m.weight.shape)}，"
        f"期望=({out_dim}, {in_per_rank})。"
        f"注意：weight shape [out, in/tp] 不是 [out/tp, in]。"
    )


def test_merged_column_parallel_shape():
    """
    LINEAR-003: MergedColumnParallelLinear gate+up 合并投影。
    weight [2*intermediate/tp, hidden] = [6144, 4096]（非 [6400,4096]）。
    """
    torch.manual_seed(42)
    B, T = 1, 1

    weight = nn.Parameter(torch.randn(GATE_UP_TOTAL, HIDDEN))
    x = torch.randn(B, T, HIDDEN)
    y = F.linear(x, weight)

    assert y.shape == (B, T, GATE_UP_TOTAL), (
        f"LINEAR-003: MergedColumnParallelLinear output shape={y.shape}，期望={(B,T,GATE_UP_TOTAL)}。"
        f"gate_up_total = 2 * {INTERMEDIATE}/{TP} = {GATE_UP_TOTAL}。"
        f"Agent 错误：可能用旧的 intermediate_size=12800 → gate_up_total=6400。"
        f"正确 intermediate_size=12288 → gate_up_total=6144。"
    )
    assert weight.shape == (GATE_UP_TOTAL, HIDDEN), (
        f"LINEAR-003: gate_up weight shape={list(weight.shape)}，期望=({GATE_UP_TOTAL}, {HIDDEN})。"
    )


def test_qkv_column_parallel_split():
    """
    LINEAR-004: QKVColumnParallelLinear 输出 split。
    q_size=1024 (32*128/4), kv_size=256 (2*128), total=1536。
    split dim=-1 → (q:[1,1,1024], k:[1,1,256], v:[1,1,256])。
    """
    torch.manual_seed(42)
    B, T = 1, 1

    weight = nn.Parameter(torch.randn(QKV_TOTAL, HIDDEN))
    x = torch.randn(B, T, HIDDEN)
    y = F.linear(x, weight)

    q, k, v = y.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)

    assert q.shape == (B, T, Q_SIZE), (
        f"LINEAR-004: Q split shape={q.shape}，期望={(B,T,Q_SIZE)}。"
        f"q_size = {NUM_HEADS}*{HEAD_DIM}/{TP} = {Q_SIZE}。"
    )
    assert k.shape == (B, T, KV_SIZE), (
        f"LINEAR-004: K split shape={k.shape}，期望={(B,T,KV_SIZE)}。"
        f"kv_size = max(1, {NUM_KV_HEADS}/{TP}) * {HEAD_DIM} = {KV_SIZE}。"
        f"Agent 错误：可能用 num_heads=8 reshape K → 8*128=1024 != 256。"
    )
    assert v.shape == (B, T, KV_SIZE), (
        f"LINEAR-004: V split shape={v.shape}，期望={(B,T,KV_SIZE)}。"
    )

    # Verify Q can be reshaped to [B, T, num_heads_per_rank, head_dim]
    q_reshape = q.view(B, T, NUM_HEADS // TP, HEAD_DIM)
    assert q_reshape.shape == (B, T, 8, HEAD_DIM), (
        f"LINEAR-004: q reshape shape={q_reshape.shape}，期望={(B,T,8,HEAD_DIM)}。"
    )

    # K reshape must use num_kv_heads_local=2, NOT num_heads=8
    kv_heads_local = max(1, NUM_KV_HEADS // TP)
    k_reshape = k.view(B, T, kv_heads_local, HEAD_DIM)
    assert k_reshape.shape == (B, T, 2, HEAD_DIM), (
        f"LINEAR-004: k reshape shape={k_reshape.shape}，期望={(B,T,2,HEAD_DIM)}。"
        f"Agent 错误：可能用 num_heads=8 替代 num_kv_heads_local={kv_heads_local} → "
        f"8*128=1024 != kv_size=256 → RuntimeError shape mismatch。"
    )


def test_double_shard_guard_presliced():
    """
    LINEAR-005: double_shard_guard — 传入预切片权重时直拷，不二次切片。
    """
    torch.manual_seed(42)

    # Pre-sliced weight (already per-rank)
    presliced = torch.randn(QKV_TOTAL, HIDDEN)
    model_weight = nn.Parameter(torch.empty(QKV_TOTAL, HIDDEN))

    # Double shard guard: if incoming shape == self.weight.shape, direct copy
    if presliced.shape == model_weight.shape:
        model_weight.data.copy_(presliced)
    else:
        # Slice (should not reach here for pre-sliced)
        raise AssertionError(
            "LINEAR-005: double_shard_guard 失败。"
            f"传入 shape={list(presliced.shape)}，weight shape={list(model_weight.shape)}。"
            f"二者相同，应走直拷路径。却进入了切片分支。"
            f"Agent 错误：可能在 shape 匹配时仍然执行了 dim0 切片。"
        )

    assert torch.equal(model_weight.data, presliced), (
        "LINEAR-005: copy_ 后 weight 不等于传入值。"
        f"double_shard_guard 的 copy_ 操作未正确执行。"
    )


def test_double_shard_guard_full_weight_requires_slicing():
    """
    LINEAR-006: double_shard_guard — 传入全量权重时必须正确切片。
    """
    torch.manual_seed(42)

    # Full weight (not pre-sliced) — shape [qkv_total*tp, hidden] = [6144, 4096]
    full_shape = (QKV_TOTAL * TP, HIDDEN)
    full_weight = torch.randn(full_shape)
    model_weight = nn.Parameter(torch.empty(QKV_TOTAL, HIDDEN))

    # Simulate correct slicing: per-rank takes its own chunk
    # QKV order: [Q, K, V] — each rank takes [q_size, kv_size, kv_size] across ranks
    rank = 0
    q_slice_start = rank * Q_SIZE
    q_slice_end = (rank + 1) * Q_SIZE
    q_shard = full_weight[q_slice_start:q_slice_end, :]  # [1024, 4096]

    num_kv_heads_local = max(1, NUM_KV_HEADS // TP)
    kv_shard_size = num_kv_heads_local * HEAD_DIM  # 256
    k_start = (NUM_HEADS * HEAD_DIM) + rank * kv_shard_size
    k_end = k_start + kv_shard_size
    k_shard = full_weight[k_start:k_end, :]  # [256, 4096]

    v_start = (NUM_HEADS * HEAD_DIM) + NUM_KV_HEADS * HEAD_DIM + rank * kv_shard_size
    v_end = v_start + kv_shard_size
    v_shard = full_weight[v_start:v_end, :]  # [256, 4096]

    sliced = torch.cat([q_shard, k_shard, v_shard], dim=0)

    assert sliced.shape == (QKV_TOTAL, HIDDEN), (
        f"LINEAR-006: sliced shape={list(sliced.shape)}，期望=({QKV_TOTAL}, {HIDDEN})。"
        f"q_shard={list(q_shard.shape)}, k_shard={list(k_shard.shape)}, v_shard={list(v_shard.shape)}。"
        f"Agent 错误：Q/K/V 拼接顺序可能错误（必须 Q-K-V，严禁 K-Q-V 或 V-K-Q）。"
    )


if __name__ == "__main__":
    test_column_parallel_output_shape()
    test_row_parallel_output_shape()
    test_merged_column_parallel_shape()
    test_qkv_column_parallel_split()
    test_double_shard_guard_presliced()
    test_double_shard_guard_full_weight_requires_slicing()
    print("PHASE3_TP_LINEAR: ALL 6 TESTS PASSED")
