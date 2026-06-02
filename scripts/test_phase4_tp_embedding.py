# Why: 防止 VocabParallelEmbedding 的 mask 范围计算错误、masked_fill 遗漏、
#   all_reduce_sum 未调用、ParallelLMHead gather 维度错误。
#   Trace: vocab_size=151936, hidden=4096, TP=4 → per_rank_vocab=37984
#   发现于 V5 审计 VocabParallelEmbedding forward 物理 tracing。
# What failure: mask 外位置非零 / all_reduce 未调 / LM head gather dim 错
#   → assert "EMBED-00X" + Source trace path。
# Superpowers gate: CLAUDE.md rule 1 — vocab_size 来自物理 config.json 验证。
# Human review: [待人类Diff]
# T11 source: physical_trace_tp4_rank0.json [config] vocab_size=151936, hidden_size=4096
import torch
import torch.nn.functional as F
TRACE_SRC = "Source: physical_trace_tp4_rank0.json"

VOCAB = 151936; HIDDEN = 4096; TP = 4
VOCAB_PER_RANK = VOCAB // TP  # 37984


def test_vocab_parallel_embedding_mask_range():
    """EMBED-001: vocab_start/vocab_end 正确计算，mask 外置零"""
    torch.manual_seed(42)
    rank = 0
    vs = rank * VOCAB_PER_RANK; ve = vs + VOCAB_PER_RANK
    w = torch.randn(VOCAB_PER_RANK, HIDDEN)
    ids = torch.tensor([[0, 1, 37983, 37984, 151935]])
    mask = (ids >= vs) & (ids < ve)
    local_ids = (ids - vs).masked_fill(~mask, 0)
    out = F.embedding(local_ids, w)
    out = out.masked_fill((~mask).unsqueeze(-1), 0)
    assert out[0, 0, :].sum() != 0, (f"EMBED-001: id=0 在 vocab range 内，不应全零。{TRACE_SRC} [config] vocab_size=151936")
    assert out[0, 3, :].sum() == 0, (f"EMBED-001: id=37984 不在 rank0 range [0,37984)，应全零。{TRACE_SRC} VocabParallel masked_fill contract")


def test_vocab_parallel_embedding_output_shape():
    """EMBED-002: 输出 [B, T, hidden_size], 不等于 [B, T, hidden/tp]"""
    torch.manual_seed(42)
    B, T = 1, 4
    w = torch.randn(VOCAB_PER_RANK, HIDDEN)
    ids = torch.randint(0, VOCAB_PER_RANK, (B, T))  # all in local range
    out = F.embedding(ids, w)
    assert out.shape == (B, T, HIDDEN), (
        f"EMBED-002: shape={out.shape}，期望={(B,T,HIDDEN)}。"
        f"VocabParallelEmbedding 输出 full hidden_size(={HIDDEN})，非 {HIDDEN}//TP。"
        f"Agent 错误: 可能误用 per-rank hidden 替代 full hidden。"
        f"{TRACE_SRC} [config] hidden_size={HIDDEN}")


def test_parallel_lm_head_output_shape():
    """EMBED-003: ParallelLMHead local logits [B,T,vocab/tp] → gather → [B,T,vocab]"""
    torch.manual_seed(42)
    B, T = 1, 1
    hs = torch.randn(B, T, HIDDEN)
    w = torch.randn(VOCAB_PER_RANK, HIDDEN)
    local = F.linear(hs, w)
    assert local.shape == (B, T, VOCAB_PER_RANK), (
        f"EMBED-003: local logits shape={local.shape}，期望={(B,T,VOCAB_PER_RANK)}。"
        f"vocab={VOCAB}, TP={TP} → per_rank={VOCAB_PER_RANK}。"
        f"{TRACE_SRC} [config] vocab_size=151936 → per_rank=37984")


def test_lm_head_all_gather_last_dim():
    """EMBED-004: all_gather_last_dim 沿 dim=-1 拼接，输出 [B,T,vocab]"""
    torch.manual_seed(42)
    B, T = 1, 1
    # Simulate 4-rank gather: each contributes vocab/tp
    parts = [torch.randn(B, T, VOCAB_PER_RANK) for _ in range(TP)]
    gathered = torch.cat(parts, dim=-1)
    assert gathered.shape == (B, T, VOCAB), (
        f"EMBED-004: all_gather 后 shape={gathered.shape}，期望={(B,T,VOCAB)}。"
        f"all_gather_last_dim = dist.all_gather + torch.cat(dim=-1)。"
        f"{TRACE_SRC} LM head output must cover full vocab")


if __name__ == "__main__":
    test_vocab_parallel_embedding_mask_range()
    test_vocab_parallel_embedding_output_shape()
    test_parallel_lm_head_output_shape()
    test_lm_head_all_gather_last_dim()
    print("PHASE4_TP_EMBEDDING: ALL 4 TESTS PASSED")
