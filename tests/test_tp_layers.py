from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.tp_layers.embedding import ParallelLMHead, VocabParallelEmbedding
from engine.tp_layers.linear import ColumnParallelLinear, RowParallelLinear


def test_column_parallel_linear_tp1_matches_full() -> None:
    x = torch.randn(2, 8)
    full_w = torch.randn(12, 8)
    m = ColumnParallelLinear(8, 12, bias=False, gather_output=False)
    m.load_weight_shard(full_w)
    y = m(x)
    ref = F.linear(x, full_w)
    assert y.shape == ref.shape
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-6)


def test_row_parallel_linear_tp1_matches_full() -> None:
    x = torch.randn(3, 8)
    full_w = torch.randn(5, 8)
    m = RowParallelLinear(8, 5, bias=False)
    m.load_weight_shard(full_w)
    y = m(x)
    ref = F.linear(x, full_w)
    assert y.shape == ref.shape
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-6)


def test_vocab_parallel_embedding_tp1_matches_full() -> None:
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    full_w = torch.randn(16, 6)
    emb = VocabParallelEmbedding(16, 6)
    emb.load_weight_shard(full_w)
    y = emb(ids)
    ref = F.embedding(ids, full_w)
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-6)


def test_parallel_lm_head_tp1_matches_full() -> None:
    hs = torch.randn(2, 4, 6)
    full_w = torch.randn(16, 6)
    head = ParallelLMHead(6, 16, gather_output=True)
    head.load_weight_shard(full_w)
    y = head(hs)
    ref = F.linear(hs, full_w)
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-6)
