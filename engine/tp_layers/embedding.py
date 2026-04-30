from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.tp_layers.distributed import (
    all_gather_last_dim,
    all_reduce_sum,
    ensure_divisible,
    get_tp_rank,
    get_tp_size,
)


class VocabParallelEmbedding(nn.Module):
    """Shard embedding on vocab dimension, then all-reduce masked outputs."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = get_tp_size()
        ensure_divisible(num_embeddings, self.tp_size, name="num_embeddings")
        self.local_vocab_size = num_embeddings // self.tp_size
        self.vocab_start = get_tp_rank() * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill((~mask).unsqueeze(-1), 0)
        return all_reduce_sum(out)

    def load_weight_shard(self, full_weight: torch.Tensor) -> None:
        if int(full_weight.shape[0]) == self.local_vocab_size:
            shard = full_weight
        else:
            shard = full_weight[self.vocab_start : self.vocab_end, :]
        self.weight.data.copy_(shard.to(device=self.weight.device, dtype=self.weight.dtype))


class ParallelLMHead(nn.Module):
    """Local vocab logits then optional all-gather to full vocab."""

    def __init__(self, hidden_size: int, vocab_size: int, gather_output: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.tp_size = get_tp_size()
        ensure_divisible(vocab_size, self.tp_size, name="vocab_size")
        self.local_vocab_size = vocab_size // self.tp_size
        self.vocab_start = get_tp_rank() * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(hidden_states, self.weight)
        if not self.gather_output:
            return local_logits
        logits = all_gather_last_dim(local_logits)
        return logits[..., : self.vocab_size]

    def load_weight_shard(self, full_weight: torch.Tensor) -> None:
        if int(full_weight.shape[0]) == self.local_vocab_size:
            shard = full_weight
        else:
            shard = full_weight[self.vocab_start : self.vocab_end, :]
        self.weight.data.copy_(shard.to(device=self.weight.device, dtype=self.weight.dtype))
