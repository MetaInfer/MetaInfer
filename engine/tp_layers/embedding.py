# engine/tp_layers/embedding.py
# Phase 4: TP Embedding — VocabParallelEmbedding + ParallelLMHead.
#
# Blueprint contract:
#   framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_embedding_and_lm_head
#
# Ref:
#   inference_blueprint.json > vocab_parallel_embedding.forward_pseudocode
#   inference_blueprint.json > parallel_lm_head.forward_pseudocode

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.tp_layers.distributed import (
    all_reduce_sum,
    all_gather_last_dim,
    get_tp_size,
    get_tp_rank,
)


# ================================================================
# VocabParallelEmbedding
# ================================================================


class VocabParallelEmbedding(nn.Module):
    """
    Vocabulary-parallel embedding layer.

    Each TP rank holds a shard of the full vocabulary: [local_vocab_size, embedding_dim].
    At forward time, input tokens outside this rank's vocab range are masked to zero,
    and all_reduce_sum reconstructs the full embedding.

    Blueprint pseudocode:
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill((~mask).unsqueeze(-1), 0)
        return all_reduce_sum(out)

    Args:
        num_embeddings: total vocabulary size (full, not per-rank)
        embedding_dim:  hidden size / embedding dimension
        tp_size:        tensor-parallel world size (default: from distributed runtime)
        padding_idx:    padding index (unused in TP path; accepted for API compatibility)
    """

    def __init__(self, num_embeddings, embedding_dim, tp_size=None, padding_idx=None):
        super().__init__()
        self.tp_size = tp_size if tp_size is not None else get_tp_size()
        self.tp_rank = get_tp_rank()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Vocab partition: each rank holds 1/tp_size of the vocabulary
        self.local_vocab_size = num_embeddings // self.tp_size
        self.vocab_start = self.tp_rank * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size

        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] int64 — token ids in range [0, num_embeddings)

        Returns:
            [B, T, embedding_dim] — full embedding after all_reduce_sum across TP ranks
        """
        # 1. Mask: which tokens belong to this rank's vocab partition?
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)

        # 2. Map global token ids → local indices; mask out-of-range to 0
        local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)

        # 3. Embedding lookup
        out = F.embedding(local_ids, self.weight)  # [B, T, embedding_dim]

        # 4. Zero out contributions from tokens not in this rank's partition
        out = out.masked_fill((~mask).unsqueeze(-1), 0)

        # 5. All-reduce sum across TP ranks to reconstruct full embedding
        return all_reduce_sum(out)

    def load_weight_shard(self, shard: torch.Tensor) -> None:
        """
        Load a weight shard with double_shard_guard.

        If shard.shape == self.weight.shape: direct copy_ (already pre-sliced).
        Otherwise: slice from full weight by tp_rank along dim 0.
        """
        if shard.shape == self.weight.shape:
            self.weight.data.copy_(shard)
        else:
            # Full weight shape: [num_embeddings, embedding_dim]
            # Slice to this rank's vocab partition along dim 0
            start = self.tp_rank * self.local_vocab_size
            end = start + self.local_vocab_size
            self.weight.data.copy_(shard[start:end, :])


# ================================================================
# ParallelLMHead
# ================================================================


class ParallelLMHead(nn.Module):
    """
    Parallel LM head (output projection layer).

    Weight is partitioned the same way as VocabParallelEmbedding:
    [local_vocab_size, embedding_dim].

    Forward computes local logits then all_gather_last_dim to reconstruct
    the full vocabulary logits.

    Blueprint pseudocode:
        local_logits = F.linear(hidden_states, self.weight)  # [B, T, local_vocab_size]
        logits = all_gather_last_dim(local_logits)            # [B, T, num_embeddings]
        return logits

    Args:
        num_embeddings: total vocabulary size (full, not per-rank)
        embedding_dim:  hidden size / embedding dimension
        tp_size:        tensor-parallel world size (default: from distributed runtime)
        bias:           if True, add bias (default: False)
    """

    def __init__(self, num_embeddings, embedding_dim, tp_size=None, bias=False):
        super().__init__()
        self.tp_size = tp_size if tp_size is not None else get_tp_size()
        self.tp_rank = get_tp_rank()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.local_vocab_size = num_embeddings // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_vocab_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, embedding_dim]

        Returns:
            [B, T, num_embeddings] — full vocabulary logits after all-gather
        """
        # 1. Local linear projection → per-rank logits
        local_logits = F.linear(hidden_states, self.weight, self.bias)  # [B, T, local_vocab_size]

        # 2. All-gather along last dim to reconstruct full vocabulary logits
        logits = all_gather_last_dim(local_logits)  # [B, T, num_embeddings]
        return logits

    def load_weight_shard(self, shard: torch.Tensor) -> None:
        """
        Load a weight shard with double_shard_guard.

        If shard.shape == self.weight.shape: direct copy_ (already pre-sliced).
        Otherwise: slice from full weight by tp_rank along dim 0.
        """
        if shard.shape == self.weight.shape:
            self.weight.data.copy_(shard)
        else:
            # Full weight shape: [num_embeddings, embedding_dim]
            # Slice to this rank's vocab partition along dim 0
            start = self.tp_rank * self.local_vocab_size
            end = start + self.local_vocab_size
            self.weight.data.copy_(shard[start:end, :])
