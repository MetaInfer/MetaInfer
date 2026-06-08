"""
Phase 4 — TP Embedding: VocabParallelEmbedding + ParallelLMHead.

Implements vocabulary-parallel embedding and LM head with per-rank vocab sharding.
All signatures must match inference_blueprint.json
  > tp_embedding_and_lm_head.vocab_parallel_embedding
  > tp_embedding_and_lm_head.parallel_lm_head
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.tp_layers.distributed import all_reduce_sum, all_gather_last_dim, _get_world_size


# ===========================================================================
# VocabParallelEmbedding — per-rank vocab shard, mask + all_reduce_sum
# ===========================================================================

class VocabParallelEmbedding(nn.Module):
    """Vocabulary-parallel embedding with masked per-rank vocab lookup.

    Each rank owns a contiguous slice of the vocabulary. Tokens outside this
    rank's slice are masked to zero, then all_reduce_sum combines contributions
    across all ranks to produce the correct full embedding.

    Weight shape: [local_vocab_size, embedding_dim]
    Input:        [B, T] int64
    Output:       [B, T, embedding_dim] after all_reduce_sum

    Constructor args:
        num_embeddings:  full vocabulary size (e.g. 151936 for Qwen3-8B)
        embedding_dim:   hidden dimension (e.g. 4096)
        tp_size:         tensor parallel size (auto-detect from env if None)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int | None = None,
    ):
        super().__init__()
        if tp_size is None:
            tp_size = _get_world_size()
        self.tp_size = tp_size
        self.tp_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Compute per-rank vocab range (handles non-divisible vocab_size)
        per_rank = num_embeddings // tp_size
        remainder = num_embeddings % tp_size
        self.local_vocab_size = per_rank + (1 if self.tp_rank < remainder else 0)
        self.vocab_start = self.tp_rank * per_rank + min(self.tp_rank, remainder)
        self.vocab_end = self.vocab_start + self.local_vocab_size

        self.weight = nn.Parameter(
            torch.empty(self.local_vocab_size, embedding_dim, dtype=torch.float32)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward: masked per-rank embedding lookup + all_reduce_sum.

        input_ids: [B, T] int64
        returns:   [B, T, embedding_dim]
        """
        # Mask: True for tokens within this rank's vocab range
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)

        # Map global ids to local ids; set out-of-range to 0
        local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)

        out = F.embedding(local_ids, self.weight)  # [B, T, embedding_dim]

        # Zero out embeddings for tokens not owned by this rank
        out = out.masked_fill((~mask).unsqueeze(-1), 0)

        # After all_reduce_sum: each token's embedding comes from exactly one rank
        return all_reduce_sum(out)

    def load_weight_shard(self, weight: torch.Tensor) -> None:
        """Load a weight shard with double-shard guard.

        If weight.shape == self.weight.shape, the weight is already pre-sliced →
        direct copy_. Otherwise, slice along dim=0 by tp_rank.

        Args:
            weight: full [num_embeddings, embedding_dim] or pre-sliced
                    [local_vocab_size, embedding_dim]
        """
        if weight.shape == self.weight.shape:
            # Already pre-sliced per-rank shard → direct copy
            self.weight.data.copy_(weight)
        else:
            # Full weight → slice along dim=0 by tp_rank
            shard = weight[self.vocab_start : self.vocab_end, :].contiguous()
            self.weight.data.copy_(shard)


# ===========================================================================
# ParallelLMHead — per-rank vocab logits, all_gather_last_dim (NOT all_reduce)
# ===========================================================================

class ParallelLMHead(nn.Module):
    """Language model head with vocabulary-parallel output.

    Each rank computes logits for its own vocab slice via F.linear,
    then all_gather_last_dim concatenates all slices into full-vocab logits.

    CRITICAL: Uses all_gather_last_dim, NOT all_reduce_sum.
    Each rank produces logits for a DIFFERENT portion of the vocabulary,
    so gathering (concatenation) is the correct operation — summing would
    incorrectly combine unrelated logit values.

    Weight shape: [local_vocab_size, embedding_dim]
    Input:        [B, T, embedding_dim]
    Output:       [B, T, num_embeddings] after all_gather_last_dim

    Constructor args:
        num_embeddings:  full vocabulary size (e.g. 151936 for Qwen3-8B)
        embedding_dim:   hidden dimension (e.g. 4096)
        bias:            whether to include bias (default False)
        tp_size:         tensor parallel size (auto-detect from env if None)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tp_size: int | None = None,
    ):
        super().__init__()
        if tp_size is None:
            tp_size = _get_world_size()
        self.tp_size = tp_size
        self.tp_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Compute per-rank vocab range (handles non-divisible vocab_size)
        per_rank = num_embeddings // tp_size
        remainder = num_embeddings % tp_size
        self.local_vocab_size = per_rank + (1 if self.tp_rank < remainder else 0)
        self.vocab_start = self.tp_rank * per_rank + min(self.tp_rank, remainder)
        self.vocab_end = self.vocab_start + self.local_vocab_size

        self.weight = nn.Parameter(
            torch.empty(self.local_vocab_size, embedding_dim, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_vocab_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward: per-rank logits + all_gather_last_dim.

        hidden_states: [B, T, embedding_dim]
        returns:       [B, T, num_embeddings] full-vocab logits
        """
        # Local logits for this rank's vocab slice
        local_logits = F.linear(hidden_states, self.weight, self.bias)
        # [B, T, local_vocab_size]

        # all_gather_last_dim (NOT all_reduce!) — each rank computes different
        # logits for its own vocab shard, so we concatenate along last dim.
        logits = all_gather_last_dim(local_logits)  # [B, T, num_embeddings]
        return logits

    def load_weight_shard(self, weight: torch.Tensor) -> None:
        """Load a weight shard with double-shard guard.

        If weight.shape == self.weight.shape, the weight is already pre-sliced →
        direct copy_. Otherwise, slice along dim=0 by tp_rank.

        Args:
            weight: full [num_embeddings, embedding_dim] or pre-sliced
                    [local_vocab_size, embedding_dim]
        """
        if weight.shape == self.weight.shape:
            # Already pre-sliced per-rank shard → direct copy
            self.weight.data.copy_(weight)
        else:
            # Full weight → slice along dim=0 by tp_rank
            shard = weight[self.vocab_start : self.vocab_end, :].contiguous()
            self.weight.data.copy_(shard)
