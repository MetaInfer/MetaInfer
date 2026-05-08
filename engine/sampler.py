"""Logits-based token sampling (greedy, top-p)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return argmax token id per row. logits: [batch, vocab]."""
    return logits.argmax(dim=-1)


def top_p_sample(
    logits: torch.Tensor,
    top_p: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling per row.
    logits: [batch, vocab], top_p in (0, 1].

    Optimization: first select top-k candidates to reduce sort cost on large vocab.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    vocab_size = logits.shape[-1]
    # For large vocab, first narrow to top-k candidates
    k = min(1024, vocab_size)
    if k < vocab_size:
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        # Do top-p on the reduced set
        sorted_logits, sorted_idx = torch.sort(top_k_logits, descending=True, dim=-1)
    else:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        top_k_indices = None

    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    # Remove tokens with cumulative mass strictly above top_p
    sorted_indices_to_remove = cumsum > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    filtered = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    probs_filtered = F.softmax(filtered, dim=-1)
    sampled_idx = torch.multinomial(probs_filtered, num_samples=1, generator=generator).squeeze(-1)

    # Map back: sampled_idx → position in top-k → original vocab index
    if top_k_indices is not None:
        local_idx = sorted_idx.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)
        return top_k_indices.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
    return sorted_idx.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)


def sample_next_tokens(
    logits: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 1.0,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Combined path: optional temperature scaling, then greedy (temp==0) or top-p or multinomial.
    logits: [batch, vocab]
    temperature: scalar or [batch] tensor broadcastable to logits rows
    """
    if temperature == 0 or (isinstance(temperature, (int, float)) and float(temperature) == 0.0):
        return greedy_sample(logits)

    scaled = logits.float()
    if isinstance(temperature, torch.Tensor):
        scaled = scaled / temperature.unsqueeze(-1).clamp(min=1e-8)
    else:
        t = float(temperature)
        if t != 1.0:
            scaled = scaled / t

    if top_p is not None and top_p < 1.0:
        return top_p_sample(scaled, top_p, generator=generator)

    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
