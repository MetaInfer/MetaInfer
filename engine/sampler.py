"""
Phase 8 — Sampler: logit → token conversion with TP-safe broadcast protocol.

TP sampling protocol (blueprint tp_sampling_protocol):
  - Only rank 0 executes sampling (greedy or multinomial).
  - Non-rank0 initializes tokens to placeholder [0]*B.
  - dist.broadcast(t, src=0) propagates rank0 result to all ranks.
  - Rationale: independent sampling on each rank → different CUDA random seeds →
    different tokens → KV cache divergence → NCCL crash.
"""

import torch
import torch.nn.functional as F


def sample_next_tokens(logits, temperature=0.0, top_p=1.0, top_k=0):
    """Sample next tokens from logits tensor.

    Args:
        logits: [B, V] float tensor
        temperature: 0.0 = greedy (argmax), >0 = multinomial with temp
        top_p: nucleus sampling threshold (1.0 = disabled)
        top_k: top-k sampling (0 = disabled)

    Returns:
        list[int] of length B
    """
    B, V = logits.shape

    # Greedy path
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1).tolist()

    # Temperature scaling (cast to float32 for numerical stability)
    logits_float = logits.float() / temperature

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits_float, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the mask right to always keep the first token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter sorted tensors to original ordering
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        logits_float = logits_float.masked_fill(indices_to_remove, float('-inf'))

    # Top-k sampling
    if top_k > 0:
        top_k = min(top_k, V)
        topk_values, _ = torch.topk(logits_float, top_k, dim=-1)
        kth_values = topk_values[:, -1].unsqueeze(-1)
        logits_float = logits_float.masked_fill(logits_float < kth_values, float('-inf'))

    probs = F.softmax(logits_float, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()


def tp_sample(logits, temperature=0.0, top_p=1.0):
    """TP-safe sampling: rank 0 samples, broadcasts to all ranks.

    Must be used in all TP multi-GPU paths. Direct calls to sample_next_tokens
    on non-rank0 would produce different tokens → KV cache desync → NCCL crash.

    Args:
        logits: [B, V] float tensor (same on all ranks after all_gather of lm_head)
        temperature: sampling temperature
        top_p: nucleus sampling threshold

    Returns:
        list[int] of length B (identical across all ranks)
    """
    import torch.distributed as dist

    if dist.is_initialized() and dist.get_world_size() > 1:
        rank = dist.get_rank()
        B = logits.shape[0]

        # Only rank 0 samples
        if rank == 0:
            tokens = sample_next_tokens(logits, temperature, top_p)
        else:
            tokens = [0] * B

        # Broadcast each token from rank 0 to all ranks
        for i, t in enumerate(tokens):
            tt = torch.tensor([t], dtype=torch.long, device=logits.device)
            dist.broadcast(tt, src=0)
            tokens[i] = tt.item()

        return tokens
    else:
        # Single GPU or no distributed — safe to sample directly
        return sample_next_tokens(logits, temperature, top_p)
