"""
Phase 8 — Sampler: token sampling with TP broadcast protocol.

TP protocol (hard rule): ONLY rank 0 samples; dist.broadcast to all ranks.
Single-GPU (TP=1): direct greedy argmax or top_p sampling.

Greedy shortcut: temperature=0.0 → argmax (skip softmax/filtering).

All signatures must match inference_blueprint.json
  > components[4] Sampler
  > tp_sampling_protocol
"""

import torch
import torch.distributed as dist


class Sampler:
    """Token sampler with TP broadcast protocol.

    TP multi-GPU: only rank 0 executes sampling logic; result is broadcast
    to all ranks via dist.broadcast(src=0). This ensures KV cache consistency
    across ranks.

    Single-GPU: direct greedy or top_p sampling.
    """

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Sample next tokens from logits.

        Args:
            logits: [B, vocab_size] or [B, seq_len, vocab_size] — raw logits.
                    For decode, logits[:, -1, :] is used (last position).
            temperature: 0.0 = greedy (argmax), > 0 = scaled softmax.
            top_p: Nucleus sampling threshold (1.0 = disabled).

        Returns:
            Tensor [B] of dtype torch.long with sampled token ids.
        """
        # Extract last-position logits if 3D input
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [B, vocab_size]

        batch_size = logits.shape[0]
        device = logits.device

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        if world_size > 1:
            # TP protocol: rank 0 samples, broadcast to all
            tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
            if rank == 0:
                sampled = self._do_sample(logits, temperature, top_p)
                tokens.copy_(sampled)
            # Broadcast: rank 0's result overwrites all other ranks
            dist.broadcast(tokens, src=0)
            return tokens
        else:
            # Single-GPU: direct sampling
            return self._do_sample(logits, temperature, top_p)

    # ------------------------------------------------------------------
    # Internal sampling logic
    # ------------------------------------------------------------------

    def _do_sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Core sampling: greedy or top_p nucleus sampling.

        Args:
            logits: [B, vocab_size] — raw logits at the last position.
            temperature: Temperature scaling factor.
            top_p: Nucleus sampling threshold.

        Returns:
            Tensor [B] of dtype torch.long with sampled token ids.
        """
        # Greedy shortcut
        if temperature == 0.0:
            return logits.argmax(dim=-1)  # [B]

        if temperature != 1.0:
            logits = logits / temperature

        if top_p < 1.0:
            logits = self._apply_top_p(logits, top_p)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

    # ------------------------------------------------------------------
    # Top-P (nucleus) filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits.

        Args:
            logits: [B, vocab_size] — scaled logits.
            top_p: Cumulative probability threshold (0 < top_p < 1).

        Returns:
            Filtered logits with low-probability tokens set to -inf.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Mask tokens beyond top_p threshold
        sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        # Always keep at least the top token
        sorted_mask[..., 0] = False

        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original order
        return torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)
