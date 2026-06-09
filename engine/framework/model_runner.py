"""
Phase 9 — TPModelRunner: loads QwenForCausalLMTP + tokenizer, executes prefill/decode
forward passes, calls Sampler for next-token selection.

TP path (qwen_tp / deepseek_tp):
  - prefill: model.forward(input_ids, past_key_values=None)
  - decode:  model.forward(input_ids, past_key_values=kv_len)
  - Sampling: Sampler (rank 0 only + broadcast for TP>1).

All signatures match inference_blueprint.json
  > components[3] ModelRunner
  > tp_runner_actual_flow
  > scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from engine.models.qwen import QwenForCausalLMTP, QwenTPConfig
from engine.framework.sampler import Sampler
from engine.tp_layers.distributed import init_custom_ar


@dataclass
class RunnerOutput:
    """Output from a single runner.run() call."""
    next_tokens: List[int] = field(default_factory=list)


class TPModelRunner:
    """Tensor-parallel model runner for Qwen3 Dense.

    Construction (5-step chain per blueprint):
      1. QwenTPConfig.from_config(model_dir) → cfg
      2. QwenForCausalLMTP(cfg, device, dtype)
      3. model.load_weights()  (includes barrier + CustomAR init when TP>1)
      4. model.eval()
      5. init_custom_ar()  (double-call is safe — guarded by world_size <= 1)

    KV cache: managed internally by QwenAttentionTP (lazy alloc, torch.arange).
    BlockManager is NOT used — num_free_blocks comes from the attendion layer.
    """

    def __init__(
        self,
        model_dir: Path,
        tp_size: int = 1,
        device: Optional[torch.device] = None,
        load_weights: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.tp_size = tp_size

        # 1. Config
        self.cfg = QwenTPConfig.from_config(self.model_dir)
        self.max_seq_len = self.cfg.max_position_embeddings

        # 2. Device
        if device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = device
        self.dtype = torch.bfloat16

        # 3. Model
        self.model = QwenForCausalLMTP(
            self.cfg, device=self.device, dtype=self.dtype
        )
        # Move to correct device and dtype before weight loading
        self.model.to(device=self.device, dtype=self.dtype)

        # 4. Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), trust_remote_code=True
        )

        # 5. Weight loading
        if load_weights:
            self.model.load_weights(self.model_dir)

        # 6. Eval mode
        self.model.eval()

        # 7. CustomAR (safe re-init — guarded internally)
        init_custom_ar(device=self.device)

        # 8. Sampler
        self.sampler = Sampler()

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        seqs: List,
        is_prefill: bool,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ) -> RunnerOutput:
        """Execute one prefill or decode step.

        Args:
            seqs: List of Sequence objects (B=1 single-seq in current impl).
            is_prefill: True for prefill, False for decode.
            temperature: Sampling temperature (0.0 = greedy argmax).
            top_p: Nucleus sampling threshold (None/1.0 = disabled).

        Returns:
            RunnerOutput with next_tokens (one per sequence).
        """
        if not seqs:
            return RunnerOutput(next_tokens=[])

        top_p_val = top_p if top_p is not None else 1.0

        if is_prefill:
            # Concatenate input_ids across sequences [1, total_tokens]
            input_ids = torch.cat(
                [s.input_ids_tensor(device=self.device).unsqueeze(0) for s in seqs],
                dim=1,
            )  # [1, total_tokens]

            logits, _ = self.model(
                input_ids,
                past_key_values=None,
                position_offset=0,
                max_seq_len=self.max_seq_len,
            )  # [1, total_tokens, vocab_size]

            # Update kv_len: each seq now has prompt_len tokens in KV cache
            for s in seqs:
                s.kv_len = s.seq_len()

        else:
            # Decode: single token per sequence (B=1 current impl)
            kv_lens = [s.kv_len for s in seqs]
            input_ids = torch.tensor(
                [[s.output_ids[-1]] for s in seqs],
                dtype=torch.long,
                device=self.device,
            )  # [B, 1]

            kv_len = kv_lens[0]  # B=1 single-seq
            logits, new_kv_lens = self.model.forward_decode(
                input_ids,
                past_key_values=kv_len,
                position_offset=kv_len,
                max_seq_len=self.max_seq_len,
            )  # [1, 1, vocab_size]

            # Update kv_len: model.forward() returns authoritative kv_lens
            for s, kv in zip(seqs, new_kv_lens if isinstance(new_kv_lens, list) else [new_kv_lens]):
                s.kv_len = kv

        # Sampling (last-position logits): rank 0 only + broadcast for TP>1
        sampled = self.sampler.sample(
            logits[:, -1, :], temperature=temperature, top_p=top_p_val
        )  # [B]

        return RunnerOutput(next_tokens=sampled.tolist())

    # ------------------------------------------------------------------
    # get_num_free_blocks
    # ------------------------------------------------------------------

    def get_num_free_blocks(self) -> int:
        """Return the number of free KV cache blocks.

        TP Runner path: reads _kv_len_gpu from the first decoder layer's
        attention.  All layers have the same kv_len at decode start.

        Formula:
            allocated = (kv_len + 255) // 256   (ceil, partial blocks count as used)
            free = max_blocks - allocated
        """
        kv_len = self.model.layers[0].self_attn._kv_len_gpu[0].item()
        max_blocks = self.cfg.max_position_embeddings // 256
        allocated = (kv_len + 255) // 256
        return max_blocks - allocated
