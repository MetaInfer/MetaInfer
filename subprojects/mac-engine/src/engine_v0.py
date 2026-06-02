# Phase 0: 最简可用 — 模型加载 + 单步生成，无 KV cache
"""Phase 0 inference engine. No mlx_lm dependency.

Full sequence re-encode per token. Slow but proves end-to-end works.
"""

from __future__ import annotations

import mlx.core as mx

from .kv_cache import make_kv_cache
from .model import Qwen3Config
from .sampler import temperature_sample
from .tokenizer import Tokenizer
from .weights import load_qwen3_model


class InferenceEngine:
    """Phase 0: minimal engine with full-sequence forward per token."""

    def __init__(self) -> None:
        self.model = None
        self.config: Qwen3Config | None = None
        self.tokenizer: Tokenizer | None = None

    def load_model(self, model_path: str) -> None:
        self.model, self.config = load_qwen3_model(model_path)
        self.tokenizer = Tokenizer(model_path)

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
        if self.model is None or self.tokenizer is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        token_ids = self.tokenizer.encode(prompt)

        for _ in range(max_tokens):
            # Full forward pass (no cache → full attention computation)
            input_ids = mx.array([token_ids])
            logits = self.model(input_ids)

            next_id = temperature_sample(logits[0, -1, :], temperature)
            token_ids.append(next_id)
            yield self.tokenizer.decode([next_id])
