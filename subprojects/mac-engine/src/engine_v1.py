# Phase 1: KV Cache 增量解码
"""Phase 1 inference engine. No mlx_lm dependency.

Prefill once, then incremental decode using our custom KVCache.
"""

from __future__ import annotations

import mlx.core as mx

from .kv_cache import KVCache, make_kv_cache
from .model import Qwen3Config
from .sampler import temperature_sample
from .tokenizer import Tokenizer
from .weights import load_qwen3_model


def _compiled_sample(logits_last: mx.array, temperature: float) -> mx.array:
    """Compiled argmax/greedy sampling. Returns single-element array [next_id]."""
    if temperature <= 0.0:
        next_id = mx.argmax(logits_last, axis=-1, keepdims=True)
    else:
        probs = mx.softmax(logits_last / temperature, axis=-1)
        next_id = mx.random.categorical(probs)
        next_id = mx.expand_dims(next_id, axis=0)
    return next_id


_compiled_sample = mx.compile(_compiled_sample, shapeless=True)


class InferenceEngine:
    """Phase 1: KV cache incremental decode."""

    def __init__(self) -> None:
        self.model = None
        self.config: Qwen3Config | None = None
        self.tokenizer: Tokenizer | None = None
        self._cache: list[KVCache] | None = None

    def load_model(self, model_path: str) -> None:
        self.model, self.config = load_qwen3_model(model_path)
        self.tokenizer = Tokenizer(model_path)

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
        if self.model is None or self.tokenizer is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        token_ids = self.tokenizer.encode(prompt)

        # Create fresh KV cache (one per layer)
        n_layers = len(self.model.layers)
        self._cache = make_kv_cache(n_layers)

        # Prefill: full prompt forward with cache
        input_ids = mx.array([token_ids])
        logits = self.model(input_ids, cache=self._cache)

        # Decode loop
        for _ in range(max_tokens):
            next_id = temperature_sample(logits[0, -1, :], temperature)
            yield self.tokenizer.decode([next_id])

            # Decode: single token forward with cache
            next_input = mx.array([[next_id]])
            logits = self.model(next_input, cache=self._cache)
