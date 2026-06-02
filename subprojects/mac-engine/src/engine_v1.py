# Phase 1: KV Cache 增量解码
"""Phase 1 inference engine. No mlx_lm dependency.

Prefill once, then incremental decode using our custom KVCache.
"""

from __future__ import annotations

import mlx.core as mx

from .kv_cache import KVCache, make_kv_cache
from .model import Qwen3Config
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

# Thread-local stream for async pipeline (see mlx_lm/generate.py:226)
_generation_stream = mx.new_thread_local_stream(mx.gpu)


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

    def generate_stream(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
        """Generate tokens with stream pipeline (mx.async_eval pattern).

        Pipeline: async_eval(token N) while building graph for token N+1,
        so GPU compute overlaps with graph building + tokenizer.decode.
        """
        if self.model is None or self.tokenizer is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        token_ids = self.tokenizer.encode(prompt)
        n_layers = len(self.model.layers)
        self._cache = make_kv_cache(
            n_layers,
            n_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_len=2048,
        )

        # Prefill: full prompt forward with cache
        input_ids = mx.array([token_ids])
        logits = self.model(input_ids, cache=self._cache)

        # Define inner step: forward + sample (runs on generation stream)
        def _step(tok_arr: mx.array) -> mx.array:
            """One decode step on generation stream. Returns sampled token id."""
            with mx.stream(_generation_stream):
                _in = tok_arr.reshape(1, 1)
                _logits = self.model(_in, cache=self._cache)
                return _compiled_sample(_logits[0, -1, :], temperature)

        # Pipeline: async_eval pattern (see mlx_lm/generate.py:453-470)
        y = _compiled_sample(logits[0, -1, :], temperature)

        n = 0
        while True:
            # Build graph for next token (while GPU evaluates current y)
            if n != max_tokens:
                next_y = _step(y)
                mx.async_eval(next_y)
            if n == 0:
                mx.eval(y)  # Ensure first token is ready
            if n == max_tokens:
                break
            next_id = int(y.item())
            yield self.tokenizer.decode([next_id])
            if n % 256 == 0:
                mx.clear_cache()  # Prevent unbounded graph cache growth
            y = next_y
            n += 1

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
        if self.model is None or self.tokenizer is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        token_ids = self.tokenizer.encode(prompt)

        # Create fresh KV cache (one per layer), pre-allocate for 2048 tokens
        n_layers = len(self.model.layers)
        self._cache = make_kv_cache(
            n_layers,
            n_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_len=2048,
        )

        # Prefill: full prompt forward with cache
        input_ids = mx.array([token_ids])
        logits = self.model(input_ids, cache=self._cache)

        # Pre-allocate next_input tensor (1x1, reused)
        _next_input = mx.zeros((1, 1), mx.int32)

        # Decode loop
        generated = 0
        while generated < max_tokens:
            # Compiled argmax: stays in MX graph, no .item() sync
            next_logits = logits[0, -1, :]
            next_id_arr = _compiled_sample(next_logits, temperature)
            next_id = int(next_id_arr.item())  # only sync point

            token_text = self.tokenizer.decode([next_id])
            generated += 1
            yield token_text

            if generated >= max_tokens:
                break

            # Reuse pre-allocated buffer, avoid new allocation
            _next_input[0, 0] = next_id
            logits = self.model(_next_input, cache=self._cache)
