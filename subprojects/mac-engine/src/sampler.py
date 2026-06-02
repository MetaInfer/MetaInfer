# Sampling utilities — zero mlx_lm dependency.
"""Greedy and temperature-based sampling from logits."""

from __future__ import annotations

import mlx.core as mx


def greedy_sample(logits: mx.array) -> int:
    """Greedy argmax sampling. Returns single token id."""
    return int(mx.argmax(logits, axis=-1).item())  # type: ignore[no-any-return]


def temperature_sample(logits: mx.array, temperature: float = 1.0) -> int:
    """Sampling with temperature. T=0 falls back to greedy."""
    if temperature <= 0.0:
        return greedy_sample(logits)
    probs = mx.softmax(logits / temperature, axis=-1)
    return int(mx.random.categorical(probs).item())  # type: ignore[no-any-return]


def _compiled_sample_fn(logits_last: mx.array, temperature: float) -> mx.array:
    """Compiled sampling for engine decode loop. Returns [next_id] array."""
    if temperature <= 0.0:
        return mx.argmax(logits_last, axis=-1, keepdims=True)
    probs = mx.softmax(logits_last / temperature, axis=-1)
    next_id = mx.random.categorical(probs)
    return mx.expand_dims(next_id, axis=0)


# Compile once — shapeless so it works for any vocab size
compiled_sample = mx.compile(_compiled_sample_fn, shapeless=True)
