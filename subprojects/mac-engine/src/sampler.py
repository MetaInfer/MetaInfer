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
