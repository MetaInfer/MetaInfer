# Custom KV Cache — zero mlx_lm dependency.
"""Custom KV cache for incremental decoding.

Design mirrors mlx_lm.models.cache.KVCache pattern but implemented from scratch.
Each layer gets one KVCache. On update_and_fetch(), new K/V are appended
to the stored cache and the full concatenated K/V is returned.
"""

from __future__ import annotations

import mlx.core as mx


class KVCache:
    """Per-layer KV cache with pre-allocated buffer.

    Stores K and V tensors of shape [B, n_kv_heads, seq_len, head_dim].
    Grows in chunks of `step` tokens to minimize reallocation.
    """

    step: int = 256

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset: int = 0

    @classmethod
    def pre_allocated(cls, n_kv_heads: int, head_dim: int, max_len: int) -> KVCache:
        """Create a KVCache with pre-allocated buffer for max_len tokens."""
        cache = cls()
        cache.keys = mx.zeros((1, n_kv_heads, max_len, head_dim), mx.float16)
        cache.values = mx.zeros((1, n_kv_heads, max_len, head_dim), mx.float16)
        cache.offset = 0
        return cache

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Store new K/V and return full cached K/V up to current offset."""
        prev = self.offset
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[3]

        # Grow buffer if needed
        if self.keys is None or (prev + S) > self.keys.shape[2]:
            n_steps = (self.step + S - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        # Store new K/V
        self.offset += S
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.keys is None


def make_kv_cache(num_layers: int, n_kv_heads: int = 8, head_dim: int = 128,
                   max_len: int = 0) -> list[KVCache]:
    """Create KV cache list. If max_len > 0, pre-allocate buffers."""
    if max_len > 0:
        return [KVCache.pre_allocated(n_kv_heads, head_dim, max_len) 
                for _ in range(num_layers)]
    return [KVCache() for _ in range(num_layers)]
