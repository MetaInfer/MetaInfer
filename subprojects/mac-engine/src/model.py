# Qwen3 model architecture — pure mlx.nn, zero mlx_lm dependency.
"""Qwen3 transformer model for MLX.

Reference: mlx_lm/models/qwen3.py (read-only architecture reference).
Built entirely with mlx.core and mlx.nn.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Qwen3Config:
    """Qwen3 model configuration, parsed from config.json."""

    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    head_dim: int = 128
    tie_word_embeddings: bool = False
    rope_scaling: dict | None = None
    model_type: str = "qwen3"

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3Config:
        return cls(
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__}  # type: ignore[arg-type]
        )


# --- Attention ---
class Qwen3Attention(nn.Module):
    """Grouped-query attention with Q/K normalization and RoPE."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# --- MLP ---
class Qwen3MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        dim = config.hidden_size
        hidden = config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# --- Transformer Block ---
class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# --- Full Model ---
class Qwen3Model(nn.Module):
    """Qwen3 decoder-only transformer."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        L = h.shape[1]
        mask = None
        if cache[0] is not None:
            if L > 1:
                # Prefill with cache: use SDPA "causal" fast path
                mask = "causal"
            # L==1 (decode): no mask needed — SDPA handles causal constraint
        elif L > 1:
            # Initial prefill: use SDPA "causal" fast path
            mask = "causal"

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model with language modeling head."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers
