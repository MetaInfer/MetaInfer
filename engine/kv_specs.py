"""
KV 占用估算：从 HuggingFace config 读取真实 MLA 维度，与 HF 中 materialized K/V 形状一致。
MoE 仅影响 FFN，不参与 KV；层数按 num_hidden_layers（每层均有 self_attn）。
"""

from __future__ import annotations

from typing import Any

import torch


def hf_deepseek_v2_kv_bytes_per_token(config: Any, dtype: torch.dtype) -> int:
    """
    每层缓存的 key_states / value_states（与 modeling_deepseek 中 past_key_value 展开形状一致）：
    - key: [..., num_heads, q_len, q_head_dim], q_head_dim = qk_nope + qk_rope
    - value: [..., num_heads, q_len, v_head_dim]
    """
    layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    if hasattr(config, "qk_nope_head_dim") and hasattr(config, "qk_rope_head_dim") and hasattr(config, "v_head_dim"):
        q_head_dim = int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim)
        v_head_dim = int(config.v_head_dim)
    else:
        # Dense/GQA fallback (e.g. Qwen): materialized K/V use kv_heads * head_dim
        kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        head_dim = int(getattr(config, "head_dim", int(config.hidden_size) // num_heads))
        q_head_dim = head_dim
        v_head_dim = head_dim
        num_heads = kv_heads
    if dtype in (torch.float16, torch.bfloat16):
        elem = 2
    elif dtype == torch.float32:
        elem = 4
    else:
        elem = 2
    per_layer = num_heads * (q_head_dim + v_head_dim) * elem
    return layers * per_layer


def hf_deepseek_v2_kv_bytes_per_block(config: Any, dtype: torch.dtype, block_size: int) -> int:
    return hf_deepseek_v2_kv_bytes_per_token(config, dtype) * block_size
