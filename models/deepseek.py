from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


@dataclass
class DeepSeekConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            kv_lora_rank=raw["kv_lora_rank"],
            qk_nope_head_dim=raw["qk_nope_head_dim"],
            qk_rope_head_dim=raw["qk_rope_head_dim"],
            v_head_dim=raw["v_head_dim"],
            moe_intermediate_size=raw["moe_intermediate_size"],
            n_routed_experts=raw["n_routed_experts"],
            n_shared_experts=raw["n_shared_experts"],
            num_experts_per_tok=raw["num_experts_per_tok"],
            first_k_dense_replace=raw["first_k_dense_replace"],
            routed_scaling_factor=float(raw.get("routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(raw.get("norm_topk_prob", False)),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class DeepSeekMLA(nn.Module):
    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.softmax_scale = self.q_head_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).expand(bsz, seqlen)

        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_rope = k_rope.view(bsz, seqlen, 1, self.qk_rope_head_dim).transpose(1, 2).expand(-1, self.num_heads, -1, -1)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_rope, k_rope = _apply_rope(q_rope, k_rope, position_ids, self.rope_theta)
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            scale=self.softmax_scale,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(self.hidden_size, config.moe_intermediate_size) for _ in range(self.n_routed_experts)]
        )
        self.shared_experts = FeedForward(
            self.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = torch.softmax(self.gate(flat).float(), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        if self.norm_topk_prob and self.num_experts_per_tok > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        routed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id
            if not mask.any():
                continue
            token_mask = mask.any(dim=-1)
            token_pos = torch.nonzero(token_mask, as_tuple=False).squeeze(-1)
            expert_out = expert(flat[token_pos])
            selected = mask[token_pos].float()
            selected_weight = (selected * topk_weight[token_pos]).sum(dim=-1, keepdim=True).to(expert_out.dtype)
            routed[token_pos] += expert_out * selected_weight

        shared = self.shared_experts(flat)
        return (routed + shared).view(orig_shape)


class DeepSeekLayer(nn.Module):
    def __init__(self, config: DeepSeekConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DeepSeekMLA(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp: nn.Module = FeedForward(config.hidden_size, config.hidden_size * 4)
        else:
            self.mlp = DeepSeekMoE(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        h = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_ids=position_ids)
        return h + self.mlp(self.post_attention_layernorm(h))


def _build_hf_key(local_name: str, layer_idx: int) -> str:
    if local_name.startswith("input_layernorm."):
        return f"model.layers.{layer_idx}.input_layernorm.weight"
    if local_name.startswith("post_attention_layernorm."):
        return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
    if local_name.startswith("self_attn."):
        suffix = local_name[len("self_attn.") :]
        return f"model.layers.{layer_idx}.self_attn.{suffix}"
    if local_name.startswith("mlp.gate.weight"):
        return f"model.layers.{layer_idx}.mlp.gate.weight"
    if local_name.startswith("mlp.shared_experts."):
        suffix = local_name[len("mlp.shared_experts.") :]
        return f"model.layers.{layer_idx}.mlp.shared_experts.{suffix}"
    if local_name.startswith("mlp.experts."):
        suffix = local_name[len("mlp.experts.") :]
        return f"model.layers.{layer_idx}.mlp.experts.{suffix}"
    if local_name.startswith("mlp."):
        suffix = local_name[len("mlp.") :]
        return f"model.layers.{layer_idx}.mlp.{suffix}"
    raise KeyError(f"Unsupported key mapping for {local_name}")


def load_weights(model: nn.Module, model_dir: str | Path, layer_idx: int) -> dict[str, str]:
    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        index = json.load(f)["weight_map"]

    loaded: dict[str, str] = {}
    tensors_by_file: dict[str, list[tuple[str, str]]] = {}
    for name, _ in model.named_parameters():
        hf_key = _build_hf_key(name, layer_idx)
        if hf_key not in index:
            continue
        shard = index[hf_key]
        tensors_by_file.setdefault(shard, []).append((name, hf_key))

    for shard, items in tensors_by_file.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for local_name, hf_key in items:
                tensor = f.get_tensor(hf_key)
                param = dict(model.named_parameters())[local_name]
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {local_name}: model {tuple(param.shape)} vs ckpt {tuple(tensor.shape)}"
                    )
                param.data.copy_(tensor.to(dtype=param.dtype))
                loaded[local_name] = hf_key
    return loaded
