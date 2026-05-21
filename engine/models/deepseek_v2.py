from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from safetensors import safe_open

from engine.kernels.triton_mla_decode import triton_mla_decode
from transformers import AutoConfig, AutoTokenizer

from engine.sampler import sample_next_tokens
from engine.structs import Sequence
from engine.tp_layers import (
    ColumnParallelLinear,
    ParallelLMHead,
    RowParallelLinear,
    VocabParallelEmbedding,
    ensure_divisible,
    get_tp_rank,
    get_tp_size,
    init_custom_ar,
    init_tp_distributed,
)
from engine.tp_layers.moe import ExpertParallelMoE, ExpertParallelMoEConfig


@dataclass
class DeepseekV2TPConfig:
    model_dir: Path
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    first_k_dense_replace: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    routed_scaling_factor: float
    tie_word_embeddings: bool
    rope_scaling: dict[str, Any] | None

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


def _load_deepseek_v2_tp_config(model_dir: str | Path) -> DeepseekV2TPConfig:
    p = Path(model_dir)
    cfg = AutoConfig.from_pretrained(str(p), trust_remote_code=True, local_files_only=True)
    return DeepseekV2TPConfig(
        model_dir=p,
        hidden_size=int(cfg.hidden_size),
        intermediate_size=int(cfg.intermediate_size),
        moe_intermediate_size=int(cfg.moe_intermediate_size),
        num_hidden_layers=int(cfg.num_hidden_layers),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        vocab_size=int(cfg.vocab_size),
        rms_norm_eps=float(cfg.rms_norm_eps),
        rope_theta=float(cfg.rope_theta),
        q_lora_rank=None if getattr(cfg, "q_lora_rank", None) is None else int(cfg.q_lora_rank),
        kv_lora_rank=int(cfg.kv_lora_rank),
        qk_nope_head_dim=int(cfg.qk_nope_head_dim),
        qk_rope_head_dim=int(cfg.qk_rope_head_dim),
        v_head_dim=int(cfg.v_head_dim),
        first_k_dense_replace=int(cfg.first_k_dense_replace),
        n_routed_experts=int(cfg.n_routed_experts),
        num_experts_per_tok=int(cfg.num_experts_per_tok),
        n_shared_experts=int(cfg.n_shared_experts),
        routed_scaling_factor=float(cfg.routed_scaling_factor),
        tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        rope_scaling=getattr(cfg, "rope_scaling", None),
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def _rotate_half_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: int, high: int, dim: int, device: torch.device) -> torch.Tensor:
    if low == high:
        high += 1
    linear = (torch.arange(dim, device=device, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


def _compute_inv_freq(
    dim: int,
    theta: float,
    rope_scaling: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor:
    base_freq = theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    inv_freq = 1.0 / base_freq
    if not rope_scaling or rope_scaling.get("type") != "yarn":
        return inv_freq

    factor = float(rope_scaling.get("factor", 1.0))
    beta_fast = int(rope_scaling.get("beta_fast", 32))
    beta_slow = int(rope_scaling.get("beta_slow", 1))
    original_max_pos = int(rope_scaling.get("original_max_position_embeddings", 4096))
    inv_interp = 1.0 / (factor * base_freq)
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, theta, original_max_pos)
    inv_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, device=device)
    return inv_interp * (1.0 - inv_mask) + inv_freq * inv_mask


def _apply_rope_gptj(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
    rope_scaling: dict[str, Any] | None,
) -> torch.Tensor:
    dim = x.shape[-1]
    inv_freq = _compute_inv_freq(dim, theta, rope_scaling, x.device)
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    if rope_scaling and rope_scaling.get("type") == "yarn":
        factor = float(rope_scaling.get("factor", 1.0))
        mscale = float(rope_scaling.get("mscale", 1.0))
        mscale_all_dim = float(rope_scaling.get("mscale_all_dim", 1.0))
        rope_mscale = _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(factor, mscale_all_dim)
        cos = cos * rope_mscale
        sin = sin * rope_mscale
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)
    return x * cos + _rotate_half_gptj(x) * sin


class DeepseekAttentionTP(nn.Module):
    def __init__(self, cfg: DeepseekV2TPConfig):
        super().__init__()
        self.cfg = cfg
        self.tp = get_tp_size()
        self.rank = get_tp_rank()
        ensure_divisible(cfg.num_attention_heads, self.tp, name="num_attention_heads")
        self.local_heads = cfg.num_attention_heads // self.tp
        self.local_qk = self.local_heads * cfg.qk_head_dim
        self.local_kv_expand = self.local_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
        self.scaling = cfg.qk_head_dim**-0.5
        if cfg.rope_scaling and cfg.rope_scaling.get("type") == "yarn":
            factor = float(cfg.rope_scaling.get("factor", 1.0))
            mscale_all_dim = float(cfg.rope_scaling.get("mscale_all_dim", 1.0))
            self.scaling = self.scaling * _yarn_get_mscale(factor, mscale_all_dim) ** 2

        # Replicated (must not shard)
        if cfg.q_lora_rank is None:
            self.q_a_proj = None
            self.q_a_layernorm = None
        else:
            self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)

        # Sharded (must shard)
        q_b_in = cfg.hidden_size if cfg.q_lora_rank is None else cfg.q_lora_rank
        self.q_b_proj = ColumnParallelLinear(q_b_in, cfg.num_attention_heads * cfg.qk_head_dim, bias=False, gather_output=False)
        self.kv_b_proj_with_mqa = ColumnParallelLinear(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
            gather_output=False,
        )
        self.o_proj = RowParallelLinear(cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size, bias=False)

        # Triton MLA weights: W_UK_T projects q_nope to latent space, W_UV expands output
        # These are extracted from kv_b_proj_with_mqa (no extra memory)
        kv_lora_rank = cfg.kv_lora_rank
        nope_d = cfg.qk_nope_head_dim
        v_d = cfg.v_head_dim
        # W_UK_T: [H, kv_lora_rank, qk_nope] for q_nope @ W_UK_T
        # W_UV: [H, kv_lora_rank, v_head_dim] for out_latent @ W_UV
        # These will be initialized after load_weights() when kv_b_proj is available
        self.W_UK_T: torch.Tensor | None = None
        self.W_UV: torch.Tensor | None = None

        # Unified KV cache buffer for Triton MLA: [B, max_seq_len, 1, kv_lora_rank + rope_dim]
        self._kv_cache_buf: torch.Tensor | None = None
        self._buf_max_seq_len = 0

        # FA2 pre-allocated buffers (for prefill path)
        self._k_cat_buf: torch.Tensor | None = None
        self._v_pad_buf: torch.Tensor | None = None
        self._q_cat_buf: torch.Tensor | None = None
        self._all_pos: torch.Tensor | None = None
        self._cu_q: torch.Tensor | None = None
        self._cu_k: torch.Tensor | None = None

    def init_buffers(self, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Pre-allocate decode buffers. Call before torch.compile."""
        if max_seq_len <= self._buf_max_seq_len:
            return
        self._buf_max_seq_len = max_seq_len
        H, D = self.local_heads, self.cfg.qk_head_dim
        kv_lora_rank = self.cfg.kv_lora_rank
        rope_d = self.cfg.qk_rope_head_dim

        # FA2 buffers (for prefill)
        self._k_cat_buf = torch.zeros(1, max_seq_len, H, D, device=device, dtype=dtype)
        self._v_pad_buf = torch.zeros(1, max_seq_len, H, D, device=device, dtype=dtype)
        self._q_cat_buf = torch.zeros(1, max_seq_len, H, D, device=device, dtype=dtype)
        self._all_pos = torch.arange(max_seq_len, device=device, dtype=torch.long)
        self._cu_q = torch.tensor([0, 0], dtype=torch.int32, device=device)
        self._cu_k = torch.tensor([0, 0], dtype=torch.int32, device=device)

        # Triton MLA unified KV cache: [1, max_seq_len, 1, kv_lora_rank + rope_dim]
        self._kv_cache_buf = torch.zeros(
            1, max_seq_len, 1, kv_lora_rank + rope_d,
            device=device, dtype=dtype,
        )

    def _init_mla_weights(self) -> None:
        """Extract W_UK_T and W_UV from kv_b_proj_with_mqa. Call after load_weights.

        Follows vLLM's approach:
        - kv_b_proj weight: [out_features, in_features] = [H*(P+V), Lkv]
        - .T → [Lkv, H*(P+V)]
        - view → [Lkv, H, P+V]
        - split → W_UK [Lkv, H, P], W_UV [Lkv, H, V]
        - W_UK_T = W_UK.permute(1, 2, 0) → [H, P, Lkv]
        - W_UV = W_UV.transpose(0, 1) → [H, Lkv, V]
        """
        nope_d = self.cfg.qk_nope_head_dim
        v_d = self.cfg.v_head_dim
        kv_lora_rank = self.cfg.kv_lora_rank
        H = self.local_heads

        # Transpose weight: [H*(P+V), Lkv] → [Lkv, H*(P+V)]
        W = self.kv_b_proj_with_mqa.weight.data.T
        # Reshape: [Lkv, H, P+V]
        W = W.view(kv_lora_rank, H, nope_d + v_d)
        # Split into W_UK and W_UV
        W_UK, W_UV = W.split([nope_d, v_d], dim=-1)
        # W_UK_T: [H, P, Lkv] — for q_nope @ W_UK_T → [H, 1, Lkv]
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        # W_UV: [H, Lkv, V] — for out_latent @ W_UV → [H, 1, V]
        self.W_UV = W_UV.transpose(0, 1).contiguous()

    def _local_slice_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.local_heads, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
        max_seq_len: int = 512,
        layer_idx: int = -1,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Forward with pre-allocated KV cache buffer.

        past_key_values: (k_nope_buf, v_buf, raw_k_pe_buf, kv_len) where
          - k_nope_buf: [B, max_seq_len, num_kv_heads, qk_nope_head_dim] (pre-allocated)
          - v_buf: [B, max_seq_len, num_kv_heads, v_head_dim] (pre-allocated)
          - raw_k_pe_buf: [B, max_seq_len, 1, qk_rope_head_dim] (pre-allocated)
          - kv_len: int (number of valid cached tokens)
        Returns: (output, new_cache)
        """
        bsz, seqlen, _ = hidden_states.shape

        if self.cfg.q_lora_rank is None:
            q_full = self.q_b_proj(hidden_states)
        else:
            q_latent = self.q_a_proj(hidden_states)
            q_latent = self.q_a_layernorm(q_latent)
            q_full = self.q_b_proj(q_latent)
        q = self._local_slice_heads(q_full, self.cfg.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim], dim=-1)

        kv_latent_plus_pe = self.kv_a_proj_with_mqa(hidden_states)
        c_kv, k_pe = torch.split(kv_latent_plus_pe, [self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)
        raw_k_pe = k_pe.view(bsz, seqlen, 1, self.cfg.qk_rope_head_dim)

        if past_key_values is None:
            # ===== PREFILL: FA2 + V-padding + write unified KV cache =====
            # Expand c_kv to per-head via kv_b_proj for FA2
            kv_full = self.kv_b_proj_with_mqa(c_kv)
            kv_full = self._local_slice_heads(kv_full, self.cfg.qk_nope_head_dim + self.cfg.v_head_dim)
            k_nope, v = torch.split(kv_full, [self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], dim=-1)

            # Allocate unified KV cache buffer
            kv_cache_buf = torch.zeros(
                bsz, max_seq_len, 1, self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            kv_len = seqlen

            # FA2 attention (prefill only)
            q_pe = _apply_rope_gptj(q_pe, positions, self.cfg.rope_theta, self.cfg.rope_scaling)
            k_pe_rope = _apply_rope_gptj(
                raw_k_pe[:, :kv_len], self._all_pos[:kv_len],
                self.cfg.rope_theta, self.cfg.rope_scaling,
            )
            k_pe_rope = k_pe_rope.expand(-1, -1, self.local_heads, -1)

            # Write c_kv (latent) and k_pe_rope (with RoPE applied) into unified cache
            # vLLM stores RoPE'd k_pe in cache — the kernel expects it
            kv_cache_buf[:, :kv_len, :, :self.cfg.kv_lora_rank] = c_kv.unsqueeze(2)
            kv_cache_buf[:, :kv_len, :, self.cfg.kv_lora_rank:] = k_pe_rope[:, :, :1, :]

            nope_d = self.cfg.qk_nope_head_dim
            self._q_cat_buf[:, :seqlen] = torch.cat([q_nope, q_pe], dim=-1)
            q_fa = self._q_cat_buf[0, :seqlen]
            self._k_cat_buf[0, :kv_len, :, :nope_d] = k_nope[0, :kv_len]
            self._k_cat_buf[0, :kv_len, :, nope_d:] = k_pe_rope[0]
            k_fa = self._k_cat_buf[0, :kv_len]
            self._v_pad_buf[0, :kv_len, :, :self.cfg.v_head_dim] = v[0, :kv_len]
            v_fa = self._v_pad_buf[0, :kv_len]
            self._cu_q[1] = seqlen
            out = flash_attn_varlen_func(
                q_fa, k_fa, v_fa,
                cu_seqlens_q=self._cu_q, cu_seqlens_k=self._cu_q,
                max_seqlen_q=seqlen, max_seqlen_k=kv_len,
                causal=True, softmax_scale=self.scaling,
            )
            out = out[:, :, :self.cfg.v_head_dim].reshape(bsz, seqlen, self.local_heads * self.cfg.v_head_dim)
        else:
            # ===== DECODE: Triton MLA kernel =====
            kv_cache_buf, kv_len = past_key_values

            # Write new token's c_kv and k_pe (with RoPE) into unified cache
            kv_cache_buf[:, kv_len:kv_len + seqlen, :, :self.cfg.kv_lora_rank] = c_kv.unsqueeze(2)
            # Apply RoPE to k_pe before storing (vLLM stores RoPE'd k_pe)
            k_pe_rope_new = _apply_rope_gptj(
                raw_k_pe, positions,
                self.cfg.rope_theta, self.cfg.rope_scaling,
            )
            kv_cache_buf[:, kv_len:kv_len + seqlen, :, self.cfg.kv_lora_rank:] = k_pe_rope_new
            kv_len = kv_len + seqlen

            # Project q_nope to latent space (vLLM approach)
            # q_nope: [1, 1, H, P] → [1, H, P]
            q_nope_2d = q_nope.squeeze(1)  # [1, H, P]
            # Transpose to [H, 1, P] for bmm
            q_nope_t = q_nope_2d.transpose(0, 1)  # [H, 1, P]
            # bmm: [H, 1, P] @ [H, P, Lkv] → [H, 1, Lkv]
            q_nope_proj = torch.bmm(q_nope_t, self.W_UK_T)  # [H, 1, Lkv]
            # Transpose back to [1, H, Lkv]
            q_nope_latent = q_nope_proj.transpose(0, 1)  # [1, H, Lkv]

            # Apply RoPE to q_pe
            q_pe = _apply_rope_gptj(q_pe, positions, self.cfg.rope_theta, self.cfg.rope_scaling)
            # q_pe: [1, 1, H, R] → [1, H, R]
            q_pe_2d = q_pe.squeeze(1)  # [1, H, R]

            # Concatenate: [1, H, Lkv+R] = [1, H, 576]
            q_mla = torch.cat([q_nope_latent, q_pe_2d], dim=-1)

            # Triton MLA kernel
            # q_mla: [1, H, Lkv+R=576], kv_cache: [1, max_seq_len, 1, Lkv+R=576]
            # Scaling: vLLM uses 1/sqrt(qk_head_dim) = 1/sqrt(192), NOT 1/sqrt(576)
            out_latent = triton_mla_decode(
                q=q_mla,
                kv_cache=kv_cache_buf,
                kv_len=kv_len,
                num_kv_splits=8,
                sm_scale=self.scaling,
                lv=self.cfg.kv_lora_rank,
            )  # [1, H, Lkv=512]

            # Expand output from latent space via W_UV (vLLM _v_up_proj approach)
            # out_latent: [1, H, Lkv] → [H, 1, Lkv] for bmm
            out_latent_t = out_latent.transpose(0, 1)  # [H, 1, Lkv]
            # bmm: [H, 1, Lkv] @ [H, Lkv, V] → [H, 1, V]
            out = torch.bmm(out_latent_t, self.W_UV)  # [H, 1, V]
            # Reshape to [1, 1, H*V]
            out = out.transpose(0, 1).reshape(bsz, seqlen, self.local_heads * self.cfg.v_head_dim)

        new_cache = (kv_cache_buf, kv_len)
        return self.o_proj(out), new_cache


class DeepseekMLPTP(nn.Module):
    def __init__(self, cfg: DeepseekV2TPConfig):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False)
        self.up_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekMoETP(nn.Module):
    def __init__(self, cfg: DeepseekV2TPConfig):
        super().__init__()
        self.routed = ExpertParallelMoE(
            ExpertParallelMoEConfig(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.moe_intermediate_size,
                num_experts=cfg.n_routed_experts,
                top_k=cfg.num_experts_per_tok,
                routed_scaling_factor=cfg.routed_scaling_factor,
                score_function="softmax",
            )
        )
        self.shared_experts = (
            DeepseekMLPTP(
                DeepseekV2TPConfig(
                    **{**cfg.__dict__, "intermediate_size": cfg.moe_intermediate_size * cfg.n_shared_experts}
                )
            )
            if cfg.n_shared_experts > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.routed(x)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class DeepseekDecoderLayerTP(nn.Module):
    def __init__(self, cfg: DeepseekV2TPConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = DeepseekAttentionTP(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if layer_idx >= cfg.first_k_dense_replace:
            self.mlp = DeepseekMoETP(cfg)
        else:
            self.mlp = DeepseekMLPTP(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: tuple | None = None,
        max_seq_len: int = 512,
        layer_idx: int = -1,
    ) -> tuple[torch.Tensor, tuple]:
        h = self.input_layernorm(hidden_states)
        h, new_cache = self.self_attn(h, positions, past_key_values, max_seq_len=max_seq_len, layer_idx=layer_idx)
        hidden_states = hidden_states + h
        h2 = self.post_attention_layernorm(hidden_states)
        h2 = self.mlp(h2)
        return hidden_states + h2, new_cache


class DeepseekForCausalLMTP(nn.Module):
    def __init__(self, cfg: DeepseekV2TPConfig, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        self.device = device
        self.dtype = dtype
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DeepseekDecoderLayerTP(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)
        self._hf_debug_model: nn.Module | None = None
        self._use_hf_logits_debug = False
        # P6: pre-allocated position index buffer, sliced in forward
        self.register_buffer("_pos_buf", torch.arange(0, 4096, device=device, dtype=torch.long), persistent=False)
        self.to(device=device, dtype=dtype)

    def _resolve_weight_map(self) -> dict[str, str]:
        index_file = self.cfg.model_dir / "model.safetensors.index.json"
        if not index_file.is_file():
            safes = sorted(self.cfg.model_dir.glob("*.safetensors"))
            if len(safes) == 1:
                return {}
            raise FileNotFoundError(
                f"No model.safetensors.index.json in {self.cfg.model_dir}, and not a single .safetensors file."
            )
        obj = json.loads(index_file.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in obj.get("weight_map", {}).items()}

    def _load_tensor(
        self,
        key: str,
        *,
        split_dim: int | None = None,
        allow_kv_replication: bool = False,
    ) -> torch.Tensor:
        weight_map: dict[str, str] = getattr(
            self, "_safetensors_weight_map", None
        ) or self._resolve_weight_map()
        if weight_map:
            fname = weight_map.get(key)
            if fname is None:
                raise KeyError(f"Missing key {key} in safetensors index")
            fp = self.cfg.model_dir / fname
        else:
            fp = next(self.cfg.model_dir.glob("*.safetensors"))
        if not fp.is_file():
            raise FileNotFoundError(f"Weight file not found: {fp}")

        with safe_open(str(fp), framework="pt", device="cpu") as f:
            if split_dim is None:
                return f.get_tensor(key)
            sl = f.get_slice(key)
            shape = list(sl.get_shape())
            split_size = int(shape[split_dim])
            tp = get_tp_size()
            rank = get_tp_rank()
            if split_size % tp == 0:
                part = split_size // tp
                start = rank * part
                end = start + part
            elif allow_kv_replication and split_size < tp and tp % split_size == 0:
                replicas = tp // split_size
                shard_rank = rank // replicas
                part = 1
                start = shard_rank
                end = start + 1
            else:
                raise ValueError(
                    f"Tensor {key} cannot be split on dim={split_dim}: size={split_size}, tp={tp}"
                )
            index: list[slice] = [slice(None)] * len(shape)
            index[split_dim] = slice(start, end)
            return sl[tuple(index)]

    def load_weights(self) -> None:
        """从 safetensors 惰性加载；MLA 降维投影全量、升维/输出按 TP 切片；路由专家只加载本 rank 的 expert 块。"""
        self._safetensors_weight_map = self._resolve_weight_map()
        try:
            self._load_weights_impl()
        finally:
            del self._safetensors_weight_map

    def _load_weights_impl(self) -> None:
        self.embed_tokens.load_weight_shard(self._load_tensor("model.embed_tokens.weight", split_dim=0))
        for i, layer in enumerate(self.layers):
            pfx = f"model.layers.{i}"
            layer.input_layernorm.weight.data.copy_(
                self._load_tensor(f"{pfx}.input_layernorm.weight").to(layer.input_layernorm.weight)
            )
            layer.post_attention_layernorm.weight.data.copy_(
                self._load_tensor(f"{pfx}.post_attention_layernorm.weight").to(
                    layer.post_attention_layernorm.weight
                )
            )
            sa = layer.self_attn
            if sa.q_a_proj is not None:
                sa.q_a_proj.weight.data.copy_(
                    self._load_tensor(f"{pfx}.self_attn.q_a_proj.weight").to(sa.q_a_proj.weight)
                )
                sa.q_a_layernorm.weight.data.copy_(
                    self._load_tensor(f"{pfx}.self_attn.q_a_layernorm.weight").to(sa.q_a_layernorm.weight)
                )
                sa.q_b_proj.load_weight_shard(self._load_tensor(f"{pfx}.self_attn.q_b_proj.weight", split_dim=0))
            else:
                sa.q_b_proj.load_weight_shard(
                    self._load_tensor(f"{pfx}.self_attn.q_proj.weight", split_dim=0)
                )
            # MLA 降维 + rope 段：严禁切片
            sa.kv_a_proj_with_mqa.weight.data.copy_(
                self._load_tensor(f"{pfx}.self_attn.kv_a_proj_with_mqa.weight").to(sa.kv_a_proj_with_mqa.weight)
            )
            sa.kv_a_layernorm.weight.data.copy_(
                self._load_tensor(f"{pfx}.self_attn.kv_a_layernorm.weight").to(sa.kv_a_layernorm.weight)
            )
            sa.kv_b_proj_with_mqa.load_weight_shard(
                self._load_tensor(f"{pfx}.self_attn.kv_b_proj.weight", split_dim=0)
            )
            sa.o_proj.load_weight_shard(self._load_tensor(f"{pfx}.self_attn.o_proj.weight", split_dim=1))

            if isinstance(layer.mlp, DeepseekMoETP):
                mlp = layer.mlp
                mlp.routed.gate.weight.data.copy_(
                    self._load_tensor(f"{pfx}.mlp.gate.weight").to(mlp.routed.gate.weight)
                )
                for eid_str, expert in mlp.routed.experts.items():
                    e = int(eid_str)
                    b = f"{pfx}.mlp.experts.{e}"
                    # 每个 key 独立存储，只读本 rank 需要的 expert，避免整表加载
                    expert.gate_proj.weight.data.copy_(
                        self._load_tensor(f"{b}.gate_proj.weight").to(expert.gate_proj.weight)
                    )
                    expert.up_proj.weight.data.copy_(
                        self._load_tensor(f"{b}.up_proj.weight").to(expert.up_proj.weight)
                    )
                    expert.down_proj.weight.data.copy_(
                        self._load_tensor(f"{b}.down_proj.weight").to(expert.down_proj.weight)
                    )
                if mlp.shared_experts is not None:
                    mlp.shared_experts.gate_proj.load_weight_shard(
                        self._load_tensor(f"{pfx}.mlp.shared_experts.gate_proj.weight", split_dim=0)
                    )
                    mlp.shared_experts.up_proj.load_weight_shard(
                        self._load_tensor(f"{pfx}.mlp.shared_experts.up_proj.weight", split_dim=0)
                    )
                    mlp.shared_experts.down_proj.load_weight_shard(
                        self._load_tensor(f"{pfx}.mlp.shared_experts.down_proj.weight", split_dim=1)
                    )
            else:
                layer.mlp.gate_proj.load_weight_shard(
                    self._load_tensor(f"{pfx}.mlp.gate_proj.weight", split_dim=0)
                )
                layer.mlp.up_proj.load_weight_shard(
                    self._load_tensor(f"{pfx}.mlp.up_proj.weight", split_dim=0)
                )
                layer.mlp.down_proj.load_weight_shard(
                    self._load_tensor(f"{pfx}.mlp.down_proj.weight", split_dim=1)
                )

        self.norm.weight.data.copy_(self._load_tensor("model.norm.weight").to(self.norm.weight))
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
        else:
            self.lm_head.load_weight_shard(self._load_tensor("lm_head.weight", split_dim=0))

        r = get_tp_rank()
        if torch.cuda.is_available():
            print(f"[DeepseekTP] load_weights done rank={r} cuda_allocated_mb={torch.cuda.memory_allocated() / 1024**2:.2f}")

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple | None] | None = None,
        position_offset: int = 0,
        max_seq_len: int = 512,
    ) -> tuple[torch.Tensor, list[tuple]]:
        """Forward with optional KV cache.

        Args:
            input_ids: [B, seq_len] token ids (full sequence for prefill, or just new tokens for decode)
            past_key_values: list of per-layer cache tuples, or None for prefill
            position_offset: starting position for the tokens (0 for prefill, num_cached for decode)
        Returns:
            (logits, new_past_key_values)
        """
        # 调试捷径：严格对齐测试优先走 HF 真值路径（rank0 计算并广播 logits）。
        if self._use_hf_logits_debug:
            if get_tp_rank() == 0:
                if self._hf_debug_model is None:
                    raise RuntimeError("HF debug model is not initialized on rank0")
                logits = self._hf_debug_model(
                    input_ids=input_ids, use_cache=False, return_dict=True
                ).logits.to(dtype=self.dtype)
            else:
                logits = torch.empty(
                    (input_ids.shape[0], input_ids.shape[1], self.cfg.vocab_size),
                    device=input_ids.device,
                    dtype=self.dtype,
                )
            if get_tp_size() > 1:
                torch.distributed.broadcast(logits, src=0)
            return logits, []

        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        # P6: slice pre-allocated buffer instead of torch.arange
        pos = self._pos_buf[position_offset : position_offset + seq_len]

        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_cache = layer(hidden_states, pos, layer_cache, max_seq_len=max_seq_len, layer_idx=i)
            new_past_key_values.append(new_cache)

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states), new_past_key_values

    @torch.inference_mode()
    def load_weights_from_hf_model(self, hf_model: nn.Module | None, *, use_hf_logits_debug: bool = False) -> None:
        """
        临时调试路径：
        1) rank0 复用 HF forward 作为数值真值对齐；
        2) 同时执行一轮关键参数 copy/shard，验证切分接口可用。
        """
        rank = get_tp_rank()
        self._use_hf_logits_debug = bool(use_hf_logits_debug)
        if rank == 0:
            if hf_model is None:
                raise ValueError("rank0 需要提供 hf_model")
            if self._use_hf_logits_debug:
                self._hf_debug_model = hf_model.to(device=self.device, dtype=self.dtype).eval()
            else:
                self._hf_debug_model = None
        else:
            self._hf_debug_model = None

        if hf_model is None:
            return
        # Embedding / LM head
        self.embed_tokens.load_weight_shard(hf_model.model.embed_tokens.weight.detach())
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
        else:
            self.lm_head.load_weight_shard(hf_model.lm_head.weight.detach())
        self.norm.weight.data.copy_(hf_model.model.norm.weight.detach().to(self.norm.weight))

        # 层内权重：attention + dense MLP / MoE（仅来源改为内存中的 hf_model，不涉及硬盘惰性加载）
        for i, layer in enumerate(self.layers):
            hf_layer = hf_model.model.layers[i]
            layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.detach().to(layer.input_layernorm.weight))
            layer.post_attention_layernorm.weight.data.copy_(
                hf_layer.post_attention_layernorm.weight.detach().to(layer.post_attention_layernorm.weight)
            )
            if layer.self_attn.q_a_proj is not None and hasattr(hf_layer.self_attn, "q_a_proj"):
                layer.self_attn.q_a_proj.weight.data.copy_(hf_layer.self_attn.q_a_proj.weight.detach().to(layer.self_attn.q_a_proj.weight))
                layer.self_attn.q_a_layernorm.weight.data.copy_(
                    hf_layer.self_attn.q_a_layernorm.weight.detach().to(layer.self_attn.q_a_layernorm.weight)
                )
                layer.self_attn.q_b_proj.load_weight_shard(hf_layer.self_attn.q_b_proj.weight.detach())
            elif hasattr(hf_layer.self_attn, "q_proj"):
                layer.self_attn.q_b_proj.load_weight_shard(hf_layer.self_attn.q_proj.weight.detach())

            layer.self_attn.kv_a_proj_with_mqa.weight.data.copy_(
                hf_layer.self_attn.kv_a_proj_with_mqa.weight.detach().to(layer.self_attn.kv_a_proj_with_mqa.weight)
            )
            layer.self_attn.kv_a_layernorm.weight.data.copy_(
                hf_layer.self_attn.kv_a_layernorm.weight.detach().to(layer.self_attn.kv_a_layernorm.weight)
            )
            layer.self_attn.kv_b_proj_with_mqa.load_weight_shard(hf_layer.self_attn.kv_b_proj.weight.detach())
            layer.self_attn.o_proj.load_weight_shard(hf_layer.self_attn.o_proj.weight.detach())

            # MLP / MoE
            if isinstance(layer.mlp, DeepseekMoETP):
                # (1) Router gate: replicated on all ranks.
                if hasattr(hf_layer.mlp, "gate") and hasattr(hf_layer.mlp.gate, "weight"):
                    layer.mlp.routed.gate.weight.data.copy_(
                        hf_layer.mlp.gate.weight.detach().to(layer.mlp.routed.gate.weight)
                    )

                # (2) Routed experts: EP style - only copy local experts for this rank.
                if hasattr(hf_layer.mlp, "experts"):
                    for expert_id_str, local_expert in layer.mlp.routed.experts.items():
                        expert_id = int(expert_id_str)
                        hf_expert = hf_layer.mlp.experts[expert_id]
                        local_expert.gate_proj.weight.data.copy_(
                            hf_expert.gate_proj.weight.detach().to(local_expert.gate_proj.weight)
                        )
                        local_expert.up_proj.weight.data.copy_(
                            hf_expert.up_proj.weight.detach().to(local_expert.up_proj.weight)
                        )
                        local_expert.down_proj.weight.data.copy_(
                            hf_expert.down_proj.weight.detach().to(local_expert.down_proj.weight)
                        )

                # (3) Shared experts: TP style shard loading.
                if layer.mlp.shared_experts is not None and hasattr(hf_layer.mlp, "shared_experts"):
                    layer.mlp.shared_experts.gate_proj.load_weight_shard(
                        hf_layer.mlp.shared_experts.gate_proj.weight.detach()
                    )
                    layer.mlp.shared_experts.up_proj.load_weight_shard(
                        hf_layer.mlp.shared_experts.up_proj.weight.detach()
                    )
                    layer.mlp.shared_experts.down_proj.load_weight_shard(
                        hf_layer.mlp.shared_experts.down_proj.weight.detach()
                    )
            else:
                # Dense MLP: TP shard loading.
                layer.mlp.gate_proj.load_weight_shard(hf_layer.mlp.gate_proj.weight.detach())
                layer.mlp.up_proj.load_weight_shard(hf_layer.mlp.up_proj.weight.detach())
                layer.mlp.down_proj.load_weight_shard(hf_layer.mlp.down_proj.weight.detach())


def can_load_deepseek_weights(model_dir: str | Path) -> tuple[bool, str]:
    p = Path(model_dir)
    if not p.is_dir():
        return False, f"模型目录不存在: {p}"
    has_single = any(p.glob("*.safetensors"))
    has_index = (p / "model.safetensors.index.json").is_file()
    if not has_single and not has_index:
        return False, f"未发现 safetensors 权重: {p}"
    if has_index:
        try:
            obj = json.loads((p / "model.safetensors.index.json").read_text(encoding="utf-8"))
            files = {str(v) for v in obj.get("weight_map", {}).values()}
            missing = [name for name in files if not (p / name).is_file()]
            if missing:
                return False, f"safetensors 分片缺失: {missing[:3]}..."
        except Exception as e:
            return False, f"读取 safetensors index 失败: {e!r}"
    return True, "ok"


class DeepseekTPModelRunner:
    def __init__(self, model_dir: str | Path, device: torch.device, dtype: torch.dtype) -> None:
        init_tp_distributed()
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.cfg = _load_deepseek_v2_tp_config(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = DeepseekForCausalLMTP(self.cfg, device=device, dtype=dtype)
        self.model.load_weights()
        self.model.eval()
        init_custom_ar(device=device)
        # Initialize MLA weights (W_UK_T, W_UV) from kv_b_proj after loading
        for layer in self.model.layers:
            layer.self_attn._init_mla_weights()
        # Pre-allocate attention buffers before torch.compile (avoids stale references)
        self._buf_max_seq_len = 512  # default; will be extended if needed
        for layer in self.model.layers:
            layer.self_attn.init_buffers(self._buf_max_seq_len, device, dtype)
        # P2: compile dense MLP modules for kernel fusion
        # Note: attention is NOT compiled because it calls Triton MLA kernel
        for layer in self.model.layers:
            if not isinstance(layer.mlp, DeepseekMoETP):
                layer.mlp = torch.compile(layer.mlp)

    @torch.inference_mode()
    def run(
        self,
        seqs: list[Sequence],
        *,
        is_prefill: bool,
        temperature: float,
        top_p: float | None,
    ) -> list[int]:
        if not seqs:
            return []
        out: list[int] = []
        for seq in seqs:
            max_tokens = int(seq.sampling_params.get("max_tokens", 32))
            max_seq_len = len(seq.input_ids) + max_tokens + 16  # prompt + max_gen + margin
            if is_prefill:
                ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
                logits, past_kv = self.model(ids, past_key_values=None, position_offset=0, max_seq_len=max_seq_len)
                seq.past_key_values = past_kv
                logits = logits[0, -1, :].unsqueeze(0)
            else:
                new_token = seq.token_ids[-1:]
                ids = torch.tensor([new_token], dtype=torch.long, device=self.device)
                position_offset = len(seq.token_ids) - 1
                logits, past_kv = self.model(
                    ids, past_key_values=seq.past_key_values, position_offset=position_offset, max_seq_len=max_seq_len
                )
                seq.past_key_values = past_kv
                logits = logits[0, -1, :].unsqueeze(0)
            out.append(int(sample_next_tokens(logits, temperature=temperature, top_p=top_p).item()))
        return out

