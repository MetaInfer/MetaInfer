"""Qwen3.5 MoE 模型实现 (单卡推理)。

架构特性:
- 混合注意力: Gated DeltaNet (线性注意力) + 标准 softmax 注意力
- MoE: 256 experts (每 token 激活 8 个) + shared expert
- Partial RoPE: 只有 head_dim 的 25% 使用旋转位置编码
- 输出门控: full attention 用 sigmoid gate, linear attention 用 gated RMSNorm
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from engine.sampler import sample_next_tokens
from engine.structs import Sequence

# ─── Config ──────────────────────────────────────────────────────────────


@dataclass
class Qwen35MoeConfig:
    model_dir: Path
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    partial_rotary_factor: float
    attn_output_gate: bool
    full_attention_interval: int
    layer_types: list[str]
    # MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    # Linear attention
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1


def load_config(model_dir: str | Path) -> Qwen35MoeConfig:
    p = Path(model_dir)
    raw = json.loads((p / "config.json").read_text(encoding="utf-8"))
    tc = raw.get("text_config", raw)

    hs = int(tc["hidden_size"])
    nah = int(tc["num_attention_heads"])
    default_head_dim = hs // nah
    num_layers = int(tc["num_hidden_layers"])
    full_interval = int(tc.get("full_attention_interval", 4))

    layer_types_list = tc.get("layer_types")
    if not layer_types_list:
        layer_types_list = [
            "full_attention" if i % full_interval == full_interval - 1 else "linear_attention"
            for i in range(num_layers)
        ]

    default_inter = hs * 4
    moe_inter = int(tc.get("moe_intermediate_size", default_inter))
    shared_inter = int(tc.get("shared_expert_intermediate_size", default_inter))

    return Qwen35MoeConfig(
        model_dir=p,
        hidden_size=hs,
        intermediate_size=int(tc.get("intermediate_size", default_inter)),
        num_hidden_layers=num_layers,
        num_attention_heads=nah,
        num_key_value_heads=int(tc.get("num_key_value_heads", nah)),
        head_dim=int(tc.get("head_dim", default_head_dim)),
        vocab_size=int(tc["vocab_size"]),
        rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        rope_theta=float(tc.get("rope_theta", 10000000.0)),
        tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
        partial_rotary_factor=float(tc.get("partial_rotary_factor", 1.0)),
        attn_output_gate=bool(tc.get("attn_output_gate", False)),
        full_attention_interval=full_interval,
        layer_types=layer_types_list,
        num_experts=int(tc.get("num_experts", 0)),
        num_experts_per_tok=int(tc.get("num_experts_per_tok", 1)),
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        linear_num_key_heads=int(tc.get("linear_num_key_heads", nah)),
        linear_num_value_heads=int(tc.get("linear_num_value_heads", nah)),
        linear_key_head_dim=int(tc.get("linear_key_head_dim", default_head_dim)),
        linear_value_head_dim=int(tc.get("linear_value_head_dim", default_head_dim)),
        linear_conv_kernel_dim=int(tc.get("linear_conv_kernel_dim", 4)),
    )


# ─── RMSNorm variants ───────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """(1 + weight) * rsqrt(var + eps) * x — Qwen3.5 MoE 格式。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return ((1.0 + self.weight) * x).to(input_dtype)


class RMSNormGated(nn.Module):
    """RMSNorm(x) * silu(gate) — 线性注意力层输出归一化。

    与 RMSNorm 不同，这里直接使用 weight（不加 1），与 HF transformers 对齐。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


# ─── Partial RoPE ────────────────────────────────────────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def _precompute_rope_inv_freq(theta: float, rotary_dim: int, device: torch.device) -> torch.Tensor:
    """Precompute inverse frequency for partial RoPE."""
    freqs = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim
    return 1.0 / (theta**freqs)


@torch.no_grad()
def precompute_rope_table(
    theta: float,
    rotary_dim: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin table for partial RoPE.

    Returns (cos_table, sin_table) each of shape (1, max_seq_len, 1, rotary_dim).
    """
    inv_freq = _precompute_rope_inv_freq(theta, rotary_dim, device)
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, rotary_dim//2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, rotary_dim)
    cos_table = emb.cos().unsqueeze(0).unsqueeze(2).to(dtype)  # (1, max_len, 1, rotary_dim)
    sin_table = emb.sin().unsqueeze(0).unsqueeze(2).to(dtype)
    return cos_table, sin_table


def apply_rope_from_table(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using precomputed cos/sin table.

    cos/sin: (1, S, 1, rotary_dim) — sliced from the precomputed table.
    q, k: (B, S, H, D) — will only rotate the first rotary_dim dims.
    """
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rotated = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rotated = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_rotated, q_pass], dim=-1), torch.cat([k_rotated, k_pass], dim=-1)


def apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只对 head_dim 的前 rotary_dim 维应用 RoPE，其余维度直接透传。

    inv_freq: precomputed via _precompute_rope_inv_freq()
    """
    rotary_dim = inv_freq.shape[0] * 2
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    t = positions.to(torch.float32)
    if t.dim() == 1:
        freqs = torch.outer(t, inv_freq)
    elif t.shape[-1] == 1:
        # Decode fast path: (B, 1) positions → (B, 1, rotary_dim//2)
        freqs = t * inv_freq[None, None, :]
    else:
        b = t.shape[0]
        freqs = (inv_freq[None, :, None].expand(b, -1, 1) @ t[:, None, :]).transpose(1, 2)

    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(2) if emb.dim() == 2 else emb.cos().unsqueeze(2)
    sin = emb.sin().unsqueeze(0).unsqueeze(2) if emb.dim() == 2 else emb.sin().unsqueeze(2)
    cos, sin = cos.to(q_rot.dtype), sin.to(q_rot.dtype)

    q_rotated = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rotated = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_rotated, q_pass], dim=-1), torch.cat([k_rotated, k_pass], dim=-1)


# ─── Full Attention (with output gating + GQA) ──────────────────────────


class Qwen35MoeFullAttention(nn.Module):
    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim**-0.5
        self.rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
        self.rope_theta = cfg.rope_theta
        self.attn_output_gate = cfg.attn_output_gate

        q_out = cfg.num_attention_heads * cfg.head_dim
        if self.attn_output_gate:
            q_out *= 2
        kv_out = cfg.num_key_value_heads * cfg.head_dim

        # Fused QKV projection: q + k + v in one matmul
        self.qkv_proj = nn.Linear(cfg.hidden_size, q_out + 2 * kv_out, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

        # Precompute RoPE inv_freq (lazy init on first forward)
        self._rope_inv_freq: torch.Tensor | None = None

        # Store split sizes for fused projection
        self._q_dim = q_out
        self._kv_dim = kv_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_len: int,
    ) -> tuple[torch.Tensor, int]:
        B, S, _ = hidden_states.shape

        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        q_out_dim = self.head_dim * 2 if self.attn_output_gate else self.head_dim
        q = qkv[..., : self._q_dim].view(B, S, self.num_heads, q_out_dim)
        k = qkv[..., self._q_dim : self._q_dim + self._kv_dim].view(
            B, S, self.num_kv_heads, self.head_dim
        )
        v = qkv[..., self._q_dim + self._kv_dim :].view(B, S, self.num_kv_heads, self.head_dim)

        gate = None
        if self.attn_output_gate:
            gate = q[..., self.head_dim :]
            q = q[..., : self.head_dim]

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self._rope_inv_freq is None:
            self._rope_inv_freq = _precompute_rope_inv_freq(
                self.rope_theta, self.rotary_dim, hidden_states.device
            )
        q, k = apply_partial_rope(q, k, position_ids, self._rope_inv_freq)

        key_cache[:, cache_len : cache_len + S] = k
        value_cache[:, cache_len : cache_len + S] = v
        new_len = cache_len + S

        k_full = key_cache[:, :new_len]
        v_full = value_cache[:, :new_len]

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(repeat, dim=2)
            v_full = v_full.repeat_interleave(repeat, dim=2)

        q = q.permute(0, 2, 1, 3)
        k_full = k_full.permute(0, 2, 1, 3)
        v_full = v_full.permute(0, 2, 1, 3)

        is_causal = S > 1
        out = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=is_causal, scale=self.scaling
        )
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, -1)

        if self.attn_output_gate and gate is not None:
            out = out * gate.reshape(B, S, -1).sigmoid()

        return self.o_proj(out), new_len


# ─── Gated DeltaNet kernels (pure PyTorch) ───────────────────────────────


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


@torch.inference_mode()
def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet chunk parallel prefill — adapted from HF transformers.

    Uses WY representation for intra-chunk attention and recurrent updates
    for inter-chunk state. All computation in float32 for precision.

    q: (B, S, H, D_k),  k: (B, S, H, D_k),  v: (B, S, H, D_v)
    g: (B, S, H),  beta: (B, S, H)
    """
    initial_dtype = query.dtype
    # QK L2 norm + scaling done inside
    query = _l2norm(query.float(), dim=-1)
    key = _l2norm(key.float(), dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]  # (B, H, S, D)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay — cumulative sum of g, then compute exp(g_i - g_j)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value).float()
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state.to(initial_dtype)


@torch.inference_mode()
def torch_recurrent_gated_delta_rule_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet single-token decode — fast path for S=1.

    Avoids loop overhead, unnecessary indexing, and extra reshapes.
    All inputs have S=1 dimension; output is squeezed to (B, H, V).

    q: (B, 1, H, D), k: (B, 1, H, D), v: (B, 1, H, V)
    g: (B, 1, H), beta: (B, 1, H)
    initial_state: (B, H, D, V)
    Returns: (output: (B, 1, H, V), new_state: (B, H, D, V))
    """
    input_dtype = q.dtype
    B, _, H, D = q.shape
    V_dim = v.shape[-1]
    BH = B * H
    scale = D**-0.5

    # Squeeze S=1, go to float32, L2 norm + scale
    q = _l2norm(q.squeeze(1).float(), dim=-1) * scale  # (B, H, D)
    k = _l2norm(k.squeeze(1).float(), dim=-1)  # (B, H, D)
    v = v.squeeze(1).float()  # (B, H, V)
    g_decay = g.squeeze(1).float().exp()  # (B, H)
    beta_sq = beta.squeeze(1).float()  # (B, H)

    # Flatten B*H for batched matmul
    q_flat = q.reshape(BH, 1, D)
    k_flat = k.reshape(BH, 1, D)
    v_flat = v.reshape(BH, 1, V_dim)
    g_flat = g_decay.reshape(BH, 1, 1)
    beta_flat = beta_sq.reshape(BH, 1, 1)

    S_state = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(B, H, D, V_dim, device=q.device, dtype=torch.float32)
    )
    S_flat = S_state.reshape(BH, D, V_dim)

    S_flat = S_flat * g_flat
    kv_mem = torch.bmm(k_flat, S_flat)  # (BH, 1, V)
    delta = (v_flat - kv_mem) * beta_flat
    S_flat = S_flat + torch.bmm(k_flat.transpose(-1, -2), delta)  # (BH, D, V)
    out_flat = torch.bmm(q_flat, S_flat)  # (BH, 1, V)

    output = out_flat.reshape(B, H, V_dim).unsqueeze(1)  # (B, 1, H, V)
    S_state = S_flat.reshape(B, H, D, V_dim)
    return output.to(input_dtype), S_state.to(input_dtype)


@torch.inference_mode()
def torch_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet 递归计算 — 逐 token 更新循环状态 S。

    内部使用 float32 计算。QK L2 norm + scaling 在内部完成。
    S=1 时委托给 single-token fast path。

    q: (B, S, H_v, D_k),  k: (B, S, H_v, D_k),  v: (B, S, H_v, D_v)
    g: (B, S, H_v) — 衰减 logits (负值, 将 exp 得到 0~1 衰减因子)
    beta: (B, S, H_v) — 更新率 (已 sigmoid)
    initial_state: (B, H_v, D_k, D_v) or None
    """
    if q.shape[1] == 1:
        return torch_recurrent_gated_delta_rule_single(q, k, v, g, beta, initial_state)

    B, S, H, D = q.shape
    V_dim = v.shape[-1]
    input_dtype = q.dtype

    # L2 norm + scaling (matching HF behavior)
    q = _l2norm(q.float(), dim=-1)
    k = _l2norm(k.float(), dim=-1)
    v = v.float()
    g = g.float()
    beta = beta.float()
    q = q * (D**-0.5)

    if initial_state is None:
        S_state = torch.zeros(B, H, D, V_dim, device=q.device, dtype=torch.float32)
    else:
        S_state = initial_state.float()

    output = torch.empty(B, S, H, V_dim, device=q.device, dtype=torch.float32)
    g_decay = g.exp()

    # Pre-flatten batch+head dims for bmm efficiency
    BH = B * H
    for t in range(S):
        q_t = q[:, t].reshape(BH, 1, D)  # (BH, 1, D)
        k_t = k[:, t].reshape(BH, 1, D)  # (BH, 1, D)
        v_t = v[:, t]  # (B, H, V)
        g_t = g_decay[:, t].reshape(BH, 1, 1)  # (BH, 1, 1)
        beta_t = beta[:, t].reshape(BH, 1, 1)  # (BH, 1, 1)

        S_flat = S_state.reshape(BH, D, V_dim)
        S_flat = S_flat * g_t
        # kv_mem = k^T @ S → (BH, 1, V)
        kv_mem = torch.bmm(k_t, S_flat).reshape(B, H, V_dim)
        delta = (v_t - kv_mem).reshape(BH, 1, V_dim) * beta_t
        # S += k * delta^T → outer product
        S_flat = S_flat + torch.bmm(k_t.transpose(-1, -2), delta)  # (BH, D, V)
        S_state = S_flat.reshape(B, H, D, V_dim)
        # output = q^T @ S → (BH, 1, V)
        output[:, t] = torch.bmm(q_t, S_flat).reshape(B, H, V_dim)

    return output.to(input_dtype), S_state.to(input_dtype)


# ─── Linear Attention (Gated DeltaNet) ───────────────────────────────────


class Qwen35MoeGatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Fused input projection: [qkv | z | b | a] in one matmul
        self.fused_proj_dim = self.conv_dim + self.value_dim + self.num_v_heads * 2
        self.in_proj = nn.Linear(cfg.hidden_size, self.fused_proj_dim, bias=False)
        self.out_proj = nn.Linear(self.value_dim, cfg.hidden_size, bias=False)

        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, 1, self.conv_kernel_size))

        A = torch.rand(self.num_v_heads) * 16
        self.A_log = nn.Parameter(torch.log(A))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Cache exp(A_log) to avoid repeated exp() during decode
        self._A_exp_cache: torch.Tensor | None = None

        self.norm = RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps)

        # GQA ratio
        self.repeat_factor = self.num_v_heads // self.num_k_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape
        K = self.conv_kernel_size

        # Fused projection → split into qkv, z, b, a
        fused = self.in_proj(hidden_states)
        splits = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
        mixed_qkv_raw, z, b, a = torch.split(fused, splits, dim=-1)

        # Causal conv1d
        if S == 1 and conv_state is not None:
            x_flat = mixed_qkv_raw.squeeze(1)  # (B, conv_dim)
            window = torch.cat([conv_state, x_flat.unsqueeze(-1)], dim=-1)
            w = self.conv1d_weight.squeeze(1)
            mixed_qkv = (window * w.unsqueeze(0)).sum(dim=-1).unsqueeze(1)
            mixed_qkv = F.silu(mixed_qkv)
            conv_state = torch.cat([conv_state[..., 1:], x_flat.unsqueeze(-1)], dim=-1)
        else:
            x_t = mixed_qkv_raw.transpose(1, 2)
            x_t = F.pad(x_t, (K - 1, 0))
            mixed_qkv = F.silu(F.conv1d(x_t, self.conv1d_weight, groups=self.conv_dim)).transpose(
                1, 2
            )
            if S >= K - 1:
                conv_state = mixed_qkv_raw[:, -(K - 1) :].transpose(1, 2).contiguous()
            else:
                pad = torch.zeros(
                    B,
                    self.conv_dim,
                    K - 1 - S,
                    device=mixed_qkv_raw.device,
                    dtype=mixed_qkv_raw.dtype,
                )
                conv_state = torch.cat([pad, mixed_qkv_raw.transpose(1, 2)], dim=2).contiguous()

        # Split QKV
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.view(B, S, self.num_k_heads, self.head_k_dim)
        key = key.view(B, S, self.num_k_heads, self.head_k_dim)
        value = value.view(B, S, self.num_v_heads, self.head_v_dim)

        # GQA: expand key heads to match value heads
        if self.repeat_factor > 1:
            query = query.repeat_interleave(self.repeat_factor, dim=2)
            key = key.repeat_interleave(self.repeat_factor, dim=2)

        # Decay & beta
        beta = b.sigmoid()
        if self._A_exp_cache is None:
            self._A_exp_cache = self.A_log.float().exp()
        g = -self._A_exp_cache * F.softplus(a.float() + self.dt_bias)

        # Prefill: chunk parallel; Decode: recurrent (single token)
        if S > 1:
            output, recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g, beta, initial_state=None, chunk_size=64
            )
        else:
            output, recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g, beta, initial_state=recurrent_state
            )

        # Gated RMSNorm
        z = z.view(B, S, self.num_v_heads, self.head_v_dim)
        output = self.norm(output, z)

        output = output.view(B, S, -1)
        return self.out_proj(output), recurrent_state, conv_state


# ─── MoE ─────────────────────────────────────────────────────────────────


class Qwen35MoeExperts(nn.Module):
    """256 个 expert 的权重存储 (3D Parameters)。"""

    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.moe_intermediate_size = cfg.moe_intermediate_size
        self.hidden_size = cfg.hidden_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(cfg.num_experts, cfg.moe_intermediate_size * 2, cfg.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )


class Qwen35MoeTopKRouter(nn.Module):
    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(hidden_states)
        scores = F.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
        return top_scores, top_indices


class Qwen35MoeMLP(nn.Module):
    """Dense MLP — fused gate_up_proj for single matmul。"""

    def __init__(self, cfg: Qwen35MoeConfig, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Qwen35MoeSparseMoeBlock(nn.Module):
    def __init__(self, cfg: Qwen35MoeConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.experts = Qwen35MoeExperts(cfg)
        self.router = Qwen35MoeTopKRouter(cfg)
        self.shared_expert = Qwen35MoeMLP(cfg, cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        flat = hidden_states.view(-1, H)

        # Router
        top_scores, top_indices = self.router(flat)

        # Routed experts
        output = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            mask = top_indices == expert_idx
            token_ids, k_ids = torch.where(mask)
            if token_ids.numel() == 0:
                continue
            weights = top_scores[token_ids, k_ids]
            expert_input = flat[token_ids]
            gate_up = F.linear(expert_input, self.experts.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, self.experts.down_proj[expert_idx])
            output[token_ids] += weights.unsqueeze(-1) * expert_out

        # Shared expert with sigmoid gate
        shared_out = self.shared_expert(flat)
        shared_gate = self.shared_expert_gate(flat).sigmoid()
        output = output + shared_gate * shared_out

        return output.view(B, S, H)


# ─── Decoder Layer ───────────────────────────────────────────────────────


class Qwen35MoeDecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen35MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        if self.layer_type == "full_attention":
            self.self_attn = Qwen35MoeFullAttention(cfg)
        else:
            self.self_attn = Qwen35MoeGatedDeltaNet(cfg)

        if cfg.is_moe:
            self.mlp = Qwen35MoeSparseMoeBlock(cfg)
        else:
            self.mlp = Qwen35MoeMLP(cfg, cfg.intermediate_size)

    def forward(self, hidden_states: torch.Tensor, kw: dict) -> torch.Tensor:
        h = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            h, new_len = self.self_attn(
                h, kw["position_ids"], kw["key_cache"], kw["value_cache"], kw["cache_len"]
            )
            kw["cache_len"] = new_len
        else:
            h, kw["recurrent_state"], kw["conv_state"] = self.self_attn(
                h, kw.get("recurrent_state"), kw.get("conv_state")
            )

        hidden_states = hidden_states + h
        h2 = self.post_attention_layernorm(hidden_states)
        h2 = self.mlp(h2)
        hidden_states = hidden_states + h2
        return hidden_states


# ─── Full Model ──────────────────────────────────────────────────────────


class Qwen35MoeForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen35MoeConfig, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        self.device = device
        self.dtype = dtype

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen35MoeDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.to(device=device, dtype=dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        """随机初始化参数（权重加载前使用，确保 forward 不会因全零权重而出错）。"""
        for name, param in self.named_parameters():
            if "A_log" in name or "dt_bias" in name:
                continue  # 这些有专门的初始化
            if param.dim() >= 2:
                torch.nn.init.normal_(param, std=0.02)
            elif param.dim() == 1 and "weight" in name:
                torch.nn.init.ones_(param)  # RMSNorm weight 初始化为 1（会被 load_weights 覆盖）

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, layer_kwargs: list[dict]) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, layer_kwargs[i])
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


# ─── Weight Loading ──────────────────────────────────────────────────────


def _resolve_weight_map(model_dir: Path) -> dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        obj = json.loads(index_file.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in obj.get("weight_map", {}).items()}
    single = model_dir / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    return {}


def _load_tensor(weight_map: dict[str, str], model_dir: Path, key: str) -> torch.Tensor:
    fname = weight_map.get(key)
    if fname is None:
        raise KeyError(f"Missing tensor {key} in safetensors index")
    fp = model_dir / fname
    with safe_open(str(fp), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _detect_prefix(weight_map: dict[str, str]) -> str:
    for k in weight_map:
        if k.startswith("model.language_model."):
            return "model.language_model."
    for k in weight_map:
        if k.startswith("model.") and "embed_tokens" in k:
            return "model."
    return ""


def load_weights(model: Qwen35MoeForCausalLM) -> None:
    cfg = model.cfg
    weight_map = _resolve_weight_map(cfg.model_dir)
    prefix = _detect_prefix(weight_map)

    def _w(key: str) -> torch.Tensor:
        full_key = prefix + key
        try:
            tensor = _load_tensor(weight_map, cfg.model_dir, full_key)
        except KeyError:
            if prefix:
                tensor = _load_tensor(weight_map, cfg.model_dir, key)
            else:
                raise
        return tensor.to(model.dtype)

    # Embedding
    model.embed_tokens.weight.data.copy_(_w("embed_tokens.weight"))

    # Layers
    for i, layer in enumerate(model.layers):
        pfx = f"layers.{i}"
        layer.input_layernorm.weight.data.copy_(_w(f"{pfx}.input_layernorm.weight"))
        layer.post_attention_layernorm.weight.data.copy_(
            _w(f"{pfx}.post_attention_layernorm.weight")
        )

        attn = layer.self_attn
        if layer.layer_type == "full_attention":
            q_w = _w(f"{pfx}.self_attn.q_proj.weight")
            k_w = _w(f"{pfx}.self_attn.k_proj.weight")
            v_w = _w(f"{pfx}.self_attn.v_proj.weight")
            attn.qkv_proj.weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            attn.o_proj.weight.data.copy_(_w(f"{pfx}.self_attn.o_proj.weight"))
            attn.q_norm.weight.data.copy_(_w(f"{pfx}.self_attn.q_norm.weight"))
            attn.k_norm.weight.data.copy_(_w(f"{pfx}.self_attn.k_norm.weight"))
        else:
            attn_prefix = f"{pfx}.linear_attn"
            qkv_w = _w(f"{attn_prefix}.in_proj_qkv.weight")
            z_w = _w(f"{attn_prefix}.in_proj_z.weight")
            b_w = _w(f"{attn_prefix}.in_proj_b.weight")
            a_w = _w(f"{attn_prefix}.in_proj_a.weight")
            attn.in_proj.weight.data.copy_(torch.cat([qkv_w, z_w, b_w, a_w], dim=0))
            attn.out_proj.weight.data.copy_(_w(f"{attn_prefix}.out_proj.weight"))
            attn.conv1d_weight.data.copy_(_w(f"{attn_prefix}.conv1d.weight"))
            attn.A_log.data.copy_(_w(f"{attn_prefix}.A_log"))
            attn.dt_bias.data.copy_(_w(f"{attn_prefix}.dt_bias"))
            attn.norm.weight.data.copy_(_w(f"{attn_prefix}.norm.weight"))

        # MLP
        mlp = layer.mlp
        if cfg.is_moe:
            mlp.router.gate.weight.data.copy_(_w(f"{pfx}.mlp.gate.weight"))
            mlp.experts.gate_up_proj.data.copy_(_w(f"{pfx}.mlp.experts.gate_up_proj"))
            mlp.experts.down_proj.data.copy_(_w(f"{pfx}.mlp.experts.down_proj"))
            mlp.shared_expert.gate_proj.weight.data.copy_(
                _w(f"{pfx}.mlp.shared_expert.gate_proj.weight")
            )
            mlp.shared_expert.up_proj.weight.data.copy_(
                _w(f"{pfx}.mlp.shared_expert.up_proj.weight")
            )
            mlp.shared_expert.down_proj.weight.data.copy_(
                _w(f"{pfx}.mlp.shared_expert.down_proj.weight")
            )
            mlp.shared_expert_gate.weight.data.copy_(_w(f"{pfx}.mlp.shared_expert_gate.weight"))
        else:
            gate_w = _w(f"{pfx}.mlp.gate_proj.weight")
            up_w = _w(f"{pfx}.mlp.up_proj.weight")
            mlp.gate_up_proj.weight.data.copy_(torch.cat([gate_w, up_w], dim=0))
            mlp.down_proj.weight.data.copy_(_w(f"{pfx}.mlp.down_proj.weight"))

    model.norm.weight.data.copy_(_w("norm.weight"))
    if cfg.tie_word_embeddings:
        model.lm_head.weight.data.copy_(model.embed_tokens.weight.data)
    else:
        model.lm_head.weight.data.copy_(_w("lm_head.weight"))

    print(f"[Qwen35Moe] weights loaded from {cfg.model_dir}")


# ─── Inference State ─────────────────────────────────────────────────────


@dataclass
class LayerState:
    layer_type: str

    # Full attention KV cache
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None
    cache_len: int = 0

    # Linear attention recurrent state
    recurrent: torch.Tensor | None = None
    conv: torch.Tensor | None = None


def create_layer_states(
    cfg: Qwen35MoeConfig,
    device: torch.device,
    dtype: torch.dtype,
    max_seq_len: int = 8192,
) -> list[LayerState]:
    # Linear attention recurrent state uses float32 for precision;
    # full attention KV cache uses model dtype (fp16 is fine for softmax attention).
    linear_dtype = torch.float32 if dtype != torch.float32 else dtype
    states = []
    for lt in cfg.layer_types:
        if lt == "full_attention":
            states.append(
                LayerState(
                    layer_type="full_attention",
                    key_cache=torch.zeros(
                        1,
                        max_seq_len,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    value_cache=torch.zeros(
                        1,
                        max_seq_len,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    cache_len=0,
                )
            )
        else:
            kv = cfg.linear_num_value_heads
            dk = cfg.linear_key_head_dim
            dv = cfg.linear_value_head_dim
            conv_dim = (
                cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
                + cfg.linear_num_value_heads * cfg.linear_value_head_dim
            )
            states.append(
                LayerState(
                    layer_type="linear_attention",
                    recurrent=torch.zeros(1, kv, dk, dv, device=device, dtype=linear_dtype),
                    conv=torch.zeros(
                        1, conv_dim, cfg.linear_conv_kernel_dim - 1, device=device, dtype=dtype
                    ),
                )
            )
    return states


def states_to_kwargs(states: list[LayerState]) -> list[dict]:
    return [
        {
            "position_ids": None,
            "key_cache": s.key_cache,
            "value_cache": s.value_cache,
            "cache_len": s.cache_len,
            "recurrent_state": s.recurrent,
            "conv_state": s.conv,
        }
        for s in states
    ]


def update_states_from_kwargs(states: list[LayerState], kwargs_list: list[dict]) -> None:
    for s, kw in zip(states, kwargs_list):
        if s.layer_type == "full_attention":
            s.cache_len = kw["cache_len"]
        else:
            s.recurrent = kw["recurrent_state"]
            s.conv = kw["conv_state"]


# ─── Model Runner ────────────────────────────────────────────────────────


class Qwen35MoeModelRunner:
    def __init__(
        self,
        model_dir: str | Path,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int = 8192,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        print(f"[Qwen35Moe] loading config from {self.model_dir}")
        self.cfg = load_config(self.model_dir)

        print("[Qwen35Moe] loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        print(f"[Qwen35Moe] tokenizer eos={eos}, pad={pad}")

        print(f"[Qwen35Moe] loading model dtype={dtype}, device={device}")
        self.model = Qwen35MoeForCausalLM(self.cfg, device=device, dtype=dtype)
        load_weights(self.model)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Qwen35Moe] model loaded, parameters={total_params:,}")

        self.eos_token_id = self.tokenizer.eos_token_id
        self._seq_states: dict[str, list[LayerState]] = {}

    def init_sequence(self, request_id: str) -> None:
        self._seq_states[request_id] = create_layer_states(
            self.cfg,
            self.device,
            self.dtype,
            self.max_seq_len,
        )

    def cleanup_sequence(self, request_id: str) -> None:
        self._seq_states.pop(request_id, None)

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
        next_tokens: list[int] = []
        for seq in seqs:
            rid = seq.request_id
            if rid not in self._seq_states:
                self.init_sequence(rid)

            states = self._seq_states[rid]
            kwargs_list = states_to_kwargs(states)

            ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
            position_ids = torch.arange(
                seq.total_tokens, device=self.device, dtype=torch.long
            ).unsqueeze(0)

            for kw in kwargs_list:
                kw["position_ids"] = position_ids

            logits = self.model(ids, kwargs_list)
            next_tok = int(
                sample_next_tokens(logits[:, -1, :], temperature=temperature, top_p=top_p).item()
            )
            next_tokens.append(next_tok)

            update_states_from_kwargs(states, kwargs_list)

            if is_prefill:
                print(
                    f"[Qwen35Moe] prefill req={rid} len={seq.total_tokens} first_token={next_tok}"
                )

        return next_tokens
