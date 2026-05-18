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
    init_tp_distributed,
    is_tp_enabled,
)


@dataclass
class QwenTPConfig:
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


def _load_qwen_tp_config(model_dir: str | Path) -> QwenTPConfig:
    p = Path(model_dir)
    cfg = AutoConfig.from_pretrained(str(p), trust_remote_code=True, local_files_only=True)
    return QwenTPConfig(
        model_dir=p,
        hidden_size=int(cfg.hidden_size),
        intermediate_size=int(cfg.intermediate_size),
        num_hidden_layers=int(cfg.num_hidden_layers),
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(cfg.num_key_value_heads),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        vocab_size=int(cfg.vocab_size),
        rms_norm_eps=float(cfg.rms_norm_eps),
        rope_theta=float(getattr(cfg, "rope_theta", 1000000.0)),
        tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
    )


class RMSNorm(nn.Module):
    """与 transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm 一致：在 fp32 中算方差/归一化，再回写激活 dtype。"""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # 与 HF Qwen3 apply_rotary_pos_emb / rotate_half 一致：前一半/后一半，而非奇偶维交错
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE 与 Qwen3RotaryEmbedding + apply_rotary_pos_emb 对齐（interleaved rotate_half 会破坏 Qwen3 的 logits）。"""
    dim = q.shape[-1]
    device = q.device
    input_dtype = q.dtype
    # modeling_rope_utils._compute_default_rope_parameters: inv_freq 与 arange(0, dim, 2) / dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = positions.to(torch.float32)
    if t.dim() == 1:
        freqs = torch.outer(t, inv_freq)
    else:
        # 与 Qwen3RotaryEmbedding: inv_freq[None, :, None].expand(B, -1, 1) @ pos[:, None, :]
        b = t.shape[0]
        inv_e = inv_freq[None, :, None].expand(b, -1, 1)
        t_e = t[:, None, :]
        freqs = (inv_e @ t_e).transpose(1, 2)
    with torch.autocast(device_type=device.type if device.type in ("cuda", "cpu", "mps") else "cpu", enabled=False):
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
    else:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    cos, sin = cos.to(input_dtype), sin.to(input_dtype)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class QwenAttentionTP(nn.Module):
    def __init__(self, cfg: QwenTPConfig):
        super().__init__()
        self.tp = get_tp_size()
        self.rank = get_tp_rank()
        self.total_num_heads = cfg.num_attention_heads
        self.total_num_kv_heads = cfg.num_key_value_heads
        ensure_divisible(self.total_num_heads, self.tp, name="num_attention_heads")
        self.num_heads = self.total_num_heads // self.tp
        if self.total_num_kv_heads >= self.tp:
            ensure_divisible(self.total_num_kv_heads, self.tp, name="num_key_value_heads")
            self.num_kv_heads = self.total_num_kv_heads // self.tp
            self.kv_head_replica = 1
            self.kv_group_rank = self.rank
        else:
            ensure_divisible(self.tp, self.total_num_kv_heads, name="tp_size/num_key_value_heads")
            self.num_kv_heads = 1
            self.kv_head_replica = self.tp // self.total_num_kv_heads
            self.kv_group_rank = self.rank // self.kv_head_replica
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = cfg.rope_theta

        self.q_proj = ColumnParallelLinear(cfg.hidden_size, self.total_num_heads * self.head_dim, bias=False, gather_output=False)
        self.k_proj = ColumnParallelLinear(cfg.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False, gather_output=False)
        self.v_proj = ColumnParallelLinear(cfg.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False, gather_output=False)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

        # Pre-allocated cu_seqlens buffers (avoid per-step torch.tensor allocation)
        self.register_buffer("_cu_q", torch.tensor([0, 0], dtype=torch.int32), persistent=False)
        self.register_buffer("_cu_k", torch.tensor([0, 0], dtype=torch.int32), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor, int] | None = None,
        max_seq_len: int = 512,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
        """Forward with pre-allocated KV cache buffer.

        past_key_values: (k_buf, v_buf, kv_len) with pre-allocated buffers [B, max_seq_len, num_kv_heads, head_dim]
        Returns: (output, new_cache)
        """
        bsz, seqlen, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rope(q, k, positions, self.rope_theta)

        use_gqa = (self.num_kv_heads != self.num_heads)

        if past_key_values is None:
            # Prefill: allocate buffers and slice (efficient for long prompts)
            k_buf = torch.zeros(bsz, max_seq_len, self.num_kv_heads, self.head_dim, device=k.device, dtype=k.dtype)
            v_buf = torch.zeros(bsz, max_seq_len, self.num_kv_heads, self.head_dim, device=v.device, dtype=v.dtype)
            k_buf[:, :seqlen] = k
            v_buf[:, :seqlen] = v
            kv_len = seqlen

            # flash-attn: reshape to (total, heads, headdim)
            q_fa = q.reshape(seqlen, self.num_heads, self.head_dim)
            k_fa = k_buf[0, :kv_len]
            v_fa = v_buf[0, :kv_len]
            self._cu_q[1] = seqlen
            out = flash_attn_varlen_func(
                q_fa, k_fa, v_fa,
                cu_seqlens_q=self._cu_q, cu_seqlens_k=self._cu_q,
                max_seqlen_q=seqlen, max_seqlen_k=kv_len,
                causal=True, softmax_scale=self.scaling,
            )
            out = out.reshape(bsz, seqlen, self.q_size)
        else:
            # Decode: append new KV, then use flash-attn on full buffer (fixed shape for torch.compile)
            k_buf, v_buf, kv_len = past_key_values
            k_buf[:, kv_len:kv_len + seqlen] = k
            v_buf[:, kv_len:kv_len + seqlen] = v
            kv_len = kv_len + seqlen

            # flash-attn: full buffer + cu_seqlens_k marks valid boundary (no dynamic slice)
            q_fa = q.reshape(seqlen, self.num_heads, self.head_dim)
            k_fa = k_buf[0]  # [max_seq_len, H, D] — fixed shape
            v_fa = v_buf[0]  # [max_seq_len, H, D] — fixed shape
            self._cu_q[1] = seqlen
            self._cu_k[1] = kv_len
            out = flash_attn_varlen_func(
                q_fa, k_fa, v_fa,
                cu_seqlens_q=self._cu_q, cu_seqlens_k=self._cu_k,
                max_seqlen_q=seqlen, max_seqlen_k=kv_len,
                causal=False, softmax_scale=self.scaling,
            )
            out = out.reshape(bsz, seqlen, self.q_size)

        new_cache = (k_buf, v_buf, kv_len)
        return self.o_proj(out), new_cache


class QwenMLPTP(nn.Module):
    def __init__(self, cfg: QwenTPConfig):
        super().__init__()
        self.tp = get_tp_size()
        ensure_divisible(cfg.intermediate_size, self.tp, name="intermediate_size")
        self.local_intermediate = cfg.intermediate_size // self.tp
        # P5a: merge gate+up into single GEMM, silu_and_mul fused
        from engine.tp_layers.linear import MergedColumnParallelLinear
        self.gate_up_proj = MergedColumnParallelLinear(
            cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False
        )
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        d = gate_up.shape[-1] // 2
        h = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        return self.down_proj(h)


class QwenDecoderLayerTP(nn.Module):
    def __init__(self, cfg: QwenTPConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = QwenAttentionTP(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = QwenMLPTP(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: tuple | None = None,
        max_seq_len: int = 512,
    ) -> tuple[torch.Tensor, tuple]:
        h = self.input_layernorm(hidden_states)
        h, new_cache = self.self_attn(h, positions, past_key_values, max_seq_len=max_seq_len)
        hidden_states = hidden_states + h
        h2 = self.post_attention_layernorm(hidden_states)
        h2 = self.mlp(h2)
        return hidden_states + h2, new_cache


class QwenForCausalLMTP(nn.Module):
    def __init__(self, cfg: QwenTPConfig, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        self.device = device
        self.dtype = dtype
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([QwenDecoderLayerTP(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)
        self.to(device=device, dtype=dtype)

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
            input_ids: [B, seq_len] token ids
            past_key_values: list of per-layer cache tuples, or None for prefill
            position_offset: starting position for the tokens
        Returns:
            (logits, new_past_key_values)
        """
        rank = get_tp_rank()
        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        pos = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device, dtype=torch.long)

        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_cache = layer(hidden_states, pos, layer_cache, max_seq_len=max_seq_len)
            new_past_key_values.append(new_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_past_key_values

    def _resolve_weight_map(self) -> dict[str, str]:
        index_file = self.cfg.model_dir / "model.safetensors.index.json"
        if not index_file.is_file():
            safes = sorted(self.cfg.model_dir.glob("*.safetensors"))
            if len(safes) == 1:
                return {}
            raise FileNotFoundError(
                f"No model.safetensors.index.json found in {self.cfg.model_dir}, and no single safetensors file."
            )
        obj = json.loads(index_file.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in obj.get("weight_map", {}).items()}

    def _load_tensor(self, key: str, *, split_dim: int | None = None, allow_kv_replication: bool = False) -> torch.Tensor:
        weight_map = self._resolve_weight_map()
        if weight_map:
            fname = weight_map.get(key)
            if fname is None:
                raise KeyError(f"Missing tensor {key} in safetensors index")
            fp = self.cfg.model_dir / fname
        else:
            fp = next(self.cfg.model_dir.glob("*.safetensors"))
        if not fp.is_file():
            raise FileNotFoundError(f"Tensor file not found: {fp}")

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
            index = [slice(None)] * len(shape)
            index[split_dim] = slice(start, end)
            return sl[tuple(index)]

    def load_weights(self) -> None:
        self.embed_tokens.load_weight_shard(self._load_tensor("model.embed_tokens.weight", split_dim=0))
        for i, layer in enumerate(self.layers):
            pfx = f"model.layers.{i}"
            layer.input_layernorm.weight.data.copy_(
                self._load_tensor(f"{pfx}.input_layernorm.weight").to(layer.input_layernorm.weight)
            )
            layer.post_attention_layernorm.weight.data.copy_(
                self._load_tensor(f"{pfx}.post_attention_layernorm.weight").to(layer.post_attention_layernorm.weight)
            )
            layer.self_attn.q_norm.weight.data.copy_(
                self._load_tensor(f"{pfx}.self_attn.q_norm.weight").to(layer.self_attn.q_norm.weight)
            )
            layer.self_attn.k_norm.weight.data.copy_(
                self._load_tensor(f"{pfx}.self_attn.k_norm.weight").to(layer.self_attn.k_norm.weight)
            )
            layer.self_attn.q_proj.weight.data.copy_(
                self._load_tensor(f"{pfx}.self_attn.q_proj.weight", split_dim=0).to(layer.self_attn.q_proj.weight)
            )
            layer.self_attn.k_proj.weight.data.copy_(
                self._load_tensor(
                    f"{pfx}.self_attn.k_proj.weight",
                    split_dim=0,
                    allow_kv_replication=True,
                ).to(layer.self_attn.k_proj.weight)
            )
            layer.self_attn.v_proj.weight.data.copy_(
                self._load_tensor(
                    f"{pfx}.self_attn.v_proj.weight",
                    split_dim=0,
                    allow_kv_replication=True,
                ).to(layer.self_attn.v_proj.weight)
            )
            layer.self_attn.o_proj.weight.data.copy_(
                self._load_tensor(f"{pfx}.self_attn.o_proj.weight", split_dim=1).to(layer.self_attn.o_proj.weight)
            )
            g = self._load_tensor(f"{pfx}.mlp.gate_proj.weight", split_dim=0)
            u = self._load_tensor(f"{pfx}.mlp.up_proj.weight", split_dim=0)
            layer.mlp.gate_up_proj.load_weight_shard(g, u)
            layer.mlp.down_proj.weight.data.copy_(
                self._load_tensor(f"{pfx}.mlp.down_proj.weight", split_dim=1).to(layer.mlp.down_proj.weight)
            )

        self.norm.weight.data.copy_(self._load_tensor("model.norm.weight").to(self.norm.weight))
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
        else:
            self.lm_head.load_weight_shard(self._load_tensor("lm_head.weight", split_dim=0))

        rank = get_tp_rank()
        first = self.layers[0]
        last = self.layers[-1]
        print(
            f"[TP-Probe] rank={rank} first q_proj shape={tuple(first.self_attn.q_proj.weight.shape)} "
            f"device={first.self_attn.q_proj.weight.device}"
        )
        print(
            f"[TP-Probe] rank={rank} first gate_up shape={tuple(first.mlp.gate_up_proj.weight.shape)} "
            f"device={first.mlp.gate_up_proj.weight.device}"
        )
        print(
            f"[TP-Probe] rank={rank} last q_proj shape={tuple(last.self_attn.q_proj.weight.shape)} "
            f"device={last.self_attn.q_proj.weight.device}"
        )
        print(
            f"[TP-Probe] rank={rank} last gate_up shape={tuple(last.mlp.gate_up_proj.weight.shape)} "
            f"device={last.mlp.gate_up_proj.weight.device}"
        )
        if torch.cuda.is_available():
            print(f"[TP-Probe] rank={rank} cuda_allocated_mb={torch.cuda.memory_allocated()/1024**2:.2f}")


class QwenTPModelRunner:
    def __init__(self, model_dir: str | Path, device: torch.device, dtype: torch.dtype) -> None:
        init_tp_distributed()
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.cfg = _load_qwen_tp_config(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = QwenForCausalLMTP(self.cfg, device=device, dtype=dtype)
        self.model.load_weights()
        self.model.eval()
        # P2: compile attention and MLP modules for kernel fusion
        for layer in self.model.layers:
            layer.self_attn = torch.compile(layer.self_attn)
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
        next_tokens: list[int] = []
        for seq in seqs:
            max_tokens = int(seq.sampling_params.get("max_tokens", 32))
            max_seq_len = len(seq.input_ids) + max_tokens + 16
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
            next_tokens.append(int(sample_next_tokens(logits, temperature=temperature, top_p=top_p).item()))
        return next_tokens


def can_load_qwen_weights(model_dir: str | Path) -> tuple[bool, str]:
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

