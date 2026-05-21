from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from engine.sampler import sample_next_tokens
from engine.structs import Sequence
from engine.tp_layers import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVColumnParallelLinear,
    ParallelLMHead,
    RowParallelLinear,
    VocabParallelEmbedding,
    ensure_divisible,
    get_tp_rank,
    get_tp_size,
    init_custom_ar,
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
    max_position_embeddings: int
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
        max_position_embeddings=int(getattr(cfg, "max_position_embeddings", 32768)),
        tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
    )


# Shared RoPE cos/sin cache (de-duplicated across attention layers)
_cos_sin_cache_registry: dict[tuple, torch.Tensor] = {}


def _get_cos_sin_cache(max_pos: int, head_dim: int, rope_theta: float) -> torch.Tensor:
    from engine.kernels.vllm_wrappers import make_cos_sin_cache
    key = (max_pos, head_dim, rope_theta)
    if key not in _cos_sin_cache_registry:
        # Keep on CPU until model.to(device) moves it
        _cos_sin_cache_registry[key] = make_cos_sin_cache(
            max_pos, head_dim, rope_theta, dtype=torch.bfloat16, device='cpu',
        )
    return _cos_sin_cache_registry[key]


class RMSNorm(nn.Module):
    """RMSNorm with vLLM fused kernel backend.

    Dual interface (matching vLLM):
        forward(x)           -> rms_norm(x)              (no residual)
        forward(x, residual) -> fused_add_rms_norm       (residual + norm)

    Where fused_add_rms_norm does:
        1. residual = residual + x          (in-place)
        2. x = rms_norm(residual) * weight  (in-place)
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            out = torch.empty_like(x)
            from engine.kernels.vllm_wrappers import rms_norm
            rms_norm(out, x.contiguous(), self.weight, self.eps)
            return out
        else:
            from engine.kernels.vllm_wrappers import fused_add_rms_norm
            fused_add_rms_norm(x.contiguous(), residual.contiguous(), self.weight, self.eps)
            return x, residual


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

        self.qkv_proj = QKVColumnParallelLinear(
            cfg.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads,
            bias=False, gather_output=False,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

        # Pre-allocated cu_seqlens buffers (avoid per-step torch.tensor allocation)
        self.register_buffer("_cu_q", torch.tensor([0, 0], dtype=torch.int32), persistent=False)
        self.register_buffer("_cu_k", torch.tensor([0, 0], dtype=torch.int32), persistent=False)

        # Paged KV cache (vLLM flash layout: [num_blocks, block_size, Hk, D])
        # block_size=256 required by flash_attn_with_kvcache
        self._kv_block_size = 256
        # KV cache allocated on first prefill; stored on module for graph access
        self._key_cache: torch.Tensor | None = None
        self._value_cache: torch.Tensor | None = None
        self._block_table: torch.Tensor | None = None  # [1, max_blocks] int32
        self._slot_mapping: torch.Tensor | None = None  # [seqlen] int64 for prefill, [1] for decode
        self.register_buffer("_kv_len_gpu", torch.zeros(1, dtype=torch.int32), persistent=False)
        self.register_buffer("_slot_mapping_decode", torch.zeros(1, dtype=torch.int64), persistent=False)

        # Shared RoPE cos/sin cache (de-duplicated across layers via module-level cache)
        self._cos_sin_cache_cpu = _get_cos_sin_cache(
            cfg.max_position_embeddings, self.head_dim, cfg.rope_theta,
        )
        self._cos_sin_cache_gpu: torch.Tensor | None = None

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
        q, k, v = self.qkv_proj(hidden_states)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        # vLLM rotary_embedding: flatten to [num_tokens, heads, head_dim], in-place
        num_tokens = bsz * seqlen
        q_flat = q.reshape(num_tokens, self.num_heads, self.head_dim)
        k_flat = k.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        from engine.kernels.vllm_wrappers import rotary_embedding
        if self._cos_sin_cache_gpu is None or self._cos_sin_cache_gpu.device != q.device:
            self._cos_sin_cache_gpu = self._cos_sin_cache_cpu.to(device=q.device)
        rotary_embedding(
            positions, q_flat, k_flat, self.head_dim,
            self._cos_sin_cache_gpu, is_neox=True,
        )
        q = q_flat.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k_flat.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        use_gqa = (self.num_kv_heads != self.num_heads)

        if past_key_values is None:
            # Prefill: allocate paged KV cache + block table
            num_blocks = max(1, (max_seq_len + self._kv_block_size - 1) // self._kv_block_size)
            self._key_cache = torch.zeros(num_blocks, self._kv_block_size, self.num_kv_heads,
                                          self.head_dim, device=k.device, dtype=k.dtype)
            self._value_cache = torch.zeros(num_blocks, self._kv_block_size, self.num_kv_heads,
                                            self.head_dim, device=v.device, dtype=v.dtype)
            # block_table: [1, num_blocks] — sequential allocation
            self._block_table = torch.arange(num_blocks, dtype=torch.int32, device=k.device).unsqueeze(0)

            # Write prefill KV to paged cache via sequential slot_mapping
            slot_mapping = torch.arange(seqlen, dtype=torch.int64, device=k.device)
            k_flat = k.reshape(seqlen, self.num_kv_heads, self.head_dim)
            v_flat = v.reshape(seqlen, self.num_kv_heads, self.head_dim)
            self._key_cache.view(-1, self.num_kv_heads, self.head_dim)[slot_mapping] = k_flat
            self._value_cache.view(-1, self.num_kv_heads, self.head_dim)[slot_mapping] = v_flat
            kv_len = seqlen

            # Prefill attention with FA2 varlen
            q_fa = q.reshape(seqlen, self.num_heads, self.head_dim)
            self._cu_q[1] = seqlen
            k_fa = self._key_cache.reshape(-1, self.num_kv_heads, self.head_dim)[:kv_len]
            v_fa = self._value_cache.reshape(-1, self.num_kv_heads, self.head_dim)[:kv_len]
            out = flash_attn_varlen_func(
                q_fa, k_fa, v_fa,
                cu_seqlens_q=self._cu_q, cu_seqlens_k=self._cu_q,
                max_seqlen_q=seqlen, max_seqlen_k=kv_len,
                causal=True, softmax_scale=self.scaling,
            )
            out = out.reshape(bsz, seqlen, self.q_size)
        else:
            # Decode: _kv_len_gpu pre-set externally (both eager and graph paths)
            kv_len = past_key_values

            # Write KV using index_copy_ (graph-compatible)
            self._slot_mapping_decode[0] = self._kv_len_gpu[0]
            k_flat = k.reshape(1, self.num_kv_heads, self.head_dim)
            v_flat = v.reshape(1, self.num_kv_heads, self.head_dim)
            self._key_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
                0, self._slot_mapping_decode, k_flat)
            self._value_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
                0, self._slot_mapping_decode, v_flat)
            self._kv_len_gpu[0] += 1

            # Decode attention (paged, graph-compatible — kv_len_gpu as 1D tensor)
            q_kv = q.reshape(1, 1, self.num_heads, self.head_dim)
            out = flash_attn_with_kvcache(
                q_kv, self._key_cache, self._value_cache,
                cache_seqlens=self._kv_len_gpu, block_table=self._block_table,
                softmax_scale=self.scaling, causal=False,
            )
            out = out.reshape(bsz, seqlen, self.q_size)

        new_cache = kv_len + 1
        return self.o_proj(out), new_cache


class QwenMLPTP(nn.Module):
    def __init__(self, cfg: QwenTPConfig):
        super().__init__()
        self.tp = get_tp_size()
        ensure_divisible(cfg.intermediate_size, self.tp, name="intermediate_size")
        self.local_intermediate = cfg.intermediate_size // self.tp
        self.gate_up_proj = MergedColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)                                 # [B, S, 2*local_inter]
        out = torch.empty(x.shape[0], x.shape[1], self.local_intermediate,
                          dtype=x.dtype, device=x.device)
        from engine.kernels.vllm_wrappers import silu_and_mul
        silu_and_mul(out, gate_up)                                     # [B, S, local_inter]
        return self.down_proj(out)


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
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states, new_cache = self.self_attn(hidden_states, positions, past_key_values, max_seq_len=max_seq_len)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, new_cache


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

        # CUDA Graph for decode
        self._decode_graph: torch.cuda.CUDAGraph | None = None
        self._graph_input_ids: torch.Tensor | None = None
        self._graph_pos: torch.Tensor | None = None
        self._graph_logits: torch.Tensor | None = None
        self._cuda_graph_enabled = os.environ.get('META_INFER_CUDA_GRAPH', '1') == '1'

    # ---- CUDA Graph support ----
    def _set_kv_len_for_layers(self, kv_lens: list[int]) -> None:
        for i, layer in enumerate(self.layers):
            layer.self_attn._kv_len_gpu[0] = kv_lens[i]

    @property
    def has_decode_graph(self) -> bool:
        return self._decode_graph is not None

    def init_decode_graph(self, input_ids: torch.Tensor, pos: torch.Tensor, kv_lens: list[int], max_seq_len: int) -> None:
        if not self._cuda_graph_enabled:
            return
        self._set_kv_len_for_layers(kv_lens)
        self._graph_input_ids = input_ids.clone()
        self._graph_pos = pos.clone()
        position_offset = int(pos[0].item())

        # Build past_key_values from per-layer kv_len
        past_kv = kv_lens  # list of ints — the model forward expects this

        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph):
            logits, _ = self.forward(self._graph_input_ids, past_kv, position_offset, max_seq_len)
            self._graph_logits = logits

    def graph_replay(self, kv_lens: list[int], input_ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        self._set_kv_len_for_layers(kv_lens)
        self._graph_input_ids.copy_(input_ids)
        self._graph_pos.copy_(pos)
        self._decode_graph.replay()
        return self._graph_logits

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
        if self.has_decode_graph and torch.cuda.is_current_stream_capturing():
            pos = self._graph_pos  # use pre-allocated (updated before replay)
        else:
            pos = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device, dtype=torch.long)

        residual = None
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            hidden_states, residual, new_cache = layer(
                hidden_states, pos, layer_cache, max_seq_len=max_seq_len, residual=residual
            )
            new_past_key_values.append(new_cache)

        hidden_states, _ = self.norm(hidden_states, residual)
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
            layer.self_attn.qkv_proj.load_weight_shard(
                self._load_tensor(f"{pfx}.self_attn.q_proj.weight", split_dim=0),
                self._load_tensor(f"{pfx}.self_attn.k_proj.weight", split_dim=0, allow_kv_replication=True),
                self._load_tensor(f"{pfx}.self_attn.v_proj.weight", split_dim=0, allow_kv_replication=True),
            )
            layer.self_attn.o_proj.weight.data.copy_(
                self._load_tensor(f"{pfx}.self_attn.o_proj.weight", split_dim=1).to(layer.self_attn.o_proj.weight)
            )
            layer.mlp.gate_up_proj.load_weight_shard(
                self._load_tensor(f"{pfx}.mlp.gate_proj.weight", split_dim=0),
                self._load_tensor(f"{pfx}.mlp.up_proj.weight", split_dim=0),
            )
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
            f"[TP-Probe] rank={rank} first qkv_proj shape={tuple(first.self_attn.qkv_proj.weight.shape)} "
            f"device={first.self_attn.qkv_proj.weight.device}"
        )
        print(
            f"[TP-Probe] rank={rank} first gate_up_proj shape={tuple(first.mlp.gate_up_proj.weight.shape)} "
            f"device={first.mlp.gate_up_proj.weight.device}"
        )
        print(
            f"[TP-Probe] rank={rank} last qkv_proj shape={tuple(last.self_attn.qkv_proj.weight.shape)} "
            f"device={last.self_attn.qkv_proj.weight.device}"
        )
        print(
            f"[TP-Probe] rank={rank} last gate_up_proj shape={tuple(last.mlp.gate_up_proj.weight.shape)} "
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
        init_custom_ar(device=device)
        # P2: no torch.compile for Qwen (SDPA decode dynamic kv_len)

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
                seq._decode_step_count = 0; self.model._decode_graph = None  # invalidate graph on new prefill
            elif self.model.has_decode_graph:
                new_token = seq.token_ids[-1:]
                ids = torch.tensor([new_token], dtype=torch.long, device=self.device)
                position_offset = len(seq.token_ids) - 1
                pos = torch.tensor([position_offset], dtype=torch.long, device=self.device)
                logits = self.model.graph_replay(seq.past_key_values, ids, pos)
                # kv_lens are updated by the graph (_kv_len_gpu incremented inside)
                seq.past_key_values = [int(layer.self_attn._kv_len_gpu[0].item()) for layer in self.model.layers]
                logits = logits[0, -1, :].unsqueeze(0)
            else:
                new_token = seq.token_ids[-1:]
                ids = torch.tensor([new_token], dtype=torch.long, device=self.device)
                position_offset = len(seq.token_ids) - 1
                # Set _kv_len_gpu before forward (required by decode path)
                self.model._set_kv_len_for_layers(seq.past_key_values)
                logits, past_kv = self.model(
                    ids, past_key_values=seq.past_key_values, position_offset=position_offset, max_seq_len=max_seq_len
                )
                seq.past_key_values = past_kv
                logits = logits[0, -1, :].unsqueeze(0)
                # Capture graph after 2 eager warmup steps
                seq._decode_step_count = seq._decode_step_count + 1
                if seq._decode_step_count == 2:
                    pos = torch.tensor([position_offset + 1], dtype=torch.long, device=self.device)
                    self.model.init_decode_graph(ids, pos, past_kv, max_seq_len)
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

