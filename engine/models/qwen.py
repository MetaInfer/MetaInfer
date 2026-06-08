"""
Phase 5 — QwenAttentionTP + RMSNorm (Attention + KV Cache).

Qwen3-8B TP=4 per-rank dimensions:
  num_heads = 32 // 4 = 8
  num_kv_heads = 8 // 4 = 2  (max(1, 8//4))
  head_dim = 128
  q_size = 8 * 128 = 1024
  kv_size = 2 * 128 = 256
  qkv_proj = [1536, 4096]
  o_proj = [4096, 1024]

Contract: inference_blueprint.json >
  qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP
  paged_kv_cache_contract
  flash_attention_integration_contract
"""

import os
import torch
import torch.nn as nn

from engine.tp_layers.linear import (
    QKVColumnParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from engine.kernels.vllm_wrappers import (
    rms_norm,
    fused_add_rms_norm,
    silu_and_mul,
    rms_norm as _rms_norm_kernel,
    fused_add_rms_norm as _fused_add_rms_norm_kernel,
    rotary_embedding as _rotary_embedding,
    _get_cos_sin_cache,
)
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


def _get_tp_size() -> int:
    """Return tp_size from env WORLD_SIZE, default 1."""
    return int(os.environ.get("WORLD_SIZE", 1))


# ===========================================================================
# RMSNorm — wraps vLLM rms_norm / fused_add_rms_norm CUDA kernels
# ===========================================================================

class RMSNorm(nn.Module):
    """RMSNorm layer backed by vLLM CUDA kernels.

    Precision law (FM-016): self.weight * x_f.to(input_dtype)
    NOT (self.weight.float() * x_f).to(input_dtype).
    HF weights trained against 'cast then multiply' precision path.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """RMSNorm forward — always returns 2-tuple (out, residual).

        - Without residual: allocates output buffer, returns (out, None).
        - With residual: in-place fused_add_rms_norm, returns (x, residual).

        Contract: rmsnorm_return_type in inference_blueprint.json —
          all call sites use tuple unpacking.
        """
        if residual is None:
            out = torch.empty_like(x)
            _rms_norm_kernel(out, x.contiguous(), self.weight, self.eps)
            return out, None
        else:
            _fused_add_rms_norm_kernel(
                x, residual, self.weight, self.eps
            )
            return x, residual


# ===========================================================================
# QwenAttentionTP — TP attention with paged KV cache + flash attention
# ===========================================================================

class QwenAttentionTP(nn.Module):
    """Tensor-parallel attention for Qwen3 Dense.

    Per-rank dimensions (TP=4):
      num_heads = 8, num_kv_heads = 2, head_dim = 128
      _kv_block_size = 256 (flash_attn_with_kvcache minimum)
      KV cache: lazy alloc [num_blocks, 256, 2, 128] bf16

    Two paths:
      forward()          — prefill: flash_attn_varlen_func(causal=True)
      forward_decode()   — decode:  flash_attn_with_kvcache(causal=False)
    """

    def __init__(self, cfg):
        super().__init__()
        tp_size = _get_tp_size()

        # --- head config (full + per-rank) ---
        self.total_num_heads = cfg.num_attention_heads    # 32
        self.total_num_kv_heads = cfg.num_key_value_heads  # 8
        self.num_heads = cfg.num_attention_heads // tp_size  # 8

        # KV head replication guard (tp > num_kv_heads → replicate)
        if cfg.num_key_value_heads >= tp_size:
            self.num_kv_heads = cfg.num_key_value_heads // tp_size  # 2
        else:
            self.num_kv_heads = 1
            self.kv_head_replica = tp_size // cfg.num_key_value_heads

        self.head_dim = cfg.head_dim                        # 128
        self.q_size = self.num_heads * self.head_dim        # 1024
        self.kv_size = self.num_kv_heads * self.head_dim    # 256
        self.scaling = self.head_dim ** -0.5                 # 1/sqrt(128)

        # --- projection layers ---
        self.qkv_proj = QKVColumnParallelLinear(
            cfg.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,   # in = 4096 (full)
            cfg.hidden_size,                         # out = 4096
            bias=False,
        )

        # --- Q/K norms ---
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

        # --- per-layer buffers ---
        self.register_buffer(
            "_cu_q", torch.tensor([0, 0], dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "_cu_k", torch.tensor([0, 0], dtype=torch.int32), persistent=False
        )

        # --- KV cache (lazy alloc) ---
        self._kv_block_size = 256
        self._key_cache = None
        self._value_cache = None
        self._block_table = None
        self._slot_mapping = None

        # --- KV tracking buffers ---
        self.register_buffer(
            "_kv_len_gpu", torch.zeros(1, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "_slot_mapping_decode",
            torch.zeros(1, dtype=torch.int64),
            persistent=False,
        )

        # --- RoPE cos/sin cache (CPU → lazy GPU) ---
        self._cos_sin_cache_cpu = _get_cos_sin_cache(
            cfg.max_position_embeddings, self.head_dim, cfg.rope_theta
        )
        self._cos_sin_cache_gpu = None

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def allocate_kv_cache(self, num_blocks: int) -> None:
        """Lazily allocate KV cache tensors.

        Args:
            num_blocks: number of paged blocks to allocate
              = (num_tokens + 255) // 256 for prefill
        """
        device = self._kv_len_gpu.device
        self._key_cache = torch.zeros(
            num_blocks,
            self._kv_block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self._value_cache = torch.zeros_like(self._key_cache)
        self._block_table = torch.zeros(
            1, num_blocks, dtype=torch.int32, device=device
        )

    def get_num_free_blocks(self) -> int:
        """Return number of free blocks (constant for TP Runner path).

        TP Runner uses lazy allocation — blocks are pre-allocated by
        torch.arange sequential assignment.  The number of remaining
        allocatable blocks is computed from the model's context window.

        Constant: cfg.max_position_embeddings // 256
        """
        # This is a placeholder — the runner computes actual free blocks
        # from the model config.  Each attention layer has the same view.
        return 0  # replaced at runner level

    # ------------------------------------------------------------------
    # Prefill forward  (flash_attn_varlen_func)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Prefill attention: qkv_proj → Q/K norm → RoPE → flash_attn_varlen
        → KV cache write → o_proj.

        Args:
            hidden_states: [B, S, hidden_size]  (B=1)
            positions:     [S] int64 position indices
            max_seq_len:   int, max model context length

        Returns:
            [B, S, hidden_size] attention output (after o_proj + all_reduce)
        """
        B, S, H = hidden_states.shape  # B=1
        device = hidden_states.device

        # 1. QKV projection
        q, k, v = self.qkv_proj(hidden_states)
        # q: [B, S, q_size=1024], k: [B, S, kv_size=256], v: [B, S, kv_size=256]

        # 2. Reshape to 4D per-head
        q = q.view(B, S, self.num_heads, self.head_dim)       # [1, S, 8, 128]
        k = k.view(B, S, self.num_kv_heads, self.head_dim)    # [1, S, 2, 128]
        v = v.view(B, S, self.num_kv_heads, self.head_dim)    # [1, S, 2, 128]

        # 3. Q/K norm
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)

        num_tokens = B * S

        # 4. RoPE — flatten to 2D [tokens, heads, dim]
        q_flat = q.reshape(num_tokens, self.num_heads, self.head_dim)
        k_flat = k.reshape(num_tokens, self.num_kv_heads, self.head_dim)

        # Lazy GPU transfer of cos/sin cache
        if self._cos_sin_cache_gpu is None:
            self._cos_sin_cache_gpu = self._cos_sin_cache_cpu.to(device)

        _rotary_embedding(
            positions,
            q_flat,
            k_flat,
            self.head_dim,
            self._cos_sin_cache_gpu,
            is_neox=True,
        )

        # 5. Flash attention (prefill: ragged, causal=True)
        # K/V come from qkv_proj output, NOT from KV cache
        v_flat = v.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        cu = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        max_s = num_tokens

        out = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu, cu, max_s, max_s,
            causal=True,
        )
        # out: [num_tokens, num_heads, head_dim]

        # 6. KV cache lazy allocation + write
        num_blocks_needed = (num_tokens + self._kv_block_size - 1) // self._kv_block_size

        if self._key_cache is None:
            self.allocate_kv_cache(num_blocks_needed)

        # Block table: sequential assignment (logical→physical identity map)
        self._block_table[0, :num_blocks_needed] = torch.arange(
            num_blocks_needed, dtype=torch.int32, device=device
        )

        # Slot mapping: vectorized (no per-token .item() loop)
        indices = torch.arange(num_tokens, device=device)
        slot_mapping = (
            self._block_table[0, indices // self._kv_block_size] * self._kv_block_size
            + (indices % self._kv_block_size)
        )  # int64

        # Write K/V to paged cache (prefill: direct index assignment, not index_copy_)
        kc_flat = self._key_cache.view(-1, self.num_kv_heads, self.head_dim)
        vc_flat = self._value_cache.view(-1, self.num_kv_heads, self.head_dim)
        kc_flat[slot_mapping] = k_flat.contiguous()
        vc_flat[slot_mapping] = v_flat.contiguous()

        # Set KV length after prefill
        self._kv_len_gpu[0] = num_tokens

        # 7. Output projection
        out = out.view(B, S, self.q_size)
        return self.o_proj(out)

    # ------------------------------------------------------------------
    # Decode forward  (flash_attn_with_kvcache)
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_len: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Decode attention (single token): qkv_proj → Q/K norm → RoPE
        → KV cache write → flash_attn_with_kvcache → o_proj.

        Args:
            hidden_states: [1, 1, hidden_size]
            positions:     [1] int64, current token position (= kv_len)
            kv_len:        int, current KV length BEFORE this step
            max_seq_len:   int, max model context length

        Returns:
            [1, 1, hidden_size] attention output (after o_proj + all_reduce)
        """
        B, S, H = hidden_states.shape  # B=1, S=1

        # 1. QKV projection
        q, k, v = self.qkv_proj(hidden_states)
        # q: [1, 1, 1024], k: [1, 1, 256], v: [1, 1, 256]

        # 2. Reshape to 4D per-head
        q = q.view(B, S, self.num_heads, self.head_dim)       # [1, 1, 8, 128]
        k = k.view(B, S, self.num_kv_heads, self.head_dim)    # [1, 1, 2, 128]
        v = v.view(B, S, self.num_kv_heads, self.head_dim)    # [1, 1, 2, 128]

        # 3. Q/K norm
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)

        # 4. RoPE — flatten to 2D
        q_flat = q.reshape(S, self.num_heads, self.head_dim)
        k_flat = k.reshape(S, self.num_kv_heads, self.head_dim)

        if self._cos_sin_cache_gpu is None:
            self._cos_sin_cache_gpu = self._cos_sin_cache_cpu.to(
                hidden_states.device
            )

        _rotary_embedding(
            positions,
            q_flat,
            k_flat,
            self.head_dim,
            self._cos_sin_cache_gpu,
            is_neox=True,
        )

        q = q_flat.view(B, S, self.num_heads, self.head_dim)
        k = k_flat.view(B, S, self.num_kv_heads, self.head_dim)

        # 5. KV cache write (decode: write 1 token at slot = kv_len)
        self._slot_mapping_decode[0] = self._kv_len_gpu[0]

        k_write = k.reshape(S, self.num_kv_heads, self.head_dim)  # [1, 2, 128]
        v_write = v.reshape(S, self.num_kv_heads, self.head_dim)  # [1, 2, 128]

        kc_flat = self._key_cache.view(-1, self.num_kv_heads, self.head_dim)
        vc_flat = self._value_cache.view(-1, self.num_kv_heads, self.head_dim)

        kc_flat.index_copy_(0, self._slot_mapping_decode, k_write)
        vc_flat.index_copy_(0, self._slot_mapping_decode, v_write)

        # Increment KV length AFTER write (new token now visible)
        self._kv_len_gpu[0] += 1

        # 6. Flash attention with paged KV cache
        # q format: [1, num_heads, head_dim] (reshape from [1,1,H,D])
        q_attn = q.reshape(1, self.num_heads, self.head_dim)

        out = flash_attn_with_kvcache(
            q_attn,
            self._key_cache,
            self._value_cache,
            cache_seqlens=self._kv_len_gpu,
            block_table=self._block_table,
            softmax_scale=self.scaling,
            causal=False,
        )
        # out: [1, 1, num_heads, head_dim]

        # 7. Output projection
        out = out.reshape(B, S, self.q_size)
        return self.o_proj(out)


# ===========================================================================
# QwenMLPTP — TP MLP with merged gate+up projection
# ===========================================================================

class QwenMLPTP(nn.Module):
    """Tensor-parallel MLP for Qwen3 Dense.

    Merged gate+up projection (MergedColumnParallelLinear) → silu_and_mul → down_proj.

    Per-rank dimensions (TP=4):
      intermediate_size = 12288 → inter_per_rank = 3072
      gate_up_out = 2 * 3072 = 6144  (NOT 6400!)
    """

    def __init__(self, cfg):
        super().__init__()
        tp_size = _get_tp_size()
        self.intermediate_per_rank = cfg.intermediate_size // tp_size  # 3072

        self.gate_up_proj = MergedColumnParallelLinear(
            cfg.hidden_size,        # 4096
            cfg.intermediate_size,  # 12288
            bias=False,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            cfg.intermediate_size,  # in_features = 12288 (full)
            cfg.hidden_size,        # out_features = 4096
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward: gate_up_proj → silu_and_mul → down_proj.

        x: [B, T, hidden_size]
        returns: [B, T, hidden_size]
        """
        gate_up = self.gate_up_proj(x)  # [B, T, 2*inter_per_rank] = [B, T, 6144]
        act = torch.empty(
            *gate_up.shape[:-1],
            self.intermediate_per_rank,  # 3072
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        silu_and_mul(act, gate_up)
        return self.down_proj(act)


# ===========================================================================
# QwenDecoderLayerTP — TP decoder layer (prefill + decode paths)
# ===========================================================================

class QwenDecoderLayerTP(nn.Module):
    """Tensor-parallel decoder layer for Qwen3 Dense.

    Two paths:
      forward()        — prefill:  rms_norm / fused_add_rms_norm → Attn → Add+Norm → MLP
      forward_decode() — decode:   fused_add_rms_norm → Attn → Add+Norm → MLP (single token)
    """

    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = QwenAttentionTP(cfg)
        self.mlp = QwenMLPTP(cfg)

    # ------------------------------------------------------------------
    # Prefill forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        layer_cache,           # unused — KV cache is internal to self_attn
        max_seq_len: int,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill decoder layer forward (B=1 single-seq).

        Residual chain:
          residual = hidden_states.clone()  (first layer only)
          hidden_states = rms_norm(residual)
          → attention → fused_add_rms_norm → mlp → return (mlp_out, residual)
        """
        if residual is None:
            residual = hidden_states.clone()
            rms_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )
        else:
            fused_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )

        attn_out = self.self_attn.forward(hidden_states, positions, max_seq_len)

        fused_add_rms_norm(
            attn_out,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )

        mlp_out = self.mlp(attn_out)
        return mlp_out, residual

    # ------------------------------------------------------------------
    # Decode forward (single token, no clone)
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_len: int,
        max_seq_len: int,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode decoder layer forward (single token, eager path: no clone).

        Residual chain:
          fused_add_rms_norm → attention → fused_add_rms_norm → mlp
          → return (mlp_out, residual)

        residual is never None in decode — it is carried from prefill.
        """
        # residual always provided in decode (carried from prefill stage)
        fused_add_rms_norm(
            hidden_states,
            residual,
            self.input_layernorm.weight,
            self.input_layernorm.eps,
        )

        attn_out = self.self_attn.forward_decode(
            hidden_states, positions, kv_len, max_seq_len
        )

        fused_add_rms_norm(
            attn_out,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )

        mlp_out = self.mlp(attn_out)
        return mlp_out, residual
