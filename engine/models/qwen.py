# engine/models/qwen.py
# Phase 5: Attention + KV Cache — Qwen3 TP Model Components.
#
# Classes:
#   RMSNorm               — lightweight vLLM rms_norm wrapper
#   QwenAttentionTP       — Attention with Paged KV Cache (prefill + decode)
#   QwenMLPTP             — MLP stub (full impl Phase 6)
#   QwenDecoderLayerTP    — Decoder Layer (prefill + decode with residual chain)
#
# Blueprint contracts:
#   framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces
#   framework_layer.data_flow_contracts.paged_kv_cache_contract
#   framework_layer.data_flow_contracts.flash_attention_integration_contract
#
# Ref:
#   inference_blueprint.json > qwen3_tp_model_interfaces.class_hierarchy
#   inference_blueprint.json > qwen3_tp_model_interfaces.prefill_forward_pattern
#   inference_blueprint.json > qwen3_tp_model_interfaces.decode_forward_pattern
#   notebooks-cn/07_improvementPlan/improvement_plan.md §P3-FA
#   notebooks-cn/07_improvementPlan/qwen3_effective_changes.md #8 #10

import torch
import torch.nn as nn

from engine.kernels.vllm_wrappers import (
    rms_norm,
    fused_add_rms_norm,
    silu_and_mul,
    rotary_embedding,
    _get_cos_sin_cache,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from engine.tp_layers.linear import (
    QKVColumnParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from engine.tp_layers.distributed import get_tp_size, get_tp_rank, init_custom_ar
from engine.tp_layers.embedding import VocabParallelEmbedding, ParallelLMHead
from dataclasses import dataclass
from pathlib import Path
import json
import torch.distributed as dist
from safetensors import safe_open


# ================================================================
# RMSNorm
# ================================================================

class RMSNorm(nn.Module):
    """Lightweight RMSNorm wrapper using vLLM CUDA kernel.

    Blueprint contract:
        rmsnorm_precision_law: vLLM kernel internally uses fp32 computation.
        Caller ensures out pre-allocated and input contiguous.

    Usage:
        # Without residual (pure norm)
        out = rms_norm_layer(x)

        # With residual chain (fused add + norm)
        rms_norm_layer(hidden_states, residual)  # modifies both in-place
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is None:
            out = torch.empty_like(x)
            rms_norm(out, x.contiguous(), self.weight, self.eps)
            return out, None
        else:
            fused_add_rms_norm(x.contiguous(), residual.contiguous(), self.weight, self.eps)
            return x, residual  # both modified in-place


# ================================================================
# QwenAttentionTP
# ================================================================

class QwenAttentionTP(nn.Module):
    """Attention with Paged KV Cache for Qwen3 TP.

    Blueprint contract:
        class_hierarchy.QwenAttentionTP — exact attr names and shapes.
        paged_kv_cache_contract — block_size=256, block_table=int32.
        flash_attention_integration_contract — prefill flash_attn_varlen_func,
        decode flash_attn_with_kvcache.

    Key constraints:
        - block_size=256 (flash_attn_with_kvcache hard requirement)
        - block_table dtype=int32 (NOT int64)
        - K/V reshape uses num_kv_heads (NOT num_heads)
        - Prefill K/V from qkv_proj output (NOT from cache)
        - Vectorized slot_mapping (no for-loop .item())
    """

    def __init__(self, cfg, tp_size=None):
        super().__init__()
        if tp_size is None:
            tp_size = get_tp_size()

        # Head counts (full)
        self.total_num_heads = cfg.num_attention_heads       # 32
        self.total_num_kv_heads = cfg.num_key_value_heads    # 8

        # Per-rank head counts
        self.num_heads = cfg.num_attention_heads // tp_size  # 8

        # KV head replication: if tp > num_kv_heads → replicate
        if cfg.num_key_value_heads >= tp_size:
            self.num_kv_heads = cfg.num_key_value_heads // tp_size  # 2
            self.kv_head_replica = 1
        else:
            self.num_kv_heads = 1
            self.kv_head_replica = tp_size // cfg.num_key_value_heads

        self.head_dim = cfg.head_dim                          # 128
        self.q_size = self.num_heads * self.head_dim          # 1024
        self.kv_size = self.num_kv_heads * self.head_dim      # 256
        self.scaling = self.head_dim ** -0.5                  # ~0.08839
        self.hidden_size = cfg.hidden_size

        # QKV projection: merged QKVColumnParallelLinear
        self.qkv_proj = QKVColumnParallelLinear(
            cfg.hidden_size, self.head_dim,
            self.total_num_heads, self.total_num_kv_heads,
            tp_size=tp_size, bias=False)

        # Output projection: RowParallelLinear
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            cfg.hidden_size, tp_size=tp_size, bias=False)

        # Q/K norms
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

        # Pre-allocated cu_seqlens buffers (persistent=False — not saved in state_dict)
        self.register_buffer('_cu_q', torch.tensor([0, 0], dtype=torch.int32), persistent=False)
        self.register_buffer('_cu_k', torch.tensor([0, 0], dtype=torch.int32), persistent=False)
        self.register_buffer('_cu_prefill', torch.zeros(2, dtype=torch.int32), persistent=False)

        # KV cache — lazy allocated on first prefill
        self._kv_block_size = 256  # HARD: flash_attn_with_kvcache requires >= 256
        self._key_cache = None
        self._value_cache = None
        self._block_table = None
        self._slot_mapping = None

        # _kv_len_gpu: GPU tensor tracking KV length. register_buffer(int32)
        self.register_buffer('_kv_len_gpu', torch.zeros(1, dtype=torch.int32), persistent=False)

        # _slot_mapping_decode: decode write target. int64 for index_copy_
        self.register_buffer('_slot_mapping_decode', torch.zeros(1, dtype=torch.int64), persistent=False)

        # Cos/Sin cache — lazy GPU transfer
        self._cos_sin_cache_cpu = _get_cos_sin_cache(
            cfg.max_position_embeddings, self.head_dim, cfg.rope_theta)
        self._cos_sin_cache_gpu = None

    def forward(self, hidden_states, positions, max_seq_len):
        """Prefill path (B=1): flash_attn_varlen_func with KV cache write.

        Blueprint contract:
            prefill_forward_pattern.layer_forward_pseudocode
            K/V from qkv_proj output, NOT from cache.
            causal=True.
        """
        B, S, H = hidden_states.shape  # B=1

        # 1. Lazy GPU transfer for cos_sin_cache
        if self._cos_sin_cache_gpu is None:
            self._cos_sin_cache_gpu = self._cos_sin_cache_cpu.to(hidden_states.device)

        # 2. QKV projection + split
        q, k, v = self.qkv_proj(hidden_states)

        # 3. Flatten to 3D + Q/K norm (P4: no intermediate 4D views, P3: view not reshape)
        num_tokens = B * S
        q_flat = q.view(num_tokens, self.num_heads, self.head_dim)
        k_flat = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        q_flat, _ = self.q_norm(q_flat)
        k_flat, _ = self.k_norm(k_flat)

        # 4. RoPE
        rotary_embedding(positions, q_flat, k_flat, self.head_dim,
                         self._cos_sin_cache_gpu, is_neox=True)

        # 5. KV cache lazy allocation (first prefill only)
        if self._key_cache is None:
            max_blocks = (max_seq_len + 255) // 256
            self._key_cache = torch.zeros(
                max_blocks, 256, self.num_kv_heads, self.head_dim,
                dtype=hidden_states.dtype, device=hidden_states.device)
            self._value_cache = torch.zeros_like(self._key_cache)
            self._block_table = torch.zeros(
                1, max_blocks, dtype=torch.int32, device=hidden_states.device)

        # 6. Flash attention (prefill) — K/V from projection, NOT from cache!
        v_flat = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        self._cu_prefill[0] = 0
        self._cu_prefill[1] = num_tokens
        out = flash_attn_varlen_func(
            q_flat, k_flat, v_flat, self._cu_prefill, self._cu_prefill,
            num_tokens, num_tokens, causal=True)

        # 7. Write KV to paged cache (VECTORIZED slot_mapping — NO for-loop .item()!)
        num_blocks = (num_tokens + 255) // 256
        self._block_table[0, :num_blocks] = torch.arange(
            num_blocks, dtype=torch.int32, device=hidden_states.device)
        _a = torch.arange(num_tokens, device=hidden_states.device)
        slot_mapping = self._block_table[0, _a // 256] * 256 + (_a % 256)
        kc_flat = self._key_cache.view(-1, self.num_kv_heads, self.head_dim)
        vc_flat = self._value_cache.view(-1, self.num_kv_heads, self.head_dim)
        kc_flat.index_copy_(0, slot_mapping, k_flat)
        vc_flat.index_copy_(0, slot_mapping, v_flat)
        self._kv_len_gpu[0] = num_tokens

        # 8. Output projection
        out = out.view(B, S, self.q_size)
        return self.o_proj(out)

    def forward_decode(self, hidden_states, positions, kv_len, max_seq_len):
        """Decode path (B=1, S=1): flash_attn_with_kvcache from paged cache.

        Blueprint contract:
            decode_forward_pattern.full_method_body — exact copy.
            causal=False for decode.
        """
        B, S, H = hidden_states.shape  # B=1, S=1

        # 1. Lazy cos_sin_cache GPU transfer
        if self._cos_sin_cache_gpu is None:
            self._cos_sin_cache_gpu = self._cos_sin_cache_cpu.to(hidden_states.device)

        # 2. QKV projection + split
        q, k, v = self.qkv_proj(hidden_states)

        # 3. Flatten to 3D + Q/K norm (P4: no intermediate 4D views, P3: view not reshape)
        q_flat = q.view(S, self.num_heads, self.head_dim)
        k_flat = k.view(S, self.num_kv_heads, self.head_dim)
        q_flat, _ = self.q_norm(q_flat)
        k_flat, _ = self.k_norm(k_flat)

        # 4. RoPE
        rotary_embedding(positions, q_flat, k_flat, self.head_dim,
                         self._cos_sin_cache_gpu, is_neox=True)
        q = q_flat.view(B, S, self.num_heads, self.head_dim)
        k = k_flat.view(B, S, self.num_kv_heads, self.head_dim)

        # 5. Write K/V to cache at slot=kv_len
        self._slot_mapping_decode[0] = self._kv_len_gpu[0]
        k_write = k.view(1, self.num_kv_heads, self.head_dim)
        v_write = v.view(1, self.num_kv_heads, self.head_dim)
        kc_flat = self._key_cache.view(-1, self.num_kv_heads, self.head_dim)
        vc_flat = self._value_cache.view(-1, self.num_kv_heads, self.head_dim)
        kc_flat.index_copy_(0, self._slot_mapping_decode, k_write)
        vc_flat.index_copy_(0, self._slot_mapping_decode, v_write)
        self._kv_len_gpu[0] += 1

        # 6. flash_attn_with_kvcache (read from paged cache)
        q_attn = q.view(1, 1, self.num_heads, self.head_dim)
        out = flash_attn_with_kvcache(
            q_attn, self._key_cache, self._value_cache,
            cache_seqlens=self._kv_len_gpu,
            block_table=self._block_table,
            softmax_scale=self.scaling,
            causal=False)

        # 7. o_proj
        out = out.view(B, S, self.q_size)
        return self.o_proj(out)


# ================================================================
# QwenMLPTP (Minimal stub — full implementation is Phase 6)
# ================================================================

class QwenMLPTP(nn.Module):
    """MLP with MergedColumnParallelLinear gate+up fusion.

    Blueprint contract:
        class_hierarchy.QwenMLPTP — exact attr names.
        mlp gate_up_proj → silu_and_mul → down_proj chain.
    """

    def __init__(self, cfg, tp_size=None):
        super().__init__()
        if tp_size is None:
            tp_size = get_tp_size()

        self.gate_up_proj = MergedColumnParallelLinear(
            cfg.hidden_size, cfg.intermediate_size,
            tp_size=tp_size, bias=False)
        self.down_proj = RowParallelLinear(
            cfg.intermediate_size, cfg.hidden_size,
            tp_size=tp_size, bias=False)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        half_ch = gate_up.shape[-1] // 2
        out = torch.empty_like(gate_up[..., :half_ch])
        silu_and_mul(out, gate_up)
        return self.down_proj(out)


# ================================================================
# QwenDecoderLayerTP
# ================================================================

class QwenDecoderLayerTP(nn.Module):
    """Decoder Layer: Attention + MLP with residual chain.

    Blueprint contract:
        class_hierarchy.QwenDecoderLayerTP — exact attr names (.self_attn, .mlp,
        .input_layernorm, .post_attention_layernorm).
        prefill_forward_pattern.layer_forward_pseudocode — residual chain.
        decode_forward_pattern.full_method_body — exact forward_decode.

    Residual chain (both prefill and decode):
        Layer 0: res = hs.clone(); rms_norm(hs, res, ...weight)
        Layer 1+: fused_add_rms_norm(hs, res, ...weight)  # res+=hs; hs=rms_norm(res)

    All 4 fused_add_rms_norm calls use self.input_layernorm.weight or
    self.post_attention_layernorm.weight (no cross-layer weight references).
    """

    def __init__(self, cfg, tp_size=None):
        super().__init__()
        if tp_size is None:
            tp_size = get_tp_size()

        self.self_attn = QwenAttentionTP(cfg, tp_size=tp_size)
        self.mlp = QwenMLPTP(cfg, tp_size=tp_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.hidden_size = cfg.hidden_size

    def forward(self, hidden_states, positions, layer_cache=None, max_seq_len=None, residual=None):
        """Prefill path: residual chain with fused_add_rms_norm.

        Blueprint contract:
            prefill_forward_pattern.layer_forward_pseudocode
        """
        hs, res = hidden_states, residual
        if res is None:
            res = hs.clone()
            rms_norm(hs, res, self.input_layernorm.weight, self.input_layernorm.eps)
        else:
            fused_add_rms_norm(hs, res, self.input_layernorm.weight, self.input_layernorm.eps)

        hs = self.self_attn.forward(hs, positions, max_seq_len)

        fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight,
                          self.post_attention_layernorm.eps)

        mlp_out = self.mlp(hs)
        return mlp_out, res

    def forward_decode(self, hidden_states, positions, kv_len, max_seq_len, residual=None):
        """Decode path (B=1, S=1): residual chain + attention decode + mlp.

        Blueprint contract:
            decode_forward_pattern.full_method_body — exact copy.
            No clone() in eager path.
        """
        hs, res = hidden_states, residual
        if res is None:
            res = hs.clone()
            rms_norm(hs, res, self.input_layernorm.weight, self.input_layernorm.eps)
        else:
            fused_add_rms_norm(hs, res, self.input_layernorm.weight, self.input_layernorm.eps)

        hs = self.self_attn.forward_decode(hs, positions, kv_len, max_seq_len)

        fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight,
                          self.post_attention_layernorm.eps)

        mlp_out = self.mlp(hs)
        return mlp_out, res


# ================================================================
# Phase 7: QwenTPConfig — dataclass, dynamic config.json reader
# ================================================================

@dataclass
class QwenTPConfig:
    """Configuration dataclass for Qwen3 TP models.

    Blueprint contract:
        class_hierarchy.QwenTPConfig — exact field names and types.
        head_dim_fallback: hidden_size // num_attention_heads.
        factory: from_model_dir(model_dir).

    All fields are read dynamically from config.json — never hardcoded.
    """
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
    tie_word_embeddings: bool = False

    @staticmethod
    def from_model_dir(model_dir):
        """Factory: config.json -> QwenTPConfig. Dynamic read, no hardcoded dims."""
        import json
        from pathlib import Path
        cfg_path = Path(model_dir) / 'config.json'
        with open(cfg_path) as f:
            cfg = json.load(f)
        head_dim = cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])
        return QwenTPConfig(
            model_dir=Path(model_dir),
            hidden_size=cfg['hidden_size'],
            intermediate_size=cfg['intermediate_size'],
            num_hidden_layers=cfg['num_hidden_layers'],
            num_attention_heads=cfg['num_attention_heads'],
            num_key_value_heads=cfg['num_key_value_heads'],
            head_dim=head_dim,
            vocab_size=cfg['vocab_size'],
            rms_norm_eps=cfg.get('rms_norm_eps', 1e-6),
            rope_theta=cfg.get('rope_theta', 1000000.0),
            max_position_embeddings=cfg['max_position_embeddings'],
            tie_word_embeddings=cfg.get('tie_word_embeddings', False),
        )


# ================================================================
# Phase 7: QwenForCausalLMTP — top-level model with weight loading
# ================================================================

class QwenForCausalLMTP(nn.Module):
    """Qwen3 TP top-level model: embed_tokens -> layers -> norm -> lm_head.

    Blueprint contract:
        class_hierarchy.QwenForCausalLMTP — exact attr names.
        construction_chain: cfg -> model -> load_weights -> eval -> init_custom_ar.
        model_forward_pseudocode — prefill/decode dispatch logic.

    Attr names:
        .embed_tokens  (VocabParallelEmbedding)
        .layers        (ModuleList[QwenDecoderLayerTP])
        .norm          (RMSNorm)
        .lm_head       (ParallelLMHead)
    """

    def __init__(self, cfg, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [QwenDecoderLayerTP(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size)
        if device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, input_ids, past_key_values=None, position_offset=0, max_seq_len=528):
        """Prefill/decode dispatch — blueprint model_forward_pseudocode exact copy.

        Args:
            input_ids:        [B, S] int64
            past_key_values:  None for prefill, list[int] kv_lens for decode
            position_offset:  start position for RoPE (decode: current kv_len)
            max_seq_len:      max model sequence length

        Returns:
            logits:  [B, S, vocab_size]
            kv_lens: None for prefill, list[int] for decode (read after all layers)
        """
        hidden_states = self.embed_tokens(input_ids)  # [B, S, hidden_size]
        seq_len = input_ids.shape[1]
        positions = torch.arange(
            position_offset, position_offset + seq_len,
            device=input_ids.device, dtype=torch.long)

        residual = None
        is_decode = past_key_values is not None

        if not is_decode:
            # === Prefill ===
            for i, layer in enumerate(self.layers):
                hidden_states, residual = layer.forward(
                    hidden_states, positions, layer_cache=None,
                    max_seq_len=max_seq_len, residual=residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            kv_lens = None
        else:
            # === Decode ===
            for i, layer in enumerate(self.layers):
                kv_len = past_key_values[i]
                hidden_states, residual = layer.forward_decode(
                    hidden_states, positions, kv_len,
                    max_seq_len=max_seq_len, residual=residual)
            # Batch read kv_lens AFTER all layers (P5: 1x .item() from layer 0,
            # since all layers share the same kv_len at decode start)
            kv_len = int(self.layers[0].self_attn._kv_len_gpu[0].item())
            kv_lens = [kv_len] * len(self.layers)
            hidden_states, _ = self.norm(hidden_states, residual)

        logits = self.lm_head(hidden_states)  # [B, S, vocab_size]
        return logits, kv_lens

    def forward_decode(self, input_ids, positions, kv_len, max_seq_len):
        """Decode forward — standalone method for runner compatibility.

        Blueprint contract:
          tp_runner_actual_flow.run_method_impl — decode path calls forward_decode.

        Prefill/decode dispatch is handled at the runner level (QwenTPModelRunner.run),
        not inside the model. This method handles only the decode path with explicit
        kv_len (same for all layers at decode start).

        Args:
            input_ids:    [B=1, S=1] int64 — single new token
            positions:    [1] int64 — current kv_len position
            kv_len:       int — current KV length (same for all layers)
            max_seq_len:  int — max model sequence length

        Returns:
            logits: [B, S, vocab_size] — logits for the single token
        """
        hidden_states = self.embed_tokens(input_ids)  # [B=1, S=1, hidden_size]
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer.forward_decode(
                hidden_states, positions, kv_len,
                max_seq_len=max_seq_len, residual=residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        logits = self.lm_head(hidden_states)  # [B, S, vocab_size]
        return logits

    # ----------------------------------------------------------------
    # Weight loading
    # ----------------------------------------------------------------

    def load_weights(self):
        """Load weights from safetensors via model.safetensors.index.json.

        Blueprint load_weights_pseudocode:
          1. Read model.safetensors.index.json -> weight_map
          2. Group by safetensors file
          3. For each file: safe_open -> iterate keys -> _dispatch_weight
          4. Merge QKV and Gate-Up buffers
          5. dist.barrier() + init_custom_ar()
        """
        model_dir = self.cfg.model_dir
        index_path = model_dir / 'model.safetensors.index.json'
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index['weight_map']

        # Group keys by safetensors file
        files = {}
        for hf_key, fname in weight_map.items():
            files.setdefault(fname, []).append(hf_key)

        # Accumulate QKV and Gate-Up shards for later concatenation
        qkv_buffers = {}     # {layer_idx: {'q': tensor, 'k': tensor, 'v': tensor}}
        gate_up_buffers = {}  # {layer_idx: {'gate': tensor, 'up': tensor}}

        for fname, keys in files.items():
            with safe_open(model_dir / fname, framework='pt') as f:
                for hf_key in keys:
                    full = f.get_tensor(hf_key)
                    self._dispatch_weight(hf_key, full, qkv_buffers, gate_up_buffers)

        # Merge QKV and Gate-Up after all files have been read
        self._merge_qkv_weights(qkv_buffers)
        self._merge_gate_up_weights(gate_up_buffers)

        # Sync all ranks, then init CustomAR P2P
        if dist.is_initialized():
            dist.barrier()
            init_custom_ar(device=self.embed_tokens.weight.device)

    def _parse_hf_key(self, hf_key):
        """Parse HF key like 'model.layers.5.self_attn.q_proj.weight' -> (layer_idx, component).

        Returns:
            (layer_idx, component) where component is the suffix after 'model.layers.N.'
            or (-1, 'embed_tokens') / (-1, 'norm') / (-1, 'lm_head') for top-level keys.
        """
        parts = hf_key.split('.')
        if hf_key == 'model.embed_tokens.weight':
            return (-1, 'embed_tokens')
        if hf_key == 'model.norm.weight':
            return (-1, 'norm')
        if hf_key == 'lm_head.weight':
            return (-1, 'lm_head')
        # model.layers.N.self_attn.q_proj.weight -> N=2, component='self_attn.q_proj.weight'
        layer_idx = int(parts[2])
        component = '.'.join(parts[3:])
        return (layer_idx, component)

    def _dispatch_weight(self, hf_key, full, qkv_buffers, gate_up_buffers):
        """Route a single HF weight tensor to the correct module attribute.

        Dispatch logic (blueprint qwen_hf_key_mapping):
          - embed_tokens     -> load_weight_shard (vocab/tp split dim 0)
          - q_proj/k_proj/v_proj -> buffer for QKV merge
          - o_proj           -> load_weight_shard (RowParallel split dim 1)
          - gate_proj/up_proj -> buffer for gate-up merge
          - down_proj        -> load_weight_shard (RowParallel split dim 1)
          - input_layernorm / post_attention_layernorm -> direct copy_
          - norm             -> direct copy_
          - lm_head          -> load_weight_shard (vocab/tp split dim 0)
        """
        layer_idx, component = self._parse_hf_key(hf_key)

        if layer_idx == -1:
            # Top-level keys
            if component == 'embed_tokens':
                self.embed_tokens.load_weight_shard(full)
            elif component == 'norm':
                self.norm.weight.data.copy_(full)
            elif component == 'lm_head':
                self.lm_head.load_weight_shard(full)
            return

        # Layer-level keys
        layer = self.layers[layer_idx]

        if component == 'self_attn.q_proj.weight':
            qkv_buffers.setdefault(layer_idx, {})['q'] = full
        elif component == 'self_attn.k_proj.weight':
            qkv_buffers.setdefault(layer_idx, {})['k'] = full
        elif component == 'self_attn.v_proj.weight':
            qkv_buffers.setdefault(layer_idx, {})['v'] = full
        elif component == 'self_attn.o_proj.weight':
            layer.self_attn.o_proj.load_weight_shard(full)
        elif component == 'mlp.gate_proj.weight':
            gate_up_buffers.setdefault(layer_idx, {})['gate'] = full
        elif component == 'mlp.up_proj.weight':
            gate_up_buffers.setdefault(layer_idx, {})['up'] = full
        elif component == 'mlp.down_proj.weight':
            layer.mlp.down_proj.load_weight_shard(full)
        elif component == 'input_layernorm.weight':
            layer.input_layernorm.weight.data.copy_(full)
        elif component == 'post_attention_layernorm.weight':
            layer.post_attention_layernorm.weight.data.copy_(full)
        elif component == 'self_attn.q_norm.weight':
            layer.self_attn.q_norm.weight.data.copy_(full)
        elif component == 'self_attn.k_norm.weight':
            layer.self_attn.k_norm.weight.data.copy_(full)

    def _merge_qkv_weights(self, qkv_buffers):
        """Merge Q/K/V buffers into QKVColumnParallelLinear weights.

        Blueprint contract:
          - Q-K-V order: torch.cat([q_shard, k_shard, v_shard], dim=0)
          - Q: ColumnParallel split dim 0
          - K/V: ColumnParallel split dim 0, KV replication if tp > num_kv_heads
          - cat(Q_shard, K_shard, V_shard) -> load_weight_shard (double_shard_guard)
        """
        if not qkv_buffers:
            return

        tp_rank = get_tp_rank()
        tp_size = get_tp_size()

        for layer_idx, parts in qkv_buffers.items():
            q_full = parts['q']
            k_full = parts['k']
            v_full = parts['v']

            # Per-rank slice sizes
            q_size_per_rank = self.cfg.num_attention_heads * self.cfg.head_dim // tp_size

            # KV: replication if tp > num_kv_heads
            kv_heads_local = max(1, self.cfg.num_key_value_heads // tp_size)
            kv_size_per_rank = kv_heads_local * self.cfg.head_dim

            # Q slice (ColumnParallel split dim 0)
            q_shard = q_full[tp_rank * q_size_per_rank:(tp_rank + 1) * q_size_per_rank, :]

            # K/V slices (KV replication: if num_kv_heads < tp_size, replicate full weight)
            if self.cfg.num_key_value_heads >= tp_size:
                k_shard = k_full[tp_rank * kv_size_per_rank:(tp_rank + 1) * kv_size_per_rank, :]
                v_shard = v_full[tp_rank * kv_size_per_rank:(tp_rank + 1) * kv_size_per_rank, :]
            else:
                k_shard = k_full  # replicated: all ranks get full KV weights
                v_shard = v_full

            # Concat Q-K-V (MUST be Q-K-V order!)
            merged = torch.cat([q_shard, k_shard, v_shard], dim=0)

            # load_weight_shard handles double_shard_guard
            self.layers[layer_idx].self_attn.qkv_proj.load_weight_shard(merged)

    def _merge_gate_up_weights(self, gate_up_buffers):
        """Merge gate/up buffers into MergedColumnParallelLinear weights.

        Blueprint contract:
          - gate-up order: torch.cat([gate_shard, up_shard], dim=0)
          - ColumnParallel split dim 0
          - cat(gate_shard, up_shard) -> load_weight_shard (double_shard_guard)
        """
        if not gate_up_buffers:
            return

        tp_rank = get_tp_rank()
        tp_size = get_tp_size()

        inter_per_rank = self.cfg.intermediate_size // tp_size

        for layer_idx, parts in gate_up_buffers.items():
            gate_full = parts['gate']
            up_full = parts['up']

            # ColumnParallel split along dim 0
            gate_shard = gate_full[tp_rank * inter_per_rank:(tp_rank + 1) * inter_per_rank, :]
            up_shard = up_full[tp_rank * inter_per_rank:(tp_rank + 1) * inter_per_rank, :]

            # Concat gate-up (MUST be gate-up order!)
            merged = torch.cat([gate_shard, up_shard], dim=0)

            # load_weight_shard handles double_shard_guard
            self.layers[layer_idx].mlp.gate_up_proj.load_weight_shard(merged)
