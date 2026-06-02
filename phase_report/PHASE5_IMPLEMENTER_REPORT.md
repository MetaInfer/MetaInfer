# Phase 5 Implementer Report

- **PID**: 936165
- **Role**: implementer
- **Timestamp**: 2026-05-30
- **Phase**: 5
- **Status**: SUBMITTED

## Implemented

### Files Created
1. `engine/models/__init__.py` — package init for engine/models/
2. `engine/models/qwen.py` — Qwen3 TP model components

### Classes Implemented

#### RMSNorm (`engine/models/qwen.py`)
- Lightweight wrapper over vLLM `rms_norm` / `fused_add_rms_norm` CUDA kernels
- `forward(x, residual=None)`: dual-mode — pure norm when `residual is None`, fused add+norm otherwise
- `rms_norm` uses pre-allocated `torch.empty_like(x)` and `x.contiguous()`
- `fused_add_rms_norm` operates in-place on both `x` and `residual`

#### QwenAttentionTP (`engine/models/qwen.py`)
- Full prefill path: QKV projection → Q/K norm → RoPE → `flash_attn_varlen_func` (causal=True) → KV cache lazy alloc + `index_copy_` write → o_proj
- Full decode path: QKV projection → Q/K norm → RoPE → KV write at `_kv_len_gpu` slot → `flash_attn_with_kvcache` (causal=False) → o_proj
- All __init__ attrs match `inference_blueprint.json > class_hierarchy.QwenAttentionTP` exactly
- `_kv_block_size = 256` (flash_attn_with_kvcache hard requirement)
- `_block_table` dtype=int32 (NOT int64)
- KV head replication logic: `num_kv_heads` > tp_size → standard shard; else → replicate
- Prefill K/V from qkv_proj output (NOT from cache read)
- Vectorized slot_mapping: `block_table[0, indices // 256] * 256 + (indices % 256)` (no for-loop .item())
- KV cache lazy allocation on first prefill: `max_blocks = (max_seq_len + 255) // 256`
- `_kv_len_gpu` as register_buffer(int32, persistent=False)
- `_slot_mapping_decode` as register_buffer(int64, persistent=False)
- `_cu_q`/`_cu_k` as pre-allocated writable register_buffers (persistent=False)
- `_cos_sin_cache` lazy GPU transfer via `_cos_sin_cache_gpu`

#### QwenMLPTP (`engine/models/qwen.py`)
- Minimal stub for Phase 6: `gate_up_proj` (MergedColumnParallelLinear) → `silu_and_mul` → `down_proj` (RowParallelLinear)
- Attr names match blueprint exactly: `.gate_up_proj`, `.down_proj`

#### QwenDecoderLayerTP (`engine/models/qwen.py`)
- Full prefill path with residual chain: `rms_norm` (layer 0 clone) / `fused_add_rms_norm` (layer 1+) → attention.forward → `fused_add_rms_norm` (post_attn) → mlp → return (hs, residual)
- Full decode path with residual chain: same norm pattern → attention.forward_decode → post_attn norm → mlp → return (hs, residual)
- All 4 `fused_add_rms_norm` calls use `self.input_layernorm.weight` or `self.post_attention_layernorm.weight` (no cross-layer references)
- Attr names match blueprint exactly: `.self_attn`, `.mlp`, `.input_layernorm`, `.post_attention_layernorm`

## Blueprint Nodes Read

### JSON Paths
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.attention` — KV cache paged format, block_table, _kv_len_gpu
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP` — exact attr names and shapes
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenMLPTP` — attr names
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenDecoderLayerTP` — attr names
- `framework_layer.data_flow_contracts.paged_kv_cache_contract` — slot_mapping algorithm, KV cache format, index_copy_
- `framework_layer.data_flow_contracts.flash_attention_integration_contract` — prefill/decode kernel signatures
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.prefill_forward_pattern` — full dataflow + layer_forward_pseudocode
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern` — full_method_body

### ref_docs
- `notebooks-cn/07_improvementPlan/improvement_plan.md` — P0 (historical only), P3-FA (Flash Attention integration)
- `notebooks-cn/07_improvementPlan/qwen3_effective_changes.md` — #8 (KV cache paged), #10 (flash_attn integration)

### Existing Code Reviewed
- `engine/kernels/vllm_wrappers.py` — 7 kernel wrapper APIs (rms_norm, fused_add_rms_norm, silu_and_mul, rotary_embedding, _get_cos_sin_cache, flash_attn_varlen_func, flash_attn_with_kvcache)
- `engine/tp_layers/linear.py` — QKVColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear APIs
- `engine/tp_layers/distributed.py` — get_tp_size() return value (1 when dist not initialized)
- `engine/tp_layers/embedding.py` — VocabParallelEmbedding, ParallelLMHead (not used in Phase 5, reviewed for future Phase 7)

## Self-Diff Review

- No modifications to `scripts/` directory
- No modifications to existing engine files (`engine/kernels/`, `engine/tp_layers/`)
- Only 2 new files created: `engine/models/__init__.py` and `engine/models/qwen.py`
- Static analysis confirms all 4 required classes present with correct attr names
- 24 blueprint contract checks all PASS (block_size=256, block_table=int32, causal=True/False, vectorized slot_mapping, index_copy_, K/V reshape num_kv_heads, etc.)
- No load_weights(), QwenForCausalLMTP, or QwenTPConfig (those are Phase 7 scope)

## Known Issues

- None. All blueprint contracts met. Self-diff clean.

## Blockers

- None.
