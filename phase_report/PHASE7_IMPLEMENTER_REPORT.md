# Phase 7 Implementer Report

- **Status**: SUBMITTED
- **Role**: implementer
- **Phase**: 7
- **Timestamp**: 2026-05-30
- **PID**: N/A (not run as Shell `claude -p` sub-agent)

## Implemented

### Files modified

1. **`engine/models/qwen.py`** — appended ~310 lines. No existing code modified.

### New classes and methods

#### QwenTPConfig (dataclass)
- 13 fields matching blueprint `class_hierarchy.QwenTPConfig` exactly
- `from_model_dir(model_dir)` static factory: reads `config.json` dynamically, no hardcoded dimensions
- `head_dim` fallback: `cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])`

#### QwenForCausalLMTP (nn.Module)
- `__init__(cfg, device, dtype)`: construct `embed_tokens` (VocabParallelEmbedding), `layers` (ModuleList[QwenDecoderLayerTP] x num_hidden_layers), `norm` (RMSNorm), `lm_head` (ParallelLMHead)
- `forward(input_ids, past_key_values, position_offset, max_seq_len)`: prefill/decode dispatch matching blueprint `model_forward_pseudocode` exactly
- `load_weights()`: reads `model.safetensors.index.json` -> groups by safetensors file -> `safe_open` -> `_dispatch_weight` -> `_merge_qkv_weights` + `_merge_gate_up_weights` -> `dist.barrier()` + `init_custom_ar()`
- `_parse_hf_key(hf_key)`: parses HF key into (layer_idx, component), handles 3 top-level keys (embed_tokens, norm, lm_head) and layer-level keys
- `_dispatch_weight(hf_key, full, qkv_buffers, gate_up_buffers)`: routes all 12 HF key patterns to correct modules
- `_merge_qkv_weights(qkv_buffers)`: Q-K-V order concat with per-rank ColumnParallel slicing + KV replication
- `_merge_gate_up_weights(gate_up_buffers)`: gate-up order concat with per-rank ColumnParallel slicing

### Imports added
- `get_tp_rank`, `init_custom_ar` from `engine.tp_layers.distributed`
- `VocabParallelEmbedding`, `ParallelLMHead` from `engine.tp_layers.embedding`
- `dataclasses.dataclass`, `pathlib.Path`, `json`, `torch.distributed`, `safetensors.safe_open`

## Blueprint Nodes Read

### JSON paths
- `class_hierarchy.QwenTPConfig` — 13 fields + head_dim_fallback + factory
- `class_hierarchy.QwenForCausalLMTP` — construction chain: embed -> layers -> norm -> lm_head
- `construction_chain` — cfg -> model -> load_weights -> eval -> init_custom_ar
- `qwen_hf_key_mapping` — 12 HF key->attr mappings + Q-K-V order + Gate-Up order + double_shard_guard
- `load_weights_pseudocode` — full loading chain: index.json -> safe_open -> _load_one_weight -> barrier + init_custom_ar
- `qwen3_8b_model_dims` — _verified_config: max_position_embeddings=40960, intermediate_size=12288
- `model_forward_pseudocode` — prefill/decode dispatch in QwenForCausalLMTP.forward()

### Ref docs
- `notebooks-cn/07_improvementPlan/kernel_replacement_plan.md` $B (QKVColumnParallelLinear assembly) — verified

## Self-Diff Review

### Changes to engine/models/qwen.py:
- **Lines 39-45**: Added 6 import lines (get_tp_rank, init_custom_ar, VocabParallelEmbedding, ParallelLMHead, dataclass, Path, json, dist, safe_open)
- **Lines 390-700**: Appended QwenTPConfig dataclass + QwenForCausalLMTP class with all 7 methods

### What was NOT touched:
- Lines 1-388: All existing Phase 1-6 code (RMSNorm, QwenAttentionTP, QwenMLPTP, QwenDecoderLayerTP) — completely unchanged
- `scripts/` directory — no changes
- Any other file in the repository — no changes

### Issues found: None

## Known Issues

None identified. Key correctness points verified:

1. **Q-K-V order**: `torch.cat([q_shard, k_shard, v_shard], dim=0)` — Q first, K middle, V last
2. **Gate-Up order**: `torch.cat([gate_shard, up_shard], dim=0)` — gate first, up second
3. **Double shard guard**: All weight loading goes through existing `load_weight_shard` methods (QKVColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, ParallelLMHead) which already have the guard
4. **KV replication**: Handled in `_merge_qkv_weights` — if `num_key_value_heads < tp_size`, full KV weights are replicated (not sliced)
5. **Dynamic config**: `from_model_dir` reads all fields from `config.json` — no hardcoded values
6. **dist.barrier + init_custom_ar**: Called after all weights loaded, only if `dist.is_initialized()`
7. **Existing code preserved**: QwenAttentionTP, QwenDecoderLayerTP, QwenMLPTP, RMSNorm untouched
