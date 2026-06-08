# Phase 5 Spec Review Report

| Field | Value |
|-------|-------|
| PID | 1052269 |
| Role | spec-reviewer |
| Timestamp | 2026-06-09T00:00:00Z |
| Phase | 5 |
| Files Reviewed | `engine/models/qwen.py`, `engine/models/__init__.py` |
| Blueprint Nodes | `qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP`, `qwen3_tp_model_interfaces.attention`, `qwen3_tp_model_interfaces.decode_forward_pattern`, `qwen3_tp_model_interfaces.prefill_forward_pattern`, `paged_kv_cache_contract`, `flash_attention_integration_contract` |

## Spec Compliance: PASS

---

## Evidence Chain

### 1. block_size = 256
- [paged_kv_cache_contract.hard_rule]: PASS @ `engine/models/qwen.py:146` -- `self._kv_block_size = 256`, hard-coded to 256 (NOT nano-vllm's 16). Used consistently in slot_mapping arithmetic at lines 289-291.
- [paged_kv_cache_contract.kv_cache_format.key_cache]: PASS @ `engine/models/qwen.py:180-187` -- `_key_cache` shape is `[num_blocks, self._kv_block_size, self.num_kv_heads, self.head_dim]` where `_kv_block_size=256`.

### 2. block_table dtype = int32
- [paged_kv_cache_contract.kv_cache_format.block_table]: PASS @ `engine/models/qwen.py:189-191` -- `torch.zeros(1, num_blocks, dtype=torch.int32, device=device)`, explicitly int32 (NOT int64).
- [paged_kv_cache_contract.hard_rule]: PASS -- `block_table 必须 int32` verified at line 189.

### 3. QKV heads usage
- [qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP]: PASS @ `engine/models/qwen.py:106` -- `self.num_heads = cfg.num_attention_heads // tp_size  # 8`.
- [qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP]: PASS @ `engine/models/qwen.py:110` -- `self.num_kv_heads = cfg.num_key_value_heads // tp_size  # 2`.
- [tp_linear_layers.qkv_column_parallel_forward]: PASS @ `engine/models/qwen.py:235-236` (prefill) -- Q uses `self.num_heads` (8), K/V use `self.num_kv_heads` (2). Confirmed at lines 336-337 for decode as well.
- [AGENT_SKILL.md §2.2]: PASS -- "Q reshape: num_heads=8 (per-rank), K/V: num_kv_heads_local=2 (per-rank). Not mixed."

### 4. Prefill K/V source
- [flash_attention_integration_contract.prefill_path.kv_source_correction]: PASS @ `engine/models/qwen.py:231,264,268-272` -- K/V come from `self.qkv_proj(hidden_states)` at line 231, reshaped at lines 235-236, flattened to `k_flat` at line 247 and `v_flat` at line 264, then passed to `flash_attn_varlen_func` at lines 268-272. KV cache is written AFTER attention at lines 293-297. Correct order: projection -> attention -> cache write.
- [flash_attention_integration_contract.prefill_path.hard_rule]: PASS -- "prefill flash_attn_varlen_func 必须使用投影产出的 K,V (非从 KV cache 读取)" verified.

### 5. slot_mapping vectorized
- [paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm.vectorized]: PASS @ `engine/models/qwen.py:287-291` -- Uses `torch.arange` + tensorized arithmetic:
  ```python
  indices = torch.arange(num_tokens, device=device)
  slot_mapping = (self._block_table[0, indices // self._kv_block_size] * self._kv_block_size
                  + (indices % self._kv_block_size))
  ```
  No `.item()` loop. One-line vectorized computation.

### 6. flash_attn_with_kvcache keyword args
- [flash_attention_integration_contract.decode_path]: PASS @ `engine/models/qwen.py:384-392` -- All parameters use keyword args (`cache_seqlens=`, `block_table=`, `softmax_scale=`, `causal=`). Compatible with flash_attn 2.8.3+.
- [flash_attention_integration_contract.decode_path.kernel]: PASS @ `engine/models/qwen.py:391` -- `causal=False` for decode (single token, no causal mask needed).
- [flash_attention_integration_contract.decode_path.softmax_scale]: PASS @ `engine/models/qwen.py:390` -- `softmax_scale=self.scaling` where `self.scaling = self.head_dim ** -0.5` (line 118), matching `1.0/sqrt(128)`.

### 7. RMSNorm returns 2-tuple
- [qwen3_tp_model_interfaces.class_hierarchy]: PASS @ `engine/models/qwen.py:58-79` -- `forward()` signature returns `tuple[torch.Tensor, torch.Tensor | None]`. Without residual returns `(out, None)` at line 74. With residual returns `(x, residual)` at line 79. Always 2-tuple.
- [AGENT_SKILL.md §1 rmsnorm_precision_law]: PASS -- Uses vLLM CUDA kernels (`_rms_norm_kernel` / `_fused_add_rms_norm_kernel` from `engine/kernels/vllm_wrappers`), NOT hand-written PyTorch. The vLLM kernel handles internal precision (FM-016: `self.weight * x_f.to(input_dtype)` path).

### 8. KV cache lazy allocation
- [paged_kv_cache_contract.kv_cache_format]: PASS @ `engine/models/qwen.py:147-149` -- `__init__` sets `self._key_cache = None`, `self._value_cache = None`, `self._block_table = None`. NOT pre-allocating max_blocks.
- [paged_kv_cache_contract.kv_cache_format]: PASS @ `engine/models/qwen.py:172-191` -- `allocate_kv_cache(num_blocks)` method creates tensors on demand.
- [paged_kv_cache_contract.prefill_kv_write]: PASS @ `engine/models/qwen.py:278-279` -- Allocation triggered on first prefill: `if self._key_cache is None: self.allocate_kv_cache(num_blocks_needed)`.

  NOTE: The code allocates `num_blocks_needed = (num_tokens + 255) // 256` (based on current prefill tokens), matching AGENT_SKILL.md's explicit iron law: "严禁一次性 torch.zeros(max_blocks=160, 256, kv_heads, dim) 全量预分配". This differs from the blueprint prefill pseudocode which calculates `max_blocks = (max_seq_len + 255) // 256` (line 1064 of blueprint). The blueprint's own `paged_kv_cache_contract.kv_cache_format.initialization` (line 647) also states `num_blocks=(config.max_position_embeddings+255)//256`. This is a blueprint internal inconsistency -- AGENT_SKILL.md is the higher authority for the lazy-allocation iron law. Recorded as a blueprint information gap below.

### 9. QwenAttentionTP attr names (class_hierarchy alignment)
- [qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP.attrs]: PASS -- All 24 attributes from the blueprint attrs list verified against code:

| # | Blueprint attr | Code line | Match |
|---|---------------|-----------|-------|
| 1 | `self.total_num_heads` | 104 | PASS |
| 2 | `self.total_num_kv_heads` | 105 | PASS |
| 3 | `self.num_heads` | 106 | PASS |
| 4 | `self.num_kv_heads` | 110 | PASS |
| 5 | `self.kv_head_replica` | 113 | PASS |
| 6 | `self.head_dim` | 115 | PASS |
| 7 | `self.q_size` | 116 | PASS |
| 8 | `self.kv_size` | 117 | PASS |
| 9 | `self.scaling` | 118 | PASS |
| 10 | `self.qkv_proj` | 121 | PASS |
| 11 | `self.o_proj` | 127 | PASS |
| 12 | `self.q_norm` | 134 | PASS |
| 13 | `self.k_norm` | 135 | PASS |
| 14 | `self._cu_q` | 138-140 | PASS |
| 15 | `self._cu_k` | 141-143 | PASS |
| 16 | `self._kv_block_size` | 146 | PASS |
| 17 | `self._key_cache` | 147 | PASS |
| 18 | `self._value_cache` | 148 | PASS |
| 19 | `self._block_table` | 149 | PASS |
| 20 | `self._slot_mapping` | 150 | PASS |
| 21 | `self._kv_len_gpu` | 153-155 | PASS |
| 22 | `self._slot_mapping_decode` | 156-160 | PASS |
| 23 | `self._cos_sin_cache_cpu` | 163-165 | PASS |
| 24 | `self._cos_sin_cache_gpu` | 166 | PASS |

### 10. Decode KV write then _kv_len_gpu += 1
- [qwen3_tp_model_interfaces.decode_forward_pattern.kv_len_timing.write]: PASS @ `engine/models/qwen.py:366-378` -- Write-then-increment order:
  - Line 366: `self._slot_mapping_decode[0] = self._kv_len_gpu[0]` (capture current length as slot)
  - Line 374-375: `kc_flat.index_copy_(0, self._slot_mapping_decode, k_write)` (write to cache at that slot)
  - Line 378: `self._kv_len_gpu[0] += 1` (increment AFTER write)
  Correct ordering ensures the newly written token is visible to subsequent attention queries.

### Additional checks

- [AGENT_SKILL.md §1 KV head replication]: PASS @ `engine/models/qwen.py:109-113` -- `if cfg.num_key_value_heads >= tp_size: self.num_kv_heads = cfg.num_key_value_heads // tp_size; else: self.num_kv_heads = 1` with `kv_head_replica`.
- [paged_kv_cache_contract.kv_cache_format.kv_len_gpu]: PASS @ `engine/models/qwen.py:153-155` -- `torch.zeros(1, dtype=torch.int32)` on GPU, not Python int.
- [paged_kv_cache_contract.kv_cache_format.slot_mapping_decode]: PASS @ `engine/models/qwen.py:156-160` -- `torch.zeros(1, dtype=torch.int64)`, matches blueprint dtype int64.
- [qwen3_tp_model_interfaces.attention.decode_attention]: PASS @ `engine/models/qwen.py:383-392` -- `q_attn = q.reshape(1, self.num_heads, self.head_dim)` (3D: [1, num_heads, head_dim]), then `flash_attn_with_kvcache`.
- [flash_attention_integration_contract.prefill_path.kernel]: PASS @ `engine/models/qwen.py:268-272` -- `flash_attn_varlen_func(q_flat, k_flat, v_flat, cu, cu, max_s, max_s, causal=True)` with causal=True for prefill.
- [AGENT_SKILL.md §1 QKV cat order]: PASS -- The code imports `QKVColumnParallelLinear` which internally does `y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)` returning `(q, k, v)` -- Q-K-V order.

### Prefill KV write method note
- [paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm.pseudocode]: The blueprint prefill pseudocode (line 1079) uses `kc_flat.index_copy_(0, slot_mapping, k_flat)`. The code at lines 296-297 uses direct index assignment: `kc_flat[slot_mapping] = k_flat.contiguous()`. This is an intentional improvement consistent with the blueprint's own O5 optimization rule (AGENT_SKILL.md line 608: "prefill KV 写入使用直接索引赋值 kc_flat[slot_mapping] = k_flat，非 index_copy_"). Both methods produce identical results; direct indexing is preferred for prefill (no CUDA graph constraints). The decode path at lines 374-375 correctly uses `index_copy_` as required by the decode contract (line 994 of blueprint).

---

## Blueprint Information Gaps

- [paged_kv_cache_contract.kv_cache_format.initialization] vs [AGENT_SKILL.md §2.2 KV cache lazy alloc]: CONFLICT -- Blueprint initialization (line 647) says allocate `num_blocks=(config.max_position_embeddings+255)//256`. Blueprint prefill pseudocode (line 1064) says `max_blocks = (max_seq_len + 255) // 256`. AGENT_SKILL.md explicitly prohibits pre-allocating max_blocks: "严禁一次性 torch.zeros(max_blocks=160, 256, kv_heads, dim) 全量预分配". The code follows AGENT_SKILL.md's iron law (num_tokens-based allocation), which is the correct behavior for memory efficiency. The blueprint pseudocode should be updated to match the AGENT_SKILL iron law and the code's actual behavior.

---

## Issues Found

None. All 10 key verification items pass. All 24 QwenAttentionTP attributes match the blueprint class_hierarchy. All data flow contracts (prefill/decode patterns, KV cache format, flash attention integration) are satisfied.

---

## Verdict

Spec PASS. Code is consistent with blueprint contracts. Can proceed to verification.

