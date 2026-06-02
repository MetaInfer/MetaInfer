# Phase 5 Spec Review Report

- **PID**: 939335
- **Role**: spec-reviewer
- **Timestamp**: 2026-05-30T00:00:00Z
- **Phase**: 5
- **Files Reviewed**: `engine/models/qwen.py`, `engine/models/__init__.py`

---

## Spec Compliance: ✅ PASS

---

## Evidence Chain

### 1. class_hierarchy.QwenAttentionTP — Exact Attribute Names

| JSON Path | Status | File:Line | Evidence |
|-----------|--------|-----------|----------|
| `QwenAttentionTP.total_num_heads` | ✅ | qwen.py:103 | `self.total_num_heads = cfg.num_attention_heads` |
| `QwenAttentionTP.total_num_kv_heads` | ✅ | qwen.py:104 | `self.total_num_kv_heads = cfg.num_key_value_heads` |
| `QwenAttentionTP.num_heads` | ✅ | qwen.py:107 | `self.num_heads = cfg.num_attention_heads // tp_size` |
| `QwenAttentionTP.num_kv_heads` | ✅ | qwen.py:110-115 | KV head replication logic: `cfg.num_key_value_heads >= tp_size → // tp_size`, else `num_kv_heads=1` |
| `QwenAttentionTP.head_dim` | ✅ | qwen.py:117 | `self.head_dim = cfg.head_dim` |
| `QwenAttentionTP.q_size` | ✅ | qwen.py:118 | `self.q_size = self.num_heads * self.head_dim` |
| `QwenAttentionTP.kv_size` | ✅ | qwen.py:119 | `self.kv_size = self.num_kv_heads * self.head_dim` |
| `QwenAttentionTP.scaling` | ✅ | qwen.py:120 | `self.scaling = self.head_dim ** -0.5` |
| `QwenAttentionTP.qkv_proj` | ✅ | qwen.py:124-127 | `QKVColumnParallelLinear(hidden_size, head_dim, total_num_heads, total_num_kv_heads, ...)` |
| `QwenAttentionTP.o_proj` | ✅ | qwen.py:130-132 | `RowParallelLinear(total_num_heads * head_dim, hidden_size, ...)` |
| `QwenAttentionTP.q_norm` | ✅ | qwen.py:135 | `RMSNorm(self.head_dim, cfg.rms_norm_eps)` |
| `QwenAttentionTP.k_norm` | ✅ | qwen.py:136 | `RMSNorm(self.head_dim, cfg.rms_norm_eps)` |
| `QwenAttentionTP._kv_block_size` | ✅ | qwen.py:143 | `self._kv_block_size = 256` |
| `QwenAttentionTP._key_cache` | ✅ | qwen.py:144 | `self._key_cache = None` |
| `QwenAttentionTP._value_cache` | ✅ | qwen.py:145 | `self._value_cache = None` |
| `QwenAttentionTP._block_table` | ✅ | qwen.py:146 | `self._block_table = None` |
| `QwenAttentionTP._kv_len_gpu` | ✅ | qwen.py:150 | `register_buffer('_kv_len_gpu', torch.zeros(1, dtype=torch.int32), persistent=False)` |
| `QwenAttentionTP._slot_mapping_decode` | ✅ | qwen.py:153 | `register_buffer('_slot_mapping_decode', torch.zeros(1, dtype=torch.int64), persistent=False)` |
| `QwenAttentionTP._cos_sin_cache` | ✅ | qwen.py:156-158 | `_cos_sin_cache_cpu = _get_cos_sin_cache(...)`; `_cos_sin_cache_gpu = None` |
| `QwenAttentionTP._cu_q / _cu_k` | ✅ | qwen.py:139-140 | `register_buffer('_cu_q', torch.tensor([0,0], dtype=torch.int32), persistent=False)`; same for `_cu_k` |

### 2. paged_kv_cache_contract

| JSON Path | Status | File:Line | Evidence |
|-----------|--------|-----------|----------|
| `kv_cache_format.key_cache` | ✅ | qwen.py:194-196 | `torch.zeros(max_blocks, 256, self.num_kv_heads, self.head_dim, ...)` — shape `[num_blocks, 256, num_kv_heads, head_dim]` bf16 |
| `kv_cache_format.value_cache` | ✅ | qwen.py:197 | `torch.zeros_like(self._key_cache)` |
| `kv_cache_format.block_table` | ✅ | qwen.py:198-199 | `torch.zeros(1, max_blocks, dtype=torch.int32, ...)` — shape `[1, max_blocks] int32` |
| `kv_cache_format.kv_len_gpu` | ✅ | qwen.py:150 | `torch.zeros(1, dtype=torch.int32)` — shape `[1] int32` |
| `kv_cache_format.max_blocks_formula` | ✅ | qwen.py:193 | `max_blocks = (max_seq_len + 255) // 256` |
| `slot_mapping_algorithm.vectorized` | ✅ | qwen.py:211-212 | `indices = torch.arange(num_tokens, ...); slot_mapping = block_table[0, indices // 256] * 256 + (indices % 256)` — 向量化，无 for-loop .item() |
| `prefill_kv_write.integrated_timeline` | ✅ | qwen.py:175-216 | Prefill 顺序: qkv_proj → RoPE → flash_attn_varlen_func(K,V from proj) → index_copy_ to cache |
| `decode_kv_write` | ✅ | qwen.py:255 | `self._slot_mapping_decode[0] = self._kv_len_gpu[0]` → `index_copy_(0, slot_mapping_decode, k/v_write)` → `kv_len_gpu[0] += 1` |
| `decode_attention` | ✅ | qwen.py:266-269 | `flash_attn_with_kvcache(q_attn, _key_cache, _value_cache, _kv_len_gpu, _block_table, scaling, causal=False)` |
| `full_reshape_chain` | ✅ | qwen.py:213-216 | `kc_flat = _key_cache.view(-1, num_kv_heads, head_dim)` → `index_copy_(0, slot_mapping, k_flat)` |

### 3. flash_attention_integration_contract

| JSON Path | Status | File:Line | Evidence |
|-----------|--------|-----------|----------|
| `prefill_path.kernel` | ✅ | qwen.py:204-205 | `flash_attn_varlen_func(q_flat, k_flat, v_flat, cu, cu, num_tokens, num_tokens, causal=True)` |
| `prefill_path.kv_source` | ✅ | qwen.py:175-205 | K/V from `qkv_proj(hidden_states)` → used in flash_attn_varlen_func → THEN index_copy_ to cache. NOT from cache read. |
| `decode_path.kernel` | ✅ | qwen.py:266-269 | `flash_attn_with_kvcache(q_attn, _key_cache, _value_cache, _kv_len_gpu, _block_table, scaling, causal=False)` |
| `decode_path.q_format` | ✅ | qwen.py:264 | `q.reshape(1, 1, self.num_heads, self.head_dim)` — `[1, 1, num_heads, head_dim]` |
| `decode_path.softmax_scale` | ✅ | qwen.py:269 | `self.scaling` = `self.head_dim ** -0.5` (line 120) |
| `nocompile scope note` | ✅ | qwen.py:31-32 | `flash_attn_with_kvcache` imported directly (not custom_op) — correct per scope `nocompile` |

### 4. prefill_forward_pattern — 逐行对照

| Blueprint Pseudocode Step | Status | File:Line | Code Match |
|---------------------------|--------|-----------|------------|
| `def forward(self, hidden_states, positions, max_seq_len)` | ✅ | qwen.py:161 | Exact signature match |
| `B, S, H = hidden_states.shape` | ✅ | qwen.py:168 | `B, S, H = hidden_states.shape  # B=1` |
| `q, k, v = self.qkv_proj(hidden_states)` | ✅ | qwen.py:175 | `q, k, v = self.qkv_proj(hidden_states)` |
| `q = q.view(B, S, self.num_heads, self.head_dim)` | ✅ | qwen.py:176 | `[1,S,8,128]` |
| `k = k.view(B, S, self.num_kv_heads, self.head_dim)` | ✅ | qwen.py:177 | `[1,S,2,128]` — **uses num_kv_heads, NOT num_heads** |
| `v = v.view(B, S, self.num_kv_heads, self.head_dim)` | ✅ | qwen.py:178 | `[1,S,2,128]` |
| `q = self.q_norm(q); k = self.k_norm(k)` | ✅ | qwen.py:181-182 | `q = self.q_norm(q); k = self.k_norm(k)` |
| `rotary_embedding(positions, q_flat, k_flat, ...)` | ✅ | qwen.py:188-189 | `rotary_embedding(positions, q_flat, k_flat, self.head_dim, self._cos_sin_cache_gpu, is_neox=True)` |
| KV cache lazy alloc | ✅ | qwen.py:192-199 | `if self._key_cache is None: ... torch.zeros(max_blocks, 256, ...)` |
| `flash_attn_varlen_func(q, k, v, cu, cu, ..., causal=True)` | ✅ | qwen.py:204-205 | Exact match |
| block_table + slot_mapping + index_copy_ | ✅ | qwen.py:208-216 | **VECTORIZED** slot_mapping (no for-loop .item()) — uses blueprint's preferred vectorized variant |
| `kv_len_gpu[0] = num_tokens` | ✅ | qwen.py:217 | `self._kv_len_gpu[0] = num_tokens` |
| `out.view(B, S, q_size) → o_proj` | ✅ | qwen.py:220-221 | `out.reshape(B, S, self.q_size); return self.o_proj(out)` |
| `QwenDecoderLayerTP.forward` residual chain | ✅ | qwen.py:340-359 | rms_norm → self_attn.forward → fused_add_rms_norm → mlp → return (mlp_out, res) |

### 5. decode_forward_pattern — 逐行对照 full_method_body

| Blueprint Pseudocode Step | Status | File:Line | Code Match |
|---------------------------|--------|-----------|------------|
| `def forward_decode(self, hidden_states, positions, kv_len, max_seq_len)` | ✅ | qwen.py:223 | Exact signature match |
| `q, k, v = self.qkv_proj(hidden_states)` | ✅ | qwen.py:237 | Exact match |
| `q = q.view(B, S, self.num_heads, self.head_dim)` | ✅ | qwen.py:238 | `[1,1,8,128]` |
| `k = k.view(B, S, self.num_kv_heads, self.head_dim)` | ✅ | qwen.py:239 | `[1,1,2,128]` — **uses num_kv_heads** |
| `q = self.q_norm(q); k = self.k_norm(k)` | ✅ | qwen.py:243-244 | Exact match |
| `q_flat = q.reshape(S, self.num_heads, self.head_dim)` | ✅ | qwen.py:247 | `[S, 8, 128]` |
| `k_flat = k.reshape(S, self.num_kv_heads, self.head_dim)` | ✅ | qwen.py:248 | `[S, 2, 128]` |
| `rotary_embedding(positions, q_flat, k_flat, ...)` | ✅ | qwen.py:249-250 | is_neox=True |
| `_slot_mapping_decode[0] = _kv_len_gpu[0]` | ✅ | qwen.py:255 | Exact match |
| `k_write = k.reshape(1, num_kv_heads, head_dim)` | ✅ | qwen.py:256-257 | Exact match |
| `kc_flat.index_copy_(0, slot_mapping_decode, k_write)` | ✅ | qwen.py:260-261 | Exact match |
| `_kv_len_gpu[0] += 1` | ✅ | qwen.py:262 | Exact match |
| `flash_attn_with_kvcache(q, kc, vc, kv_len, block_table, scale, causal=False)` | ✅ | qwen.py:266-269 | Direct import (nocompile scope) |
| `out.reshape(B, S, q_size) → o_proj` | ✅ | qwen.py:272-273 | Exact match |
| `QwenDecoderLayerTP.forward_decode` residual chain | ✅ | qwen.py:361-381 | Clone(1st) → fused_add_rms_norm → attn.decode → fused_add_rms_norm → mlp → return (mlp_out, res) |

### 6. ⚠️ Phase 5 高发错误逐条检查

| # | 检查项 | Status | File:Line | Evidence |
|---|--------|--------|-----------|----------|
| 1 | **block_size=256（非16）** | ✅ | qwen.py:143, 193-195 | `_kv_block_size = 256`; `max_blocks = (max_seq_len + 255) // 256`; `torch.zeros(max_blocks, 256, ...)` |
| 2 | **block_table dtype=int32（非int64）** | ✅ | qwen.py:198-199, 210 | `torch.zeros(1, max_blocks, dtype=torch.int32, ...)`; `torch.arange(num_blocks, dtype=torch.int32, ...)` |
| 3 | **K/V reshape 用 num_kv_heads（非 num_heads）** | ✅ | qwen.py:177-178, 239-240 | Prefill: `k.view(B,S,num_kv_heads,head_dim)`; Decode: same. All 4 locations use `num_kv_heads` not `num_heads` |
| 4 | **prefill K/V 来自 qkv_proj 产出（非 cache 读取）** | ✅ | qwen.py:175-216 | qkv_proj → flash_attn_varlen_func(K,V from proj) → index_copy_ to cache. Correct order. |
| 5 | **slot_mapping 向量化（非 for-loop .item()）** | ✅ | qwen.py:211-212 | `indices = torch.arange(...); slot_mapping = block_table[0, indices//256] * 256 + (indices%256)` — one-line vectorized, no .item() |

### 7. RMSNorm — vLLM kernel wrapper contract

| JSON Path | Status | File:Line | Evidence |
|-----------|--------|-----------|----------|
| `rmsnorm_precision_law` | ✅ | qwen.py:46-73 | Uses vLLM `rms_norm` CUDA kernel (fp32 internal). `out` pre-allocated, `input` contiguous. |
| `fused_add_rms_norm` dual in-place | ✅ | qwen.py:72 | `fused_add_rms_norm(x.contiguous(), residual.contiguous(), self.weight, self.eps)` |
| All `fused_add_rms_norm` use `self.*layernorm.weight` | ✅ | qwen.py:349, 351, 355-356, 369, 374, 377-378 | Every call uses `self.input_layernorm.weight` or `self.post_attention_layernorm.weight` — no cross-layer references |

### 8. QwenMLPTP — class_hierarchy alignment

| JSON Path | Status | File:Line | Evidence |
|-----------|--------|-----------|----------|
| `QwenMLPTP.gate_up_proj` | ✅ | qwen.py:293-295 | `MergedColumnParallelLinear(hidden_size, intermediate_size, ...)` |
| `QwenMLPTP.down_proj` | ✅ | qwen.py:296-298 | `RowParallelLinear(intermediate_size, hidden_size, ...)` |
| `mlp chain: gate_up → silu_and_mul → down` | ✅ | qwen.py:300-305 | `gate_up = gate_up_proj(x)` → `silu_and_mul(out, gate_up)` → `down_proj(out)` |

### 9. QwenDecoderLayerTP — class_hierarchy alignment

| JSON Path | Status | File:Line | Evidence |
|-----------|--------|-----------|----------|
| `QwenDecoderLayerTP.self_attn` | ✅ | qwen.py:334 | `QwenAttentionTP(cfg, tp_size=tp_size)` — **exact name `.self_attn`** (not `.attention`) |
| `QwenDecoderLayerTP.mlp` | ✅ | qwen.py:335 | `QwenMLPTP(cfg, tp_size=tp_size)` — **exact name `.mlp`** |
| `QwenDecoderLayerTP.input_layernorm` | ✅ | qwen.py:336 | `RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)` |
| `QwenDecoderLayerTP.post_attention_layernorm` | ✅ | qwen.py:337 | `RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)` |

---

## Issues Found

**None.** 所有契约节点均已通过核验，代码与蓝图精确对齐。

---

## Blueprint Information Gaps

**None detected within Phase 5 scope.** 蓝图的 prefill_forward_pattern.pseudocode 中使用了 for-loop `.item()` 的 slot_mapping 实现（非向量化版本），但同时提供了 vectorized 替代方案。代码正确选择了向量化版本，此为蓝图自身的双轨描述，非信息断裂。

---

## Cross-Cutting Observations

1. **dtype 灵活性**: KV cache allocation 使用 `hidden_states.dtype` 而非伪代码中的 `torch.bfloat16` 硬编码。功能等价且更通用，不违反契约。
2. **cos_sin_cache lazy GPU transfer**: 代码在 `forward()` 和 `forward_decode()` 开头均执行 `if self._cos_sin_cache_gpu is None: ...to(device)`。此 lazy 模式在 `__init__` 中的 `_cos_sin_cache_cpu` 创建时已定义，属于蓝图 `cos_sin_cache_strategy` 的忠实实现。
3. **nocompile scope**: 代码使用 `flash_attn_with_kvcache` 直接导入（非 custom_op 注册），与蓝图 `_scope_note` 中 "nocompile 场景下 flash_attn_with_kvcache 直接 from flash_attn.flash_attn_interface import" 一致。
4. **`engine/models/__init__.py`**: 仅含模块注释，无契约项。Pass by default。

---

## Verdict

**Spec 审查通过，代码与蓝图契约一致，可移交 verification。**
