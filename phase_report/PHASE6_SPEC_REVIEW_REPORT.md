# PHASE6_SPEC_REVIEW_REPORT.md

| 字段 | 值 |
|------|-----|
| PID | 3873212 |
| Role | spec-reviewer |
| Timestamp | 2026-06-09T00:00:00Z |
| Phase | 6 |

---

## Spec Compliance: ✅ PASS

---

## Evidence Chain

### Contract 1: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.mlp`

- **JSON Path**: `qwen3_tp_model_interfaces.mlp.gate_up_merged`
  ✅ @ `engine/models/qwen.py:444-451` — `gate_up = self.gate_up_proj(x)` produces `[B, T, 2*intermediate_size/tp]` = `[B, T, 6144]`. Then `silu_and_mul(act, gate_up)` consumed correctly, output `[B, T, intermediate_size/tp]` = `[B, T, 3072]`. Uses `torch.empty(*gate_up.shape[:-1], self.intermediate_per_rank, ...)` (not `torch.empty_like`) per AGENT_SKILL.md line 595 guidance.

- **JSON Path**: `qwen3_tp_model_interfaces.mlp.down_out`
  ✅ @ `engine/models/qwen.py:452` — `self.down_proj(act)` returns `[B, T, hidden_size]` = `[B, T, 4096]`. RowParallelLinear internally does all_reduce_sum.

- **JSON Path**: `qwen3_tp_model_interfaces.mlp._note` (merged gate+up)
  ✅ @ `engine/models/qwen.py:426-431` — Uses `MergedColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False)`, confirmed single-merged-GEMM approach as blueprint requires.

---

### Contract 2: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern`

- **JSON Path**: `decode_forward_pattern.unified_signature.function`
  ✅ @ `engine/models/qwen.py:525-531` — Method name `forward_decode`. Signature: `(self, hidden_states, positions, kv_len, max_seq_len, residual=None)`. Return type: `tuple[torch.Tensor, torch.Tensor]` = `(mlp_out, residual)`.

- **JSON Path**: `decode_forward_pattern.unified_signature.hidden_states`
  ✅ @ `engine/models/qwen.py:336` — `B, S, H = hidden_states.shape  # B=1, S=1`. Contract `[1, 1, hidden_size]` satisfied.

- **JSON Path**: `decode_forward_pattern.unified_signature.return`
  ✅ @ `engine/models/qwen.py:561` — `return mlp_out, residual`. Returns `(hidden_states, residual)` 2-tuple.

- **JSON Path**: `decode_forward_pattern.full_method_body` (decoder layer portion)
  ✅ @ `engine/models/qwen.py:542-561` — `fused_add_rms_norm(hs, residual, self.input_layernorm.weight, self.input_layernorm.eps)` → `self.self_attn.forward_decode(...)` → `fused_add_rms_norm(attn_out, residual, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)` → `self.mlp(attn_out)` → `return mlp_out, residual`. Dataflow matches pseudocode at line level.

- **JSON Path**: `decode_forward_pattern.kv_len_timing.hard_rule` (`.item()` forbidden in forward_decode)
  ✅ @ `engine/models/qwen.py:525-561` — No `.item()` call anywhere in `forward_decode`. The `_kv_len_gpu[0] += 1` is a GPU-side increment (line 385, inside QwenAttentionTP.forward_decode), and `.item()` is reserved for the model-level forward loop (outside compiled region), per contract.

---

### Contract 3: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.prefill_forward_pattern`

- **JSON Path**: `prefill_forward_pattern.layer_forward_pseudocode` (decoder layer)
  ✅ @ `engine/models/qwen.py:478-519` — Exact dataflow match:
  - `residual is None` → `residual = hidden_states.clone()` + `rms_norm(hidden_states, residual, self.input_layernorm.weight, self.input_layernorm.eps)` (lines 493-500)
  - `residual is not None` → `fused_add_rms_norm(hidden_states, residual, self.input_layernorm.weight, self.input_layernorm.eps)` (lines 502-507)
  - `attn_out = self.self_attn.forward(hidden_states, positions, max_seq_len)` (line 509)
  - `fused_add_rms_norm(attn_out, residual, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)` (lines 511-516)
  - `mlp_out = self.mlp(attn_out)` (line 518)
  - `return mlp_out, residual` (line 519)

- **JSON Path**: `prefill_forward_pattern.key_differences_vs_decode[3]` (layer method: forward())
  ✅ @ `engine/models/qwen.py:478` — Prefill uses `forward()`, decode uses `forward_decode()`. Separate methods as required.

---

### Contract 4: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenMLPTP`

- **JSON Path**: `class_hierarchy.QwenMLPTP.attrs[0]`
  ✅ @ `engine/models/qwen.py:426-431` — `self.gate_up_proj = MergedColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False)`. Attribute name, constructor args match blueprint exactly.

- **JSON Path**: `class_hierarchy.QwenMLPTP.attrs[1]`
  ✅ @ `engine/models/qwen.py:432-436` — `self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, bias=False)`. Attribute name, constructor args match blueprint exactly.

---

### Contract 5: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenDecoderLayerTP`

- **JSON Path**: `class_hierarchy.QwenDecoderLayerTP.attrs[0]`
  ✅ @ `engine/models/qwen.py:471` — `self.self_attn = QwenAttentionTP(cfg)`. Name is `self_attn` (NOT `attention`), matches blueprint + HF key mapping convention.

- **JSON Path**: `class_hierarchy.QwenDecoderLayerTP.attrs[1]`
  ✅ @ `engine/models/qwen.py:472` — `self.mlp = QwenMLPTP(cfg)`. Name matches blueprint.

- **JSON Path**: `class_hierarchy.QwenDecoderLayerTP.attrs[2]`
  ✅ @ `engine/models/qwen.py:469` — `self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)`. Name matches blueprint.

- **JSON Path**: `class_hierarchy.QwenDecoderLayerTP.attrs[3]`
  ✅ @ `engine/models/qwen.py:470` — `self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)`. Name matches blueprint.

---

## 四大高发错误专项核验

### 1. FM-003 跨层 weight — 全部 fused_add_rms_norm 调用

| # | 位置 | weight 参数 | 是本层 self.weight? |
|---|------|------------|-------------------|
| 1 | `engine/models/qwen.py:502-507` (prefill, input_norm) | `self.input_layernorm.weight` | ✅ 本层 |
| 2 | `engine/models/qwen.py:511-516` (prefill, post_attn_norm) | `self.post_attention_layernorm.weight` | ✅ 本层 |
| 3 | `engine/models/qwen.py:542-547` (decode, input_norm) | `self.input_layernorm.weight` | ✅ 本层 |
| 4 | `engine/models/qwen.py:553-558` (decode, post_attn_norm) | `self.post_attention_layernorm.weight` | ✅ 本层 |

**结论**: 全部 4 处 `fused_add_rms_norm` 使用本层 `self.input_layernorm.weight` 或 `self.post_attention_layernorm.weight`，无跨层引用。✅

### 2. gate_up 维度

- MergedColumnParallelLinear 内部计算: `intermediate_per_rank = 12288 / 4 = 3072`, `gate_up_out_dim = 2 * 3072 = 6144` — ✅ @ `engine/tp_layers/linear.py:202-203`
- QwenMLPTP 注释标注: `# [B, T, 2*inter_per_rank] = [B, T, 6144]` — ✅ @ `engine/models/qwen.py:444`
- **NOT 6400**. ✅

### 3. Eager 路径 clone()

- `forward_decode()` 方法体 (`engine/models/qwen.py:525-561`): 全文搜索 `.clone()` — **零匹配**。✅

### 4. Residual 链

| 路径 | 条件 | 调用的函数 | 位置 |
|------|------|-----------|------|
| prefill 首层 | `residual is None` | `rms_norm(...)` (非 fused) | `engine/models/qwen.py:495-500` ✅ |
| prefill 后续层 | `residual is not None` | `fused_add_rms_norm(...)` | `engine/models/qwen.py:502-507` ✅ |
| prefill post-attn | — | `fused_add_rms_norm(...)` | `engine/models/qwen.py:511-516` ✅ |
| decode input_norm | — | `fused_add_rms_norm(...)` | `engine/models/qwen.py:542-547` ✅ |
| decode post-attn | — | `fused_add_rms_norm(...)` | `engine/models/qwen.py:553-558` ✅ |

**结论**: Prefill 首层正确使用 `rms_norm`（非 `fused_add_rms_norm`），所有后续层使用 `fused_add_rms_norm`。Residual 链无断裂。✅

### 5. QwenAttentionTP / RMSNorm 未被修改

- `RMSNorm` 类定义: `engine/models/qwen.py:52-86` — Phase 5 产物，未被 Phase 6 修改。✅
- `QwenAttentionTP` 类定义: `engine/models/qwen.py:93-404` — Phase 5 产物，未被 Phase 6 修改。✅
- Phase 6 新增代码始于 line 407（QwenMLPTP + QwenDecoderLayerTP），结束于 line 562，清晰分离。✅

### 6. 类名/属性名精确一致

| 蓝图 class_hierarchy attr | 代码 attr | 位置 | 匹配? |
|---------------------------|-----------|------|-------|
| `QwenMLPTP.gate_up_proj` | `self.gate_up_proj` | `engine/models/qwen.py:426` | ✅ |
| `QwenMLPTP.down_proj` | `self.down_proj` | `engine/models/qwen.py:432` | ✅ |
| `QwenDecoderLayerTP.self_attn` | `self.self_attn` | `engine/models/qwen.py:471` | ✅ |
| `QwenDecoderLayerTP.mlp` | `self.mlp` | `engine/models/qwen.py:472` | ✅ |
| `QwenDecoderLayerTP.input_layernorm` | `self.input_layernorm` | `engine/models/qwen.py:469` | ✅ |
| `QwenDecoderLayerTP.post_attention_layernorm` | `self.post_attention_layernorm` | `engine/models/qwen.py:470` | ✅ |

---

## 全局约束检查

- **rmsnorm_precision_law**: RMSNorm 使用 vLLM kernel (`rms_norm`, `fused_add_rms_norm` from `engine.kernels.vllm_wrappers`), 非手写 PyTorch。✅
- **tp_linear_load_no_double_shard**: `MergedColumnParallelLinear.load_weight_shard()` 在 `engine/tp_layers/linear.py:240-248` 有 `if weight.shape == self.weight.shape` 的 double-shard guard。✅
- **block_table dtype**: `torch.zeros(1, num_blocks, dtype=torch.int32, ...)` — int32, 符合契约。✅
- **block_size**: `self._kv_block_size = 256` — 符合 256 约定。✅

---

## Blueprint Information Gaps

- 🟡 `decode_forward_pattern.full_method_body` 中的 QwenDecoderLayerTP pseudocode 包含 `if res is None: res = hs.clone(); rms_norm(...)` 的防御分支。代码实现 (`engine/models/qwen.py:539-547`) 省略此分支，注释声明 "residual is never None in decode — it is carried from prefill stage"。该逻辑正确（prefill 结束后 residual 始终为 Tensor），但严格来说偏离了蓝图 pseudocode 的防御性写法。功能上等效。建议在 blueprints 中标明此分支为 dead-code safety net，或实现补上防御分支以严格匹配 blueprint。

---

## Issues Found

无。

---

## Conclusion

Spec 审查通过。代码与 `inference_blueprint.json` 中 Phase 6 相关的全部契约节点一致：
- `qwen3_tp_model_interfaces.mlp` — gate_up→silu_and_mul→down 链完全匹配
- `qwen3_tp_model_interfaces.decode_forward_pattern` — forward_decode 签名、fused_add_rms_norm 链、零 clone 完全匹配
- `qwen3_tp_model_interfaces.prefill_forward_pattern` — residual 链 (None→rms_norm, else→fused_add_rms_norm) 完全匹配
- `qwen3_tp_model_interfaces.class_hierarchy.QwenMLPTP` — 属性名 gate_up_proj / down_proj 精确一致
- `qwen3_tp_model_interfaces.class_hierarchy.QwenDecoderLayerTP` — 属性名 self_attn / mlp / input_layernorm / post_attention_layernorm 精确一致

四大高发错误全部核验通过：FM-003 跨层 weight 无违规、gate_up=6144 (非 6400)、forward_decode 零 clone、residual 链无断裂。

QwenAttentionTP 和 RMSNorm（Phase 5 代码）未被修改。

**可移交 verification。**
