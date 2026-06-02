# Phase 3 Spec Review Report

```
PID:       907464
Role:      spec-reviewer
Timestamp: 2026-05-30T05:17:19.888436
Phase:     3
```

## Spec Compliance: ✅ PASS

---

## Evidence Chain (逐条列出核验过的 JSON Path)

### 1. ColumnParallelLinear — Shape Contract

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.column_parallel_linear`
- ✅ @ `linear.py:52-53` — weight_shape `[out/tp, in]`: `self.weight = nn.Parameter(torch.empty(local_out, in_features))` where `local_out = out_features // self.tp_size`
- ✅ @ `linear.py:59-64` — input `[B, T, in]`, output_no_gather `[B, T, out/tp]`: `F.linear(x, self.weight, self.bias)` returns `[B, T, local_out]`
- ✅ @ `linear.py:62-63` — output_with_gather `[B, T, out]`: `all_gather_last_dim(y)` when `self.gather_output and self.tp_size > 1`

### 2. RowParallelLinear — Shape Contract

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.row_parallel_linear`
- ✅ @ `linear.py:108` — weight_shape `[out, in/tp]`: `self.weight = nn.Parameter(torch.empty(out_features, local_in))` where `local_in = in_features // self.tp_size`
- ✅ @ `linear.py:114-120` — forward: `F.linear(x, self.weight)` → partial_output `[B, T, out]` → `all_reduce_sum(y)` → `+self.bias` → output_after_all_reduce `[B, T, out]`
- ✅ @ `linear.py:114-120` — Forward exactly matches blueprint `row_parallel_linear_forward` pseudocode: `F.linear` (no bias in call) → `all_reduce_sum` → `+self.bias`

### 3. QKVColumnParallelLinear — Forward Pseudocode

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.qkv_column_parallel_forward`
- ✅ @ `linear.py:182-183` — Merged QKV weight shape `[q_size + 2*kv_size, hidden_size]`: `local_out = self.q_size + 2 * self.kv_size; self.weight = nn.Parameter(torch.empty(local_out, hidden_size))`
- ✅ @ `linear.py:189-198` — Forward flow matches blueprint: `F.linear` → optional `all_gather_last_dim` → `y.split([q_size, kv_size, kv_size], dim=-1)` → returns `(q, k, v)`
- ✅ @ `linear.py:174-175` — Per-rank head counts: `num_heads = total_num_heads // tp_size`; `num_kv_heads = max(1, total_num_kv_heads // tp_size)` matches blueprint KV head replication rule

### 4. MergedColumnParallelLinear — Gate+Up Merge

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.qkv_column_parallel_forward` (MergedColumnParallelLinear pseudocode section)
- ✅ @ `linear.py:270-273` — Weight shape `[2*out_features/tp, in_features]`: `local_out = 2 * out_features // self.tp_size; self.weight = nn.Parameter(torch.empty(local_out, in_features))`
- ✅ @ `linear.py:279-287` — Forward: `F.linear` → optional `all_gather_last_dim` → return `[B, T, 2*out_features/tp]` (first half=gate, second half=up)
- ✅ @ `linear.py:272` — For Qwen3-8B TP=4 (intermediate_size=12288): `local_out = 2*12288//4 = 6144`. Matches blueprint `tp4_per_rank.gate_up_weight: [6144, 4096]`

### 5. Double Shard Guard — ColumnParallelLinear

- **JSON Path**: `model_layer.architecture_knowledge_base.global_primitives_constraints.tp_linear_load_no_double_shard`
- ✅ @ `linear.py:73-79` — Guard exists: `if shard.shape == self.weight.shape: self.weight.data.copy_(shard)` else slice by `tp_rank` along dim 0
- ✅ @ `linear.py:76-79` — Slicing logic: `start = tp_rank * local_out; end = start + local_out; ... copy_(shard[start:end, :])`

### 6. Double Shard Guard — RowParallelLinear

- **JSON Path**: `model_layer.architecture_knowledge_base.global_primitives_constraints.tp_linear_load_no_double_shard`
- ✅ @ `linear.py:129-130` — Guard exists: `if shard.shape == self.weight.shape: self.weight.data.copy_(shard)` else slice by `tp_rank` along dim 1
- ✅ @ `linear.py:132-135` — Slicing logic: `local_in = in_features // tp_size; start = tp_rank * local_in; ... copy_(shard[:, start:end])`

### 7. Double Shard Guard — QKVColumnParallelLinear

- **JSON Path**: `model_layer.architecture_knowledge_base.global_primitives_constraints.tp_linear_load_no_double_shard`
- ✅ @ `linear.py:212-239` — Guard exists: `if shard.shape == self.weight.shape: self.weight.data.copy_(shard)` else Q-K-V section slicing
- ✅ @ `linear.py:224-238` — Full weight layout correctly partitioned: Q section → K section (offset `total_q`) → V section (offset `total_q + total_kv`) → `torch.cat([q_shard, k_shard, v_shard], dim=0)`

### 8. Double Shard Guard — MergedColumnParallelLinear

- **JSON Path**: `model_layer.architecture_knowledge_base.global_primitives_constraints.tp_linear_load_no_double_shard`
- ✅ @ `linear.py:300-316` — Guard exists: `if shard.shape == self.weight.shape: self.weight.data.copy_(shard)` else gate-up section slicing
- ✅ @ `linear.py:306-315` — Full weight layout correctly partitioned: gate section (0..out_features-1) → up section (out_features..2*out_features-1) → `torch.cat([gate_shard, up_shard], dim=0)`

### 9. Qwen3-8B Model Dimensions — gate_up

- **JSON Path**: `model_layer.architecture_knowledge_base.qwen_series_dense.qwen3_8b_model_dims`
- ✅ @ `linear.py:270-273` — `out_features` = intermediate_size = 12288 (passed by caller from config.json). Internally: `local_out = 2 * 12288 // tp_size`. For TP=4: `local_out = 6144`. Weight `[6144, 4096]` matches `tp4_per_rank.gate_up_weight: [6144, 4096]`.
- ✅ @ `linear.py:270-273` — All dimensions are constructed from parameters passed by the caller (no hardcoded values in linear.py). Complies with `engine_routing_contract`: "TP Runner 初始化时必须动态读取 config.json 提取全部维度参数。严禁硬编码。"

### 10. QKV Cat Order — Q-K-V (Encoding Iron Law)

- **JSON Path**: Encoding iron law from `AGENT_SKILL.md §1`: "QKV cat 顺序 Q-K-V（非 K-Q-V）"
- ✅ @ `linear.py:149` — Docstring: "Q/K/V order: Q first (front), K middle, V last."
- ✅ @ `linear.py:197` — Forward split: `y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)` → Q, K, V
- ✅ @ `linear.py:238` — Weight loading: `torch.cat([q_shard, k_shard, v_shard], dim=0)` → Q-K-V

### 11. Gate-Up Cat Order — gate-up (Encoding Iron Law)

- **JSON Path**: Encoding iron law from `AGENT_SKILL.md §1`: "Gate-Up cat 顺序 gate-up（非 up-gate）"
- ✅ @ `linear.py:248-249` — Docstring: "first half=gate, second half=up"
- ✅ @ `linear.py:306-315` — Weight loading: gate section first (offset 0..out_features-1), up section second (offset out_features..2*out_features-1) → `torch.cat([gate_shard, up_shard], dim=0)` → gate-up

### 12. Class Hierarchy — QwenAttentionTP (QKV + O projections)

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP`
- ✅ @ `linear.py:162-163` — Constructor signature `__init__(self, hidden_size, head_dim, total_num_heads, total_num_kv_heads, ...)` matches blueprint call `QKVColumnParallelLinear(cfg.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads)`
- ✅ @ `linear.py:100-108` — RowParallelLinear for o_proj matches blueprint `RowParallelLinear(self.total_num_heads * self.head_dim, cfg.hidden_size, bias=False)`

### 13. Class Hierarchy — QwenMLPTP (Gate-Up + Down)

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenMLPTP`
- ✅ @ `linear.py:262-273` — MergedColumnParallelLinear constructor signature `__init__(self, in_features, out_features, ...)` where `out_features` = intermediate_size (full). Matches blueprint `MergedColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False)`
- ✅ @ `linear.py:100-108` — RowParallelLinear for down_proj: `(in_features=intermediate_size, out_features=hidden_size)` matches blueprint `RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, bias=False)`

### 14. KV Head Replication — Edge Case Safety

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP` (KV head replication rule)
- ✅ @ `linear.py:175` — `self.num_kv_heads = max(1, total_num_kv_heads // self.tp_size)`
  - Normal case (Qwen3-8B TP=4): `max(1, 8//4) = 2` ✅
  - Edge case (TP=16): `max(1, 8//16) = max(1, 0) = 1` ✅ (replication enforced)
  - Matches blueprint pseudocode: `if cfg.num_key_value_heads >= tp_size: num_kv_heads = ... // tp_size else: num_kv_heads = 1`

---

## Issues Found: None

所有 14 条契约逐条核验通过，无 FAIL 项。

---

## Blueprint Information Gaps (if any)

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.qkv_column_parallel_forward` MergedColumnParallelLinear 伪代码注释
- 🟡 @ `inference_blueprint.json:908` — 注释写 `# [B, T, 2*intermediate/tp] e.g. [1,1,6400]`，但 `qwen3_8b_model_dims.tp4_per_rank.gate_up_weight` 明确为 `[6144, 4096]`。`6400` 对应的 intermediate_size 应为 `6400*2/2*4 = 12800`，而非已验证的 `12288`。此差异在同蓝图 line 1753 处已自纠：`"gate_up_total=6144 非 6400，inter_per_rank=3072 非 3200"`。代码实现动态计算 `local_out = 2 * out_features // tp_size` 得到正确值 6144，无影响。建议：修正 line 908 的注释为 `e.g. [1,1,6140]→应改 [1,1,6144]`。

---

## Summary

Spec 审查通过，代码与蓝图契约一致，可移交 verification。
