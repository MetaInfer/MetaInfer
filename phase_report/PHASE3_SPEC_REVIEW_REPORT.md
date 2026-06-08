# Phase 3 Spec Review Report

| Field | Value |
|-------|-------|
| PID | 1810157 |
| Role | spec-reviewer |
| Timestamp | 2026-06-09T00:00:00Z |
| Phase | 3 |
| Track | 完整串行路径（impl → spec → verify） |

---

## Spec Compliance: **PASS**

所有 14 项关键核验全部通过，代码与 `inference_blueprint.json` 契约精确一致。

---

## Evidence Chain

### 1. ColumnParallelLinear

- **`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.column_parallel_linear.weight_shape`**: **PASS** @ `linear.py:56-58`
  - 核验: `weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))` where `out_features_per_rank = out_features // tp_size`
  - 即 weight shape = `[out/tp, in]`，与蓝图 `"[out/tp, in]"` 一致。

- **`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.column_parallel_linear.forward`**: **PASS** @ `linear.py:70-72`
  - 核验: `y = F.linear(x, self.weight, self.bias)` + 可选 `all_gather_last_dim(y)`
  - `gather_output` 参数名与蓝图一致，`all_gather_last_dim` 在 `tp_size > 1` 且 `gather_output=True` 时调用。

- **`framework_layer.todo_generation_playbook.phase_3_tp_linear.load_weight_shard double shard guard`**: **PASS** @ `linear.py:84-92`
  - 核验: `if weight.shape == self.weight.shape:` → 直接 `copy_`（防双切片）
  - else 分支: `weight[r * per_rank : (r + 1) * per_rank, :]` 沿 dim=0 切片

### 2. RowParallelLinear

- **`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.row_parallel_linear.weight_shape`**: **PASS** @ `linear.py:128-130`
  - 核验: `weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))`
  - 即 weight shape = `[out, in/tp]`，与蓝图 `"[out, in/tp]"` 一致。

- **`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.row_parallel_linear_forward`**: **PASS** @ `linear.py:136-146`
  - 核验: `y = F.linear(x, self.weight, None)` → `y = all_reduce_sum(y)` → `if self.bias is not None: y = y + self.bias`
  - `all_reduce_sum` 调用 1 次，bias 在 reduce 之后加，与蓝图伪代码（lines 923-927）完全一致。

- **`framework_layer.todo_generation_playbook.phase_3_tp_linear.load_weight_shard dim=1 guard`**: **PASS** @ `linear.py:148-165`
  - 核验: `if weight.shape == self.weight.shape` guard 存在 (line 157)
  - else 分支: `weight[:, r * per_rank : (r + 1) * per_rank]` 沿 dim=1 切片

### 3. MergedColumnParallelLinear

- **`model_layer.architecture_knowledge_base.qwen_series_dense.qwen3_8b_model_dims.tp4_per_rank.gate_up_weight`**: **PASS** @ `linear.py:202-207`
  - 核验: `intermediate_per_rank = intermediate_size // tp_size = 12288 // 4 = 3072`
  - `gate_up_out_dim = 2 * intermediate_per_rank = 2 * 3072 = 6144`
  - weight shape = `[6144, hidden_size]` = `[6144, 4096]`，与蓝图 `"[6144,4096]"` 一致。
  - 非 `[6400, 4096]`（6400 源自旧 intermediate_size=12800，代码已正确使用 12288）。

- **`framework_layer.todo_generation_playbook.phase_3_tp_linear.merged_load_weight_shard`**: **PASS** @ `linear.py:224-252`
  - 核验: gate shard 从 `weight[r*per_rank:(r+1)*per_rank]`，up shard 从 `weight[inter + r*per_rank: inter + (r+1)*per_rank]`
  - cat 顺序: `torch.cat([gate_shard, up_shard], dim=0)` → gate 在前、up 在后（gate-up 顺序）
  - guard: `if weight.shape == self.weight.shape` (line 240)

- **`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.qkv_column_parallel_forward` (merged no gather)**: **PASS** @ `linear.py:214-222`
  - 核验: forward 只做 `F.linear(x, self.weight)`，无 all_gather 调用
  - 注释明确说明 `gather_output=False for gate_up → no all_gather`

### 4. QKVColumnParallelLinear

- **`model_layer.architecture_knowledge_base.qwen_series_dense.qwen3_8b_model_dims.tp4_per_rank.qkv_weight`**: **PASS** @ `linear.py:295-309`
  - 核验: `num_heads = 32//4 = 8` → `q_size = 8*128 = 1024`
  - `num_kv_heads = max(1, 8//4) = 2` → `kv_size = 2*128 = 256`
  - `out_features_per_rank = 1024 + 2*256 = 1536`
  - weight shape = `[1536, 4096]`，与蓝图 `"[1536,4096]"` 一致。

- **`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.qkv_column_parallel_forward` split order**: **PASS** @ `linear.py:329`
  - 核验: `y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)` → Q first, then K, then V
  - 代码注释明确标记 "STRICTLY Q-K-V, NOT K-Q-V"

- **`framework_layer.todo_generation_playbook.phase_3_tp_linear.qkv_load_weight_shard`**: **PASS** @ `linear.py:332-375`
  - 核验: Q 从 `[r*q_per_rank, r*q_per_rank + q_per_rank]` 提取
  - K 从 `[k_offset + r*kv_per_rank, k_offset + (r+1)*kv_per_rank]` 提取 (k_offset = 4096)
  - V 从 `[v_offset + r*kv_per_rank, v_offset + (r+1)*kv_per_rank]` 提取 (v_offset = 5120)
  - cat 顺序: `torch.cat([q_shard, k_shard, v_shard], dim=0)` — Q-K-V
  - guard: `if weight.shape == self.weight.shape` (line 350)

### 5. General Requirements

- **Class names vs blueprint `class_hierarchy`**: **PASS** @ `linear.py:23,99,172,259`
  - `ColumnParallelLinear`, `RowParallelLinear`, `MergedColumnParallelLinear`, `QKVColumnParallelLinear` 全部与蓝图精确一致。

- **Import source**: **PASS** @ `linear.py:16`
  - `from engine.tp_layers.distributed import all_reduce_sum, all_gather_last_dim, _get_world_size`
  - 从本工程导入，非外部 pip 包。`distributed.py` 中的实现逐行验证：
    - `all_gather_last_dim` 使用 `dist.all_gather(outs, x)` + `torch.cat(outs, dim=-1)` (distributed.py:220-234) — 符合编码铁律
    - `all_reduce_sum` 含 CustomAR P2P → NCCL fallback (distributed.py:169-204)

- **q_size / kv_size formula**: **PASS** @ `linear.py:295-303`
  - `q_size = self.num_heads * head_dim` where `num_heads = total_num_heads // tp_size = 8`
  - `kv_size = self.num_kv_heads * head_dim` where `num_kv_heads = total_num_kv_heads // tp_size = 2` (when tp <= num_kv_heads)
  - `q_size = 8*128 = 1024`, `kv_size = 2*128 = 256`

- **KV head replication**: **PASS** @ `linear.py:296-299`
  - `if total_num_kv_heads >= tp_size: num_kv_heads = total_num_kv_heads // tp_size` else `num_kv_heads = 1`
  - 符合编码铁律: tp > num_kv_heads 时 num_kv_heads=1

---

## Blueprint Information Gaps

- **`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.qkv_column_parallel_forward[merged_comment]`**: **YELLOW** @ `inference_blueprint.json:919`
  - 蓝图注释: `# [B, T, 2*intermediate/tp] e.g. [1,1,6400]` — `6400` 基于旧 intermediate_size=12800 的示例值（12800*2/4=6400）
  - 实际 Qwen3-8B 物理 verified_config: intermediate_size=12288 → 6144
  - 蓝图数值契约（tp4_per_rank.gate_up_weight="[6144,4096]", test_interface_contracts "gate_up_total=6144 非 6400"）一致指向 6144
  - 代码已正确使用 6144，不影响判定。建议更新蓝图注释为 `e.g. [1,1,6144]` 以消除歧义。

---

## Final Verdict

**PASS** — Spec 审查通过，代码与蓝图契约一致，可移交 verification。

所有 4 类 TP Linear 的 weight shape、forward 逻辑、load_weight_shard double-shard guard、Q-K-V 顺序、gate-up 顺序均与 `inference_blueprint.json` 的三个契约节点精确对齐。无 ERROR 级别的偏差。
