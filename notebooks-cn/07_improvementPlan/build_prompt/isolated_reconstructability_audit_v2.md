# 蓝图重构完备性漏洞报告 v2（补漏后复审）

> **审计对象**：`inference_blueprint.json` (v2.3.0, 已补漏) + `AGENT_SKILL.md`
> **审计目标**：独立重构 Qwen3-8B TP=4 调度层 + 推理引擎
> **审计日期**：2026-05-26
> **排除范围**：所有 `deepseek_v2_v3_mla_moe` 相关节点
> **对比基线**：v1 报告 (48% → 本复审重新评分)

---

## 0. 补漏变更摘要

上一轮审计指出的 10 个 🔴 GAP 和 10 个 🟡 AMB 中，**绝大多数已被定点修复**。关键补漏包括：

| 原 GAP/AMB | 补漏内容 | 位置 |
|-----------|---------|------|
| GAP-1 | `BlockManager.get_num_free_blocks()` + `block_size_selection` | `components[2].api_spec` / `scheduler_to_runner.max_num_batched_tokens` |
| GAP-2 | `cu_seqlens_construction` 伪代码 | `flash_attention_integration_contract.prefill_path` |
| GAP-3 | `unified_signature` | `qwen3_tp_model_interfaces.decode_forward_pattern` |
| GAP-4 | `slot_mapping_algorithm` 公式+伪代码 | `paged_kv_cache_contract.prefill_kv_write` |
| GAP-5 | `_responsibility_boundary` | `components[1]` (KVMemoryPool) |
| GAP-6 | `all_reduce_sum_custom_op_template` | `cuda_graph_execution_contract` |
| GAP-7 | `max_blocks_formula` + `initialization` | `paged_kv_cache_contract.kv_cache_format` |
| GAP-8 | `kv_len_timing` | `qwen3_tp_model_interfaces.decode_forward_pattern` |
| GAP-9 | 动态读取规则 + `qwen3_8b_model_dims` + `tp4_per_rank` | `global_primitives_constraints` / `qwen_series_dense` |
| GAP-10 | `lazy_loading_pseudocode` | `qwen3_kernel_contracts.rotary_embedding` |
| AMB-1 | `algorithm_detail` (hash 实现) | `BlockManager.api_spec.compute_hash` |
| AMB-2 | `chunked_prefill_rule` | `scheduler_to_runner.schedule_algorithm` |
| AMB-3 | `residual_chain_pseudocode` | `qwen3_kernel_contracts.fused_add_rms_norm` |
| AMB-4 | `swap_mechanism.implementation` | `torch_compile_contract.forward_decode_design` |
| AMB-5 | `ipc_exchange_pseudocode` | `qwen3_kernel_contracts.custom_ar_all_reduce` |
| AMB-6 | `complete_template` | `flash_attention_integration_contract.decode_path.custom_op_registration` |
| AMB-7 | `prefill_kv_len_semantics` | `qwen3_tp_model_interfaces.attention` |
| AMB-8 | `runtime_enforcement` (assert 守卫) | `global_primitives_constraints.paging_dual_track_contract` |
| AMB-9 | `engine_integration` (调度-执行 pipeline) | `ModelRunner.tp_runner_actual_flow` |
| AMB-10 | `initialization` (KV cache 创建) | `paged_kv_cache_contract.kv_cache_format` |

---

## 1. 🔴 Reconstructability Gaps (重构死锁漏洞项)

原 10 个 GAP 全部修复。复审发现 2 个新的中等严重度缺口。

### GAP-N1: `softmax_scale` 值未在任何节点中显式定义

| 属性 | 值 |
|------|-----|
| **JSON Path** | `flash_attention_integration_contract.decode_path.kernel` + `paged_kv_cache_contract.decode_attention` |
| **严重等级** | 🟠 Medium-High |

**问题描述**：

两处 decode attention 调用签名都包含 `softmax_scale` 或 `scale` 参数，但整个 blueprint 中没有任何地方给出其计算公式。Agent 可能猜测为 `1.0 / sqrt(head_dim)`（标准值），也可能误用 `1.0 / head_dim` 或直接传 `1.0`，导致 attention 权重异常——softmax 饱和或过于平坦。

对于 Qwen3-8B，`head_dim=128`，`scale = 1.0 / sqrt(128) ≈ 0.08839`。这个值虽然可以从 `head_dim` 推导，但 blueprint 应在 decode_path 中显式声明公式以避免 Agent 猜测。

**必须补充的规格**：
```json
"softmax_scale": "1.0 / sqrt(head_dim)  # 从 config.json 动态读取 head_dim 后计算"
```

---

### GAP-N2: Prefill batch 张量组装逻辑缺失

| 属性 | 值 |
|------|-----|
| **JSON Path** | `scheduler_to_runner` → `runner_prefill_tensors` → `tp_runner_actual_flow.prefill` |
| **严重等级** | 🟠 Medium-High |

**问题描述**：

Scheduler 输出 `list[Sequence]` 的 batch，但 TP Runner 的 `model.forward(input_ids, ...)` 需要一个具体的 `input_ids` 张量。对于 prefill batch 中包含多个不同长度序列的场景：

- `input_ids` 张量的 shape 是什么？`[B, L_max]`（padding）还是 `[1, total_tokens]`（concatenated ragged）？
- 如果使用 padding，是 left-padding 还是 right-padding？
- `positions` 张量如何为每个序列构造？
- 如何从 batch 中提取每个序列的 `block_table` 并合并？

图谱在 `cu_seqlens_construction` 中引用了 `seq.seq_len()`，暗示使用 ragged concatenation（`[1, total_tokens]`），但没有明确说明 input_ids 张量的组装方式。对于 TP Runner prefill，embedding 层需要完整的 input_ids，lm_head 需要按序列分离 logits——这个边界在 `tp_runner_actual_flow.prefill`（一行描述）和实际的张量准备代码之间存在断层。

**必须补充的规格**：
```python
# Prefill batch tensor assembly pseudocode
input_ids = torch.cat([seq.input_ids_tensor() for seq in batch], dim=1)  # [1, total_tokens]
positions = torch.cat([torch.arange(len(seq), device=device) for seq in batch])
# block_table 合并（每序列独立）
block_tables = [seq.block_table_tensor() for seq in batch]
```

---

## 2. 🟡 Ambiguous Descriptions (信息熵不足警告项)

### AMB-N1: `q_size` / `kv_size` 计算公式未显式给出

| 属性 | 值 |
|------|-----|
| **JSON Path** | `qwen3_kernel_contracts.qkv_merged_projection.constraint` |
| **严重等级** | 🟡 Low-Medium |

QKV 合并投影的 weight loading 三段复制（`[0:q_size], [q_size:q_size+kv_size], [q_size+kv_size:]`）已经记录，但 `q_size` 和 `kv_size` 的推导公式未显式给出。Agent 可从 `qwen3_8b_model_dims.tp4_per_rank` 推导（`qkv_weight: "[1536,4096]"`），但公式本身未声明：

```python
q_size = num_heads * head_dim // tp_size
kv_size = num_kv_heads * head_dim // tp_size  # 考虑 GQA
```

由于 `qwen3_8b_model_dims` 已经包含 tp4_per_rank 的具体数字，这个缺口影响较小。

---

### AMB-N2: `hash_to_block_id` 映射表数据结构未指定

| 属性 | 值 |
|------|-----|
| **JSON Path** | `BlockManager.api_spec.compute_hash.algorithm_detail` |
| **严重等级** | 🟡 Low |

`algorithm_detail` 描述了 hash 算法，并提及 "hash→block_id 映射表实现 prefix caching"，但没有指定映射表的数据结构——是 Python `dict[int, int]`（单 block hash→单 block_id）还是 `dict[int, list[int]]`（一个 hash 可能对应多个 block chain）？prefix caching 的查找/插入算法也未展开。

由于 BlockManager 的 ref_code 指向 `ref_projects/nano-vllm/nanovllm/engine/block_manager.py`，Agent 可参照实现，但独立重构时这是一处需要自行设计的逻辑。

---

### AMB-N3: split_graph 切图方案仅描述名称，缺少伪代码

| 属性 | 值 |
|------|-----|
| **JSON Path** | `cuda_graph_execution_contract.current_status.planned_fix` |
| **严重等级** | 🟡 Low-Medium |

TP=4 CUDA Graph 的修复方案描述为 "sglang 切图方案: torch.fx.split_module 在 all_reduce_sum 处拆分 FX 图"，但未提供 split_module 的调用伪代码。虽然 `all_reduce_sum_custom_op_template` 已经给出了 custom op 注册骨架，但如何在 compile 流程中插入 split 点（是在 `_setup_cuda_graph_piecewise` 中？是在 compile 之后手动修改 FX graph？）仍不明确。

对于当前审计范围（"独立重构"），这是一个待实施的 Stage D 任务，Agent 在到达此阶段时会需要更多细节。

---

### AMB-N4: `engine_integration` pipeline 缺少 `runner.run()` 的完整接口签名

| 属性 | 值 |
|------|-----|
| **JSON Path** | `ModelRunner.tp_runner_actual_flow.engine_integration` |
| **严重等级** | 🟡 Low |

Pipeline 描述为 `scheduler.schedule()→runner.run(batch,is_prefill,...)→scheduler.postprocess(batch,is_prefill,tokens)`，但 `runner.run()` 的具体参数列表和返回值未定义。`...` 中的内容是什么？返回的是 `tokens` 还是 `(tokens, kv_lens)`？Agent 需要从 LLMEngine 的 step() 主循环伪代码中推断这些细节。

---

### AMB-N5: 第一层 residual 初始值的特殊处理未充分强调

| 属性 | 值 |
|------|-----|
| **JSON Path** | `qwen3_kernel_contracts.fused_add_rms_norm.constraint.residual_chain_pseudocode` |
| **严重等级** | 🟡 Low |

残差链伪代码处理了 `res is None` 首次调用的分支（`res=hs.clone(); rms_norm(...)`），但未强调这个分支**仅在每层第一次 fused_add_rms_norm（input_layernorm）时可能触发**。在 post_attention_layernorm 调用时，`res` 已经不会是 None。Agent 可能在一个 layer 内对两个 norm 调用都做 None 检查，虽然不会出错但冗余。

---

## 3. 🟢 Verdict (最终图纸判决)

### 信息完备度量化评估

| 系统模块 | v1 完备度 | v2 完备度 | 变化 |
|---------|----------|----------|------|
| 推理框架调度层 (Scheduler / BlockManager / KVMemoryPool) | 55% | **90%** | +35% |
| 框架层增量修改 (P0 KV / P2 Compile / P3-FA) | 60% | **93%** | +33% |
| Kernel 层 7 大标品替换 | 70% | **95%** | +25% |
| CUDA Graph 静态执行契约 | 40% | **85%** | +45% |
| 模型维度具体参数 | 0% | **95%** | +95% |

### 总体完备度：**90%**（v1: 48%）

### 最终结论：**允许交付给全新 Agent 闭环开工，附条件**

当前图谱已具备使 Agent 从零构建 Qwen3-8B TP=4 推理引擎的核心指导力。所有 10 个原 🔴 阻断级 GAP 均已解决，Agent 不会因缺少关键参数或算法而死锁。

### 条件与建议

1. **条件 1（开工前必须）**：在 `flash_attention_integration_contract.decode_path` 中补充 `softmax_scale = 1.0 / sqrt(head_dim)` 的显式声明（GAP-N1）。

2. **条件 2（开工前建议）**：在 `scheduler_to_runner` 或新增 `batch_assembly_contract` 节点中补充 prefill batch 张量组装的伪代码，明确 input_ids 是 ragged concatenation 而非 padding（GAP-N2）。

3. **条件 3（实施中期）**：当 Agent 推进到 Stage D (TP=4 CUDA Graph) 时，需补充 `torch.fx.split_module` 切图伪代码（AMB-N3）——此为已知阻塞项，当前蓝图已标注但未提供代码骨架。

4. **辅助依赖**：Agent 仍需阅读 `ref_projects/nano-vllm/` 和 `notebooks-cn/` 中的参考文档来补充细节（BlockManager 的 hash_to_block_id 映射实现等），但蓝图本身已覆盖所有不可替代的核心约束和避免崩溃的关键知识（failure_mode_library 15 条 FM 条目覆盖了所有已知踩坑点）。

### 与原 v1 报告的对比

| 维度 | v1 | v2 |
|------|-----|-----|
| 🔴 Critical Gaps | 10 | 0 |
| 🟡 Ambiguous Warnings | 10 | 0 (原) + 5 (新) |
| 总体完备度 | 48% | **90%** |
| 可开工判定 | 不允许 | **允许（附 4 条件）** |
| Top 1 阻断 | 缺少模型维度 | softmax_scale 未声明 (minor) |

---

*复审完成。图谱从"参考手册"级别（48%）升级至"施工图纸"级别（90%）。剩余缺口不影响框架层、Kernel 层和模型适配层的独立重构，仅在 CUDA Graph TP=4（Stage D，已知未完成）和少量边缘细节上留有 TODO。*
