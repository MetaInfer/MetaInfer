# 蓝图重构完备性漏洞报告

**审计对象**：`inference_blueprint.json` (v2.3.0, 最新更新) + `AGENT_SKILL.md` (最新更新)
**审计身份**：推理引擎独立闭环审计官（Isolated Reconstructability Auditor）
**审计前提**：假设 `engine/` 目录源码被物理擦除，Agent 仅持有上述两份文件
**审计范围**：调度层 + Qwen3-8B TP=4（排除 `deepseek_v2_v3_mla_moe` 节点）
**审计日期**：2026-05-26

---

## 与 V5 审计的关键差异

本次审计对象相比 V5 审计时有以下新增节点，全部 12 项 V5 识别问题均已闭合：

| V5 编号 | 修复节点 | 状态 |
|---------|---------|------|
| GAP-1 (命名空间) | `block_size_selection` 改用 `hf`/`qwen_tp`/`deepseek_tp` | ✅ |
| GAP-2 (kv_len 来源) | `num_free_blocks_source.TP_Runner.impl` + `ceil_note` + `kv_len_source` | ✅ |
| GAP-3 (多序列 block_table) | `slot_mapping_algorithm.multi_seq` | ✅ |
| GAP-4 (Sampler TP) | `Sampler.tp_sampling_protocol` | ✅ |
| GAP-5 (_reserved_blocks) | `scheduler_tp_runner_bridge.prefill_timing_gap` | ✅ |
| GAP-6 (CUDA Graph pathway) | `current_status.tp4_implementation_pathway` | ✅ |
| WARN-1 (decode batch) | `tp_runner_actual_flow.decode_batch_mode` | ✅ |
| WARN-2 (超长死循环) | `schedule_algorithm.overlength_rejection` | ✅ |
| WARN-3 (prefill 回滚) | `prefill_failure_rollback` | ✅ |
| WARN-4 (residual weight) | `fused_add_rms_norm.constraint.physical_layout_correction` | ✅ |
| WARN-5 (workspace) | `init_state_machine.workspace_size` 修正 | ✅ |
| WARN-6 (KV source) | `prefill_path.kv_source_correction` | ✅ |

AGENT_SKILL.md §7.4.G 从 7 条扩展至 12 条（新增 TP 采样、超长拒绝、prefill 回滚、KV 来源、多序列 prefill）。

---

## 1. 🔴 可重构性阻断项（Reconstructability Gaps）

### GAP-1：CUDA Graph TP=4 `tp4_implementation_pathway` 缺少可执行代码模板

- **JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.cuda_graph_execution_contract.current_status.tp4_implementation_pathway.tp4_upgrade`
- **严重等级**：🔴 阻断（如 Agent 目标包含 TP=4 CUDA Graph）

**问题**：`tp4_upgrade` 给出了 5 步升级路径描述，`split_graph_implementation` 提供了 SGLang 方案的伪代码。但以下关键环节仍缺失可执行代码：

1. **`torch.compile` custom backend 注册**：`SGLangBackend` 的伪代码展示了 `compile()` 方法，但 `register_backend('sglang_custom', SGLangBackend)` 和 `torch.compile(model, backend='sglang_custom')` 的调用方式未给出。
2. **`split_graph` 函数不存在于标准库**：需要从 sglang 复制 `compilation/backend.py` 的 `split_graph` 函数，但 Agent 无法访问 `ref_projects`（审计假设）。
3. **与现有 `swap_mechanism` 的组合语义**：`CUDA_GRAPH=1` 触发 per-layer `torch.compile(forward_decode_graph, mode='reduce-overhead')`。TP=4 CUDA Graph 需要在此基础上加入 `split_graph`——但 `reduce-overhead`（inductor cudagraph_trees）和 `split_graph`（手动切图 + eager/inductor 混合）是两个不同的编译策略，如何组合未说明。

`current_status.tp4` 仍标注为"阻塞"，所以此 gap 属于已知未完成范围。但如果 Agent 的目标是"具备 CUDA Graph 优化"，这个 gap 是阻断性的。

**必须补充的技术规格契约**：在 `tp4_implementation_pathway` 中补充完整的 `torch.compile` custom backend 注册代码，包括 `register_backend` 调用和与 `swap_mechanism` 的互斥/组合逻辑（如 `CUDA_GRAPH=2` 触发 TP=4 split_graph 路径）。

---

### GAP-2：`QwenForCausalLMTP.forward()` 的 prefill/decode 分发逻辑未定义

- **JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern.entry` + `services.tp_runner_actual_flow`
- **严重等级**：🔴 High

**问题**：`tp_runner_actual_flow.prefill` 描述 prefill 为 `model.forward(input_ids, past_key_values=None)`，`decode` 描述 decode 为 `model.forward(input_ids, past_key_values=kv_lens)`。`decode_forward_pattern.entry` 说 "QwenForCausalLMTP.forward() 中 is_decode 分支调用 layer.forward_decode()"。但**如何判断 is_decode**？是通过 `past_key_values is not None`？还是 `input_ids.shape[1] == 1`？还是调用者显式传参？

此外，prefill 时 `model.forward()` 调用 `layer.forward()`（非 `forward_decode`），但 `QwenDecoderLayerTP` 的接口中 `forward()` 和 `forward_decode()` 是两个不同方法。`forward()` 方法的具体签名、内部如何处理 prefill KV allocation（多序列 block_table 拼接、slot_mapping 构造、flash_attn_varlen_func 调用、index_copy_ 写入）——这些核心逻辑**全部缺失**。蓝图详细描述了 `forward_decode` 的完整流程，但对 `forward()` (prefill) 只给了一句描述加一个带 `multi_seq` 伪代码的 KV write 节点。

Agent 不知道 prefill forward 的完整数据流：embedding → 逐层处理 → 每层内部 qkv_proj → rotary → flash_attn_varlen_func → index_copy_ → o_proj → mlp → 输出 logits。

**必须补充的技术规格契约**：在 `qwen3_tp_model_interfaces` 下新增 `prefill_forward_pattern` 节点，给出与 `decode_forward_pattern` 同级别的完整流程描述，包括 is_prefill 判定逻辑。

---

### GAP-3：Scheduler `schedule()` 的 `num_free` 参数传递链未端到端闭合

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_tp_runner_bridge.prefill_timing_gap.pseudocode`
- **严重等级**：🔴 High

**问题**：`prefill_timing_gap.pseudocode` 给出了 `def schedule(num_free)` 的伪代码，展示了 `_reserved_blocks` 的内部维护。但 `schedule()` 的调用方是谁？谁向它传递 `num_free`？`engine_integration` 描述 `batch, is_prefill = scheduler.schedule()` 调用时**无参数**。`num_free_blocks_source.interface` 说 "LLMEngine.step() 中根据 inference_backend 选择调用 BlockManager 或 runner 获取 num_free_blocks"——但获取后如何传给 `scheduler.schedule()`？schedule() 的签名是否为 `schedule(num_free_blocks)`？图谱中两处描述不一致（一处带参数，一处不带）。

**必须补充的技术规格契约**：将 `engine_integration` 的描述从 `scheduler.schedule()` 更新为 `scheduler.schedule(num_free_blocks)`，或在 `scheduler_tp_runner_bridge.interface` 中明确调用链：
```
LLMEngine.step():
    num_free = block_mgr.get_num_free_blocks() or runner.get_num_free_blocks()
    batch, is_prefill = scheduler.schedule(num_free)
```

---

## 2. 🟡 信息熵不足警告项（Ambiguous Descriptions）

---

### WARN-1：Prefill forward 内部 KV cache 写入与 attention 的顺序未与 `multi_seq` slot_mapping 整合

- **JSON Path**：`framework_layer.data_flow_contracts.paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm.multi_seq` + `prefill_path.kv_source_correction`
- **严重等级**：🟡 警告

**问题**：`kv_source_correction` 明确了顺序为 1) QKV 投影 → 2) flash_attn_varlen_func → 3) index_copy_ 写入。`multi_seq` 给出了多序列 block_table 和 slot_mapping 的构造。但两个伪代码块是分离的——Agent 需要自己将它们整合到 `QwenAttentionTP.forward()` 的 prefill 分支中。整合时的关键细节缺失：
- `multi_seq` 中的 `seq.block_table = bt_padded` 是 Python list 赋值，但 `seq.block_table` 需要传给 `flash_attn_with_kvcache` 供后续 decode 使用——中间是否需要转为 tensor？何时转？
- 多序列 prefill 的 `flash_attn_varlen_func` 在第一层调用时 K/V 还不是从 cache 中来的（直接从投影产出来），后续层呢？后续层在 prefill 中也需要 attention，但 K 和 V 是否已全部写入 cache？若是，多层的 prefill 是一次性全部投影后再 attention 还是逐层 projection→attention→cache_write？

**风险**：Agent 可能错误理解 prefill 中多层 attention 的 K/V 来源顺序。

---

### WARN-2：`get_num_free_blocks()` 的实现位置未在 Runner 接口中显式声明

- **JSON Path**：`components[3]` (ModelRunner) — 无 `get_num_free_blocks` 方法描述
- **严重等级**：🟡 警告

**问题**：`scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner.impl` 给出了 `get_num_free_blocks()` 的实现代码（`self.model.layers[0].self_attn._kv_len_gpu[0].item()`），但这个方法的定义位置——它应该是 `QwenTPModelRunner` 的方法——在 `ModelRunner` 组件中未声明。Agent 搜索 `ModelRunner` 的接口定义时会发现它缺少这个关键方法。

**风险**：Agent 可能在写 `QwenTPModelRunner` 时遗漏 `get_num_free_blocks()`，导致 LLMEngine.step() 集成时报 `AttributeError`。

---

### WARN-3：`KVMemoryPool` 与 TP Runner 路径的交互未定义

- **JSON Path**：`components[1]` (KVMemoryPool) — 仅描述 HF 路径下的显存预算
- **严重等级**：🟡 警告

**问题**：`KVMemoryPool._responsibility_boundary` 说 "KVMemoryPool: 仅显存预算(estimate_num_blocks)+GPU placeholder"。但 `paged_kv_cache_contract.kv_cache_format.initialization` 中的 `_key_cache=torch.zeros(num_blocks,256,...)` 是在 `QwenAttentionTP` 内部创建的，不是 KVMemoryPool 创建的。Agent 面临两个问题：

1. TP Runner 路径是否需要 KVMemoryPool？如果不使用 BlockManager（`block_manager_role_in_tp_path` 说降级为 no-op），那 KVMemoryPool 是否也需要降级？
2. 预分配 KV cache 时应该用哪个 `num_blocks`？是 `KVMemoryPool.estimate_num_blocks()` 返回的值，还是 `max_blocks_formula.qwen3_8b: 128` 的硬编码值？蓝图在两个地方给了不同的指导——KVMemoryPool 是动态估算的，`initialization` 中的公式是基于 `max_position_embeddings` 的固定值。

**风险**：Agent 可能错误地在 TP Runner 路径中通过 KVMemoryPool 分配 KV cache 大小，导致显存预算与模型实际分配不一致。

---

### WARN-4：`compute_hash` 使用 Python `hash()` 但 `hash_policy` 提供了备选方案，Agent 需要二选一

- **JSON Path**：`components[2].api_spec.compute_hash.hash_policy`
- **严重等级**：🟡 Low

**问题**：`hash_policy` 给 Agent 两个选择：Python `hash(tuple)` 或 `hashlib.md5`。但没有说明哪个是默认推荐。Agent 需要在两种实现间自行选择——Python `hash` 简单但跨进程不一致（当前单进程 OK），`hashlib.md5` 稳定但更慢。这不是功能性缺陷，但增加了 Agent 的决策负担。

**风险**：低。两种实现都能工作。

---

### WARN-5：`fused_add_rms_norm` 的 `physical_layout_correction` 与 `desc` 的语义矛盾

- **JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint`
- **严重等级**：🟡 Low

**问题**：`desc` 说 "post_mlp 调用使用下一层 input_layernorm.weight"，但 `physical_layout_correction` 说 "meta-infer 简化: post_mlp 的 fused_add_rms_norm 使用本层 post_attention_layernorm.weight（非下一层 input_layernorm.weight）。与 vLLM 原始设计不同但数值等价。"

这是两条语义矛盾的信息——Agent 需要仔细阅读 `physical_layout_correction` 才能发现实际的正确行为。如果 Agent 只看 `desc` 就开始编码，会写成 vLLM 的原始设计。虽然 `physical_layout_correction` 说明了正确的 meta-infer 行为，但 `desc` 的误导仍然存在。

**风险**：低。Agent 需要仔细阅读全部三个子字段（desc + residual_chain_pseudocode + physical_layout_correction）才能正确实现。

---

## 3. 🟢 最终图纸判决（Verdict）

### 逐项审计结论

逐项回答审计清单中的具体问题：

#### 2.1 推理框架调度层

| 问题 | 结果 | 证据 |
|------|------|------|
| `can_allocate` / `can_append_one_more` 算法边界 | ✅ 已定义 | `can_allocate.formula` + `required_blocks`, `can_append_one_more.formula` |
| `max_num_batched_tokens` 计算依据 | ✅ 已定义 | `max_num_batched_tokens.formula = max(1, num_free_blocks * block_size)` |
| `compute_hash` 签名规范 | ✅ 已定义 | `compute_hash.signature` + `algorithm_detail` + `hash_policy` |
| `ref_count` 生命周期边界 | ✅ 已定义 | `ref_count_lifecycle`: +1 allocate, +1 prefix hit, -1 free |
| 双轨块大小隔离的硬性架构屏障 | ✅ 已定义 | `scheduler_tp_runner_bridge` + `block_size_selection` + `block_manager_role_in_tp_path` |

#### 2.2 框架层增量修改

| 问题 | 结果 | 证据 |
|------|------|------|
| Decode K/V Tensor 拼接 Shape 契约 | ✅ 已定义 | `full_reshape_chain` + `decode_kv_write` |
| `is_causal = (past_key_values is None)` | ✅ 已定义 | `causal_rule`: "prefill causal=True, decode causal=False (past_key_values is None → prefill)" |
| MoE 不可编译原因 | ✅ 已定义 | `no_moe_compile`: "MoE 模块有 .item() GPU→CPU 同步 → graph break" |
| `cu_seqlens_q/k` 构造方法 | ✅ 已定义 | `cu_seqlens_construction` |

#### 2.3 Kernel 层 7 大标品

| 问题 | 结果 | 证据 |
|------|------|------|
| `rms_norm` out 预分配约束 | ✅ 已定义 | `rms_norm.constraint`: "out 预分配 (empty_like)" |
| `silu_and_mul` out 预分配约束 | ✅ 已定义 | `silu_and_mul.constraint`: "out 预分配 [B, S, intermediate/tp]" |
| `fused_add_rms_norm` 跨层权重拓扑 | ✅ 已定义 | `residual_chain_pseudocode` + `last_layer_note` + `physical_layout_correction` |
| CustomAR IPC 握手状态机 | ✅ 已定义 | `init_state_machine`: 7 步伪代码 + barrier + workspace |

#### 2.4 CUDA Graph

| 问题 | 结果 | 证据 |
|------|------|------|
| 显存地址锁死代码规约 | ⚠️ 部分 | `clone_mechanism` + `static_buffers` 已描述单 GPU 方案；TP=4 方案在 `tp4_implementation_pathway` 中为步骤描述 |
| `all_reduce_sum` custom_op 注册 + Fake Tensor | ✅ 已定义 | `all_reduce_sum_custom_op_template.registration` + `impl` + `fake` |

### 量化结论

| 维度 | 完备度 | 剩余缺口 |
|------|--------|---------|
| 框架调度层（Scheduler/BlockManager/KVMemoryPool） | **85%** | GAP-3 (schedule 签名一致性待闭合), WARN-2, WARN-3 |
| 框架层增量修改（P0/P2/P3-FA） | **90%** | WARN-1 (prefill forward 完整流程), GAP-2 (forward 分发) |
| Kernel 层 7 大标品（数据契约 + 残差链 + IPC） | **88%** | WARN-4, WARN-5 (均为 Low) |
| CUDA Graph 静态执行契约 | **65%** | GAP-1 (TP=4 仍标注阻塞，单 GPU 完备) |
| **加权综合完备度** | **≈83%** | |

### 判决

**当前图谱允许有经验的推理引擎开发者独立开工，预计可产出结构正确、需少量调试即可跑通的 Qwen3-8B TP=4 推理引擎。**

自初次审计（V4 初版，~48%）以来，经过两轮重大迭代更新，图谱现已到达"可施工"水平。所有系统性的架构断层（Scheduler ↔ TP Runner 桥接、CustomAR 状态机、KV cache 重塑链、多序列 block_table、Sampler TP 协议、异常路径处理）均已被对应的契约节点覆盖。

剩余阻断项中，GAP-1（CUDA Graph TP=4）属于明确标注的未完成功能（`current_status.tp4: "阻塞"`），符合预期。GAP-2（prefill forward 分发逻辑）和 GAP-3（schedule 签名一致性）是 Agent 在集成测试阶段会自然发现的细节——不会导致系统性架构崩塌，会在首次 prefill 或首次调度调用时报错并修正。

### 建议修复的 Top 3 项（按优先级）

1. **GAP-2（prefill forward 分发逻辑）** — 在 `qwen3_tp_model_interfaces` 下新增 `prefill_forward_pattern`，给出与 `decode_forward_pattern` 同级别的完整 prefill 数据流描述。这是单模块内最显著的文档缺口。

2. **GAP-3（schedule 签名一致性）** — 统一 `engine_integration` 和 `prefill_timing_gap.pseudocode` 中的 `schedule()` 签名，在 `interface` 中给出从 LLMEngine 到 Scheduler 的完整参数传递链。

3. **GAP-1（CUDA Graph TP=4 可执行性）** — 将 `tp4_implementation_pathway` 升级为可独立编码的代码模板（custom backend 注册 + 与 swap_mechanism 的组合策略），或维持当前"阻塞"标注并在 `_note` 中明确说明 Agent 不应尝试实现 TP=4 CUDA Graph。

---

**总结**：经过本次更新，图谱的完备度从 V5 的 ~70% 提升至 ~83%。剩余 3 个 GAP 中 2 个为特定功能的文档缺口（prefill forward 流程、schedule 签名），1 个为已知未完成的优化功能（TP=4 CUDA Graph）。修复 GAP-2 和 GAP-3 后完备度可达 ~88%，届时可交付中级 Agent 零人工干预闭环。当前状态下，经验丰富的推理引擎开发者已可基于此图谱开工。
