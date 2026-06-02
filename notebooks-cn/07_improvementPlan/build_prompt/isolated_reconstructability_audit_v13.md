# 蓝图重构盲测漏洞报告 v1.3

**审计对象**: `inference_blueprint.json` v2.3.0（排除 deepseek 章节）
**审计范围**: Qwen3-8B nocompile B=1 TP=4 引擎全链路（框架调度层 + 模型层 + kernel 层）
**审计官**: 独立第三方系统级审计官（Zero-Shot Reconstructability Auditor）
**审计资产**: `inference_blueprint.json` + `AGENT_SKILL.md` + RAG 可查 ref_code + ref_docs

---

## 审计前置：ref_docs 抽样交叉验证记录（维度四）

根据强制操作要求，打开以下 ref_doc 并验证蓝图引用：

| ref_doc | 验证的引用点 | 文档中是否存在？ | 结论 |
|---------|-------------|:------------:|------|
| `kernel_replacement_plan.md §九` | `rms_norm` 调用契约 (out 预分配, input contiguous) | ✅ 完整 | §9.1-9.6 包含 6 个 kernel 的完整调用契约 |
| `kernel_replacement_plan.md §九` | `fused_add_rms_norm` 跨层 weight 依赖 | ✅ 但存疑 | §9.2 说 post-mlp 用"下一层 input_layernorm.weight"，但 JSON `_kernel_replacement_plan_conflict` 指出 meta-infer 实际用本层 post_attention_layernorm.weight。两文档矛盾，JSON 已注明优先以 JSON 为准。 |
| `kernel_replacement_plan.md §九` | `cos_sin_cache` 格式 [max_pos, head_size] | ✅ 完整 | §9.6 + Snippet E 提供完整构造代码 |
| `kernel_replacement_plan.md custom_op` | `flash_attn_with_kvcache_op` 注册模板 | ✅ 存在 | 但 `_fa_kvcache` 符号在模板中未显式 import，需推断 |
| `qwen3_effective_changes.md` | 10 个有效改动点的追溯 | ✅ 完整 | 可作为改动清单交叉参考 |

**维度四结论**: ref_docs 中核心知识点存在，但存在 1 处文档间矛盾（fused_add_rms_norm weight），蓝图已注明处理方式。`_fa_kvcache` 符号缺失是唯一的真正信息断裂，但可通过上下文推断。

---

## 🔴 Fatal Gaps（致命脑补漏洞项）

### FG-1: `flash_attn_with_kvcache` 的底层导入符号在蓝图及 ref_docs 中均未显式定义

- **JSON Path**: `framework_layer.data_flow_contracts.flash_attention_integration_contract.decode_path.custom_op_registration.complete_template`
- **ref_docs 交叉验证**: `kernel_replacement_plan.md` custom_op 模板（line 1201-1203）同样使用了未定义的 `_fa_kvcache` 符号
- **致命理由**: 模板中 `return _fa_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, block_table=block_table, ...)` 的 `_fa_kvcache` 在所有文档中均无 import 声明或定义。重构者必须推断它是 `from flash_attn.flash_attn_interface import flash_attn_with_kvcache` 的别名。这个 import 是 decode 路径的单点故障——错误推断会导致 `NameError` 直接崩溃。
- **修复建议**: 在 custom_op 模板前增加一条注释 `# _fa_kvcache = flash_attn.flash_attn_interface.flash_attn_with_kvcache`

### FG-2: `all_gather_last_dim` 仅有功能描述，无实现

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime.collectives`（line 529: "输入 [..., local_dim] -> 输出 [..., local_dim * tp_size]"）
- **涉及组件**: `ParallelLMHead`（vocab gather）和 `ColumnParallelLinear(gather_output=True)` 均依赖此函数
- **致命理由**: 全图谱仅一条功能描述。实际实现需要 `dist.all_gather_into_tensor` 调用，涉及 output tensor 预分配（shape 为 `[*B, local_dim * tp_size]`）、dtype 一致性和 NCCL backend 约束。`kernel_replacement_plan.md` 关注 all_reduce (CustomAR) 而非 all_gather，此 gap 在所有 ref_docs 中也未被填补。
- **修复建议**: 在 `tp_distributed_runtime.collectives.all_gather_last_dim` 增加完整伪代码

---

## 🟡 Override Warnings（重载失效警告项）

### OW-1: Scheduler `_nano_vllm_override` 行号与 ref_code 实际行号不完全对齐

- **JSON Path**: `framework_layer.components[0]._nano_vllm_override`
- **问题**: "line 52 can_append(seq)" 实际在 nano-vllm scheduler.py line 53。"line 54 self.running.pop()" 与实际一致但行号上下文偏移。重构者按行号删除时可能误触周边代码。
- **严重程度**: 低——override 描述的是语义（"preempt() 逻辑必须删除"），行号偏差不影响理解。

### OW-2: `fused_add_rms_norm` 跨层 weight 在蓝图与 kernel_replacement_plan.md 之间矛盾

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint._kernel_replacement_plan_conflict`（line 958）
- **问题**: `kernel_replacement_plan.md §9.2` 描述 vLLM 原始方案——post-mlp 使用**下一层**的 `input_layernorm.weight`。JSON 明确声明 meta-infer 简化方案——使用**本层** `post_attention_layernorm.weight`。JSON 已注明优先以 JSON 为准，但重构者如果先读到 `kernel_replacement_plan.md §9.2` 可能按 vLLM 原始方案实现，产生跨层 weight 错误。
- **严重程度**: 中——JSON 已标注冲突和优先级，但依赖重构者注意到此警告

### OW-3: BlockManager `compute_hash` 算法与 nano-vllm ref_code 不一致，但缺少 `_nano_vllm_override` 格式标注

- **JSON Path**: `framework_layer.components[2].api_spec.compute_hash`
- **问题**: 蓝图规定 `hash(tuple(token_ids))` 使用 Python builtin hash。nano-vllm `block_manager.py` 实际使用 `xxhash.xxh64()` + `numpy.array(token_ids).tobytes()`。TP 路径不使用 compute_hash，所以这不是功能问题——但 override 信息仅存在于 `api_spec` 的 prose 中，未显式标注 `_nano_vllm_override` 标签，容易在代码审查时遗漏。

### OW-4: Sequence.block_table 双态表示缺乏 `_nano_vllm_override`

- **JSON Path**: `framework_layer.data_flow_contracts.request_level.sequence_fields.block_table`
- **问题**: 蓝图规定 block_table 需同时支持 `list[int]`（HF 路径）和 `torch.Tensor [1, max_blocks] int32`（TP 路径）。nano-vllm `sequence.py` 仅支持 `list[int]`。蓝图未提供 Tensor 表示的切换逻辑、block_table_tensor() 方法签名或 padding 约定（-1 还是 0）。重构者需自行设计双态转换层。

### OW-5: Scheduler.block_manager 引用在 TP 路径下未完全切断

- **JSON Path**: `framework_layer.components[0]._nano_vllm_override` + `scheduler_tp_runner_bridge`
- **问题**: nano-vllm `Scheduler.__init__` 硬编码 `self.block_manager = BlockManager(...)`。蓝图 override 说 TP 路径不调用 BlockManager 方法，但 Scheduler.schedule() 内部有 3 处 `self.block_manager.xxx()` 调用。重构者在改写 schedule() 时必须**重写整个方法体**而非仅外部包装——这一点蓝图的 override 说得不够直白。
- **缓解因素**: `scheduler_tp_runner_bridge.block_manager_role_in_tp_path`（line 1262-1269）提供了降级伪代码。

---

## 🔵 附：FG-2 (原) 已撤回

原报告 FG-2（`past_key_values` 传递协议不完整）经 ref_docs + 蓝图重审后确认为误判。`is_decode = past_key_values is not None` 分派逻辑干净一致，prefill 路径传 `None`、decode 路径传 `list[int]` 的协议严格分离，不存在边界条件漏洞。**此条从 Fatal Gaps 中移除**。

---

## 🟢 Reconstructability Score（重构可行性判决）

### 量化分数：**82 / 100**

当前图谱 + ref_docs 对于"独立重构 Qwen3 TP=4 引擎"的信息完备度。

### 阻碍达到 100% 的 Top 3 阻断因素：

1. **`_fa_kvcache` 符号未定义**（权重：35%）
   - flash_attn custom_op 的底层导入路径在所有文档中均为隐式引用。重构者必须推断正确的 flash_attn 包路径和函数别名。错误推断 → decode attention 崩溃。修复成本极低（一行 import 注释）。

2. **`all_gather_last_dim` 无实现，仅功能描述**（权重：30%）
   - 涉及 lm_head 输出 gather 和 ColumnParallelLinear gather_output 路径。重构者需自行编写 `dist.all_gather_into_tensor` 调用，涉及 output tensor 预分配 shape 计算。实现难度低但接口契约缺失。

3. **fused_add_rms_norm 的蓝图 vs kernel_replacement_plan.md 矛盾**（权重：20%）
   - 两文档对 post-mlp weight 来源描述不同（本层 vs 下一层）。JSON 已标注优先以 JSON 为准，但依赖重构者注意到此警告。若疏忽——使用下一层 weight → 输出数值异常。

### 得分细节：

| 能力维度 | 得分 | 说明 |
|---------|:----:|------|
| 框架调度层 (Scheduler/BlockManager/序列) | 75% | 双轨 bridge 清晰，但 schedule() 改写存在引用切断风险 |
| Qwen3 模型结构 (类层级 + 维度 + 算子签名) | 95% | 类定义、dims、TP=4 per-rank 分解均完整 |
| TP 线性层 (QKV/MLP/Embedding) | 90% | QKV merge 伪代码完整，all_gather_last_dim 缺失 |
| Attention (prefill + decode + paged KV) | 85% | 主路径完整，custom_op import 断层 |
| 权重加载 (HF key mapping + lazy loader) | 85% | HF→meta-infer 映射完整，QKV 状态机隐含但可实现 |
| Kernel 层 7 大标品 (RMSNorm/RoPE/Silu/CustomAR) | 90% | kernel_replacement_plan.md §九 有完整调用契约 |
| NCCL/CustomAR 初始化 | 80% | init 伪代码详细，调用位置在蓝图与 AGENT_SKILL.md 间略有模糊 |
| CUDA Graph (torch.compile reduce-overhead) | 70% | 单 GPU 路径完整，TP=4 切图方案有 reference 但缺集成代码 |
| Scheduler↔Runner 桥接 | 80% | block_size 注入、num_free_blocks 来源均指定 |

### 结论

**允许交付给全新 Agent 闭环开工，但需在开工前补齐 FG-1 和 FG-2。** 蓝图在模型层静态结构和 kernel 调用契约上达到了高完备性。剩余断层集中在两个接口级细节（一个 import 声明、一个 all_gather 实现），每个的修复成本均在一行至十行代码以内。修复后预期得分 ≥ 92%。

---

*审计日期: 2026-05-27 | 审计版本: v1.3 | 审计官模型: deepseek-v4-pro*
