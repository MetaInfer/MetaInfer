# 知识图谱合规审计报告（Trinity Verification Audit Report）

**审计身份**：AutoLLM 系统独立三方审计官（Third-Party Auditor）
**审计日期**：2026-05-26
**审计目标**：核验 `inference_blueprint.json` v2.3.0 与自研工程代码的物理保真度
**审计范围**：`engine/` + `llm_engine.py` + `engine/tp_layers/` + `engine/kernels/` + `engine/models/`

---

## 1. 审计资源清单

| 资源类型 | 路径 | 状态 |
|---------|------|------|
| 核心图谱 | `inference_blueprint.json` v2.3.0 | 已读取 |
| 变更文档 | `notebooks-cn/07_improvementPlan/cuda_graph_plan.md` | 已读取 |
| 变更文档 | `notebooks-cn/07_improvementPlan/qwen3_effective_changes.md` | 已读取 |
| 变更文档 | `notebooks-cn/07_improvementPlan/kernel_replacement_plan.md` | 已读取 |
| 变更文档 | `notebooks-cn/07_improvementPlan/improvement_plan.md` | 已读取 |
| 自研源码 | `engine/` 全部 `.py` 文件 | 已审计 |
| 参考源码 | `ref_projects/nano-vllm/nanovllm/engine/` | 已验证存在 |
| 参考文档 | `notebooks/01_framework_design/` | 已验证存在 |
| 参考文档 | `notebooks-cn/` | 已验证存在 |

---

## 2. 审计方法论

针对 `inference_blueprint.json` 内每个组件节点或全局约束，依次提取 `ref_docs`、`ref_code`、`impl_code` 字段，执行三方交叉印证（Trinity Verification）：

- **2.1 标品与实现对齐核验**：对比 `ref_code` 与 `impl_code`，核对 Tensor Shape、Dtype、连续性契约
- **2.2 角色与实现对齐核验**：对照 `ref_docs` 定义的职责，检索 `impl_code` 真实控制流
- **2.3 资源缺失补偿机制**：对缺失 `ref_code`/`ref_docs` 的高性能节点，启用变更文档补偿
- **2.4 故障模式探针核验**：扫描 `failure_mode_library`，确认代码中注入了对应防御性探针

---

## 3. 审计结论

### 🟢 Passed（完全合规项，12 项）

| # | 知识节点 | JSON Path | 证据 |
|---|---------|-----------|------|
| 1 | **Scheduler** prefill 优先 + 不抢占 | `framework_layer.components[0]` | `engine/scheduler.py:24-55`: `_schedule_prefill` 优先从 waiting 取，资源不足 `break` 不抢占 |
| 2 | **BlockManager** 链式前缀哈希 + ref_count 共享 | `framework_layer.components[2]` | `engine/block_manager.py:39-48`: `compute_hash(prefix=...)` 链式；`engine/block_manager.py:88-89`: `ref_count += 1` 共享 |
| 3 | **Sequence** 状态机字段完整 | `data_flow_contracts.request_level` | `engine/structs.py`: `input_ids`, `output_ids`, `block_table`, `num_cached_tokens`, `status` 全部匹配，状态转移 `WAITING→RUNNING_PREFILL→RUNNING_DECODE→FINISHED` |
| 4 | **Sampler** greedy + top_p + temperature 组合 | `framework_layer.components[4]` | `engine/sampler.py`: `greedy_sample` argmax + `top_p_sample` nucleus + `sample_next_tokens` 组合路径，logits.float() 后计算 |
| 5 | **QKVColumnParallelLinear** 合并投影 QKV | `tp_layer_interface_contracts.qwen3_tp_model_interfaces.qkv_projection` | `engine/tp_layers/linear.py:129-133`: `F.linear → split([q_size, kv_size, kv_size])` |
| 6 | **MergedColumnParallelLinear** gate+up 合并 | `model_layer.qwen_series_dense.mlp_routing.gate_up_proj` | `engine/tp_layers/linear.py:47-61`: `weight=[2*local_out, in]` |
| 7 | **ColumnParallelLinear** 防双切片 guard | `global_primitives_constraints.tp_linear_load_no_double_shard` | `engine/tp_layers/linear.py:36`: `if full_weight.shape == self.weight.shape: copy_()` |
| 8 | **RowParallelLinear** 防双切片 guard | 同上 | `engine/tp_layers/linear.py:95`: 同 pattern |
| 9 | **VocabParallelEmbedding** 防双切片 guard | `global_primitives_constraints.tp_linear_load_no_double_shard` | `engine/tp_layers/embedding.py:38`: `if int(full_weight.shape[0]) == self.local_vocab_size` |
| 10 | **FM-005** (CustomAR gloo group) 已注入探针 | `failure_mode_library.entries[4]` | `engine/tp_layers/distributed.py:57`: `gloo_group = dist.new_group(backend="gloo")` |
| 11 | **FM-011** (flash_attn custom_op) 已注入探针 | `failure_mode_library.entries[10]` | `engine/kernels/custom_ops.py:15-34`: `@torch.library.custom_op` + `register_fake` |
| 12 | **FM-013** (all_reduce_sum custom_op 屏蔽 Dynamo) 已注入探针 | `failure_mode_library.entries[12]` | `engine/tp_layers/distributed.py:75-93`: `@torch.library.custom_op("meta_infer::all_reduce_sum")` + `register_fake` |

---

### 🔴 Critical Violations（严重缺陷项，3 项）

#### CRITICAL-01：Scheduler/BlockManager 与模型 KV Cache 使用两套不连通的块大小

- **JSON Path**：`data_flow_contracts.paged_kv_cache_contract.kv_cache_format.block_table`
- **蓝图声称**：`block_size` 必须 >= 256（flash_attn_with_kvcache 硬性要求），block_table 由 BlockManager 分配
- **代码冲突**：
  - `llm_engine.py:184` — `block_size: int = 16`（默认值）
  - `engine/scheduler.py:45` — `seq.block_size = self.memory_pool.block_size`（继承 16）
  - `engine/block_manager.py:33` — `self.block_size = block_size`（继承 16）
  - **但** `engine/models/qwen.py:196` — `self._kv_block_size = 256`（硬编码）
  - `engine/models/qwen.py:255` — `self._block_table = torch.arange(...)`（**独立顺序分配，完全绕过 BlockManager**）
- **物理事实**：QwenTPModelRunner **不使用** Scheduler 分配的 `Sequence.block_table`，也不使用 BlockManager 的逻辑块。模型的 KV cache 是自管理的，按 256 token 块大小顺序分配。Scheduler 的 16-token 粒度 paging 系统在 TP Runner 路径下是**死代码**。
- **修正规格**：
  1. 蓝图应明确标注："框架层 paging（Scheduler/BlockManager）仅在 HF 兜底路径（RealModelRunner）生效，TP Runner 自管 KV cache"
  2. 或者：将 LLMEngine 的 `block_size` 默认值改为 256，并使 QwenTPModelRunner 使用 BlockManager 分配的 block_table

#### CRITICAL-02：RealModelRunner 违反 HF OOM Guard

- **JSON Path**：`global_primitives_constraints.hf_baseline_test_oom_guard`
- **蓝图要求**："HF 模型保持 CPU 驻留；仅在 forward 时将 input_ids/attention_mask 放到目标设备"
- **代码冲突**：`llm_engine.py:118-125`：
  ```python
  self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, ...)
  self.model.to(self.device)  # ← 模型常驻 GPU
  ```
- **物理事实**：多进程 `torchrun` 下，每个 rank 的 RealModelRunner 将完整 HF 模型 `.to(cuda)`，N 个 rank 同时占用 N 份全量模型显存 → OOM 高风险
- **修正规格**：RealModelRunner 保持 `.to('cpu')`，forward 时按需搬运输入

#### CRITICAL-03：蓝图 rmsnorm_precision_law.reference_pattern 与实际实现不匹配

- **JSON Path**：`model_layer.architecture_knowledge_base.global_primitives_constraints.rmsnorm_precision_law`
- **蓝图 reference_pattern**：`x_fp32 = x.float(); var = x_fp32.pow(2).mean(...); out = (x_fp32 * rsqrt(var+eps) * weight).to(x.dtype)`（手动 PyTorch 实现）
- **实际实现**：`engine/kernels/vllm_wrappers.py:12-27` — `_vllm_rms_norm(out, input, weight, epsilon)`（vLLM CUDA kernel wrapper）
- **物理事实**：两者数值语义等价（vLLM kernel 内部也是 fp32 计算），但蓝图从未提及 kernel wrapper 的存在。Sub-Agent 若按蓝图 pattern 写出纯 PyTorch RMSNorm，将丢失 vLLM fused kernel 的性能优势
- **修正规格**：补充 `rmsnorm_precision_law`，注明实际使用 `engine/kernels/vllm_wrappers.py::rms_norm` 作为黑盒 kernel wrapper

---

### 🟡 Warnings（描述模糊项，4 项）

| # | 节点 | JSON Path | 问题 |
|---|------|-----------|------|
| W1 | **routed_probability_patch** 不完整 | `model_layer.deepseek_v2_v3_mla_moe.routed_probability_patch` | 蓝图要求 "若开启 norm_topk_prob 必须除以权重和"。代码 `engine/tp_layers/moe.py:76` 只做了 `routed_scaling_factor`，无 `norm_topk_prob` 的显式检查和归一化分支 |
| W2 | **FM-008** 探针仅在模型层存在 | `failure_mode_library.entries[7]` | 蓝图 FM-008 警告 "block_size < 256 导致 flash_attn_with_kvcache 报错"，防御探针存在于 `qwen.py`（硬编码 256），但 LLMEngine/Scheduler 无门禁检查 |
| W3 | **FM-009** (.item() 安全) 未运行期验证 | `failure_mode_library.entries[8]` | 蓝图描述 `.item()` 必须在 "compiled region 外部执行"。`qwen.py` 的 kv_len 读取机制是否满足此约束，本次审计未实际执行编译路径，无法确认 |
| W4 | **engine/model_runner.py** trace artifact | `data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime` | `engine/model_runner.py:20` 使用 `engine.tp_distributed`（旧模块），与 `engine/tp_layers/distributed.py`（新模块）不同。该文件仅用于 TDD 玩具测试，建议标注 deprecated |

---

## 4. 故障模式库探针覆盖率

| FM ID | 类别 | 代码中探针 | 状态 |
|-------|------|-----------|------|
| FM-001 | TP Embedding 双重切片 | `embedding.py:38` | ✅ 已注入 |
| FM-002 | RMSNorm contiguous 约束 | `vllm_wrappers.py` 文档注释声明 | ✅ 已注入（文档级） |
| FM-003 | fused_add_rms_norm 跨层 weight | 未在本次审计的模型代码中逐层验证 | ⚠ 待验证 |
| FM-004 | CosSinCache 格式 + 显存 | `vllm_wrappers.py:90-113` 缓存格式正确 | ✅ 已注入 |
| FM-005 | CustomAR gloo ProcessGroup | `distributed.py:57` | ✅ 已注入 |
| FM-006 | QKV weight 拼接索引 | `linear.py:140-142` 三段复制 | ✅ 已注入 |
| FM-007 | RoPE Neox vs GPT-J | `vllm_wrappers.py:74` is_neox 参数 | ✅ 已注入 |
| FM-008 | paged KV block_size >= 256 | `qwen.py:196` 硬编码 256，LLMEngine 未同步 | ⚠ 探针仅模型层有 |
| FM-009 | compiled region .item() | 未运行期验证 | ⚠ 待验证 |
| FM-010 | reduce-overhead vs KV cache | `custom_ops.py` custom_op 注册 | ✅ 已注入 |
| FM-011 | flash_attn 无法 compile trace | `custom_ops.py:15-34` | ✅ 已注入 |
| FM-012 | 无条件 clone 性能回退 | 待验证 forward_decode/forward_decode_graph 分支 | ⚠ 待验证 |
| FM-013 | Dynamo RNG 重编译 (TP=4) | `distributed.py:75-93` custom_op | ✅ 已注入 |
| FM-014 | cuBLAS 图池地址漂移 | 通信 eager 执行（sglang 切图方案，未实施） | ⚠ 待验证 |
| FM-015 | mutated inputs → cudagraphs 跳过 | 待验证 clone 机制 | ⚠ 待验证 |

---

## 5. 审计统计

```
审计节点总数：           32
🟢 Passed:              12 (37.5%)
🔴 Critical Violations:  3 (9.4%)
🟡 Warnings:             4 (12.5%)
⚠ 待运行期验证:         13 (40.6%)
```

---

## 6. 核心发现总结

蓝图描述了一个 Scheduler → BlockManager → ModelRunner **统一 paging** 的架构，但实际 **TP Runner（QwenTPModelRunner、DeepseekTPModelRunner）完全绕过了这套系统**，自己管理 KV cache（硬编码 `_kv_block_size=256`、`torch.arange` 自建 block_table）。框架层的 paging 系统仅在 HF 兜底路径（RealModelRunner）生效。

这意味着当蓝图作为"Agent 实现指南"时，会把 Sub-Agent 引向实现一套**不会被 TP Runner 使用的 paging 逻辑**，浪费开发时间且引入了两套不一致的块管理语义。这是当前知识图谱误导 Agent 实现的核心根因。
