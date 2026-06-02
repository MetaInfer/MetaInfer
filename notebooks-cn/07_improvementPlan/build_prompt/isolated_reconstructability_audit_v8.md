# 蓝图重构盲测漏洞报告 v8

**审计对象**: `inference_blueprint.json` v2.3.0 + `AGENT_SKILL.md`
**审计范围**: 排除 deepseek 章节，聚焦 Qwen3 TP=4 重构可行性
**审计日期**: 2026-05-27

---

## 🔴 Fatal Gaps（致命脑补漏洞项）

### FG-1: Python 类层级与 `__init__` 签名完全缺失

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces`（全体）
- **理由**: 图谱详细描述了 QwenAttentionTP、QwenMLPTP、QwenDecoderLayerTP、QwenForCausalLMTP、QwenTPModelRunner 五个类的**行为语义**，但完全没有给出任何一类的 `__init__` 方法签名、模块属性名、或类继承关系。例如：
  - `QwenAttentionTP.__init__` 需要创建 `self.qkv_proj`（QKVColumnParallelLinear）、`self.o_proj`（RowParallelLinear）、`self.q_norm`、`self.k_norm`（RMSNorm）、paged KV cache buffer。这些模块的具体参数（如 `q_size`、`kv_size`、`num_blocks` 计算公式）散落在 JSON 各处但从未被组装成一个明确的构造函数。
  - `QwenDecoderLayerTP` 需要创建 `self.self_attn`、`self.mlp`、`self.input_layernorm`、`self.post_attention_layernorm`。这些属性名若猜错（如写成 `self.attention` 而非 `self.self_attn`），权重加载的 HF key 映射将全部失效。
- **后果**: 我无法写出正确的 Python 类定义。属性名靠猜 → 权重加载必然失败 → 引擎无法启动。

### FG-2: QwenDecoderLayerTP.forward() prefill 路径缺乏实现级伪代码

- **JSON Path**: `qwen3_tp_model_interfaces.prefill_forward_pattern.full_dataflow`
- **理由**: 该字段给出了 8 条高层步骤描述（如 "逐层 QwenDecoderLayerTP.forward(hidden_states, positions, layer_cache=None, max_seq_len, residual=None)"），但相比 `decode_forward_pattern` 拥有明确的函数签名、参数类型、返回值合约，prefill 的 `forward()` 方法缺少：
  - 完整的 residual chain 伪代码（fused_add_rms_norm 的约束中给了通用残差链，但未区分 prefill/decode 路径差异）
  - prefill 中 `layer_cache=None` 的含义和传递方式
  - 多序列 ragged batch 下 positions 的构造方式
- **后果**: decode 路径可精确实现，但 prefill 路径需要大量脑补。prefill 写错 → 首 token 错误 → 后续 decode 全链崩塌。

### FG-3: 权重加载的 HF key → 自定义模块属性映射表未在 JSON 本体中给出

- **JSON Path**: `model_layer.lazy_loader_synthesis_rules.qwen_dense_loader`
- **理由**: JSON 仅给出 `split_dim_0: ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]`，但没有说明：
  - HF key `model.layers.0.self_attn.q_proj.weight` 如何映射到自定义的 `QKVColumnParallelLinear`（属性名是 `qkv_proj`？）
  - 三个独立 HF 权重（q_proj, k_proj, v_proj）如何在合并线性层中拼接（顺序是 q-k-v？）
  - `gate_proj` 和 `up_proj` 如何在 `MergedColumnParallelLinear` 中拼接
- **注**: 该映射在 `ref_docs` 的 `02_qwen_dense_tp_implementation_guide.md` 第 175-179 行中有记录（`.q_proj` → `.qkv_proj` 等），但 JSON 本体未内联此关键信息。若 RAG 检索失败或 Agent 未主动查阅该文档，则 100% 写错。
- **后果**: 权重加载是引擎启动的第一道关卡，此映射错误 → 直接 shape mismatch 崩溃。

### FG-4: QwenTPModelRunner 的 `run()` 方法和 prefill/decode 分发逻辑缺失

- **JSON Path**: `framework_layer.components[3].tp_runner_actual_flow`（ModelRunner）
- **理由**: 该字段给出了 engine 层的调用伪代码（`tokens = runner.run(batch, is_prefill=is_prefill, ...)`），但没有给出 `run()` 方法内部的实现逻辑：
  - 如何从 batch（list[Sequence]）组装 prefill ragged tensors（input_ids, positions, block_tables）
  - 如何从 batch 组装 decode tensors
  - `is_prefill=True` 时调用 `model.forward()` vs `is_prefill=False` 时的分发逻辑
  - `kv_lens` 的 batch 读取和更新时机
- **后果**: `run()` 是连接 Scheduler 和 Model 的枢纽。此方法写错 → 整个推理循环断裂。

### FG-5: NCCL 初始化与设备放置的精确时序未指定

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime`
- **理由**: 该节仅给出 `tp_rank = dist.get_rank()` 和 `all_reduce_sum` 的输入输出 shape。但缺失：
  - `dist.init_process_group("nccl", ...)` 的确切调用时机（在 LLMEngine 还是 ModelRunner？）
  - `torch.cuda.set_device(rank)` 的调用时机
  - 多进程 torchrun 下的初始化序列（barrier 位置、timeout 设置）
  - 默认 device 的设置策略（`torch.set_default_device("cuda")` 还是手动 `.to(device)`？）
- **后果**: NCCL 初始化时序错误 → 死锁或 "NCCL WARN Duplicate init" 崩溃。

---

## 🟡 Override Warnings（重载失效警告项）

### OW-1: nano-vllm Scheduler 使用抢占（preempt），但 JSON 明确禁止抢占——差异分散在多处

- **ref_code**: `ref_projects/nano-vllm/nanovllm/engine/scheduler.py`（第 52-57 行有 `preempt()` 逻辑）
- **JSON 约束**: `scheduler_to_runner.preemption_policy: "disabled; 资源不足时不抢占，只等待"`
- **风险**: nano-vllm 的 `schedule()` 在 decode 阶段资源不足时主动抢占运行中序列（第 54 行 `self.preempt(self.running.pop())`）。若重构者直接抄入 nano-vllm 的 `schedule()` 方法，会引入 JSON 明确禁止的抢占行为。虽 JSON 在多处强调此差异，但 `schedule()` 方法的实现伪代码未给出，重构者大概率直接从 nano-vllm 搬运整段代码。

### OW-2: nano-vllm BlockManager 使用 xxhash + 链式哈希，JSON 要求 Python builtin hash

- **ref_code**: `ref_projects/nano-vllm/nanovllm/engine/block_manager.py`（第 2、36-41 行使用 `xxhash.xxh64`）
- **JSON 约束**: `BlockManager.api_spec.compute_hash.algorithm_detail: "hash(tuple(token_ids[block_start:block_start+block_size])) — Python builtin hash"`
- **风险**: JSON 的 `api_spec` 给出了详细的 Python hash 方案，但若重构者直接复制 nano-vllm 的 `BlockManager` 代码（xxhash 依赖），会引入额外依赖且与 JSON 的 `hash_to_block_id` 映射表设计不一致。JSON 的 `compute_hash` 签名和算法与 nano-vllm 完全不同。

### OW-3: nano-vllm ModelRunner 使用 contiguous KV cache + CUDA Graph，但 TP Runner 使用 paged KV cache

- **ref_code**: `ref_projects/nano-vllm/nanovllm/engine/model_runner.py`（第 37 行 `self.capture_cudagraph()`，第 35 行 `self.allocate_kv_cache()`）
- **JSON 约束**: paged KV cache `[num_blocks, 256, num_kv_heads, head_dim]` + `block_size=256`
- **风险**: nano-vllm 的 KV cache 是 contiguous buffer（非 paged），其 `allocate_kv_cache()` 和 CUDA Graph capture 逻辑与 TP Runner 完全不同。JSON 在多处强调此差异（`_dual_track_note`、`paged_kv_cache_contract`），但若重构者不仔细阅读全部 JSON 节点，极易被 nano-vllm 的 contiguous 方案带偏。

### OW-4: `fused_add_rms_norm` 的 post_mlp weight 使用本层而非下一层——与 vLLM 原始设计不同

- **JSON Path**: `qwen3_kernel_contracts.fused_add_rms_norm.constraint.physical_layout_correction`
- **JSON 约束**: "meta-infer 简化: post_mlp 的 fused_add_rms_norm 使用本层 post_attention_layernorm.weight（非下一层 input_layernorm.weight）"
- **风险**: kernel_replacement_plan.md §9.2 明确写着 "post-mlp 调用时 weight 必须是**下一层**的 `input_layernorm.weight`"，与 JSON 约束直接矛盾。若重构者按 kernel_replacement_plan.md 实现 → 数值错误。JSON 虽声明了此简化并称"数值等价"，但这是一个反直觉的设计决策，极易被忽略。

### OW-5: nano-vllm Scheduler 的 `block_size=16` 与 TP Runner 的 `block_size=256` 冲突

- **ref_code**: nano-vllm 默认 block_size=16（由其 Config 注入）
- **JSON 约束**: `scheduler_tp_runner_bridge.llm_engine_block_size_injection` — TP 路径必须注入 `scheduler._block_size = 256`
- **风险**: 若重构者先实现框架层 Scheduler（block_size=16），再接入 TP Runner 时忘记注入 256 → `max_num_batched_tokens` 计算错误 → 调度器行为异常 → 显存预算偏差。

### OW-6: BlockManager 在 TP 路径下降级为 no-op——架构角色翻转

- **JSON Path**: `scheduler_tp_runner_bridge.block_manager_role_in_tp_path`
- **JSON 约束**: "allocate()/free() 为 no-op。block_table 的实际分配由 QwenAttentionTP torch.arange 完成。"
- **风险**: 这是整个框架最反直觉的架构决策。nano-vllm 的 BlockManager 是核心分配器，但在 TP Runner 路径下它被完全架空。若重构者不仔细阅读 `scheduler_tp_runner_bridge` 整节（该节位于文件末尾，容易被遗漏），必然写出双重分配的 bug。

---

## 🟢 Reconstructability Score（重构可行性判决）

### 量化分数: **68%**

以当前掌握的信息（JSON + AGENT_SKILL.md + RAG 访问 ref_code/ref_docs），我有 68% 的把握写出能一次性跑通的 Qwen3 TP=4 引擎（不含 CUDA Graph）。

### Top 3 阻断因素（阻碍达到 100%）:

1. **类层级与 `__init__` 签名的系统性缺失（权重 ~40%）**: 图谱在"做什么"层面极其详尽（shape 合约、数据流、算法伪代码），但在"怎么声明"层面几乎空白。5 个核心类的构造函数签名和属性名必须靠猜测 + 参考代码拼凑。任何一个属性名猜错 → 权重加载失败 → 0% 可运行。

2. **prefill 路径的实现精度远低于 decode 路径（权重 ~35%）**: decode 有 `forward_decode` 的完整签名、residual chain 伪代码、每步 KV write/read 语义。prefill 只有 8 条高层描述，缺少 `forward()` 方法的等价伪代码。prefill 正确性是 decode 正确性的前提——prefill 错了，后续全错。

3. **RAG 依赖链的风险集中（权重 ~25%）**: 权重映射表（FG-3）、CustomAR 的完整 wrapper 代码（kernel_replacement_plan.md §三 Snippets）、Qwen TP 实现指南中的合并投影细节——这些关键信息都在 JSON 本体之外。JSON 的 `source_refs` 字段是间接引用而非内联。若 RAG 检索流程中任何一个环节遗漏或 Agent 未主动查阅相关文档，重构将立即失败。

### 积极方面（为什么不是 0%）:

- paged KV cache 合约和 slot mapping 伪代码几乎是"抄写级"完整
- 所有 Tensor shape 和 dtype 精确到 dim 级别
- Qwen3-8B 的 10 个物理维度参数和 tp4_per_rank 计算全部给出
- 16 条 Failure Mode 知识库覆盖了最常见的踩坑点
- kernel_replacement_plan.md §九 提供了 7 个 vLLM kernel 的完整 API 契约
- `scheduler_tp_runner_bridge` 详细解决了双轨 block_size 和 num_free_blocks 来源问题

### 结论

这份蓝图在**数据流合约层**达到了工业级精度，但在**软件工程结构层**（类设计、模块组合、初始化序列）存在系统性盲区。它是一份优秀的"白盒合约说明书"，但不是一份"可执行的重构施工图"。要从这份蓝图重建 Qwen3 TP=4 引擎，一个有经验的分布式推理工程师大约需要 3-5 天，其中约 40% 的时间会花在逆向推断类结构和权重加载映射上——而这些信息在本该被"销毁"的 `impl_code` 文件中原本是显然的。
