# 蓝图重构盲测漏洞报告

**审计对象**: `inference_blueprint.json` (v2.3.0, 2026-04-30) + `AGENT_SKILL.md`
**审计范围**: Qwen3-8B, nocompile, B=1, TP=4（排除 deepseek 章节）
**审计身份**: Zero-Shot Reconstructability Auditor
**审计日期**: 2026-05-27

---

## 前置：ref_docs 交叉验证执行记录（维度四强制操作）

按照审计要求，实际打开了以下 ref_docs 并逐段验证：

| # | ref_doc | 验证的知识点 | 结果 |
|---|---------|-------------|------|
| 1 | `kernel_replacement_plan.md §九` (line 1429) | 7 个 kernel 调用契约（rms_norm, fused_add_rms_norm, silu_and_mul, rotary_embedding, CustomAR all_reduce, cos_sin_cache） | **存在，签名完整**。§9.1-9.6 包含精确的函数签名、dtype 约束、shape 要求、in-place 语义。与蓝图 `qwen3_kernel_contracts` 中引用的签名一致。 |
| 2 | `kernel_replacement_plan.md §三` (line 168) | Snippets A-F 纯净 wrapper 代码 | **存在，代码完整**。每个 Snippet 含完整 Python 函数体、数据契约注释、vLLM 源码行号引用。 |
| 3 | `improvement_plan.md §P0` | 增量 KV Cache 解码实现记录 | **存在**。描述了 P0 的实现方式（方案 A：自定义 TP attention 层内置 KV cache）和踩坑记录，但没有提供具体的 paged KV cache 实现伪代码。 |
| 4 | `qwen3_effective_changes.md` | P0-P3 全部有效改动点追溯 | **存在**。包含 7 个 kernel 替换 + 3 个框架层改动，含参考源码路径和踩坑记录。 |
| 5 | `notebooks/01_framework_design/02_scheduler.md` | Scheduler 设计文档 | **存在**。包含调度算法描述、prefill/decode 选择逻辑、postprocess 模板。是设计文档而非实现契约。 |

**跨文档冲突发现**: `kernel_replacement_plan.md §9.2` 明确要求 post-mlp 的 `fused_add_rms_norm` 使用**下一层**的 `input_layernorm.weight`（跨层依赖），但蓝图 `qwen3_kernel_contracts.fused_add_rms_norm.constraint` 明确声明使用**本层** `post_attention_layernorm.weight`。蓝图已标注此为已知差异（`_kernel_replacement_plan_conflict`）并裁决"以蓝图为准"。重构者若先读 ref_doc 而遗漏蓝图中的冲突标注，会按错误方案实现。

---

## 🔴 Fatal Gaps（致命脑补漏洞项）

### FG-1: Scheduler.schedule() 完整方法体缺失

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm` + `scheduler_tp_runner_bridge.prefill_timing_gap`
- **严重程度**: 高
- **现状**: 蓝图以英文描述了 `schedule_algorithm` 的阶段划分：
  - phase1_prefill: 从 waiting 队列取 seq，满足 can_allocate 且累计 tokens ≤ max_num_batched_tokens
  - phase2_decode: waiting 空时从 running 取 seq，满足 can_append_one_more
  - empty_return: `([], False)` — 调用方忙等或检查 all_finished 退出
  - chunked_prefill_rule: prefill/decode 不混批
  - state_transition_safety: 状态在 postprocess 中更新
  - overlength_rejection: enqueue 时检测 REJECTED

  `prefill_timing_gap.pseudocode` 给出了带 `_reserved_blocks` 计数器的骨架：
  ```python
  def schedule(num_free):
    reserved = 0; batch = []
    for seq in waiting:
      req = ceil(len(seq.input_ids) / self._block_size)
      if reserved + req <= num_free and _check_max_tokens(batch, seq):
        batch.append(seq); reserved += req
    return (batch, True) if batch else _select_decode(num_free - reserved)
  ```

- **缺失的关键逻辑**:
  1. `_check_max_tokens(batch, seq)` 的完整实现（token 累计、max_num_batched_tokens 动态计算）
  2. `_select_decode(remaining_free)` 从 running 队列中选择 can_append_one_more 序列的完整逻辑
  3. `schedule()` 返回的 `is_prefill` 标志与 `batch` 的对应关系保证（同 batch 同阶段）
  4. `_reserved_blocks` 计数器的生命周期（每次 schedule() 调用重置？还是持续累积？）
  5. REJECTED 状态序列在 schedule() 中的过滤逻辑（蓝图说"已拒绝不参与调度"但 scheduling 伪代码中未体现）

- **ref_docs 补充情况**: `notebooks/01_framework_design/02_scheduler.md` 是概念性设计文档，只描述了 3 种调度策略的理念差异（prefill-priority vs interleaved vs overlap），提供了 5 行 nano-vllm 原始 `schedule()` 骨架，**不含** TP 路径修改后的 `schedule(num_free)` 签名和 `_reserved_blocks` 逻辑。`07_request_lifecycle.md` L48-73 提供了 loop 级别伪代码但跟蓝图 `engine_integration` 同一抽象层级。**两个设计文档均未填补此缺口。**

- **重构后果**: 重构者需从英文描述中自行合成 ~60 行的 schedule() 方法。最可能的 bug：（1）未正确处理 `num_free_blocks` 在 prefill 调度时的"乐观调度"问题——prefill 前 _kv_len_gpu 为旧值，导致 num_free 高估；（2）空批次处理不当引发死循环或过早退出；（3）`_reserved_blocks` 未在 decode 阶段正确扣除导致 OOM。

---

### FG-2: LLMEngine.step() + generate() 完整主循环缺失

- **JSON Path**: `framework_layer.components[5]` (LLMEngine) + `framework_layer.components[3].tp_runner_actual_flow.engine_integration`
- **严重程度**: 高
- **现状**: 蓝图 `engine_integration` 以一行伪代码概括：
  ```python
  num_free = block_mgr.get_num_free_blocks() or runner.get_num_free_blocks()
  batch, is_prefill = scheduler.schedule(num_free)
  tokens = runner.run(batch, is_prefill=is_prefill, temperature=temperature, top_p=top_p)
  scheduler.postprocess(batch, is_prefill, generated_tokens=tokens)
  ```

- **缺失的关键逻辑**:
  1. `all_finished()` 判断条件与循环退出逻辑（waiting 空 + running 空？还是所有 seq FINISHED？）
  2. `generate()` 方法的完整实现：prompt → tokenize → Sequence 构造 → enqueue → step-loop → 收集 output_ids → decode 文本
  3. 空批次 `([], False)` 时的处理（忙等重试？直接 break？sleep 后 retry？）
  4. `num_free_blocks` 来源选择的具体分支代码（`if self.inference_backend in ('qwen_tp', 'deepseek_tp'): num_free = runner.get_num_free_blocks(); else: num_free = block_mgr.get_num_free_blocks()`）
  5. REJECTED 状态的序列如何从 waiting 队列移除并向调用方反馈错误
  6. Scheduler block_size 注入的时机和方式（LLMEngine.__init__ 中设置 `scheduler._block_size = 256`）

- **ref_docs 补充情况**: `01_architecture.md` 只有架构层次图和组件列表。`07_request_lifecycle.md` L48-73 提供了跟蓝图同层级的伪代码循环（`while not seq.is_finished(): schedule → run → postprocess`），且使用的是单序列循环而非队列模式。**未填补此缺口。**

- **重构后果**: 重构者需自行设计 ~40 行的引擎主循环。空批次处理错误会导致（1）死循环 100% CPU 占用（忙等）或（2）等待队列还有请求但引擎提前退出。REJECTED 处理缺失会导致超长 prompt 永久卡在 WAITING 状态使引擎死循环。

---

### FG-3: Scheduler.postprocess() 完整方法体缺失

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner` + `framework_layer.data_flow_contracts.request_level.status_transitions`
- **严重程度**: 高
- **现状**: 蓝图多处提到 postprocess 负责"推进状态并触发 token 级块扩展"，但从未提供 postprocess 的伪代码。状态转移表只有 3 行：
  ```
  WAITING -> RUNNING_PREFILL
  RUNNING_PREFILL -> RUNNING_DECODE
  RUNNING_DECODE -> FINISHED
  ```

- **缺失的关键逻辑**:
  1. generated_tokens 如何写入 Sequence.output_ids（`seq.output_ids.append(token)`？`seq.output_ids.extend(tokens)`？）
  2. 状态转移的具体执行时机和条件——prefill batch 处理完后所有序列批量转 DECODE？还是只转已完成 prefill 的？
  3. FINISHED 的触发条件检测代码（EOS token ID 比较？max_tokens 比较？）
  4. FINISHED 时释放 KV cache 的具体调用链（TP 路径下 `free()` 是 no-op，实际 block 如何回收？`_kv_len_gpu` 需重置吗？）
  5. KV_len 更新——蓝图说 prefill 后 `s.kv_len = s.seq_len()`，decode 后 `s.kv_len = new_kv_lens[i]`，但这些更新是在 postprocess 还是 runner.run() 中？

- **ref_docs 补充情况**: `07_request_lifecycle.md` L67-70 以 4 行注释描述了 postprocess 做什么（"Appends token / Checks EOS / If finished: deallocate"），但**无一字伪代码**。`02_scheduler.md` L159-168 有 nano-vllm 原始 postprocess 伪代码，但那是 HF 路径（调用 `block_manager.deallocate`），与 TP 路径（no-op）不同。

- **重构后果**: 状态机转移错误（如 prefill 完了不转 decode、decode 永久不转 finished）导致序列永远卡在某状态。EOS 检测遗漏导致模型无限生成直到 max_tokens。资源释放逻辑错误在长对话中造成显存泄漏。

---

### FG-4: QwenAttentionTP prefill slot_mapping 向量化实现缺失

- **JSON Path**: `framework_layer.data_flow_contracts.paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm`
- **严重程度**: 中
- **现状**: 蓝图给出的 pseudocode 使用 Python for 循环 + `.item()`：
  ```python
  for i in range(num_tokens):
      slot_mapping[i] = block_table[0, i//256].item() * 256 + (i % 256)
  ```
- **问题**: 对 max_position_embeddings=32768 的 Qwen3-8B，长 prompt 会产生 32768 次 GPU→CPU 同步（`.item()`），每次 ~5μs，累计 ~160ms 纯同步开销，prefill 延迟不可接受。蓝图未提供向量化替代方案（如 `torch.arange` + `repeat_interleave` 的组合公式）。
- **重构后果**: 功能正确但性能灾难。在 nocompile B=1 场景下，短 prompt（< 512 tokens）可正常工作，长 prompt（如 32K tokens）会产生 32768 次 GPU→CPU 同步，prefill 延迟不可接受。不是正确性缺陷，但限制了实际可用范围。
- **ref_docs 补充情况**: 无。`improvement_plan.md` 和 `qwen3_effective_changes.md` 均未涉及 slot_mapping 向量化。

---

---

## 🟡 Override Warnings（重载失效警告项）

### OW-1: nano-vllm Scheduler — 修改范围以文字描述为主，缺少逐行对照表

- **JSON Path**: `framework_layer.components[0]._nano_vllm_override`
- **现状**: 蓝图用一段文字描述了 10+ 处修改（删除 preempt、替换 block_size 引用、修改 can_append 逻辑、替换 running.pop 为顺序迭代、替换 3 处 self.block_manager.xxx() 调用等）。但没有给出**逐行修改对照表**（原行号 → 新代码）。
- **风险**: 重构者可能遗漏某处 `self.block_manager` 调用，导致 TP 路径下调用了被降级为 no-op 的 BlockManager 方法。
- **严重程度**: 中

### OW-2: nano-vllm ModelRunner.prepare_prefill() — PARTIAL 修改指引不够精确

- **JSON Path**: `framework_layer.components[3].tp_runner_actual_flow._nano_vllm_per_function.prepare_prefill()`
- **现状**: 蓝图标注为 "PARTIAL — cu_seqlens 构造可参考 L129-150, contiguous slot_mapping (L154-162) 必须替换为 paged slot_mapping_algorithm"。但 nano-vllm 的 `prepare_prefill()` 中还有 `attn_metadata` 构造、`block_table` 传递等其他内部逻辑，哪些保留、哪些删除、哪些替换，全凭重构者自行判断。
- **风险**: 重构者可能保留 contiguous slot_mapping（与 paged KV cache 不兼容），导致 flash_attn_with_kvcache 读取错误 KV 位置。
- **严重程度**: 中

### OW-3: fused_add_rms_norm 蓝图与 kernel_replacement_plan.md 的跨层 weight 冲突

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint._kernel_replacement_plan_conflict`
- **现状**: 蓝图说 post-mlp fused_add_rms_norm 用本层 `post_attention_layernorm.weight`；ref_doc `kernel_replacement_plan.md §9.2` 说用下一层 `input_layernorm.weight`。蓝图已标注冲突并裁决"以蓝图为准，数值等价"。但 **ref_doc §9.2 的跨层 weight 描述非常详细且位于文档显眼位置**（速查表），重构者可能先信了 ref_doc 而遗漏蓝图深层 JSON 中的冲突标注。
- **风险**: 按 ref_doc 实现会用下一层 weight，第 36 层（最后一层）访问不存在的第 37 层 weight 导致 IndexError 崩溃。
- **严重程度**: 中。蓝图已有冲突标注，但标注位置深（JSON path 嵌套 8 层），易被遗漏。

### OW-4: nano-vllm BlockManager — TP 路径降级的 allocate/free no-op 缺实现伪代码

- **JSON Path**: `framework_layer.components[2]._nano_vllm_override`
- **现状**: 蓝图说 "allocate/free 改为 no-op (仅 count+=1/count-=1)"。但未给出改造后的方法体伪代码。allocate 的返回值类型（原返回 `list[int]` block IDs，no-op 后返回什么？）和处理边界不明确。
- **风险**: 如果 allocate no-op 返回值类型不匹配，LLMEngine 中调用方可能类型错误（期望 list 但得到其他类型）。
- **严重程度**: 低

### OW-5: QKVColumnParallelLinear 构造函数参数命名不一致

- **JSON Path**: 对比 `qkv_merged_projection.inline_signature` 与 `class_hierarchy.QwenAttentionTP.constructor`
- **现状**: Kernel 契约签名是 `(hidden_size, head_size, total_num_heads, total_num_kv_heads, ...)`，但 class_hierarchy 实际调用是 `(cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads)`。参数名 `head_size` vs `head_dim` 不一致，`total_num_heads` vs `cfg.num_attention_heads` 描述方式不同。语义等价但容易混淆。
- **风险**: 重构者可能用 `head_size` 参数名但传入 head_dim 值，导致 QKV 总尺寸计算偏差。
- **严重程度**: 低

---

## 🟢 Reconstructability Score（重构可行性判决）

### 量化分数: **78%**

以当前掌握的信息（JSON 蓝图 + AGENT_SKILL.md + RAG 可检索的全部 ref_docs + ref_code），有 **78%** 的把握写出能一次性跑通的 Qwen3 TP=4 nocompile B=1 引擎。

---

### 得分明细

| 类别 | 得分 | 满分 | 说明 |
|------|:----:|:----:|------|
| 模型维度参数 | 100% | 10 | `qwen3_8b_model_dims` + `tp4_per_rank` 完整给出所有数值 |
| QwenAttentionTP forward/decode | 90% | 15 | forward_decode 完整方法体可直接使用；prefill forward 完整；扣分在 slot_mapping 向量化 |
| QwenDecoderLayerTP + QwenForCausalLMTP | 90% | 10 | 层 forward 伪代码 + 顶层编排伪代码均完整 |
| TP 线性层 (QKV/Column/Row/Merged) | 90% | 10 | 含伪代码和签名；扣分在构造函数参数命名不一致 |
| TP Embedding/LM Head | 95% | 10 | VocabParallelEmbedding + ParallelLMHead 伪代码完整，含边界情况 |
| TP 通信 (all_reduce/all_gather/CustomAR) | 85% | 10 | init 序列 + collectives 签名完整；扣分在 register_buffer 细节复杂 |
| 7 个 vLLM kernel 调用契约 | 95% | 10 | kernel_replacement_plan.md §九 自包含，含完整签名和约束 |
| Scheduler | 30% | 10 | schedule() 只有骨架，postprocess() 只有文字描述 |
| BlockManager + KVMemoryPool | 80% | 5 | API 签名完整，TP 降级 no-op 规则明确 |
| LLMEngine 主循环 | 30% | 5 | engine_integration 一行伪代码 |
| Sequence 状态机 | 60% | 5 | 字段和状态转换列出，但缺 postprocess 中的实际转移代码 |

---

### 阻碍达到 100% 的 Top 3 阻断因素

| # | 阻断因素 | 丢失的代码量（估算） | 缺失的具体信息 |
|---|---------|:---:|------|
| 1 | **Scheduler 完整方法体** | ~80 行 | `schedule(num_free)` 的完整决策树（prefill 分派 → token budget 检查 → decode 回退 → 空批次返回）；`postprocess(batch, is_prefill, tokens)` 的状态转移 + EOS 检测 + 资源释放代码 |
| 2 | **LLMEngine step()/generate() 主循环** | ~40 行 | enqueue → schedule → run → postprocess → check_finished 的完整 while 循环；inference_backend 路由；空批次处理策略 |
| 3 | **跨文档冲突融合与 slot_mapping 向量化** | ~15 行 | fused_add_rms_norm weight 来源的冲突裁决分散在多处；slot_mapping 构造需要从 .item() 循环改为向量化公式 |

---

### 做得好的部分（不需要脑补的区域）

- **Kernel 调用层**: `kernel_replacement_plan.md §九` 的 7 个 kernel 契约自包含、可复制。每个 kernel 的签名、dtype、shape、in-place 语义、边界约束均已明确。
- **模型类层级**: `class_hierarchy` 给出了 QwenTPConfig → QwenAttentionTP → QwenMLPTP → QwenDecoderLayerTP → QwenForCausalLMTP 的完整构造函数和属性列表，属性命名与 HF key 映射直接对应。
- **Prefill/Decode 热路径**: `forward_decode` 完整方法体（~30 行）和 `prefill_forward_pattern`（~40 行伪代码）基本可直接使用。
- **TP 权重加载**: `qwen_hf_key_mapping` + `load_weights_pseudocode` + `_load_tensor` 提供了完整的 HF key → module attribute 映射和 QKV/MLP 合并逻辑。
- **失败模式库**: FM-001 到 FM-015 覆盖了 15 个已知踩坑点，含 symptom → check → fix 三元组和源码引用。
- **Paged KV Cache 契约**: slot_mapping algorithm、decode KV write、block_table 初始化、kv_len 追踪均详细。
- **Flash Attention 集成**: prefill (`flash_attn_varlen_func`) 和 decode (`flash_attn_with_kvcache`) 的完整调用模板已提供。nocompile 场景下直接 `from flash_attn.flash_attn_interface import flash_attn_with_kvcache` 即可，无需 custom_op 注册（custom_op + register_fake 仅在 torch.compile tracing 时需要）。
- **TP 分布式运行时**: 5 步 init_sequence + 每步的伪代码 + CustomAR 初始化 7 步完整流程。

---

## 审计后记

本报告基于"源码物理销毁 + 仅持有 JSON 蓝图 + AGENT_SKILL.md + RAG 可检索 ref_docs/ref_code"的零样本重构假设撰写。实际 `meta-infer` 工程的完整源码（`engine/` 目录下所有文件）提供了可工作的参考实现，上述 Fatal Gaps 在现有源码中已解决。本报告的 gap 标注反映的是**蓝图作为独立工程文档的完备性**，而非现有代码的质量问题。
