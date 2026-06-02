# 蓝图重构盲测漏洞报告 v1.5

**审计对象**: `inference_blueprint.json` v2.3.0 (Qwen3 nocompile B=1 TP=4 路径，排除 deepseek 章节)  
**审计日期**: 2026-05-27  
**审计方法**: 实际打开 7 个 ref_doc 文件 + 5 个 ref_code 文件，逐段交叉验证

---

## 实证追溯记录

| 审计维度 | 验证项 | 实际打开的文件 | 结果 |
|---------|--------|-------------|------|
| 维度四 | kernel 调用契约 | `kernel_replacement_plan.md` §九 (L1429-1510) | **存在且完整** — 7 个 kernel 的签名、参数、约束均有完整表格 |
| 维度四 | Snippet A-F 代码模板 | `kernel_replacement_plan.md` §三 (L168-370) | **存在** — Snippet A(rms_norm), B(fused_add_rms_norm), C(silu_and_mul), D(rotary_embedding), E(cos_sin_cache), F(CustomAR) 均有完整代码 |
| 维度四 | sglang split_graph 参考 | `ref_projects/sglang/.../compilation/backend.py` L214-257 | **存在且匹配** — `split_graph()` 函数逐行对应蓝图伪代码，`keep_original_order=True` 注释一致 |
| 维度四 | P0 增量 KV Cache 记录 | `improvement_plan.md` §P0 (L201-260) | **存在** — 记录了实现方案和踩坑经验 |
| 维度一 | nano-vllm scheduler preempt | `ref_projects/nano-vllm/nanovllm/engine/scheduler.py` L66-69 | **匹配** — `preempt()` 方法确实存在，蓝图要求删除 |
| 维度一 | nano-vllm block_manager xxhash | `ref_projects/nano-vllm/nanovllm/engine/block_manager.py` L36-41 | **匹配** — 使用 `xxhash.xxh64()`，蓝图要求替换为 Python `hash(tuple())` |
| 维度一 | nano-vllm layernorm PyTorch 实现 | `ref_projects/nano-vllm/nanovllm/layers/layernorm.py` | **匹配** — 包含手写 `rms_forward`/`add_rms_forward` 方法，蓝图要求整体替换为 vLLM kernel |
| 维度二 | nano-vllm model_runner 方法 | `ref_projects/nano-vllm/nanovllm/engine/model_runner.py` L103-258 | **指针精确** — 蓝图逐方法（allocate_kv_cache, prepare_prefill, prepare_decode, capture_cudagraph, run_model）给出 DELETE/PARTIAL 指示 |
| 维度四 | kernel_replacement_plan §9.2 跨层 weight | `kernel_replacement_plan.md` L1458 | **存在矛盾** — 文档说 post_mlp 用下一层 weight，蓝图说用本层 weight。蓝图已显式标注此差异。 |

---

## 🔴 Fatal Gaps (致命脑补漏洞项)

### FG-1: `max_blocks_for_model` 未定义 — Scheduler overlength rejection 不可实现

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm.schedule_complete_method` (L362-385)
- **具体问题**: 伪代码第 363 行 `if req > max_blocks_for_model:` 引用了未定义的变量。注释 `e.g. (32768+255)//256 = 128` 暗示了值但未说明来源。
- **死锁理由**: 重构者不知道 `max_blocks_for_model` 是从 config.json 推导（`max_position_embeddings // 256`）还是从 runner 查询（`runner.max_blocks`），还是从 KVMemoryPool 获取。更关键的是，这个值必须在 Scheduler.enqueue 时可用，而 config.json 是由 LLMEngine 读取的，二者之间的传递链路（LLMEngine → Scheduler.`_max_blocks` 注入）在蓝图中没有任何 pseudocode 或 hard_rule。
- **影响范围**: 无法正确实现超长 prompt 拒绝逻辑，导致两种错误可能：拒绝合法 prompt（值偏小）或 OOM（值偏大，REJECTED 未触发）。

### FG-2: Sequence 双轨 `block_table` 初始化时序缺口

- **JSON Path**: `framework_layer.data_flow_contracts.request_level.sequence_fields.block_table_dual_repr` (L291-304)
- **具体问题**: `block_table_tensor()` 的惰性初始化需要 `max_blocks` 和 `device` 参数。`max_blocks` 来自 `config.max_position_embeddings // 256`，但 Sequence 的构造器签名（继承自 nano-vllm）不接受这两个参数。伪代码 (L296-302) 使用 `torch.zeros(1, max_blocks, ...)` 但 `max_blocks` 和 `device` 的来源和作用域未指定。
- **死锁理由**: 必须决定：是在 Sequence 构造时传入 `max_blocks`/`device`（增加耦合），还是在 `block_table_tensor()` 首次调用时从全局状态获取（引入隐式依赖）。两种方案在蓝图和任一 ref_doc 中均无说明。

### FG-3: `QwenForCausalLMTP` 构造链路中 config 对象实例化缺失

- **JSON Path**: `tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy` (L993-1045)
- **具体问题**: `QwenForCausalLMTP.__init__(self, cfg, device, dtype)` 依赖 `cfg` 对象，但 `QwenTPConfig`（L995-1007）只是一组字段列表，没有给出从 `config.json` 到 `QwenTPConfig` 实例化的代码。蓝图外部有 `engine_routing_contract.hard_rule` 要求动态读取 config.json，但没有将 "read config.json → construct QwenTPConfig" 这个关键步骤连接到类层级。
- **死锁理由**: 类属性 `num_heads=cfg.num_attention_heads//tp` 等 per-rank 计算（L1016）能否正确执行，取决于 `cfg` 包含的是全量值还是 per-rank 值。如果重构者用原始 HF config 对象（全量值），`num_heads` 计算结果会是 32 而非 8，导致后续所有 shape 计算错误。

---

## 🟡 Override Warnings (重载失效警告项)

### OW-1: nano-vllm layernorm.py 替换链路隐蔽

- **JSON Path**: `global_primitives_constraints.rmsnorm_precision_law._nano_vllm_override` (L1565)
- **风险**: `layernorm.py` 不在任何 framework 组件的 `ref_code` 列表中。重构者按组件逐项实现时，会先参考 nano-vllm 的 Scheduler/BlockManager/ModelRunner，这些文件不涉及 RMSNorm。只有当重构者独立发现 `global_primitives_constraints` 时，才会注意到需要替换 RMSNorm。此时重构者可能已经将 nano-vllm 的 `RMSNorm`（L1-51，手写 PyTorch 实现）集成到了 attention/mlp 中。
- **建议**: 在 `ModelRunner` 或 `qwen3_kernel_contracts.rms_norm` 的 `ref_code` 中显式添加 `ref_projects/nano-vllm/nanovllm/layers/layernorm.py`，并标注 DELETE+REPLACE。

### OW-2: `kernel_replacement_plan.md` §9.2 跨层 weight 矛盾（已标注但未在 ref_doc 侧修复）

- **JSON Path**: `qwen3_kernel_contracts.fused_add_rms_norm.constraint._kernel_replacement_plan_conflict` (L1107)
- **风险**: 蓝图 (L1107) 和 AGENT_SKILL.md (L161) 明确说 "post_mlp 使用本层 post_attention_layernorm.weight"。但 kernel_replacement_plan.md L1458 原文字段 "post-mlp 调用时 weight 必须是**下一层**的 input_layernorm.weight"。重构者如果只看 ref_doc（更详细），会实现跨层方案；如果只看蓝图，会实现同层方案。虽然蓝图标注了 WARNING，但在源码销毁假设下，重构者可能只看到 ref_doc 而错过蓝图的矛盾声明。
- **建议**: 在 `kernel_replacement_plan.md` §9.2 中添加与蓝图一致的勘误注释。

### OW-3: nano-vllm ModelRunner `prepare_prefill` 的 slot_mapping 标记为 PARTIAL 但差异不够精确

- **JSON Path**: `framework_layer.components[3].tp_runner_actual_flow._nano_vllm_per_function.prepare_prefill()` (L184)
- **风险**: 蓝图说 "contiguous slot_mapping (L154-162) 必须替换为 paged slot_mapping_algorithm"。但 nano-vllm 的 slot_mapping 已经使用了 `block_table[i] * block_size + offset`（L155-162），这本身就是 paged 计算。真正的差异在于：(a) nano-vllm 的 contiguous slot_mapping 是按 block 范围展开为连续区间（`range(slot_start, slot_end)`），而蓝图的 slot_mapping 是按 token 逐个计算；(b) nano-vllm 使用 `self.block_size`（可能是 16），而蓝图强制 256。重构者可能认为 nano-vllm 的实现"已经是 paged 了"而直接复用，忽略了 block_size=256 的硬性要求和 index_copy_ 的接口差异。
- **建议**: 明确写出 nano-vllm slot_mapping 与蓝图 slot_mapping 的 3 个具体差异点（block_size 值、range vs per-token、int32 vs int64 dtype），而非仅标注 PARTIAL。

### OW-4: Scheduler 的 `_reserved_blocks` 生命周期与 runner 的实际 KV 分配不同步

- **JSON Path**: `scheduler_tp_runner_bridge.prefill_timing_gap` (L1425-1437)
- **风险**: 蓝图设计了一个 `_reserved_blocks` 计数器来补偿 "prefill 调度在 forward 前，_kv_len_gpu 为旧值" 的时序问题。但 `_reserved_blocks` 的递减（在 postprocess 的 `_release` 中重置为 0，L422）仅适用于 B=1 单序列。如果未来扩展到 B>1，多序列并发 prefill/decode 时 `_reserved_blocks=0` 的粗暴重置会导致空闲 block 计数错误。蓝图在 `_b1_scope` 中已声明 B=1 范围，但 `_reserved_blocks` 的实现（L421-422）没有标注 "B=1 only" 的 scope 限制，容易在扩展时遗漏。

### OW-5: nano-vllm ModelRunner `run_model()` 使用 `use_cache=False` 强制模式

- **JSON Path**: `framework_layer.components[3]._nano_vllm_override` (L181)
- **风险**: 蓝图说 "删除 nano-vllm 的 model(input_ids, use_cache=False, return_dict=True) 调用模式"。但 nano-vllm 的 `run_model()` (L197-213) 实际上没有显式传 `use_cache`——它调用 `self.model(input_ids, positions)` 然后是 `self.model.compute_logits(...)`。`use_cache` 是 HF 默认行为。重构者可能不理解 "use_cache=False 模式" 具体指什么（实际上是指 HuggingFace 模型的 `past_key_values` 机制），而在 TP Runner 中错误地保留了类似语义。

---

## 🟢 Reconstructability Score (重构可行性判决)

### 得分: 62%

### Top 3 阻断因素（阻碍达到 100%）:

1. **`max_blocks_for_model` 未定义 (FG-1)** — 这是 scheduler pseudocode 中的硬错误。不修复它，Scheduler 的 overlength rejection 逻辑完全无法实现。占阻断权重的 40%。

2. **Config → QwenTPConfig → QwenForCausalLMTP 构造链路不完整 (FG-3)** — 蓝图给出了类层级和字段，但没有给出组装代码。在源码销毁假设下，重构者必须"猜"出 config.json → dataclass → model 的完整调用链。一个字段类型错误（如 `num_heads` 是全量值 vs per-rank 值）会导致整个模型 shape 错误。占阻断权重的 35%。

3. **kernel_replacement_plan.md §9.2 与蓝图的跨层 weight 矛盾 (OW-2)** — 这是最具体的 "必定抄错" 风险点。ref_doc 是重构者的主要编码参考（比 JSON 更易读），而它的描述直接与正确实现相反。虽然蓝图标注了矛盾，但标注在 JSON 深层嵌套中，发现概率低。占阻断权重的 25%。

### 修补后可达分数: 85-90%

修补 FG-1 和 FG-3 即可消解大部分不确定性。OW-2 在 ref_doc 中加勘误注释后也可消除。剩余分数损失主要来自：
- nano-vllm 与蓝图之间的 7 处 override note 需要逐一人工核对，没有自动化验证
- TP=4 CUDA Graph 路径（Stage D）被显式标记为未实现，不在本次审计范围但影响完整度
- B>1 和 Continuous Batching 场景的 glue logic 仅在 `_b1_scope` 中被提及为"不在范围"

### 亮点

- `kernel_replacement_plan.md` §九 是本次审计中**最完整的参考资产**，7 个 kernel 的签名、参数、约束、import 路径均有精确表格。即使源码销毁，仅凭此节即可正确调用所有 vLLM kernel。
- QwenAttentionTP 的 `forward_decode()` 完整方法体 (L835-878) 是**可直接抄入代码的权威参考**，包含了 reshape 链、cache write、flash_attn_with_kvcache 调用的完整顺序。
- AGENT_SKILL.md 的集成接口陷阱 (G 节) 提供了 12 条来自物理 tracing 验证的具体约束，构成了有效的"防抄错检查清单"。
- 蓝图的 `_nano_vllm_override` 和 `_nano_vllm_per_function` 字段体现了对参考代码的深度理解，DELETE/PARTIAL 标记精确到方法级。
