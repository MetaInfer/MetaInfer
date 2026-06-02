# 蓝图重构盲测漏洞报告 v9

**审计对象**: `inference_blueprint.json` v2.3.0 + `AGENT_SKILL.md`
**审计范围**: 排除 deepseek 章节，聚焦框架层 + Qwen3 TP=4 重构可行性
**审计日期**: 2026-05-27
**审计身份**: 独立第三方系统级审计官（Zero-Shot Reconstructability Auditor）

---

## 🔴 Fatal Gaps (致命脑补漏洞项)

### FG-1: `QwenAttentionTP.forward_decode()` 完整方法体缺失

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern`
- **致命理由**: decode 路径是推理主循环的核心热路径。蓝图将关键实现信息碎片化散落在至少 5 个不同节点中（`decode_kv_write`、`decode_attention`、`qkv_projection`、`rotary_embedding`、`decode_forward_pattern.unified_signature`），但**没有任何一处给出从 qkv_proj → split Q/K/V → Q/K norm → rotary → KV cache index_copy_ 写入 → flash_attn_with_kvcache → o_proj → all_reduce 的完整串联伪代码**。重构者需要自行从多个碎片中拼凑调用顺序和张量 reshape 链，其中 QKVColumnParallelLinear 输出如何 split 为 Q、K、V 三个不同 shape 的张量（Q: `[1, num_heads/tp, head_dim]`, K/V: `[1, num_kv_heads_local, head_dim]`），以及 K 从 `[1, num_kv_heads, head_dim]` reshape 为 `[1, num_kv_heads, head_dim]` 写入 cache flat view 的具体维度操作，均无伪代码。**必定写错**。

### FG-2: `VocabParallelEmbedding.forward()` 和 `ParallelLMHead.forward()` 无伪代码

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_embedding_and_lm_head`
- **致命理由**: 蓝图给出了 input/output shape（如 `local_embedding: [B,T,hidden_size]`，`output_after_all_reduce: [B,T,hidden_size]`），但**完全没有 forward() 的实现伪代码**。VocabParallelEmbedding 的核心逻辑——输入 token ids 中超出 local vocab 范围的 token 需要 mask 为 0（或 pad_token_id），以及何时调用 all_reduce_sum——没有任何说明。ParallelLMHead 的 all_gather 操作（`output_logits_gather: [B,T,vocab_size]`）仅在 shape 层面提及，实际 gather 维度、是否使用 `dist.all_gather` 还是手动拼接均未指定。**盲目实现会导致 token 映射错误或通信死锁**。

### FG-3: QKVColumnParallelLinear/MergedColumnParallelLinear forward() 的 split 逻辑无伪代码

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers` + `qwen3_kernel_contracts.qkv_merged_projection`
- **致命理由**: 蓝图说明 QKVColumnParallelLinear 权重为 `[q_size + 2*kv_size, hidden_size]`，输出需 `.split([q_size, kv_size, kv_size], dim=-1)`。但 ColumnParallelLinear 基类的 forward() 行为（是否做 all_reduce、是否 gather 输出）未定义。QKVColumnParallelLinear 重写 forward() 时，是先调用基类 `F.linear(x, weight)` 得到 `[B, T, q_size+2*kv_size]` 再 split，还是另有 fused 实现？split 后 Q 需要 reshape 为 `[B, num_heads/tp, head_dim]`，K/V 为 `[B, num_kv_heads_local, head_dim]`——这些 reshape 操作的具体维度计算（涉及 TP 除法、GQA 的 num_kv_heads 与 num_heads 关系）是**极易出错的算术盲区**。蓝图没有一行伪代码覆盖。

### FG-4: `load_weights()` 完整编排逻辑缺失

- **JSON Path**: `model_layer.lazy_loader_synthesis_rules.qwen_dense_loader` + `qwen_hf_key_mapping`
- **致命理由**: HF key mapping 表格非常详尽（如 `q_proj.weight → qkv_proj [0:q_size]`），但**实际的 load_weights() 方法体完全没有伪代码**。这包括：遍历 `model.safetensors.index.json` 的 weight_map、按层号分组、对 QKV 三个 key 做 ColumnParallel 切片后 `torch.cat` 拼接、对 gate/up 同理拼接、对 o_proj/down_proj 做 RowParallel 切片、以及 `double_shard_guard` 的 shape 相等短路判断。如果重构者未正确实现 cat 顺序（Q-K-V 顺序错误会导致静默数值错误），或未正确处理 safetensors 跨文件分片（`get_slice`），**权重加载必定静默失败**——不会报错但输出乱码。

### FG-5: `QwenForCausalLMTP.forward()` 完整方法体缺失

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces`
- **致命理由**: 顶层模型的 forward() 是连接所有子模块的编排器。蓝图给出了 `prefill_forward_pattern.full_dataflow`（7 步文本描述）和 `decode_forward_pattern.entry`（一句话），但**完整的 forward() 方法体伪代码缺失**。关键盲区包括：
  - `past_key_values is None` 判断分发到 prefill/decode 分支的具体代码
  - prefill 中 `past_key_values=None` 时各层 KV cache 的首次分配时机
  - decode 中如何将 `past_key_values`（kv_lens 列表）传入各层
  - 最后一层 `model.norm(hidden_states, residual)` 的调用方式（residual 来源是第 36 层的 return 值）
  - `.item()` 读取 kv_lens 的精确位置（必须在非编译 forward() 中）
  - 多序列 prefill 时 block_table 如何从 Runner 层传递到 QwenAttentionTP 内部

---

## 🟡 Override Warnings (重载失效警告项)

### OW-1: Scheduler 重载 nano-vllm — 双轨 block_size 修改点不精确

- **JSON Path**: `framework_layer.components[0]._dual_track_note` + `framework_layer.data_flow_contracts.scheduler_tp_runner_bridge`
- **警告**: nano-vllm 的 scheduler.py（84 行）内部使用 `block_size` 做 `max_num_batched_tokens` 计算和 `required_blocks` 估算。蓝图给出的注入方案是 `LLMEngine.__init__` 中设置 `scheduler._block_size = 256`。但 nano-vllm scheduler 可能在其他方法中也有 block_size 依赖（如 `can_allocate`、`can_append_one_more`），蓝图仅指明了一处注入点。**重构者可能只改一处而漏掉其他调用点**，导致 `can_allocate` 使用 256 而某处内部仍用 16 的不一致状态。

### OW-2: BlockManager 在 TP 路径下降级 — 未逐方法标注

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_tp_runner_bridge.block_manager_role_in_tp_path`
- **警告**: 蓝图规定 TP 路径下 `allocate()/free()` 降级为 no-op，但 BlockManager 的 ref_code（nano-vllm block_manager.py, 112 行）还包含 `compute_hash`、`may_append`、`get_num_free_blocks` 等方法。**重构者在抄入 nano-vllm 代码后，需要自行判断哪些方法保留原逻辑、哪些需要修改**。例如 `get_num_free_blocks()` 在 TP 路径下是从 `runner.get_num_free_blocks()` 获取，而非 BlockManager 自身——但这个决策点在 LLMEngine.step() 中，BlockManager 本身的 `get_num_free_blocks()` 是否需要保留为 fallback 未说明。

### OW-3: KVMemoryPool GPU placeholder — 与 nano-vllm 的差异未显式警告

- **JSON Path**: `framework_layer.components[1].tp_path_note` + `_responsibility_boundary`
- **警告**: nano-vllm 的 `model_runner.py`（258 行）可能在 `__init__` 中调用 `KVMemoryPool` 创建 GPU KV placeholder。蓝图说明了 TP 路径不使用这些 placeholder（实际 KV cache 由 QwenAttentionTP 内部创建），但**没有逐方法标注 KVMemoryPool 中哪些方法在 TP 路径下不应被调用**。重构者可能因为惯性调用了 `KVMemoryPool` 的 GPU placeholder 创建逻辑，导致显存被两份 KV cache 占用。

### OW-4: nano-vllm ModelRunner 的 use_cache/return_dict — 缺少禁用指引

- **JSON Path**: `framework_layer.components[3].tp_runner_actual_flow._note`
- **警告**: nano-vllm 的 ModelRunner 调用 `model(input_ids=ids, use_cache=False, return_dict=True)`。蓝图在 `runner_decode_tensors` 中标注了 "仅 RealModelRunner 使用"，且 `tp_runner_actual_flow._note` 声明 TP Runner 不使用此路径。但**重构者在抄入 nano-vllm 代码时，可能将 `use_cache=False` 的 HF 调用模式错误地套用到 TP Runner**。蓝图未在 "与参考代码的差异" 中显式列出"删除 use_cache=False 约束"或"替换为 forward_decode 调用"。

### OW-5: Sampler TP 协议 — nano-vllm 单卡采样器的 patch 点未标注

- **JSON Path**: `framework_layer.components[4].tp_sampling_protocol`
- **警告**: nano-vllm 的 `sampler.py`（12 行）是单卡实现（直接 `torch.multinomial` 或 `argmax`）。蓝图给出了 TP 协议的完整伪代码（rank 0 采样 + broadcast），但**没有标注"nano-vllm sampler 的哪些行需要被 if world_size>1 分支包裹"**。重构者可能把 broadcast 逻辑放在错误的位置（例如放在 sampler 内部而非 runner 的 `_sample` 方法中），或者遗漏了 `dist.broadcast` 的 src=0 参数。

### OW-6: RMSNorm kernel 替换 — nano-vllm 原始实现 vs vLLM kernel 的切换点

- **JSON Path**: `model_layer.architecture_knowledge_base.global_primitives_constraints.rmsnorm_precision_law`
- **警告**: nano-vllm 参考工程大概率使用 PyTorch 原生 RMSNorm。蓝图要求所有 RMSNorm 使用 vLLM CUDA kernel（`rms_norm(out, x.contiguous(), weight, eps)`），但**未在框架组件或 Qwen3 适配章节中显式标注"RMSNorm 类需整体替换为 vLLM kernel wrapper，不能保留 nano-vllm 原始实现"**。重构者可能写了一个混合体——RMSNorm 类框架来自 nano-vllm，但 forward 中手工调用 vLLM kernel，导致 `out` 预分配方式和 `contiguous` 约束不一致。

---

## 🟢 Reconstructability Score (重构可行性判决)

### 量化分数: **52%**

以当前掌握信息（JSON + AGENT_SKILL.md + RAG 可检索 ref_code），写出**一次跑通**的 Qwen3 TP=4 引擎的综合把握约 52%。

#### Top 3 阻断因素（阻碍达到 100%）:

1. **forward_decode() 完整伪代码缺失（阻断力: 35%）**: 这是 decode 主循环的核心热路径。蓝图将 QKV split、norm、rotary、KV cache write、flash_attn_with_kvcache、o_proj 的实现分散在 5+ 个节点中，缺少统一的伪代码。重构者必须在没有参考实现的情况下自行缝合——而 decode 路径的每个张量 reshape（特别是 GQA 下 num_kv_heads 与 num_heads 不同的维度处理）都会直接导致静默数值错误或 CUDA error。

2. **load_weights() 编排逻辑缺失（阻断力: 30%）**: HF key mapping 表详尽，但从 "知道怎么映射" 到 "写出遍历 safetensors 索引、跨文件 get_slice、拼接 QKV/gate_up、处理 double_shard_guard 的完整 load_weights" 之间有巨大鸿沟。权重加载错误通常是**静默的**——张量 shape 对但值错位，只会在最终输出中表现为乱码，极难定位。

3. **VocabParallelEmbedding/ParallelLMHead forward() 缺失 + QKVColumnParallelLinear split 逻辑缺失（阻断力: 20%）**: 这两个 TP 基础算子是模型的首尾关键路径。Embedding 的 token mask 逻辑和 LM Head 的 all_gather 通信模式没有伪代码，重构者只能猜测。QKVColumnParallelLinear 的 split→reshape 链涉及 TP 除法算术——在 `num_kv_heads=8, tp=4, num_kv_heads_local=2` 的情况下，Q 的 head 数是 8，K/V 的 head 数是 2，split 后的 reshape 需要正确理解 GQA 的 head mapping 关系。

#### 蓝图做得好的部分（得分项）:

- Qwen3-8B 完整物理维度（含 TP=4 per-rank 计算值）: 满分
- Paged KV cache 格式、slot_mapping 算法、reshape 链伪代码: 优秀
- Prefill forward 伪代码 (`layer_forward_pseudocode`): 较完整
- Class hierarchy + 精确属性命名: 优秀（权重加载的 HF key 映射直接可用）
- 16 条 failure_mode_library (symptom/check/fix): 极有价值
- CustomAR 初始化伪代码: 足够完整
- flash_attn_with_kvcache custom_op 注册模板: 精确可用
- fused_add_rms_norm residual_chain_pseudocode: 清晰完整
- Scheduler-TP Runner bridge 契约: 架构接口清晰

#### 关键发现:

蓝图在**静态知识表达**（维度、属性、约束、shape）层面质量很高，但在**动态执行流**（完整方法体伪代码、跨模块调用链）层面存在系统性的碎片化问题。信息是"全的"但需要重构者在脑中做大量缝合——对于 LLM Agent 而言这种缝合极易引入幻觉错误，对于人类工程师而言则需要反复 cross-reference 多个节点才能写出一行正确的代码。
