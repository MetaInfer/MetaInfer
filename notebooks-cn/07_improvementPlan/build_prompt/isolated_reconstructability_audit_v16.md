# 蓝图重构盲测漏洞报告 — `inference_blueprint.json` v2.3.0

**审计日期**: 2026-05-27
**审计对象**: Qwen3-8B nocompile B=1 TP=4 推理引擎（排除 DeepSeek 章节）
**审计方法**: 四维度盲测重建可行性评估 + 3 份 ref_doc 抽样交叉验证

---

## 维度四：扩展存储完备性核验（抽检结果）

已实际打开的 ref_doc 文件：
1. `kernel_replacement_plan.md`（1511 行）— 验证了 §九 的 7 个 kernel 调用契约
2. `improvement_plan.md`（905 行）— 验证了 §P0、§P2、§P3-FA 的改动记录
3. `qwen3_effective_changes.md`（137 行）— 验证了 #8、#9、#10 改动点的追溯信息
4. `ref_projects/nano-vllm/nanovllm/engine/scheduler.py`（85 行）— 验证了 override 指令的锚点存在

**交叉验证结论**: 3/3 份 ref_doc 中蓝图引用的知识点均被确认存在。`kernel_replacement_plan.md` §九 的 7 个 kernel 调用签名（行 1429-1511）与蓝图 `qwen3_kernel_contracts` 中的引用一致。`improvement_plan.md` §P0（行 201-259）完整记录了增量 KV Cache 的实现方式、踩坑经验和数值验证结果。`qwen3_effective_changes.md` 的 10 个改动点追溯链（行 124-136）与蓝图的 kernel/framework 分层吻合。

---

## 🔴 Fatal Gaps（致命脑补漏洞项）

### FG-1: `fused_add_rms_norm` 跨层 weight 的信息断裂（文档间自相矛盾）

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint._kernel_replacement_plan_conflict`
- **矛盾链条**:
  - `kernel_replacement_plan.md` §9.2（行 1458）明确要求：post-mlp 调用时 weight 必须是**下一层**的 `input_layernorm.weight`
  - `inference_blueprint.json`（行 1132-1134）WARNING：meta-infer 实际使用**本层** `post_attention_layernorm.weight`
  - `qwen3_effective_changes.md` Stage 1 "踩坑 2"（行 17）仍描述为"需要**下一层**的 `input_layernorm.weight`"——与蓝图矛盾
- **致死原因**: 三份文档对同一关键约束给出两个互斥答案。若重构者先读了 `kernel_replacement_plan.md`（最自然的顺序——先研究 kernel 调用签名），会直接抄入跨层 weight 方案；随后即使见到蓝图的 WARNING，也无法判断哪个是正确的。这个冲突只能靠实际运行—报错—debug 发现。
- **等级**: 🔴 Fatal——导致 post_mlp 的 residual 链全部错误，输出不可读。

### FG-2: CustomAR `init_custom_ar` 双套 IPC buffer 的分配公式缺口

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce.constraint.init_state_machine`
- **缺失内容**:
  - 蓝图的 meta_ptrs 和 buf_ptrs 是两套独立的 IPC buffer 集合（行 1228-1232）。但 `init_state_machine` 伪代码（行 1174-1195）**仅描述了 meta_ptrs 的分配和 exchange 流程**，未包含 buf_ptrs/register_buffer 的完整调用序列。
  - `ops.allocate_shared_buffer_and_handle(max_size)` 返回 `(raw_ptr, ipc_handle)`，但 `max_size` 的精确计算公式未给出（仅说"16 MB"足够）。重构者需要理解 `max_size >= max(per_layer_allreduce_size)` 才能正确设定，但 `per_layer_allreduce_size`（TP=4 下 RowParallel 输出 `[B,T,4096]` bf16 = 8KB）的推导链散布在文档各处。
  - buf_ptrs 的 IPC handle exchange 使用了 `dist.broadcast_object_list`（行 1222），而 meta_ptrs 使用了 `dist.all_gather_object`（行 1185）——两种不同的 IPC exchange 模式。蓝图只给了 buf_ptrs 的 broadcast_object_list 伪代码一句话，缺乏完整的 gloo init + exchange + open_handles + register_buffer 四步序列。
- **致死原因**: CustomAR 初始化是所有 RowParallel 通信的基础设施。如果 `register_buffer` 步骤遗漏或参数错误，all_reduce 在运行时访问无效的远程 buffer 指针 → CUDA illegal memory access → 整个 TP 推理崩溃。重构者只能凭经验脑补 buf_ptrs 的完整初始化序列。
- **等级**: 🔴 Fatal——核心通信基础设施的初始化步骤不完整。

### FG-3: paged KV cache 的 `max_seq_len` 语义缺失

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.model_forward_pseudocode`（行 1074-1098）
- **缺失内容**: `QwenForCausalLMTP.forward(input_ids, past_key_values=None, position_offset=0, max_seq_len=528)` 的 `max_seq_len` 默认值是 528。但没有任何文档或蓝图解释 528 这个值的来源和含义。Qwen3-8B 的 `max_position_embeddings=32768`，`max_blocks=128`。528 对应什么？
- **致死原因**: KV cache 的大小 `num_blocks = (max_seq_len + 255) // 256` 直接依赖此参数。如果重构者错误地将 `max_seq_len=config.max_position_embeddings=32768` 传入，会分配 128 个 block 的 KV cache (`128*256*num_kv_heads*head_dim*2*bfloat16*36层 ≈ 4.7GB per rank`)，而非按 528 分配的 ~80MB。错误的 `max_seq_len` 不会导致 crash（有足够显存），但会导致 KV cache 大小与预期不符，进而影响 BlockManager num_free_blocks 计算，Scheduler 的调度决策出错。
- **等级**: 🔴 Fatal——核心架构参数 magic number 无溯源。

### FG-4: `vocab_size` 不能被 `tp_size` 整除时的边界处理不完整

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_embedding_and_lm_head.vocab_parallel_embedding.forward_pseudocode`（行 706-719）
- **缺失内容**: Qwen3-8B `vocab_size=151936`，`151936 // 4 = 37984`，`151936 % 4 = 0`——恰好整除。但蓝图的注释说"vocab_size 不可被 tp_size 整除时: 最后一个 rank 的 local_vocab_size = vocab_size - (tp-1)*(vocab_size//tp)。pad_token_id 映射到 0"。这个描述：
  1. 未说明 pad_token_id 是什么（config.json 中有吗？默认多少？）
  2. 未说明 mask 逻辑如何处理 pad_token 的边界 case——如果 input_ids 里出现了 pad_token，mask 和 safe_ids 的逻辑是否仍然正确？
  3. `ParallelLMHead`（lm_head）的 output `all_gather` 在 vocab_size 不整除时也需要对应的 gather 边界处理，蓝图完全未提。
- **致死原因**: 虽然 Qwen3-8B 恰好整除，但蓝图声明了覆盖 Qwen2ForCausalLM 等系列，这些模型的 vocab_size 可能不整除。重构者若更换模型，LM head 的 `all_gather` 输出维度会出错。
- **等级**: 🔴 Fatal（跨模型场景）——当前 Qwen3-8B 恰好整除，但蓝图声明了更广的覆盖范围。

## 🟡 Override Warnings（重载失效警告项）

### OW-1: nano-vllm 行号依赖与蓝图自包含的脱节

- **影响节点**: Scheduler, BlockManager, KVMemoryPool, ModelRunner, Sampler
- **详情**: 蓝图每个组件的 `_nano_vllm_override` 中包含精确行号引用（如 `nano-vllm scheduler.py line 52-57`）。经验证，nano-vllm 源码确实存在且行号准确。但：
  - 这些行号是瞬时快照——nano-vllm 若被修改，行号即失效
  - 蓝图自己的 `schedule_complete_method` 和 `postprocess_complete_method` 伪代码（行 344-430）已经提供了独立、完整的实现，实际上**不需要** nano-vllm 作为基础
  - Sampler 的 TP 改写依赖 nano-vllm sampler.py 做基础实现，但蓝图的 `tp_sampling_protocol.pseudocode`（行 206-213）已提供完整的 TP 采样逻辑
- **风险**: 重构者可能把 override 行号当唯一真理，而忽略蓝图已有的完整伪代码，或者反过来——在两个信息源之间反复横跳。
- **等级**: 🟡 Warning——混淆风险但非阻塞。

### OW-2: improvement_plan.md 的 Past-key-values 方案 A 与实际实现的冲突

- **影响节点**: `ModelRunner.tp_runner_actual_flow`
- **详情**: `improvement_plan.md` §P0 方案 A（行 207-223）描述的是基于 HF `past_key_values` 的 KV cache 方案（`use_cache=True`，缓存 `past_key_values` 到 `seq.past_key_values`）。但蓝图 `_nano_vllm_per_function.allocate_kv_cache()`（行 183）明确说"KV 由 QwenAttentionTP 自管"。P0 文档和蓝图描述的是两套不同的 KV cache 机制——P0 是基于 HF 的 contiguously stored，蓝图是基于 paged KV cache（block_size=256）。
- **风险**: 重构者若遵循 improvement_plan.md 的 P0 记录，会实现错误的 KV cache 格式。
- **等级**: 🟡 Warning——文档版本进化但旧方案记录未标记为废弃。

### OW-3: Scheduler TP/HF 双轨下 BlockManager 降级的安全边界缺失

- **影响节点**: `framework_layer.components.BlockManager._nano_vllm_override`
- **详情**: 蓝图规定 TP 路径下 BlockManager "allocate/free 改为 no-op（仅 count+=1/count-=1）"（行 137）。但未明确：
  1. 降级后的 `allocate()` 是否需要返回值给 Scheduler（Scheduler 的 `can_allocate()` 依赖 `num_free_blocks` 但不依赖 allocate 的返回值）
  2. 如果 LLMEngine 在某条路径中误调用了 `BlockManager.allocate()`（TP 路径下），no-op 行为是 silent succeed 还是 assert 抛异常
  3. `BlockManager` 的 `allocate()` 在 TP 路径下仍需返回 `list[int]` 类型（api_spec 行 116 签名），no-op 返回什么——空 list？
- **风险**: 静默的 no-op 可能导致调度器在 TP 路径下行为正确但逻辑不可验证。
- **等级**: 🟡 Warning——不阻塞功能但有隐式契约风险。

### OW-4: `flash_attn_varlen_func` 的 cu_seqlens 注入路径缺少最后的 dtype 保证

- **影响节点**: `framework_layer.data_flow_contracts.flash_attention_integration_contract.prefill_path.cu_seqlens_construction`
- **详情**: 蓝图给出 cu_seqlens 构造伪代码（行 629-633），生成 `cu_seqlens_q = torch.zeros(len(batch)+1, dtype=torch.int32, device='cuda')`。但 `flash_attn_varlen_func` 要求 `cu_seqlens` 必须是 **int32**（非 int64），且 `max_seqlen` 必须是 Python int（非 0-dim tensor）。蓝图指明了 dtype=int32，但未明确 `max_seqlen` 必须是 Python int 而非 0-dim tensor（常见陷阱）。
- **风险**: 重构者可能将 `max_seqlen_q` 写成 `torch.tensor(max(...))` → flash_attn 静默接受但输出错误形状。
- **等级**: 🟡 Warning——常见陷阱未被标注。

### OW-5: `last_layer_note` 中第 36 层的特殊处理硬编码了层数

- **影响节点**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint.residual_chain_pseudocode.last_layer_note`（行 1132）
- **详情**: 蓝图说"最后一层（第36层）的 post_mlp 不调用 fused_add_rms_norm"。但 `num_hidden_layers` 应该从 `config.json` 动态读取（AGENT_SKILL.md 行 59-63 要求动态读取）。如果重构者用 `num_hidden_layers=36` 做硬编码判断，换了 32 层或 40 层的模型就会错误处理最后一层 norm。
- **风险**: 蓝图上下文明确是针对 Qwen3-8B（36 层），但 last_layer_note 的描述方式容易让人理解为硬编码 36。
- **等级**: 🟡 Warning——跨模型移植时需要意识到此处。

---

## 🟢 Reconstructability Score（重构可行性判决）

**量化分数: 78 / 100**

### 分数依据

蓝图 + ref_docs 提供了：
- 完整的 Qwen3-8B 物理维度参数（`qwen3_8b_model_dims`，行 1704-1724）：含全量值和 per-rank (TP=4) 值
- 完整的张量 shape 推导链（`data_flow_contracts` 覆盖所有关键接口）
- 完整的 Scheduler/LLMEngine 伪代码（行 344-430）——直接可转为 Python
- 完整的 QwenForCausalLMTP 类层级与 `__init__` 签名（`class_hierarchy`，行 1000-1072）
- QwenAttentionTP 的 decode 热路径 `forward_decode` 完整伪代码（行 843-886）：QKV 投影 → Q/K norm → RoPE → KV cache write（index_copy_）→ flash_attn_with_kvcache → o_proj 全链路
- Prefill 路径 `forward` 伪代码（行 910-955）：含 KV cache 懒初始化、cu_seqlens 自动构造、slot_mapping 构建
- 完整的 HF weight key mapping + load_weights 伪代码（行 1860-1916）：QKV 三段拼接和 gate/up 两段拼接的索引边界清晰
- 7 个 vLLM kernel 的调用契约（`kernel_replacement_plan.md` §九）：调用签名、dtype 约束、输入 shape、预分配要求精确到参数级别
- 可实际访问的 nano-vllm 参考代码和完整的 override 指令

Qwen3 TP 的知识覆盖度约 **90%**——前向推理的完整计算图、所有权重映射、所有张量 shape 变换链都是闭环的。

### Top 3 阻断因素（阻止达到 100%）

1. **`fused_add_rms_norm` 跨层 weight 文档矛盾**（FG-1）——必须通过实际运行时测试才能确定正确的 weight 引用。占 -8 分。
2. **CustomAR buf_ptrs/register_buffer 初始化序列不完整**（FG-2）——缺少完整可执行的四步初始化伪代码。占 -7 分。
3. **`max_seq_len=528` 的 magic number 无溯源和 `vocab_size` 不整除的边界处理欠缺**（FG-3、FG-4）——核心架构参数和跨模型边界缺失。占 -7 分。
