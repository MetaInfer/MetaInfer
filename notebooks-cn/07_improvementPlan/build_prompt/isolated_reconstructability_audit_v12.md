# 蓝图重构盲测漏洞报告

**审计对象**：`inference_blueprint.json` v2.3.0  
**审计范围**：排除 deepseek 章节，仅审计 Qwen3 Dense TP=4 路径  
**审计前提**：业务源码已销毁，仅持有 JSON + AGENT_SKILL.md + RAG 访问权

---

## 🔴 Fatal Gaps（致命脑补漏洞项）

### FG-1：prefill forward 参数传递自相矛盾

**JSON Path**：
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.prefill_forward_pattern.model_forward_pseudocode` (L877-901)
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.prefill_forward_pattern.cu_seqlens_injection_path` (L807-821)
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.prefill_forward_pattern.layer_forward_pseudocode` (L749-805)

**死锁理由**：三处给出了三个不同的接口约定。

`model_forward_pseudocode` 定义的签名是：

```python
def forward(self, input_ids, past_key_values=None, position_offset=0, max_seq_len=528)
```

调用层时为 `layer.forward(hidden_states, positions, layer_cache=None, max_seq_len=max_seq_len, residual=residual)`，**无 cu_seqlens 参数**。

但 `cu_seqlens_injection_path` 要求**多序列 prefill 时**：

```python
model.forward(input_ids, past_key_values=None, cu_seqlens_q=..., cu_seqlens_k=..., max_seqlen_prefill=...)
```

→ 逐层传递 `cu_seqlens_q/cu_seqlens_k/max_seqlen`。

而 `layer_forward_pseudocode` 定义的层签名又多出三个可选参数：

```python
def forward(self, hidden_states, positions, layer_cache, max_seq_len, residual=None,
            cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen=None)
```

**后果**：若按 `model_forward_pseudocode` 实现 QwenForCausalLMTP.forward()，当遇到多序列 prefill 时 cu_seqlens 无注入通道，flash_attn_varlen_func 将拿到错误的 cu_seqlens_q/k。单序列 prefill 尚可借自动构造 `[0, num_tokens]` 短路绕过，但 B>1 时必然报错或产生静默数值错误。

---

### FG-2：多序列 decode batch 扩展无实现规格

**JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern` (L665-728)  
**具体位置**：`tp_runner_actual_flow.decode_batch_mode` (L153)

**死锁理由**：当前只给出了 B=1 的逐序列 for-loop 方案，多序列扩展仅以一句话带过：

> "batch 需扩展所有 attention 层为 [B,...] 数据结构"

没有：

- `_block_table` 如何从 `[1, max_blocks]` 扩展到 `[B, max_blocks]` 的具体代码
- `_kv_len_gpu` 如何从 `[1]` 扩展到 `[B]` 的具体代码
- flash_attn_with_kvcache 在多 block_table 下的调用方式变化
- QwenForCausalLMTP.forward() 的 decode 分支如何传入 batch 维度的 block_table
- 多序列时 kv_lens 的读取逻辑（`_kv_len_gpu[0].item()` vs `_kv_len_gpu[i].item()`）

**后果**：若要在 B>1 场景下从零实现，必须在完全不理解物理布局的前提下脑补 batch dim 的维度和语义。按当前蓝图，有约 50% 概率写错 broadcast/slice 语义，导致 KV cache 写串扰（静默正确性 bug，难以检测）。

---

### FG-3：CustomAR 初始化的完整张量分配与 register 逻辑断层

**JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce.constraint.init_state_machine` (L975-1017)

**死锁理由**：JSON 给出了伪代码骨架（gloo group 创建 → allocate_shared_buffer_and_handle → all_gather_object → open_mem_handle → init_custom_ar → register_buffer），但存在三个裂缝：

1. **注册缓冲区大小选择**：`init_state_machine.workspace_size` 声明 16 MB，但 `register_buffer` 需要传递一个**预先分配的张量列表** `buf_pointers`，该列表的构造方式（每个 buffer 的 shape/dtype/分配代码）未给出。`custom_all_reduce` wrapper 的 `reg_buffer` 参数来历不明——是 register_buffer 的返回值还是 buffer 列表的索引？

2. **kernel_replacement_plan.md 的向后引用**：Stage 4 组装说明 (L496) 明确写道 *"具体组装代码在 Stage 4 实施时从 vLLM `parallel_state.py` 和 `custom_all_reduce.py` 提取"*，但根据核心假设，vLLM **源码**已随 meta-infer 一起销毁，仅剩 installed package。这些 `.py` 文件在 pip 安装中通常不保留。你只能靠 `dir(vllm._custom_ops)` 反推函数签名，然后用试错法猜测参数语义。

3. **meta_size() 与 buffer 的关系**：`init_state_machine.vllm_imports` 列出了 `ops.meta_size()` 但未说明其用途——是 init_custom_ar 的返回值还是元数据缓冲区的预分配大小？这个信息断层会导致 init 调用传递错误的参数。

**后果**：即使 vLLM 已安装，CustomAR 的完整初始化代码有约 30% 概率因参数顺序/大小错误而 SIGSEGV（访问非法 GPU 地址）。

---

### FG-4：VocabParallelEmbedding 的不整除边界处理只有注释无代码

**JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_embedding_and_lm_head.vocab_parallel_embedding` (L543-562)

**死锁理由**：forward_pseudocode 末尾注释提到 *"vocab_size 不可被 tp_size 整除时: 最后一个 rank 的 local_vocab_size = vocab_size - (tp-1)*(vocab_size//tp)。pad_token_id 映射到 0。mask 逻辑不变。"*

但 Qwen3-8B 的 vocab_size=151936，tp_size=4，**151936 / 4 = 37984，恰好整除**——所以这个边界路径在当前模型上永远不会执行。然而伪代码**未给出整除性检测逻辑**，也未说明 `local_vocab_size` 的计算代码应放在 `__init__` 还是 `forward` 中。负责重构的人若按正向逻辑*恰好写对*（因为整除），但在未来切换到 vocab_size 不整除模型时，这段未实现的边界处理会引入难排查的词表越界。

**后果**：当前可工作，但留有技术债务。在"盲测"假设下，重构者会完全不知有此边界情况，导致模型切换时出现 CUDA out-of-bounds 访问。

---

## 🟡 Override Warnings（重载失效警告项）

### OW-1：nano-vllm `allocate_kv_cache` 删除指令缺少精确行号

**JSON Path**：`framework_layer.components.KVMemoryPool._nano_vllm_override` (L83)

**描述**：指示删除 nano-vllm `model_runner.py` 的 `allocate_kv_cache` 模式，但 `model_runner.py` 共 258 行，`allocate_kv_cache` 是一个方法模式而非单一函数。重构者需要自行搜索并判断哪些代码属于该模式。若误删 `prepare_prefill` 的 `cu_seqlens` 构造（L129-150，蓝图声明 "可参考"），将同时丢失有用的 ragged batch 参考代码。

**建议**：提供 `allocate_kv_cache` 在 `model_runner.py` 中的精确行范围。

---

### OW-2：flash_attn_varlen_func 的 K/V 来源约定在多处重复但缺乏 `hard_rule` 锁定

**JSON Path**：
- `prefill_kv_write.integrated_timeline` (L413-424)：QKV 投影 → flash_attn(Q,K,V) → index_copy_ 写入
- `prefill_forward_pattern.key_differences_vs_decode` (L743-748)：KV cache: allocate+write vs read+append
- `flash_attention_integration_contract.prefill_path.kv_source_correction` (L491)：再次强调 K/V 来自投影

**警告理由**：蓝图三处均坚称 prefill 的 flash_attn_varlen_func 使用投影 K/V（非 cache K/V）。这个设计在数学上是正确的。但 `layer_forward_pseudocode` (L766-805) 中，步骤 7 在 attention 之后才写入 cache——这意味着如果你调用 `flash_attn_varlen_func(q, k_proj, v_proj, ...)` 然后在 attention 后写入 cache，cache 中的 K/V 与投影 K/V 是**数值相同的**（RoPE 后的值）。这没有问题。

但某些优化路径可能希望 attention 从 cache 读取（验证时可以直接对比投影与 cache 的一致性）。蓝图未给出这种"cache 写后读"替代模式的评价，如果重构者自行选择从 cache 读取（看起来更自然——"cache 里已经有 KV 了为什么不读？"），会产生数值等价但性能不同的路径，在 CUDA Graph 场景下可能导致 cache 流水线行为不同。

**建议**：增加 "禁止从 cache 读取用于 prefill attention" 的 hard_rule，或论证"从 cache 读取亦可"的等价性证明。

---

### OW-3：nano-vllm `can_append`/`may_append` 改写逻辑依赖隐式 block_size 语义

**JSON Path**：`framework_layer.components.Scheduler._nano_vllm_override` (L64)

**警告理由**：蓝图书 *"nano-vllm scheduler.py line 52 can_append(seq) 和 line 60 may_append(seq) 需改为 can_append_one_more (num_free_blocks>=1) 逻辑"*。但 nano-vllm 原 `can_append` 的逻辑是 `len(self.free_block_ids) >= (len(seq) % self.block_size == 1)`，它仅在**块边界**（token 填满一个 block 时）检查是否需要新 block。

改名为 `can_append_one_more` 并简化为 `num_free_blocks >= 1` 后，语义从"是否需要分配新 block"变为"是否有空闲 block"。但**调度器不实际分配 block**（TP 路径下 BlockManager 降级为 no-op），所以调度器需要的是"是否需要新 block"还是"是否有空闲 block"？蓝图未明确说明这个语义变化的调用方影响。

**后果**：重构者可能直接抄 nano-vllm 的 `can_append` + `may_append` 逻辑但不理解 block_size=256 vs 16 的差异，或者在改写时遗漏 TP 路径下 `may_append` 应变为 no-op 的事实。

---

### OW-4：nano-vllm 使用 xxhash，蓝图 api_spec 使用 Python builtin hash——迁移影响未声明

**JSON Path**：
- `framework_layer.components.BlockManager.api_spec.compute_hash` (L98-103)
- `ref_projects/nano-vllm/nanovllm/engine/block_manager.py:36-41`

**警告理由**：nano-vllm 的 BlockManager 使用 `xxhash.xxh64` 计算 hash，而 blueprint 的 `api_spec.compute_hash` 推荐使用 Python builtin `hash(tuple(token_ids))`。两者不是同一种 hash 算法，即使输入相同 token_ids 也会产生不同 hash 值。若重构者直接抄 nano-vllm 的 hash 计算（xxhash），同时按蓝图 api_spec 构建 hash_to_block_id 映射表，内部不一致不会报错——只是 prefix caching 永远 miss（不同 hash 算法）。这在 B=1 场景下无影响（prefix caching 本来就不生效），但浪费了 prefix caching 的代码路径。

**建议**：明确声明 "TP 路径禁用 prefix caching" 时是否需要完全删除 hash 相关代码。

---

### OW-5：vLLM kernel 的 `.contiguous()` 要求散布在契约而非调用点

**JSON Path**：`qwen3_kernel_contracts.rms_norm.inline_signature` (L919)

**警告理由**：蓝图书 *"rms_norm(out, x.contiguous(), weight, eps)"*，其中 `x.contiguous()` 是调用方的责任。但 caller（如 QwenAttentionTP.forward_decode）中有多达 5-6 处调用 RMSNorm（Q/K norm 各一次 × 36 层 + input_layernorm + post_attention_layernorm），如果任何一处忘记 `.contiguous()` 而传入 view，vLLM CUDA kernel 会静默输出错误值（不会报错，因为 kernel 不检查 stride）。蓝图把 `.contiguous()` 要求写在 kernel 契约中而非每个调用点的伪代码中，增加了调用方的遗漏风险。

---

## 🟢 Reconstructability Score（重构可行性判决）

### 量化分数：**72/100**

### Top 3 阻断因素（阻止达到 100%）：

1. **prefill forward 参数传递自相矛盾（FG-1）** — 阻碍权重 12%。多序列 prefill 时，cu_seqlens 的注入通道在两个权威来源（`model_forward_pseudocode` vs `cu_seqlens_injection_path`）中描述不一致，你必须自行"猜"哪个版本是权威的，以及如何修改 QwenForCausalLMTP.forward() 签名来容纳多出来的参数。

2. **CustomAR init 的完整张量分配与 register 逻辑断层（FG-3）** — 阻碍权重 9%。JSON 的 IPC exchange 伪代码给出了步骤序列，但 `register_buffer` 的具体调用（buffer 个数、每个 buffer 的 shape/stride/分配方式）缺失，而 `kernel_replacement_plan.md` §四 又将此责任推给了已销毁的 vLLM 源码。你可以在安装的 vLLM wheel 中通过 `inspect` 反推 API，但这不保证一次正确。

3. **多序列 decode 的 batch 扩展无规格（FG-2）** — 阻碍权重 7%。当前 B=1 可一次跑通，但蓝图对 B>1 只有一句话提示，没有 _block_table/_kv_len_gpu 的 batch 维度扩展代码。即使 B=1 可以工作，一旦扩展到连续批处理，你必须从零脑补所有张量维度的 batch 扩展语义。

---

### 可扣除但非阻断的缺陷：

- RoPE 的 cos/sin cache 创建公式由 `kernel_replacement_plan.md` §0 (make_cos_sin_cache) 补齐，JSON 自身仅描述格式。若该 md 文件丢失，你无法从 JSON 中推导 `inv_freq` 公式（+3%）
- nano-vllm 部分 override 依赖行号引用（如 scheduler.py L52-57），若 nano-vllm 版本升级导致行号漂移，override 将失效（+2%）
- `_nano_vllm_per_function` 中 `prepare_prefill()` 标记为 "PARTIAL" 但未给出"保留哪些/删除哪些"的精确边界（+2%）
- Qwen3-8B 维度已在 `qwen3_8b_model_dims` 明确列出（hidden_size=4096, num_layers=36, etc.），**此点满分，无缺口**

---

### 可工作范围正面评估：

蓝图的以下方面表现良好，可以在盲测下直接实现：

- **paged KV cache contract**：slot_mapping 算法、index_copy_ 写入链、block_table 初始化、block_size=256 硬约束——均给出可执行伪代码
- **TP 线性层签名**：ColumnParallel/RowParallel/QKVColumnParallel/MergedColumnParallel 的 forward 逻辑均有完整伪代码
- **decode 单序列热路径**：forward_decode 完整方法体（L684-728）可以直接翻译为 Python
- **TP 采样协议**：rank 0 采样 + broadcast 的伪代码（L196-203）可直接抄写
- **qwen_hf_key_mapping**：HF key → 自定义属性映射表完整，load_weights 伪代码（L1661-1696）可执行
- **Scheduler 的双轨 block_size 注入**：`scheduler_tp_runner_bridge` 给出了 LLMEngine 层面注入 16/256 的明确逻辑
- **ref_code 精确度**：除 model_runner.py (258行) 外，其他 ref_code 引用（scheduler.py 84行, block_manager.py 112行, sampler.py 12行）文件规模小，line-level override 指示精确
- **Failure Mode Library**：22 条 FM 条目覆盖了常见 bug（双切片、contiguous、compiled .item() 等），配合故障排查

---

**结论**：对于 B=1, 单序列, nocompile 的 Qwen3 TP=4 推理，该蓝图**有能力引导完成重建**（约 85% 置信度）。但一旦需要多序列 batch prefill 或 CustomAR 的正确初始化，蓝图中的接口矛盾和信息断层将强制你进行代码脑补。建议优先修复 FG-1（统一 prefill 参数传递约定）和 FG-3（补齐 CustomAR register_buffer 的完整分配代码），可将得分提升至 ~90%。
