# 蓝图重构盲测漏洞报告 v10

**审计对象**: `inference_blueprint.json` v2.3.0 + `AGENT_SKILL.md`
**审计范围**: 排除 deepseek 章节与 CUDA Graph 优化路径，聚焦框架层 + Qwen3 TP=4 **nocompile** 引擎重构可行性
**审计日期**: 2026-05-27
**审计身份**: 独立第三方系统级审计官（Zero-Shot Reconstructability Auditor）
**审计前提**: CUDA Graph（`cuda_graph_execution_contract` 全章）不在本次审计范围内，不构成重构阻断

---

## 🔴 Fatal Gaps (致命脑补漏洞项)

### FG-1: 多序列 Prefill 的 cu_seqlens 传递链路断裂

- **JSON Path**: `framework_layer.data_flow_contracts.flash_attention_integration_contract.prefill_path.cu_seqlens_construction` + `qwen3_tp_model_interfaces.prefill_forward_pattern.layer_forward_pseudocode`
- **致命理由**: cu_seqlens 的构造伪代码（lines 472-477）位于 `flash_attention_integration_contract` 节点，但 `QwenDecoderLayerTP.forward()` 的签名（line 706）仅接受 `(hidden_states, positions, layer_cache, max_seq_len, residual)`，**不包含 cu_seqlens 参数**。而 `QwenAttentionTP.forward(prefill)` 伪代码（lines 715-729）在**内部**自行构造 cu_seqlens：

  ```python
  cu=torch.zeros(len(batch)+1,dtype=torch.int32,device=device)
  cu[1:]=torch.tensor([s.seq_len() for s in batch]).cumsum(0).to(device)
  ```

  `batch` 变量在此处凭空出现——QwenAttentionTP 无法自行获取 batch 信息（它只接收 hidden_states 和 positions，不知道有几个序列、每个多长）。这个 cu_seqlens 要么应由 Runner 层构造后沿 `forward() → layer.forward() → attn.forward()` 传递，要么通过其他隐式参数（如 `layer_cache`）传入。**蓝图在 cu_seqlens 的依赖注入路径上存在架构断层。**

- **后果**: 单序列 prefill 可以硬编码绕开（cu_seqlens = [0, num_tokens]），但**多序列 ragged prefill 必定崩溃**——各序列 KV 互相可见导致 attention 输出错误，且 flash_attn_varlen_func 的 cu_seqlens_k 参数语义完全错误。

---

### FG-2: QKVColumnParallelLinear 输出 split→reshape 链的维度计算过于简略

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers.qkv_column_parallel_forward` + `qwen3_tp_model_interfaces.qkv_projection`
- **致命理由**: QKVColumnParallelLinear.forward() 的伪代码（lines 580-590）给出了 split 操作：

  ```python
  q, k, v = y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
  # reshape: q→[B,T,num_heads/tp,head_dim], k→[B,T,num_kv_heads_local,head_dim]
  ```

  但 reshape 以**注释**而非代码给出。对于 Qwen3-8B TP=4：
  - `q_size = num_heads * head_dim // tp = 32 * 128 / 4 = 1024`
  - `kv_size = num_kv_heads_local * head_dim = max(1, 8//4) * 128 = 2 * 128 = 256`
  - Q reshape: `[1,1,1024] → [1,1,8,128]` — 使用 `.view(B, T, -1, head_dim)` 即可
  - K/V reshape: `[1,1,256] → [1,1,2,128]` — 使用 `.view(B, T, -1, head_dim)` 即可

  但由于 `num_kv_heads_local = max(1, num_kv_heads // tp)`，且 GQA 下 `num_kv_heads=8` 可能小于 `tp=4` 时触发的 head 复制逻辑（`allow_kv_replication`），K/V 的 reshape 目标维度 `(-1, head_dim)` 中的 `-1` 推导依赖 `num_kv_heads_local` 的正确计算。**蓝图未给出这一 reshape 的完整代码，仅靠注释提示。** 若重构者错误地使用 `num_heads // tp`（32//4=8）而非 `num_kv_heads_local`（2）来 reshape K/V，K/V 的 head 维度将膨胀 4x 且静默通过 shape 检查。

- **后果**: GQA + TP 组合下的 K/V reshape 维度错误→静默数值错误（K/V head 维度错误，但 total elements 仍然匹配）。

---

### FG-3: Prefill Attention forward() 完整方法体简化为注释级描述

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.prefill_forward_pattern.layer_forward_pseudocode` (lines 714-729)
- **致命理由**: 对比 decode 路径的 `full_method_body`（lines 639-683，含完整逐行伪代码），prefill 的 QwenAttentionTP.forward() 伪代码多处使用注释代替代码：
  - `reshape to [B,S,heads,dim]` — 注释而非代码
  - `rotary_embedding(positions,q_flat,k_flat,self.head_dim,cos_sin_cache,is_neox=True)` — 未指定 `cos_sin_cache` 来源（是 `self._cos_sin_cache_gpu` 还是参数？）
  - `out = flash_attn_varlen_func(q_flat,k_flat,v_flat,cu,cu,max_seqlen,max_seqlen,causal=True)` — 未指定 `max_seqlen` 的计算方式（应为 `max(seq.seq_len() for seq in batch)` 或 `q_flat.shape[0]` 取决于 batch 模式）
  - `slot_mapping=build_slot_mapping(batch,self._block_table,256)` — `build_slot_mapping` 函数未定义，需自行实现（多序列伪代码在另一个节点 `paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm.multi_seq`）
  - `kc_flat.index_copy_(0,slot_mapping,k_flat)` — 但 `k_flat` 的 shape 为 `[total_tokens, num_kv_heads, head_dim]`，需确认已从 4D reshape 到 3D

  蓝图的信息是“全的”但分散在 4 个不同节点，prefill attention 伪代码只是索引而非完整实现。

- **后果**: 重构者容易遗漏 reshape 步骤（尤其是 4D→3D 的展平用于 index_copy_），导致 KV 写入 shape mismatch 或静默写入错误位置。**与 decode 路径有完整逐行伪代码可抄形成鲜明对比。**

---

### FG-4: CustomAR 初始化所需 vLLM 符号的精确 import 路径缺失

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce.init_state_machine`
- **致命理由**: IPC exchange 伪代码（lines 880-897）使用了 4 个 vLLM 内部符号：
  - `ops.allocate_shared_buffer_and_handle(size)`
  - `ops.open_mem_handle(h)`
  - `ops.init_custom_ar(pointers, rank_data, rank, fully_connected)`
  - `ops.register_buffer(self._ptr, buf_pointers)`

  但**未给出任何一个符号的完整 import 路径**。`kernel_replacement_plan.md §九` Snippet F（line 343-372）只包含 `all_reduce` wrapper，不含初始化函数。在源码销毁假设下，重构者不知道：
  - 这 4 个函数是从 `vllm._custom_ops` 还是 `vllm._C`（C++ 扩展）导入？
  - 它们的精确 Python 函数签名（`rank_data` 和 `fully_connected` 参数的类型和含义）？
  - `allocate_shared_buffer_and_handle` 返回的 tuple 格式？

- **后果**: 重构者必须自行 grep vLLM installed package 源码定位这些符号。一旦导入路径选错（如 `vllm._C.ops` vs `vllm._custom_ops`），会触发 `AttributeError` 且错误信息不包含正确的导入路径提示。

---

### FG-5: ColumnParallelLinear / RowParallelLinear 基类 forward() 隐含行为未定义

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers`
- **致命理由**: QKVColumnParallelLinear 的 forward() 伪代码（line 582）写 `y = F.linear(x, self.weight)`，这暗示**它不调用基类 forward()**，而是直接做 GEMM。但 ColumnParallelLinear 基类可能包含 `gather_output` 和 `all_reduce` 逻辑。RowParallelLinear 的 `return self.o_proj(out)` 注释说“internally calls all_reduce_sum”（line 670），但：

  1. RowParallelLinear.forward() 的完整代码未被给出
  2. `all_reduce_sum` 的实现（CustomAR vs `dist.all_reduce` 切换逻辑）未被给出——蓝图仅说“透明切换 CustomAR P2P”（line 906）
  3. 当 `tp_size=1` 时 RowParallelLinear 是否跳过 all_reduce（no-op）？伪代码未说明

- **后果**: 重构者可能写出一个做了双重 all_reduce 的 RowParallelLinear，或一个在 tp_size=1 时报错的 CustomAR 调用。

---

## 🟡 Override Warnings (重载失效警告项)

### OW-1: nano-vllm Scheduler — `can_append`/`may_append` 替换为 `can_append_one_more` 未逐行标注

- **JSON Path**: `framework_layer.components.Scheduler._nano_vllm_override` + `data_flow_contracts.scheduler_to_runner.can_append_one_more`
- **警告**: nano-vllm scheduler.py（84 行）decode 阶段调用 `self.block_manager.can_append(seq)` 和 `self.block_manager.may_append(seq)`（lines 52, 60）。蓝图将 block_size 和 preempt 的修改说清楚了，但**未显式说明这两行也需要改为 `can_append_one_more` 逻辑**。TP 路径下 `can_append`/`may_append` 依赖 BlockManager 的内部状态，而 BlockManager 在 TP 路径已降级为纯计数器——这两行不改会导致 TP 路径下调度器错误地调用已降级的 BlockManager 方法。

- **风险**: decode 调度逻辑静默失效——调度器认为可以 append 但实际上 BlockManager 的降级 no-op 不反映真实 KV 容量。

---

### OW-2: fused_add_rms_norm post_mlp weight — 蓝图与 kernel_replacement_plan.md 存在矛盾

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint`
- **警告**: 蓝图 line 834-843 明确声明 *"post_mlp 的 fused_add_rms_norm 使用本层 post_attention_layernorm.weight（非下一层 input_layernorm.weight）。与 vLLM 原始设计不同但数值等价"*。然而 `kernel_replacement_plan.md §九` line 234 写的是 *"weight 参数是**下一层**的 input_layernorm.weight"*。更麻烦的是，蓝图 line 842 的 `last_layer_note` 说第 36 层 post_mlp 不调用 fused_add_rms_norm——但如果重构者按 kernel_replacement_plan 的跨层 weight 方案实现了非最后一层，却在最后一层发现没有"下一层"可以取 input_layernorm.weight，会陷入困惑。

- **风险**: 重构者实现时在两种 weight 来源方案之间摇摆，写出混合体。正确的做法是以蓝图的 `residual_chain_pseudocode`（lines 834-838）为准。

---

### OW-3: QKVColumnParallelLinear weight loading — q_size 公式中 `// tp_size` 的双重除风险

- **JSON Path**: `model_layer.lazy_loader_synthesis_rules.qwen_hf_key_mapping` + `qwen3_kernel_contracts.qkv_merged_projection.constraint`
- **警告**: 蓝图 line 917 的 constraint 明确写了 `num_heads = config.num_attention_heads (全量值，非 per-rank)。q_size = num_heads * head_dim // tp_size 已包含除法，勿双重除`。但 `lazy_loader_synthesis_rules` 的 `_load_tensor` 伪代码（line 1578-1583）对所有 `split_dim=0` 权重统一使用 `full.chunk(tp_size, dim=0)[self.tp_rank]`。对于 QKV 合并投影，传入的 `full` tensor 已经是 ColumnParallel 切片后的 `[q_size+2*kv_size, hidden_size]`，如果 `_load_tensor` 再次 chunk，就会双重切片。虽然 `double_shard_guard`（shape 相等检查）能拦截部分情况，但**蓝图的 load_weights 伪代码和 constraint 文字分布在两个不相邻的节点**——重构者容易读前者而忽略后者。

- **风险**: QKV 权重双重切片导致 Q/K/V 段错位，静默数值错误。

---

### OW-4: nano-vllm RMSNorm — 整体替换要求在多个位置分别声明，易漏

- **JSON Path**: `model_layer.architecture_knowledge_base.global_primitives_constraints.rmsnorm_precision_law._nano_vllm_override` (line 1260)
- **警告**: 蓝图 line 1260 明确写了"nano-vllm 原始 RMSNorm 实现需整体替换为 vLLM kernel wrapper"且"不能保留 nano-vllm 的 PyTorch RMSNorm 框架"。但这是 `global_primitives_constraints` 中的约束（line 1249+），距离 nano-vllm 的 RMSNorm 参考代码路径（`ref_projects/nano-vllm/nanovllm/layers/layernorm.py`）在文档中相距约 1200 行。重构者按"先读框架组件→再抄参考代码→再读约束"的顺序工作时，可能已经照抄了 nano-vllm RMSNorm 后才看到替换要求。

- **风险**: 写了两遍 RMSNorm 类（nano-vllm 版 + vLLM kernel wrapper 版），或者更糟——混合体（保留 nano-vllm 的 `nn.Parameter` 声明 + vLLM kernel forward 调用，但 `out` tensor 预分配方式不一致）。

---

### OW-5: KVMemoryPool GPU placeholder — 禁止调用但调用路径未逐条列出

- **JSON Path**: `framework_layer.components.KVMemoryPool._nano_vllm_override` + `tp_path_note`
- **警告**: 蓝图说 TP 路径下"禁止调用 KVMemoryPool 的 GPU placeholder 创建逻辑"。但 nano-vllm 的 `model_runner.py`（258 行）可能在 `__init__` 的多处调用 KVMemoryPool 方法创建 placeholder。重构者抄入 nano-vllm ModelRunner 代码后，需要自行判断：`allocate_kv_cache` 方法是只禁 TP 路径还是整体删除？`estimate_num_blocks` 在 TP 路径保留（做显存预算日志）——这个区分只在 `tp_path_note` 中提到，未在 override 中逐方法标注。

- **风险**: 显存中同时存在 QwenAttentionTP 内部创建的 paged KV cache（~256 MB per rank）和 KVMemoryPool 分配的 HF 风格 contiguous KV placeholder（~256 MB per rank），总计浪费 ~512 MB/rank。

---

## 🟢 Reconstructability Score (重构可行性判决)

### 量化分数: **85 / 100**

以当前掌握信息（JSON + AGENT_SKILL.md + kernel_replacement_plan.md + RAG 可检索 ref_code），写出**一次跑通**的 Qwen3 TP=4 **nocompile** 引擎的综合把握约 85%。

### Top 3 阻断因素（阻碍达到 100%）:

| # | 阻断因素 | 影响范围 | 减分 |
|---|---------|---------|------|
| 1 | **cu_seqlens 在多序列 prefill 中的传递链路缺失** (FG-1) — 单序列可工作，多序列 ragged prefill 的 cu_seqlens 注入点不明确 | 多序列 prefill 正确性 | -5% |
| 2 | **QKV split→reshape 链从注释到代码的鸿沟** (FG-2) + **Prefill attention forward 伪代码碎片化** (FG-3) — GQA+TP 下 K/V reshape 的 `num_kv_heads_local` 算术和 `build_slot_mapping` 依赖正确理解多个分散节点的约束 | Prefill 路径正确性 | -5% |
| 3 | **CustomAR 初始化所需的 vLLM 内部符号 API 接口盲区** (FG-4) + **ParallelLinear 基类隐含行为** (FG-5) — 导入路径不可知 + all_reduce_sum 的 CustomAR/fallback 切换逻辑缺失 | TP 通信初始化 | -5% |

### 蓝图做得好的部分（分值贡献）:

| 维度 | 评价 | 贡献 |
|------|------|------|
| **Decode 热路径** — `full_method_body` (lines 639-683) 给出 QwenAttentionTP.forward_decode() + QwenDecoderLayerTP.forward_decode() 完整逐行伪代码 | 可直接抄入实现 | +25% |
| **模型维度** — `qwen3_8b_model_dims` 含 TP=4 per-rank 计算值 (lines 1371-1391) + `class_hierarchy` 含精确 `__init__` 签名 (lines 732-784) | 零脑补即可实例化所有模块 | +15% |
| **权重加载** — `qwen_hf_key_mapping` (lines 1527-1548) 含 Q-K-V 拼接顺序 + `load_weights_pseudocode` (lines 1549-1584) 含完整 load_weights/_load_one_weight/_load_tensor 三函数 | 权重加载链路可直接翻译为代码 | +15% |
| **Paged KV Cache** — `paged_kv_cache_contract` (lines 339-422) 含 slot_mapping 算法、multi_seq 伪代码、reshape chain、timeline | Prefill KV 写入可完整实现 | +12% |
| **Kernel Wrappers** — `kernel_replacement_plan.md §九` 的 6 个 Snippet (rms_norm/fused_add_rms_norm/silu_and_mul/rotary_embedding/cos_sin_cache/custom_all_reduce) 每个都有精确的函数签名 + 约束 | 所有 kernel 调用可直接复制 | +10% |
| **Failure Mode Library** — 16 条 symptom/check/fix 词条 (FM-001~FM-015) | 排障效率大幅提升 | +5% |
| **顶层 forward 编排** — `model_forward_pseudocode` (lines 786-809) 给出 QwenForCausalLMTP.forward() 完整 prefill/decode 分发逻辑 | 直接可用 | +5% |
| **TP 采样协议** — `tp_sampling_protocol` 给出完整 broadcast 伪代码 (lines 186-196) | 直接可用 | +3% |

### v10 vs v9 分数变化说明:

v9（52%）→ v10（85%）的 +33% 修正源于对蓝图实际内容的重新逐行核实：

1. **+20%** — v9 声称 forward_decode() / load_weights() / model_forward() / VocabParallelEmbedding.forward() "无伪代码"。经核实，这些节点在 blue print lines 530-561 / 639-683 / 786-809 / 1549-1584 中均有**完整逐行伪代码**，可直接翻译为 Python。
2. **+8%** — v9 将 CUDA Graph 的未完成状态计为阻断因子。本次审计排除 CUDA Graph 范围后，该减分移除。
3. **+5%** — v9 未识别 `full_method_body` 节点（decode 完整方法体）的存在，低估了蓝图对动态执行流的覆盖度。

### 剩余 15% gap 的本质:

蓝图在 **静态知识表达**（维度、属性、约束、shape、签名）层面覆盖度约 95%，在**单序列执行流**（decode path、weight loading、top-level forward）层面覆盖度约 90%。剩余 gap 集中在一个模式：**多序列 batch 场景下，组件间的动态参数传递链路**（cu_seqlens 从 Runner 到 QwenAttentionTP，batch metadata 在各层间的流动）。这是"每个模块单独看都有伪代码，但连起来就断了根线"的系统集成问题。
