# 蓝图重构完备性漏洞报告

> 审计范围：`inference_blueprint.json` (834 行)，排除所有 `deepseek_v2_v3_mla_moe` 节点
> 审计身份：推理引擎独立闭环审计官
> 审计日期：2026-05-26

---

## 1. 🔴 Reconstructability Gaps（重构死锁漏洞项）

### Gap-1: Scheduler 核心算法为纯自然语言，零量化规格

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner`
- **当前内容**：一句自然语言描述 "必须满足 `max_num_batched_tokens` 与 `can_allocate`"，但 `max_num_batched_tokens` 和 `can_allocate` 两个关键变量在 JSON 全局均无定义。
- **后果**：Agent 无法实现 `schedule()` 方法。Prefill 批次到底能装多少 token？`max_num_batched_tokens` 是等于 `block_size * num_free_blocks` 还是另有公式？完全不可知。
- **缺失的规格**：
  ```python
  max_num_batched_tokens: int = max(1, num_free_blocks * block_size)
  can_allocate(seq) -> bool = num_free_blocks >= seq.required_blocks()
  can_append_one_more(seq) -> bool = num_free_blocks >= 1
  ```

### Gap-2: BlockManager 链式哈希与引用计数算法完全缺位

- **JSON Path**: `framework_layer.components[2]` (BlockManager)
- **当前内容**：`role` 字段仅一句话 "通过链式哈希执行 prefix caching 与 ref_count 共享"。
- **关键词扫描结果**：`compute_hash`、`ref_count`、`can_append` 在蓝图全局 **不存在**。
- **后果**：Agent 不知道：
  - `compute_hash(token_ids)` 的输入是完整 token 序列还是定长 block？
  - hash 冲突如何处理（无链式结构定义）？
  - `ref_count` 何时 +1（分配时？共享时？）、何时 -1（序列完成时？最后一 token 消费时？）？
  - `may_append` 与 `can_append_one_more` 是否同一语义？
- **缺失的规格**：
  ```python
  compute_hash(token_ids: tuple[int]) -> int  # 需指定 hash 函数与输入范围
  allocate(seq, num_blocks) -> list[int]       # 返回 block_table
  free(block_id) -> None: ref_count -= 1; if 0: return to free pool
  ```

### Gap-3: 双轨块大小隔离硬门禁在蓝图中完全不可见

- **JSON Path**: *(不存在)*
- **当前内容**：蓝图 **没有** 任何 `_dual_track_note`、`block_size` 隔离规则或 "禁止 TP Runner 路径使用 BlockManager" 的硬性约束。
- **实际源码事实**（AGENT_SKILL.md CRITICAL-01）：框架层 `block_size=16`，TP Runner 自管 KV cache `block_size=256`，两者严禁互通。
- **后果**：纯蓝图 Agent 必然将 `BlockManager.allocate()` 接入 QwenTPModelRunner，写出 `block_manager.allocate(seq)` + `seq.block_table` 与模型内部 `_block_table` 的对接代码——这正是 CRITICAL-01 违规。
- **缺失的规格**：必须在 `framework_layer` 或 `qwen3_tp_model_interfaces` 下新增：
  ```json
  "_dual_track_isolation": {
    "framework_path": {"block_size": 16, "manager": "BlockManager"},
    "tp_runner_path": {"block_size": 256, "manager": "model internal (_kv_block_size)"},
    "hard_gate": "禁止在 inference_backend='qwen_tp'/'deepseek_tp' 路径下调用 BlockManager API"
  }
  ```

### Gap-4: P0 增量 KV Cache 的 Decode Shape 契约缺失

- **JSON Path**: `framework_layer.data_flow_contracts.runner_decode_tensors`
- **当前内容**：`forward_call: "model(input_ids=ids, attention_mask=m, use_cache=False, return_dict=True)"` — 这是 **HF 兜底路径**的调用方式，不是 TP Runner 的。
- **缺失的关键信息**：
  - TP Runner decode 走的是 `layer.forward_decode()` 而非 `model.forward(use_cache=True)`
  - KV cache 是 paged 格式 `[num_blocks, block_size=256, num_kv_heads, head_dim]`
  - Decode 每步对 K/V Tensor 执行 `index_copy_` 写入选定 slot
  - `is_causal = (past_key_values is None)` 逻辑：prefill causal=True，decode causal=False
  - KV 长度用 GPU tensor `_kv_len_gpu` 追踪，不能用 Python int（与 CUDA Graph 兼容）
- **后果**：Agent 会写出 `model(input_ids, use_cache=True, past_key_values=past_kv)` 风格的 HuggingFace 路径，而非实际的 paged KV cache + `index_copy_` + `flash_attn_with_kvcache` 路径。

### Gap-5: CUDA Graph 执行契约在蓝图中完全不存在

- **JSON Path**: *(不存在)* — 蓝图全局搜索 `cuda_graph` 返回 "NOT FOUND"
- **实际需求**（来自 CLAUDE.md）：Stage D TP=4 CUDA Graph 是待实施的核心目标，涉及：
  - sglang 切图方案：`torch.fx.split_module` 在 `all_reduce_sum` 处拆分 FX 图
  - 通信子图 `backend='eager'` 不入 CUDA Graph
  - `out=static_buffer` 切断 cuBLAS 到 NCCL 的显存指针漂移
  - `@torch.library.custom_op` 注册 `all_reduce_sum` 屏蔽 Dynamo 追踪
  - Fake Tensor 形状推导
  - RNG 状态访问禁止规则
- **后果**：Agent 对 CUDA Graph 实现完全无指导。写出的 raw CUDA Graph 必定触发：cuBLAS workspace 地址漂移 → 通信读到垃圾数据 → 输出 hs diff 3.4 → 静默数值错误。

### Gap-6: Kernel 层 7 大标品替换黑盒零覆盖

- **JSON Path**: *(不存在)* — 蓝图全局搜索 `kernel`、`rms_norm`、`silu_and_mul`、`fused_add_rms_norm` 均返回 "NOT FOUND"
- **实际源码事实**（来自 CLAUDE.md 和 AGENT_SKILL.md）：
  - `rms_norm`: 须预分配 output buffer，fp16 下先转 fp32
  - `silu_and_mul`: gate/up 融合算子
  - `fused_add_rms_norm`: 跨层权重依赖（下一层的 `input_layernorm.weight`）
  - `custom_ar_all_reduce`: Gloo 组句柄交换，发生在 `load_weights` 与第一次 `forward` 之间
  - 残差链拓扑：层级参数传递顺序
- **后果**：Agent 无法实现任何自定义 CUDA/Triton kernel。写出的纯 PyTorch eager 路径性能将远差于目标。

### Gap-7: Flash Attention 集成契约不存在

- **JSON Path**: *(不存在)* — 蓝图全局搜索 `flash_attn` 返回 "NOT FOUND"
- **缺失的关键信息**：
  - Qwen3 prefill 使用 `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)`
  - Qwen3 decode 使用 `flash_attn_with_kvcache`（paged attention, block_size=256）
  - `cu_seqlens_q` 和 `cu_seqlens_k` 的构造方法：`torch.tensor([0, seqlen], dtype=torch.int32, device=...)`
  - 与传统 `scaled_dot_product_attention` 的 API 差异
- **后果**：Agent 大概率写出 `F.scaled_dot_product_attention` 路径或错误的 FA API 调用。

---

## 2. 🟡 Ambiguous Descriptions（信息熵不足警告项）

### A-1: `scheduler_to_runner` 中的控制流模糊

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_to_runner`
- **问题**：`"waiting 无可调度项时，从 running 中选择 can_append_one_more 的序列"` — 如果 waiting 非空但都 `can_allocate=False`，怎么办？是忙等循环还是跳过？调度器 `schedule()` 的返回值在不满足任何条件时是什么？
- **建议**：补充状态机伪代码，包括空批次返回行为。

### A-2: `model_output` shape 的 vocab_size 歧义

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.model_output`
- **问题**：`"[B, T, vocab_size]"` — 在 TP 路径下，每 rank 的 logits 实际是 `[B, T, vocab_size/tp]`，需要 `all_gather_last_dim` 才能得到完整 `vocab_size`。蓝图未标注这个 local vs global 的区别，Agent 可能误以为 raw model output 直接就是完整 logits。

### A-3: `all_reduce_sum` 的精度路径不完备

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime.collectives.all_reduce_sum`
- **问题**：`"fp16/bf16 先转 fp32 归约再 cast 回"` — 但 `RowParallelLinear` 的 partial output 是 `[B, T, out]` (通常是 hidden_size=4096)，`all_reduce_sum` 是否对该形状也适用？Custom AR（`custom_ar_all_reduce`）与 NCCL `dist.all_reduce` 的切换条件完全未提。

### A-4: `qwen3_tp_model_interfaces` 缺少 prefill/decode 分支语义

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces`
- **问题**：该节点直接列出 attention 和 MLP 的 shape，但没有区分 prefill 和 decode 两条不同路径。实际源码中，prefill 走 `forward()` 带 `causal=True`，decode 走 `forward_decode()` 带 KV cache write + `causal=False`。Agent 可能会写成一个统一路径，缺少 KV cache 累积逻辑。

### A-5: `rope_routing` 规则完整但缺 tensor 维度

- **JSON Path**: `model_layer.architecture_knowledge_base.qwen_series_dense.rope_routing`
- **问题**：`"Neox half-half rotate"` + `"严禁使用 GPT-J 奇偶交错 rotate"` — 规则清晰但缺少具体实现细节：`cos/sin` 的 `dtype`、展布方式（按 head_dim 展或按 position 展？）、Qwen3 使用 `rope_scaling` 时的 `mscale` 修正（类似 DeepSeek 的 YaRN）。

### A-6: `qwen_dense_loader` 规格与 `global_primitives_constraints.tp_linear_load_no_double_shard` 的交互边界模糊

- **JSON Path**: `model_layer.lazy_loader_synthesis_rules.qwen_dense_loader` vs `model_layer.architecture_knowledge_base.global_primitives_constraints.tp_linear_load_no_double_shard`
- **问题**：lazy_loader 定义了 `split_dim_0` / `split_dim_1`，但也存在 "输入已是 local shard 直接 copy" 的情况。Agent 在实现时不知道何时调用切片、何时直接 copy——需要显式的调用契约。

---

## 3. 🟢 Verdict（最终图纸判决）

### 量化结论

| 维度 | 蓝图覆盖度 | 说明 |
|------|-----------|------|
| 调度层 (Scheduler/BlockManager) | **15%** | 仅自然语言描述，无算法规格 |
| P0/P2/P3-FA 增量修改 | **10%** | 无 Flash Attention、torch.compile、增量 KV 的量化契约 |
| Kernel 层 7 大标品 | **0%** | 蓝图未涉及任何 custom kernel |
| CUDA Graph 静态执行 | **0%** | 蓝图未涉及 CUDA Graph |
| TP 通信层 (distributed/linear/embedding) | **55%** | Shape 契约相对完整，但缺 Custom AR |
| 模型架构路由 (Qwen3) | **50%** | 注意力/MLP 路由规则清晰，但缺 prefill/decode 分支 |
| 惰性加载器 | **60%** | 切片规则较完整，缺双重切片拦截的调用契约 |
| 验收/监控 | **35%** | 有目标但无具体采集方法 |

### 总体信息完备度：**~22%**

### 可否直接交付给全新 Agent 闭环开工：**否**

必须修复的 Top 3 阻断项：

1. **Gap-1 + Gap-2 合并**：补充 `Scheduler` 和 `BlockManager` 的 API 签名、核心算法伪代码和数据结构（`max_num_batched_tokens` 公式、`compute_hash` 签名、`ref_count` 生命周期、`can_allocate`/`can_append_one_more` 判定逻辑）。这是框架层的骨架，没有它连 `schedule()` 和 `allocate()` 都写不出来。

2. **Gap-3 + Gap-4 合并**：补充 TP Runner paged KV cache 的完整契约（`block_size=256` 自管内存、`index_copy_` 写入时序、`causal=True/False` 判定、"禁止 BlockManager 接入 TP 路径"的硬门禁）。没有这个，Agent 必然写出框架 BlockManager ↔ 模型 KV cache 的错误对接代码。

3. **Gap-5 + Gap-6 + Gap-7 合并**：补充 Flash Attention、Kernel 替换、CUDA Graph 三大缺失区块的技术规格。当前蓝图对这三大块的覆盖率均为 0%，Agent 无法产出任何高性能推理路径。

---

**一句话总结**：当前 `inference_blueprint.json` 本质上是一个 **"模型架构路由 + TP Shape 契约 + 踩坑知识库"**，对模型层 TP 切分的描述尚可（~50-60%），但对**调度层核心算法**（~15%）和**高性能执行引擎**（Kernel/FlashAttention/CUDAGraph, ~0%）基本是空白。纯蓝图 Agent 最多能写出一个"能跑但性能极差"的 eager TP 推理，产物距工业级引擎（55+ tok/s、CUDA Graph、paged KV cache）差距巨大。
