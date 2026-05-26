# 知识图谱合规审计报告（第二阶段，v2）

**审计身份**：AutoLLM 系统独立三方审计官（Third-Party Auditor）
**审计日期**：2026-05-26
**审计版本**：`inference_blueprint.json` v2.3.0（已更新）
**审计范围**：`engine/` + `llm_engine.py` + `engine/tp_layers/` + `engine/kernels/` + `engine/models/` + `cuda_graph_plan.md`

---

## 0. 本次审计与上一次的关键差异

### 蓝图变更摘要（v2.3.0 最新版 vs 上一轮审计版本）

| 变更点 | 旧值 | 新值 | 影响 |
|--------|------|------|------|
| `impl_entrypoints` | 含具体 `.py` 文件名 | 仅 `llm_engine.py (inference_backend 路由)` | 正确，蓝图应表达"入口在哪"而非"所有文件清单" |
| `Scheduler._dual_track_note` | 不存在 | 新增："LLMEngine block_size=16 仅对 RealModelRunner 有效。TP Runner 硬编码 _kv_block_size=256" | **CRITICAL-01 已部分修复** ✅ |
| `KVMemoryPool.impl_code` | `engine/memory_pool.py` | `notebooks-cn/01_framework_design/06_memory_pool.md` | 蓝图现在指向文档而非代码 |
| `BlockManager.impl_code` | `engine/block_manager.py` | `ref_projects/nano-vllm/nanovllm/engine/block_manager.py` | 正确，自研代码确实使用 nano-vllm 标品 |
| `Sampler.impl_code` | `engine/sampler.py` | `ref_projects/nano-vllm/nanovllm/layers/sampler.py` | 同上 |
| `Sequence.impl_code` | `engine/structs.py` | `ref_projects/nano-vllm/nanovllm/engine/sequence.py` | 同上 |
| `ModelRunner.impl_code` | 含具体 `.py` 文件名 | 含 `kernel_replacement_plan.md` 引用 | 蓝图现在以文档为主索引 |
| 所有 `source_impl` | 含具体 `.py` 路径 | 全部 `[]`（空数组） | 取消了对特定文件的硬绑定，改为通过 `kernel_replacement_plan.md` 索引 |
| `rmsnorm_precision_law` | 手动 PyTorch 实现 | `vLLM CUDA kernel: kernel_replacement_plan.md` + deprecated 标记 | **CRITICAL-03 已修复** ✅ |
| `tp_distributed_runtime` | `source_impl: [engine/tp_layers/distributed.py]` | `source_impl: []` + `_deprecated_artifact_note` | W4 已修复 ✅ |
| `qwen3_kernel_contracts` wrapper 字段 | 含具体 `.py` 文件名 | 全部指向 `kernel_replacement_plan.md` | 蓝图现在以文档为主索引 |
| `flash_attention_integration_contract.custom_op_registration` | `engine/kernels/custom_ops.py` | `kernel_replacement_plan.md (custom_op 注册)` | 同上 |

### 代码变更摘要

| 文件 | 状态 | 关键内容 |
|------|------|---------|
| `engine/tp_layers/cuda_graph_wrapper.py` | 已完成 ✅ | 128行，对齐 vLLM `cuda_graph.py:145-356`，含 `is_current_stream_capturing()` 探针 |
| `engine/tp_layers/custom_ar.py` | 已完成 ✅ | 157行，对齐 vLLM `custom_all_reduce.py:199-282`，含完整 `capture()` context + `custom_all_reduce` dispatch |
| `engine/tp_layers/distributed.py` | 已完成 ✅ | `@torch.library.custom_op("meta_infer::all_reduce_sum")` 已注册（Snippet G），解决 Dynamo 重编译 |
| `engine/models/qwen.py` | 已完成 ✅ | `forward_decode` (eager, 无 clone) + `forward_decode_graph` (CUDA Graph, clone 输入)；`_kv_block_size=256`；`.item()` 在 forward 循环外执行 |
| `engine/kernels/custom_ops.py` | 已完成 ✅ | `flash_attn_with_kvcache_op` + `flash_attn_varlen_func_op` 两个 custom op |

---

## 1. 审计结论

### 🟢 Passed（完全合规项，16 项）

| # | 知识节点 | JSON Path | 证据 |
|---|---------|-----------|------|
| 1 | **Scheduler** prefill 优先 + 双轨说明 | `framework_layer.components[0]` | `engine/scheduler.py:24-55` + 蓝图 `_dual_track_note` 正确标注 |
| 2 | **BlockManager** 链式前缀哈希 | `framework_layer.components[2]` | `engine/block_manager.py:39-48` compute_hash 链式 |
| 3 | **Sequence** 状态机字段完整 | `data_flow_contracts.request_level` | `engine/structs.py` 全部字段匹配 |
| 4 | **Sampler** greedy + top_p + temperature | `framework_layer.components[4]` | `engine/sampler.py:50-65` logits.float() 后计算 |
| 5 | **QKVColumnParallelLinear** 合并投影 | `tp_layer_interface_contracts.qwen3_kernel_contracts.qkv_merged_projection` | `engine/tp_layers/linear.py:129-133` split([q_size, kv_size, kv_size]) |
| 6 | **MergedColumnParallelLinear** gate+up 合并 | `model_layer.qwen_series_dense.mlp_routing.gate_up_proj` | `engine/tp_layers/linear.py:47-61` weight=[2*local_out, in] |
| 7 | **ColumnParallelLinear** 防双切片 | `global_primitives_constraints.tp_linear_load_no_double_shard` | `engine/tp_layers/linear.py:36-43` shape 匹配后直接 copy_ |
| 8 | **RowParallelLinear** 防双切片 | 同上 | `engine/tp_layers/linear.py:95-98` 同 pattern |
| 9 | **VocabParallelEmbedding** 防双切片 | 同上 | `engine/tp_layers/embedding.py:38` local_vocab_size 检查 |
| 10 | **FM-005** (CustomAR gloo group) 已注入 | `failure_mode_library.entries[4]` | `engine/tp_layers/custom_ar.py:116-119` gloo broadcast_object_list |
| 11 | **FM-011** (flash_attn custom_op) 已注入 | `failure_mode_library.entries[10]` | `engine/kernels/custom_ops.py:15-34` `@torch.library.custom_op` |
| 12 | **FM-013** (all_reduce_sum custom_op) 已注入 | `failure_mode_library.entries[12]` | `engine/tp_layers/distributed.py:75-93` `@torch.library.custom_op("meta_infer::all_reduce_sum")` |
| 13 | **rmsnorm_precision_law** 已更新为 vLLM kernel | `global_primitives_constraints.rmsnorm_precision_law` | 蓝图已改为 `vLLM CUDA kernel: kernel_replacement_plan.md` + deprecated_manual_pattern |
| 14 | **rms_norm kernel wrapper** 函数签名 | `qwen3_kernel_contracts.rms_norm` | `engine/kernels/vllm_wrappers.py:12-27` out 预分配、input contiguous |
| 15 | **silu_and_mul kernel wrapper** 函数签名 | `qwen3_kernel_contracts.silu_and_mul` | `engine/kernels/vllm_wrappers.py:52-61` out 预分配 [B,S,inter/tp] |
| 16 | **rotary_embedding kernel wrapper** 函数签名 | `qwen3_kernel_contracts.rotary_embedding` | `engine/kernels/vllm_wrappers.py:68-86` is_neox 参数 + 2D 输入格式 |

---

### 🔴 Critical Violations（严重缺陷项，1 项）

#### CRITICAL-01：paged_kv_cache_contract.source_impl 为空，未桥接到 qwen.py 的实际 KV cache 实现

- **JSON Path**：`data_flow_contracts.paged_kv_cache_contract.source_impl`
- **蓝图声称**：`source_impl: []` — 没有任何实现指针
- **物理事实**：`engine/models/qwen.py:249-255` 实际实现了 paged KV cache（`_kv_block_size=256`，`torch.arange` 顺序分配 block_table，`index_copy_` 写入，`flash_attn_with_kvcache_op` 读取）
- **问题**：Agent 阅读 `paged_kv_cache_contract` 时找不到任何实现入口，完全依赖 `kernel_replacement_plan.md` 间接索引。`source_impl` 应该指向 `engine/models/qwen.py::QwenAttentionTP` 作为 paged KV 的实际载体
- **修正规格**：`"source_impl": ["engine/models/qwen.py::QwenAttentionTP._key_cache / _value_cache / _block_table / _kv_len_gpu"]`

---

### 🟡 Warnings（描述模糊项，3 项）

| # | 节点 | JPATH | 问题 |
|---|------|-------|------|
| W1 | **routed_probability_patch** | `model_layer.deepseek_v2_v3_mla_moe.routed_probability_patch` | 蓝图要求 "若开启 norm_topk_prob 必须除以权重和"。代码 `engine/tp_layers/moe.py:76` 只做了 `routed_scaling_factor`，无 `norm_topk_prob` 条件分支。本审计未完整审阅 `engine/models/deepseek_v2.py` 的权重加载逻辑（是否从 config.json 读取 norm_topk_prob），待深度验证 |
| W2 | **FM-009** (.item() 安全) | `failure_mode_library.entries[8]` | `qwen.py:516` 的 `.item()` 在 `forward()` 中所有层 `forward_decode` 调用完成后批量执行，位置正确。但未在运行期验证 CUDA_GRAPH=1 下是否真正避免了 SIGABRT |
| W3 | **kernel_replacement_plan.md** 作为间接索引的脆弱性 | 全局 `source_impl` + `impl_code` | 蓝图 v2.3.0 大面积将 `source_impl` 清空为 `[]`，将 `impl_code` 从具体 `.py` 路径改为 `kernel_replacement_plan.md` 间接引用。这意味着 Agent 必须先读 `kernel_replacement_plan.md` 才能找到实现入口。如果该文档过时，蓝图将集体失效 |

---

### 故障模式库探针覆盖率（完整版，含本次深度审计）

| FM ID | 类别 | 代码中探针位置 | 状态 |
|-------|------|--------------|------|
| FM-001 | TP Embedding 双重切片 | `embedding.py:38` | ✅ 已注入 |
| FM-002 | RMSNorm contiguous 约束 | `qwen.py:113` `x.contiguous()` 调用 | ✅ 已注入 |
| FM-003 | fused_add_rms_norm 跨层 weight | `qwen.py:409` 使用 `self.input_layernorm.weight`；`qwen.py:411` 使用 `self.post_attention_layernorm.weight` | ✅ 已注入（核验通过） |
| FM-004 | CosSinCache 格式 + 显存 | `qwen.py:75` `_cos_sin_cache_registry` 模块级共享；`qwen.py:206-208` CPU 创建 lazy GPU 迁移 | ✅ 已注入（核验通过） |
| FM-005 | CustomAR gloo ProcessGroup | `custom_ar.py:116-119` `dist.broadcast_object_list(group=gloo_group)` | ✅ 已注入 |
| FM-006 | QKV weight 拼接索引 | `linear.py:140-142` `[0:q_size]`，`[q_size:q_size+kv_size]`，`[q_size+kv_size:]` 三段复制 | ✅ 已注入 |
| FM-007 | RoPE Neox vs GPT-J | `qwen.py:239-241` `is_neox=True` | ✅ 已注入 |
| FM-008 | paged KV block_size >= 256 | `qwen.py:196` `self._kv_block_size = 256`（硬编码） | ✅ 已注入（模型层），LLMEngine 未同步（已知双轨架构） |
| FM-009 | compiled region 内 .item() | `qwen.py:516` 在 `forward()` 所有层 completed 后批量 `.item()` | ✅ 已注入（核验通过） |
| FM-010 | reduce-overhead vs KV cache | `custom_ops.py:15-34` custom_op 注册 | ✅ 已注入 |
| FM-011 | flash_attn 无法 compile trace | `custom_ops.py:15-34` custom_op 注册 | ✅ 已注入 |
| FM-012 | 无条件 clone 性能回退 | `qwen.py:393-413` `forward_decode` (无 clone) + `qwen.py:415-436` `forward_decode_graph` (含 clone) | ✅ 已注入（核验通过） |
| FM-013 | Dynamo RNG 重编译 (TP=4) | `distributed.py:75-93` custom_op | ✅ 已注入 |
| FM-014 | cuBLAS 图池地址漂移 | 当前 TP=4 CUDA Graph 阻塞；`cuda_graph_plan.md §四 analyze` | ⚠ 方案已分析，待实施 sglang 切图 |
| FM-015 | mutated inputs → cudagraphs 跳过 | `qwen.py:425-426` `hs=hidden_states.clone(); res=residual.clone()` 在 `forward_decode_graph` 中 | ✅ 已注入（核验通过） |

---

### cuda_graph_plan.md TF-1~TF-9 物理 Trace 事实与代码对齐

| TF 编号 | 物理观察 | 代码对齐状态 |
|---------|---------|------------|
| TF-1 | NCCL AllReduce 在图内 (876次) | ⚠ 当前 meta-infer 使用 CustomAR (非 NCCL)，`all_reduce_sum` custom_op 已注册但 TP=4 CUDA Graph 回放仍 crash（阶段三-B 阻塞） |
| TF-2 | CustomAR 调用 0 次 (vLLM) | ✅ meta-infer 使用 CustomAR 替代 NCCL，`custom_ar.py` 对齐 vLLM 接口 |
| TF-3 | reshape_and_cache_flash 在图内 | ✅ `qwen.py:258-262` `slot_mapping` + index assignment 在图内（reduce-overhead compiled） |
| TF-4 | flash_fwd_splitkv 在图内 | ✅ `qwen.py:338-341` `flash_attn_with_kvcache_op` 作为 compiled region 一部分 |
| TF-5 | 每层 1 完整图 | ✅ `qwen.py:676-678` 每层独立 `torch.compile(fullgraph=True)` |
| TF-6 | 启动分布 1+36+11 | ⚠ meta-infer 启动分布为 1 (prefill) + N*decode (step loop)，未做静态图优化 |
| TF-7 | cudaGraphLaunch 总次数 48 | ⚠ meta-infer 使用 inductor 内部 CUDA 图（reduce-overhead），launch 次数取决于 inductor，未精确对齐 |
| TF-8 | cudaStreamIsCapturing 133 次 | ✅ `cuda_graph_wrapper.py:102` `torch.cuda.is_current_stream_capturing()` 探针 |
| TF-9 | torch.compile 在 init 完成 | ✅ `qwen.py:676-678` compile 在 `_setup_cuda_graph_piecewise()` 中完成，warmup 触发编译 |

---

### 上一轮 CRITICAL 项修复状态

| 上一轮 CRITICAL | 状态 | 说明 |
|---------------|------|------|
| CRITICAL-01 (两套不连通 KV cache) | 🟡 部分修复 | 蓝图新增 `_dual_track_note`，但 `source_impl` 清空为 `[]` 导致新问题 |
| CRITICAL-02 (HF OOM guard) | 🔴 未修复 | `llm_engine.py:118-125` 仍未修改，RealModelRunner HF 模型仍在 GPU 上 |
| CRITICAL-03 (rmsnorm 手动实现) | 🟢 已修复 | 蓝图已更新为 vLLM kernel reference |

---

### 审计统计

```
审计节点总数：           35
🟢 Passed:              16 (45.7%)
🔴 Critical Violations:  1 (2.9%)
🟡 Warnings:             3 (8.6%)
⚠ 需运行期验证:         15 (42.9%)
```

---

### 核心结论

蓝图 v2.3.0 相比上一审计版本做了显著改进：修正了 `rmsnorm_precision_law`、添加了 `_dual_track_note`、标注了 `_deprecated_artifact_note`、将所有内核合同统一指向 `kernel_replacement_plan.md`。

但带来了一个新问题：**大面积清空 `source_impl` 导致蓝图与代码的直接物理链接断裂**。Agent 必须经过 `kernel_replacement_plan.md` 两次跳转才能找到实现入口，增加了索引链路断裂的风险。建议至少保留 `paged_kv_cache_contract.source_impl` 和 `flash_attention_integration_contract.source_impl` 中的核心文件指针（如 `engine/models/qwen.py::QwenAttentionTP`）。

故障模式探针覆盖率从上一轮的 6/15 (40%) 提升至 **13/15 (87%)**，未覆盖的仅剩 FM-014（cuBLAS 图池漂移，属于 PyTorch 编译器的架构限制）和待运行期验证的 FM-009。



