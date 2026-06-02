# 蓝图重构盲测漏洞报告 v17

**审计对象**：`inference_blueprint.json` v2.3.0 + `AGENT_SKILL.md` + 全部 `ref_code` + `ref_docs`
**审计假设**：`meta-infer` 业务源码已物理销毁，技术资产仅剩上述文件
**目标重构范围**：Qwen3-8B, nocompile, B=1, TP=4（排除 DeepSeek 章节，排除 CUDA Graph/sglang 切图）
**审计日期**：2026-05-27

---

## 一、审计方法论执行记录

### 1.1 ref_docs 强制交叉验证（已实际打开 5 个文件）

| # | 文件 | 蓝图引用的知识点 | 验证结果 |
|---|------|----------------|---------|
| 1 | `notebooks-cn/07_improvementPlan/kernel_replacement_plan.md` | §九 "vLLM 标品 Kernel 自包含调用契约速查" | **完整存在** — 7 个 kernel 的调用契约（函数签名、dtype/shape 约束、import 路径、代码模板）与蓝图一致 |
| 2 | `notebooks/01_framework_design/02_scheduler.md` | Scheduler 组件 ref_doc | **设计文档，非实现规范** — 涵盖调度概念但不含蓝图伪代码级别的实现细节 |
| 3 | `notebooks/01_framework_design/03_kv_cache.md` | KVMemoryPool/BlockManager ref_doc | **设计文档** — paged attention 概念概述，但不含蓝图的具体 `[num_blocks,256,heads,dim]` 实现 |
| 4 | `notebooks-cn/07_improvementPlan/qwen3_effective_changes.md` | 多次引用（#8 paged KV, #9 compile, #10 FA） | **知识追溯表完整** — 10 个改动点的元信息完整，但不含具体伪代码 |
| 5 | `notebooks-cn/07_improvementPlan/improvement_plan.md` §P0 | paged_kv_cache_contract source_refs | **内容与蓝图矛盾** — 详见 FG-4 |

### 1.2 ref_code 路径核验（全部组件）

所有 `ref_code` 路径（nano-vllm 相关）均存在且可读。行号引用经逐条 grep 验证准确。

---

## 二、维度一：重载契约核验 (Override & Adaptation Check)

### 🟢 总体评价：nano-vllm 重载规则精确

蓝图对 5 个 nano-vllm 组件标注了 `_nano_vllm_override` 规则，且精确到行号和具体方法名：

| 组件 | 覆盖的 nano-vllm 行号 | 验证结果 |
|------|----------------------|---------|
| Scheduler | L52-57 (preempt), L54 (running.pop), L66-69 (preempt 方法体) | 行号精确匹配 |
| BlockManager | 整体降级为 no-op | 已指明 |
| KVMemoryPool | 禁用 GPU placeholder | 已指明 |
| ModelRunner | allocate_kv_cache/capture_cudagraph/IPC SharedMemory 删除，prepare_prefill 部分保留 | 逐函数指明 |
| Sampler | rank 0 采样 + broadcast | 协议完整 |

### 🟡 Override Warnings

#### OW-1：`prepare_prefill` slot_mapping 部分保留与重写的边界模糊

- **JSON Path**: `framework_layer.components[3]._nano_vllm_per_function.prepare_prefill()`
- **内容**：蓝图说保留 `cu_seqlens` 构造逻辑（L129-150），将 `slot_mapping` 替换为 `paged_slot_mapping_algorithm`
- **风险**：nano-vllm L150-162 的 slot_mapping 是 contiguous block 迭代（`block_table[i] * block_size` 连续区间），而蓝图的 paged 公式是 per-token `block_table[0, i//256] * 256 + (i%256)`。二者数据结构完全不同（list[int] vs Tensor[1,max_blocks] int32），**重建者需要独立编写集成代码**，蓝图未能直接拼接

#### OW-2：BlockManager TP 降级的分叉接口未定义

- **JSON Path**: `framework_layer.components[2]._nano_vllm_override`
- **内容**：蓝图说 "allocate/free 改为 no-op"，但未指定分叉方式（继承/条件分支/替换）
- **风险**：ref_code `engine/block_manager.py` 有完整的 prefix caching 实现。重建者可能直接抄入完整 BlockManager 而未做 TP 降级

#### OW-3：`improvement_plan.md §P0` 描述旧版方案，与蓝图新版方案矛盾

- **见 FG-4**

#### OW-4：BlockManager ref_code 是文件级引用

- **JSON Path**: `framework_layer.components[2].ref_code`
- **风险**：`ref_projects/nano-vllm/nanovllm/engine/block_manager.py` 是完整文件（156 行），蓝图虽然提供 `api_spec` 定义了 6 个方法签名，但 nano-vllm 实际实现有 prefix caching 等额外逻辑，重建者可能过度抄入

#### OW-5：CustomAR register_buffer 两种 IPC exchange 方式缺少解释

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime.collectives.custom_ar_all_reduce.constraint.init_state_machine`
- **风险**：meta_ptrs exchange 使用 `dist.all_gather_object`，buf_ptrs exchange 使用 `dist.broadcast_object_list`。两种 collective 模式对 gloo group 的同步要求不同，蓝图未解释为何需要两种方式以及混用的后果

#### OW-6：Sequence block_table 双轨制切换逻辑不完整

- **JSON Path**: `framework_layer.data_flow_contracts.request_level.sequence_fields.block_table_dual_repr`
- **风险**：蓝图定义了 `block_table_list()` 和 `block_table_tensor()` 两个方法，但未定义以什么条件切换（通过 `inference_backend` 传入构造函数，还是外部判断调用哪个方法），缺少开关逻辑

---

## 三、维度二：引路信噪比评估 (Pointer Precision Check)

### 🟢 总体评价：精确定位为主

| 引路精度 | 占比 | 示例 |
|---------|------|------|
| 精确到类/函数/行号 | ~80% | `ref_projects/nano-vllm/nanovllm/engine/scheduler.py`, `vllm/_custom_ops.py:420-423`, `vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda` |
| 精确到文件 | ~15% | `ref_projects/nano-vllm/nanovllm/engine/block_manager.py` |
| 模糊/缺失 | ~5% | sglang 路径（nocompile 范围不依赖，无影响） |

---

## 四、维度三：缝合代码与维度盲区 (Glue Logic & Shape Blind Spots)

### 4.1 Tensor Shape 推导完备性

蓝图的 Shape 推导链对 **nocompile B=1 TP=4** 路径是自包含且可自验证的：

- Qwen3-8B: `hidden=4096` → TP4 per-rank `hidden=1024`, `q_size=32*128/4=1024`, `kv_size=8*128/4=256`, `qkv_weight=[1024+2*256=1536, 4096]`, `gate_up_weight=[2*intermediate/4, 4096]`
- QwenAttentionTP: `num_heads=32//4=8`, `num_kv_heads_local=max(1,8//4)=2`, `head_dim=128`
- `block_table=[1, max_blocks] int32`, `_key_cache=[num_blocks, 256, num_kv_heads, head_dim] bf16`
- `slot_mapping` 向量化公式: `block_table[0, indices//256] * 256 + (indices%256)`

### 4.2 自研黑盒算子伪代码完备性

- CustomAR init（gloo group → allocate_shared_buffer_and_handle → all_gather_object → init_custom_ar → register_buffer）：**完整伪代码**，含两个 buffer 集合分配细节
- all_reduce_sum custom_op 注册：**完整模板**（`@torch.library.custom_op` + `register_fake`）
- fused_add_rms_norm 全 4 处 weight 均为 `self.xxx.weight`：**明确约束**

### 🔴 Fatal Gaps

#### FG-1：Qwen3-8B 维度练习值与真实 config.json 不一致

- **JSON Path**: `model_layer.architecture_knowledge_base.qwen_series_dense.qwen3_8b_model_dims`

| 参数 | 蓝图值 | 实际 config.json | 偏差 |
|------|--------|-----------------|------|
| `max_position_embeddings` | **32768** | **40960** | -25% |
| `intermediate_size` | **12800** | **12288** | +4.2% |

- **影响**：
  - `max_blocks = max_position_embeddings // 256`：蓝图算得 128，实际应为 160。KV block 数量系统性低估 25%，导致长序列场景 block 耗尽或 OOM
  - `gate_up_proj weight shape = [2*intermediate_size/tp, hidden_size]`：蓝图算得 `[6400, 4096]`，实际应为 `[2*12288/4=6144, 4096]`。safetensors 加载时 shape mismatch
- **注意**：蓝图有 `_note: "严禁硬编码。必须动态读取 config.json。"`，但在 "源码销毁" 假设下示例值是重建者唯一的数值参考。信任示例值将导致上述错误

#### FG-2：`_estimate_kv_blocks` 的 Dense 公式在 ref_code 中不存在

- **JSON Path**: `framework_layer.components[5].full_api_surface.__init__._estimate_kv_blocks`
- **蓝图声称**："若 Dense 用 K+V 公式: `layers*kv_heads*head_dim*2*elem_size*block_size`"
- **实际 ref_code**：`engine/memory_pool.py::KVMemoryPool.estimate_num_blocks()` **仅实现了 MLA 公式**（调用 `hf_deepseek_v2_kv_bytes_per_block`），无 Dense 公式代码路径
- **死锁理由**：对 Qwen3 Dense 模型，调用 `estimate_num_blocks` 会进入 MLA 公式路径（依赖 `kv_lora_rank`, `qk_nope_head_dim` 等 DeepSeek 特有字段），得到错误结果。蓝图给的一行 K+V 公式仅是数学公式，缺少完整实现（free_bytes 获取、边界处理、reserve_bytes 扣减）。**重建者需自行实现整个 Dense 预算路径**

#### FG-3：`improvement_plan.md §P0` 与蓝图描述矛盾

- **JSON Path**: `framework_layer.data_flow_contracts.paged_kv_cache_contract.source_refs[0]`
- **蓝图引用**: `notebooks-cn/07_improvementPlan/improvement_plan.md §P0`
- **ref_doc 实际内容**: 描述 HF `past_key_values` 增量解码方案（"修改 QwenForCausalLMTP.forward: 接受 past_key_values 参数, use_cache=True"），使用 contiguous KV buffer
- **蓝图实际内容**: 描述 paged KV cache + `flash_attn_with_kvcache`，分页 buffer `[num_blocks, 256, heads, dim]`
- **死锁理由**：重建者打开 ref_doc 会看到**过时的实现方案**。contiguous buffer + HF past_key_values 与分页 buffer + flash_attn_with_kvcache 在架构上不兼容。蓝图虽有 "以蓝图为准" 的 policy，但重建者无法确认哪个是正确的——ref_doc 中的具体代码比蓝图的 pseudocode 更有 "可信度" 错觉

#### FG-4：Scheduler-LLMEngine 桥接缺少 `step()` 完整方法体

- **JSON Path**: `framework_layer.data_flow_contracts.scheduler_tp_runner_bridge` + `framework_layer.components[5].full_api_surface.step`
- **问题**：蓝图定义了 `scheduler._block_size`、`scheduler._max_blocks` 注入点，但：
  1. `_max_blocks` 注入代码仅为注释（`# in LLMEngine.__init__: self.scheduler._max_blocks = ...`）而非完整方法体
  2. `LLMEngine.step()` 仅有 5 行文字描述，缺少完整伪代码。而 step() 是 OpenAI server 的唯一调用入口
  3. `num_free_blocks` 来源在 step() 中的路由仅文字描述，无具体条件判断代码
- **死锁理由**：generate() 和 step() 内部逻辑不同——step() 需要返回 finished 列表，generate() 是 while-loop 直到 all_finished。`num_free` 参数在两种接口中的传入路径不同。缺少 step() 完整方法体意味着 OpenAI server TP 同步将不可靠

---

## 五、量化评分

### 🟢 Reconstructability Score: **74%**

| 维度 | 得分 | 说明 |
|------|------|------|
| pseudocode 覆盖率 | 85% | QwenAttentionTP/QwenDecoderLayerTP/QwenForCausalLMTP prefill+decode 完整 |
| ref_code 精确度 | 80% | 大部分精确到行号/函数 |
| ref_docs 有效覆盖率 | 60% | 3/5 不含实现细节，1/5 内容矛盾 |
| 维度参数正确率 | 60% | 2 个关键参数错误 |
| 致命漏洞节点数 | 4 | FG-1~FG-4 |

### 🔴 阻碍达到 100% 的 Top 3 阻断因素

1. **FG-1: 维度参数练习值错误** — `max_position_embeddings` 32768→40960 导致 max_blocks=128→160（-25%），KV 预算系统性低估；`intermediate_size` 12800→12288 导致 gate_up_proj weight shape 错误
2. **FG-2: Dense KV 预算公式在 ref_code 缺失** — `KVMemoryPool.estimate_num_blocks` 仅有 MLA 公式，Qwen3 Dense 模型无法正确估算 block 数量
3. **FG-3: ref_doc 与蓝图矛盾** — `improvement_plan.md §P0` 描述的是 HF past_key_values 方案，而蓝图是 paged KV cache + flash_attn_with_kvcache 方案，重建者将获得矛盾信号

---

## 六、附录：完整发现清单

### 🔴 Fatal Gaps (4 项)

| ID | 摘要 | JSON Path |
|----|------|----------|
| FG-1 | Qwen3-8B 维度练习值错误（max_position_embeddings, intermediate_size） | `model_layer.architecture_knowledge_base.qwen_series_dense.qwen3_8b_model_dims` |
| FG-2 | Dense KV 预算公式在 ref_code 中缺失 | `framework_layer.components[5].full_api_surface.__init__._estimate_kv_blocks` |
| FG-3 | improvement_plan.md §P0 与蓝图架构矛盾 | `framework_layer.data_flow_contracts.paged_kv_cache_contract.source_refs[0]` |
| FG-4 | LLMEngine.step() 缺完整方法体 | `framework_layer.components[5].full_api_surface.step` |

### 🟡 Override Warnings (6 项)

| ID | 摘要 | JSON Path |
|----|------|----------|
| OW-1 | prepare_prefill slot_mapping 保留/重写边界模糊 | `framework_layer.components[3]._nano_vllm_per_function` |
| OW-2 | BlockManager TP 降级分叉接口未定义 | `framework_layer.components[2]._nano_vllm_override` |
| OW-3 | → 归入 FG-3 | — |
| OW-4 | BlockManager ref_code 文件级泛引用，需 api_spec 补齐 | `framework_layer.components[2].ref_code` |
| OW-5 | CustomAR register_buffer 两种 IPC exchange 方式缺少解释 | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime.collectives.custom_ar_all_reduce` |
| OW-6 | Sequence block_table 双轨制切换逻辑不完整 | `framework_layer.data_flow_contracts.request_level.sequence_fields.block_table_dual_repr` |
