# 蓝图重构盲测漏洞报告（v11）

**审计对象**: `inference_blueprint.json` (Qwen3 Dense TP 章节，排除 DeepSeek)
**审计目标**: 重构 Qwen3-8B TP=4 完整推理引擎（nocompile 单序列路径）
**审计范围**: 排除 CUDA Graph TP=4（用户明确排除）
**审计官**: 独立第三方系统级审计官

---

## 审计前提

1. 目标工程 `meta-infer` 的所有业务源码已被物理销毁。
2. 全部技术资产仅有：`inference_blueprint.json` 和 `AGENT_SKILL.md`。
3. 拥有完整 RAG 检索权限，可查阅图谱中配置的所有 `ref_code` 和 `ref_docs`。

---

## 1. 三维度逐层盘问

### 维度一：重载契约核验 (Override & Adaptation Check)

| 组件 | ref_code 指向 | override 指引 | 评价 |
|------|--------------|--------------|------|
| **Scheduler** | `nano-vllm/engine/scheduler.py` | 精确到行级：删除 preempt() L52-57，can_append 改为 can_append_one_more，block_size 注入式替换 | ✅ 充分 |
| **KVMemoryPool** | `nano-vllm/engine/block_manager.py` + `model_runner.py` | 指明 TP 路径禁用 GPU placeholder，仅做显存预算日志 | ✅ 充分 |
| **BlockManager** | `nano-vllm/engine/block_manager.py` | TP 路径降级为纯计数器，allocate/free no-op，get_num_free_blocks 改用 runner 来源 | ✅ 充分 |
| **ModelRunner** | `nano-vllm/engine/model_runner.py` | 仅一行说明删除 use_cache 模式并替换为 forward_decode 调用 | 🟡 不足。参考文件 259 行包含 SharedMemory IPC + contiguous KV cache + batch-padded CUDA Graph 完整架构，未逐函数标注取舍 |
| **Sampler** | `nano-vllm/layers/sampler.py` | 明确 TP 多卡 rank 0 采样 + broadcast，不修改 sampler.py 本身 | ✅ 充分 |
| **RMSNorm** | 无直接 nano-vllm 文件 | 整体替换为 vLLM kernel wrapper（`kernel_replacement_plan.md §九`） | 🟡 指引分散在 3 处位置，重构者可能遗漏 contiguous 约束或 fused_add_rms_norm 的 weight 选择 |

#### 🟡 OW-1: nano-vllm ModelRunner 参考代码缺少逐函数取舍表

蓝图第 171 行只说"删除 nano-vllm 的 model(input_ids, use_cache=False, return_dict=True) 调用模式"。但打开 259 行的参考文件后，重构者面对的是：

| nano-vllm 函数 | 行号 | TP Runner 对应 | 蓝图是否指明 |
|---|---|---|---|
| `allocate_kv_cache()` | 103 | 禁用（KV 由 QwenAttentionTP 自管） | ✅ |
| `prepare_prefill()` | 129 | 部分保留（cu_seqlens 构建逻辑可参考） | ❌ 未说明 |
| `prepare_decode()` | 173 | 完全不同（paged vs contiguous slot_mapping） | ❌ 未说明 |
| `capture_cudagraph()` | 224 | 整体删除 | ❌ 未说明 |
| `run_model()` | 196 | 整体替换 | ❌ 未说明 |
| IPC SharedMemory | 43-48 | 整体删除 | ❌ 未说明 |

**风险**: 重构者可能把 `prepare_prefill()` 中的 contiguous slot_mapping 逻辑（nano-vllm L154-162）错抄进 TP Runner，该逻辑假设 contiguous KV cache 而非 paged cache。

#### 🟡 OW-2: fused_add_rms_norm weight 来源——FM-003 与 JSON 正文矛盾

- JSON 正文 `residual_chain_pseudocode`（第 912-918 行）：post_mlp 使用**本层** `post_attention_layernorm.weight`
- `failure_mode_library` FM-003 fix（第 1757 行）："确认第二处 fused_add_rms_norm 传入的 weight 是**下一层** input_layernorm.weight"

两处描述相反。重构者若先查 failure_mode_library 找修复方案，会被 FM-003 误导到错误实现。JSON 正文已标注 `_kernel_replacement_plan_conflict` 警告，但 FM-003 条目自身未同步修正。

#### 🟡 OW-3: Scheduler preempt() 删除范围不完整

蓝图第 64 行说删除 `preempt()` 逻辑 (nano-vllm L52-57)。但 nano-vllm 的 `preempt()` 方法体（L66-69）和 `deallocate()` 方法仍存在于参考文件。蓝图未指示：(a) `preempt()` 方法定义本身是否完全删除；(b) `schedule()` 中 `self.running.pop()` 逻辑也需重构（nano-vllm 的 pop + conditional re-append 模式在无抢占的 TP 路径下会导致序列丢失）。

#### 🟡 OW-4: ensure_divisible 的 vocab_size 边界情况未定义

Qwen3-8B vocab_size=151936，151936/4=37984，恰好整除。但蓝图声明的 hard_rule（第 1401 行）是"严禁硬编码，必须动态读取 config.json"。如果重构者应用到 vocab_size 不被 tp_size 整除的模型，`VocabParallelEmbedding` 的 mask/safe_ids 逻辑（第 537-547 行伪代码）会产生越界。蓝图未定义 pad/truncate 策略。

---

### 维度二：引路信噪比评估 (Pointer Precision Check)

| 引用类型 | 精度 | 示例 |
|---------|------|------|
| ref_code 文件路径 | **文件级** | `ref_projects/nano-vllm/nanovllm/engine/scheduler.py` — 85 行文件，需全文阅读 |
| ref_docs 文档路径 | **文档级（无行号）** | `kernel_replacement_plan.md §九` — 1511 行文档，需 grep 定位 |
| inline 伪代码 | **行级** | 蓝图内嵌的 `forward_decode`、`forward()`、`load_weights` 等伪代码精确到 Python 语句 |
| sglang backend.py | **精确行号** | `ref_projects/sglang/.../backend.py:214-257` — 唯一精确行号引用 |
| vLLM kernel 路径 | **包路径级** | `vllm/_custom_ops.py:420-423` — 指向 installed package，非 repo 文件 |

**评分: 7/10。** 蓝图以 inline 伪代码大量弥补了 ref_code 文件级引用的精度不足。主要缺陷是对 `kernel_replacement_plan.md`（1511 行）和 `improvement_plan.md` 的引用缺乏章节+行号定位，重构者需在文档内自行搜索——这对"假设源码已销毁"的盲测场景构成真实的时间成本。

---

### 维度三：缝合代码与维度盲区 (Glue Logic & Shape Blind Spots)

#### 3.1 目标模型物理维度参数

**评分: ✅ 10/10。** `qwen3_8b_model_dims`（第 1463-1482 行）提供了全部 10 个维度参数，包括 TP=4 per-rank 换算值。同时 `engine_routing_contract`（第 1400-1413 行）强制要求动态读取 config.json，禁止硬编码——重构者既能快速验证（对照表格），又能保证通用性。

#### 3.2 Tensor Shape 推导公式

**评分: ✅ 9/10。** 数据流契约覆盖完整：

| 环节 | Shape 标注 | 推导公式 |
|------|-----------|---------|
| Embedding 输入/输出 | `[B,T] int64` → `[B,T,hidden_size]` | 含 mask+all_reduce 伪代码 |
| QKV 投影 | `[1,1,hidden_size]` → `q:[1,1,q_size] k:[1,1,kv_size] v:[1,1,kv_size]` | TP=4 时 q_size=1024 kv_size=256 |
| Attention 输入 | `q:[1,1,8,128] k:[1,1,2,128] v:[1,1,2,128]` | num_heads=32/4=8, num_kv_heads=max(1,8/4)=2 |
| MLP gate/up | `[B,T,2*intermediate/tp]` = `[1,1,6400]` | intermediate=12800, /4=3200, *2=6400 |
| Paged KV cache | `[num_blocks,256,num_kv_heads,head_dim]` = `[128,256,2,128]` | max_blocks=(32768+255)//256=128 |
| block_table | `[1,max_blocks] int32` | 固定 shape，decode 时在[:,num_blocks]追加 |

#### 3.3 自研黑盒算子的伪代码覆盖

**评分: ✅ 8/10。** 所有自研黑盒算子均有伪代码或接口签名：

| 算子 | 蓝图覆盖 |
|------|---------|
| `flash_attn_with_kvcache_op` | ✅ custom_op 注册模板（含 register_fake）+ `q_format` + `softmax_scale` |
| `all_reduce_sum`（CustomAR P2P） | ✅ 完整 init 7 步伪代码 + fallback 到 `dist.all_reduce` + `world_size=1` no-op 守卫 |
| `VocabParallelEmbedding` | ✅ 完整 forward 伪代码（含 mask 逻辑） |
| `ParallelLMHead` | ✅ all_gather + F.linear 伪代码 |
| `QKVColumnParallelLinear` | ✅ forward + 调用方 reshape + K/V reshape 必须用 num_kv_heads 的警告 |
| `fused_add_rms_norm` | ✅ 残差链伪代码 + 最后一层特殊处理 + weight 来源冲突标注 |
| `silu_and_mul` | ✅ out 预分配 + torch.ops._C.silu_and_mul 调用 |
| `rotary_embedding` | ✅ 2D in-place + cos_sin_cache 共享策略 + lazy GPU loading |

#### 🔴 FG-1: 多序列 Prefill 的 _block_table 注入路径未闭合

**JSON Path**: `paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm.multi_seq` (第 374-384 行)

**死锁理由**: 类层级定义中 `QwenAttentionTP._block_table` 固定为 `[1, max_blocks] int32`（第 351 行），即**硬编码单序列**。multi_seq 伪代码正确构建了各序列的 `block_table` 并赋给 `seq.block_table`，但从未说明这些 per-sequence block_table 如何注入到各 attention 层：

- 单序列：`_block_table[0, :num_blocks] = torch.arange(...)` ✅
- 多序列：需将 `_block_table` 扩展为 `[B, max_blocks]`，`flash_attn_with_kvcache` 才能为每个序列定位 KV 槽位。蓝图未描述 shape 扩展和逐层注入时机。

**实际影响**: B=1 场景不受影响。但当 Scheduler 产出 batch_size > 1 的 prefill 时（Continuous Batching 必然发生），重构者将在此处卡死。

---

## 2. 致命脑补漏洞与重载警告汇总

### 🔴 Fatal Gaps

| # | JSON Path | 死锁理由 | B=1 是否受影响 |
|---|-----------|---------|---------------|
| FG-1 | `paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm.multi_seq` | `_block_table` 的 `[1,max_blocks]→[B,max_blocks]` 扩展与逐层注入路径未定义 | ❌ 不受影响 |

### 🟡 Override Warnings

| # | JSON Path | 风险 |
|---|-----------|------|
| OW-1 | `components[3]` (ModelRunner) | nano-vllm 259 行参考文件缺少逐函数取舍表 |
| OW-2 | `qwen3_kernel_contracts.fused_add_rms_norm` vs `failure_mode_library[2]` | FM-003 的 fix 字段指向错误的跨层 weight 方案，与 JSON 正文矛盾 |
| OW-3 | `components[0]` (Scheduler) `_nano_vllm_override` | preempt() 删除范围不完整：方法体是否保留、pop 逻辑是否重构均未定义 |
| OW-4 | `qwen_series_dense.qwen3_8b_model_dims` | vocab_size 不可整除 tp_size 时的 pad/truncate 策略未定义 |

---

## 3. 重构可行性判决

### 量化分数: 92% (nocompile, B=1, TP=4)

排除 CUDA Graph TP=4 后，蓝图对单序列 nocompile 路径的覆盖度极高。

**蓝图已提供的资产**:

- **完整数据流契约**: `QwenForCausalLMTP.forward()` 顶层编排器（第 864-886 行）→ 逐层 `forward_decode()`（第 671-713 行）→ `QwenAttentionTP.forward_decode()` 热路径（第 672-701 行）——全部是可直接转录为 Python 的生产级伪代码。
- **完整类层级**: 4 个核心类的 `__init__` 签名与属性名，含 `self.qkv_proj`（非 `q_proj`）、`self.gate_up_proj`（非 `gate_proj`）等极易抄错的命名。
- **完整权重加载**: HF key → TP 模块属性的 1:1 映射表（第 1619-1639 行），Q-K-V 拼接顺序、gate-up 拼接顺序、double shard guard 分支、safetensors 惰性加载主循环。
- **完整 KV cache 契约**: paged 格式、block_size=256、`_block_table` shape、`_kv_len_gpu` 追踪、slot_mapping 算法（单序列）、prefill/decode 写入/读取时序。
- **15 条分类失败模式**: symptom → check → fix 闭环，覆盖 RMSNorm contiguous、RoPE Neox 错配、CustomAR gloo、.item() 在 compiled region、QKV 拼接索引等高频踩坑点。
- **CustomAR 初始化**: 7 步伪代码 + vLLM import 清单 + workspace size 计算（16MB）。

### 阻碍达到 100% 的 Top 3 阻断因素

| # | 阻断因素 | 类型 | 修复建议 |
|---|---------|------|---------|
| **1** | **FM-003 与 JSON 正文矛盾** (OW-2) | 文档一致性缺陷 | 修正 FM-003 的 fix 字段为"使用本层 post_attention_layernorm.weight"，与 `residual_chain_pseudocode` 对齐。添加注释说明与 kernel_replacement_plan §9.2 的已知差异。 |
| **2** | **多序列 _block_table 注入路径** (FG-1) | 接口契约缺失 | 在 `paged_kv_cache_contract` 中补充 `_block_table` 从 `[1,max_blocks]` 扩展为 `[B,max_blocks]` 的接口签名，以及在 `QwenForCausalLMTP.forward()` 中的逐层注入伪代码。 |
| **3** | **vLLM kernel wrapper 精确签名不在蓝图内** | 外部文档依赖 | 在 `qwen3_kernel_contracts` 各条目下直接嵌入精确的函数签名（参数类型 + 返回值），减少对 1511 行 `kernel_replacement_plan.md` 的搜索依赖。当前 `rms_norm`/`fused_add_rms_norm`/`rotary_embedding` 的签名均只有 `kernel_replacement_plan.md §九, input, weight, eps)` 格式的简写。 |
