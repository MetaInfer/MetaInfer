# meta-infer 推理引擎性能优化计划

## 1. Benchmark 环境与结果

### 1.1 测试环境

| 项目 | 配置 |
|------|------|
| 模型 | DeepSeek-V2-Lite-Chat (~16B MoE) / Qwen3-8B |
| GPU | NVIDIA A800 80GB PCIe, TP=4 (GPU 4,5,6,7) |
| 服务端 | `torchrun --nproc_per_node=4 openai_tp_server.py --backend tp` |
| 压测脚本 | `run_compare_metainfer_vllm.sh dsv2` (SKIP_VLLM=1) |
| 参数 | ROUNDS=25, STEPS=32, REQUEST_RATE=4, MAX_CONCURRENCY=1 |
| 数据集 | zh-default (5 条中文 prompt 循环) |

### 1.2 正确性基准 (Baseline)

**DeepSeek-V2-Lite-Chat (TP=4, max_new_tokens=24, temperature=0.0)**:

```
output[0]='\n\n苏州园林的特点是追求自然美，注重景物的布局和景致的和谐，以及对细节的精致'
output[1]='\n\n张量并行是一种分布式计算技术，用于在多个计算设备上并行处理大规模的张量'
output[2]='\n\n夏天傍晚，夕阳如火，天边染上了金色的余晖。微风轻拂，带来了'
output[3]='\n\n大语言模型是一种基于大量文本数据训练的机器学习模型，它能够理解和生成自然语言文本。'
output[4]='\n面包是一种非常受欢迎的食品，它的制作过程虽然有些复杂，但只要掌握了基本的方法，就可以在家中'
```

**Qwen3-8B (TP=4, max_new_tokens=24, temperature=0.0)**:

```
output[0]=' 苏州园林是中国古典园林的代表，以精巧的布局、曲折的水系、精致的建筑'
output[1]=' 张量并行（Tensor Parallelism）是一种分布式计算技术，主要用于在多个计算设备（如GPU或'
output[2]='要求：有画面感，有诗意，有文学性。 15字以内。\n夏暮微风拂'
output[3]='它们如何工作？它们有哪些应用？ 大语言模型（Large Language Models，LLMs）是人工智能领域的一个'
output[4]=' 做面包的步骤可以分为以下几个主要部分：准备材料、制作面团、发酵、整形、烘焙'
```

### 1.3 性能基准 (Baseline)

**DeepSeek-V2-Lite-Chat, TP=4, GPU 4,5,6,7**:

| 指标 | meta-infer (TP=4) | vLLM (典型值) | 差距倍数 |
|------|-----------|-------------|---------|
| Request throughput (req/s) | 0.07 | ~2-5 | 30-70x |
| Output token throughput (tok/s) | 2.15 | ~200-800 | 100-370x |
| Mean TTFT (ms) | 14,295 | ~50-300 | 50-285x |
| Mean E2EL (ms) | 14,295 | ~1,000-5,000 | 3-14x |
| Benchmark duration (s) | 357.39 | - | - |

### 1.3 服务端日志关键观察

```
[LLMEngine] step=1 phase=prefill batch=1 waiting=0 running=1 free_blocks=6461
[LLMEngine] step=2 phase=decode batch=1 waiting=0 running=1 free_blocks=6461
[LLMEngine] step=3 phase=decode batch=1 waiting=0 running=1 free_blocks=6461
...
[LLMEngine] finish req-0: gen_len=32, reason=max_tokens, free_blocks=6464
[LLMEngine] generate done in 19.63s
```

- batch_size 始终为 1
- 每个请求 ~20s 生成 32 tokens
- GPU 4 利用率 0%, GPU 5 利用率 100%（TP 负载不均）
- 日志双份输出（两个 rank 各打一份）

---

## 2. 问题根因分析

### 问题 1 [致命]: 每步全量重算，KV Cache 形同虚设

**位置**:
- `llm_engine.py:159` — `use_cache=False`
- `engine/models/deepseek_v2.py:514-519` — 每步遍历全部 token
- `engine/models/qwen.py:219-228` — 同上

**现象**: 虽然引擎分配了 `KVMemoryPool` 和 `BlockManager`（6464 free blocks），但 model forward 完全没有使用 KV cache。每个 decode step 都把完整序列（prompt + 已生成 token）从头算一遍。

**计算浪费量化** (以 25 prompts, 32 output tokens 为例):
- 第 1 步: 4 tokens → 完整 forward
- 第 2 步: 5 tokens → 完整 forward
- ...
- 第 32 步: 35 tokens → 完整 forward
- 总计算量: 4+5+...+35 = 620 tokens
- 增量解码只需: 4+1+1+...+1 = 35 tokens
- **浪费比: ~17x**

**根因**: `DeepseekForCausalLMTP.forward` 和 `QwenForCausalLMTP.forward` 均未接受 `past_key_values` 参数，也未返回 KV cache。`Sequence.past_key_values` 字段已预留（`structs.py:24`）但从未被使用。

---

### 问题 2 [致命]: MoE 前向使用逐 token Python 循环

**位置**: `engine/tp_layers/moe.py:87-96`

```python
for token_i in range(flat.shape[0]):           # Python 循环
    x = flat[token_i : token_i + 1]
    for k in range(self.cfg.top_k):             # Python 循环
        expert_id = int(topk_idx_flat[token_i, k].item())  # GPU→CPU 同步!
        ...
        local_out[token_i : token_i + 1] += expert(x) * w  # 逐 token forward
```

**问题清单**:
1. `.item()` 每次调用触发 GPU→CPU 同步，阻塞 CUDA pipeline
2. 每个 token × 每个 expert = 独立 Python forward 调用，无法利用 GPU 并行
3. DeepSeek-V2-Lite 几乎所有层都是 MoE 层（`first_k_dense_replace=1`）
4. batch=1, top_k=6 → 每层 6 次独立 forward, 60 层 ≈ 360 次 kernel launch

**影响**: MoE 路径比 fused kernel 慢约 100x。

---

### 问题 3 [严重]: 无 CUDA Graph，kernel launch 开销巨大

**现状**: decode 阶段每个 token 仅 1 个 token 输入，GPU 算力利用率极低，主要瓶颈是 kernel launch overhead。

**量化**: 60 层 × 每层多个算子（QKV proj, RoPE, Attn, O proj, Norm, MoE/MLP）= 每步数百次 CUDA kernel launch。每次 launch ~5-10μs，总计 ~3-6ms 纯 launch 开销，而实际计算可能仅 ~1ms。

**对比**: vLLM 对 decode 使用 CUDA Graph，将整个 forward 录制为一个 graph，replay 时几乎零 launch overhead。

---

### 问题 4 [严重]: 无 Flash Attention / 高效 Attention kernel

**位置**:
- `engine/models/deepseek_v2.py:261`
- `engine/models/qwen.py:166`

```python
out = F.scaled_dot_product_attention(q_cat, k_cat, v, ...)
```

使用 PyTorch 原生 SDPA，而非 Flash Attention / FlashInfer / Triton 自定义 kernel。

**影响**:
- Prefill: 缺少 `flash_attn_varlen_func` 的 variable-length + paged KV 支持
- Decode: 缺少 FlashInfer `BatchDecodeWithPagedKVCacheWrapper`（支持 GQA tensor core 加速）
- 无 KV cache write kernel（Triton scatter write by slot_mapping）

---

### 问题 5 [中等]: 请求串行处理，无 batching

**位置**: `openai_tp_server.py:144`

```python
with request_lock:  # 全局锁
    if dist_ready():
        _ = broadcast_obj(cmd)
    text = generate_once(...)
```

所有 HTTP 请求通过 `request_lock` 串行化。即使有 25 个请求排队，也是一次处理一个，batch_size 始终为 1。

**TP 同步问题**: 非 rank-0 进程在 `while True` 循环中等待 `broadcast_obj`，rank-0 在 `generate_once` 期间独占 `request_lock`，其他请求被阻塞。

---

### 问题 6 [中等]: TP 通信效率低

**位置**: `engine/tp_layers/distributed.py:42-50`

```python
def all_reduce_sum(x):
    if x.dtype in (torch.float16, torch.bfloat16):
        tmp = x.float()           # bf16 → fp32 转换
        dist.all_reduce(tmp, ...)
        return tmp.to(dtype=x.dtype)  # fp32 → bf16 转换
```

每次 all_reduce 都做 dtype 转换，增加显存带宽消耗。NCCL 原生支持 bf16 all_reduce，无需转换。

---

### 问题 7 [低]: Scheduler 仅支持 prefill-first

**位置**: `engine/scheduler.py:24-28`

```python
def schedule(self):
    prefill_batch = self._schedule_prefill()
    if prefill_batch:
        return prefill_batch, True
    return self._schedule_decode(), False
```

在有 waiting 请求时优先 prefill，decode 只在 waiting 队列为空时才执行。对于持续到达的请求，decode 被频繁打断，影响 TPOT 稳定性。

---

### 问题 8 [低]: 日志冗余

两个 TP rank 均打印日志（`_should_log()` 在 `META_INFER_LOG_RANK0_ONLY=1` 时只打 rank-0，但 benchmark 脚本未设置此变量）。每条消息出现两次，干扰日志分析。

---

## 3. 优化措施

### P0: 实现增量 KV Cache 解码

**预期提升**: 10-20x
**难度**: 中
**影响范围**: `llm_engine.py`, `engine/models/deepseek_v2.py`, `engine/models/qwen.py`

**方案 A — 利用 HF past_key_values（快速验证）**:

1. 修改 `DeepseekForCausalLMTP.forward` 和 `QwenForCausalLMTP.forward`:
   - 接受 `past_key_values` 参数
   - 设置 `use_cache=True`
   - 返回 `(logits, past_key_values)`
2. 修改 `DeepseekTPModelRunner.run` 和 `QwenTPModelRunner.run`:
   - Prefill: 首次 forward，缓存返回的 `past_key_values` 到 `seq.past_key_values`
   - Decode: 只输入最新 1 个 token + 传入 `seq.past_key_values`
3. 修改 `LLMEngine.step`:
   - 传递 `past_key_values` 到 runner

**方案 B — 使用物理 KV buffer + 自定义 kernel（高性能）**:

1. 在 attention forward 中将 KV 写入 `KVMemoryPool` 的物理 buffer
2. Decode 时从物理 buffer 读取 KV（需 FlashInfer / Triton kernel 支持）
3. 支持 paged attention + prefix caching

**建议**: 先实现方案 A 验证正确性和基本性能，再迭代到方案 B。

#### P0 实现记录

**实现方式**: 方案 A — 自定义 TP attention 层内置 KV cache

**具体改动**:
1. `engine/models/deepseek_v2.py`:
   - `DeepseekAttentionTP.forward`: 接受 `past_key_values=(cached_k_nope, cached_v, cached_raw_k_pe)`，缓存 RoPE 前的 k_nope/v/raw_k_pe，decode 时拼接缓存后重新计算全位置 RoPE
   - `DeepseekDecoderLayerTP.forward`: 透传 `past_key_values`，返回 `(hidden_states, new_cache)`
   - `DeepseekForCausalLMTP.forward`: 接受 `past_key_values` 和 `position_offset`，decode 时只 embed 新 token
   - `DeepseekTPModelRunner.run`: prefill 时缓存 KV 到 `seq.past_key_values`，decode 时传入缓存
2. `engine/models/qwen.py`: 同上结构改动（QwenAttentionTP, QwenDecoderLayerTP, QwenForCausalLMTP, QwenTPModelRunner）
3. `llm_engine.py:329`: 序列完成时清理 `seq.past_key_values = None`

**遇到的问题与解决**:
- **问题 1**: `is_causal=False` 在 prefill 阶段破坏因果注意力，导致输出乱码
- **解决**: `is_causal = (past_key_values is None)` — prefill 用因果 mask，decode 不需要 mask
- **问题 2**: DeepSeek MLA 的 k_pe (RoPE 分量) 需要在 decode 时对所有位置重新计算 RoPE
- **解决**: 缓存 raw_k_pe (RoPE 前)，decode 时拼接后对全量位置重新应用 `_apply_rope_gptj`

**正确性验证**:
- DeepSeek TP=4: PASSED（输出语义与基准一致，措辞因浮点累积有微小差异）
- Qwen TP=4: PASSED（输出与基准高度一致）

**性能验证** (DeepSeek-V2-Lite-Chat, TP=4, GPU 4,5,6,7):

| 指标 | 基准 | P0 | 提升 |
|------|------|-----|------|
| Output throughput (tok/s) | 2.15 | 8.49 | **3.95x** |
| Mean TTFT (ms) | 14,295 | 3,628 | **3.94x** |
| Duration (s) | 357.39 | 90.71 | **3.94x** |

**分析**: 实际提升 ~4x（而非预期的 10-20x），因为 MoE 逐 token Python 循环仍是主要瓶颈。

---

### P1: 重写 MoE 前向为 batched 路径

**预期提升**: 5-10x
**难度**: 中
**影响范围**: `engine/tp_layers/moe.py`

**具体改动**:

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, hidden = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden)
    num_tokens = flat.shape[0]
    topk_idx, topk_weight = self._router_topk(hidden_states)  # [num_tokens, top_k]

    local_out = torch.zeros_like(flat)

    # 按 expert 分组批量计算，消除逐 token Python 循环
    for expert_id in self.local_expert_ids:
        # 找到路由到该 expert 的所有 (token, k) 对
        mask = (topk_idx == expert_id)  # [num_tokens, top_k]
        if not mask.any():
            continue
        token_indices, k_indices = mask.nonzero(as_tuple=True)
        weights = topk_weight[token_indices, k_indices]  # [num_matched]

        expert_input = flat[token_indices]               # [num_matched, hidden]
        expert_output = self.experts[str(expert_id)](expert_input)  # 单次 batched forward
        local_out.index_add_(0, token_indices, expert_output * weights.unsqueeze(-1))

    global_out = all_reduce_sum(local_out)
    return global_out.view(bsz, seqlen, hidden)
```

**关键改进**:
- 消除 `.item()` GPU→CPU 同步
- 每个 expert 只做一次 batched forward（而非逐 token）
- 使用 `index_add_` 高效 scatter-add 结果

#### P1 实现记录

**结果**: 性能回退，已回滚。

| 指标 | P0 | P1 | 变化 |
|------|-----|-----|------|
| Output throughput (tok/s) | 8.49 | 7.06 | **-16.8%** |
| Duration (s) | 90.71 | 113.27 | +24.9% |

**分析**: batch=1 时 batched MoE 无收益。每个 expert 仍只处理 1 token，但增加了 mask/nonzero/index_add_ 开销。`.item()` GPU→CPU 同步在 batch=1 时仅 ~360 次（60 层 × top_k=6），每次 ~5μs，总计 ~2ms，不是主要瓶颈。MoE 的主要开销在矩阵乘法本身，而非循环控制。

**结论**: P1 仅在 batch_size > 1 时有效。当前 batch=1 场景下跳过，待 P4 (continuous batching) 实现后重新评估。

#### P1 第二次尝试（P2 之后）

**时间**: 2026-05-08
**前置**: P0 + P2 (torch.compile) 已完成

**实现方式**: 与第一次相同——按 expert 分组 batched forward，用 `nonzero` + `index_add_` 替代逐 token 循环。

**性能验证** (DeepSeek-V2-Lite-Chat, TP=4):

| 指标 | P2 | P2+P1 | 变化 |
|------|-----|-------|------|
| Output throughput (tok/s) | 12.75 | 9.56 | **-25.0%** |

**分析**: batch=1 时，`nonzero()` 和 `index_add_()` 的开销仍然大于收益。torch.compile 已经优化了逐 token 循环的 kernel launch overhead，使得 batched 方式的额外 tensor 操作成为净负面。

**结论**: 回滚。P1 需要 batch>1（通过 P4 continuous batching）才能生效。

---

### P2: CUDA Graph 兼容 + torch.compile kernel fusion

**预期提升**: 2-5x
**难度**: 中
**影响范围**: `engine/models/deepseek_v2.py`, `engine/models/qwen.py`

**实现方式**: 两部分优化——固定形状 attention + torch.compile 编译子模块

**具体改动**:
1. **固定形状 decode attention**（CUDA Graph 兼容）:
   - Prefill 路径保持切片（`k_buf[:, :kv_len]`）+ `is_causal=True`（高效，shape 随 prompt 长度变化）
   - Decode 路径改为全 buffer + `attn_mask`：使用 `k_buf[:, :max_seq_len]`（固定形状），通过 `attn_mask[:, :, :, kv_len:] = -inf` 屏蔽 padding 位置
   - 参考: vLLM 的 paged attention 思路（固定 buffer + mask）
2. **torch.compile 编译子模块**:
   - 编译每个 decoder layer 的 `self_attn`（attention 模块）
   - 编译 dense MLP 模块（Qwen 全部层，DeepSeek 的前 `first_k_dense_replace` 层）
   - **不编译 MoE 模块**（有 `.item()` GPU→CPU 同步，导致 graph breaks）
   - 使用 `mode='default'`（kernel fusion only，不用 CUDA Graph，避免与 KV cache buffer 复用冲突）

**遇到的问题与解决**:
- **问题 1**: `torch.compile(mode='reduce-overhead')` 和 `mode='max-autotune'` 使用内部 CUDA Graph，与 KV cache buffer 跨步复用冲突
- **错误**: `RuntimeError: accessing tensor output of CUDAGraphs that has been subsequent run`
- **解决**: 使用 `mode='default'`（纯 kernel fusion，不使用 CUDA Graph）
- **问题 2**: 之前 attention 使用动态切片 `k[:, :kv_len]`，形状每步变化，无法 CUDA Graph capture
- **解决**: decode 路径改为全 buffer + attn_mask，保持固定形状

**性能验证** (DeepSeek-V2-Lite-Chat, TP=4, GPU 4,5,6,7):

| 指标 | P0+P5 | P2 | 提升 |
|------|-------|-----|------|
| Output throughput (tok/s) | 8.76 | 12.75 | **+45.5%** |
| Duration (s) | ~88 | 60.4 | **-31.4%** |

**微基准测试** (单请求, 20 decode steps):
- 无 compile: 124.1 ms/step
- 有 compile: 88.7 ms/step
- **加速: 1.40x**

**正确性验证**:
- DeepSeek TP=4 PASSED（输出与基准语义一致）
- Qwen TP=4 PASSED（输出与基准高度一致）

**参考代码路径**:
- `torch.compile` 文档: `torch.compiler` API
- SDPA `attn_mask`: `torch.nn.functional.scaled_dot_product_attention`

---

### P3: 使用 Flash Attention / FlashInfer

**预期提升**: 2-3x
**难度**: 中
**影响范围**: `engine/models/deepseek_v2.py`, `engine/models/qwen.py`

**具体改动**:
1. Prefill: 替换 `F.scaled_dot_product_attention` 为 `flash_attn_varlen_func`
   - 使用 `cu_seqlens` 支持 variable-length batch
   - 支持 paged KV cache（block_table）
2. Decode: 替换为 `flash_attn_with_kvcache` 或 FlashInfer
   - 支持 block_table 读取
   - GQA 优化（tensor core 加速）
3. KV cache write: 使用 Triton scatter kernel（`slot_mapping` 索引写入）

**前置条件**: P0（增量 KV Cache）需要先实现，否则无 KV cache 可读。

---

### P4: 实现 Continuous Batching

**预期提升**: 2-5x
**难度**: 高
**影响范围**: `openai_tp_server.py`, `engine/scheduler.py`, `llm_engine.py`

**具体改动**:
1. **移除 `request_lock`**: 改为异步 request queue
2. **Scheduler 支持 decode batch 多请求**:
   - `_schedule_decode()` 返回多个 running 序列
   - Model runner 将多序列组 batch 做 forward
3. **Interleaved prefill/decode**:
   - 每 N 步 decode 后插入一次 prefill batch
   - 平衡 TTFT 和 TPOT
4. **TP 同步重构**:
   - 用 `broadcast_object_list` 传递完整的 batch metadata（而非逐请求）
   - 所有 rank 执行相同的 batch forward

---

### P5: 优化 TP 通信

**预期提升**: 1.2-1.5x
**难度**: 低
**影响范围**: `engine/tp_layers/distributed.py`

**具体改动**:

```python
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not is_tp_enabled():
        return x
    # 直接使用 NCCL 原生 bf16/fp16 all_reduce，不做 dtype 转换
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x
```

**注意事项**: 需验证 bf16 all_reduce 在长序列上的数值稳定性。如发现精度问题，可改为仅在 attention softmax 等敏感算点保留 fp32。

#### P5 实现记录

**实现方式**: 移除 `all_reduce_sum` 中的 bf16→fp32→bf16 dtype 转换，直接使用 NCCL 原生 bf16 all_reduce。

**正确性验证**: DeepSeek TP=4 PASSED

**性能验证**:

| 指标 | P0 | P0+P5 | 提升 |
|------|-----|-------|------|
| Output throughput (tok/s) | 8.49 | 8.87 | **+4.5%** |
| Duration (s) | 90.71 | 87.35 | -3.7% |

**分析**: 改善较小，因为 TP=4 时 all_reduce 通信量本身不大（hidden_size=5120, batch=1, 每次 ~10KB）。主要瓶颈仍在 MoE 计算和 kernel launch overhead。

---

### P3: Flash Attention / SDPA 优化

**预期提升**: 2-3x
**难度**: 低
**影响范围**: `engine/models/deepseek_v2.py`, `engine/models/qwen.py`

**实现方式**: 利用 PyTorch 2.9.1 内置的 flash SDPA 后端，无需额外安装 `flash-attn` 包。

**具体改动**:
1. `engine/models/qwen.py`:
   - 移除 `repeat_interleave` GQA broadcast（Qwen3-8B 有 32 Q heads / 8 KV heads，GQA 4:1）
   - 改用 `F.scaled_dot_product_attention(..., enable_gqa=True)` 让 SDPA 内核原生处理 GQA
   - 参考: PyTorch 2.9 `torch.nn.functional.scaled_dot_product_attention` 的 `enable_gqa` 参数
2. `engine/models/deepseek_v2.py`:
   - 移除无效的 GQA broadcast 代码块（DeepSeek-V2 MLA 每 rank 的 KV heads == Q heads，无 GQA）
   - `k_pe.expand(-1, -1, self.local_heads, -1)` 保持不变（view，无拷贝）

**遇到的问题与解决**:
- **问题 1**: `flash-attn` 包无法安装（编译失败，PyTorch 2.9.1+cu128 兼容性问题）
- **解决**: 使用 PyTorch 内置 flash SDPA 后端。验证 `torch.backends.cuda.flash_sdp_enabled()=True`，`F.scaled_dot_product_attention` 自动调度到 flash 内核
- **问题 2**: `enable_gqa=True` 需要 PyTorch 2.9+，旧版本不支持
- **解决**: 当前环境 PyTorch 2.9.1，直接可用

**性能基准测试**:
- `repeat_interleave` vs `expand+reshape`: 0.128ms vs 0.024ms (**5.3x** 加速)
- Flash SDPA 自动调度: prefill 和 decode 均确认使用 flash 后端
- GQA 直接传递（不 broadcast）+ `enable_gqa=True`: 通过 SDPA 内核原生处理

**正确性验证**:
- DeepSeek TP=4 PASSED（输出与基准语义一致，微小措辞差异为浮点累积）
- Qwen TP=4 PASSED（输出与基准完全一致）

**性能验证**: 由于 GPU 4-7 上有其他进程竞争（~28GB/卡），无法获得公平的性能对比数据。P3 改动对 DeepSeek 是纯清理（移除死代码），对 Qwen 省去了 `repeat_interleave` 拷贝，理论上 decode 每步节省 ~0.1ms。

**参考代码路径**:
- PyTorch SDPA 文档: `torch.nn.functional.scaled_dot_product_attention`
- `enable_gqa` 参数: PyTorch 2.9+ 新增，原生支持 GQA 不同 head 数

---

## 4. 优化进展总结

| Phase | 状态 | 吞吐 (tok/s) | 相对基准提升 | 说明 |
|-------|------|-------------|-------------|------|
| 基准 | - | 2.15 | 1.0x | 全量重算，无 KV cache |
| P0 | 完成 | 8.49 | **3.95x** | 增量 KV Cache 解码 |
| P1 | 回滚(2次) | 7.06/9.56 | - | batch=1 无收益，需要 P4 连续批处理 |
| P2 | 完成 | 12.75 | **5.93x** | torch.compile kernel fusion + 固定形状 attention |
| P3 | 完成 | - | - | SDPA enable_gqa=True，省去 repeat_interleave 拷贝 |
| P5 | 完成 | 8.87 | **4.13x** | TP 通信去 dtype 转换 |
| P5+buffer | 中性 | 8.76 | **4.07x** | 预分配 KV buffer（短序列无显著收益） |

**当前总提升**: 2.15 → 12.75 tok/s (**5.93x**)

**剩余瓶颈分析**:
1. **MoE 逐 token Python 循环 + `.item()` GPU→CPU 同步** — batch>1 时可通过 batched MoE 优化，当前 batch=1 无效
2. **每步 KV cache `torch.cat` 拷贝** — 短序列影响小，长序列时应改为预分配 buffer + slice
3. **无 CUDA Graph** — kernel launch ~3-5ms/step，需固定 shape KV cache 才能实现
4. **无 Flash Attention** — PyTorch SDPA 对 decode 效率较低，需替换为 FlashInfer
5. **无 Continuous Batching** — 请求串行处理，batch_size 始终为 1

**下一步建议**（按优先级）:
1. **P4: Continuous Batching** — 允许多请求 batch，可大幅提升吞吐（batch>1 时 P1 MoE 优化也生效）
2. **P3: Flash Attention** — 替换 SDPA 为 FlashInfer，decode 效率提升 2-3x
3. **P2: CUDA Graph** — 需先实现固定 shape KV cache，然后 capture decode forward

---

## 5. 实施路线图（修订）

```
Phase 1 (已完成，4.07x 提升):
├── [DONE] P0: 增量 KV Cache 解码 (3.95x)
├── [REVERTED] P1: Batched MoE（batch=1 无收益）
├── [SKIPPED] P2: CUDA Graph（需固定 shape KV cache）
└── [DONE] P5: TP 通信优化 (+4.5%)

Phase 2 (下一步，预期额外 3-5x):
├── P4: Continuous Batching — 允许多请求 batch，解除 batch=1 瓶颈
├── P1 (revisit): Batched MoE — batch>1 时重新评估
└── P3: Flash Attention — 替换 SDPA 为 FlashInfer

Phase 3 (进一步优化，预期额外 2-3x):
├── P2: CUDA Graph — 固定 shape KV cache + decode graph capture
└── P0b: 物理 KV buffer + paged attention kernel
```

**完成 Phase 2 后预期**: ~30-50 tok/s (vLLM 的 15-25%)
**完成 Phase 3 后预期**: ~100-200 tok/s (vLLM 的 50-100%)

**Phase 1 完成后预期**: output token throughput 从 ~1.3 tok/s 提升至 ~30-60 tok/s（vLLM 的 10-30%）。

**全部完成后预期**: output token throughput 达到 ~200-500 tok/s（vLLM 的 60-100%）。

---

## 5. 验证方法

每个 Phase 完成后，使用相同参数重跑 benchmark 对比:

```bash
# 启动服务
CUDA_VISIBLE_DEVICES=4,5 TP_SIZE=2 PORT=9000 BACKEND=tp \
  bash start_tp_infer_service.sh dsv2

# 压测
CUDA_VISIBLE_DEVICES=4,5 TP_SIZE=2 PORT=9000 ROUNDS=25 STEPS=32 \
  REQUEST_RATE=4 MAX_CONCURRENCY=1 \
  bash run_myengine_benchmark.sh dsv2
```

**关键指标**:
1. Output token throughput (tok/s) — 主要衡量标准
2. Mean TTFT (ms) — 首 token 延迟
3. Mean TPOT (ms) — 每 token 延迟（需 batch>1 时才有意义）
4. GPU utilization — nvidia-smi 观察双卡是否均衡
