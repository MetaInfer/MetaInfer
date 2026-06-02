# Apple Silicon LLM 推理框架对比调研

> 日期: 2026-06-02
> 对比基准: mac-engine 18.0 tok/s (Qwen3-8B, bf16, Apple M5 Pro 48GB)
> 目的: 找出 mac-engine 缺失的优化手段

---

## 1. mlx-lm (Apple 官方)

### 性能

| 模型 | 平台 | 量化 | 吞吐 (tok/s) | 来源 |
|------|------|------|-------------|------|
| Qwen3-8B | M4 Max 128GB | 4-bit | 79.9 | vllm-mlx 论文 (arXiv:2601.19139) |
| Qwen3-8B | M5 Pro 48GB | bf16 | 17.8 | mac-engine 实测 |
| Qwen3-8B | M4 Pro 24GB | bf16 | ~17.8 | Apple ML Research 博客 |
| Qwen3-14B | M5 24GB | 4-bit | ~23 (推算) | Apple ML Research (TTFT 3.97x, gen 1.24x M4) |

### 关键优化

1. **wired_limit 提升** — 当模型接近内存上限时，自动设置 `mx.set_wired_limit(max_recommended_working_set_size)`，让 Metal 驱动将模型权重 "wired" 到 GPU 常驻内存，避免页面换入换出。这是 mlx-lm 对大模型（接近可用内存上限）的关键优化。
   - 位置: `generate.py` 中的 `wired_limit()` context manager
   - 条件: `model_bytes > 0.9 * max_rec_size` 时触发

2. **async_eval + stream pipeline** — 核心生成循环使用 `mx.async_eval()` + `generation_stream` 实现异步流水线：
   - `generation_stream = mx.new_thread_local_stream(mx.default_device())`
   - 每步 `_step()` 在 generation_stream 上运行 model forward + sample
   - `mx.async_eval(next_y, next_logprobs)` 启动异步求值后立即返回
   - 在 yield/decode 期间，GPU 已经在计算下一个 token

3. **"causal" 字符串 mask** — `create_attention_mask()` 对 decode (N=1) 返回 `None`，对 prefill (N>1) 但无需显式 mask 的情况返回字符串 `"causal"`，让 `mx.fast.scaled_dot_product_attention` 内部使用优化的 causal fast path。这比我们创建显式 float tensor mask 更高效。

4. **KV Cache 量化** — 支持 `QuantizedKVCache`，将 KV cache 压缩为 4-bit/8-bit：
   - `mx.quantize()` / `mx.quantized_matmul()` 原生支持
   - 参数: `--kv-bits 4 --kv-group-size 64`
   - 可从指定 step 开始量化: `--quantized-kv-start 5000`
   - KVCache 有 `.to_quantized()` 方法

5. **RotatingKVCache** — 固定大小的旋转 KV cache，支持无限长文本生成：
   - 使用环形缓冲区，超出 max_size 后覆盖最老的 token
   - 保留前 `keep=4` 个 token（attention sinks）
   - `--max-kv-size` 参数控制

6. **Prompt Caching** — KV cache 可保存到磁盘 (`safetensors`) 并加载复用：
   - `save_prompt_cache()` / `load_prompt_cache()`
   - 支持 LRU 淘汰策略 (`LRUPromptCache`)
   - 支持 Trie 前缀匹配 + cache trimming

7. **BatchGenerator 连续批处理** — 完整的连续批处理系统：
   - `PromptProcessingBatch`: 批量 prefill，支持 right-padding + 动态长度
   - `GenerationBatch`: 批量 decode，`mx.async_eval` 异步采样
   - `BatchGenerator`: 自动管理 prompt → generation 的状态转换
   - 支持 prefill 和 decode 交错进行
   - BatchKVCache / BatchRotatingKVCache 支持变长序列

8. **Speculative Decoding** — 支持投机解码：
   - `speculative_generate_step()`: draft model 生成 N 个候选 token
   - 主模型一次 forward 验证所有候选
   - cache trimming 回退机制
   - `--draft-model` + `--num-draft-tokens` 参数

9. **Chunked Prefill** — 长 prompt 分块处理：
   - `prefill_step_size=2048`，超长 prompt 分块执行
   - 每块完成后 `mx.eval([c.state for c in prompt_cache])` + `mx.clear_cache()`

10. **Distributed Inference** — 支持 `mx.distributed` 多设备推理：
    - `model.shard()` 方法将 linear 层切分
    - `shard_linear()` 支持 "all-to-sharded" / "sharded-to-all" 模式

### 我们缺失的

| 优化 | mlx-lm | mac-engine | 影响 |
|------|--------|------------|------|
| wired_limit | ✅ 自动检测并设置 | ❌ 未使用 | 大模型可能有 10-30% 提升 |
| "causal" 字符串 mask | ✅ SDPA fast path | ❌ 手动创建/跳过 mask | 已通过 L=1 跳过 mask 部分解决 |
| KV Cache 量化 | ✅ 4/8-bit | ❌ 未实现 | 长上下文场景内存节省 2-4x |
| RotatingKVCache | ✅ 环形缓冲 | ❌ 未实现 | 无限长文本场景 |
| Prompt Caching | ✅ 磁盘+LRU | ❌ 未实现 | 多轮对话/重复前缀场景 |
| BatchGenerator | ✅ 连续批处理 | ❌ 仅 round-robin | 并发吞吐提升 2-4x |
| Speculative Decoding | ✅ 内置 | ❌ 未实现 | 延迟优化 |
| async_eval pipeline | ✅ 有效 | ⚠️ 已尝试但无效 | MLX 0.31.2 上可能需 dtype 修复后重试 |

### 参考来源
- GitHub: https://github.com/ml-explore/mlx-lm
- 源码分析: generate.py, models/cache.py, models/base.py, models/qwen3.py
- Apple ML Research: https://machinelearning.apple.com/research/exploring-llms-mlx-m5

---

## 2. llama.cpp (Metal backend)

### 性能

| 模型 | 平台 | 量化 | 吞吐 (tok/s) | 来源 |
|------|------|------|-------------|------|
| Qwen3-8B | M4 Max 128GB | 4-bit (Q4_K_M) | 76.9 | vllm-mlx 论文 |
| Qwen3-0.6B | M4 Max 128GB | 4-bit | 281.5 | vllm-mlx 论文 |
| Llama-2 7B | M2 Max 38-core | Q4_K_S | ~60 | Medium 博客 |

### 关键优化

1. **手工优化的 Metal Kernels** — ggml-metal 包含大量针对 Apple Silicon GPU 的手工优化 Metal Shading Language kernel：
   - 量化矩阵乘法 kernel (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0)
   - 融合 dequantize + matmul — 一个 pass 完成反量化 + 矩阵乘
   - 优化的 batched matmul
   - 针对 batch_size=1 (单请求 decode) 的特殊 kernel

2. **KV Cache 量化** — 支持 Q4_0/Q4_1/Q8_0 等格式的 KV cache 量化：
   - `--cache-type-k q4_0 --cache-type-v q8_0`
   - 4-bit KV cache 节省 4x 内存
   - Q8 KV cache 节省 2x 内存，质量损失极小
   - **内存带宽优化的关键**：decode 时 KV cache 读取也是内存瓶颈的一部分

3. **GGUF 量化格式体系** — 完整的量化等级：
   - Q4_K_M: 4-bit K-quant，per-block scale + min，3.3% 质量损失，75% 大小缩减
   - Q5_K_M: 5-bit，更高精度
   - Q6_K: 6-bit，接近 FP16 质量
   - Q8_0: 8-bit，几乎无损
   - IQ 量化 (imatrix): 基于重要性矩阵的量化

4. **CPU + GPU 混合推理** — 支持将模型层分配到 CPU 和 GPU：
   - `-ngl N` 控制有多少层在 GPU 上运行
   - 当 GPU 内存不足时，溢出到 CPU
   - CPU 侧有优化的 SIMD kernel (NEON/AVX2)

5. **Thread Pinning** — CPU 推理时线程绑定到性能核心：
   - 优化 cache locality
   - 限制 CPU 核心数反而可能更快（避免在 efficiency core 上调度）

6. **Locked/Plane Memory** — 支持 Metal managed buffer 模式优化：
   - 减少内存拷贝
   - 但不如 MLX 的 unified memory 零拷贝

7. **Batched Decoding** — 支持并行 decode 多个序列：
   - batched attention kernel
   - 但缺少 continuous batching 调度器

### 我们缺失的

| 优化 | llama.cpp | mac-engine | 影响 |
|------|-----------|------------|------|
| KV Cache 量化 | ✅ Q4/Q8 | ❌ | decode 阶段 KV 读取带宽优化 |
| 量化权重 | ✅ GGUF 全系列 | ❌ 仅 bf16 | 4-bit 可提升 2-4x 吞吐 |
| 融合 dequantize+matmul | ✅ Metal kernel | ❌ | 单 pass 减少内存读写 |
| CPU+GPU 混合 | ✅ 层切分 | ❌ | 超大模型场景 |

### 参考来源
- GitHub: https://github.com/ggerganov/llama.cpp
- Blog: https://medium.com/@andreask_75652/llama-cpp-performance-apple-silicon-051241dd6eae
- 论文: arXiv:2601.19139 (vllm-mlx 对比数据)

---

## 3. vllm-mlx

> 注: 社区项目 (waybarrios/vllm-mlx)，非官方 vLLM 组织项目，已被 EuroMLSys '26 接收。

### 性能

| 模型 | 平台 | 量化 | 吞吐 (tok/s) | 并发 | 来源 |
|------|------|------|-------------|------|------|
| Qwen3-8B | M4 Max 128GB | 4-bit | 93.3 | 单请求 | arXiv:2601.19139 |
| Qwen3-0.6B | M4 Max 128GB | 4-bit | 525.5 | 单请求 | 同上 |
| Qwen3-8B | M4 Max 128GB | 4-bit | ~120+ | 4 并发 | 4.3x scaling at 16 req |

**vs 其他框架**: 比 llama.cpp 快 21-87%，比 mlx-lm 快 7-21%，比 vllm-metal 快 7-21%（单请求场景）。

### 关键优化

1. **Continuous Batching** — 核心差异化功能：
   - 请求在任意位置加入/离开 batch
   - prefill 和 decode 交错进行
   - 16 并发请求下实现 4.3x 聚合吞吐提升
   - 基于 MLX BatchKVCache

2. **Paged KV Cache** — 类似 vLLM PagedAttention 的 KV cache 管理：
   - 按 page/block 分配 KV cache
   - 避免内存碎片
   - 支持动态增长和收缩

3. **Content-Based Prefix Caching** — 多模态场景的前缀缓存：
   - 通过内容哈希识别相同图片
   - 消除重复 vision encoding
   - 重复图片查询 28x 加速
   - 视频分析 24.7x 加速

4. **SSD-Tiered Cache** — KV cache 持久化到 SSD：
   - 重复前缀的 TTFT 从 30-90s 降到 1-3s
   - 适用于 agent 场景的重复系统 prompt

5. **OpenAI + Anthropic API 兼容** — 生产级服务能力：
   - `/v1/chat/completions` (OpenAI)
   - `/v1/messages` (Anthropic)
   - MCP tool calling 支持
   - 多模态 (text/vision/audio/embeddings)

### 我们缺失的

| 优化 | vllm-mlx | mac-engine | 影响 |
|------|----------|------------|------|
| Continuous Batching | ✅ | ❌ round-robin | 并发吞吐 2-4x |
| Paged KV Cache | ✅ | ❌ 线性增长 | 内存效率 |
| Prefix Caching | ✅ | ❌ | 重复前缀场景 |
| SSD Cache | ✅ | ❌ | Agent 场景 TTFT |

### 参考来源
- GitHub: https://github.com/waybarrios/vllm-mlx
- 论文: arXiv:2601.19139 "Native LLM and MLLM Inference at Scale on Apple Silicon"
- EuroMLSys '26 论文

---

## 4. vllm-metal (官方 vLLM 插件)

### 性能

| 模型 | 平台 | 量化 | 吞吐 (tok/s) | 来源 |
|------|------|------|-------------|------|
| Qwen3-8B | M4 Max 128GB | 4-bit | 87.1 | vllm-mlx 论文 |
| Qwen3-8B | M5 Pro 48GB | bf16 | 17.8 | mac-engine 实测 |

### 关键优化

1. **MLX Backend + vLLM Scheduler** — 桥接 vLLM 调度器到 MLX：
   - 复用 vLLM 的 scheduler/sampler/logits processor
   - MLX 作为计算后端
   - 统一内存零拷贝

2. **PagedAttention (Metal)** — 实验性的 Metal PagedAttention：
   - 高效 KV cache 管理
   - 长序列支持

3. **GQA Support** — Grouped-Query Attention 优化

4. **Rust Frontend (实验)** — `vllm-rs` 替代 Python 服务层：
   - 减少 Python 开销
   - 保持 MLX/Metal 引擎

### 我们缺失的

同 vllm-mlx，但 vllm-metal 更侧重 vLLM 生态集成。

### 参考来源
- 文档: https://docs.vllm.ai/projects/vllm-metal/en/latest/
- GitHub: https://github.com/vllm-project/vllm-metal

---

## 5. MLC-LLM

### 性能

| 模型 | 平台 | 量化 | 吞吐 (tok/s) | 来源 |
|------|------|------|-------------|------|
| Llama-3.2-3B | M2 Ultra | AWQ 4-bit | ~210 | arXiv:2511.05502 |
| 同上 | 同上 | FP16 | ~190 | 同上 |

> 注: MLC-LLM 在长上下文 (64K-128K) 下扩展性最好，但绝对吞吐低于 MLX。

### 关键优化

1. **TVM 编译优化** — 基于 Apache TVM Unity 的编译流程：
   - 算子自动融合 (operator fusion)
   - 智能内存规划 (memory planning)
   - 生成低级 Metal kernel
   - 针对目标硬件的代码生成

2. **Paged KV Cache** — 类似 vLLM 的分页 KV cache：
   - 长上下文场景 (64K-128K tokens) 表现稳定
   - 吞吐约 190 tok/s (比 MLX 低 17%)

3. **多种量化支持** — 社区标准量化格式：
   - AWQ, GPTQ, FP8, 混合精度
   - AWQ 4-bit: 内存更低 (~1.6GB for 3B), 吞吐略高于 FP16

4. **跨平台部署** — 统一编译到多后端：
   - Apple Silicon Metal
   - NVIDIA CUDA
   - AMD ROCm
   - WebGPU (浏览器)

### 我们缺失的

| 优化 | MLC-LLM | mac-engine | 影响 |
|------|---------|------------|------|
| TVM 算子融合 | ✅ | ❌ | 减少 kernel launch 开销 |
| Paged KV Cache | ✅ | ❌ | 长上下文内存效率 |
| 多种量化格式 | ✅ | ❌ | 灵活的精度-性能权衡 |

### 参考来源
- GitHub: https://github.com/mlc-ai/mlc-llm
- 论文: arXiv:2511.05502 "A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp"
- Blog: https://www.latent.space/p/llms-everywhere

---

## 6. 其他重要发现

### 6.1 Apple M5 Neural Accelerators

- M5 GPU 每个核心内集成了 Neural Accelerator，专门优化矩阵乘法
- MLX 通过 Metal 4 TensorOps 框架原生利用这些加速器
- Qwen3-14B-4bit TTFT 加速 4.06x（vs M4），token 生成加速 1.19x
- 需要 macOS 26.2+ 和最新 MLX 版本
- **mac-engine 已在 M5 Pro 上运行，自动受益于此**

### 6.2 内存带宽是核心瓶颈

- LLM decode 阶段是纯内存带宽受限（memory-bandwidth-bound）
- 理论吞吐上限 = 内存带宽 (GB/s) ÷ 模型大小 (GB)
- M5 Pro: ~153 GB/s ÷ 15 GB (Qwen3-8B bf16) ≈ **10.2 tok/s 理论上限**
- mac-engine 实测 18.0 tok/s，**超出理论值**，说明 bf16 实际内存占用可能更小（~8.5 GB，可能因为权重共享/量化存储）
- 4-bit 量化可将模型降到 ~4 GB，理论上限 ~38 tok/s

### 6.3 Rapid-MLX

- 2026-03-23 发布，定位为 Ollama 替代品
- 声称比 Ollama (llama.cpp backend) 快 2-4.2x (M3 Ultra)
- 基于 MLX 构建

### 6.4 oMLX

- 专为 coding agent 场景优化
- SSD 持久化 KV cache，重复前缀 TTFT 从 30-90s 降到 1-3s

### 6.5 LM Studio mlx-engine

- v0.4.2 加入 MLX 连续批处理
- 支持自动切换 MLX/llama.cpp 后端
- 内置 KV cache 量化

---

## 7. mac-engine 缺失优化优先级排序

### 🔴 高优先级（预期收益 > 20%）

| # | 优化 | 来源 | 预期收益 | 实施难度 |
|---|------|------|---------|---------|
| 1 | **4-bit 量化推理** | llama.cpp, mlx-lm | 2-4x 吞吐 | 中（MLX 原生支持量化权重） |
| 2 | **KV Cache 量化 (4/8-bit)** | mlx-lm, llama.cpp | 长上下文 2-4x 内存节省 | 低（mlx.quantize 已有 API） |
| 3 | **wired_limit 设置** | mlx-lm | 10-30%（大模型场景） | 极低（1 行代码） |

### 🟡 中优先级（预期收益 10-20%）

| # | 优化 | 来源 | 预期收益 | 实施难度 |
|---|------|------|---------|---------|
| 4 | **Continuous Batching** | vllm-mlx, mlx-lm BatchGenerator | 并发 2-4x | 高 |
| 5 | **async_eval pipeline 重试** | mlx-lm | 10-20% | 低（dtype 修复后可能生效） |
| 6 | **Prompt Caching** | mlx-lm | 多轮对话场景 | 中 |

### 🟢 低优先级（场景受限或收益 < 10%）

| # | 优化 | 来源 | 预期收益 | 实施难度 |
|---|------|------|---------|---------|
| 7 | **RotatingKVCache** | mlx-lm | 无限长文本 | 低 |
| 8 | **Speculative Decoding** | mlx-lm | 延迟优化 | 高 |
| 9 | **SSD Tiered Cache** | oMLX | Agent 场景 | 高 |
| 10 | **Custom Metal Kernels** | llama.cpp | 20-40% | 极高 |
| 11 | **TVM 编译优化** | MLC-LLM | 不确定 | 极高 |

---

## 8. 汇总对比表

| 优化手段 | llama.cpp | mlx-lm | vllm-mlx | vllm-metal | MLC-LLM | mac-engine | 差距说明 |
|---------|-----------|--------|----------|------------|---------|------------|---------|
| bf16/fp16 推理 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 持平 |
| KV Cache 增量解码 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 持平 |
| Decode 无 mask 优化 | ✅ | ✅ (返回 None/"causal") | ✅ | ✅ | ✅ | ✅ (L=1 跳过) | 已解决 |
| async_eval pipeline | ❌ (C++ 控制) | ✅ | ✅ | ✅ | N/A | ⚠️ (已尝试，无效) | dtype 修复后需重试 |
| mx.compile 采样 | N/A | ✅ | ✅ | ✅ | N/A | ✅ | 持平 |
| **wired_limit** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | **缺失，大模型 10-30%** |
| **4-bit 量化推理** | ✅ GGUF | ✅ MLX | ✅ MLX | ✅ MLX | ✅ AWQ/GPTQ | ❌ | **缺失，2-4x 吞吐** |
| **KV Cache 量化** | ✅ Q4/Q8 | ✅ 4/8-bit | ✅ | ❌ | ✅ | ❌ | **缺失，长上下文关键** |
| RotatingKVCache | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | 缺失，无限长文本 |
| Prompt Caching | ✅ 磁盘 | ✅ 磁盘+LRU | ✅ SSD | ❌ | ❌ | ❌ | 缺失，多轮对话 |
| **Continuous Batching** | ❌ | ✅ BatchGenerator | ✅ | ✅ | ❌ | ❌ (round-robin) | **缺失，并发 2-4x** |
| Paged KV Cache | ❌ | ❌ | ✅ | ✅ 实验性 | ✅ | ❌ | 缺失，内存效率 |
| Speculative Decoding | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | 缺失，延迟优化 |
| Prefix Caching | ❌ | ✅ Trie | ✅ 哈希 | ❌ | ❌ | ❌ | 缺失，重复前缀 |
| SSD Tiered Cache | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 缺失，Agent 场景 |
| Custom Metal Kernels | ✅ ggml-metal | ❌ (用 MLX 内置) | ❌ | ✅ | ✅ TVM | ❌ | 高难度优化 |
| 融合 dequant+matmul | ✅ | ✅ (MLX 内部) | ✅ | ✅ | ✅ TVM | ✅ (MLX 自动) | 持平（MLX 内部优化） |
| Chunked Prefill | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 缺失，长 prompt 场景 |
| Distributed Inference | ❌ | ✅ mx.distributed | ❌ | ❌ | ✅ | ❌ | 非单机场景 |
| M5 Neural Accelerator | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ (MLX 自动) | 持平 |

---

## 9. 建议实施路线

### 阶段 A: 快速收益 (1-2 天)

1. **添加 wired_limit** — 复制 mlx-lm 的 `wired_limit()` context manager，1 行改动
2. **重试 async_eval pipeline** — 在 bf16 dtype 修复后重新测试 stream pipeline 效果
3. **实现 KV Cache 量化** — 使用 `mx.quantize/mx.quantized_matmul`，参考 mlx-lm 的 `QuantizedKVCache`

### 阶段 B: 量化推理 (3-5 天)

4. **4-bit 量化权重加载** — 支持 MLX 格式量化模型
5. **量化 benchmark** — 对比 bf16 vs 4-bit 的吞吐和精度

### 阶段 C: 批处理优化 (1-2 周)

6. **Continuous Batching** — 参考 mlx-lm 的 `BatchGenerator` 或 vllm-mlx 的实现
7. **Prompt Caching** — 磁盘持久化 + Trie 前缀匹配

---

## 附录: 数据来源

1. arXiv:2601.19139 - "Native LLM and MLLM Inference at Scale on Apple Silicon" (vllm-mlx 论文)
2. arXiv:2511.05502 - "A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp on Apple Silicon"
3. Apple ML Research - "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"
4. https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide
5. https://yage.ai/share/mlx-apple-silicon-en-20260331.html
6. mlx-lm 源码: generate.py, models/cache.py, models/base.py, models/qwen3.py
7. mac-engine 实测数据: experiment_baseline.md

---
日期: 2026-06-02
