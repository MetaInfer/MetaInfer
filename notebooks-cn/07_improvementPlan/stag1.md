# meta-infer Stage 1 优化总结

## 测试环境

| 项目 | 配置 |
|------|------|
| 模型 | DeepSeek-V2-Lite-Chat (~16B MoE) / Qwen3-8B |
| GPU | NVIDIA A800 80GB PCIe × 4, TP=4 |
| 压测参数 | ROUNDS=5, STEPS=8, REQUEST_RATE=4, MAX_CONCURRENCY=1 |

---
## DeepSeek-V2-Lite 全链路

```
2.15 tok/s (baseline, 全量重算)
    ↓ P0: 增量 KV Cache  (8.49, +3.95x)
    ↓ P2: torch.compile   (12.75, +45.5%)
    ↓ P3-Triton: MLA kernel (13.08, +2.6%)
    ↓ P5b: MoE GPU map    (13.20, +0.9%)
    ↓ P6: position buffer (13.42, +1.7%)

最终: 2.15 → 13.42 tok/s (6.24x，峰值)
vLLM 基准: 36.94 tok/s (差距 2.75x)
  > 此前测得过 26.70 tok/s，原因为当时机器显存被其他进程占用，被迫设置 `gpu_memory_utilization=0.15`，导致 vLLM paged KV cache pool 过小，decode 时频繁 block eviction/swap 引入额外延迟（-28%）。36.94 为显存充足（gpu_memory_utilization=0.8）下的对比值。
```

**各阶段说明**:

| 阶段 | 改动 |
|------|------|
| **P0 增量 KV Cache** | attention 层内置 KV buffer；prefill 缓存 k_nope/v/raw_k_pe，decode 拼接缓存后重新计算全位置 RoPE |
| **P2 torch.compile** | `mode='default'` 编译 dense MLP（MoE 不编译）；decode attention 改用全 buffer + `attn_mask` 固定 shape |
| **P3-Triton MLA kernel** | 新增 `triton_mla_decode` 两阶段 kernel；从 `kv_b_proj` 提取 W_UK_T/W_UV 权重（无额外显存）；统一 KV cache `[c_kv\|k_pe_rope]`（节省 55% 显存） |
| **P5b MoE GPU map** | 新增 `_expert_map` GPU tensor 映射 global expert → local index；prefill 用 `mask.nonzero()` 分组 batch 处理，decode 保留 `.item()` 循环 |
| **P6 position buffer** | `register_buffer("_pos_buf")` 预分配 4096 个 int64；forward 中 `self._pos_buf[offset:offset+seqlen]` 零拷贝 slice 替代 `torch.arange` |

## Qwen3-8B 全链路

```
23.06 tok/s (baseline, 全量重算, commit acf2191)
    ↓ P0: 增量 KV Cache  (23.51, +2.0%)
    ↓ P2: torch.compile   (5.95, -74.7%)
    ↓ P5a: flash_attn+fused MLP (16.57)
    ↓ P6: position buffer (16.45)
    ↓ decode→SDPA, 去 P5a/P6/compile，回退P0，(23.25)

最终: 23.06 → 23.25 tok/s (+0.8%)
vLLM 基准: 36.92 tok/s (差距 1.59x)
```

**各阶段说明**:

| 阶段 | 改动 |
|------|------|
| **P0 增量 KV Cache** | attention 层内置 KV buffer；decode 用 SDPA auto-dispatch（P0 已是最优 decode 路径） |
| **P2 torch.compile** | compile attention+MLP；decode 改用全 buffer + `attn_mask` → 动态 shape 导致 torch.compile 严重重编译，性能暴跌 |
| **P5a flash_attn+fused MLP** | 引入 `flash_attn_varlen_func` + `MergedColumnParallelLinear`（合并 gate+up 为单次 GEMM）；FA2 部分修复 P2 回退，但仍未回到 P0 水平 |
| **P6 position buffer** | `register_buffer("_pos_buf")` 预分配位置索引，波动范围内 |
| **decode→SDPA** | 回退 P5a/P6/compile，恢复 P0 的 slice KV + SDPA 路径，吞吐回到 23.25 |

**关键发现**: Qwen3-8B 的 P0（增量 KV Cache）仅提升 2%（23.06→23.51），远低于 DeepSeek 的 4x。因为 Qwen decode 瓶颈是 NCCL 通信（占 GPU 55.8%）而非计算——全量重算虽然多了 6.5x GEMM，但 GEMM 只占 GPU 16.6%，通信才是大头。Qwen 的正确优化方向是 P7b Custom AllReduce + fused elementwise kernel。

---

## 代码文件清单

| 文件 | Phase | 变更 |
|------|-------|------|
| `engine/kernels/triton_mla_decode.py` | P3-Triton | 新增 Triton MLA decode kernel |
| `engine/models/deepseek_v2.py` | P3-Triton, P6 | MLA 权重提取, 统一 KV cache, position buffer |
| `engine/models/qwen.py` | P5a, P6 | gate_up_proj 合并 GEMM, position buffer |
| `engine/tp_layers/linear.py` | P5a | 新增 MergedColumnParallelLinear |
| `engine/tp_layers/moe.py` | P5b | GPU-side expert_map + hybrid forward |
| `engine/tp_layers/distributed.py` | P7a | all_reduce 去 dtype 转换 |

---

## torch.profiler 对比分析 (Qwen3-8B, TP=4, 22 decode steps)

> **为何用 nano-vllm 而不用 vLLM**: vLLM 使用多进程架构（EngineCore 在独立 worker 进程运行推理），`nsys` 只能抓取主进程的 CPU 活动，GPU kernel 数据全在子进程中无法捕获（`nsys stats` 报 "no CUDA kernel data"）。nano-vllm 单进程运行，`torch.profiler` 可直接捕获完整 GPU trace，因此暂时先选它作为profiling对比基准。

### 时间分解对比

| 类别 | meta-infer TP=4 | nano-vllm TP=4 | 差异 |
|------|----------------|----------------|------|
| 总 CPU 时间 | 2,435ms | 2,344ms | 1.04x (相近) |
| 总 GPU 时间 | 457ms (20.8ms/step) | 115ms (~5.2ms/step) | 4.0x |
| Kernel 数量 | 53,020 | 13,093 | 4.0x |
| **NCCL AllReduce** | **254.9ms (55.8%)** | 20.4ms (17.8%) | 12.5x |
| GEMM/Linear | 75.8ms (16.6%) | 74.8ms (65.1%) | 1.0x (相同!) |
| Elementwise | **117.8ms (25.8%)** | 0.5ms (0.5%) | 236x |
| Attention | 6.2ms (1.4%) | 10.2ms (8.9%) | 0.6x |
| Copy/Memcpy | **34.3ms (7.5%)** | 12.5ms (10.9%) | 2.7x |
| Triton fused | — | 17.5ms (15.2%) | nano-vllm 专属 |

### Top 8 GPU Kernel 对比

| meta-infer TP=4 | 耗时 | 占比 | nano-vllm TP=4 | 耗时 | 占比 |
|------|------|------|------|------|------|
|  **ncclDevKernel_AllReduce_Sum** | 253.4ms | 55.5% | cutlass gemm_relu | 25.0ms | 21.8% |
| ampere_s16816gemm_bf16 | 28.3ms | 6.2% |  **ncclDevKernel_AllReduce_Sum** | 20.0ms | 17.4% |
| gemvx::kernel (GEMV) | 16.6ms | 3.6% | ampere gemm_bf16_64x64 | 16.2ms | 14.1% |
| ampere gemm_bf16_64x64 | 14.4ms | 3.1% | gemvx::kernel (GEMV) | 10.8ms | 9.4% |
| direct_copy_kernel | 13.9ms | 3.1% | gemv2T_kernel_val (GEMV) | 7.0ms | 6.1% |
| reduce_kernel (RMSNorm) | 12.1ms | 2.6% | flash_fwd_splitkv | 6.1ms | 5.3% |
| elementwise_kernel | 9.7ms | 2.1% | triton_red_fused (RMSNorm) | 5.6ms | 4.8% |
| BinaryFunctor (add/mul) | 9.0ms | 2.0% | ampere gemm_bf16_128x64 | 3.9ms | 3.4% |

### 分项分析

#### 1. NCCL AllReduce — 12.5x 差异

**原因**: nano-vllm 使用 `QKVParallelLinear`（Q/K/V 合并为一个 ColumnParallel 层）和 `MergedColumnParallelLinear`（gate/up 合并），减少了独立 GEMM launch 和 stream 同步开销。nano-vllm 的 BlockTable、slot_mapping 等辅助结构在 GPU 上，避免了 CPU-GPU 同步打断 pipeline。meta-infer 的 Python 控制流（`.item()`、`if` 分支等）频繁打断 GPU stream，使 NCCL 通信无法与计算 overlap。

**提速办法**: 
- P7b Custom AllReduce（P2P 内存映射替代 NCCL ring，对小 tensor 延迟降低 5-10x）
- QKV 合并投影（减少独立 GEMM launch）
- 减少 CPU-GPU 同步（用 CUDA Graph 消除 Python 开销）

#### 2. Elementwise — 236x 差异

**原因**: nano-vllm 使用 Triton fused kernel（`triton_red_fused`、`triton_poi_fused`、`triton_poi_fused_mul_silu`）将 RMSNorm 归一化、SiLU 激活、残差加法融合为单个 kernel。meta-infer 使用 PyTorch 原生 elementwise kernel，每个操作（`.pow(2)`、`.mean()`、`.rsqrt()`、`*weight`、`silu(gate)*up`、`hidden+h`、`hidden+h2`）都是独立 kernel launch。36 层 × 每层 ~10 个 elementwise kernel = 360+ kernel/step。

**提速办法**:
- 实现 Triton fused RMSNorm + 残差加法 kernel（参考 nano-vllm 的 `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt`）
- 实现 Triton fused SiLU+Mul kernel（参考 nano-vllm 的 `triton_poi_fused_mul_silu`）
- 两者合并可节省 ~5ms/step

#### 3. Copy/Memcpy — 2.7x 差异

**原因**: meta-infer 的 KV cache 写入使用 Python slice（`k_buf[:, kv_len:kv_len+1] = k`），触发 implicit copy。GQA broadcast 使用 `repeat_interleave`（分配新 tensor 并拷贝）。nano-vllm 使用 `store_kvcache_kernel` Triton kernel 和 head-major 布局，减少了数据搬运。

**提速办法**:
- 用 scatter kernel 或 `index_copy_` 替代 slice 赋值
- GQA 用 `expand` 替代 `repeat_interleave`

#### 4. Attention — meta-infer 更省

**原因**: Qwen3-8B decode 是 q_len=1 的单 token 推理。meta-infer 使用 SDPA auto-dispatch（自动选最优 flash kernel），nano-vllm 使用 `flash_attn_with_kvcache`。短 kv_len (~5-30) 时 SDPA 的 auto-dispatch 开销更小（少了 kvcache wrapper 的参数构建）。

**注意**: 此优势仅在短 kv_len decode 场景成立。长 prefill 时 FA2 的 paged KV 支持更高效。

---

## torch.profiler 对比分析: meta-infer vs vLLM (Qwen3-8B, TP=4, 12 output tokens)

> 使用 vLLM 内置 `profiler_config` 离线 profiling（参考 `examples/offline_inference/simple_profiling.py`），rank-0 GPU trace 提取。两者均 `enforce_eager=True`（无 CUDA graph），温度均接近 0（greedy）。输出一致：`'（ ） A：建筑与园林结合 B：建筑'`。

### GPU 时间分解对比

| 类别 | meta-infer TP=4 | vLLM TP=4 | 差异 |
|------|----------------|-----------|------|
| 总 GPU 时间 | 1,006ms | 412ms | 2.44x |
| Kernel 数量 | 31,920 | 6,299 | 5.07x |
| **NCCL AllReduce** | **884.6ms (87.9%)** | **358.0ms (87.0%)** | 2.47x |
| GEMM/GEMV | 45.5ms (4.5%) | 40.5ms (9.8%) | 1.12x (相近) |
| **Elementwise** | **71.3ms (7.1%)** | 0.7ms (0.2%) | 102x |
| Attention/FA2 | 3.9ms (0.4%) | 5.7ms (1.4%) | 0.68x |
| **Copy/Memcpy** | **20.3ms (2.0%)** | 0.1ms (0.0%) | 203x |
| Fused ops (rms/silu/rope) | 1.2ms (0.1%) | 9.7ms (2.4%) | 0.12x |

### 分项分析

#### 1. NCCL AllReduce — 同为最大瓶颈，但 meta-infer 2.47x 更慢

两者 NCCL 均占 GPU 87%+。但 meta-infer 总体 NCCL 耗时 2.47x——不是因为单次 allreduce 不同，而是 Kernel 数量差异（meta-infer 31,920 vs vLLM 6,299）。更多的独立 GEMM launch（Q/K/V 未合并）导致更多的 stream 同步点，NCCL 等待时间被拉长。

vLLM 使用 `QKVParallelLinear` 合并 Q/K/V 投影和 `all_reduce` 的 `record_param_comms` CUDA 事件管理，减少了 TP 同步碎片。

#### 2. Elementwise — 102x 差异

meta-infer 的 RMSNorm（`pow→mean→rsqrt→mul`）、SiLU（`silu(gate)*up`）、残差加（`h+attn_out`, `h+mlp_out`）全部使用独立 PyTorch elementwise kernel。vLLM 使用 C++ fused kernel：`fused_add_rms_norm`（残差+RMSNorm 融合）、`silu_and_mul`（SiLU+Mul 融合）、`rotary_embedding`（RoPE 融合）。36 层 × 每层 ~8 个 elementwise = 288+ kernel/step，vLLM 融合后仅 ~4-5 个。

#### 3. Copy/Memcpy — 203x 差异

meta-infer 的 GQA broadcast（`repeat_interleave` 分配+拷贝）和 KV cache 写入（Python slice 触发 implicit copy）产生了大量显存搬运。vLLM 使用 `reshape_and_cache_flash` C++ kernel 直接写入 paged KV cache，无中间拷贝。

### CPU 时间分解对比

| 类别 | meta-infer TP=4 | vLLM TP=4 | 差异 |
|------|----------------|-----------|------|
| 总 CPU 时间 | 2,672ms | 737ms | 3.63x |
| **aten::linear/mm (GEMM dispatch)** | 862.1ms (32.3%) | 197.4ms (26.8%) | 4.37x |
| **CUDA stream/event/record ops** | 411.2ms (15.4%) | 0.9ms (0.1%) | 457x |
| **c10d::allreduce 通信 dispatch** | 269.1ms (10.1%) | 3.2ms (0.4%) | 84x |
| aten::mul/add/div/silu (activation) | 229.1ms (8.6%) | 4.5ms (0.6%) | 51x |
| **aten::to + _to_copy (dtype 转换)** | 218.6ms (8.2%) | 4.3ms (0.6%) | 51x |
| **aten::item/detach** | 142.4ms (5.3%) | 0.1ms (0.0%) | 1424x |
| aten::norm/pow/rsqrt/mean (RMSNorm) | 124.9ms (4.7%) | 18.5ms (2.5%) | 6.75x |
| aten::copy_/clone | 106.6ms (4.0%) | 11.1ms (1.5%) | 9.60x |

**CPU 瓶颈分析**: 

- **`record_param_comms` 411ms (15.4%)**: meta-infer 的每次 TP allreduce 都记录 CUDA event，36 层 × 2 allreduce × ~12 steps = 864 次 record。vLLM 使用 `all_reduce` 的 `record_param_comms` 优化，开销仅 0.9ms（457x 差异）。
- **`c10d::allreduce_` 269ms**: NCCL 通信的 CPU dispatch 开销。vLLM 使用 `vllm::all_reduce` 自定义 op 路径，耗时仅 63.5ms 但归类在 Other 中。
- **`aten::to/_to_copy` 219ms**: bf16↔fp32 dtype 转换开销。vLLM 全程保持 bf16，仅 4.3ms。
- **`aten::item/detach` 142ms**: `aten::detach` 在 torch.compile guard 和模型 forward 中被频繁调用。

#### 4. GEMM 几乎相同 — 45.5ms vs 40.5ms

纯矩阵计算性能接近，说明 GPU compute 利用率在同一水平。差距在通信和杂项开销。

### 最大收益路径

| 优先级 | 优化 | 当前耗时 | 目标 | 预计节省 |
|--------|------|---------|------|---------|
| P7b | Custom AllReduce | 884ms | ~200ms | ~684ms |
| fused | Triton fused RMS+SiLU+RoPE | 71ms | ~5ms | ~66ms |
| KV | reshape_and_cache + expand GQA | 20ms | ~1ms | ~19ms |

三项合计可节省 ~770ms，GPU 时间从 1006ms 降至 ~236ms，接近 vLLM 的 412ms 水平（vLLM 仍有 NCCL 税 358ms）。

---

## torch.profiler 对比分析 (DeepSeek-V2-Lite, TP=4, 12 decode steps)

> vLLM profiling 使用 nsys capture 主进程，GPU kernel 数据全在 worker 子进程中，`nsys stats` 报 "no CUDA kernel data"。vLLM 多进程架构下 nsys 不可行。以下仅分析 meta-infer 自身。

### meta-infer DeepSeek 时间分解

| 类别 | 耗时 | 占比 |
|------|------|------|
| 总 GPU 时间 | 551ms (~45.9ms/step) | — |
| Kernel 数量 | 43,621 | — |
| **NCCL AllReduce+AllGather** | **435.3ms** | **79.0%** |
| Elementwise | 78.2ms | 14.2% |
| GEMM/GEMV | 28.4ms | 5.2% |
| Copy/Memcpy | 18.9ms | 3.4% |

### CPU 时间分解

| 类别 | 耗时 | 占比 |
|------|------|------|
| 总 CPU 时间 | 2,066ms | — |
| **aten::to + _to_copy** (dtype 转换) | 245.6ms | 11.9% |
| **aten::item + _local_scalar_dense** (MoE `.item()` GPU→CPU 同步) | 227.3ms | 11.0% |
| aten::mul/add/div/silu (activation) | 208.9ms | 10.1% |
| aten::linear/addmm (GEMM dispatch) | 155.4ms | 7.5% |
| aten::copy_/clone | 123.0ms | 6.0% |
| c10d::allreduce + record_param_comms | 155.9ms | 7.5% |
| aten::norm/pow/rsqrt/mean (RMSNorm) | 82.3ms | 4.0% |
| Other | 710.1ms | 34.4% |

**CPU 瓶颈**: MoE 的 `.item()` 同步占 CPU 11%（227ms），加上 `aten::to` 的 dtype 转换（246ms），两者合计 23%。这些在 GPU profiling 中不可见但在端到端延迟中贡献显著。

### Top 5 GPU Kernels

| Kernel | 耗时 | 占比 |
|------|------|------|
| `ncclDevKernel_AllReduce_Sum` | 430.8ms | 78.2% |
| `gemvx::kernel` (GEMV) | 24.8ms | 4.5% |
| `unrolled_elementwise (direct_copy)` | 8.6ms | 1.6% |
| `BinaryFunctor` (add/mul) | 5.9ms | 1.1% |
| `ncclDevKernel_AllGather_RING` | 4.5ms | 0.8% |

### DeepSeek vs Qwen 瓶颈对比

| 类别 | DeepSeek TP=4 | Qwen TP=4 | 说明 |
|------|-------------|----------|------|
| 每步 GPU 时间 | 45.9ms | 20.8ms | DeepSeek 2.2x 慢 |
| **NCCL** | **79.0%** | **55.8%** | DeepSeek MoE 的 EP allreduce 额外增加通信 |
| GEMM | 5.2% | 16.6% | DeepSeek decode 只有 1 token，MoE GEMV 利用率低 |
| Elementwise | 14.2% | 25.8% | Qwen 的 dense MLP 有更多 elementwise op |

**分析**: DeepSeek 的 NCCL 占比 (79%) 比 Qwen (56%) 更高，因为 MoE 的 expert-parallel allreduce 在所有 60 层 MoE 中都触发。此外 `ncclDevKernel_AllGather_RING` (4.5ms) 来自 LM head 的 `all_gather_last_dim`。GEMM 占比极低 (5.2%) 因为 decode 只有 1 token，MoE 的 `.item()` 循环中每个 expert 只做 GEMV（batch=1），tensor core 利用率极低。

**提速路径**: P7b Custom AllReduce（79%→20%）是 DeepSeek 最快最大的收益来源。

---

## 阻塞/延后的优化

| Phase | 状态 | 阻塞原因 |
|-------|------|---------|
| P2 (CUDA Graph) | 阻塞 | Qwen: tensor 索引 KV 写 vs Python slice 数值发散。DeepSeek: MoE `.item()` + Triton 动态分配 |
| 完整 Fused MoE | 延后 | batch=1 时 nonzero+index_add_ 开销 > `.item()`，需 batch>1 |
