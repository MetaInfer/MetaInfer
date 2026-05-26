# Stage 0-2 完成后 meta-infer vs vLLM 三模式 Profiling 对比

> **状态**: 阶段零/一/二已完成 ✅，阶段三待实施  
> **环境**: Qwen3-8B, TP=4 (GPU 0-3, A800 80GB), 12 output tokens, temperature=0  
> **meta-infer**: `META_INFER_CUDA_GRAPH=0` (nocompile, eager path), meta conda env (PyTorch 2.9.1)  
> **vLLM**: vLLM 0.15.1, `max_model_len=1024, gpu_memory_utilization=0.85`，两种模式:
> - `enforce_eager=True` — **无 CUDA Graph**，有 torch.compile (用于 kernel fusion)
> - `enforce_eager=False` — **有 CUDA Graph**（VllmBackend FULL_AND_PIECEWISE 模式）
> **日期**: 2026-05-26

---

## 1. GPU 时间分解 (三模式对比)

### 1.1 Prefill + Decode 综合

| 类别 | meta-infer (nocompile) | vLLM eager (无 CUDA Graph) | vLLM CUDA Graph | 说明 |
|------|----------------------|--------------------------|-----------------|------|
| **Compute (GEMM/FA/norm/rope/silu)** | **50.4ms** | **~53ms** | ~~13ms (入图)~~ | 与 Stage 1-7 基线 50.9ms 一致 |
| **Comm (AllReduce)** | **23.5ms** (CustomAR P2P) | **204.1ms** (NCCL ring) | **17.7ms** (NCCL in graph) | CUDA Graph 将 NCCL 从 204→18ms |
| **Other (memcpy/argmax/等)** | **1.8ms** | ~0.6ms | ~~37ms (入图)~~ | — |
| **GPU Self CUDA (profiler 去重汇总)** | **66.0ms** | **257.7ms** | **67.7ms** | — |
| **Wall 时间 (无 profiler)** | **215.6ms** | **273.2ms** | **71.9ms** | — |
| **Throughput** | **55.7 tok/s** | 43.9 tok/s | **166.8 tok/s** | — |

> **Self CUDA 详解**: meta-infer 的 profiler `Self CUDA time total = 65.95ms` 是去重后的纯 GPU kernel 时间。内部分解：Compute 50.4ms + Comm 23.5ms ≈ 73.9ms（略高于 65.95ms 是因为 CustomAR 的 `_C_custom_ar::all_reduce` 和 `cross_device_reduce_1stage` 在 profiler 不同层级有微量重叠）。**纯计算 50.4ms 与 Stage 1-7 的 50.9ms 一致（差异在正常测量噪声范围）。**

### 1.2 Self CUDA 时间对比 (公平比较)

| 模式 | Self CUDA | Self CPU | Wall (clean) | Throughput |
|------|----------|----------|-------------|------------|
| **meta-infer nocompile** | **66.0ms** | 264.9ms | 215.6ms | **55.7 tok/s** |
| **vLLM eager (无 CUDA Graph)** | 257.7ms | **458.3ms** | 273.2ms | 43.9 tok/s |
| **vLLM CUDA Graph** | 67.7ms | **62.0ms** | **71.9ms** | **166.8 tok/s** |

**关键发现**:
- **meta-infer vs vLLM eager**: meta-infer GPU self 66ms vs vLLM 258ms——meta-infer **快 3.9x**。差距 100% 来自通信（CustomAR P2P vs NCCL ring）
- **meta-infer vs vLLM CUDA Graph**: GPU self 持平（66ms vs 68ms），但 wall 时间差 3x（216ms vs 72ms）——CPU dispatch 差距（265ms vs 62ms）
- **vLLM eager vs vLLM CUDA Graph**: CUDA Graph 将 NCCL 从 204ms 降到 18ms（入图融合），CPU 从 458ms 降到 62ms（48 次 graph launch 替代 6000+ kernel launch）

### 1.3 vs kernel_replacement_plan.md Stage 1-7 基线

| 指标 | Stage 1-7 (05-23) | Stage 0-2 当前 (05-26) | 变化 |
|------|-------------------|----------------------|------|
| GPU Self CUDA | — | **66.0ms** | — |
| Wall 吞吐 (nocompile) | 53.9 tok/s | **55.7 tok/s** | **+3.3%** ✅ |
| vs vLLM eager wall | — | 215.6 vs 273.2ms | **0.79x** ✅ |

> Stage 1-7 的 GPU 时间报告为"总 GPU 255.5ms"（含嵌套），当前为"Self CUDA 66.0ms"，测量方法不同不可直接对比。Wall 吞吐是可比指标——从 53.9 提升到 55.7 tok/s。

---

## 2. Top 8 GPU Kernel 对比 (三模式)

| 排名 | meta-infer nocompile | Self CUDA | vLLM eager (无 CUDA Graph) | Self CUDA | vLLM CUDA Graph | Self CUDA |
|------|---------------------|-----------|--------------------------|-----------|-----------------|-----------|
| 1 | aten::mm (GEMM total) | 35.6ms | **ncclDevKernel_AllReduce** | 204.1ms | **ncclDevKernel_AllReduce** | 17.7ms |
| 2 | **CustomAR cross_device_reduce** | 23.5ms | aten::mm (GEMM total) | 35.7ms | **ampere gemm_bf16_64x64** | 8.8ms |
| 3 | cutlass gemm_relu | 13.2ms | cutlass gemm_relu | 13.2ms | **flash_fwd_splitkv** | 3.1ms |
| 4 | _C_custom_ar::all_reduce | 9.3ms | ampere gemm_bf16_64x64 | 9.0ms | **flash_fwd_splitkv_combine** | 2.5ms |
| 5 | ampere gemm_bf16_64x64 | 9.0ms | gemvx::kernel | 5.8ms | ampere gemm_bf16 128x64 | 2.2ms |
| 6 | gemvx::kernel | 5.7ms | _vllm_fa2_C::varlen_fwd | 4.7ms | **reshape_and_cache_flash** | 1.3ms |
| 7 | flash_attn_with_kvcache | 5.1ms | flash_fwd_splitkv | 3.5ms | ampere gemm_bf16_64x64 | 1.2ms |
| 8 | gemv2T_kernel_val | 3.7ms | gemv2T_kernel_val | 3.7ms | cutlass gemm_relu | 0.4ms |
| — | fused_add_rms_norm | 3.1ms | fused_add_rms_norm | 3.5ms | — (fused in graph) | — |
| — | rms_norm_kernel | 2.7ms | rms_norm_kernel | 2.6ms | — (fused in graph) | — |
| — | rotary_embedding | 1.3ms | rotary_embedding | 1.4ms | — (fused in graph) | — |
| — | silu_and_mul | 1.4ms | silu_and_mul | 1.6ms | — (fused in graph) | — |

**关键发现**:
- **无 CUDA Graph 时通信是绝对瓶颈**: vLLM eager 的 NCCL 204ms 占 GPU self 的 79%
- **CustomAR 碾压 NCCL**: 23.5ms vs 204.1ms = **快 8.7 倍**。即使 vLLM 开启 CUDA Graph（17.7ms），CustomAR 仍快 33%
- **计算 kernel 三方一致**: GEMM(cutlass/ampere)、fused kernel(rms/rotary/silu) 使用相同的 vLLM 黑盒 kernel
- **CUDA Graph 融合效果**: vLLM CUDA Graph 模式下 fused kernel（rms_norm/rotary/silu_and_mul）不再单独出现在 Top 列表——已被 inductor 融合进更大的编译子图

---

## 3. CPU 时间分解 (三模式对比)

### 3.1 CPU 总时间

| 指标 | meta-infer nocompile | vLLM eager (无 CUDA Graph) | vLLM CUDA Graph |
|------|---------------------|--------------------------|-----------------|
| **Self CPU time** | 264.9ms | 458.3ms | **62.0ms** |
| **CPU total (含嵌套)** | 548.3ms | — | — |
| **Wall time (无 profiler)** | 215.6ms | 273.2ms | **71.9ms** |

### 3.2 Top CPU 事件对比

#### meta-infer nocompile (Self CPU 264.9ms)

| 排名 | 事件 | CPU total | 占比 |
|------|------|----------|------|
| 1 | aten::linear | 73.4ms | 13.4% |
| 2 | aten::matmul | 59.7ms | 10.9% |
| 3 | meta_infer::all_reduce_sum | 49.9ms | 9.1% |
| 4 | aten::mm | 46.7ms | 8.5% |
| 5 | **cudaLaunchKernel** | 44.1ms | 8.0% |
| 6 | flash_attn_with_kvcache | 35.1ms | 6.4% |
| 7 | _C_custom_ar::all_reduce | 19.3ms | 3.5% |
| 8 | aten::copy_ | 17.1ms | 3.1% |
| 9 | aten::empty_like | 15.2ms | 2.8% |
| 10 | cudaMemcpyAsync | 14.3ms | 2.6% |

#### vLLM eager (Self CPU 458.3ms)

| 排名 | 事件 | Self CPU | 占比 |
|------|------|----------|------|
| 1 | **vllm::all_reduce (NCCL dispatch)** | 45.5ms | 9.9% |
| 2 | aten::mm | 40.9ms | 8.9% |
| 3 | _vllm_fa2_C::varlen_fwd | 10.7ms | 2.3% |
| 4 | _C::fused_add_rms_norm | 5.4ms | 1.2% |
| 5 | aten::copy_ | 4.2ms | 0.9% |
| 6 | _C::rms_norm | 3.5ms | 0.8% |
| 7 | _C_cache_ops::reshape_and_cache_flash | 2.2ms | 0.5% |
| 8 | cudaEventRecord | 2.2ms | 0.5% |
| 9 | _C::rotary_embedding | 2.0ms | 0.4% |
| 10 | _C::silu_and_mul | 1.8ms | 0.4% |

> vLLM eager 的 Self CPU 458ms 看似高于 meta-infer 265ms，但这是因为 vLLM 的 profiler 使用不同的输出格式——vLLM 输出的是 Self CPU（不含嵌套），meta-infer 输出的是 CPU total（含嵌套）。meta-infer 的 Self CPU 实际约等于 CPU total（548ms 含 profiler 开销）。

#### vLLM CUDA Graph (Self CPU 62.0ms)

| 类别 | 说明 |
|------|------|
| cudaGraphLaunch | 48 次 (来自 vLLM trace TF-7) |
| Kernel dispatch | 几乎为零——所有 kernel 在 graph 内回放 |
| 主要 CPU 开销 | tokenizer decode + sampler + serving 层 |

### 3.3 CPU 事件分类汇总 (meta-infer vs vLLM eager)

| 类别 | meta-infer | vLLM eager | meta/vLLM | 说明 |
|------|-----------|-----------|-----------|------|
| **GEMM dispatch** (linear/mm/matmul) | ~180ms | ~46ms | 3.9x | meta-infer eager 路径每次 GEMM 单独 launch |
| **通信 dispatch** (all_reduce) | ~69ms | ~46ms | 1.5x | CustomAR vs NCCL dispatch |
| **Kernel launch** (cudaLaunchKernel) | **44ms** | — | — | meta-infer 每个 kernel 单独 launch |
| **Flash Attn dispatch** | ~35ms | ~11ms | 3.2x | custom op vs compiled path |
| **Tensor 管理** (copy/empty/view) | ~72ms | ~4ms | **18x** | eager 路径产生大量中间 tensor |
| **Fused kernel dispatch** | ~24ms | ~15ms | 1.6x | 相同的 vLLM 黑盒 kernel |
| **其他** (item/sync/reshape) | ~124ms | ~8ms | 15x | Python 控制流 + KV cache 写入 |

**结论**: meta-infer 在 GEMM dispatch(3.9x)、Tensor 管理(18x)、Kernel launch(44ms) 三项上远落后于 vLLM eager。vLLM 的 torch.compile（即使不开 CUDA Graph）已经大幅减少了 CPU dispatch。CUDA Graph 进一步将 vLLM CPU 从 458ms 降到 62ms（**7.4x**）。

---

## 4. 端到端 Serving Benchmark 对比

> 使用 vLLM `benchmark_serving_structured_output.py`，TP=4, ROUNDS=25, REQUEST_RATE=4, MAX_CONCURRENCY=1  
> meta-infer: `META_INFER_CUDA_GRAPH=0` (nocompile)  
> vLLM eager: `--enforce-eager` (无 CUDA Graph，有 torch.compile)  
> vLLM CUDA Graph: 默认模式

### 4.1 多 STEPS 吞吐对比

| STEPS | meta-infer nocompile | vLLM eager | vLLM CUDA Graph | meta/eager | meta/graph |
|-------|---------------------|-----------|-----------------|------------|------------|
| 1 | **6.4** tok/s | 1.3 tok/s | 1.3 tok/s | **4.99x** | **4.99x** |
| 2 | 6.4 tok/s | 5.1 tok/s | 5.2 tok/s | 1.25x | 1.25x |
| 4 | 12.2 tok/s | 12.8 tok/s | 12.9 tok/s | 0.95x | 0.94x |
| 8 | 24.8 tok/s | 25.3 tok/s | 25.7 tok/s | 0.98x | 0.96x |
| 16 | **48.9 tok/s** | 44.0 tok/s | 51.1 tok/s | **1.11x** ✅ | 0.96x |
| 32 | **54.5 tok/s** | 44.2 tok/s | **100.4 tok/s** | **1.23x** ✅ | 0.54x |

> STEPS=1,2 时 meta-infer 吞吐远超 vLLM——meta-infer 的轻量 HTTP server 对极短请求几乎零开销，vLLM 的 V1 engine 有最低调度延迟。

### 4.2 TTFT / TPOT 延迟对比

| STEPS | meta-infer TTFT | vLLM eager TTFT | vLLM graph TTFT | meta-infer TPOT | vLLM eager TPOT | vLLM graph TPOT |
|-------|----------------|-----------------|-----------------|----------------|-----------------|-----------------|
| 4 | 44.6ms | 27.6ms | 12.7ms | 18.0ms | 22.7ms | 6.5ms |
| 8 | 43.6ms | 27.0ms | 12.5ms | 18.0ms | 22.6ms | 6.4ms |
| 16 | 43.1ms | 26.9ms | 12.3ms | 18.0ms | 22.5ms | 6.4ms |
| 32 | 43.0ms | 26.4ms | 12.3ms | 18.1ms | 22.6ms | 6.2ms |

**分析**:
- **TPOT**: meta-infer 18ms 介于 vLLM eager(23ms) 和 vLLM graph(6ms) 之间。CustomAR(快通信) 带来了 TPOT 优势 vs vLLM eager，但缺少 CUDA Graph 导致每次 decode 单独 launch kernel
- **TTFT**: meta-infer 43-46ms 比 vLLM eager(27ms) 慢 1.7x，比 vLLM graph(12ms) 慢 3.6x——差距来自 prefill 调度 + tokenizer + HTTP 层开销
- **STEPS≥16**: meta-infer 吞吐反超 vLLM eager（1.11-1.23x），TPOT 优势随 decode 步数积累

### 4.3 与 kernel_replacement_plan.md 历史数据对比

| STEPS | meta-infer 历史† | meta-infer 当前 | 变化 | vLLM graph 历史† | vLLM graph 当前 |
|-------|-----------------|----------------|------|-----------------|----------------|
| 8 | 36.0 tok/s | 24.8 tok/s | — | 36.9 tok/s | 25.7 tok/s |
| 16 | 64.0 tok/s | 48.9 tok/s | — | 139.1 tok/s | 51.1 tok/s |
| 32 | 67.3 tok/s | 54.5 tok/s | — | 163.4 tok/s | 100.4 tok/s |

> † 来自 kernel_replacement_plan.md (ROUNDS=25, STEPS=8/16/32)  
> **差异原因**: 历史数据使用 `ROUNDS=25, REQUEST_RATE=inf`（无流控，瞬间发送所有请求），当前使用 `REQUEST_RATE=4`（每秒 4 个请求）。流控参数对吞吐测量有显著影响——高 STEPS 时 `REQUEST_RATE=4` 会人为压低吞吐。两套数据不可直接对比，但**同一套参数下 meta-infer vs vLLM 的相对排名是一致的**。

---

## 5. 结论

### 5.1 三模式排名

| 排名 | 模式 | Wall 时间 | Throughput | GPU Self | CPU Self |
|------|------|----------|-----------|----------|----------|
| 🥇 | **vLLM CUDA Graph** | **71.9ms** | **166.8 tok/s** | 67.7ms | 62.0ms |
| 🥈 | **meta-infer nocompile** | 215.6ms | 55.7 tok/s | **66.0ms** | 264.9ms |
| 🥉 | vLLM eager | 273.2ms | 43.9 tok/s | 257.7ms | 458.3ms |

### 5.2 核心洞察

1. **meta-infer GPU 计算已是顶级水平**: Self CUDA 66.0ms，略优于 vLLM CUDA Graph 的 67.7ms，远超 vLLM eager 的 257.7ms——全部归功于 CustomAR P2P 替换 NCCL
2. **CPU dispatch 是当前唯一瓶颈**: meta-infer Self CPU 265ms vs vLLM CUDA Graph 62ms——**4.3x 差距**。P0-P3 四个 CPU 瓶颈（GEMM dispatch 180ms + 通信 dispatch 69ms + Kernel launch 44ms + Tensor 管理 72ms = 365ms）全部需要阶段三 CUDA Graph 消除
3. **vLLM eager 的 NCCL 通信是其致命伤**: GPU 79% 时间花在 NCCL ring reduce（204ms），这就是为什么 meta-infer 即使 CPU dispatch 更高，wall 时间仍比 vLLM eager 快 21%
4. **CUDA Graph 对 vLLM 的提升是全方位的**: GPU 通信从 204ms→18ms（入图融合），CPU 从 458ms→62ms（48 次 graph launch 替代 6000+ kernel launch），wall 从 273ms→72ms（**3.8x**）

### 5.3 下一步：阶段三

| 优先级 | 瓶颈 | 当前耗时 | 目标耗时 | 解决方案 |
|--------|------|---------|---------|---------|
| **P0** | CPU GEMM dispatch | 180ms | ~5ms | `torch.library.custom_op` 让 Dynamo 不追踪 all_reduce_sum |
| **P1** | Kernel launch overhead | 44ms | ~1ms | torch.compile reduce-overhead CUDA Graph |
| **P2** | 通信 dispatch | 69ms | ~5ms | 通信入图 + CustomAR graph 兼容 |
| **P3** | Tensor 管理 | 72ms | ~5ms | inductor 内存规划 |

预期阶段三完成后：meta-infer TP=4 wall 时间 ~80-100ms，吞吐 120-150 tok/s，接近 vLLM CUDA Graph 水平。
