# Qwen3-8B Z200 HIP 算子与调用契约

> 用途：给生成 C++ 单卡推理框架的 Agent 一份可以直接照着实现和接线的算子知识库。本文描述当前算子的真实接口、Q8_0 反量化路径、张量布局、workspace 生命周期、prefill/decode 调用顺序和已知缺口。
>
> 算子参考源码（实现事实来源）：`notebooks/reference/qwen3_z200_kernels.hip.cpp`
>
> 模型数学与权重 shape：`notebooks/model/qwen3/qwen3_8b_contract.md`

## 0. 核心约定：按冻结权重格式调用 hipBLAS

**所有带权重的矩阵乘都调用 hipBLAS，不在自定义 HIP kernel 中手写 GEMM。** Runtime
先读取冻结的 `weight_format`：

```cpp
Q8_0 -> qwen3_z200_q8_linear_fp32(..., M, N, K, stream);
F16  -> qwen3_z200_f16_linear_fp32(..., M, N, K, stream);
```

两条路径都把 FP32 activation cast 为 FP16 并使用 FP32 accumulation；只有 Q8_0 路径
额外反量化 weight workspace。F16 权重常驻 device，不能错误调用 Q8_0 wrapper。

这个 wrapper 内部已经按同一条 stream 串联：

```text
x_fp32[M,K]
    -> 自定义 cast kernel 得到 x_fp16[M,K]

W_q8_0[N,K]
    -> 自定义 dequant kernel 得到 W_fp16[N,K]

x_fp16 @ W_fp16^T
    -> hipblasGemmEx(FP16, FP16, FP32 compute)
    -> y_fp32[M,N]
```

下表列出 Q8_0 分支的入口；F16 分支保持相同 M/N/K，把每个
`qwen3_z200_q8_linear_fp32` 替换为 `qwen3_z200_f16_linear_fp32`，Embedding 替换为
`qwen3_z200_launch_embedding_lookup_f16`：

| 模型算子 | 是否调用 hipBLAS | 实际入口 |
| --- | --- | --- |
| Q projection | 是 | `qwen3_z200_q8_linear_fp32(M=T,N=4096,K=4096)` |
| K projection | 是 | `qwen3_z200_q8_linear_fp32(M=T,N=1024,K=4096)` |
| V projection | 是 | `qwen3_z200_q8_linear_fp32(M=T,N=1024,K=4096)` |
| O projection | 是 | `qwen3_z200_q8_linear_fp32(M=T,N=4096,K=4096)` |
| gate projection | 是 | `qwen3_z200_q8_linear_fp32(M=T,N=12288,K=4096)` |
| up projection | 是 | `qwen3_z200_q8_linear_fp32(M=T,N=12288,K=4096)` |
| down projection | 是 | `qwen3_z200_q8_linear_fp32(M=T,N=4096,K=12288)` |
| LM Head | 是 | `qwen3_z200_q8_linear_fp32(M=1,N=151936,K=4096)` |
| Embedding | 否 | Q8_0 查表 kernel，只解 token 对应行 |
| RMSNorm / RoPE / KV write / SwiGLU / residual | 否 | 对应自定义 HIP kernel |
| causal GQA attention | 否 | 当前使用自定义 prefill/decode attention kernel |

所以完整边界是：**矩阵投影交给 hipBLAS，数据变换与非线性算子交给自定义 HIP kernel。** Attention 内部虽然包含 QK 点积和对 V 的加权，但它带 causal/KV-cache/GQA 访问语义，当前参考版本不把它当作普通 dense GEMM 调 hipBLAS。

## 1. 首版目标和硬件边界

目标运行环境是一张 Hygon Z200：`gfx906`、64 CU、wavefront 64、Fast F16、约 15.98 GiB device memory。当前支持：

- Qwen3-8B Dense；
- 单卡、单请求、`B=1`；
- F16 或 Q8_0 权重常驻 device memory；
- 自定义 HIP kernel 处理 Embedding、RMSNorm、RoPE、KV cache、GQA attention、SwiGLU 和残差；
- hipBLAS 处理所有带权重的线性层；
- 先保证 prefill/decode 数值正确，再优化 decode 性能。

> **多并发实现覆盖说明：** 上述 `B=1` 是本算子参考实现的基线。选择 Continuous Batching
> 时以 `runtime/continuous_batching.md` 为准，kernel 按 row 的 position、token row 和 KV view
> 隔离 scratch、scores 与 logits；Continuous-only 使用 dense sequence slots，只有同时选择
> Paged KV 时才使用 block tables 和 Paged Attention。

> **Tensor Parallel 覆盖说明：** 本文的 `qwen3_z200_q8_linear_fp32()` 是单卡 Q8_0 基线。基础 TP 任务使用 `distributed/tensor_parallel.md` 定义的非量化 F16 Rank-local 权重，直接调用 local hipBLAS GEMM，不先运行 Q8_0 整矩阵反量化 wrapper。本文的 gfx906、Wave64、stream、workspace 和非线性 HIP kernel 约束仍然有效。

当前源码是**正确性参考 kernel**，不是最终高性能实现。它没有 GGUF loader、模型类、buffer allocator 或完整 sampling policy，也没有 fused Q8_0 GEMV。这些由生成的 C++ runtime 补齐。源码已经提供确定性 greedy argmax primitive，但 stop、temperature、top-k/top-p 和生成状态仍由外部 sampler/engine 管理。

**首版执行决策：分别调用 Q/K/V 三个 linear，并分别调用 gate/up 两个 linear。** GGUF 本来就是这些独立权重，这样输出天然是 compact 多行 tensor，可直接连接当前 norm/KV/SwiGLU kernel。fused QKV/gate-up 只作为后续优化；在补齐第 6 节 helper 前不能用于 `T>1` prefill。

## 2. 全局张量布局和 dtype

除非某个接口另有说明，所有数组都是 contiguous row-major：

| 张量 | shape | 当前 dtype |
| --- | --- | --- |
| hidden state | `[T, H]` | FP32 |
| compact Q | `[T, 32, 128]` | FP32 |
| compact K/V | `[T, 8, 128]` | FP32 |
| attention 输出 | `[T, 32, 128]`，字节上等价于 `[T,H]` | FP32 |
| compact gate/up | 各自 `[T, 12288]` | FP32 |
| SwiGLU 输出 | `[T, 12288]` | FP32 |
| dense K/V cache | `[max_seq_len, 8, 128]` | FP32；Paged/Continuous scalable path 为 FP16 |
| logits | `[1, 151936]` | FP32 |
| 模型矩阵权重 | `[N,K]`，HF 的 `[out,in]` | 冻结格式 F16 或 Q8_0 |
| hipBLAS 激活/权重 workspace | `[M,K]` / `[N,K]` | FP16 |

线性层统一采用：

```text
Y[M,N] = X[M,K] @ W[N,K]^T
```

`M=T` 表示同时处理本次 token；decode 时 `M=1`。所有尺寸参数都是**元素数或维度**，不是字节数。

当前 dtype 数据流为：

```text
FP32 hidden
  -> FP32 custom op
  -> cast FP16 activation + dequant FP16 weight
  -> hipBLAS FP32 accumulation/output
  -> FP32 custom op
```

## 3. 当前公开算子索引

所有 public wrapper 都在参考源码中以 `extern "C"` 导出。

| 接口 | 主要作用 | 输出 |
| --- | --- | --- |
| `qwen3_z200_launch_cast_fp32_to_fp16` | FP32 激活转 FP16 | FP16 |
| `qwen3_z200_launch_dequant_q8_0_to_fp16` | 整个 Q8_0 矩阵解到 FP16 workspace | FP16 |
| `qwen3_z200_launch_embedding_lookup_q8_0` | 查找 token 行并在读取时解量化 | FP32 `[T,H]` |
| `qwen3_z200_launch_embedding_lookup_f16` | 从常驻 F16 embedding 查找 token 行 | FP32 `[T,H]` |
| `qwen3_z200_f16_linear_fp32` | activation cast + resident F16 `hipblasGemmEx` | FP32 `[M,N]` |
| `qwen3_z200_q8_linear_fp32` | cast + dequant + `hipblasGemmEx` | FP32 `[M,N]` |
| `qwen3_z200_launch_embedding_lookup` | FP32 embedding 查表备用路径 | FP32 `[T,H]` |
| `qwen3_z200_launch_rms_norm` | hidden/最终 RMSNorm | FP32 `[rows,dim]` |
| `qwen3_z200_launch_per_head_rms_norm` | Q/K 每个 head 独立 RMSNorm | FP32 |
| `qwen3_z200_launch_rope` | 对 Q 或 K 原地施加 RoPE | 原地 FP32 |
| `qwen3_z200_launch_kv_cache_write` | 连续写入一层 K/V cache | FP32 cache |
| `qwen3_z200_launch_prefill_gqa_attention` | causal prefill GQA | FP32 `[T,H]` |
| `qwen3_z200_launch_decode_gqa_attention` | 单 token decode GQA | FP32 `[H]` |
| `qwen3_z200_launch_greedy_sample` | 对 device logits 做确定性 argmax | device `int32_t[1]` |
| `qwen3_z200_launch_swiglu` | `silu(gate) * up` | FP32 `[T,I]` |
| `qwen3_z200_launch_add` | 两个数组相加 | FP32 |
| `qwen3_z200_launch_add_inplace` | 将 src 累加到 dst | 原地 FP32 |

生成框架时应为这些接口建立一个独立头文件，声明必须与 `.hip.cpp` 完全一致；不要在多个 `.cpp` 中手写不同版本的 prototype。

## 4. Q8_0、反量化与 hipBLAS 线性层

### 4.1 Q8_0 内存契约

参考源码使用与 GGML/GGUF Q8_0 相同的 34-byte block：

```cpp
struct alignas(2) BlockQ8_0 {
    __half d;       // 一个 FP16 scale
    int8_t qs[32];  // 32 个 signed quant
};

static_assert(sizeof(BlockQ8_0) == 34);
```

第 `i` 个元素的反量化公式：

```text
block  = weight[i / 32]
offset = i % 32
value  = fp32(block.d) * fp32(block.qs[offset])
```

Loader 必须保证：

- tensor 的 GGUF type 确实是 Q8_0；
- 被量化矩阵最后一维 `K % 32 == 0`；
- 数据起点满足 `BlockQ8_0` 的对齐要求；
- tensor shape 和字节数严格匹配，不能把 GGUF metadata/padding 当权重；
- QKV/gate-up 融合只能按输出行拼接完整量化行，不能打乱一行中的 block。

### 4.2 单独反量化接口

```cpp
hipError_t qwen3_z200_launch_dequant_q8_0_to_fp16(
    __half* out,
    const BlockQ8_0* weight,
    size_t n_elements,
    hipStream_t stream);
```

`n_elements` 是解量化后的标量元素数，必须是 32 的倍数；`out` 至少容纳 `n_elements * sizeof(__half)` bytes。kernel 每个 thread 负责一个标量，结果写入 FP16。

这个接口适合调试和通用线性封装。不要在每次调用中临时申请 `out`；它应指向 runtime 初始化时分配的可复用权重 workspace。

### 4.3 通用 Q8_0 线性接口

```cpp
hipblasStatus_t qwen3_z200_q8_linear_fp32(
    hipblasHandle_t handle,
    float* out,                         // [M,N]
    const float* x,                     // [M,K]
    const BlockQ8_0* weight,            // [N,K]
    __half* x_fp16_workspace,
    size_t x_workspace_elements,
    __half* weight_fp16_workspace,
    size_t weight_workspace_elements,
    int m, int n, int k,
    hipStream_t stream);
```

一次调用按同一 stream 的顺序 enqueue：

```text
1. x_fp32[M,K] -> x_fp16_workspace[M,K]
2. W_q8_0[N,K] -> weight_fp16_workspace[N,K]
3. hipblasGemmEx: FP16 x FP16 -> FP32 accumulation -> out_fp32[M,N]
```

workspace 最小容量：

```text
x_workspace_elements      >= M * K
weight_workspace_elements >= N * K
```

这里的容量单位是 FP16 **元素数**。实际申请字节数要再乘 `sizeof(__half)`。

调用示例：

```cpp
hipblasStatus_t st = qwen3_z200_q8_linear_fp32(
    handle,
    out_fp32,
    x_fp32,
    weight_q8,
    x_fp16_workspace,
    x_workspace_elements,
    weight_fp16_workspace,
    weight_workspace_elements,
    M, N, K,
    stream);
if (st != HIPBLAS_STATUS_SUCCESS) {
    // 报出 layer、算子名、M/N/K 后终止本次 forward。
}
```

内部用 hipBLAS column-major 视图实现 row-major `X @ W^T`，其关键配置是 `HIPBLAS_OP_T, HIPBLAS_OP_N`、两个输入 `HIP_R_16F`、输出 `HIP_R_32F` 和 `HIPBLAS_COMPUTE_32F`。调用者不要再次转置权重。

普通 hipBLAS INT8 GEMM 不能直接消费“FP32 激活 + GGUF Q8_0 权重”，因为 Q8_0 scale 每 32 个 K 元素变化。首版不能把 `BlockQ8_0*` 强转成 `int8_t*` 后直接 GEMM。

### 4.4 每层线性调用参数

| 线性层 | M | N | K | 输出 shape |
| --- | ---: | ---: | ---: | --- |
| Q projection | `T` | 4096 | 4096 | `[T,4096]` |
| K projection | `T` | 1024 | 4096 | `[T,1024]` |
| V projection | `T` | 1024 | 4096 | `[T,1024]` |
| O projection | `T` | 4096 | 4096 | `[T,4096]` |
| gate projection | `T` | 12288 | 4096 | `[T,12288]` |
| up projection | `T` | 12288 | 4096 | `[T,12288]` |
| down projection | `T` | 4096 | 12288 | `[T,4096]` |
| LM Head | **1** | 151936 | 4096 | `[1,151936]` |

prefill 也只对最后一个 hidden row 调 LM Head，所以它的 `M=1`，不是 prompt 长度 `T`。

## 5. 非线性算子的调用规则

### 5.1 Q8_0 Embedding

```cpp
qwen3_z200_launch_embedding_lookup_q8_0(
    out_fp32, embedding_q8, token_ids_device,
    T, V, H, stream);
```

- `token_ids_device` 是 device pointer；
- 输出为 compact `[T,H]` FP32；
- `H % 32 == 0`；
- 只解本次 token 对应的行，禁止先展开整个 `[V,H]`；
- 当前 kernel 遇到越界 token id 会写 0，runtime 应在更上层校验并报告错误。

### 5.2 Hidden RMSNorm

```cpp
qwen3_z200_launch_rms_norm(
    out, x, weight, rows, dim, eps, stream);
```

- attention/FFN norm：`rows=T, dim=4096`；
- final norm：若只生成最后 logits，可传最后一行，`rows=1, dim=4096`；
- `weight` 为 FP32 `[dim]`；
- kernel 使用 256 threads 和 256 个 FP32 shared elements 做归约；
- `out` 与 `x` 使用 `__restrict__` 声明，按契约视为不可 alias，准备独立输出 buffer。

### 5.3 Q/K per-head RMSNorm

```cpp
qwen3_z200_launch_per_head_rms_norm(
    out, x, weight,
    T, n_heads, 128, row_stride,
    eps, stream);
```

- Q：`n_heads=32`，compact `row_stride=4096`；
- K：`n_heads=8`，compact `row_stride=1024`；
- `weight` 是 `[128]`，每个 head 复用；
- V 不做 per-head RMSNorm；
- 与 hidden RMSNorm 一样，输入输出按不 alias 处理。

### 5.4 RoPE

```cpp
qwen3_z200_launch_rope(
    x, cos_table, sin_table,
    T, n_heads, 128, row_stride,
    start_pos, max_position,
    rope_mode, stream);
```

- RoPE 对已经 per-head RMSNorm 的 Q/K **原地**执行；
- Qwen3 使用 half-split `rotate_half` 语义，对应 `ROPE_NEOX = 1`；
- compact Q/K 的 `row_stride` 分别是 4096/1024；
- table layout 是 `[max_position, head_dim/2]`；
- 第 `t` 行的绝对位置是 `start_pos + t`；
- runtime 必须在调用前检查 `start_pos + T <= max_position`，不能依赖 kernel 静默跳过越界位置。

### 5.5 KV cache 写入

```cpp
qwen3_z200_launch_kv_cache_write(
    k_compact, v_compact,
    layer_k_cache, layer_v_cache,
    T, 8, 128,
    max_seq_len, start_pos,
    stream);
```

输入必须是 compact `[T,8,128]`，cache layout 是 `[max_seq_len,8,128]`。写入的是已经做完 Q/K norm 和 RoPE 的 K；V 直接来自 projection。每层有独立的 K/V cache。

kernel 遇到越界 position 会直接 return，runtime 必须提前检查 `start_pos + T <= max_seq_len`，否则可能出现“调用成功但 cache 没写完整”的静默错误。

### 5.6 Prefill GQA attention

```cpp
qwen3_z200_launch_prefill_gqa_attention(
    q_compact, layer_k_cache, layer_v_cache, out,
    T, start_pos, max_seq_len,
    32, 8, 128,
    1.0f / sqrtf(128.0f), stream);
```

- Q 输入必须 compact `[T,32,128]`；
- grid 是 `[T,32]`，每个 block 处理一个 token 的一个 Q head；
- `q_head / 4` 得到对应 KV head；
- 第 `t` 个 token 只看 `0 .. start_pos+t`，已经包含 causal mask；
- 这是朴素两遍 softmax 的正确性 kernel，长 prompt 会慢。

### 5.7 Decode GQA attention

```cpp
qwen3_z200_launch_decode_gqa_attention(
    q_one_token, layer_k_cache, layer_v_cache,
    out, scores_workspace,
    current_pos, max_seq_len,
    32, 8, 128,
    1.0f / sqrtf(128.0f), stream);
```

- `current_pos` 是刚写入 K/V 的绝对位置；读取长度为 `current_pos + 1`；
- `scores_workspace` 最少为 `[32,max_seq_len]` FP32，可在 36 层间串行复用；
- `out` 是 `[32,128]`，字节上可作为 `[1,4096]`；
- caller 必须验证 `0 <= current_pos < max_seq_len` 和 `32 % 8 == 0`。

### 5.8 SwiGLU 和残差

```cpp
qwen3_z200_launch_swiglu(out, gate, up, T * 12288, stream);
qwen3_z200_launch_add(out, residual, branch, T * 4096, stream);
// 或
qwen3_z200_launch_add_inplace(residual, branch, T * 4096, stream);
```

SwiGLU 计算 `silu(gate) * up`。现有接口要求 `gate` 和 `up` 都是独立 compact `[T,12288]`；它不能直接正确读取 `T>1` 的 row-major fused `[T,24576]` 两个半区，相关缺口见下一节。

### 5.9 Greedy argmax

```cpp
qwen3_z200_launch_greedy_sample(
    d_logits, 151936, d_next_token, stream);
hipMemcpyAsync(
    &next_token, d_next_token, sizeof(int32_t),
    hipMemcpyDeviceToHost, stream);
hipStreamSynchronize(stream);
```

- `d_logits` 是 LM Head 产生的 FP32 `[151936]`；
- `d_next_token` 是外部 sampler 初始化时一次性分配的 device `int[1]`，不能每个 decode step 临时 `hipMalloc/hipFree`；
- kernel 对相同最大值选择较小 token id，适合 `temperature=0` 的确定性基准；
- stop token、temperature、top-k/top-p、repetition penalty 不属于该 kernel。需要修改 logits 的策略在未实现 GPU logit processor 前走 CPU sampler；
- kernel、D2H copy 和 forward 使用同一条 runtime stream，确保读取的是刚产生的 logits。

## 6. 可选 fused 优化的两个布局前置条件

首版独立 Q/K/V 和 gate/up 路径不需要本节 helper。若后续为了减少反量化/GEMM 调度而融合权重，必须注意：线性层输出虽然数学上可以 `split`，但 C++ 指针偏移不一定得到跨 token compact tensor。当前参考源码还没有下面两个 pack/strided helper；不补齐时 decode 的 `T=1` 可能工作，而 prefill 的 `T>1` 会读错行。

### 6.1 Fused QKV 必须 split/pack

`q8_linear` 输出是 row-major `[T,6144]`：

```text
row 0: Q0[4096] K0[1024] V0[1024]
row 1: Q1[4096] K1[1024] V1[1024]
...
```

仅令 `q = qkv + 0`、`k = qkv + 4096`、`v = qkv + 5120`，三者在 `T>1` 时都不是各自的 compact 多行数组。prefill attention 要求 compact Q，KV writer 要求 compact K/V。

fused 优化必须选择一种方式：

1. **推荐：**补一个 HIP split/pack kernel，将 fused 输出复制为 compact `Q[T,4096]`、`K[T,1024]`、`V[T,1024]`，然后调用现有 per-head norm、RoPE、KV write 和 attention；
2. 回退到首版的 Q、K、V 三个独立线性层，直接得到 compact 输出，但会增加两次完整权重解量化和 GEMM 调度。

不能只在文档或 C++ 类型中做 reshape 而不搬运数据。必须用 `T>=2` 的递增值测试验证每一行。

### 6.2 Fused gate-up 必须 pack 或使用 strided SwiGLU

fused gate-up 输出是 `[T,2I]`：

```text
row 0: gate0[I] up0[I]
row 1: gate1[I] up1[I]
...
```

现有 `qwen3_z200_launch_swiglu()` 假设 gate/up 分别 compact。对 `T>1` 仅用 `up = gate_up + I` 会让线性遍历跨入错误半区。

fused 优化必须选择：

1. 补一个 strided SwiGLU kernel，输入行 stride 为 `2I`，每行分别读 `row[0:I]` 和 `row[I:2I]`；或
2. 先把 gate/up pack 为两个 compact buffer，再调用现有 SwiGLU。

对 fused 路径而言，这两个布局问题是 prefill 正确性的前置条件，不是可以忽略的性能细节。

## 7. 单卡完整调用顺序

### 7.1 初始化阶段

```text
read config -> 校验 H/I/L/Nq/Nkv/D/V
read GGUF -> 校验并上传冻结格式的 F16/Q8_0 权重
create HIP stream + hipBLAS handle
hipblasSetStream(handle, stream)
build cos/sin RoPE table
allocate 36-layer KV cache
allocate reusable FP32 activations/intermediates
allocate decode scores workspace
allocate x_fp16_workspace
allocate shared weight_fp16_workspace
create external sampler and allocate d_next_token[1]
```

所有 workspace 只分配一次。逐层 forward 和逐 token decode 中禁止 `hipMalloc/hipFree`。

### 7.2 Prefill

```text
Q8_0 embedding lookup -> x[T,H]

for layer = 0..35:
    residual = x
    RMSNorm(x) -> h

    q8_linear(h, q_weight, M=T,N=4096,K=4096) -> q_compact
    q8_linear(h, k_weight, M=T,N=1024,K=4096) -> k_compact
    q8_linear(h, v_weight, M=T,N=1024,K=4096) -> v_compact
    per_head_rms(q_compact) -> q_norm
    per_head_rms(k_compact) -> k_norm
    RoPE(q_norm, start_pos, NEOX)
    RoPE(k_norm, start_pos, NEOX)
    KV write(k_norm, v_compact, start_pos)
    prefill GQA(q_norm, layer cache) -> attn

    q8_linear(attn, o_proj, M=T,N=4096,K=4096) -> attn_proj
    add(residual, attn_proj) -> x

    residual = x
    RMSNorm(x) -> h
    q8_linear(h, gate_weight, M=T,N=12288,K=4096) -> gate
    q8_linear(h, up_weight, M=T,N=12288,K=4096) -> up
    SwiGLU(gate, up) -> ffn[T,12288]
    q8_linear(ffn, down, M=T,N=4096,K=12288) -> down
    add(residual, down) -> x

final RMSNorm 仅处理 x[T-1]
q8_linear(last_hidden, lm_head, M=1,N=151936,K=4096) -> logits[V]
```

若 prompt 分 chunk，`start_pos` 是当前 chunk 的绝对起点；每个 chunk 的 attention 必须能看到之前已经写入的 cache。仅在最后一个 chunk 执行 final norm 和 LM Head。

### 7.3 Decode

```text
Q8_0 embedding lookup(token_id) -> x[1,H]
current_pos = cache.length

for layer = 0..35:
    attention RMSNorm
    q8 Q/K/V independent linears, M=1
    per-head Q/K RMSNorm
    Q/K RoPE(position=current_pos)
    write K/V at current_pos
    decode GQA(current_pos)
    q8 O linear, M=1
    residual add
    FFN RMSNorm
    q8 gate/up independent linears, M=1
    SwiGLU
    q8 down linear, M=1
    residual add

final RMSNorm
q8 LM Head, M=1
cache.length = current_pos + 1
external greedy sampler -> d_next_token[1] -> copy one int32 to host
```

Runtime 在 forward 成功后提交 cache length，并把 logits 交给外部 sampler。后续若启用 fused 路径，decode 也应复用与 prefill 相同的 split/pack 接口，减少两条执行路径产生的错误。

## 8. 单卡 workspace 与显存规划

### 8.1 权重 workspace

一个 FP16 权重 workspace 可以在所有线性层间串行复用。若 LM Head 不分块，最大容量是：

```text
151936 * 4096 FP16 elements
= 1,244,659,712 bytes
= 1187 MiB ≈ 1.159 GiB
```

首版独立 gate/up 每个只需 96 MiB；可选 fused gate-up 也只有 192 MiB，所以都不能作为全局最大值。LM Head 首版可以不分块；若启动时 `hipMemGetInfo` 显示余量不足，再沿 vocab 行分块解量化和 GEMM。

### 8.2 激活 workspace

FP16 激活 workspace 至少容纳当前最大 `M*K`。decode 的 `M=1` 很小；prefill 可能由 down projection 的 `T*12288` 决定。应按允许的 prefill chunk size 计算并复用，不能直接按 40960 token 申请。

FP32 激活/中间 buffer 应做生命周期复用，但不能让一个 kernel 的输入被异步覆盖。推荐至少规划：

- hidden/residual ping-pong `[T,H]`；
- norm output `[T,H]`；
- compact Q/K/V 独立输出；若启用 fused 优化，再增加 fused QKV 临时区；
- attention/O projection `[T,H]`；
- compact gate/up 和 SwiGLU `[T,I]`；若启用 fused 优化，再增加 fused gate-up 临时区；
- logits `[V]`；
- decode scores `[32,max_seq_len]`。

### 8.3 KV cache

当前单序列/TP-only dense FP32 cache 在 4096 context 时：

```text
2 * 36 * 4096 * 8 * 128 * 4 bytes = 1.125 GiB
```

40960 context 会达到 11.25 GiB，单卡无法再同时容纳 Q8_0 模型和线性 workspace。默认
context 建议 4096 或更小，并在启动时按实际空闲显存拒绝不安全配置。选择 Paged KV 或
Continuous Batching 时，冻结合同改为 FP16 KV，并必须使用对应 FP16 writer/attention，不能
只按 FP16 预算却继续分配 `float*`。

## 9. Stream、错误处理和同步

- cast、dequant、GEMM、后续 custom op 必须 enqueue 到同一 stream，或用 event 明确建立跨 stream 依赖；
- 共享 weight workspace 时，不得并发执行两个使用它的线性层；
- 每个 wrapper 返回后检查 `hipError_t` / `hipblasStatus_t`；错误日志包含 layer、op、shape、position；
- `hipGetLastError()` 只能发现 launch 配置错误，异步执行错误需要在调试边界用 `hipStreamSynchronize()` 捕获；
- 正常逐层路径不要调用 `hipDeviceSynchronize()`；
- `qwen3_z200_q8_linear_fp32()` 会设置 handle 的 stream，并在需要时临时切换/恢复 hipBLAS host pointer mode；caller 不要在同一个 handle 上并发修改这些状态；
- cache、RoPE、token id 的越界必须由 caller 先检查，不能依赖 kernel 静默 return 或写 0。

## 10. 必须先做的正确性测试

1. Q8_0 单块：构造已知 scale/quant，比较反量化 FP16 与 CPU 公式。
2. Q8 linear 小矩阵：CPU 反量化 + FP32 matmul 对比 `qwen3_z200_q8_linear_fp32()`。
3. Embedding：包含重复 token，验证只返回对应行；另外测试越界由 runtime 拒绝。
4. RMSNorm：对比 CPU reference，并分别测试 hidden `[T,4096]` 和 per-head `[T,N,128]`。
5. RoPE：测试 `start_pos=0` 和非零位置，确认使用 `ROPE_NEOX`。
6. 独立 Q/K/V：必须用 `T>=2` 且每一行数值不同，验证 compact layout；启用 fused 优化时再增加 pack stride 测试。
7. 独立 gate/up + SwiGLU：必须用 `T>=2`，验证每行 gate/up 配对；启用 fused 优化时再增加 strided/pack 测试。
8. KV + attention：比较 prefill 最后位置与逐 token decode 最后位置输出。
9. 整层：比较 attention residual、FFN residual。
10. 全模型：固定 prompt 比较最后 logits/greedy token，再做多步 decode。

静态源码契约测试只能检查接口和关键配置存在，不能替代 `gfx906` 实卡编译与数值测试。
以下命令只展示 system-owned `build.sh` 内部应产生的编译效果，Implementer 不得直接执行：

```bash
hipcc --offload-arch=gfx906 -c qwen3_z200_kernels.hip.cpp \
  -o qwen3_z200_kernels.o
```

最终链接 hipBLAS，并在同一台 Z200 上运行上述测试。

## 11. 正确后再做的优化顺序

1. decode `M=1`：实现直接读取 Q8_0 block 的 fused GEMV，去掉整矩阵 FP16 解量化；
2. 将 K/V cache 与 attention 统一为 FP16，降低显存和带宽；
3. 优化 decode/prefill attention，替换当前朴素 softmax kernel；
4. 将 QKV split、Q/K norm、RoPE、KV write 做适度融合；
5. 实测显存不足时再做 LM Head vocab 分块；
6. 对当前任务未选择的多请求、Paged KV、图捕获或 TP 不做隐式扩展；已选择的能力必须在
   correctness vertical slice 中实现，不能被当作“以后再考虑”。

不要在正确性测试通过前同时改量化格式、attention 算法和执行调度，否则最终 logits 出错时很难定位生产者/消费者。
