# 原生Operator接口与正确性契约

> 实现指南：`03_operators/`。Backend能力和错误边界见
> `03_operators/05_hip_blas_backend.md`。

目标是建立“正确、可分派、可测量”的算子层，而不是一开始追求 Fusion。

## 1. Operator 接口与 Registry

```cpp
struct OperatorContext {
  int device = -1;
  BackendStream stream;
  const BackendCapabilities* capabilities = nullptr;
  WorkspacePool* workspace_pool = nullptr;
};

class OperatorSet {
 public:
  virtual Status RmsNorm(const TensorView& input,
                         const TensorView& weight,
                         float epsilon,
                         TensorView output,
                         const OperatorContext& ctx) = 0;
  virtual Status Rope(TensorView q,
                      TensorView k,
                      const TensorView& positions,
                      const RopeConfig& config,
                      const OperatorContext& ctx) = 0;
  virtual Status SiluMultiply(const TensorView& gate,
                              const TensorView& up,
                              TensorView output,
                              const OperatorContext& ctx) = 0;
};
```

分派 Key 至少包含：

```cpp
struct OperatorKey {
  OperatorKind kind;
  DType input_dtype;
  DType output_dtype;
  Layout layout;
  std::string architecture;
};
```

Debug/Profile 日志输出最终 Implementation ID，例如 `rms_norm.hip.reference.fp32`。禁止只按 `gpu_name` 分派。

## 2. BLAS Adapter

由于 DTK 提供的 BLAS Header/API 版本需要现场验证，Model Layer 不应直接调用具体 BLAS API，而应依赖统一接口：

```cpp
enum class MatrixOp { kNone, kTranspose };

struct GemmProblem {
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
  MatrixOp op_a;
  MatrixOp op_b;
  DType a_dtype;
  DType b_dtype;
  DType c_dtype;
  DType compute_dtype;
  std::int64_t lda;
  std::int64_t ldb;
  std::int64_t ldc;
};

class Gemm {
 public:
  virtual Status Run(const GemmProblem& problem,
                     const void* a,
                     const void* b,
                     void* c,
                     BackendStream stream) = 0;
};
```

如果 Probe 证实 `hipBLAS` 接口可用，可以实现 RAII Handle：

```cpp
class HipBlasHandle {
 public:
  static Result<HipBlasHandle> Create() {
    hipblasHandle_t handle = nullptr;
    HIPBLAS_RETURN_IF_ERROR(hipblasCreate(&handle));
    return HipBlasHandle(handle);
  }
  ~HipBlasHandle() {
    if (handle_ != nullptr) RecordIfError(hipblasDestroy(handle_));
  }
  Status SetStream(hipStream_t stream) {
    HIPBLAS_RETURN_IF_ERROR(hipblasSetStream(handle_, stream));
    return Status::Ok();
  }
 private:
  explicit HipBlasHandle(hipblasHandle_t h) : handle_(h) {}
  hipblasHandle_t handle_ = nullptr;
};
```

只有目标 DTK Header/Library 的 Compile/Link/Runtime Probe 通过后才能启用。矩阵 Layout、Leading Dimension、Transpose 和 Compute Type 必须在 Adapter 中集中处理。

## 3. RMSNorm Float Reference Kernel

下面给出易验证的 FP32 基线：一个 Block 处理一行。它不是最终高性能实现，但可作为 BF16/FP16 Kernel 的 Reference。

```cpp
__global__ void RmsNormFp32Kernel(const float* input,
                                  const float* weight,
                                  float* output,
                                  int rows,
                                  int hidden,
                                  float epsilon) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float value = input[row * hidden + col];
    sum_sq += value * value;
  }

  extern __shared__ float scratch[];
  scratch[threadIdx.x] = sum_sq;
  __syncthreads();

  // 该 Reduction 要求 blockDim.x 为 2 的幂；Host Launch 前必须验证。
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }

  const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(hidden) + epsilon);
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    output[row * hidden + col] =
        input[row * hidden + col] * inv_rms * weight[col];
  }
}
```

Host Wrapper：

```cpp
Status LaunchRmsNormFp32(const TensorView& input,
                         const TensorView& weight,
                         float epsilon,
                         TensorView output,
                         hipStream_t stream) {
  RETURN_IF_ERROR(ValidateRmsNormViews(input, weight, output));
  const int rows = static_cast<int>(input.shape[0]);
  const int hidden = static_cast<int>(input.shape[1]);
  const int threads = SelectPowerOfTwoThreads(hidden);  // <= device limit
  RmsNormFp32Kernel<<<rows, threads, threads * sizeof(float), stream>>>(
      static_cast<const float*>(input.data),
      static_cast<const float*>(weight.data),
      static_cast<float*>(output.data), rows, hidden, epsilon);
  HIP_RETURN_IF_ERROR(hipGetLastError());
  return Status::Ok();
}
```

BF16/FP16 版本应在 Float 中累加，并显式转换 Load/Store。转换类型和 Intrinsic 必须在 DTK 编译探针验证。

## 4. RoPE FP32 基线

```cpp
__global__ void RopeFp32Kernel(float* tensor,
                               const int* positions,
                               const float* cos_cache,
                               const float* sin_cache,
                               int tokens,
                               int heads,
                               int head_dim) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int pairs_per_head = head_dim / 2;
  const int total_pairs = tokens * heads * pairs_per_head;
  if (pair >= total_pairs) return;

  const int local_pair = pair % pairs_per_head;
  const int token_head = pair / pairs_per_head;
  const int token = token_head / heads;
  const int position = positions[token];
  const int base = token_head * head_dim + 2 * local_pair;
  const int cache = position * pairs_per_head + local_pair;

  const float x0 = tensor[base];
  const float x1 = tensor[base + 1];
  const float c = cos_cache[cache];
  const float s = sin_cache[cache];
  tensor[base] = x0 * c - x1 * s;
  tensor[base + 1] = x0 * s + x1 * c;
}
```

必须验证 `head_dim` 偶数、Position 不越过 Cache、Q/K Layout 与模型 RoPE 语义一致。若模型采用不同 Rotary Layout，需要独立 Implementation ID。

## 5. SiLU * Up 基线

```cpp
__global__ void SiluMultiplyFp32(const float* gate,
                                 const float* up,
                                 float* output,
                                 std::size_t elements) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < elements;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float x = gate[i];
    output[i] = (x / (1.0f + expf(-x))) * up[i];
  }
}
```

后续可以将 Gate/Up Projection 和 Activation 做 Pack/Fusion，但必须与此基线逐元素对比。

## 6. Attention 实现阶梯

### 6.1 Tiny Reference

```text
QK GEMM -> Scale/Mask Kernel -> Stable Softmax -> PV GEMM
```

只用于小序列测试，允许物化 Score。

### 6.2 Prefill Tiled Path

按 Query/Key Tile 加载，在线维护 Row Max 和 Row Sum，避免 `[seq, seq]` 全量 Score。必须支持变长 `cu_seqlens` 和 Causal Position。

### 6.3 Decode Paged Path

Kernel 直接遍历每个请求 Block Table：

```cpp
for (int logical = 0; logical < kv_length; ++logical) {
  const int table_slot = logical / block_size;
  const int block_offset = logical % block_size;
  const int physical_block = block_table[table_slot];
  // 计算 K/V 地址，更新在线 Softmax/Context。
}
```

禁止先 Gather 完整连续 KV 再 Attention。

## 7. Kernel Launch 与错误边界

```cpp
kernel<<<grid, block, shared_bytes, stream>>>(...);
HIP_RETURN_IF_ERROR(hipGetLastError());

// Debug 模式可在每个算子同步；生产模式在 Step/Event 边界观察异步错误。
if (debug_sync) HIP_RETURN_IF_ERROR(hipStreamSynchronize(stream));
```

不得在每个生产算子后无条件 `hipDeviceSynchronize()`。

## 8. 数值与测试矩阵

至少覆盖：

```text
DType: FP32 reference、BF16、FP16（能力通过时）
Shape: 非方阵 GEMM、尾部维度、不同 Token 数
RMSNorm: hidden 小于/大于 blockDim、epsilon、全零/大值
RoPE: position=0/边界、不同 head_dim、Q/K 一致性
Attention: 变长、Causal、GQA、Block Boundary、全 Mask 防护
Sampling: Argmax Tie、NaN/Inf 防护
```

每个 Specialized/Fused Kernel 要登记：

```cpp
struct KernelRegistration {
  std::string implementation_id;
  ShapePredicate supports;
  DType input_dtype;
  std::vector<std::string> architectures;
  double absolute_tolerance;
  double relative_tolerance;
  std::string fallback_id;
};
```

只在一个 Shape 上更快而破坏其他 Shape 的 Kernel 不能成为无条件默认值。
