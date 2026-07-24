# K100/gfx928 INT8 MMAC（TensorCore）通用 GEMM 与 Split-K 选择

## 1. 结论

目标算子：

```text
Y_bf16 = bf16((A_int8[M,K] @ W_int8[K,N])
              * A_scale[M,None] * W_scale[None,N])
```

当前 K100/gfx928、TP=1 真权重实测结论：

- `M<=16 && K>=2048`：优先使用专用 split-K SDOT4。本轮 16 个 small-M、
  large-K case 中，最佳 split-K 全部快于 MMAC/dispatcher，优势 `1.32x~4.11x`。
- `M>=32`：使用四 Wave MMAC general kernel。当前 split-K 实验 kernel 只支持
  `M<=16`，不能用于 large-M。
- `M<=8 && K<=2048`：`gemm_opt_MMAC.cpp` 当前 dispatcher 走 small-M SDOT4，
  不是 MMAC；若允许 INT32 workspace，应替换为最佳 split-K。
- `M=16` 已走 MMAC，但 K=2048/4096/8192 的 TP=1 shape 仍是 split-K 更快。

数字来自一次 `3 warmup + 10 samples` GPU Event 中位数，没有锁频和方差，只应作为
当前机器上的 dispatch/tuning 起点。

## 2. 布局与正确性

```text
A: [M,K], row-major, INT8
W: [K,N], row-major, INT8
Y: [M,N], row-major, BF16
```

W 不在 host 侧转置。kernel 沿 N 连续读取 W，在写 LDS 时转成 K 连续布局。

本文所有 iteration-008、MMAC/dispatcher 和 split-K case 均通过完整 BF16 输出核对：

```text
abs(candidate - Triton) <= 1e-3
max_abs = 0
mismatches = 0
```

## 3. 四 Wave MMAC 设计

源码：`kernels/gemm_opt_MMAC.cpp`

```text
BM=16, BN=64, BK=64, block=256 threads=4 Wave64
wave 0 -> C[0:16,  0:16]
wave 1 -> C[0:16, 16:32]
wave 2 -> C[0:16, 32:48]
wave 3 -> C[0:16, 48:64]
```

四个 wave 共享一次加载的 `A[16,64]`，各自消费 B 的 16 列。相比四个独立
16x16 block，A 的 global traffic 理论上降为 1/4；B 数据量不变，但改为 256 个
线程共同完成连续向量搬运。

### 3.1 128-bit global load

```cpp
struct alignas(16) GlobalVec128 { int32_t dwords[4]; };

GlobalVec128 av = *reinterpret_cast<const GlobalVec128*>(
    a + int64_t(gm) * K + k0 + lk);
GlobalVec128 bv = *reinterpret_cast<const GlobalVec128*>(
    w + int64_t(k0 + lk) * N + gn);
```

gfx928 device ISA 已确认主路径生成：

```asm
global_load_dwordx4
v_mmac_i32_16x16x32_i8
```

### 3.2 MMAC fragment/lane layout

```cpp
const int lane = threadIdx.x & 63;
const int wave = threadIdx.x / 64;
const int row_in_tile = lane & 15;
const int k_group = lane >> 4;

for (int kk = 0; kk < BK; kk += 32) {
    const int fragment_k = kk + k_group * 8;
    int8x8_t af = load_lds_int8x8(&a_tile[row_in_tile][fragment_k]);
    int8x8_t bf = load_lds_int8x8(
        &b_tile[wave * 16 + row_in_tile][fragment_k]);
    acc = mmac_i32_16x16x32_i8(af, bf, acc);
}
```

`load_lds_int8x8` 的两个 dword load 在 ISA 中合并为 `ds_read2_b32`。fragment layout
已由全部实测 shape 的 `max_abs=0` 验证，不是当前主要瓶颈。

一个 lane 的四个 accumulator 输出到：

```cpp
col = block_n + wave * 16 + k_group + i * 4; // i=0..3
```

### 3.3 MMAC kernel 的主要实现

下面是通用 kernel 的关键骨架；边界 tail 的标量 fallback 在实际源码中保留，不能
为了追求向量化而取消，否则 N/K 非完整 tile 时会越界：

```cpp
template <int BM, int BN, int BK>
__global__ __launch_bounds__(256)
void w8a8_scaled_gemm_kernel(
    const int8_t* a, const int8_t* w,
    const float* a_scale, const float* w_scale,
    hip_bfloat16* y, int M, int N, int K) {
  constexpr int PAD = BK + 4;
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wave = tid >> 6;
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  __shared__ int8_t as[BM][PAD];
  __shared__ int8_t bs[BN][PAD];
  int32x4_t acc = {0, 0, 0, 0};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A: contiguous K dimension, 16 bytes per request.
    for (int v = tid; v < BM * BK / 16; v += 256) {
      int m = v / (BK / 16), k = (v % (BK / 16)) * 16;
      load_vec128_or_zero(as[m] + k, a + int64_t(blockIdx.y * BM + m) * K
                                      + k0 + k, M, K, ...);
    }
    // B: contiguous N dimension; store transposed into LDS.
    for (int v = tid; v < BK * BN / 16; v += 256) {
      int k = v / (BN / 16), n = (v % (BN / 16)) * 16;
      vec128_or_zero x = load_vec128_or_zero(
          w + int64_t(k0 + k) * N + blockIdx.x * BN + n, N, ...);
      for (int j = 0; j < 16; ++j) bs[n + j][k] = x[j];
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 32) {
      int fk = kk + kg * 8;
      int8x8_t af = load_lds_int8x8(&as[row16][fk]);
      int8x8_t bf = load_lds_int8x8(&bs[wave * 16 + row16][fk]);
      acc = mmac_i32_16x16x32_i8(af, bf, acc);
    }
    __syncthreads();
  }

  if (blockIdx.y * BM + row16 < M) {
    float s = a_scale[blockIdx.y * BM + row16];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int n = blockIdx.x * BN + wave * 16 + kg + i * 4;
      if (n < N) y[int64_t(blockIdx.y * BM + row16) * N + n] =
          hip_bfloat16(float(acc[i]) * s * w_scale[n]);
    }
  }
}
```

上面 `load_vec128_or_zero` 是伪代码里的概念名，实际源码没有这个函数名，而是将
下面的逻辑直接内联在 A/B 搬运循环中。需要复用时可以使用这个可编译的 helper：

```cpp
struct alignas(16) GlobalVec128 { int8_t v[16]; };

__device__ __forceinline__ GlobalVec128 load_vec128_or_zero(
    const int8_t* base, int valid) {
  GlobalVec128 out{};
  if (valid >= 16) {
    // 只有调用方已经确认地址16-byte对齐且完整有效时才走这里。
    out = *reinterpret_cast<const GlobalVec128*>(base);
  } else {
    #pragma unroll
    for (int j = 0; j < 16; ++j)
      out.v[j] = (j < valid) ? base[j] : int8_t{0};
  }
  return out;
}
```

调用方必须先保证 `valid` 不为负，并避免对越界地址做指针计算；例如 A 的调用为：

```cpp
int valid = (gm < M && k0 + lk < K)
              ? min(16, K - (k0 + lk)) : 0;
GlobalVec128 x = (valid > 0)
    ? load_vec128_or_zero(a + int64_t(gm) * K + k0 + lk, valid)
    : GlobalVec128{};
for (int j = 0; j < 16; ++j) as[lm][lk + j] = x.v[j];
```

B 的调用同理，但 `valid = (gk < K && gn < N) ? min(16, N-gn) : 0`，并写入
`bs[ln+j][lk]` 完成 LDS 转置。当前生产 kernel 为了减少 helper 调用开销，使用的就是
这段逻辑的内联版本，见 `kernels/gemm_opt_MMAC.cpp` 第 149–195 行。

## 4. 公平比较口径

脚本：`benchmark/compare_mmac_splitk.py`

- 四方：Triton、iteration-008、MMAC/dispatcher、split-K。
- split-K 时间包含 partial GEMM 和 INT32 reduction/scale/BF16 两个 kernel。
- 输入生成、权重 H2D、输出及 workspace 分配不计时。
- 3 次 warmup、10 次 sample，GPU Event 中位数。
- split 扫描 `{1,2,4,8,16,32}`，要求 `K % (split*32) == 0`。
- small-M 范围为 `M={1,4,8,16}`。

## 4.1 split-K 的主要实现

split-K 不改变数值算法：每个 split 仍用 INT8 dot4 得到 INT32，先写 partial，再由
第二个 kernel 做整数归约、scale 和 BF16 转换。核心实现如下：

```cpp
template <int BM, int BN, int BK>
__global__ __launch_bounds__(512)
void small_m_splitk_dot4_kernel(
    const int8_t* a, const int8_t* w, int32_t* partial,
    int M, int N, int K, int split_k) {
  int split = blockIdx.z;
  int span = K / split_k;
  int k_begin = split * span;
  int row = blockIdx.y * BM + threadIdx.y;
  int col = blockIdx.x * BN + threadIdx.x;
  int32_t acc = 0;

  for (int k0 = k_begin; k0 < k_begin + span; k0 += BK) {
    // A/B 128-bit global load -> LDS；B 在 LDS 中转置。
    load_a_vec128_to_lds(...);
    load_b_vec128_to_transposed_lds(...);
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 4) {
      int32_t ap = *reinterpret_cast<int32_t*>(&a_tile[threadIdx.y][kk]);
      int32_t bp = *reinterpret_cast<int32_t*>(&b_tile[threadIdx.x][kk]);
      acc = __builtin_amdgcn_sdot4(ap, bp, acc, false);
    }
    __syncthreads();
  }
  if (row < M && col < N)
    partial[(int64_t(split) * M + row) * N + col] = acc;
}
```

归约 kernel 必须保持 INT32 精度到最后：

```cpp
__global__ void reduce_splitk_scale_kernel(
    const int32_t* partial, const float* as, const float* ws,
    hip_bfloat16* y, int M, int N, int split_k) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t elems = int64_t(M) * N;
  if (idx >= elems) return;
  int32_t sum = 0;
  #pragma unroll 1
  for (int s = 0; s < split_k; ++s)
    sum += partial[int64_t(s) * elems + idx];
  int row = idx / N, col = idx % N;
  y[idx] = hip_bfloat16(float(sum) * as[row] * ws[col]);
}
```

workspace 大小为 `split_k*M*N*sizeof(int32_t)`；split 越大，partial 和 reduction 代价
越大。两个 kernel 必须在同一 stream 连续提交，计时要覆盖二者。

## 4.2 launcher 分流与参数检查

当前 MMAC launcher 的关键分流逻辑：

```cpp
if (M <= 8 && K <= 2048) {
  // BM=8, BN=64, BK=32, 512 threads, SDOT4
  launch_small_m_dot4(...);
} else {
  // BM=16, BN=64, BK=64, 256 threads, 4-wave MMAC
  launch_mmac_16x64(...);
}
```

split-K launcher 还必须拒绝不满足布局/分块契约的输入：

```cpp
if (M <= 0 || M > 16 || N % 16 != 0 || K % 16 != 0 ||
    split_k <= 0 || K % (split_k * 32) != 0)
  return hipErrorInvalidValue;

dim3 block(64, 8);
dim3 grid((N + 63) / 64, (M + 7) / 8, split_k);
```

如果生产 shape 不满足这些条件，应回退到 MMAC/general kernel 或单独实现 tail
kernel；不能把输入 padding/越界读取隐藏在 benchmark 里。

## 5. TP=1 small-M、大 K 实测

最后一列为 `MMAC/dispatcher latency / best split-K latency`；大于 1 表示 split-K 快。

### 5.1 wqkv-a：K=4096，N=1536

| M | Triton ms | iter008 ms | MMAC ms | 最佳 split | split-K ms | split-K 优势 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.241436 | 0.225276 | 0.133278 | 16 | 0.033120 | 4.024x |
| 4 | 0.250236 | 0.245595 | 0.144478 | 16 | 0.037439 | 3.859x |
| 8 | 0.214557 | 0.223516 | 0.140158 | 16 | 0.037599 | 3.728x |
| 16 | 0.252796 | 0.326714 | 0.144318 | 8 | 0.044639 | 3.233x |

### 5.2 wo-b TP=1：K=8192，N=4096

| M | Triton ms | iter008 ms | MMAC ms | 最佳 split | split-K ms | split-K 优势 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.398393 | 0.398393 | 0.243676 | 32 | 0.100478 | 2.425x |
| 4 | 0.319515 | 0.394234 | 0.246396 | 32 | 0.107358 | 2.295x |
| 8 | 0.325275 | 0.446393 | 0.250556 | 32 | 0.113758 | 2.202x |
| 16 | 0.323514 | 0.544471 | 0.252956 | 16 | 0.164798 | 1.535x |

### 5.3 shared gate/up TP=1：K=4096，N=4096

| M | Triton ms | iter008 ms | MMAC ms | 最佳 split | split-K ms | split-K 优势 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.284156 | 0.225276 | 0.138557 | 16 | 0.061119 | 2.267x |
| 4 | 0.233276 | 0.197116 | 0.127998 | 32 | 0.063039 | 2.030x |
| 8 | 0.233276 | 0.226236 | 0.131197 | 16 | 0.066879 | 1.962x |
| 16 | 0.235996 | 0.281275 | 0.135518 | 16 | 0.094399 | 1.436x |

### 5.4 shared down TP=1：K=2048，N=4096

M=1/4/8 的 dispatcher 实际走 small-M dot4，M=16 才是真 MMAC。

| M | Triton ms | iter008 ms | dispatcher ms | 最佳 split | split-K ms | split-K 优势 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.211036 | 0.181917 | 0.165757 | 16 | 0.040319 | 4.111x |
| 4 | 0.195997 | 0.161437 | 0.157757 | 16 | 0.041439 | 3.807x |
| 8 | 0.194876 | 0.169438 | 0.165277 | 16 | 0.042559 | 3.883x |
| 16 | 0.194077 | 0.151517 | 0.077759 | 8 | 0.059039 | 1.317x |

## 6. Dispatch 建议

| M | K | N | 建议路径 | split |
|---:|---:|---:|---|---:|
| 1~8 | 2048 | 4096 | split-K SDOT4 | 16 |
| 16 | 2048 | 4096 | split-K SDOT4 | 8 |
| 1~8 | 4096 | 1536 | split-K SDOT4 | 16 |
| 16 | 4096 | 1536 | split-K SDOT4 | 8 |
| 1~8 | 4096 | 4096 | split-K SDOT4 | 16/32，需按 N 调优 |
| 16 | 4096 | 4096 | split-K SDOT4 | 16 |
| 1~8 | 8192 | 4096 | split-K SDOT4 | 32 |
| 16 | 8192 | 4096 | split-K SDOT4 | 16 |
| >=32 | 当前真权重 K/N | 四 Wave MMAC | - |

选择 split 时至少同时计算：

```text
base_blocks = ceil(M/8) * ceil(N/64)
split_blocks = base_blocks * split_k
workspace_bytes = split_k * M * N * 4
```

split 增大可以增加 K 方向并行度，但也增加 grid、INT32 workspace 和 reduction 成本，
不能只按 K 最大化 split。

## 7. 限制与下一步

- MMAC 尚无 global/LDS double buffer，加载与计算未流水重叠。
- B global-to-LDS 仍需转置；输出仍是分散的标量 BF16 store。
- split-K 需要 `split*M*N*4` workspace 和第二次 kernel launch。
- split-K 当前只支持 `M<=16`、`N%16==0`、`K%16==0`，并要求 K 可被
  `split*32` 整除。
- 新 shape、不同 DTK 或不同频率策略必须重新核对正确性并 tuning。

## 8. 复现实验

```bash
ROCR_VISIBLE_DEVICES=0 python3 benchmark/compare_mmac_splitk.py \
  --workload wqkv-a wo-b-tp1 shared-gate-up-proj-tp1 shared-down-proj-tp1 \
  --M 1 4 8 16 --splits 1 2 4 8 16 32 --warmup 3 --samples 10
```
