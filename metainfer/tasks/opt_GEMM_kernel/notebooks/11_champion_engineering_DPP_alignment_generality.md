# K100/gfx928 Champion 工程化：DPP、Split-K、MMAC、安全对齐与通用性

## 1. 文档目的与源码基线

本文记录对以下算子的实际修改、失败尝试、修复、性能证据和工程边界，供后续 planning、implementer 和 reviewer agent 直接参考：

```text
/data/work/int8-w8a8-gemm/benchmark/gemm_champon.cpp
```

目标计算：

```text
Y_bf16 = bf16((A_int8[M,K] @ W_int8[K,N])
              * A_scale[M,None] * W_scale[None,N])
```

布局与 ABI：

```text
A:       contiguous row-major [M,K], INT8
W:       contiguous row-major [K,N], INT8
A_scale: contiguous [M], FP32
W_scale: contiguous [N], FP32
Y:       contiguous row-major [M,N], BF16
```

本版本面向海光 K100、`gfx928`、Wave64。DPP 和 `v_mmac_i32_16x16x32_i8` 都是架构专用实现，不能未经检查复制到 Wave32 或其他 ISA。

本文中的性能数字来自物理 HCU3，使用冻结 evaluator：10 次 warmup、100 次 sample、GPU Event、逐 sample 同步、取中位数。PMC 使用 `hipprof --pmc --pmc-type 3`，逐 case 单独进程采集。

## 2. 最终 dispatch 路由

当前 launcher 按下面顺序分流：

```text
1. M<=4, N=4096, K=256，且 A/W 基地址均 4-byte 对齐
      -> register-only Wave64 DPP + SDOT4

2. small-M 且 choose_split_k() 返回 split_k>1
      -> split-K SDOT4 partial kernel
      -> INT32 reduction + scale + BF16 kernel

3. M<=8 且没有合法 split-K
      -> 标量 GEMV fallback

4. 其他 shape
      -> general MMAC kernel
```

关键 launcher 代码：

```cpp
if (M <= 4 && N == 4096 && K == 256 &&
    (reinterpret_cast<uintptr_t>(x_q) & 3u) == 0 &&
    (reinterpret_cast<uintptr_t>(weight_kn) & 3u) == 0) {
    constexpr int threads = 256;
    constexpr int columns_per_block = 16;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(wave_dpp_sdot4_k256_n4096_kernel),
        dim3(N / columns_per_block), dim3(threads), 0, stream,
        x_q, weight_kn, x_scale, weight_scale, output, M, N, K);
    return hipGetLastError();
}
```

DPP 的 4-byte 门槛非常重要：该 kernel 使用 `int32_t*` 加载。非对齐输入不能进入 DPP，而是继续落入已经支持安全标量搬运的 split-K 路径。

## 3. Split-K SDOT4 Champion

### 3.1 Tile 与计算流程

```text
BM=8, BN=64, BK=32
block=(64,8)=512 work-items=8 Wave64
LDS A=[BM,BK]
LDS B=[BN,BK+4]
```

全局 W 是 `[K,N]`。global load 沿 N 读取，写 LDS 时转置成 `[N,K_PAD]`，之后每个线程得到 K 连续的 4-byte pack：

```cpp
int a_pack = *reinterpret_cast<const int32_t*>(&a_tile[ty][kk]);
int b_pack = *reinterpret_cast<const int32_t*>(&b_tile[tx][kk]);
acc = __builtin_amdgcn_sdot4(a_pack, b_pack, acc, false);
```

每个 K split 先写 INT32 partial：

```text
partial[split,M,N]
```

第二个 kernel 必须先做 INT32 reduction，最后才做 scale 和 BF16：

```cpp
int32_t sum = 0;
for (int s = 0; s < split_k; ++s)
    sum += partial[int64_t(s) * M * N + idx];
output[idx] = hip_bfloat16(float(sum) * a_scale[row] * w_scale[col]);
```

禁止提前转 BF16，也不能对 BF16 做原子累加。

### 3.2 当前 split 表与整除保证

| Shape 条件 | M<=8 | M=9..16 |
|---|---:|---:|
| K=4096,N=1536 | 16 | 8 |
| K=4096,N=1024 | 32 | 16 |
| K=4096,N=512 | 32 | 16 |
| K=2048,N=4096 | 16 | 8 |
| K=1024,任意N且M<=8 | 4 | 1 |
| K=512,任意N且M<=8 | 4 | 1 |
| K=256,任意N且M<=8 | 2 | 1 |

这些表项全部满足：

```text
K % (split_k * BK) == 0, BK=32
```

当前不会把任意不能整除的 K 强行送进 split-K。未知 K 返回 `split_k=1`：M<=8 走 GEMV，M>8 走 MMAC。

若 agent 扩展 split 表，必须同时检查：

```cpp
split_k > 1
K / split_k >= BK
K % (split_k * BK) == 0
```

否则 `span=K/split_k` 会截断，或者 K-loop 最后一 tile 破坏 split 边界。

## 4. General MMAC kernel

最终 general 路径：

```text
BM=32, BN=64, BK=128
block=512 threads=8 Wave64
每个 wave 计算一个 16x16 子块
A/B LDS 双缓冲
128-bit global request -> LDS
v_mmac_i32_16x16x32_i8
FP32 scale -> BF16
```

Wave 映射：

```text
row_group = wave >> 2
col_group = wave & 3
waves 0..3 -> tile rows 0..15, four 16-column slices
waves 4..7 -> tile rows 16..31, four 16-column slices
```

MMAC fragment：

```cpp
const int row16 = lane & 15;
const int kg = lane >> 4;
const int fragment_k = kk + kg * 8;
int2_t af = load_lds_int8x8(&a_tile[buf][row][fragment_k]);
int2_t bf = load_lds_int8x8(
    &b_tile[buf][col_group * 16 + row16][fragment_k]);
acc = mmac_i32_16x16x32_i8(af, bf, acc);
```

M/N/K 尾块通过 global load 补零和 epilogue 边界判断处理。因此 shape 不必是 32/64/128 的整数倍。

## 5. `shared-down-proj-tp8` 的 DPP 专用路径

### 5.1 为什么要增加新路径

目标 shape：

```text
A[M,256] @ W[256,4096], M in {1,2,4,8}
```

原 Champion 使用 `split_k=2`：主 kernel + reduction 两个 dispatch，并产生 `2*M*4096*4` bytes INT32 workspace。原 `gemm_opt` 是单 kernel、小 M SDOT4，因此在这几个 case 上更快。

原始 benchmark：

| M | Champion split-K | gemm_opt |
|---:|---:|---:|
| 1 | 37.760 us | 35.841 us |
| 2 | 35.840 us | 32.161 us |
| 4 | 36.000 us | 32.160 us |
| 8 | 36.160 us | 33.440 us |

### 5.2 K100 使用 DPP，不是 CUDA shuffle

CUDA 常用 warp shuffle；K100/gfx928 应明确使用 AMD DPP。最终规约使用：

```cpp
__device__ __forceinline__ int32_t wave64_reduce_sum_dpp(int32_t value) {
    value += __builtin_amdgcn_mov_dpp(value, 0x111, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x112, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x114, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x118, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x142, 0xf, 0xf, false);
    value += __builtin_amdgcn_mov_dpp(value, 0x143, 0xf, 0xf, false);
    return value; // Wave64 total is in lane 63
}
```

含义：

```text
row_shr:1/2/4/8 -> 每个 16-lane row 内规约
row_bcast:15/31 -> 合并 Wave64 的四个 row
lane 63 -> 最终结果
```

### 5.3 向量加载与寄存器转置

一个 Wave64 计算同一行连续 4 列。每个 lane 负责连续 4 个 K：

```text
lane 0 -> K 0..3
lane 1 -> K 4..7
...
lane 63 -> K 252..255
```

A 沿 K 做一个4-byte加载。W 是 `[K,N]`，所以对4个K行分别沿N做4-byte加载，得到寄存器中的4x4 byte块，再转置成4个 K-contiguous SDOT4 operand：

```cpp
const uint32_t w0 = *reinterpret_cast<const uint32_t*>(W[(k0+0),col0]);
const uint32_t w1 = *reinterpret_cast<const uint32_t*>(W[(k0+1),col0]);
const uint32_t w2 = *reinterpret_cast<const uint32_t*>(W[(k0+2),col0]);
const uint32_t w3 = *reinterpret_cast<const uint32_t*>(W[(k0+3),col0]);

const int32_t wp0 = pack_weight_column(w0,w1,w2,w3,0);
const int32_t wp1 = pack_weight_column(w0,w1,w2,w3,1);
const int32_t wp2 = pack_weight_column(w0,w1,w2,w3,2);
const int32_t wp3 = pack_weight_column(w0,w1,w2,w3,3);
```

之后每个 lane 做4个 SDOT4，DPP 规约，lane63写4列。该路径：

```text
LDS=0
workspace=0
single dispatch
VGPR=24
SGPR=32
```

### 5.4 第一版失败：按 M 重复加载 W

第一版 grid 使用 `grid.y=M`，每行独立 block，导致同一 W 被重新读取 M 次：

| M | DPP v1 | gemm_opt | 结论 |
|---:|---:|---:|---|
| 1 | 30.079 us | 37.919 us | 快 |
| 2 | 33.759 us | 35.519 us | 略快 |
| 4 | 79.037 us | 35.519 us | 严重退化 |
| 8 | 118.237 us | 36.639 us | 严重退化 |

这是一个重要教训：去掉 LDS 不等于减少 global traffic。对于多个 M 行，必须复用 W。

### 5.5 第二版：W 寄存器跨 M 复用

最终 kernel 只按 N 发 grid。每个 wave 将 W 的4个 SDOT4 operand 保存在寄存器，然后在 kernel 内循环 M：

```cpp
const int32_t wp0 = ...; // load/transpose W once
const int32_t wp1 = ...;
const int32_t wp2 = ...;
const int32_t wp3 = ...;

#pragma unroll
for (int row = 0; row < M; ++row) {
    int32_t ap = *reinterpret_cast<const int32_t*>(A + int64_t(row)*K + k0);
    int32_t acc0 = __builtin_amdgcn_sdot4(ap, wp0, 0, false);
    int32_t acc1 = __builtin_amdgcn_sdot4(ap, wp1, 0, false);
    int32_t acc2 = __builtin_amdgcn_sdot4(ap, wp2, 0, false);
    int32_t acc3 = __builtin_amdgcn_sdot4(ap, wp3, 0, false);
    acc0 = wave64_reduce_sum_dpp(acc0);
    acc1 = wave64_reduce_sum_dpp(acc1);
    acc2 = wave64_reduce_sum_dpp(acc2);
    acc3 = wave64_reduce_sum_dpp(acc3);
    if (lane == 63) { /* scale + BF16 store */ }
}
```

同口径重跑：

| M | DPP v2 | gemm_opt | DPP v2 speedup |
|---:|---:|---:|---:|
| 1 | 31.359 us | 37.919 us | 1.209x |
| 2 | 31.040 us | 35.519 us | 1.144x |
| 4 | 31.840 us | 35.519 us | 1.116x |
| 8 | 37.760 us | 36.639 us | 0.970x |

最终只对 `M<=4` 启用 DPP；M=8 保留原路径。

PMC 证据：DPP v2 的 `LDS bytes=0`、`SQ_INSTS_LDS=0`、`SQ_LDS_BANK_CONFLICT=0`、`SQ_WAIT_INST_LDS=0`；`gemm_opt` 使用2592 bytes LDS，存在22K~51K LDS instructions和4.8K~79.8K LDS wait。

## 6. 128-bit global load 的对齐安全

### 6.1 不能只检查 tile 尾部

MMAC 和 split-K 使用：

```cpp
struct alignas(16) GlobalVec128 { int32_t dwords[4]; };
*reinterpret_cast<const GlobalVec128*>(src)
```

即使 HIP allocation 的基地址对齐，每行地址仍取决于 stride：

```text
A address = A_base + gm*K + gk
W address = W_base + gk*N + gn
```

A 全行自然16-byte对齐通常要求 `A_base%16==0 && K%16==0`；W 通常要求 `W_base%16==0 && N%16==0`。带 storage offset 的切片还可能让基地址本身不对齐。

K100 的 flat/global load 可能容忍某些非对齐地址，但 C++ 的对齐类型转换仍不应依赖这种行为，而且可能拆分内存事务。

### 6.2 最终策略：MMAC不回退，只让搬运标量化

保留 MMAC/SDOT4 核心。每一个16-byte片检查“完整有效且实际地址对齐”：

```cpp
GlobalVec128 value{};
if (gm < M && gk + 16 <= K &&
    (reinterpret_cast<uintptr_t>(
        a + int64_t(gm) * K + gk) & 15u) == 0) {
    value = *reinterpret_cast<const GlobalVec128*>(
        a + int64_t(gm) * K + gk);
} else if (gm < M && gk < K) {
    const int remaining = K - gk;
    const int valid = remaining < 16 ? remaining : 16;
    #pragma unroll
    for (int j = 0; j < valid; ++j)
        reinterpret_cast<int8_t*>(&value)[j] =
            a[int64_t(gm) * K + gk + j];
}
```

W 同理：

```cpp
if (gk < K && gn + 16 <= N &&
    (reinterpret_cast<uintptr_t>(
        w + int64_t(gk) * N + gn) & 15u) == 0) {
    value = *reinterpret_cast<const GlobalVec128*>(
        w + int64_t(gk) * N + gn);
} else if (gk < K && gn < N) {
    const int remaining = N - gn;
    const int valid = remaining < 16 ? remaining : 16;
    for (int j = 0; j < valid; ++j)
        reinterpret_cast<int8_t*>(&value)[j] =
            w[int64_t(gk) * N + gn + j];
}
```

这样只有 global-to-LDS 搬运降级，后面的 LDS tile、SDOT4/MMAC 和 epilogue 不变。

### 6.3 修复中发现的关键 bug：标量长度必须钳制到16

原尾部代码使用：

```cpp
const int valid = K - gk;
```

它原本隐含假设“进入else一定是尾部不足16字节”。加入非对齐分支后，一个完整但非对齐的片也进入else，`K-gk`可能是256或更大，导致写爆16-byte临时对象。

必须改为：

```cpp
const int valid = min(16, K - gk);
```

W 使用 `min(16,N-gn)`。这是后续 agent 最容易遗漏的地方。

### 6.4 LDS 显式对齐

所有被转换为 `int32_t*`/`int2_t*` 的 LDS 数组显式声明：

```cpp
__shared__ __align__(16) int8_t a_tile[...];
__shared__ __align__(16) int8_t b_tile[...];
```

虽然当前 stride `32/36/68/132` 均为4的倍数，但不能只依赖编译器碰巧对齐 LDS symbol。`gemm_opt.cpp` 的LDS数组也做了相同修正。

## 7. 通用 shape 与正确性证据

### 7.1 非标准 shape

下列非标准、非 tile 整除 shape 与精确参考逐元素一致：

```text
(1,13,17)
(7,67,259)
(8,70,256)
(5,73,512)
(3,65,1024)
(9,13,17)
(17,65,129)
(33,70,257)
(31,63,2049)
(16,77,2051)
```

覆盖 GEMV、split-K 和 MMAC；全部返回0、零 mismatch、零误差。

### 7.2 真实非对齐指针

不能用“奇数 K/N”代替指针对齐测试。最终另外构造A/W allocation，并传入基地址 `+1 byte` 的连续逻辑矩阵：

```text
A offset=1, W offset=0
A offset=0, W offset=1
A offset=1, W offset=1
```

覆盖：

```text
DPP guard -> split-K fallback: (1,4096,256)
split-K odd N:             (5,73,512), (3,65,1024)
MMAC odd strides/tails:    (17,65,129), (33,70,257)
scalar GEMV:               (7,67,259)
```

18组全部零 mismatch、零误差。报告：

```text
benchmark/alignment_safe/reports/misaligned-pointer-correctness.json
```

最终完整冻结正确性也是64/64、零 mismatch：

```text
benchmark/alignment_safe/reports/champion-final-correctness.json
```

## 8. 安全对齐修改的性能影响

使用同一物理 HCU3 按下列顺序交替跑完整60 case：

```text
old -> new -> old -> new
```

每个case取两次延迟的中位数。新版本相对旧版本：

| 分组 | 等权几何平均 | case数 |
|---|---:|---:|
| 全部 | 1.0146x | 60 |
| DPP | 1.0416x | 3 |
| split-K | 1.0141x | 42 |
| MMAC | 1.0108x | 15 |

整体没有性能下降。39个case新版本更快，19个更慢，短小 split-K kernel 存在数微秒波动。若以后确认逐片动态地址判断是稳定瓶颈，可模板化为：

```text
kernel<Aligned=true>  -> host已确认base和stride对齐，内核无动态判断
kernel<Aligned=false> -> 所有16-byte片走安全标量搬运
```

但在没有旧/新交替证据前，不要仅凭源码指令数宣称性能变化。

完整对比：

```text
benchmark/alignment_safe/reports/ab-summary.json
```

### 8.1 Benchmark 与 PMC 不可混用

`hipprof --pmc` 会插桩并改变 dispatch 时间，PMC CSV 中的 `DispatchNs` 只能用于同次 profile 的辅助观察，不能代替正式 benchmark 中的 GPU Event 中位数。报告性能时：

```text
latency/speedup/TOPS -> benchmark JSON
VGPR/SGPR/LDS/counter -> hipprof PMC CSV/parsed JSON
```

Triton 必须只选择 `matmul_kernel`；DPP 选择 `wave_dpp_sdot4...`；`gemm_opt` 选择目标 `w8a8_scaled_gemm_kernel...`。split-K 必须同时保留主 kernel 和 reduction 两个 dispatch，不能只报其中一个。

本实验所有跨 case 总结均为60 case等权或逐 case原始结果，不使用任务评分权重。等权几何平均 speedup 适合比较实现，不能把不同 shape 的 latency 直接解释成同一工作量。

### 8.2 已知编译警告

当前 DTK/hipcc 仍报告若干已有警告：

```text
void kernel is missing a return statement
DPP runtime-M loop requested unroll but was not fully unrolled
```

`gemm_opt` 运行时还曾提示512-thread launch与缺失 `__launch_bounds__` 元数据不一致。它们没有造成当前正确性失败，但正式发布前应清理，不能把“编译成功”当成零风险。不要为消除警告盲目改变 thread/block 形状；修复后必须重新做正确性、benchmark和PMC。

## 9. 吞吐量解释

INT8乘加应报告 TOPS，不是 TFLOPS：

```text
TOPS = 2*M*N*K / latency_seconds / 1e12
```

不同case差异很大，因为小M主要受launch、访存和scale开销限制，大M才能充分使用MMAC。

当前实测范围：

| M | median TOPS | max TOPS |
|---:|---:|---:|
| 1 | 0.173 | 0.274 |
| 2 | 0.358 | 0.558 |
| 4 | 0.693 | 1.095 |
| 8 | 1.400 | 2.246 |
| 16 | 2.754 | 4.132 |
| 4096 | 22.111 | 24.143 |

最大实测约24.143 TOPS：`wqkv-a-tp4-m4096`，shape `(4096,1536,4096)`，MMAC，约2.135 ms。

## 10. 工程通用性：可以做什么，仍不能承诺什么

### 10.1 已经具备

- M/N/K 不必等于1024/2048/4096；未知shape有 fallback。
- M/N/K tile tail 有边界保护和补零。
- 当前 split 表保证 K 可整除。
- 16-byte global地址不对齐时，MMAC/split-K只将搬运标量化，核心计算不降级。
- DPP 4-byte不对齐时自动避开DPP。
- LDS显式16-byte对齐。
- 标准64 case、10个非标准shape、18组真实非对齐指针均有正确性证据。

### 10.2 仍然不是完整生产通用库

当前 split-K workspace：

```cpp
static int32_t* g_workspace = nullptr;
static size_t g_workspace_capacity = 0;
```

存在以下问题：

1. 多 stream 并发会写同一 partial workspace；
2. 多 host thread 扩容存在竞争；
3. 同一进程多 GPU 时 pointer/device ownership 不安全；
4. 没有显式销毁接口；
5. 扩容的 `hipFree/hipMalloc` 可能同步；
6. workspace 分配不在计时内，生产首调用延迟不同。

因此当前可承诺范围是：

```text
K100/gfx928
单GPU
单stream串行调用
contiguous row-major A/W/Y
A_scale[M]、W_scale[N]
当前INT32累加不溢出的K范围
```

尚不支持或未验证：

```text
多stream并发
同进程多GPU
任意stride/transpose view
batched GEMM
不同scale布局
跨架构运行
极大K导致INT32溢出
```

生产化优先事项：让调用方传入workspace，或按 `(device,stream)` 管理独立workspace，并添加线程安全与生命周期管理。

## 11. Agent 实施与评审清单

修改本算子时必须逐项回答：

1. 新shape命中哪个dispatch？未知shape是否安全fallback？
2. 新split是否满足 `K%(split*BK)==0`？
3. global vector地址是否实际对齐，而不只是基地址对齐？
4. 非对齐标量搬运是否严格 `min(vector_width,remaining)`？
5. LDS基地址和每行stride是否满足后续dword/int2加载？
6. K/N尾部是否补零，M/N输出是否做边界判断？
7. split partial是否保持INT32直到最终reduction？
8. benchmark是否包含所有dispatch但不包含workspace首次分配？
9. PMC是否过滤到了目标kernel；split-K是否同时分析主kernel和reduction？
10. 是否跑64-case、非标准shape和真实misaligned pointer测试？
11. 是否用旧/新交替benchmark，而不是跨时段单次结果？
12. 是否误把INT8吞吐写成TFLOPS；应使用TOPS？
13. 是否说明了static workspace的并发限制？

## 12. 证据文件索引

```text
源码:
  /data/work/int8-w8a8-gemm/benchmark/gemm_champon.cpp
  /data/work/int8-w8a8-gemm/benchmark/gemm_opt.cpp

安全加载最终产物与报告:
  benchmark/alignment_safe/champion/libmetainfer_gemm_candidate.so
  benchmark/alignment_safe/reports/champion-final-correctness.json
  benchmark/alignment_safe/reports/champion-final-benchmark.json
  benchmark/alignment_safe/reports/misaligned-pointer-correctness.json
  benchmark/alignment_safe/reports/performance-vs-before.json
  benchmark/alignment_safe/reports/ab-summary.json

DPP实验:
  benchmark/three_way_profile/reports_dpp/champion-dpp-v2-correctness.json
  benchmark/three_way_profile/reports_dpp/champion-dpp-v2-benchmark.json
  benchmark/three_way_profile/reports_dpp/pmc-v2-vs-opt-four-cases.json
  benchmark/three_way_profile/pmc_dpp_v2/

三实现完整对比:
  benchmark/three_way_profile/reports/benchmark-all-cases.csv
  benchmark/three_way_profile/reports/pmc-all-cases.csv
  benchmark/three_way_profile/reports/summary-unweighted.json
```

性能报告是机器与当次运行的实验证据，不是跨机器生产SLA。后续agent应复用方法和约束，不应把具体微秒数硬编码成普遍结论。
