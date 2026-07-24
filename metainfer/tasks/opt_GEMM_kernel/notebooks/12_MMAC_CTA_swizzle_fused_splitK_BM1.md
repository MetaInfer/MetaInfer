# K100/gfx928 W8A8 GEMM：MMAC CTA Swizzle、Fused Split-K 与 BM=1 特化

## 1. 文档范围与证据基线

本文记录在 K500SM_AI / gfx928 / Wave64 上实测过的三项改动：

1. large-M MMAC 的 L2-aware CTA swizzle；
2. 使用 last-arriving CTA 的 fused split-K；
3. `wq-b-tp4-m1` 的精确 `BM=1` fused split-K 特化。

目标 ABI 和布局：

```text
Y_bf16 = bf16((A_int8[M,K] @ W_int8[K,N])
              * A_scale[M,None] * W_scale[None,N])

A:       row-major [M,K], INT8
W:       row-major [K,N], INT8
A_scale: [M], FP32
W_scale: [N], FP32
Y:       row-major [M,N], BF16
```

源码版本：

```text
原 Champion:
  benchmark/gemm_champon.cpp

001（fused split-K + MMAC CTA swizzle）:
  benchmark/gemm_champon_001.cpp

002（001 + 精确 BM=1 特化）:
  benchmark/gemm_champon_002.cpp
```

所有最终结论均要求：

```text
64-case correctness 全通过
目标 case 单独 rocprof，核对实际 kernel 名称
旧/新独立 .so
同卡交替 benchmark
10 warmup + 100 samples + GPU event + 每 sample 同步
```

性能数字是当前机器的实验事实，不是跨机器 SLA。

## 2. 最终推荐 dispatch 增量

本文只描述相对 notebook 11 的新增规则：

```text
1. M=1,N=8192,K=1024
      -> BM=1, BN=64, BK=32 fused split-K SDOT4

2. 其他满足 split-K 表的 small-M
      -> BM=8, BN=64, BK=32
         根据逐 shape 证据选择：
           a. 原两-kernel split-K
           b. BM=8 fused split-K

3. M<=4,N=4096,K=256，且 A/W 4-byte 对齐
      -> register-only Wave64 DPP + SDOT4

4. large-M
      -> BM=32,BN=64,BK=128 MMAC
         使用 L2-aware CTA swizzle
```

不得仅凭 `M==1` 将所有 workload 送入 BM=1。BM 同时决定计算线程和
global-to-LDS 搬运并行度，必须按 `(M,N,K,split_k)` 精确认证。

## 3. Large-M MMAC：L2-aware CTA swizzle

### 3.1 原问题

MMAC tile：

```text
BM=32, BN=64, BK=128
block=512 threads=8 Wave64
每个 wave 计算一个 16x16 子块
A/B 经 LDS 双缓冲
v_mmac_i32_16x16x32_i8
```

原二维 grid 的调度顺序不能保证短时间内运行的 CTA 共享同一个 B panel。
对 `(M,N,K)=(4096,8192,1024)`，rocprof 观察到：

```text
理论最小逻辑数据量: 约 76.05 MiB
实际 HBM 事务:      约 1091.83 MiB
流量放大:           约 14.36x
L2 hit:             约 83.18%
延迟:               约 3.828 ms
性能:               约 17.94 TOPS
```

高达约 299 GB/s 的物理带宽主要来自重复流量，不代表高效率。

### 3.2 Swizzle 目标

让连续 CTA 先遍历一组 M tile，再移动 N tile：

```text
(m0,n0), (m1,n0), ... (m15,n0),
(m0,n1), (m1,n1), ... (m15,n1), ...
```

同一 M group 内多个 CTA 共享 B panel，提高 B 在 L2 中的时间局部性。

### 3.3 必要源码

```cpp
constexpr int CTA_GROUP_M = 16;

const int num_m_tiles = (M + BM - 1) / BM;
const int num_n_tiles = (N + BN - 1) / BN;
const int linear_block = static_cast<int>(blockIdx.x);
const int blocks_per_m_group = CTA_GROUP_M * num_n_tiles;
const int m_group = linear_block / blocks_per_m_group;
const int first_m_tile = m_group * CTA_GROUP_M;
const int group_m_size =
    (num_m_tiles - first_m_tile < CTA_GROUP_M)
        ? (num_m_tiles - first_m_tile)
        : CTA_GROUP_M;
const int block_in_m_group =
    linear_block - m_group * blocks_per_m_group;
const int tile_m =
    first_m_tile + block_in_m_group % group_m_size;
const int tile_n = block_in_m_group / group_m_size;
```

launcher 必须改为一维 grid：

```cpp
const int num_m_tiles = (M + BM - 1) / BM;
const int num_n_tiles = (N + BN - 1) / BN;
dim3 grid(num_m_tiles * num_n_tiles);
```

kernel 内所有原 `blockIdx.x/y` 的 tile 计算必须统一替换为
`tile_n/tile_m`，包括：

```text
A global load 的 gm
W global load 的 gn
双缓冲预取
epilogue 的 global_row/col
```

只改加载而漏改 epilogue 会产生静默错位。

### 3.4 实测结果与正确解释

同一 `(4096,8192,1024)`：

| 指标 | 原 MMAC | Swizzled MMAC |
|---|---:|---:|
| 延迟 | 3.828 ms | 3.016 ms |
| 性能 | 17.94 TOPS | 22.80 TOPS |
| Fetch | 802876 KiB | 70039 KiB |
| Write | 315157 KiB | 175739 KiB |
| 总物理流量 | 1091.83 MiB | 240.02 MiB |
| L2 hit | 83.18% | 97.38% |
| 物理 HBM BW | 298.9 GB/s | 83.5 GB/s |

结论：

```text
流量下降约 78%
延迟下降约 21%
速度提升约 1.269x
```

物理 GB/s 下降不是退化。新的 MMAC 更少访问 HBM，因此更快。分析 GEMM
时必须同时报告：

```text
逻辑最小数据量
rocprof FetchSize + WriteSize
流量放大倍数
L2 hit
无 profiler benchmark 延迟
```

禁止把“更高的物理 GB/s”直接等同于“更好的 kernel”。

## 4. Fused Split-K：实现与代价

### 4.1 原两-kernel结构

```text
kernel 1:
  每个 split CTA 计算 INT32 partial[split,M,N]

kernel 2:
  对 split 维做 INT32 reduction
  FP32 scale
  BF16 output
```

kernel 边界天然提供全局同步。

### 4.2 Last-arriving CTA fusion

fused 版本增加每个输出 tile 的计数器：

```cpp
static uint32_t* g_tile_done = nullptr;
```

每个 split CTA 完成 partial 后：

```cpp
partial[(int64_t(split) * M + row) * N + col] = acc;
__syncthreads();

if (tid == 0) {
    __threadfence();
    const uint32_t ticket = atomicAdd(&tile_done[tile_id], 1u);
    is_last_split = (ticket == uint32_t(split_k - 1));
}
__syncthreads();
```

最后到达的 CTA 完成规约和 epilogue：

```cpp
if (is_last_split && row < M && col < N) {
    const int64_t idx = int64_t(row) * N + col;
    const int64_t elements = int64_t(M) * N;
    int32_t sum = 0;
#pragma unroll 1
    for (int s = 0; s < split_k; ++s)
        sum += partial[int64_t(s) * elements + idx];
    output[idx] = hip_bfloat16(
        float(sum) * a_scale[row] * w_scale[col]);
}
```

输出发布后计数器复位：

```cpp
__syncthreads();
if (tid == 0 && is_last_split) {
    __threadfence();
    atomicExch(&tile_done[tile_id], 0u);
}
```

### 4.3 “融合”不代表免费

该方案省去一个 kernel launch，但增加：

```text
每个 split CTA 的 threadfence
每个 split CTA 的 atomicAdd
多个 __syncthreads
最后 CTA 的 reduction 拖尾
最后 CTA 的 threadfence + atomicExch
更高 SGPR、LDS 和代码体积
```

`wq-b-tp4-m1` 的 BM=8 fused：

```text
原两-kernel: 约 69.12 us
BM=8 fused: 约 69.92 us
```

因此不得默认“少一个 launch 一定更快”。必须逐 shape A/B。

### 4.4 并发与生命周期限制

static partial workspace 和 `tile_done` 只在以下条件下成立：

```text
单 GPU
单 stream
串行调用
同一 host thread 或外部串行化
```

多 stream 会共享：

```text
g_workspace
g_tile_done
```

从而造成 partial 和 ticket 竞争。生产方案应由调用方传 workspace，或按
`(device,stream)` 管理独立状态。扩容时的 `hipFree/hipMalloc` 也可能同步，
不得计入 steady-state benchmark。

## 5. 精确 BM=1 Fused Split-K

### 5.1 为什么 BM=8 在 M=1 有浪费

BM=8：

```text
block=(64,8)=512 threads=8 Wave64
```

当 M=1 时只有 `ty==0` 的 64 个线程拥有有效输出行。其余 448 个线程虽然可
参与 tile 搬运，但不进行有效输出计算和最后规约。

BM=1：

```text
block=(64,1)=64 threads=1 Wave64
```

所有线程都拥有有效输出列，同时仍保留 fused split-K ticket/reduction。

### 5.2 必要 launcher

只对已经认证的精确 shape启用：

```cpp
constexpr int BN = 64;
constexpr int BK = 32;

if (M == 1 && N == 8192 && K == 1024) {
    constexpr int BM = 1;
    const int tile_count = (N + BN - 1) / BN;
    int err = ensure_tile_counters(
        static_cast<size_t>(tile_count), stream);
    if (err != 0) return err;

    dim3 block(BN, BM);              // one Wave64
    dim3 grid((N + BN - 1) / BN, 1, split_k);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            small_m_splitk_dot4_fused_kernel<BM, BN, BK>),
        grid, block, 0, stream,
        x_q, weight_kn, g_workspace, g_tile_done,
        x_scale, weight_scale, output, M, N, K, split_k);
} else {
    // certified BM=8 fused or certified two-kernel path
}
```

### 5.3 实测证据

目标：

```text
wq-b-tp4-m1
M=1,N=8192,K=1024
split_k=4
```

| 实现 | 延迟 |
|---|---:|
| 原两-kernel split-K | 69.12 us |
| BM=8 fused | 69.92 us |
| 精确 BM=1 fused | 65.44 us |

```text
BM=1 vs BM=8 fused: 1.0685x
BM=1 vs 原两-kernel: 1.0562x
```

rocprof：

| 指标 | BM=8 fused | BM=1 fused |
|---|---:|---:|
| workgroup threads | 512 | 64 |
| Wavefronts | 4096 | 512 |
| VGPR | 24 | 20 |
| SGPR | 64 | 64 |
| LDS | 11264 B | 10752 B |
| FetchSize | 8281 KiB | 8262 KiB |
| VALUUtilization | 90.43% | 86.24% |

BM=1 的 `LDSInsts/wave` 会增加，因为一个 wave 承担原先八个 wave 合作的搬运。
这不等于总 LDS 工作增加八倍。应同时结合 Wavefronts：

```text
BM=8: 44.1 LDSInsts/wave * 4096 waves
BM=1: 339  LDSInsts/wave *  512 waves
```

### 5.4 失败边界：不能推广到所有 M=1

将所有 M=1 split-K 都改成 BM=1 后，多数其他 workload 退化。原因：

```text
BM=1 消除空闲计算 wave
但 global-to-LDS 搬运线程也从 512 降到 64
小 N、不同 K、不同 split 数可能更依赖搬运并行度
```

因此 BM 是 `(M,N,K,split_k,内存布局)` 的联合参数，不是只由 M 决定。
新增 BM 特化时必须：

1. 精确 shape guard；
2. 64-case correctness；
3. 目标 kernel rocprof；
4. 旧/新交替 benchmark；
5. 检查所有同 M workload，防止错误泛化。

## 6. BM=2 / BM=4 模板实例与认证表

BM=1 的结果说明应继续测试与 M 匹配的 CTA：

```text
M=2 -> BM=2, block=128 threads=2 Wave64
M=4 -> BM=4, block=256 threads=4 Wave64
M=8 -> BM=8, block=512 threads=8 Wave64（已有路径）
```

探索时曾让所有合法 fused split-K 的 M=2/4 分别进入 BM=2/4，然后使用
`002 -> 003 -> 002 -> 003` 和反向顺序各两轮。未改动的 DPP case也出现
数个百分点波动，因此最终只固化正反顺序均有明显收益的 shape：

```cpp
const bool certified_small_bm_shape =
    (K == 1024 && (N == 8192 || N == 4096)) ||
    (K == 4096 && N == 1024);

if (M == 2 && certified_small_bm_shape) {
    launch_fused_splitk<2, 64, 32>(...);
} else if (M == 4 && certified_small_bm_shape) {
    launch_fused_splitk<4, 64, 32>(...);
} else {
    launch_fused_splitk<8, 64, 32>(...);
}
```

认证 shape：

| M | K | N | 主要 workload | 模板 |
|---:|---:|---:|---|---|
| 2/4 | 1024 | 8192 | wq-b TP4 | BM=2/BM=4 |
| 2/4 | 1024 | 4096 | wq-b/wo-b TP8 | BM=2/BM=4 |
| 2/4 | 4096 | 1024 | gate-up TP4 | BM=2/BM=4 |

`wq-b-tp4` 四轮中位数：

| Case | BM=8 | 匹配 BM | speedup |
|---|---:|---:|---:|
| M=2 | 66.319 us | 55.519 us | 1.1945x |
| M=4 | 67.679 us | 57.359 us | 1.1799x |

rocprof 确认：

| Case | 实际 kernel | workgroup | Wavefronts | VGPR | SGPR |
|---|---|---:|---:|---:|---:|
| M=2 | `fused_kernel<2,64,32>` | 128 | 1024 | 24 | 64 |
| M=4 | `fused_kernel<4,64,32>` | 256 | 2048 | 24 | 64 |

判断开销发生在 host launcher。十几个整数比较相对 HIP launch 和
30--70 us 的 small-M kernel 可忽略；真正需要控制的是模板 code size、
首次 module load、workspace 分配和额外 kernel dispatch。

## 7. 小 M 的瓶颈判断

在当前 workload 中，M=1/2/4/8 的延迟大多约 30--70 us，M 增加八倍时延迟
变化很小。这说明 small-M 主要受：

```text
kernel launch 与短 kernel 固定成本
权重首次读取和内存延迟
split-K threadfence / atomic ticket
CTA 数与可用并行度
last-arriving CTA 拖尾
BM 与实际 M 的匹配
```

当前通常不是首要瓶颈：

```text
LDS bank conflict
HBM 峰值 GB/s
SDOT4 lane 利用率
```

register-only DPP 的 `(M<=4,N=4096,K=256)` 仍是独立策略：

```text
单 dispatch
LDS=0
workspace=0
DPP Wave64 reduction
```

它不应推广到 `wq-b-tp4-m1=(1,8192,1024)`；该 shape 的无 split DPP
实测慢于 split-K，因为 W 访问跨 K 行分散且 K 串行化。

## 8. Profiling 方法与工具注意

本机有效工具：

```text
/opt/dtk/rocprofiler/bin/rocprof
```

派生指标：

```text
FetchSize
WriteSize
L2CacheHit
MemUnitBusy
Wavefronts
VALUUtilization
VALUBusy
LDSInsts
ALUStalledByLDS
LDSBankConflict
```

HBM 带宽：

```text
physical_HBM_BW =
    (FetchSize + WriteSize) * 1024 / unprofiled_benchmark_seconds
```

Fetch 和 Write 可能来自不同 replay，最终延迟应使用无 profiler 的冻结
benchmark。Profiler 时间只用于校验，不应作为正式延迟。

`rocprofv2` 在当前 DTK 中可以枚举 gfx928 counter，但实际运行 file plugin
不生成输出，连 smoke test 也如此。未修复 runtime/plugin 安装前，不得把
其零退出码当作采集成功。

## 9. Agent 实施清单

1. MMAC swizzle 是否在加载、预取、epilogue 全部使用同一 tile 映射？
2. 最后 M group 是否正确处理不足 `CTA_GROUP_M` 的情况？
3. 是否同时报告物理流量和逻辑流量，而不是只比较 GB/s？
4. fused partial 是否保持 INT32 到最终 reduction？
5. `__threadfence()` 是否位于发布 ticket 之前？
6. 是否只有最后 CTA 规约，并在输出发布后复位 counter？
7. 是否明确 static workspace/ticket 的单 stream 限制？
8. BM=1 是否使用精确 shape guard，而不是泛化所有 M=1？
9. 是否验证实际 kernel 名、workgroup size 和 Wavefronts？
10. 是否跑完整 64-case correctness？
11. 是否使用独立 `.so` 交替 benchmark？
12. 若 fused 退化，是否允许该 shape 回退原两-kernel？

## 10. 证据索引

```text
001:
  benchmark/gemm_champon_001.cpp
  benchmark/champion_001/candidate/libmetainfer_gemm_candidate.so
  benchmark/champion_001/reports/correctness.json
  benchmark/champion_001/reports/ab-old1.json
  benchmark/champion_001/reports/ab-new1.json
  benchmark/champion_001/rocprof/

002:
  benchmark/gemm_champon_002.cpp
  benchmark/champion_002/candidate/libmetainfer_gemm_candidate.so
  benchmark/champion_002/reports/correctness-narrow.json
  benchmark/champion_002/reports/narrow1.json
  benchmark/champion_002/reports/narrow2.json
  benchmark/champion_002/rocprof/wq-b-tp4-m1/

003:
  benchmark/gemm_champon_003.cpp
  benchmark/champion_003/candidate/libmetainfer_gemm_candidate.so
  benchmark/champion_003/reports/final-correctness.json
  benchmark/champion_003/reports/final-benchmark.json
  benchmark/champion_003/rocprof/wq-b-tp4-m2/
  benchmark/champion_003/rocprof/wq-b-tp4-m4/
```
