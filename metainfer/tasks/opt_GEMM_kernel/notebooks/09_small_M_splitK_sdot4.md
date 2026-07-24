# K100/gfx928 小 M、大 K：128-bit Load + Split-K + SDOT4

## 1. 结论

目标算子：

```text
Y_bf16 = bf16((A_int8[M,K] @ W_int8[K,N])
              * A_scale[M,None] * W_scale[None,N])
```

一次 TP=1 真权重实验观测到：当 `M <= 16` 且 K 较大时，以下专用路径明显优于
iteration 008，并在当前 evaluator 计时口径下超过 Triton baseline。这里的数字是
待复验的 tuning 证据，不是生产性能承诺：

1. 源码用16-byte对齐类型请求 A、B 的128-bit global-to-LDS load；最终ISA宽度尚需
   反汇编确认；
2. B 保持全局 `[K,N]` row-major，在写入 LDS 时转成 K 连续布局；
3. 使用 `__builtin_amdgcn_sdot4`；
4. 多个 workgroup 沿 K 并行，每个 split 写 INT32 partial；
5. 第二个 kernel 做 INT32 reduction、scale 和 BF16 写回；
6. 性能计时包含 GEMM 和 reduction 两个 kernel。

本路线适合 small-M，不应替代 large-M 的 MMAC/MFMA 路径。


## 2. 数据布局与128-bit load

输入契约不变：

```text
A: [M,K], row-major, stride=[K,1], INT8
W: [K,N], row-major, stride=[N,1], INT8
Y: [M,N], row-major, stride=[N,1], BF16
```

实验源码使用：

```cpp
struct alignas(16) Vec128 { int32_t x[4]; };
```

A 沿 K 请求读取 `A[m,k:k+16]`；B 沿 N 请求读取 `W[k,n:n+16]`。编译器可能
保留、拆分或重排这些访问，必须以反汇编为准。源码将 B 的16个连续N元素写入
转置 LDS：

```text
Global W[BK,BN] row-major
       ↓ 128-bit coalesced load along N
LDS b_tile[BN][BK_PAD]
       ↓ four adjacent K bytes
sdot4
```

当前约束：

```text
M <= 16
N % 16 == 0
K % 16 == 0
K % (split_k * 32) == 0
```

## 3. Split-K 与正确 reduction

K 等分为连续区间：

```text
split_span = K / split_k
k_begin = split_id * split_span
k_end = k_begin + split_span
```

grid：

```text
grid.x = ceil(N / BN)
grid.y = ceil(M / BM)
grid.z = split_k
```

每个 split 写：

```text
partial[split_k,M,N], dtype=int32
```

正确顺序：

```text
split GEMM → INT32 partial
            ↓ reduce across split
INT32 total → float * A_scale * W_scale → BF16
```

不得先转 BF16 或直接对 BF16 原子累加。workspace 大小：

```text
split_k * M * N * 4 bytes
```

例如 `split=32,M=16,N=4096` 使用8 MiB。

当前 tile：

```text
BM=8, BN=64, BK=32, BK_PAD=36
block=(64,8)=512 work-items=8 Wave64
```

### 4.1 核心代码摘录

以下摘自 `benchmark/small_m_splitk.hip`。实际使用以源码和反汇编为准。

#### A的16-byte请求加载

```cpp
constexpr int A_VECS = BM * BK / 16;
for (int vi = tid; vi < A_VECS; vi += threads) {
    const int lm = vi / (BK / 16);
    const int lk = (vi % (BK / 16)) * 16;
    const int gm = blockIdx.y * BM + lm;
    Vec128 value{};
    if (gm < M) {
        value = *reinterpret_cast<const Vec128*>(
            a + int64_t(gm) * K + k0 + lk);
    }
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        a_tile[lm][lk + j] = reinterpret_cast<const int8_t*>(&value)[j];
}
```

#### B的16-byte请求加载与LDS转置

```cpp
constexpr int B_VECS = BK * BN / 16;
for (int vi = tid; vi < B_VECS; vi += threads) {
    const int lk = vi / (BN / 16);
    const int ln = (vi % (BN / 16)) * 16;
    const int gn = blockIdx.x * BN + ln;
    Vec128 value{};
    if (gn + 16 <= N) {
        value = *reinterpret_cast<const Vec128*>(
            w + int64_t(k0 + lk) * N + gn);
    } else {
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            reinterpret_cast<int8_t*>(&value)[j] =
                gn + j < N ? w[int64_t(k0 + lk) * N + gn + j] : 0;
    }
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        b_tile[ln + j][lk] = reinterpret_cast<const int8_t*>(&value)[j];
}
```

#### SDOT4与INT32 partial

```cpp
int32_t acc = 0;
#pragma unroll
for (int kk = 0; kk < BK; kk += 4) {
    const int a_pack = *reinterpret_cast<const int32_t*>(&a_tile[ty][kk]);
    const int b_pack = *reinterpret_cast<const int32_t*>(&b_tile[tx][kk]);
    acc = __builtin_amdgcn_sdot4(a_pack, b_pack, acc, false);
}
partial[(int64_t(split) * M + row) * N + col] = acc;
```

#### INT32 reduction、scale与BF16写回

```cpp
const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
const int64_t elements = int64_t(M) * N;
if (idx >= elements) return;

int32_t acc = 0;
for (int split = 0; split < split_k; ++split)
    acc += partial[int64_t(split) * elements + idx];

const int row = idx / N;
const int col = idx - int64_t(row) * N;
output[idx] = hip_bfloat16(float(acc) * a_scale[row] * w_scale[col]);
```

主kernel与reduction连续提交到同一stream，中间无需host同步；计时必须覆盖两次
launch。

## 5. 正确性

该次运行中所有列出的 TP=1 配置：

```text
max_abs_error = 0
mismatches = 0
```

完整 BF16 输出逐元素与 Triton 比较，门槛为 `abs(diff) <= 1e-3`。这验证了
split边界、INT32 partial/reduction、scale和BF16写回，但新增非对齐 shape 时仍须
重新做完整正确性。

## 6. TP=1 真权重实测

以下数字逐项抄录自该次运行输出，没有做跨轮平均。GPU Event取10次sample的中位
数；split-K延迟包含reduction；量化、权重加载和workspace分配不计时。3次warmup、
10次sample仅用于方向判断，未锁定频率，也未报告方差。正式dispatch需增加样本并
重复多轮。

### 6.1 `wqkv-a`: M×4096 @ 4096×1536

| M | Triton ms | iter008 ms | 最佳 split | split-K ms | vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.243674 | 0.227194 | 16 | 0.034239 | 7.117× |
| 4 | 0.252153 | 0.243674 | 16 | 0.035040 | 7.196× |
| 8 | 0.210395 | 0.220474 | 16 | 0.035199 | 5.977× |
| 16 | 0.260793 | 0.332951 | 8 | 0.044959 | 5.801× |

### 6.2 `wq-b-tp1`: M×1024 @ 1024×32768

| M | Triton ms | iter008 ms | 最佳 split | split-K ms | vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.282553 | 0.120637 | 2 | 0.095678 | 2.953× |
| 4 | 0.287673 | 0.122877 | 4 | 0.099678 | 2.886× |
| 8 | 0.290233 | 0.123997 | 2 | 0.108477 | 2.676× |
| 16 | 0.293593 | 0.201275 | 2 | 0.195675 | 1.500× |

N=32768 本身产生大量 N-direction blocks，大 split 只会增加 reduction 和调度成本。

### 6.3 `wo-b-tp1`: M×8192 @ 8192×4096

| M | Triton ms | iter008 ms | 最佳 split | split-K ms | vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.326552 | 0.385750 | 32 | 0.127997 | 2.551× |
| 4 | 0.324312 | 0.394870 | 32 | 0.129917 | 2.496× |
| 8 | 0.327832 | 0.438389 | 32 | 0.136476 | 2.402× |
| 16 | 0.328631 | 0.543667 | 32 | 0.168636 | 1.949× |

### 6.4 `shared-gate-up-proj-tp1`: M×4096 @ 4096×4096

| M | Triton ms | iter008 ms | 最佳 split | split-K ms | vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.238234 | 0.199995 | 32 | 0.066558 | 3.579× |
| 4 | 0.240154 | 0.275353 | 32 | 0.070559 | 3.404× |
| 8 | 0.245434 | 0.238874 | 32 | 0.073439 | 3.342× |
| 16 | 0.240954 | 0.289113 | 16 | 0.095838 | 2.514× |

### 6.5 `shared-down-proj-tp1`: M×2048 @ 2048×4096

| M | Triton ms | iter008 ms | 最佳 split | split-K ms | vs Triton |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.192955 | 0.155196 | 16 | 0.039199 | 4.922× |
| 4 | 0.195675 | 0.163356 | 16 | 0.040799 | 4.796× |
| 8 | 0.196155 | 0.173436 | 16 | 0.043679 | 4.491× |
| 16 | 0.195995 | 0.154876 | 8 | 0.059038 | 3.320× |

## 7. 初始 dispatch 建议

这只是当前 K100、当前 tile、TP=1实测 shape 的 tuning 起点：

| Shape class | 建议 split-K |
|---|---:|
| K=1024,N=32768,M<=16 | 2；M=4比较2/4 |
| K=2048,N=4096,M<=8 | 16 |
| K=2048,N=4096,M=16 | 8 |
| K=4096,N=1536,M<=8 | 16 |
| K=4096,N=1536,M=16 | 8 |
| K=4096,N=4096,M<=8 | 32 |
| K=4096,N=4096,M=16 | 16 |
| K=8192,N=4096,M<=16 | 32 |

不能只看 K。至少计算：

```text
base_blocks = ceil(N/BN) * ceil(M/BM)
total_blocks = base_blocks * split_k
```

当 base_blocks 已远大于 CU 数，例如 N=32768，不应使用大 split。当 M/N block不足、
K-loop很长时，split-K才有价值。候选 split 建议限制为 `{1,2,4,8,16,32}`，并要求：

```text
K / split_k >= BK
K % (split_k * BK) == 0
```

### 7.1 M/K/N参数范围

当前 `BM=8, BN=64, BK=32`。先计算未split时的基础并行度：

```text
m_blocks = ceil(M / 8)
n_blocks = ceil(N / 64)
base_blocks = m_blocks * n_blocks
```

#### M

| M | 建议 |
|---:|---|
| 1、2、4、8 | 核心适用区间，优先扫描较大split |
| 9～16 | 仍适用；通常比M<=8少一档split |
| 17～32 | 本轮无证据；先比较split 1/2/4和普通GEMM |
| >32 | 不依据本文启用；优先tiled GEMM或MMAC/MFMA |

#### K

| K | 初始候选 |
|---:|---|
| <1024 | 通常不split |
| 1024 | split 1/2/4 |
| 2048 | split 8/16；M=16加入4 |
| 4096 | split 8/16/32 |
| 8192 | split 16/32 |
| >8192 | 未测，不得直接外推 |

#### N与base grid

| N/grid特征 | 初始判断 |
|---|---|
| `base_blocks >= 512` | N方向已充分并行，split通常只需1～4 |
| N约4096、base_blocks约64～128 | K>=2048时split 8～32通常有价值 |
| N约1024～2048、base_blocks约16～64 | grid不足，大K优先较大split |
| N<512 | 本轮数据不足，必须实测 |

`512`只是本轮 `N=32768` 观测给出的保守分界，不是通用硬件常数。

### 7.2 未测shape的保守heuristic

```text
if M > 16:
    不走本路径
else if base_blocks >= 512 or K <= 1024:
    扫 {1,2,4}
else if N <= 2048:
    M<=8：目标每split K约256
    M>8 ：目标每split K约512
else if N <= 4096:
    M<=8：目标每split K约128～256
    M>8 ：目标每split K约256～512
else:
    扫 {2,4,8,16}
```

估算值向 `{1,2,4,8,16,32}` 中合法的2次幂取整，并同时满足：

```text
split_k <= 32
K / split_k >= 32
K % (split_k * 32) == 0
```

实测对应关系：

```text
M=4,N=1536,K=4096：base_blocks=24，split=16最佳
M=16,N=1536,K=4096：base_blocks=48，split=8最佳
M=4,N=4096,K=2048：base_blocks=64，split=16最佳
M=4,N=32768,K=1024：base_blocks=512，split=4最佳，split=2非常接近
```

### 7.3 精确表优先，heuristic兜底

经过50～100 sample、多轮复测后，可把稳定shape写入精确dispatch表：

```cpp
int choose_split_k(int M, int N, int K) {
    if (M > 16) return 1; // 改走其他kernel
    if (K == 8192 && N == 4096) return 32;
    if (K == 4096 && N == 1536) return M <= 8 ? 16 : 8;
    if (K == 4096 && N == 4096) return M <= 8 ? 32 : 16;
    if (K == 2048 && N == 4096) return M <= 8 ? 16 : 8;
    if (K == 1024 && N == 32768) return M == 4 ? 4 : 2;
    return 1; // 未测shape保守fallback或离线tune
}
```

以上精确值当前仍是待复验参数，不能仅凭单轮10-sample结果直接发布。

## 8. 适用与不适用

适用：

- `M <= 16`，K-loop长；
- M/N方向 block 不足以填满设备；
- dense row-major INT8 W；
- K/N满足向量对齐；
- 允许 INT32 workspace 和第二个 kernel。

不适用或需谨慎：

- large-M/prefill，应优先MMAC/MFMA；
- N极大且base grid已充分；
- K很小，launch/reduction成本占主导；
- 无workspace或禁止第二个kernel；
- 非对齐shape且没有安全fallback。

## 9. Agent 实施规则

1. 保留无split或原kernel fallback；
2. partial必须为INT32；
3. 计时包含主kernel和reduction；
4. workspace分配放在timed region外；
5. 覆盖所有受dispatch影响的shape，不能只报最优case；
6. 每个split先做完整正确性；
7. 同时报Triton、原candidate、新candidate；
8. 正式结论至少50～100 samples并重复三轮；
9. 反汇编确认 `v_dot4_i32_i8` 和预期宽度的global load；
10. 检查VGPR、LDS、occupancy和workspace上限。

## 10. 已知风险与待办

### Launch bounds

实验曾出现：

```text
Launch params (64, 8, 1) are larger than launch bounds (256)
```

当前 block 是512 work-items。虽然测试成功且正确，正式集成前必须显式验证/设置
`__launch_bounds__(512)`，或重构成256 work-items并重新测试。不得忽略此警告。

### 样本量与非单调结果

部分shape的split-8慢于split-4/16/32，可能来自cache、调度、频率或采样不足。正式
dispatch必须增加样本重复确认，不能仅凭本表硬编码。

### Triton计时语义

当前 Triton wrapper 会创建输出tensor；GPU Event范围可能包含相关device zeroing，
而HIP candidate使用预分配输出。这与冻结evaluator一致，适合比较任务分数；分析纯
kernel ISA时应另做预分配输出或直接profile生成kernel。

### 后续方向

- 256-thread与512-thread版本比较；
- reduction/epilogue融合或更紧凑partial layout；
- 自动选择split；
- 双缓冲global-to-LDS；
- TP4/TP8单-rank shape；
- 真正4卡TP4 + RCCL端到端测试；
- large-M使用 `__builtin_hcu_mmac_i32_16x16x32_i8`。

## 11. 工程判断

这批数据证明的不是“split越大越好”，而是：

> small-M/large-K的重要瓶颈之一是grid并行度不足。Split-K把长K-loop变成更多
> 独立workgroups；128-bit load与SDOT4保证每个分片仍高效。最佳split由M、N、K、
> CU数量和reduction成本共同决定。

应采用shape-aware dispatch：small-M使用split-K SDOT4，large-M使用普通tiled
GEMM或MMAC/MFMA，不能用一个配置覆盖全部workload。
