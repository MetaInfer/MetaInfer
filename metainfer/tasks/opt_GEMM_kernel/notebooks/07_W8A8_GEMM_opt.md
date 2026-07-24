# 海光 K100 W8A8 INT8 GEMM 算子优化思路

## 1. 算子目标与当前基线

目标算子完成以下流程：

```text
x_bf16[M,K]
    │
    ├─ per-token 动态量化
    │    x_scale[m] = max(abs(x[m,:])) / 127
    │    x_q[M,K]   = round(x_bf16 / x_scale)
    ▼
x_q[M,K] INT8
    │
    │  x_q[M,K] @ weight[K,N]
    ▼
acc[M,N] INT32
    │
    │  float(acc) * x_scale[m] * weight_scale[n]
    ▼
out[M,N] BF16/FP16
```

当前基线代码包含三个部分：

1. `quantize_bf16_per_token_kernel`：BF16 激活按 token 动态量化为 INT8；
2. `w8a8_scaled_gemv_kernel`：`M <= 8` 的 decode/small-M 路径；
3. `w8a8_scaled_gemm_kernel`：`M > 8` 的通用 LDS tiled GEMM 路径。

当前版本的定位是**正确性基线**，主要问题是仍使用标量：

```cpp
acc += int(a) * int(b);
```

尚未使用 K100 上可能存在的 packed INT8 dot、矩阵指令、权重预打包、双缓冲和专门的 wave 映射。

---

## 2. 目标 Shape

需求中主要需要覆盖以下本地 GEMM Shape：

| 模块 | TP=1 | TP=4 | TP=8 |
|---|---|---|---|
| `wqkv_a` | `(M,4096)@(4096,1536)` | 同 TP1 | 同 TP1 |
| `wq_b` | `(M,1024)@(1024,32768)` | `(M,1024)@(1024,8192)` | `(M,1024)@(1024,4096)` |
| `indexer.wq_b` | `(M,1024)@(1024,8192)` | 同 TP1 | 同 TP1 |
| `wo_b` | `(M,8192)@(8192,4096)` | `(M,2048)@(2048,4096)` | `(M,1024)@(1024,4096)` |
| `shared gate_up_proj` | `(M,4096)@(4096,4096)` | `(M,4096)@(4096,1024)` | `(M,4096)@(4096,512)` |
| `shared down_proj` | `(M,2048)@(2048,4096)` | `(M,512)@(512,4096)` | `(M,256)@(256,4096)` |

按性能特征可分为三组：

- **Decode/small-M**：`M=1、2、4、8`，更接近 GEMV 或 small-M GEMM；
- **中等 M**：`M=16～128`，既要考虑权重带宽，也要开始利用 B 在 M 维的复用；
- **Prefill/large-M**：`M>=128`，更接近标准 GEMM，矩阵计算吞吐与 LDS pipeline 更重要。

不建议所有 Shape 强行共用同一个 kernel 配置，应至少按 `M` 分流，并进一步按 `K/N` 选择 tile。

---

## 3. Decode 阶段的核心判断

计算为：

```text
A[M,K] INT8 @ B[K,N] INT8
```

其中：

- A 是激活 `x_q`，decode 时通常只有几百字节到几十 KB；
- B 是模型权重，通常为几 MB 到几十 MB；
- 对小 M 而言，算子大多受**权重读取带宽**限制，而不是 A 的读取限制。

仅考虑 B 的流量，算术强度近似为：

```text
2 * M * K * N ops / (K * N bytes) ≈ 2M ops/byte
```

因此：

- `M=1`：约 2 ops/byte，典型带宽受限；
- `M=4`：约 8 ops/byte；
- `M=8`：约 16 ops/byte，才开始有更明显的权重复用价值。

Decode 优化的第一原则是：

> 减少 B 的重复读取，并让每次 B 读取服务尽可能多的 M 行和多个 INT8 MAC。

---

## 4. 当前 small-M Kernel 的主要瓶颈

当前 small-M 路径采用：

```text
grid.y = M
一个 block 只处理一个 token 行
一个线程只计算一个输出列
每个线程串行遍历完整 K
```

### 4.1 不同 M 行重复读取完整权重

当 `M=8` 时，当前实现为每个 token 单独启动一组 N tiles：

```text
row 0 的 block 扫描一遍 B
row 1 的 block 再扫描一遍 B
...
row 7 的 block 再扫描一遍 B
```

即便 L2 能提供部分命中，B 仍会从 L2 到 CU 重复传输，且无法在同一个 workgroup 内直接复用。

**建议优先改为一个 block 同时计算多个 M 行**：

```text
一个 block：处理 M_TILE 个 token × N_TILE 个输出通道
A tile：[M_TILE, K_TILE] 放 LDS/寄存器
B tile：[K_TILE, N_TILE] 只加载一次
同一份 B 同时更新 M_TILE 组 accumulator
```

对 `M=2/4/8`，这通常是收益最大的结构性改动之一。

### 4.2 每线程只计算一个输出，ILP 偏低

当前每个线程只有一个 `int32 acc`。可以尝试：

```text
每线程计算 2～4 个 N 列
```

收益：

- 同一个 A pack 可复用到多个 B pack；
- 增加 instruction-level parallelism；
- 更容易隐藏 B load 延迟。

代价：

- 寄存器使用量增加；
- block 的 N 并行度下降；
- 需要结合 CU 数量和 occupancy 调参。

### 4.3 K 维完全标量串行

当前：

```cpp
for (kk = 0; kk < K; ++kk) {
    acc += int(a[kk]) * int(weight[kk * N + n]);
}
```

每次只处理一对 INT8。应优先确认 K100 是否支持：

- 4×INT8 packed dot；
- 8×INT8 packed dot；
- INT8 矩阵/MFMA 类指令；
- 编译器内建函数或可生成对应 ISA 的向量类型。

如果存在 dot4，一次指令可完成类似：

```text
4 个 int8 × 4 个 int8 → int32 累加
```

理论上可显著减少整数乘加指令数。

---

## 5. 权重预打包：small-M 优化的关键

原始权重布局为：

```text
weight[K,N] row-major
地址 = k * N + n
```

该布局的优点是：固定 `k` 时，相邻线程读取相邻 `n`，访存连续。

但对于某个固定输出列 `n`，连续 4 个 K 元素的地址为：

```text
weight[k+0,n]
weight[k+1,n]
weight[k+2,n]
weight[k+3,n]
```

它们相隔 `N`，无法直接作为一个连续 `int32` 读入，因此不适合直接使用 K 维 dot4。

### 5.1 推荐的 blocked packed layout

可以在模型加载阶段把权重预处理为：

```text
B_pack[N_TILE_ID][K_GROUP][N_INNER]
```

其中每个元素是一个 32-bit packed word：

```text
uint32 = {B[k+0,n], B[k+1,n], B[k+2,n], B[k+3,n]}
```

逻辑布局可表示为：

```text
[N / BN][K / 4][BN] uint32
```

访问方式：

- 固定 `K_GROUP`；
- 相邻线程处理相邻 `N_INNER`；
- 相邻线程读取相邻的 `uint32`；
- 每个 `uint32` 内包含该输出列连续 4 个 K 权重。

这样同时满足：

1. wave 内 B 读取合并；
2. 单线程获得连续 K 的 4 个 INT8；
3. 可直接执行 packed dot4；
4. 不需要运行时 transpose/pack。

### 5.2 预打包注意事项

- 预处理只在模型加载时做一次，不能每次 forward 重新 pack；
- 若 N 维发生 permutation，`weight_scale[n]` 必须同步重排；
- 建议为不同主要 Shape 单独选择 `BN/BK` pack；
- packed weight 应保持 16B/32B 对齐；
- 原始权重是否保留，取决于框架内存预算与其他算子是否还需要原布局。

---

## 6. A 激活的缓存策略

A 在 decode 阶段很小，应优先利用：

```text
寄存器 > LDS > L1/L2 > HBM
```

### 6.1 workgroup 内复用

建议按 K 分块：

```text
A[M_TILE, BK] global → LDS
B[BK, BN] global/packed → register 或 LDS
计算 M_TILE × BN 输出
```

A tile 被整个 workgroup 的所有 N 线程复用。

### 6.2 persistent N-loop

一个 block 加载 A 后，可连续处理多个 N tiles：

```text
加载 A
计算 N tile 0
计算 N tile 1
计算 N tile 2
...
```

这能进一步降低 A 重载和 block 启动开销，但不能让 block 数量过少，否则会损失 CU 并行度。

建议尝试：

```text
每个 block 连续处理 2～4 个 N_TILE
```

并对比：

- CU active ratio；
- occupancy；
- 总权重带宽；
- A 的 L2/LDS 流量。

---

## 7. B 是否需要经过 LDS

完整 B 必须保存在 HBM，不能整体放入 LDS。只能对局部 tile 做暂存。

### small-M

对于 `M=1`：

- B 元素基本只服务一个激活行；
- `global → LDS → register` 可能增加一次写 LDS 和读 LDS；
- 如果 packed dot 指令可直接使用 global load 到 register，B 不一定要经过 LDS。

优先尝试：

```text
A → LDS/寄存器
B_pack → 向量化 global/L2 load → 寄存器
packed dot → INT32 acc
```

### M=4/8 或更大

同一份 B 可服务多行 A，此时 B tile 放 LDS 的价值明显提高：

```text
B tile 加载一次
被 M_TILE 行共同复用
```

因此可设计两条 decode 路径：

```text
M=1/2：B 直接 global → register
M=4/8：B global → LDS，共享给多行 A
```

最终以 benchmark 结果决定阈值。

---

## 8. 向量化和访存优化

### 8.1 A 的向量化加载

`x_q` 连续存储，可尝试：

```text
int32/uint32：一次加载 4 个 INT8
int4/16B：一次加载 16 个 INT8
```

前提：

- 地址满足对齐；
- K 是 4/16 的倍数；
- 尾块做 mask 或单独处理。

当前目标 K 均为 256 的倍数或更大，天然适合 packed load。

### 8.2 B 的向量化加载

原始 `[K,N]` 布局适合固定 k、沿 N 向量加载；预打包布局则适合每线程沿 K 获取 dot4 word。

两条方案应分别 benchmark：

1. **不 pack**：保持沿 N 合并，标量/向量 N load；
2. **预 pack**：每线程读取 K-packed 32-bit word，执行 dot4。

### 8.3 输出与 scale

当前 epilogue 已融合：

```cpp
float y = float(acc) * x_scale[row] * weight_scale[col];
out = bf16(y);
```

这是正确方向，不应把完整 INT32 accumulator 写回显存再单独反量化。

还可进一步：

- 每线程多输出时，向量化读取多个 `weight_scale`；
- BF16 两个一组或更宽地写回；
- 将 `x_scale[row]` 预加载到标量寄存器；
- 避免在内层重复计算地址。

---

## 9. 动态量化 Kernel 的优化

当前量化 kernel 对每行执行：

1. 第一遍读取 BF16，求 absmax；
2. 第二遍再次读取 BF16，除 scale、舍入、写 INT8。

### 9.1 用 wave reduction 替代完整 LDS reduction

当前使用 256 个 float 的 LDS reduction，并在每轮调用 `__syncthreads()`。

可改为：

```text
线程局部 max
→ wave 内 shuffle/max reduction
→ 每个 wave 写一个 partial max 到 LDS
→ 第一条 wave 再做最终 reduction
```

可减少：

- LDS 访问；
- barrier 数量；
- reduction 延迟。

### 9.2 BF16 向量化读取

可尝试一次加载：

```text
2/4/8 个 BF16
```

在寄存器中转 FP32、求绝对值和局部 max。

### 9.3 一次全局读取 + LDS 暂存

对 K=256/512/1024 时，一行 BF16 大小为：

```text
512B / 1KB / 2KB
```

可考虑第一次读取时直接放入 LDS，求完 scale 后从 LDS 量化，避免第二次 HBM/L2 读取。

对 K=4096/8192 时，一行大小为 8KB/16KB，仍可尝试，但要评估 LDS 占用与 occupancy。

### 9.4 是否融合 quant + GEMM

融合难点是：

- 必须先完成整行 absmax，才能得到 scale；
- 若每个 N block 都独立量化，会重复做同一份 A 量化；
- 若一个 persistent block 处理多个 N tiles，则有机会把 `x_q` 保留在 LDS/寄存器中。

建议顺序：

1. 先优化独立 quant kernel；
2. small-M GEMM 达到较高带宽后再评估 quant 占比；
3. 只有 quant 启动和中间写读确实成为显著比例时，再做融合。

---

## 10. 通用/large-M GEMM 优化

当前通用 kernel 为：

```text
BM=16, BN=16, BK=32
每线程只计算 1 个 C 元素
A/B 均经 LDS
标量 INT8 MAC
```

主要优化方向如下。

### 10.1 每线程计算多个输出

从：

```text
1 thread → 1 accumulator
```

改为：

```text
1 thread → TM × TN accumulators
```

例如：

```text
TM=2/4，TN=2/4
```

可以显著提高 A/B 寄存器片段复用，降低每个输出对应的 LDS load 次数。

### 10.2 更大的 CTA tile

需要测试：

```text
BM ∈ {16, 32, 64}
BN ∈ {32, 64, 128}
BK ∈ {32, 64, 128}
```

选择受以下因素约束：

- LDS 容量；
- accumulator 寄存器数量；
- occupancy；
- K100 的 wave size；
- INT8 矩阵指令支持的 tile 形状。

### 10.3 双缓冲/多 stage pipeline

LDS 分为两个 buffer：

```text
stage 0：当前计算
stage 1：预取下一 K tile
```

目标是实现：

```text
global load 下一 tile
与
当前 tile 的 dot/MMA
重叠执行
```

需检查编译后的 ISA 是否真正产生异步/重叠，而不是源码上看似双缓冲但实际串行。

### 10.4 LDS bank conflict 与 padding

当前：

```cpp
int8_t a_tile[BM][BK];
int8_t b_tile[BK][BN];
```

需要结合 wave lane 访问模式检查 bank conflict。常见处理包括：

- 行末 padding；
- blocked/swizzle layout；
- 以 32-bit/128-bit word 存储；
- 让矩阵指令要求的 lane layout 与 LDS layout 匹配。

### 10.5 使用 K100 INT8 矩阵/点积指令

这是 large-M 性能提升的核心。需要先确认：

- 目标架构名称；
- 编译器支持的 builtins/intrinsics；
- `-mattr` 可用特性；
- 对应 ISA 的输入布局、累加类型与 tile 约束。

未确认前不要直接假定其等同于 AMD 某一代 MFMA 指令。

---

## 11. Shape 专用 dispatch

建议 dispatch 至少考虑：

```text
M、K、N、是否预打包、输出类型
```

初始策略可以是：

```cpp
if (M == 1) {
    launch_decode_m1(...);
} else if (M <= 8) {
    launch_decode_multi_m(...);
} else if (M <= 64) {
    launch_small_gemm(...);
} else {
    launch_large_gemm(...);
}
```

进一步按 K 分类：

```text
small K：256 / 512
medium K：1024 / 1536 / 2048
large K：4096 / 8192
```

原因：

- K=256 时 pipeline 很短，kernel launch/epilogue 占比更高；
- K=8192 时主循环很长，更适合多 stage 和大 BK；
- N=32768 时 N 方向并行度极大；
- N=512 时应避免 BN 过大导致 tile 数不足。

---

## 12. TP 语义与通信边界

### Column parallel

如：

```text
wq_b
gate_up_proj
```

切 N，每个 rank 计算不同输出列，通常保持分片或后续 AllGather。

### Row parallel

如：

```text
wo_b
shared down_proj
```

切 K，每个 rank 计算 `[M,N]` partial output，随后需要求和。

对于动态量化，要特别确认：

- 每个 rank 的 `x_scale` 是基于本地 K 求 max，还是全局 K 求 max；
- 若各 rank 的 scale 不同，必须先在本地把 INT32 partial accumulator 乘本地 scale 恢复为浮点，再做 AllReduce；
- 不能在 scale 不一致时直接 AllReduce INT32 accumulator 后统一乘一个 scale。

通信是否融合进 kernel，应在单卡计算稳定后再考虑。

---

## 13. 性能测量指标

不能只看 TOPS，建议同时统计：

### 13.1 分阶段耗时

```text
quant time
scaled GEMM time
quant + GEMM end-to-end time
```

### 13.2 有效权重带宽

Decode 时可近似：

```text
weight_GBps = K * N bytes / kernel_time
```

如果一个 kernel 同时处理 M 行且 B 只加载一次，该指标更能反映是否接近硬件 HBM/L2 带宽上限。

### 13.3 计算吞吐

```text
TOPS = 2 * M * N * K / time
```

对于 bandwidth-bound 的 M=1，TOPS 不高并不一定说明 kernel 差，应同时看 GB/s。

### 13.4 Profiling 指标

重点观察：

- HBM/DRAM 带宽利用率；
- L2 hit rate 与 L2→CU 流量；
- vector/scalar global load 比例；
- INT8 dot/矩阵指令是否实际生成；
- active waves、occupancy；
- VGPR/SGPR 使用量；
- LDS 使用量和 bank conflict；
- barrier 等待占比；
- 每个 kernel 的 launch overhead。

---

## 14. 正确性要求

### 动态量化

应检查：

```text
x_scale[m] = max(abs(x[m,:])) / 127
x_q = round-to-nearest-even(x / x_scale)
clamp 到 [-128,127]
```

必须显式处理全零 token，避免 `0/0`。

### GEMM

数学语义必须为：

```text
INT8 × INT8 → INT32 accumulator
```

不要使用 FP32 GEMM 作为最终严格语义，只可作为宽松 reference。

### Epilogue

```text
float(acc_int32) * x_scale[m] * weight_scale[n]
```

在 FP32 中完成，最后一次性转换为 BF16/FP16。

### 预打包

确保：

- packed weight 与原始权重逐元素一致；
- `weight_scale[n]` 与输出通道映射一致；
- 不同 TP rank 的权重片段没有错位。

---

## 15. 推荐的优化实施顺序

### 阶段 0：建立可靠基线

- 所有目标 Shape 正确；
- 输出与 INT32 CPU reference 对齐；
- quant、GEMM、end-to-end 分别计时；
- 固定 warmup、迭代次数和设备频率条件。

### 阶段 1：small-M 结构优化

优先级：

1. 一个 block 同时处理多行 M，B 加载一次服务 `M_TILE` 行；
2. 每线程计算多个 N 输出；
3. A 使用 32/128-bit 向量化加载；
4. 减少地址计算和分支；
5. 评估 persistent N-loop。

### 阶段 2：权重 pack + packed dot

1. 确认 K100 的 INT8 dot/矩阵指令；
2. 设计 `[N/BN][K/4][BN]` 或硬件要求的 blocked layout；
3. 模型加载时一次性 pack；
4. kernel 使用 32-bit packed B 和 packed A；
5. 对比 ISA，确认不是被编译器展开回标量乘法。

### 阶段 3：large-M 矩阵核版本

1. 更大的 BM/BN/BK；
2. 每线程多 accumulator；
3. A/B vectorized global→LDS；
4. LDS swizzle/padding；
5. double buffering；
6. INT8 matrix instruction；
7. 自动调参或 Shape 专用配置表。

### 阶段 4：量化与系统级融合

1. wave reduction 优化 quant；
2. 评估一次读取并在 LDS 暂存 BF16；
3. 评估 quant+small-M GEMM 融合；
4. 最后再考虑 TP AllReduce 融合或与上游算子融合。

---

## 16. 建议首先验证的硬件信息

在容器中收集：

```bash
hipconfig --version
hipcc --version
rocminfo | grep -E 'Name:|Marketing Name|Wavefront Size' | head -40
```

查找 LLVM 工具：

```bash
find /opt/dtk -type f \
  \( -name llvm-objdump -o -name llvm-mc -o -name llc \) \
  2>/dev/null
```

保存中间产物并反汇编：

```bash
hipcc -O3 -std=c++17 --save-temps \
  w8a8_gemm_k100_baseline.hip \
  -o w8a8_gemm_k100_baseline
```

然后重点确认：

- 实际 target CPU/架构名；
- 标量 INT8 乘法对应的 ISA；
- 是否存在 packed INT8 dot；
- 是否存在 INT8 矩阵指令；
- 128-bit load 是否真的生成；
- 使用 LDS 后的 load/store 与 barrier 数量。

---

## 17. 当前最值得先做的两个实验

### 实验 A：多 M 行共享一份 B

把当前：

```text
一个 block 处理 1 行 M
```

改成：

```text
一个 block 处理 2/4/8 行 M
```

对比 `M=1/2/4/8` 下：

- kernel time；
- effective weight GB/s；
- L2 traffic；
- register/LDS 占用。

### 实验 B：预打包 B + dot4

先针对固定 Shape：

```text
(M,1024) @ (1024,4096)
```

实现：

```text
B_pack[N/BN][K/4][BN] uint32
```

对比：

- 原始 `[K,N]` 标量 MAC；
- packed B + 4-way INT8 dot；
- 生成 ISA；
- 实际加速比。

如果实验 B 无法生成 packed dot 指令，应先停止继续复杂化 pack，回到 ISA/编译器能力确认。

---

## 18. 总结

该算子的优化应分成两条主线：

### Decode/small-M

```text
核心瓶颈：权重 B 的流式读取
核心策略：一份 B 服务多个 M 行 + packed INT8 dot + 权重预打包
```

### Prefill/large-M

```text
核心瓶颈：矩阵计算吞吐、LDS pipeline 和矩阵指令利用率
核心策略：更大 tile + 多 accumulator + 双缓冲 + INT8 matrix instruction
```

当前最优先的工作不是简单增大 shared memory，而是：

1. 让一个 B tile 同时服务多个 token；
2. 让一次 B load 携带多个 K 方向 INT8；
3. 确认 K100 的 packed dot/矩阵指令并设计匹配的权重布局；
4. 使用 Shape 专用 dispatch，而不是一套 kernel 覆盖所有 M/K/N。
