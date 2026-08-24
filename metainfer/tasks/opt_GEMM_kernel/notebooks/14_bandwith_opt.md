# 从旧 SDOT4 算子到 W8A8 packed Marlin/MMAC 的优化总结

本文专门解释目录中的算子相对以下旧实现做了什么改进：

```text
/data/FF/MetaInfer/nodes/worker26/workspaces/
opt-gemm-kernel-2067149b/010/submission/myGEMM_kernel.hip
```

目标读者是需要参考这些经验继续编写 gfx928 HIP kernel 的 AI 或工程师。
旧版本对照是 `w8a8-marlin-fused_backup_20260730_pre_stripe`；最终生产证据来自
`/home/FF/workspace/003/build.sh`、`w8a8_gemm_fused.cpp` 及其实际编译的 kernel
source。源码存在或进入最终共享库不等于生产启用；判断某个 shape 的真实路径时，
必须同时找到 dispatcher 的 shape guard 和对应 launch。

## 0. 最终 `/home/FF/workspace/003` 的证据边界

### 0.1 三种状态必须分开

- **生产 dispatch**：最终 `w8a8_gemm_fused.cpp` 对某个 shape family 有可达 guard，
  并调用对应 launcher。这是“最终版本实际启用”的必要证据。
- **编译进最终库**：`build.sh` 将 source 编译进 `libw8a8_marlin_fused.so`，但某个
  kernel 仍可能只被特定 guard 使用，不能推广到其他 shape。
- **实验记录**：workspace 中未进入最终 `build.sh`，或最终 dispatcher 没有可达调用的
  source/report，只能作为候选假设，不能写成生产 dispatch。

最终 `build.sh` 明确编译 packed-weight、small-M packed MMAC、BM64、N8192 BM32、
两个 M32/K4096 winner、M4096 kernel 和最终 dispatcher。生产 route 由 dispatcher
进一步限定：

| shape guard | 最终 launcher | 已启用的主要机制 | 主要代价/复核点 |
|---|---|---|---|
| `M<=32, N=4096, K=1024` | `launch_bm64_b_vgpr` | BM64；B-in-VGPR；A-only LDS；A direct-to-LDS；细粒度 `ds_read_b64`/MMAC interleave；bit-reverse N scheduling；stride-17 epilogue | B fragment 和 accumulator 增加 VGPR；重新检查 occupancy、HBM 和尾部 mask |
| `M<=32, N=8192, K=1024` | `launch_bm32_vgpr_db_n8192` | BM32；A direct-to-LDS；B global read 使用 SLC；A/B double buffering；B prefetch 与 MMAC 重叠 | SLC 和 prefetch 是该 shape 的实测选择，不应泛化；检查 cache、VGPR 和并发 wave |
| `M<=32, N=1536, K=4096` | `launch_m32_k4096_wqkv_winner` | shape-specific M32/K4096 pipeline | 只对该 N guard 有生产证据；与通用 split-K 比较总 dispatch operator time |
| `M<=32, N=512, K=4096` | `launch_m32_k4096_shared_winner` | shape-specific shared pipeline | LDS 容量、barrier 和 occupancy 必须重新实测 |
| `M=4096` 的 dispatcher 列举 N/K family | `launch_m4096_bm128_bn128_group4` | BM128xBN128xBK64；8 waves；`GROUP_M=4` CTA ordering 复用 L2 中的 B；`ds_read_b64` fragments feeding MMAC | tile 大、wave 多；检查边界 guard、寄存器/LDS 和不同 N/K 下的 L2 收益 |
| dispatcher 的其他明确 family | BM16/BN32、BM16/BN16 或 BM32/BN32 shared launcher | shape-specific tile 路由，避免一个通用 tile 覆盖所有 M/N/K | 每个 guard 独立验证，不能按相近 shape 推断 winner |
| 未命中上述专用路径的支持 small-M shape | packed MMAC split-K | packed W8A8；caller-owned persistent `partial`/`tile_done`；last-arriving CTA fused reduction | partial HBM、fence/atomic、split 数和 workspace 并发所有权 |

因此，本 notebook 记录的是优化机制和已知适用 guard，不是要求 agent 复制最终 dispatch
表。新迭代必须从当前 `submission/` 和当前 frozen shapes 重新建立 route map，再用当前
hipprof trace 的每次调用全部 dispatch `DurationNs` 之和比较。PMC 只解释 HBM、L2、
register、LDS/scratch 和可靠可用的 wave/occupancy 信息；PMC replay duration 不是 latency。
绝对微秒数和 workspace 中的历史 winner 名称都不是跨机器 SLA。

### 0.2 最终版本中确认启用的技术

| 技术 | 最终证据 | 适用范围 | 不应忽略的 trade-off |
|---|---|---|---|
| packed W8A8 MMAC | pack source、MMAC consumers、最终 dispatcher | 静态、可复用权重；多个 small/large-M route | packing 必须在计时区间外完成；layout/consumer lane mapping 必须一致 |
| B-in-VGPR、A-only LDS | `w8a8_bm64_b_vgpr_opt.cpp` 的生产 route | `M<=32,N=4096,K=1024` | VGPR 压力可能降低 resident waves |
| Direct-to-LDS | BM64、N8192 等生产 source | 对齐且 guard 满足的 A tile load | alignment、尾部和 builtin 语义必须保持安全 |
| A/B double buffering | N8192 production source；其他 route 以各自 source 为准 | memory latency 可与 MMAC 重叠的固定 shape | LDS/VGPR 增长和 barrier 次序可能抵消收益 |
| 细粒度 `ds_read`/MMAC interleave | BM64/M4096 production source | 有足够独立 fragment 的 MMAC loop | 调度改变必须以 operator time 和资源计数复核 |
| SLC read | N8192/K1024 production route | 当前只有该 guard 的启用证据 | cache policy 的收益依赖工作集和并发，不可全局开启 |
| bit-reverse N scheduling | BM64 production route | 当前 BM64 N-block ordering | 可能改变 L2 locality/负载均衡，需按 shape 复测 |
| stride-17/padded epilogue | BM64 production route | LDS epilogue transpose/store | 多占 LDS；必须验证 bank conflict 与 occupancy 的净效果 |
| shape-specific BM16/BM32/BM64 | dispatcher guards | dispatcher 明确列举的 family | 不能只按 M 或 N 相似就复用 |
| BM128xBN128 + `GROUP_M=4` | M4096 production route | dispatcher 明确列举的 M4096 family | 大 tile 资源压力和边界浪费 |
| caller-owned persistent split-K workspace | V2 query/launch ABI 与 fallback route | 需要 split-K 的支持 shape | 同一 workspace 不得被无保护的并发 stream 复用 |
| last-arriving-CTA fused reduction | packed small-M split-K consumer | partial/ticket protocol 使用的 route | device fence、ticket 清零、跨调用和并发正确性 |

这些技术进入知识库的理由是最终 source 和 dispatch 证据，而不是历史报告里的单次
speedup。任何后续采用都必须先提出一个有边界、可测、可回滚的假设，并以当前全部
shape 的 hipprof operator time 及对应 PMC 证据决定保留或回滚。

## 1. 旧算子已经做了什么

旧 `myGEMM_kernel.hip` 并非朴素基线。它已经包含：

- 128-bit global load；
- `__builtin_amdgcn_sdot4` INT8 点积；
- A/B LDS tiling；
- B 从 `[K,N]` 到 `[N,K]` 的运行时 LDS 转置；
- `BK_PAD=BK+4` 的 LDS padding；
- small-M Split-K；
- INT32 partial 后再做 FP32 scale 和 BF16 conversion；
- M>16 的 BM16/BN16/BK64 SDOT4 kernel。

旧 small-M 核心是“一线程负责一个输出元素”：

```cpp
// myGEMM_kernel.hip
const int row = int(blockIdx.y) * BM + ty;
const int col = int(blockIdx.x) * BN + tx;
int32_t acc = 0;

for (int kk = 0; kk < BK; kk += 4) {
    const int32_t a_pack =
        *reinterpret_cast<const int32_t*>(&a_tile[ty][kk]);
    const int32_t b_pack =
        *reinterpret_cast<const int32_t*>(&b_tile[tx][kk]);
    acc = __builtin_amdgcn_sdot4(a_pack, b_pack, acc, false);
}
```

每处理一个 K tile，旧算子都要：

```text
读取 A + 读取 raw W + 在 LDS 中转置 W
    → __syncthreads()
    → SDOT4
    → __syncthreads()
    → 下一个 K tile
```

因此新版本的改进重点不是再加一层 vector load，而是改变计算 primitive、静态
权重布局、同步结构、Split-K 归约协议和 workspace ABI。

## 2. 改进总览

| 方面 | 旧 myGEMM | packed Marlin/MMAC | 原理 |
|---|---|---|---|
| 计算 | 每线程一个 C，循环 SDOT4 | Wave64 执行 16×16×32 INT8 MMAC | 硬件矩阵指令复用 operand，减少点积/地址指令 |
| W 布局 | forward 读取 raw `[K,N]` | 模型加载期 pack 成 MMAC lane layout | 一次性预处理静态权重 |
| B 路径 | global→LDS transpose→VGPR | packed global→VGPR | 删除 B LDS、转置和 bank conflict |
| A staging | 每 K32 tile staging | 每 split staging 完整 A slice | 主 K 循环从每 tile 两个 barrier 降到一次 barrier |
| M tile | BM=1/2/4/8；M9..16 用 BM8 | 固定 BM16，非法行 mask | 同一 A slice 覆盖完整 M16 |
| Split-K reduce | 两次 kernel launch | last-arriving CTA 融合归约 | 减少短 kernel 的 launch 开销 |
| K 切分 | `span=K/split_k` | 按 K32 tile 比例切分 | 不要求等长整数 span，无遗漏和重复 |
| workspace | 库内 map/mutex/hipMalloc | 调用方查询并持久分配 | forward 不分配、不做 host mutex/sync |
| split_k | 常见 8/16/32 | shape 实测后常见 2/4/8 | 平衡并行度与 partial/reduce 成本 |
| M>16 | BM16/BN16/BK64 SDOT4 | BM16/BN64 packed MMAC no-split | 一次覆盖更多 N，消除运行时 B 转置 |

## 3. 改进一：SDOT4 改成 Wave64 MMAC

新算子使用 gfx928 的：

```cpp
v_mmac_i32_16x16x32_i8
```

关键封装：

```cpp
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef int int4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int4_t mmac_i32_16x16x32_i8(
    int2_t a, int2_t b, int4_t c) {
    __builtin_amdgcn_sched_barrier(0);
    __asm__ __volatile__(
        "v_mmac_i32_16x16x32_i8 %0, %1, %2, %0"
        : "+v"(c) : "v"(a), "v"(b));
    __builtin_amdgcn_sched_barrier(0);
    return c;
}
```

正式 small-M tile 是：

```text
BM=16, BN=64, BK=32
4 Wave64/CTA，256 threads
wave 0..3 分别计算四个 N16
row16  = lane & 15
k_group = lane >> 4
```

每 lane 提供 8B A 和 8B B，并持有四个 INT32 accumulator。MMAC accumulator
对应输出：

```cpp
row = row_base + (lane & 15);
col = n_block * 64 + wave * 16 + (lane >> 4) + i * 4; // i=0..3
```

原理：旧 SDOT4 仍由软件逐个输出组织点积；MMAC 把一个 Wave64 的数据组织交给
矩阵指令，一次推进 16×16×32 tile，显著减少 dot/loop/address 指令。

AI 写新 kernel 时不能只复制 asm。必须同时保持 lane→A fragment、lane→B fragment、
lane→C accumulator 三个映射完全一致。

## 4. 改进二：把 B 转置移到模型加载阶段

### 4.1 旧算子的重复成本

旧算子每个 forward、每个 K tile 都从 row-major W 读取 16 个连续 N，再写成
K-contiguous LDS：

```cpp
// old: global W[K,N] -> LDS b_tile[N][K_PAD]
Vec128 value = *reinterpret_cast<const Vec128*>(w + w_off);
#pragma unroll
for (int j = 0; j < 16; ++j)
    b_tile[ln + j][lk] = reinterpret_cast<const int8_t*>(&value)[j];
```

这会反复产生：

- B LDS 空间；
- B transpose store 指令；
- LDS bank conflict；
- B 可消费前的 CTA barrier。

### 4.2 新 packed layout

静态 W 在模型初始化时重排为：

```text
packed[n_block64][k_tile32][n_group16][lane64][byte8]
```

精确映射：

```cpp
kg   = lane / 16;
col  = lane % 16;
Ksrc = kt * 32 + kg * 8 + i;
Nsrc = nb * 64 + n_group * 16 + col;
packed[nb][kt][n_group][lane][i] = W[Ksrc][Nsrc];
```

pack kernel 的关键代码：

```cpp
const int n_group = tid >> 6;
const int lane = tid & 63;
const int k_group = lane >> 4;
const int col16 = lane & 15;
const int local_n = n_group * 16 + col16;

const int2_t fragment =
    load_int8x8(&weight_nk[local_n][k_group * 8]);
const size_t chunk =
    (((size_t(n_block) * k_tile_count + k_tile) * 4 + n_group)
      * 64 + lane);
reinterpret_cast<int2_t*>(packed_weight)[chunk] = fragment;
```

forward 中每个 wave 的 64 lanes 读取连续 `64×8=512B`：

```cpp
const size_t chunk =
    (((size_t(n_block) * k_tile_count + global_tile) * 4 + wave)
      * 64 + lane);
const int2_t b_fragment =
    reinterpret_cast<const int2_t*>(packed_w)[chunk];
```

这使 B 完全绕过 LDS，直接 global→VGPR→MMAC。packing 只适合可长期复用的静态
推理权重；如果 W 每次调用都变化，必须把 packing 成本纳入端到端评价。

packed buffer 字节数：

```text
ceil(N/64) * ceil(K/32) * 4 * 64 * 8
```

raw 和 packed 指针不能互换，否则数值会错误但通常不会触发内存异常。

## 5. 改进三：whole-A-slice，删除 K 循环内 barrier

旧 small-M kernel 每 K32 tile staging 一次 A/B，前后各一次 barrier。新 kernel
因为 B 不再使用 LDS，可以把一个 Split-K CTA 负责的全部 A 一次放进 LDS：

```cpp
constexpr int kOptMaxTilesPerSplit = 16;
__shared__ __align__(16)
int8_t a_slice[kOptMaxTilesPerSplit][16][36];

// CTA cooperative 128-bit load all A tiles in this split
// ...
__syncthreads(); // 整个 slice 的主计算只需要这一次同步
```

随后直接循环 LDS A 与 global packed B：

```cpp
for (int local_tile = 0; local_tile < slice_tiles; ++local_tile) {
    const int2_t a = load_int8x8(
        &a_slice[local_tile][row16][k_group * 8]);
    const int2_t b = load_b(local_tile);
    acc = mmac_i32_16x16x32_i8(a, b, acc);
}
```

例如 K=4096、split_k=8 时，每 split 为 16 个 K32 tile。旧主循环约有 32 个
barrier 点；新主计算只需一次 staging barrier。

`36=32+4` 的 stride 保留旧算子已经验证过的 padding 思路：改变相邻行的 LDS
bank 起点，同时保持 dword 对齐。

每 split 最多 16 个 K32 tile 不是算法限制，而是 LDS/occupancy 限制。dispatcher
必须选择足够大的 split_k，使每段不超过此上限。

## 6. 改进四：B prefetch 与双 MMAC accumulator

新 kernel 在消费当前 B 前，提前发出下一 fragment 的 global load，并让偶/奇 K tile
写入不同 accumulator：

```cpp
int4_t acc0{}, acc1{};
int2_t b_next = load_b(0);

for (int t = 0; t < slice_tiles; ++t) {
    const int2_t b = b_next;
    if (t + 1 < slice_tiles)
        b_next = load_b(t + 1);
    const int2_t a = load_int8x8(&a_slice[t][row16][k_group * 8]);
    if ((t & 1) == 0) acc0 = mmac_i32_16x16x32_i8(a, b, acc0);
    else              acc1 = mmac_i32_16x16x32_i8(a, b, acc1);
}
acc = acc0 + acc1;
```

原理：

- prefetch 让下一次 B memory latency 与当前 MMAC 尝试重叠；
- 两条 accumulator dependency chain 给调度器更多独立工作。

代价是增加 live VGPR。不要继续盲目增加 prefetch 深度或 accumulator 数；必须用
hipprof 检查 VGPR、occupancy 和实际 latency。

## 7. 改进五：两次 launch 的 Split-K 改成单 kernel 融合归约

旧实现启动两个 kernel：

```cpp
hipLaunchKernelGGL(small_m_splitk_sdot4_kernel, ...);
hipLaunchKernelGGL(reduce_splitk_scale_kernel, ...);
```

新实现保留 INT32 partial，但加入每个 `(m_block,n_block)` 的 ticket。每个 split 写完
partial 后发布；最后到达的 CTA 负责归约、scale 和 BF16：

```cpp
if (valid_output_lane)
    __threadfence();
__syncthreads();

if (tid == 0) {
    const uint32_t old = atomicAdd(&tile_done[ticket_index], 1u);
    is_last_split = (old == uint32_t(split_k - 1));
}
__syncthreads();

if (is_last_split) {
    // sum partial[s][row][col]
    // float(sum) * a_scale[row] * w_scale[col] -> BF16
    __syncthreads();
    if (tid == 0)
        tile_done[ticket_index] = 0u;
}
```

关键原理：

- `__threadfence()` 必须发生在 atomic ticket 前，保证其他 CTA 能观察 partial；
- atomic 返回 `split_k-1` 的 CTA 是最后到达者；
- ticket 完成后清零，从而让持久 workspace 可用于下一次调用；
- 归约仍为 INT32，所以数值语义没有改变。

这样删除了独立 reduction GPU dispatch 及其 GPU 工作。当前任务的 hipprof operator
latency 不包含 host launch API 时间，但会包含 standalone reduction kernel 的
`DurationNs`；因此是否获益必须看每次调用全部 GPU dispatch 的时长总和，而不是只看
main MMAC。注意同一个 `tile_done` workspace 不可被多个并发 stream 无保护复用。

## 8. 改进六：外部持久 workspace 代替库内 map/mutex/hipMalloc

旧算子在共享库内部维护 per-stream workspace：

```cpp
static std::mutex g_ws_mutex;
static std::unordered_map<DeviceStreamKey,
    std::unique_ptr<StreamWorkspace>, DeviceStreamKeyHash> g_workspaces;

hipStreamSynchronize(stream);
hipFree(ws->partial);
hipMalloc(&ws->partial, required_bytes);
```

扩容时会同步 stream，且 launch path 涉及 host mutex 和隐藏状态。新 V2 ABI 改成：

```cpp
query_w8a8_gemm_v2_workspace(
    M, N, K,
    &partial_elements,
    &tile_done_elements,
    &packed_weight_bytes,
    &split_k);
```

调用者在初始化阶段分配并持久保存：

```text
partial:   split_k * M * N 个 int32
tile_done: ceil(M/16) * ceil(N/64) 个 uint32
packed W:  模型加载时生成一次
```

forward 只传已有指针。收益是：

- 无 forward-time `hipMalloc/hipFree`；
- 无库内全局 map 和 host mutex；
- workspace 生命周期和并发归调用方明确管理；
- server/runtime 可以提前规划显存。

## 9. 改进七：重新标定 Split-K

旧 SDOT4 为增加 CTA 数，使用过较大的 split：

```cpp
// old examples
if (K == 4096 && N == 1536) return M <= 8 ? 16 : 8;
if (K == 4096 && N == 1024) return M <= 4 ? 32 : 16;
if (K == 2048 && N == 4096) return M <= 8 ? 16 : 8;
```

MMAC CTA 每次完成更多计算，过大 split 会增加：

- partial 写入量；
- fence/atomic 数量；
- reducer 读取量；
- 每 CTA 太短造成的效率损失。

因此新版本对关键 shape 扫描后常用 2/4/8：

| N,K | M<=8 | M=16 |
|---|---:|---:|
| 1536,4096 | 8 | 8 |
| 8192,1024 | M1=4，其他=2 | 2 |
| 4096,2048 | 4 | 4 |
| 1024,4096 | 8 | 8 |
| 4096,512 | 2 | 2 |

切分按 K32 tile 做：

```cpp
begin = split * k_tile_count / split_k;
end   = (split + 1) * k_tile_count / split_k;
```

相比旧 `span=K/split_k`，这种写法允许 tile 数不能被 split 整除，同时保证所有 K32
tile 恰好被处理一次。

选择 split_k 的真实目标是平衡：

```text
CTA 并行度收益
vs.
短 CTA 效率 + partial HBM 流量 + fence/atomic + reduction
```

旧 SDOT4 的 split 表不能直接复制给 MMAC kernel。

## 10. M=9..16 和 M>16 的变化

旧算子：

- M=9..16、Split-K 时固定使用 BM8，需要两个 M blocks；
- M=9..16、非 Split-K 时回退到 scalar GEMV；
- M>16 使用 BM16/BN16/BK64 SDOT4，并继续运行时 B LDS 转置。

新算子：

- small-M 固定 BM16，M=9..16 一次覆盖，多余行自然 mask；
- pre-stripe 的 M>16 使用 BM16/BN64 packed MMAC no-split；
- no-split large path 无 partial、ticket 和 reducer，直接做 scale/BF16。

BN16→BN64 的意义是同一次 A staging 服务四个 N16 wave。配合 packed B，运行时不再
转置 B。

但 pre-stripe large-M 仍会让不同 M16 CTA 重读同一 B。它不是最终 large-M 方案；
目录里的 stripe/shared-LDS/direct-LDS 实验就是继续探索跨 M block 的 B reuse。实验
kernel 未必进入正式 `launch_w8a8_gemm_v2`，不可仅凭存在源码就当生产路径。

## 11. 哪些旧思路被保留

新版本并非推倒重来，保留了旧算子中正确的基础：

- A/raw W 优先使用 128-bit aligned load，并保留 tail guard；
- K 基本 tile 仍为 32；
- LDS stride 使用 `+4` padding；
- Split-K partial 保持 INT32；
- scale 只在完整 INT32 accumulation 后执行；
- epilogue 仍为 FP32 scale 后 BF16 conversion；
- legacy raw-weight kernels 作为不能使用 packed 路径时的 fallback。

完整演进关系：

```text
旧 SDOT4 tiled GEMM
  + 保留 vector load / K32 / INT32 partial / BF16 epilogue
  + 静态 W 离线 packing
  + SDOT4 -> Wave64 MMAC
  + B LDS transpose -> packed B direct VGPR load
  + per-K32 A staging -> whole-split A slice
  + separate reduce launch -> last-arriving-CTA fused reduce
  + hidden workspace -> caller-owned persistent workspace
= packed Marlin-fused kernel
```

## 12. 对 AI 最重要的实现约束

1. 不要把 raw `[K,N]` 指针传给 packed consumer。
2. 修改 pack layout 时必须同步修改 consumer offset 和 lane mapping。
3. `v_mmac` 的 A/B/C lane mapping必须先数学推导再编码。
4. whole-A-slice 每 split 不得超过 16 个 K32 tile。
5. 所有 CTA threads 必须以一致控制流到达 `__syncthreads()`。
6. Split-K partial 发布必须先于 ticket atomic，并有 device-scope fence。
7. `tile_done` 首次使用前清零；调用结束必须安全复位。
8. 同一 workspace 不得被并发 stream 无保护复用。
9. 增加 accumulator/prefetch 会提高 VGPR，必须检查 occupancy。
10. 新 shape 必须扫描 split_k，而不是沿用旧 SDOT4 参数。

## 13. 验证与性能分析

构建：

```bash
bash build.sh
```

正确性优先跑：

```bash
python3 smoke_test_v2.py
python3 smoke_test_tp8_v2.py
```

reference 应采用 CPU INT32 GEMM，再做完全相同的 FP32 scale/BF16 conversion，要求
逐元素 `mismatch == 0`。应覆盖 TP4/TP8、M=1/2/3/4/5/8/16/32 和尾部 shape。

性能最终使用当前任务冻结的 hipprof protocol：

- 10 次 warmup dispatch group；
- 统计后续 100 次 operator call，trace 共 110 次；
- 每次调用把该 GEMM 的全部 GPU dispatch `DurationNs` 求和，再对最终 100 次取算术平均；
- 用 realtime host interval 和稳定 dispatch pattern 排除 preparation、packing、JIT 与无关 kernel；
- PMC/read/write replay 单独采集 HBM、L2、VGPR/AGPR/SGPR、LDS/scratch 等诊断，
  replay duration 不进入 latency。

阶段归因可使用 `profile_phase_breakdown.py`：

```text
main-only             = MMAC + partial store
publish mode          = main + fence/ticket
reduce-only           = partial reduce + scale + BF16
fused                 = 完整路径
```

不要只优化 main MMAC。如果 publish/reducer 已占主要比例，应降低 split_k、改变 partial
layout 或 reducer mapping，而不是继续展开 MMAC。

## 14. 可直接交给 AI 的精简提示

```text
旧基线 myGEMM_kernel.hip 已有 128-bit load、SDOT4、A/B LDS tiling 和两阶段
Split-K。新方案的结构性优化是：静态 W 在模型加载期 pack 成
[N64][K32][N16-group][lane64][8B]；forward 中 B 直接 global->VGPR；使用 Wave64
v_mmac_i32_16x16x32_i8，BM16/BN64/BK32、4 waves/CTA；每个 split 最多 16 个
K32 tile，一次性 staging 完整 A slice 到 LDS [tiles][16][36]，主计算只做一次 CTA
barrier；预取下一 B fragment，偶/奇 tile 使用两条 accumulator chain。

Split-K 写 INT32 partial。有效 lanes threadfence 后，tid0 atomicAdd per-output-tile
ticket，最后 CTA 在同 kernel 中归约、scale、写 BF16并清 ticket，从而替代旧版第二个
reduction launch。workspace 和 packed W 均由调用方初始化时持久分配，forward 不做
hipMalloc/host mutex。split_k 必须按 MMAC 实测，不能照搬旧 SDOT4 的 16/32。

任何修改都必须保持 pack/consumer lane mapping、INT32 精确累加和 BF16 epilogue；
用 CPU INT32 reference 覆盖 TP4/TP8 与边界 M，并用 hipprof 检查 timing、HBM、L2、
VGPR 和 LDS。
```
