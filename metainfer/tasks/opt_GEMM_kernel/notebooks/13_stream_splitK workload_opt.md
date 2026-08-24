# Split-K 按 Stream 隔离 Workspace：实现说明

> **历史适用范围与验证状态**：本文记录的是 `001_swizzle.cpp` 的
> per-stream 内部 workspace/fused Split-K 方案，不是 `/home/FF/workspace/003`
> 最终采用的 caller-owned persistent workspace ABI，也不是当前任务必须遵循的
> dispatch recipe。文末列出的 gfx928 编译、双 stream/多 device 正确性压力测试和
> 新旧交替 hipprof benchmark 尚未完成。Agent 只能把这里的并发风险、ticket
> 不变量和候选实现当作证据；必须先检查当前 submission、shape、真实 hipprof
> operator time/dispatch breakdown 和 PMC，再决定是否移植、修改或完全舍弃此方案。

本文档记录 `001_swizzle.cpp` 中新增的 Split-K host 侧改造，供后续
agent 继续实现、移植和验证。

对应源码：

- `001_swizzle.cpp`
- fused Split-K kernel：`small_m_splitk_dot4_fused_kernel`
- host 模板启动器：`launch_splitk_fused_instance`

## 1. 改造目标

原实现只有一组进程级静态设备指针：

```cpp
static int32_t* g_workspace;
static uint32_t* g_tile_done;
```

所有 Split-K 调用都复用这两个地址。不同 HIP stream 上的 kernel 可以
并发执行，因此两个请求可能同时覆盖相同的 partial，并把各自的
`atomicAdd` ticket 混在一起。

可能结果：

- Reduce 读到两个请求混合的 partial；
- 某个 CTA 被错误地判定为最后一个 split；
- counter 提前清零；
- 后续调用继承非零 counter；
- 输出发生偶发、非确定性错误。

本次改造的目标是：

1. 每个 `(HIP device, hipStream_t)` 使用独立的 partial 和 tile counter；
2. 同一 stream 上的 host 提交线程安全；
3. 不同 stream 的 GPU kernel 仍能并发；
4. 稳态调用不执行 stream/device synchronize；
5. 不改变 BM、BN、BK、split 数和 device 计算逻辑。

## 2. 内存所有权

算子只为 Split-K 中间结果分配设备内存：

```text
partial:
    split_k * M * N * sizeof(int32_t)

tile_done:
    ceil(M / BM) * ceil(N / BN) * sizeof(uint32_t)
```

以下内存由调用方管理，算子不为其执行 `hipMalloc`：

```text
x_q             A 矩阵
weight_kn       B 矩阵
x_scale
weight_scale
output_bf16
```

kernel 内的 `a_tile`、`b_tile` 是每个 CTA 自动分配的 LDS；accumulator
属于寄存器，也不在 host workspace 中。

由于 kernel launch 是异步的，调用方必须保证 A、B、scale 和 output
在对应 stream 完成之前保持有效。

## 3. Per-stream 状态

每个 stream 对应一个 `StreamWorkspace`：

```cpp
struct StreamWorkspace {
    int32_t* partial = nullptr;
    size_t partial_capacity = 0;

    uint32_t* tile_done = nullptr;
    size_t tile_done_capacity = 0;

    std::mutex launch_mutex;
};
```

含义：

- `partial`：该 stream 的 Split-K INT32 partial；
- `tile_done`：该 stream 每个输出 tile 的完成计数；
- `partial_capacity`：按字节记录；
- `tile_done_capacity`：按 counter 元素数记录；
- `launch_mutex`：保护同一 stream 的扩容与 kernel 提交顺序。

进程级容器使用 `(device_id, stream_handle)` 找到状态。最新版不能只用
stream 数值作为 key，因为默认 stream 在不同 device 上通常都表现为
空句柄，同一个进程管理多个 GPU 时会发生冲突。

key 和 hash 的源码如下：

```cpp
struct DeviceStreamKey {
    int device;
    uintptr_t stream;

    bool operator==(const DeviceStreamKey& other) const noexcept {
        return device == other.device && stream == other.stream;
    }
};

struct DeviceStreamKeyHash {
    size_t operator()(const DeviceStreamKey& key) const noexcept {
        const size_t device_hash = std::hash<int>{}(key.device);
        const size_t stream_hash =
            std::hash<uintptr_t>{}(key.stream);
        return device_hash ^
               (stream_hash + size_t{0x9e3779b9} +
                (device_hash << 6) + (device_hash >> 2));
    }
};

static std::mutex g_stream_workspaces_mutex;

static std::unordered_map<
    DeviceStreamKey,
    std::unique_ptr<StreamWorkspace>,
    DeviceStreamKeyHash
> g_stream_workspaces;
```

全局 map mutex 只在查找或首次创建状态时短暂持有，不会持有到 GPU
kernel 完成。

## 4. 获取 StreamWorkspace

获取 workspace 前先查询当前 HIP device，然后和原始 stream 数值共同
组成 key：

```cpp
__host__ static StreamWorkspace*
get_stream_workspace(hipStream_t stream) {
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess)
        return nullptr;

    const DeviceStreamKey key{
        device,
        reinterpret_cast<uintptr_t>(stream)
    };

    std::lock_guard<std::mutex> lock(
        g_stream_workspaces_mutex);

    auto it = g_stream_workspaces.find(key);
    if (it != g_stream_workspaces.end())
        return it->second.get();

    auto workspace = std::make_unique<StreamWorkspace>();
    StreamWorkspace* result = workspace.get();
    g_stream_workspaces.emplace(
        key, std::move(workspace));
    return result;
}
```

获取流程：

1. 调用 `hipGetDevice` 获取当前 device；
2. 构造 `(device_id, stream_handle)`；
3. 锁住 `g_stream_workspaces_mutex`；
4. 查找 workspace；
5. 不存在时创建 `StreamWorkspace`；
6. 返回稳定的 `StreamWorkspace*`；
7. 释放全局 map mutex。

如果 `hipGetDevice` 失败，`get_stream_workspace` 返回空指针。模板启动器
必须在解引用前检查：

```cpp
StreamWorkspace* workspace =
    get_stream_workspace(stream);

if (!workspace)
    return static_cast<int>(hipErrorInvalidDevice);
```

map 的 value 使用 `std::unique_ptr`，因此 unordered_map rehash 后，
`StreamWorkspace` 本体地址仍保持稳定。

## 5. Workspace 扩容

`ensure_stream_workspace` 同时检查 partial 和 counter：

```cpp
grow_partial =
    required_partial_bytes > partial_capacity;

grow_tile_done =
    required_tile_count > tile_done_capacity;
```

容量足够时直接返回，不执行：

- `hipStreamSynchronize`；
- `hipMalloc`；
- `hipFree`；
- `hipMemsetAsync`。

需要扩容时，旧指针可能仍被该 stream 中较早提交的 kernel 使用。
因此替换旧内存前执行：

```cpp
hipStreamSynchronize(stream);
```

这里只同步当前 stream，不调用 `hipDeviceSynchronize`。

扩容后：

- partial 不需要初始化，因为每个有效输出元素都会被当前 split CTA
  覆盖；
- 新 counter 必须在同一 stream 中清零：

```cpp
hipMemsetAsync(
    workspace.tile_done,
    0,
    tile_count * sizeof(uint32_t),
    stream);
```

同一 stream 的后续 kernel launch 排在 memset 后面，因此首次使用时
counter 一定为零。

注意：传统 `hipMalloc/hipFree` 本身可能包含运行时级同步成本。该成本
只应出现在首次分配或容量增长阶段，不能出现在稳态热路径。

## 6. Host 模板启动器

重复的 grid、counter、workspace 和 kernel launch 逻辑被封装为：

```cpp
template <int BM, int BN, int BK>
__host__ __forceinline__ int
launch_splitk_fused_instance(...);
```

模板参数用于编译期确定：

- workgroup 大小；
- LDS 数组尺寸；
- load 循环边界；
- SDOT4 循环边界；
- `__launch_bounds__`；
- 具体 device kernel 符号。

启动流程必须保持以下顺序：

```text
计算 m_tiles / n_tiles / tile_count / partial_bytes
                ↓
根据 hipStream_t 获取 StreamWorkspace
                ↓
锁住 workspace.launch_mutex
                ↓
ensure_stream_workspace
                ↓
在同一 stream 启动 fused Split-K kernel
                ↓
hipGetLastError
                ↓
释放 workspace.launch_mutex
```

`launch_mutex` 只保护 host 侧的资源变更和 enqueue，不等待 kernel
执行结束。

同一 stream 的两个 kernel 依靠 HIP stream FIFO 自动串行；不同
stream 使用不同设备指针，可以在 GPU 上重叠执行。

## 7. 编译期 Launch Bounds

Split-K kernel 从固定：

```cpp
__launch_bounds__(512)
```

改为：

```cpp
template <int BM, int BN, int BK>
__global__ __launch_bounds__(BM * BN)
void small_m_splitk_dot4_fused_kernel(...);
```

当前实例：

```text
BM=1, BN=64  -> launch_bounds(64)
BM=2, BN=64  -> launch_bounds(128)
BM=4, BN=64  -> launch_bounds(256)
BM=8, BN=64  -> launch_bounds(512)
```

host 侧仍需根据运行时 M/N/K 选择模板实例；这些 `if/else` 不会进入
GPU kernel，也不会造成 wave divergence。

## 8. Fused Ticket 协议必须保持

每个 split CTA 写完 partial 后：

```cpp
__syncthreads();

if (tid == 0) {
    __threadfence();
    ticket = atomicAdd(&tile_done[tile_id], 1u);
}
```

顺序不能随意交换：

1. 所有线程先写完当前 CTA 的 partial；
2. CTA barrier 确认 block 内写入已经发出；
3. device fence 发布 global partial；
4. 最后增加完成计数；
5. 获得 `split_k - 1` 票号的 CTA 执行 Reduce。

`atomicAdd` 只负责同一次 GEMM 的 splits。Per-stream workspace 的作用
是防止不同 GEMM 调用共享同一个 counter 和 partial。

`is_last_split` 由 thread 0 写入 shared memory，并在 barrier 后供整个
CTA 读取，因此它是 block-uniform 条件。非最后 CTA 可以在 ticket 后
直接结束；只有最后 CTA 执行 Reduce：

```cpp
if (is_last_split) {
    if (row < M && col < N) {
        // Reduce partial、scale、写 BF16 output。
    }

    __syncthreads();

    if (tid == 0)
        tile_done[tile_id] = 0u;
}
```

尾部不再需要第二次 `__threadfence()` 和 `atomicExch()`，理由是：

1. 其他 stream 使用独立 counter；
2. 同一 stream 的下一次 kernel 必须等当前 kernel 完成；
3. 最后 ticket 出现时，其他 split CTA 已经完成对 counter 的最后一次访问；
4. last CTA 内的 barrier 保证所有输出线程先完成 store 指令，再由 thread 0
   清零 counter。

不能据此删除 ticket 前的第一次 `__threadfence()`。第一次 fence 负责在
发布完成计数前发布 global partial，是跨 CTA Reduce 正确性的必要条件。

## 9. 当前 Dispatch

当前 split 数不变，但加入了 notebook 12 中已有交替 benchmark 证据的
BM=2/BM=4 精确实例。

当前特殊实例：

```cpp
if (M == 1 && N == 8192 && K == 1024) {
    return launch_splitk_fused_instance<1, 64, 32>(...);
}

const bool certified_small_bm_shape =
    (K == 1024 && (N == 8192 || N == 4096)) ||
    (K == 4096 && N == 1024);

if (M == 2 && certified_small_bm_shape) {
    return launch_splitk_fused_instance<2, 64, 32>(...);
}

if (M == 4 && certified_small_bm_shape) {
    return launch_splitk_fused_instance<4, 64, 32>(...);
}

return launch_splitk_fused_instance<8, 64, 32>(...);
```

不要仅根据 `M==2/4` 泛化到所有 shape。BM 同时影响计算 wave 和
global-to-LDS 搬运并行度；当前只固化：

```text
M=2/4, K=1024, N=8192
M=2/4, K=1024, N=4096
M=2/4, K=4096, N=1024
```

后续仍可实验：

```text
BK=64
BK=128
```

新增实例时应继续通过 `launch_splitk_fused_instance<BM, BN, BK>`
启动，不要重新复制 workspace 管理代码。

## 10. 并发语义

### 安全

- 同一显式 stream 连续调用；
- 多个显式 stream 并发调用；
- 多个 CPU 线程向同一显式 stream 提交；
- 同一进程内不同 HIP device 的显式或默认 stream；
- 不同 shape 在容量足够的 workspace 上复用。

### 需要注意

1. **默认 stream**

   当前 key 已包含 device ID，可以区分不同 GPU 上的空 stream 句柄。
   但如果传入 `nullptr` 且运行环境使用 per-thread default stream
   语义，同一 device 的不同 host 线程仍可能具有相同空句柄、却对应
   不同执行序列。此时需要把 host thread 信息加入默认-stream key，
   或要求调用方传入显式 stream。

2. **Stream 生命周期**

   当前 map 不知道外部 stream 何时销毁，因此不会主动释放对应设备
   workspace。适合 SGLang/PyTorch 中固定、长期存在的 stream。
   若频繁创建和销毁临时 stream，应增加显式释放接口，或改成由 HIP
   event 管理的有界 workspace slot pool。

3. **首次分配和扩容**

   首次调用包含 `hipMalloc` 和 counter memset；扩容会同步对应 stream。
   benchmark 应区分 cold-start 与 steady-state。

## 11. Agent 后续实现建议

如果需要生产级完善，优先级如下：

1. 明确处理 `nullptr` / per-thread default stream；
2. 为临时 stream 增加 workspace 回收机制；
3. 已知 shape 下可为每个 stream 预分配最大 workspace，避免热路径扩容；
4. 若 runtime 支持且验证稳定，可评估 `hipMallocAsync/hipFreeAsync`；
5. 保留 `(device_id, stream_handle)` 复合 key，不要退回单 stream key；
6. 不要用全局 `hipDeviceSynchronize` 代替 workspace 隔离；
7. 不要仅用 host mutex 保护单一 workspace——host mutex 在 launch 返回后
   会释放，而不同 stream 的异步 kernel 仍可能重叠。

## 12. 验证清单

### 编译

- 使用目标 HIP 编译器；
- 目标架构为 gfx928；
- 检查 BM1/BN64 实例的 workgroup 上限为 64；
- 检查 BM8/BN64 实例的 workgroup 上限为 512。

### 单 stream

- 连续运行不同 Split-K shape；
- 先小 shape、后大 shape，覆盖扩容路径；
- 先大 shape、后小 shape，覆盖容量复用；
- 与 CPU 或可信 GEMM reference 比较。

### 双 stream 正确性压力测试

建议两个 stream 使用不同输入模式，避免错误相互抵消：

```text
stream A:
    A/weight 填充模式 A
    output A

stream B:
    A/weight 填充模式 B
    output B
```

循环交错提交：

```text
launch(A, streamA)
launch(B, streamB)
launch(A, streamA)
launch(B, streamB)
...
```

最后分别同步两个 stream，并验证所有输出。

至少覆盖：

- 相同 shape、不同数据；
- 不同 shape；
- BM1 与 BM8 同时运行；
- 两个 stream 同时首次分配；
- 一个 stream 扩容时另一个 stream 正在计算。

### 多设备

- 在两个 HIP device 上分别使用显式 stream；
- 在两个 device 上分别使用默认 stream；
- 确认相同的空 stream 句柄映射到不同 `DeviceStreamKey`；
- 交错提交 Split-K，并分别与 reference 比较；
- 检查 workspace 的设备归属和 kernel 当前 device 一致。

### 性能

分别记录：

- 首次调用；
- workspace 容量稳定后的调用；
- 单 stream；
- 双 stream 并发吞吐；
- `hipStreamSynchronize` 是否只在扩容时出现。

## 13. 当前验证状态

已完成：

- 旧单例 `g_workspace/g_tile_done` 引用清理；
- 每-stream 指针接入 fused kernel launch；
- workspace key 扩展为 `(device_id, stream_handle)`；
- `hipGetDevice` 失败路径检查；
- host map mutex 与每-stream launch mutex 分层；
- 扩容前同-stream 同步；
- notebook 12 已认证 shape 的 BM=2/BM=4 精确 dispatch；
- 非最后 CTA 跳过 fused 尾部 barrier；
- 最后 CTA 使用普通 store 复位专属 counter，删除第二次 fence/atomic；
- 源码格式和引用静态检查。

尚未完成：

- 当前环境没有 `hipcc`，未执行 gfx928 编译；
- 未执行实机双 stream 正确性压力测试；
- 未对 BM2/BM4 和 fused 尾部精简执行新旧交替 benchmark；
- 未验证默认 stream/per-thread default stream；
- 未执行实机多 device 并发压力测试；
- 未实现 stream 销毁后的 workspace 回收。
