# Hygon Z200SM_80 DCU 底层架构与 HIP C++ 开发知识库

> 本知识库用于指导海光 Hygon Z200SM_80 DCU 的 HIP C++ Kernel 开发与性能调优。  
> 文中明确区分：
>
> 1. 本机 `rocminfo` 实测属性；
> 2. `gfx906` 编译目标所能确认的 ISA 约束；
> 3. 需要通过海光专有文档或实际反汇编进一步确认的微架构行为。

---

## 1. 硬件规格与系统拓扑

### 1.1 本机系统拓扑

CPU 数量、DCU 数量以及 PCIe/NUMA 连接关系属于服务器级配置，不属于 Z200SM_80 芯片本身的固定属性。

在系统中需要区分以下编号：

- HIP Device Ordinal
- HSA Agent ID
- HSA Node ID
- GPU UUID
- PCIe BDF
- Linux NUMA Node

这些编号不能直接互相替代。

例如：

- `rocminfo` 中的 `Node: 5` 是 HSA Node 编号；
- 它不一定等于 Linux NUMA Node 5；
- 它也不等于 HIP Device ID 5；
- 物理拓扑应结合 PCIe BDF 与 `/sys` 中的 NUMA 信息确认。

建议执行：

```bash
lscpu -e=CPU,NODE,SOCKET,CORE
numactl --hardware
rocminfo
rocm-smi --showbus
rocm-smi --showtoponuma
```

若 `rocm-smi` 不支持相关选项，可以通过 PCIe BDF 查询：

```bash
cat /sys/bus/pci/devices/0000:BB:DD.F/numa_node
```

多 GPU Host 代码应尽量满足：

- 调用 `hipSetDevice(device)` 选择目标设备；
- Host 提交线程绑定到目标 DCU 的近端 CPU NUMA 节点；
- 页锁定 Host 内存尽量在近端 NUMA 节点完成 first-touch；
- H2D/D2H 提交线程与目标 DCU 保持 NUMA 亲和性；
- 避免无必要的跨 Socket PCIe 数据传输。

---

### 1.2 Z200SM_80 实测属性

根据本机 `rocminfo` 输出，Z200SM_80 的主要属性如下。

| 项目 | 实测值 |
|---|---:|
| Agent Name | `ZIFANG` |
| Marketing Name | `Z200SM_80` |
| Vendor | `HYGON` |
| Device Type | `DCU` |
| Target ID | `gfx906:sramecc+:xnack-` |
| Compute Units | 64 |
| SIMDs per CU | 4 |
| Shader Engines | 4 |
| Wavefront Size | 64 |
| Workgroup Max Size | 1024 threads |
| Max Waves per CU | 40 |
| Max Work-items per CU | 2560 |
| L1 Cache | 16 KB |
| L2 Cache | 8192 KB |
| GROUP/LDS Pool | 64 KB |
| Global Memory | 16,760,832 KB，约 15.98 GiB |
| Cache Line | 64 B |
| Max Reported Clock | 1319 MHz |
| Fast FP16 | TRUE |
| Max FBarriers per Workgroup | 32 |

其中：

- `GROUP` Pool 对应 Workgroup 共享的 LDS 空间；
- 单个 CU 的 LDS 上限为 64 KB；
- `Fast F16 Operation: TRUE` 表示设备具有快速 FP16 运算能力；
- 该字段本身不能证明设备支持 MFMA、WMMA 或类似 Tensor Core 的矩阵指令。

---

### 1.3 Wave64 执行模型

Z200SM_80 使用 Wave64：

```text
1 wavefront = 64 work-items
```

常见 Workgroup 大小对应的 Wave 数量如下。

| Workgroup Threads | Waves per Workgroup |
|---:|---:|
| 64 | 1 |
| 128 | 2 |
| 256 | 4 |
| 512 | 8 |
| 1024 | 16 |

因此，从 NVIDIA Warp32 Kernel 迁移时，以下逻辑必须重新检查：

- Warp/Wave 内规约；
- Shuffle offset；
- Lane ID；
- 每 Warp 持有的数据 Fragment；
- 每 Block 中 Warp/Wave 的数量；
- Warp-specialization；
- Softmax、RMSNorm、LayerNorm 的规约树；
- GEMM 的线程到数据映射。

不能将写死的 `32` 直接沿用到 Wave64。

推荐在 HIP Kernel 中使用：

```cpp
const int lane_id = threadIdx.x % warpSize;
const int wave_id = threadIdx.x / warpSize;
```

在当前设备上：

```cpp
warpSize == 64
```

---

### 1.4 Occupancy 基本约束

单 CU 的已知上限：

```text
Max Waves per CU     = 40
Max Work-items per CU = 2560
LDS per CU           = 64 KB
```

只按线程数计算，不考虑其他资源时：

| Workgroup Size | Waves/Block | 线程数约束下的最大 Blocks/CU | 对应 Waves/CU |
|---:|---:|---:|---:|
| 128 | 2 | 20 | 40 |
| 256 | 4 | 10 | 40 |
| 512 | 8 | 5 | 40 |
| 1024 | 16 | 2 | 32 |

实际驻留量还会受到以下因素限制：

- VGPR 使用量；
- SGPR 使用量；
- LDS 使用量；
- Scratch Spill；
- 最大 Resident Workgroup 数量；
- 编译器生成的隐式资源开销。

例如：

- 每 Block 使用 64 KB LDS，通常只能驻留 1 Block/CU；
- 每 Block 使用 32 KB LDS，从 LDS 角度最多允许 2 Blocks/CU；
- 1024-thread Block 即使 LDS 使用很少，也最多只有 2 Blocks/CU，即 32 Waves/CU；
- VGPR 过多时，实际驻留 Wave 数可能远低于理论上限。

因此，优化时不能只观察 Block Size，还必须同时检查寄存器和 LDS 使用量。

---

## 2. 编译与构建环境

> **任务级构建优先级：** 本节的 `hipcc`、CMake 和反汇编命令用于说明 system-owned
> `build.sh`、`CMakeLists.txt` 或人工平台诊断应采用的参数。gen-cpp Implementer 只能编辑
> `CMakeLists.txt`/源码并执行 `bash build.sh`，不得直接运行 `cmake`、`hipcc`、`make` 或
> `ninja`；直接执行会触发 `bypass-system-build-sh` 策略。

### 2.1 hipcc 编译

system-owned build path 内部对应的命令形态：

```bash
hipcc -O3 \
      --offload-arch=gfx906 \
      kernel.cpp \
      -o exec
```

`gfx906` 是当前 DTK/LLVM 工具链面向该设备报告的基础 ISA 目标。

不要在没有验证的情况下将目标替换为：

```text
gfx908
gfx90a
gfx940
gfx941
gfx942
```

否则可能产生设备不支持的 ISA 指令。

---

### 2.2 完整 Target ID

本机 `rocminfo` 报告：

```text
amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-
```

其含义包括：

- 基础架构：`gfx906`
- SRAM ECC：开启
- XNACK：关闭

通常在 CMake 和普通 `hipcc` 编译命令中使用：

```text
gfx906
```

即可。

只有在满足以下条件时，才考虑显式携带 feature：

- 当前 DTK 编译器接受完整 Target ID；
- 生成的二进制只部署在同类设备；
- 已经验证 feature 不会造成兼容性问题。

---

### 2.3 CMake 配置

```cmake
cmake_minimum_required(VERSION 3.21)

project(z200_kernel LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

set(CMAKE_HIP_ARCHITECTURES gfx906)

add_executable(exec kernel.cpp)

target_compile_options(exec PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)
```

若存在多个源文件：

```cmake
add_executable(exec
    main.cpp
    kernel.cpp
    operator.cpp
)
```

必要时可增加调试和保存中间文件选项：

```bash
hipcc -O3 \
      -gline-tables-only \
      --offload-arch=gfx906 \
      -save-temps \
      kernel.cpp \
      -o exec
```

---

## 3. Global Memory 与向量化访存

### 3.1 Global → VGPR → LDS

Z200SM_80 不具备 NVIDIA Hopper TMA，也不能直接套用 CUDA `cp.async` 的硬件流水模型。

常见数据路径为：

```text
Global Memory
    ↓
VGPR
    ↓
LDS
```

Global Memory 数据首先被加载到每个 Lane 的 VGPR，然后由线程显式写入 LDS。

---

### 3.2 128-bit 向量加载

可使用 HIP 内置向量类型，例如：

```cpp
float4
int4
uint4
```

示例：

```cpp
__global__ void vector_copy(
    const float* __restrict__ input,
    float* __restrict__ output,
    int vector_count)
{
    const auto* input4 =
        reinterpret_cast<const float4*>(input);

    auto* output4 =
        reinterpret_cast<float4*>(output);

    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vector_count) {
        const float4 value = input4[idx];
        output4[idx] = value;
    }
}
```

需要保证：

- `input` 地址满足 16 字节对齐；
- `output` 地址满足 16 字节对齐；
- 起始 offset 不破坏对齐；
- 长度尾部单独处理；
- 指针别名关系尽量通过 `__restrict__` 明确；
- Wave64 中各 Lane 的地址尽量连续。

运行时可检查：

```cpp
assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
assert(reinterpret_cast<uintptr_t>(output) % 16 == 0);
```

---

### 3.3 `float4` 不保证一定生成单条 128-bit Load

以下代码：

```cpp
float4 value = ptr4[index];
```

有机会生成：

```text
global_load_dwordx4
flat_load_dwordx4
buffer_load_dwordx4
```

但也可能因为以下原因被拆分：

- 地址对齐无法证明；
- Pointer aliasing；
- 条件分支；
- 非连续 offset；
- Tail 处理；
- 寄存器压力；
- 编译器优化决策。

不能仅根据 C++ 类型判断最终 ISA。

需要人工平台诊断时可查看反汇编；Implementer 任务内不要直接运行下面的编译命令：

```bash
hipcc -O3 \
      --offload-arch=gfx906 \
      -save-temps \
      kernel.cpp \
      -o exec
```

然后执行：

```bash
llvm-objdump -d exec | \
    grep -E 'global_load|flat_load|buffer_load'
```

---

### 3.4 Wave64 下的访存规模

若每个 Lane 加载一个 `float4`：

```text
float4 = 16 bytes
Wave64 = 64 lanes
```

一个 Wave 的逻辑请求量为：

```text
64 × 16 B = 1024 B
```

是否能高效利用显存带宽，仍然取决于：

- Lane 地址是否连续；
- 是否对齐；
- 是否跨越过多 Cache Line；
- 是否存在严重的非合并访问；
- 是否存在随机 Gather；
- L1/L2 命中率；
- 实际生成的 Load 指令宽度。

---

## 4. LDS 与同步机制

### 4.1 LDS 的作用

LDS 是 Workgroup 内共享的低延迟存储空间，类似 CUDA Shared Memory。

主要用途包括：

- GEMM Tile；
- FlashAttention 的 Q/K/V Tile；
- Block Reduction；
- 多 Wave 结果交换；
- 数据重排；
- 合并非连续 Global Memory 访问。

本机每 CU 的 GROUP/LDS 上限为：

```text
64 KB
```

设计 Tile 时必须同时考虑：

- LDS 容量；
- Bank Conflict；
- Block 驻留数量；
- 双缓冲或多缓冲额外占用；
- Padding；
- 不同数据类型的字节数。

---

### 4.2 `s_waitcnt` 不是 Workgroup Barrier

`s_waitcnt` 是 Wavefront 局部的未完成访存操作等待指令。

常见计数器包括：

- `vmcnt`：Vector Memory 操作；
- `lgkmcnt`：LDS、GDS、部分 Scalar/Constant 相关操作。

示例：

```cpp
asm volatile(
    "s_waitcnt vmcnt(0)"
    :
    :
    : "memory"
);
```

表示当前 Wave 等待相应 VMEM 操作完成。

但它不能：

- 等待其他 Wave 到达相同位置；
- 保证其他 Wave 已经写完 LDS；
- 提供 Workgroup 范围的 rendezvous；
- 替代 `__syncthreads()`；
- 替代 Hopper `mbarrier`。

---

### 4.3 跨 Wave 共享 LDS 必须使用 Workgroup Barrier

当一个 Wave 写 LDS，另一个 Wave 随后读取时，需要：

```cpp
__syncthreads();
```

示例：

```cpp
shared_data[threadIdx.x] = value;

__syncthreads();

float other = shared_data[other_index];
```

仅使用：

```cpp
asm volatile("s_waitcnt lgkmcnt(0)");
```

不能保证其他 Wave 已经完成 LDS 写入。

---

### 4.4 普通 HIP C++ 中不要过度手写 `s_waitcnt`

编译器会根据数据依赖自动插入必要的 wait 指令。

例如：

```cpp
float value = input[idx];
shared[idx] = value;
```

编译器知道 LDS Store 依赖 Global Load 的结果，因此通常会插入必要等待。

手动无条件写：

```asm
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
```

会等待所有相关未完成请求清零，可能导致：

- Global Memory 并发度下降；
- Software Pipeline 被串行化；
- 无法重叠访存和计算；
- Kernel 延迟增加。

只有在以下流程后才建议手工插入：

1. 检查编译器生成 ISA；
2. 分析 waitcnt 位置；
3. 确认存在过早或过晚等待；
4. 使用 benchmark 验证；
5. 确保不同编译器版本下仍然正确。

---

## 5. 软件流水与 Double Buffering

### 5.1 基本思路

可以在 LDS 中设置两块 Tile：

```text
LDS Ping Buffer
LDS Pong Buffer
```

理想的软件流水：

```text
预取 Tile n+1 到 VGPR
        ↓
计算 Tile n
        ↓
等待 Tile n+1 的 Global Load 完成
        ↓
将 Tile n+1 写入备用 LDS
        ↓
Workgroup Barrier
        ↓
交换 Ping/Pong
```

---

### 5.2 与 Hopper TMA Pipeline 的差异

Hopper TMA 可以：

- 由专用硬件执行多维 Global→Shared 搬运；
- 配合 `mbarrier` 管理异步事务；
- 减少参与搬运的线程数；
- 降低寄存器中转开销。

Z200SM_80 上通常需要：

- 每个 Lane 执行 Global Load；
- 数据先进入 VGPR；
- 再由 Lane 写入 LDS；
- 使用普通 Workgroup Barrier；
- 由编译器和 `s_waitcnt` 管理访存依赖。

因此软件流水的主要难点包括：

- 预取距离；
- VGPR 增长；
- LDS 双缓冲容量；
- waitcnt 位置；
- Barrier 开销；
- 计算与搬运任务划分；
- Wave64 下的工作分配。

---

### 5.3 Double Buffering 伪代码

```cpp
extern __shared__ half lds[];

half* tile0 = lds;
half* tile1 = lds + TILE_ELEMENTS;

int current = 0;

// 首个 Tile
load_global_to_lds(tile0, global_ptr, 0);

__syncthreads();

for (int tile = 0; tile < num_tiles; ++tile) {
    half* current_tile = current == 0 ? tile0 : tile1;
    half* next_tile = current == 0 ? tile1 : tile0;

    // 尝试预取下一个 Tile 到 VGPR
    RegisterFragment prefetched;

    if (tile + 1 < num_tiles) {
        prefetched =
            load_global_to_register(
                global_ptr,
                tile + 1);
    }

    // 计算当前 Tile
    compute_tile(current_tile);

    if (tile + 1 < num_tiles) {
        store_register_to_lds(
            next_tile,
            prefetched);
    }

    __syncthreads();

    current ^= 1;
}
```

实际实现时应避免：

- 预取 Fragment 过大导致 VGPR 激增；
- 双缓冲 LDS 占用导致 Occupancy 降低；
- Barrier 位置错误；
- 最后一个 Tile 越界；
- Tail Tile 未处理。

---

## 6. GEMM 计算核心

### 6.1 不应默认使用 MFMA

当前设备的编译目标是：

```text
gfx906
```

因此，在没有海光专有 ISA 文档或实际反汇编证据前，不应假定支持：

- MFMA；
- WMMA；
- WGMMA；
- Tensor Core；
- CDNA Matrix Core Fragment。

以下类型的代码不应直接用于该设备：

```cpp
__builtin_amdgcn_mfma_f32_16x16x16f16(...)
```

它属于后续 CDNA/MAI 指令体系的编程模型，不应依据 `Fast F16` 字段推断其可用。

---

### 6.2 `Fast F16` 的正确理解

`rocminfo` 中：

```text
Fast F16 Operation: TRUE
```

表示 FP16 运算具备快速硬件路径。

它可能支持：

- FP16 标量或向量运算；
- Packed FP16；
- FP16 FMA；
- FP16 数据搬运和转换。

但不能单独证明存在：

- 矩阵乘专用单元；
- MFMA 指令；
- Tensor Core；
- WGMMA 类 Warp Group 指令。

---

### 6.3 自定义 GEMM 推荐路线

自定义 GEMM 应围绕以下机制设计：

- Wave64 VALU；
- FP32 FMA；
- Packed FP16；
- Register Blocking；
- LDS Tiling；
- Vectorized Global Load/Store；
- 多 Wave Latency Hiding；
- Software Pipeline；
- 合理的 Block Tile；
- 避免 VGPR Spill；
- 避免 LDS Bank Conflict。

推荐性能开发顺序：

1. 使用 DTK 提供的 BLAS/hipBLAS 作为性能基线；
2. 实现朴素 GEMM；
3. 增加 LDS Tiling；
4. 增加 Register Blocking；
5. 增加向量化 Load/Store；
6. 处理 Wave64 数据映射；
7. 增加 Double Buffering；
8. 检查 VGPR、LDS 和 Occupancy；
9. 检查反汇编；
10. 使用 Profiler 验证瓶颈。

---

### 6.4 示例 Tile 资源估算

假设 FP16 GEMM 使用：

```text
A Tile = 128 × 32
B Tile = 32 × 128
```

单缓冲 LDS 占用：

```text
A = 128 × 32 × 2 B = 8192 B
B = 32 × 128 × 2 B = 8192 B
Total = 16 KB
```

双缓冲：

```text
2 × 16 KB = 32 KB
```

从 LDS 角度看，一个 CU 最多容纳：

```text
64 KB / 32 KB = 2 Blocks
```

实际还需检查：

- 每 Block 的线程数；
- VGPR；
- SGPR；
- Scratch；
- Padding；
- Epilogue 临时存储。

---

## 7. Wave 内数据交换与规约

### 7.1 HIP Shuffle

普通 HIP Kernel 优先使用：

```cpp
__shfl()
__shfl_down()
__shfl_up()
__shfl_xor()
```

Wave64 最大值规约示例：

```cpp
__device__ inline float wave_reduce_max(float value)
{
    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down(
                value,
                offset,
                warpSize));
    }

    return value;
}
```

当前设备上：

```text
warpSize = 64
```

规约 offset 顺序为：

```text
32 → 16 → 8 → 4 → 2 → 1
```

---

### 7.2 DPP 与 `ds_bpermute` 不是同一个机制

DPP 和 DS Permute 是不同机制。

#### DPP

DPP 通常对应：

```cpp
__builtin_amdgcn_update_dpp(...)
```

它通过 VALU 数据路径执行规则化 Lane Permutation。

适合：

- 固定模式广播；
- Row Shift；
- Row Mirror；
- Wave Reduction 的固定交换模式。

#### DS BPermute

```cpp
__builtin_amdgcn_ds_bpermute(index, value)
```

属于 DS/LDS Permute 通路，用于从任意 Lane 拉取 32-bit 数据。

其索引为字节地址：

```text
source_lane = index / 4
```

读取 `src_lane` 的值时：

```cpp
uint32_t byte_index = src_lane * 4;
```

因此不能把 `ds_bpermute` 直接称为 DPP。

---

### 7.3 多 Wave Block Reduction

`__shfl_*` 只能在单个 Wave 内通信。

例如 256-thread Block 包含：

```text
256 / 64 = 4 Waves
```

完整 Block Reduction 通常需要两级规约：

```text
每个 Wave 内规约
        ↓
Wave Leader 写入 LDS
        ↓
__syncthreads()
        ↓
第一个 Wave 读取各 Wave 结果
        ↓
第一个 Wave 完成最终规约
```

示例：

```cpp
__shared__ float wave_results[16];

float value = local_value;

value = wave_reduce_max(value);

const int lane_id = threadIdx.x % warpSize;
const int wave_id = threadIdx.x / warpSize;

if (lane_id == 0) {
    wave_results[wave_id] = value;
}

__syncthreads();

if (wave_id == 0) {
    float block_value =
        lane_id < num_waves
            ? wave_results[lane_id]
            : -INFINITY;

    block_value =
        wave_reduce_max(block_value);

    if (lane_id == 0) {
        wave_results[0] = block_value;
    }
}

__syncthreads();

float final_value = wave_results[0];
```

该模式适用于：

- Softmax；
- Log-Sum-Exp；
- RMSNorm；
- LayerNorm；
- Attention Score Reduction；
- Block Sum/Max。

---

## 8. FlashAttention 迁移注意事项

### 8.1 不能直接照搬 NVIDIA Kernel

从 NVIDIA Warp32 Kernel 迁移时，需要重新设计：

- 一个 Wave 负责多少行 Q；
- 一个 Wave 中 64 Lane 的职责；
- Q/K/V Tile 大小；
- Softmax Reduction；
- LDS 容量；
- LDS Bank Conflict；
- Global Load 合并；
- Register Fragment；
- 双缓冲；
- Block 中 Wave 数；
- Barrier 次数。

---

### 8.2 LDS 容量限制

Z200SM_80 每 CU 的 LDS 为 64 KB。

若同时保存：

- Q Tile；
- K Tile；
- V Tile；
- Softmax 临时量；
- 双缓冲 K/V；

需要精确计算 LDS。

例如 FP16：

```text
K Tile: Bc × D × 2 bytes
V Tile: Bc × D × 2 bytes
```

若 `Bc=128`、`D=64`：

```text
K = 128 × 64 × 2 = 16 KB
V = 128 × 64 × 2 = 16 KB
K + V = 32 KB
```

双缓冲 K/V：

```text
2 × 32 KB = 64 KB
```

此时几乎没有空间存放 Q 或其他临时数据，并且每 CU 只能驻留一个使用完整 64 KB LDS 的 Block。

因此实际设计可能需要：

- 缩小 `Bc`；
- 缩小 Head Dimension Tile；
- 只缓存 K 或 V 的一部分；
- Q 保存在 VGPR；
- 放弃完整双缓冲；
- 使用更细粒度的流水。

---

### 8.3 Softmax Reduction

Wave64 下 Online Softmax 常见步骤：

1. 每 Lane 计算若干 Score；
2. Wave 内求局部最大值；
3. 多 Wave 时通过 LDS 合并；
4. 更新 Online Max；
5. 计算指数；
6. Wave 内求和；
7. 多 Wave 时通过 LDS 合并；
8. 更新归一化因子；
9. 对 V 进行加权累加。

需注意：

- `__shfl_down` 的起始 offset 是 32；
- 一个 Wave64 的规约树比 Warp32 多一级；
- 多 Wave Block 仍需要 LDS 和 `__syncthreads()`；
- 不应使用 `s_waitcnt` 替代 Workgroup 同步。

---

## 9. 性能监控与 Profiling

### 9.1 基础监控

常用工具：

```bash
hy-smi
rocm-smi
rocminfo
```

先查看本机支持的参数：

```bash
hy-smi --help
rocm-smi --help
```

不同 DTK 版本可能存在：

- 参数名称不同；
- 部分功能不可用；
- 权限要求不同；
- 海光定制扩展。

---

### 9.2 性能等级与频率

可以尝试：

```bash
sudo rocm-smi --setperflevel high
```

然后检查：

```bash
rocm-smi --showclocks
rocm-smi --showperflevel
```

需要注意：

```text
high ≠ 严格锁定到一个固定频率
```

它通常表示选择高性能 DPM 策略。

如果设备和驱动支持手动 DPM Level，可尝试：

```bash
sudo rocm-smi --setperflevel manual
sudo rocm-smi --setsclk <level>
sudo rocm-smi --setmclk <level>
```

随后必须重新读取：

- 实际核心频率；
- 显存频率；
- 功耗；
- 温度；
- Perf Level。

最终应以本机实际输出为准。

---

### 9.3 rocprof 基础统计

基础 Kernel 时序：

```bash
rocprof --stats ./exec
```

HIP API Trace：

```bash
rocprof --hip-trace ./exec
```

查询支持的基础 Counter：

```bash
rocprof --list-basic
```

查询 Derived Metric：

```bash
rocprof --list-derived
```

通过配置文件采集：

```bash
rocprof -i counters.txt ./exec
```

---

### 9.4 Counter 名称不能直接照搬 ROCm 或 Nsight

以下名称不应在知识库中写成所有环境都必然支持：

```text
L2CacheHit
MemUnitBusy
```

原因包括：

- DTK 版本不同；
- 海光驱动定制；
- Counter 命名不同；
- 部分 Counter 不开放；
- Counter 组合存在互斥限制；
- 不同架构支持范围不同。

应以本机输出为准：

```bash
rocprof --list-basic
rocprof --list-derived
```

---

### 9.5 与 Nsight 工具的对应关系

只能做功能层面的近似类比：

| NVIDIA | ROCm/DTK 近似工具 |
|---|---|
| `nvidia-smi` | `hy-smi` / `rocm-smi` |
| Nsight Systems | `rocprof` Trace / 相关系统分析工具 |
| Nsight Compute | `rocprof` Hardware Counters / 相关性能分析工具 |
| `cuobjdump` | `llvm-objdump` |
| SASS | AMDGCN ISA |

但不能认为：

- Counter 名称一一对应；
- 指标含义一一对应；
- 报告粒度完全一致；
- Roofline 数据可以直接照搬；
- Occupancy 模型完全相同。

---

## 10. ISA 与反汇编检查

本节同样属于平台维护/人工诊断附录，不授权 Implementer 绕过 `bash build.sh`。

### 10.1 保存中间文件

```bash
hipcc -O3 \
      --offload-arch=gfx906 \
      -save-temps \
      kernel.cpp \
      -o exec
```

---

### 10.2 查看反汇编

```bash
llvm-objdump -d exec
```

过滤 Global Memory 指令：

```bash
llvm-objdump -d exec | \
    grep -E 'global_load|global_store|flat_load|flat_store|buffer_load|buffer_store'
```

过滤 LDS/DS 指令：

```bash
llvm-objdump -d exec | \
    grep -E 'ds_read|ds_write|ds_bpermute'
```

过滤等待和同步指令：

```bash
llvm-objdump -d exec | \
    grep -E 's_waitcnt|s_barrier'
```

过滤 FMA 或 FP16 指令：

```bash
llvm-objdump -d exec | \
    grep -E 'v_fma|v_mac|f16|pk_'
```

---

### 10.3 反汇编重点检查项

检查向量 Load：

```text
是否生成 dwordx4
是否被拆成 dwordx2 或单 dword
是否使用 global/flat/buffer 路径
```

检查等待：

```text
s_waitcnt 是否过早
是否每次循环都 vmcnt(0)
是否破坏访存并行
```

检查 LDS：

```text
ds_read / ds_write 数量
是否出现过多 LDS 往返
是否可能存在 Bank Conflict
```

检查寄存器：

```text
VGPR 是否过高
是否出现 Scratch Load/Store
是否发生 Spill
```

检查计算：

```text
是否使用 Packed FP16
是否存在过多类型转换
是否正确使用 FMA
```

---

## 11. Kernel 优化检查顺序

建议按以下顺序进行性能分析。

1. 验证数值正确性；
2. 验证边界和 Tail；
3. 检查 Kernel Launch 配置；
4. 检查 Wave64 映射；
5. 检查 Workgroup Size；
6. 检查 VGPR 使用量；
7. 检查 SGPR 使用量；
8. 检查 LDS 使用量；
9. 检查 Scratch Spill；
10. 计算理论 Waves/CU；
11. 检查实际 Occupancy；
12. 检查 Global Load/Store 合并；
13. 检查向量 Load 是否被标量化；
14. 检查 L1/L2 命中；
15. 检查 LDS Bank Conflict；
16. 检查 Workgroup Barrier 数量；
17. 检查 `s_waitcnt` 是否过早等待；
18. 检查 Software Pipeline 是否真正重叠；
19. 检查计算与访存比例；
20. 对关键 Kernel 反汇编；
21. 使用 Profiler Counter 定位瓶颈；
22. 与库实现和朴素实现对比。

---

## 12. 常见错误认知

### 错误 1：`gfx906` 可以直接使用 MFMA

不应这样假定。

正确结论：

```text
当前应按 gfx906 Wave64 VALU 架构设计，
除非海光专有文档或实际 ISA 明确证明存在扩展矩阵指令。
```

---

### 错误 2：`Fast F16` 等于有 Tensor Core

错误。

`Fast F16` 只能证明 FP16 有快速硬件支持，不能证明存在专用矩阵单元。

---

### 错误 3：`s_waitcnt` 可以替代 `__syncthreads()`

错误。

- `s_waitcnt`：当前 Wave 的访存依赖等待；
- `__syncthreads()`：Workgroup 内所有 Wave 的同步。

---

### 错误 4：`ds_bpermute` 就是 DPP

错误。

- DPP：VALU 数据路径中的规则化 Lane Permutation；
- `ds_bpermute`：DS/LDS Permute 通路中的任意 Lane Gather。

---

### 错误 5：使用 `float4` 一定生成 128-bit Load

错误。

必须通过反汇编确认实际生成的 ISA。

---

### 错误 6：`rocm-smi --setperflevel high` 一定严格锁频

不准确。

它通常代表高性能策略，实际频率仍应通过监控命令确认。

---

## 13. 当前可确认与不可确认边界

### 13.1 已由终端输出确认

- Z200SM_80；
- ZIFANG；
- HYGON；
- `gfx906:sramecc+:xnack-`；
- Wave64；
- 64 CU；
- 4 SIMD/CU；
- 64 KB LDS；
- 16 KB L1；
- 8 MB L2；
- 约 16 GiB Global Memory；
- 1024 threads/workgroup；
- 40 waves/CU；
- 2560 work-items/CU；
- Fast FP16；
- 最大报告频率 1319 MHz。

### 13.2 仍需额外证据确认

- 完整服务器是否为 4 张 DCU；
- 每张 DCU 对应的 Linux NUMA Node；
- Host 到各卡的 PCIe 拓扑；
- 是否存在海光专有矩阵扩展指令；
- 实际 FP16 峰值；
- 实际显存带宽；
- LDS Bank 数量和 Bank 宽度；
- VGPR/SGPR 物理数量；
- 各类指令吞吐和延迟；
- `hy-smi` 支持的完整锁频命令；
- 本机 `rocprof` 可用的 Counter 名称。

这些内容应通过以下方式进一步确认：

```text
海光官方硬件手册
DTK 编程指南
编译器 Builtin 文档
rocprof Counter 列表
实际反汇编
微基准测试
```

---

## 14. 总结

针对 Z200SM_80 进行高性能 HIP C++ Kernel 开发时，应采用以下总体思路：

```text
Wave64 执行模型
+ Global→VGPR→LDS
+ LDS Tiling
+ Register Blocking
+ Packed FP16
+ HIP Shuffle / DS Permute
+ Workgroup Barrier
+ 软件流水
+ 反汇编验证
+ rocprof 性能计数器
```

不应直接照搬 Hopper/Blackwell 的：

```text
TMA
mbarrier
WGMMA
Warp32 Fragment
Tensor Core Pipeline
```

其中最关键的架构边界是：

1. 当前设备按 `gfx906` 编译；
2. 不应默认支持 MFMA；
3. `s_waitcnt` 不是 Workgroup Barrier；
4. `ds_bpermute` 不等于 DPP；
5. Wave64 会改变规约、线程映射和 Tile 设计；
6. LDS 仅 64 KB，FlashAttention 和 GEMM 双缓冲必须精确核算资源；
7. 所有向量化和 ISA 推断最终都应由反汇编验证。
