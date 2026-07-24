# Hygon/C-3000 K500SM_AI（gfx928）硬件与 ISA 开发知识库

> 适用对象：基于 DTK、HIP C++、DCC/Clang 开发高性能 Kernel 的开发者。  
> 重点场景：INT8 W8A8 GEMM、FP16/BF16 GEMM、规约、Softmax、FlashAttention、数据搬运与软件流水。  
> 文档依据：本机 `rocminfo`、`hipconfig --version`、`hipcc --version` 与 `llvm-mc -mcpu=gfx928 -mattr=help` 输出。

---

## 0. 结论先行

当前机器实际报告的加速卡型号为：

```text
Marketing Name: K500SM_AI
Target ID: gfx928:sramecc+:xnack-
Vendor Name: C-3000
Device Type: HCU
```

因此，本知识库按以下对象编写：

```text
K500SM_AI / gfx928
```

如果项目内部把该卡称为“K100”，应将“K100”理解为项目、整机或产品线内部名称；底层编译和 ISA 判断仍应以设备实际报告的 `gfx928` 为准。

当前已经可以确认：

- 单卡 120 CU；
- 每 CU 4 SIMD；
- Wave64；
- 单卡约 64 GiB 显存；
- 每 CU 64 KB LDS；
- 每 CU 16 KB L1；
- 单卡 8 MB L2；
- Cache Line 64 B；
- 最大 40 Waves/CU；
- 最大 2560 Work-items/CU；
- 最大 Workgroup 1024 Threads；
- 4 卡系统；
- DTK/HIP 6.2；
- DCC 25.10；
- Clang 17；
- `gfx928` 是海光/C-3000 定制 LLVM 后端明确识别的 Target。

当前还不能直接确认：

- `gfx928` 默认启用了哪些 `-mattr` Feature；
- 是否默认启用 INT8 DOT；
- 是否默认启用 `mAI`/MMOP 矩阵指令；
- 是否默认启用 FP8/BF8；
- 是否默认启用 Global/Buffer → LDS 直接搬运；
- 各类矩阵指令的 Lane-to-Fragment 映射；
- 指令吞吐、延迟和流水线数量；
- LDS Bank 数量是否为 16 或 32；
- VGPR/SGPR 物理容量；
- 显存类型、总线宽度与峰值带宽。

必须特别注意：

> `llvm-mc -mattr=help` 输出的是后端可识别的 Feature 列表，不是“gfx928 当前默认启用 Feature 列表”。

因此本文将信息分为三级：

| 标记 | 含义 |
|---|---|
| **已确认** | 来自本机 `rocminfo`、版本命令或实际设备输出 |
| **后端可识别** | LLVM/DCC 后端包含对应 Feature 或指令定义 |
| **待验证** | 需要实际编译、汇编或反汇编确认 gfx928 是否启用 |

---

# 1. 系统与设备拓扑

## 1.1 HSA Runtime

本机 HSA 系统报告：

```text
Runtime Version: 1.11
Machine Model: LARGE
System Endianness: LITTLE
DMAbuf Support: YES
Mwaitx: DISABLED
```

`DMAbuf Support: YES` 表示运行时具备 DMA-BUF 互操作能力，但是否能被具体框架、进程间共享或外部设备使用，还取决于驱动、权限和上层软件。

## 1.2 CPU 配置

系统暴露 4 个 CPU HSA Agent：

```text
Hygon C86 7285 32-core Processor
```

HSA Node 编号为 `0, 1, 2, 3`。每个 CPU Agent 报告约 32 GiB 可分配内存池。

注意：

- HSA Node 不应直接等同于 Linux NUMA Node；
- 需要结合 `numactl --hardware`、`lspci` 和 `/sys/bus/pci/devices/.../numa_node` 确认物理拓扑；
- 每张 HCU 应尽量绑定到其近端 CPU NUMA Node。

## 1.3 HCU 配置

系统暴露 4 个 gfx928 HCU Agent：

```text
Agent 5 → HSA Node 4
Agent 6 → HSA Node 5
Agent 7 → HSA Node 6
Agent 8 → HSA Node 7
```

四张卡的硬件参数一致。

BDFID 分别为：

```text
1024
9728
17152
25344
```

按常见 BDF 编码推测，可能对应：

```text
0000:04:00.0
0000:26:00.0
0000:43:00.0
0000:63:00.0
```

该映射仍需通过以下命令确认：

```bash
lspci -D | grep -i -E 'C-3000|Hygon|VGA|Display|3D'
```

---

# 2. 单卡硬件规格

## 2.1 基础信息

| 属性 | 实测值 | 状态 |
|---|---:|---|
| Marketing Name | `K500SM_AI` | 已确认 |
| Agent Name | `gfx928` | 已确认 |
| Vendor | `C-3000` | 已确认 |
| Device Type | `HCU` | 已确认 |
| Chip ID | `0x6210` | 已确认 |
| ASIC Revision | 1 | 已确认 |
| Target ID | `gfx928:sramecc+:xnack-` | 已确认 |
| Max Clock | 1400 MHz | 已确认 |
| SRAM ECC | Enabled | 已确认 |
| XNACK | Disabled | 已确认 |
| Fast FP16 | TRUE | 已确认 |

完整 ISA Target：

```text
amdgcn-amd-amdhsa--gfx928:sramecc+:xnack-
```

## 2.2 计算资源

| 属性 | 实测值 |
|---|---:|
| Compute Units | 120 |
| SIMDs per CU | 4 |
| Shader Engines | 8 |
| Shader Arrays per SE | 1 |
| Wavefront Size | 64 |
| Max Waves per CU | 40 |
| Max Work-items per CU | 2560 |
| Max Workgroup Size | 1024 |
| Max FBarriers per Workgroup | 32 |

全卡 SIMD 数：

```text
120 CU × 4 SIMD/CU = 480 SIMD
```

按最大驻留 Wave 数计算：

```text
120 CU × 40 Waves/CU = 4800 Waves
```

Wave64 对应理论最大在途 Work-items：

```text
4800 × 64 = 307200 Work-items
```

这只是线程数量上限。实际 Occupancy 还受 VGPR、SGPR、LDS、Scratch Spill、Workgroup 大小和编译器资源分配粒度共同限制。

## 2.3 存储层级

| 层级 | 实测容量 |
|---|---:|
| Global Memory | 67,092,480 KB |
| Global Memory | 约 63.98 GiB |
| L2 Cache | 8192 KB |
| L1 Cache | 16 KB/CU |
| LDS/GROUP | 64 KB/CU |
| Cache Line | 64 B |
| Allocation Granule | 4 KB |
| Allocation Alignment | 4 KB |

单卡显存可按约 64 GiB 理解，4 卡总显存约 256 GiB。

当前信息没有说明显存类型、总线宽度、Memory Channel 数量、显存频率和理论峰值带宽。

---

# 3. Wave64 执行模型

## 3.1 Wave 与 Workgroup

gfx928 使用：

```text
1 Wavefront = 64 Work-items
```

| Workgroup Threads | Waves/Workgroup |
|---:|---:|
| 64 | 1 |
| 128 | 2 |
| 256 | 4 |
| 512 | 8 |
| 1024 | 16 |

HIP 中可使用：

```cpp
const int lane_id = threadIdx.x % warpSize;
const int wave_id = threadIdx.x / warpSize;
```

当前设备 `warpSize == 64`。

从 CUDA Warp32 移植时，不能直接保留：

```cpp
lane = threadIdx.x & 31;
warp = threadIdx.x >> 5;
```

## 3.2 理论 Occupancy

只按 Work-items 和 Waves 计算：

| Threads/Block | Waves/Block | 最大 Blocks/CU | Waves/CU |
|---:|---:|---:|---:|
| 64 | 1 | 40 | 40 |
| 128 | 2 | 20 | 40 |
| 256 | 4 | 10 | 40 |
| 512 | 8 | 5 | 40 |
| 1024 | 16 | 2 | 32 |

1024-thread Block 即使寄存器和 LDS 都不构成限制，也只能达到 32 Waves/CU，无法达到 40 Waves/CU。

## 3.3 LDS 对 Occupancy 的限制

单 CU LDS 为 64 KB。

| LDS/Block | 最大 Blocks/CU |
|---:|---:|
| 8 KB | 8 |
| 16 KB | 4 |
| 24 KB | 2 |
| 32 KB | 2 |
| 48 KB | 1 |
| 64 KB | 1 |

例如 W8A8 GEMM：

```text
Block Size = 256 Threads
LDS/Block = 48 KB
```

每 CU 只能驻留一个 Block，即 4 Waves/CU，仅为理论 40 Waves/CU 的 10%。

---

# 4. 软件工具链

## 4.1 HIP Runtime

```text
hipconfig --version
6.2.0-0
```

当前 HIP Runtime/SDK 基线为 HIP 6.2。

## 4.2 DCC 编译器

```text
dcc version: 25.10.0-0
clang version 17.0.0
Target: x86_64-unknown-linux-gnu
InstalledDir: /opt/dtk/llvm/bin
```

实际工具位置包括：

```text
/opt/dtk/dcc/bin/llc
/opt/dtk/aillvm/bin/llvm-mc
/opt/dtk/aillvm/bin/llc
/opt/dtk/aillvm/bin/llvm-objdump
```

说明 DTK 中至少存在 DCC LLVM 与 AI LLVM 两套相关后端。

初步理解：

- `/opt/dtk/dcc`：普通 HIP/DCC 编译链；
- `/opt/dtk/aillvm`：包含 gfx928 AI/矩阵扩展定义的 LLVM 工具链。

最终应通过 `hipcc -###` 确认正常 HIP 编译实际使用哪一套后端。

## 4.3 编译命令

```bash
hipcc -O3 \
    --offload-arch=gfx928 \
    kernel.cpp \
    -o kernel
```

CMake：

```cmake
cmake_minimum_required(VERSION 3.21)
project(gfx928_kernel LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_ARCHITECTURES gfx928)

add_executable(kernel kernel.cpp)

target_compile_options(kernel PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)
```

保存中间文件：

```bash
hipcc -O3 \
    --offload-arch=gfx928 \
    -save-temps \
    kernel.cpp \
    -o kernel
```

---

# 5. 如何解释 `-mattr=help`

执行：

```bash
/opt/dtk/aillvm/bin/llvm-mc \
    -triple=amdgcn-amd-amdhsa \
    -mcpu=gfx928 \
    -mattr=help \
    </dev/null
```

输出确认：

```text
gfx928 - Select the gfx928 processor
```

这说明 AI LLVM 后端明确识别 gfx928。

但 `Available features for this target` 表示该 LLVM AMDGPU 后端注册的 Feature 集合，并不等于 gfx928 默认启用了所有 Feature。

例如列表同时存在：

```text
wavefrontsize16
wavefrontsize32
wavefrontsize64
```

而实机明确是 Wave64。因此不能因为列表中出现 `dot1-insts`、`mai-insts`、`mmop1-insts`、`fp8-insts` 或 `glb2lds-dword-x2x4`，就直接断言 gfx928 已默认启用这些能力。

正确判断链：

```text
Feature 出现在列表
        ↓
后端认识该 Feature
        ↓
检查 gfx928 默认 Subtarget
        ↓
编译测试
        ↓
反汇编
        ↓
确认硬件实际执行
```

---

# 6. gfx928 后端可识别的主要指令 Feature

以下内容表示 AI LLVM 后端存在对应 Feature 或指令定义；默认启用状态仍需验证。

## 6.1 基础架构扩展

```text
gfx9
gfx9-insts
gfx928-insts
```

说明 gfx928 基于 GFX9 体系扩展，并包含专用 `gfx928-insts`。

不能将 gfx928 完全等同于公开的 gfx906、gfx908 或 gfx90a。

## 6.2 标量和向量 ALU

后端可识别：

```text
16-bit-insts
fp64
fmaf
fast-fmaf
mad-mac-f32-insts
fma-mix-insts
mad-mix-insts
```

候选指令族：

```text
v_add_*
v_sub_*
v_mul_*
v_fma_*
v_fmac_*
v_mac_*
v_mad_*
s_add_*
s_mul_*
s_cmp_*
s_branch
```

地址、循环和统一分支一般由 SALU 处理；每 Lane 数据计算一般由 VALU 处理。

## 6.3 Packed FP16/BF16/FP32

后端可识别：

```text
vop3p
pk-fmac-f16-inst
hcu-packed-bf16-ops
hcu-packed-fp32-ops
packed-fp32-ops
real-true16
true16
```

候选包括：

```text
v_pk_fmac_f16
v_pk_fma_bf16
v_pk_mul_bf16
v_pk_add_bf16
v_pk_fma_f32
v_pk_mul_f32
v_pk_add_f32
v_pk_mov_b32
```

其中 `hcu-` 前缀表明 AI LLVM 中包含 HCU 专用扩展定义，但是否由 gfx928 默认启用仍需验证。

---

# 7. INT8/INT4 DOT 指令

W8A8 GEMM 最重要的候选 Feature：

```text
dot1-insts
dot2-insts
dot3-insts
dot4-insts
dot5-insts
dot6-insts
dot7-insts
dot8-insts
dot9-insts
dot10-insts
dp4x
```

## 7.1 INT8 候选指令

### `dot1-insts`

```text
v_dot4_i32_i8
v_dot8_i32_i4
```

`v_dot4_i32_i8` 的概念操作：

```text
dst_i32 =
    a.i8[0] * b.i8[0] +
    a.i8[1] * b.i8[1] +
    a.i8[2] * b.i8[2] +
    a.i8[3] * b.i8[3] +
    acc_i32
```

一条 32-bit 寄存器可打包 4 个 INT8。

### `dot6-insts`

```text
v_dot4c_i32_i8
```

后缀 `c` 通常表示带累加语义的变体，具体操作数格式应以海光 ISA 或汇编器接受的语法为准。

### `dot7-insts`

```text
v_dot4_u32_u8
v_dot8_u32_u4
```

### `dot8-insts`

```text
v_dot4_i32_iu8
v_dot8_i32_iu4
```

对 W8A8 优先验证：

```text
v_dot4_i32_i8
v_dot4c_i32_i8
v_dot4_u32_u8
v_dot4_i32_iu8
```

## 7.2 其他 DOT 候选

| Feature | 候选指令 |
|---|---|
| `dot2-insts` | `v_dot2_i32_i16`, `v_dot2_u32_u16` |
| `dot3-insts` | `v_dot8c_i32_i4` |
| `dot4-insts` | `v_dot2c_i32_i16` |
| `dot5-insts` | `v_dot2c_f32_f16` |
| `dot9-insts` | FP16/BF16 DOT |
| `dot10-insts` | `v_dot2_f32_f16` |

---

# 8. 矩阵指令候选：mAI 与 MMOP

后端可识别：

```text
mai-insts
mmop-insts
mmop1-insts
mmop-16X16X8f32-insts
mmop-16X16X4f64-insts
mmop-16X16X64i4u4-insts
mmop-fp8-insts
```

## 8.1 `mai-insts`

描述：

```text
Has mAI instructions
```

说明后端具备 mAI 指令类别，但 `-mattr=help` 没有直接给出完整指令名。

## 8.2 MMOP/MMAC 候选

### 基础 MMOP

```text
v_mmac_16x16x4_f32
```

### MMOP1

```text
v_mmac_16x16x4_tf32
v_mmac_16x16x4_f16
v_mmac_16x16x4_bf16
v_mmac_16x16x4_i8
v_mmac_16x16x4_u8
```

对 W8A8 最重要：

```text
v_mmac_16x16x4_i8
v_mmac_16x16x4_u8
```

其名称暗示输出矩阵 Tile 为 16×16，K 步长为 4，但必须进一步验证：

- A/B 每 Lane 数据布局；
- Accumulator Fragment；
- Wave64 的协作方式；
- 累加类型；
- LDS Matrix Layout；
- 是否需要 `ds_read_m...`；
- 操作数修饰符和符号语义。

### INT4 MMOP

```text
v_mmac_16x16x64_i4
v_mmac_16x16x64_u4
```

### FP8/BF8 MMOP

```text
v_mmac_16x16x32_fp8
v_mmac_16x16x32_bf8
```

## 8.3 普通 DOT 与 MMOP 的区别

| gfx928 候选机制 | 执行模式 | NVIDIA 近似概念 |
|---|---|---|
| `v_dot4_i32_i8` | 每 Lane packed DOT | DP4A |
| `v_mmac_*` | Wave64 协同矩阵 Tile | Warp-level MMA/Tensor Core |
| `ds_read_m*` | 矩阵友好 LDS Load | Fragment Load |
| Matrix Format DS | LDS Layout 转换 | LDMATRIX/布局重排类能力 |

类比只用于理解，不表示线程映射和 ISA 完全相同。

---

# 9. LDS 矩阵读取和格式转换

后端可识别：

```text
ds-matrix-insts
ds-matrix-fmt-insts
ds-pk-read-insts
```

## 9.1 DS Matrix 指令

描述：

```text
Has ds_read_mXXX_tf32/f16/bf16/i8/u8/i4/u4 instructions
```

可能的数据路径：

```text
LDS
 ↓
按矩阵 Fragment 规则读取和重排
 ↓
VGPR Fragment
 ↓
v_mmac
```

W8A8 重点搜索：

```text
ds_read_m...i8
ds_read_m...u8
```

## 9.2 Matrix Format 指令

候选：

```text
ds_read_matrix_format
ds_read_matrix_trans_format
ds_write_matrix_format
ds_scale_copy
ds_read_matrix_padbyte
```

可能用于：

- 矩阵格式化读取；
- 转置格式读取；
- 矩阵专用 Layout 写入；
- Scale Copy；
- INT8/INT4 Padding；
- 为 MMOP 准备 Fragment。

## 9.3 Packed LDS Read

候选：

```text
ds_read_pack_l_b16
ds_read_pack_h_b16
ds_read_pack_0_b8
ds_read_pack_1_b8
ds_read_pack_2_b8
ds_read_pack_3_b8
ds_read_mask_b32
```

这些指令可能降低手工位移、掩码和字节打包开销。

---

# 10. DPP、Shuffle 与 Lane Permute

后端可识别：

```text
dpp
dpp8
dpp-64bit
dpp-src1-sgpr
permlane16-swap
permlane32-swap
```

DPP 用于同一 Wave64 中 Lane 之间的寄存器数据交换，类似 CUDA Shuffle 的底层硬件能力。

HIP 层优先使用：

```cpp
__shfl()
__shfl_down()
__shfl_xor()
```

Wave64 求和：

```cpp
__device__ inline int wave_reduce_sum(int value)
{
    for (int offset = 32; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 64);
    }
    return value;
}
```

Offset：

```text
32 → 16 → 8 → 4 → 2 → 1
```

DPP 与 DS BPermute 的区别：

| 机制 | 特点 |
|---|---|
| DPP | 固定模式的 Lane Permutation |
| DS BPermute | 按动态 Lane Index Gather |
| HIP Shuffle | 上层接口，由编译器选择底层实现 |

---

# 11. Global/Buffer → LDS 直接搬运候选

后端包含：

```text
glb2lds-dword-x2x4
buf2lds-dword-x2x4
buffer-load-lds-dword
```

描述：

```text
support global_load_dwordx2/x4 to lds
support buffer_load_dwordx2/x4 to lds
```

如果 gfx928 默认启用，可能支持：

```text
Global/Buffer Memory
        ↓
由 Wave Lane 发起的直接 Load-to-LDS
        ↓
LDS
```

这不等同于 Hopper TMA：

- 不一定有独立搬运引擎；
- 不一定支持多维 Tensor Descriptor；
- 仍可能需要每个 Lane 发起操作；
- 仍受 `s_waitcnt` 管理；
- 跨 Wave 使用 LDS 仍需要 Workgroup Barrier；
- 是否默认启用需反汇编确认。

这是 gfx928 上非常值得优先验证的能力。

---

# 12. Global、Flat、Buffer 与 LDS 指令族

## 12.1 Global/Flat

候选：

```text
global_load_dword[x2/x4]
global_store_dword[x2/x4]
flat_load_dword[x2/x4]
flat_store_dword[x2/x4]
```

后端可识别：

```text
flat-address-space
flat-global-insts
flat-inst-offsets
```

## 12.2 Buffer

候选：

```text
buffer_load_dword[x2/x4]
buffer_store_dword[x2/x4]
```

Buffer 指令通常通过 Resource Descriptor、Scalar Base、Vector Offset 和 Immediate Offset 完成地址计算。

## 12.3 LDS/DS

候选：

```text
ds_read_b32/b64/b128
ds_write_b32/b64/b128
```

后端可识别：

```text
enable-ds128
unaligned-ds-access
lds-direct-read-64bit
```

默认启用状态仍需验证。

---

# 13. 同步与 Wait Counter

常见 Wait Counter：

```text
vmcnt
lgkmcnt
vscnt
```

后端可识别：

```text
vscnt
auto-waitcnt-before-barrier
back-off-barrier
```

跨 Wave 共享 LDS 时仍需：

```cpp
__syncthreads();
```

区别：

| 机制 | 作用 |
|---|---|
| `s_waitcnt` | 当前 Wave 等待访存完成 |
| `s_barrier` | Workgroup 中多个 Wave 汇合 |
| `__syncthreads()` | HIP C++ Workgroup Barrier |

`auto-waitcnt-before-barrier` 不能简单理解为可以删除所有 Wait 和同步；其默认启用状态及具体语义仍需验证。

---

# 14. 原子操作候选

后端可识别：

```text
atomic-fadd-rtn-insts
atomic-fadd-no-rtn-insts
flat-atomic-fadd-f32-inst
atomic-buffer-global-pk-add-f16-insts
atomic-ds-pk-add-16-insts
atomic-ds-pk-compute-16-insts
atomic-flat-pk-add-16-insts
atomic-global-pk-add-bf16-inst
scalar-atomics
```

候选能力：

- Global/Buffer FP32 Atomic Add；
- 返回原值和不返回原值版本；
- Packed FP16/BF16 Atomic；
- LDS Packed FP16/BF16 Atomic；
- Scalar Memory Atomic。

可能用于 Scatter、Embedding Gradient、Histogram、MoE Token Dispatch 和多 Block 累加。

---

# 15. FP8、BF8 与 MXFP 候选能力

后端可识别：

```text
fp8-insts
fp8-conversion-insts
hcu-fp8-insts
mmop-fp8-insts
mxfp864-cvt-scale-insts
mxfp864-cvt-scale-insts1
```

候选能力：

- FP8/BF8 转换；
- HCU 专用 FP8/BF8 指令；
- FP8/BF8 MMAC；
- MXFP8/6/4 Scale Conversion。

实际使用前需确认具体数据格式、Scale 规则、Accumulator 类型、舍入和饱和语义。

---

# 16. W8A8 GEMM 推荐实现路线

## 16.1 第一阶段：普通 DOT Kernel

```text
Global INT8
    ↓
128-bit Vector Load
    ↓
LDS Tile
    ↓
每 Lane 读取打包 INT8
    ↓
v_dot4_i32_i8 候选
    ↓
INT32 Accumulator
    ↓
Scale / Bias / Convert
    ↓
Output
```

关键点：

- 一个 32-bit 寄存器打包 4 个 INT8；
- 每条 DOT 候选指令处理 4 对 INT8；
- 输出累加优先保持 INT32；
- Epilogue 再转 FP16/BF16/INT8。

## 16.2 第二阶段：MMOP/MMAC Kernel

若 `v_mmac_16x16x4_i8` 经验证可用：

```text
Global
  ↓
LDS Matrix Tile
  ↓
ds_read_m...i8 / Matrix Format Load
  ↓
Wave64 Matrix Fragment
  ↓
v_mmac_16x16x4_i8
  ↓
INT32 Accumulator Tile
  ↓
Epilogue
```

主要难点：

- Thread-to-Data Mapping；
- LDS Matrix Layout；
- Accumulator Fragment；
- 每 Wave 输出 Tile；
- 多 Wave Block Tile；
- B Matrix 转置或 Swizzle；
- K 维循环；
- 双缓冲；
- VGPR 数量；
- Epilogue 数据重排。

## 16.3 Tile 资源估算

假设：

```text
A Tile = 128 × 64 INT8
B Tile = 64 × 128 INT8
```

单缓冲 LDS：

```text
A = 128 × 64 × 1 B = 8192 B
B = 64 × 128 × 1 B = 8192 B
Total = 16 KB
```

双缓冲为 32 KB，从 LDS 角度每 CU 最多两个 Block。

若 Block 为 256 Threads：

```text
2 Blocks × 4 Waves = 8 Waves/CU
```

仅为 40 Waves/CU 上限的 20%。矩阵指令 Kernel 不一定需要很高 Occupancy，最终应结合指令吞吐、计算/访存比和延迟隐藏效果判断。

---

# 17. Cache Line 与 W8A8 访存

gfx928 Cache Line 为 64 B。

Wave64 每 Lane 读取一个 INT8：

```text
64 × 1 B = 64 B
```

恰好覆盖一条 Cache Line。

每 Lane 读取一个打包 INT32：

```text
64 × 4 B = 256 B
```

对应 256 个 INT8 元素，覆盖 4 条 64B Cache Line。

每 Lane 读取 16B 向量：

```text
64 × 16 B = 1024 B
```

覆盖 16 条 Cache Line。

优化重点：

- 起始地址至少 16B 对齐；
- Tile 行首尽量 64B 对齐；
- Wave 的 Lane 地址连续；
- 避免大 Stride；
- Tail 单独处理；
- 通过反汇编确认 `dwordx4`；
- 通过 Counter 验证 L2 和显存效率。

---

# 18. 推荐验证流程

## 18.1 确认 hipcc 实际后端

```bash
hipcc -### \
    --offload-arch=gfx928 \
    -x hip \
    -c /dev/null \
    -o /tmp/null.o \
    2>&1 | tee hipcc_gfx928_driver.txt
```

搜索：

```bash
grep -E '/opt/dtk|clang|llc|lld|amdgcn|gfx928' \
    hipcc_gfx928_driver.txt
```

## 18.2 保存中间文件

```bash
hipcc -O3 \
    --offload-arch=gfx928 \
    -save-temps \
    kernel.cpp \
    -o kernel
```

## 18.3 反汇编

```bash
/opt/dtk/aillvm/bin/llvm-objdump \
    -d \
    --mcpu=gfx928 \
    ./kernel \
    > gfx928_disassembly.txt
```

搜索关键指令：

```bash
grep -Ei \
'dot|mmac|mfma|mma|mai|pk_|dpp|bpermute|global_load|buffer_load|ds_read|ds_write|waitcnt|barrier' \
    gfx928_disassembly.txt
```

## 18.4 W8A8 重点搜索

```bash
grep -Ei \
'v_dot4.*i8|v_mmac.*i8|ds_read_m.*i8|ds_read_pack_.*b8' \
    gfx928_disassembly.txt
```

出现 `v_dot4_i32_i8` 表示使用 packed INT8 DOT；出现 `v_mmac_16x16x4_i8` 表示使用矩阵级 INT8 MMAC。

## 18.5 Global → LDS 搜索

```bash
grep -Ei \
'global_load.*lds|buffer_load.*lds|load_lds' \
    gfx928_disassembly.txt
```

---

# 19. 与 Z200SM_80/gfx906 对比

| 属性 | Z200SM_80 | K500SM_AI |
|---|---:|---:|
| Target | gfx906 | gfx928 |
| CU | 64 | 120 |
| SIMD/CU | 4 | 4 |
| Shader Engines | 4 | 8 |
| Wave Size | 64 | 64 |
| Max Waves/CU | 40 | 40 |
| Max Work-items/CU | 2560 | 2560 |
| LDS/CU | 64 KB | 64 KB |
| L1/CU | 16 KB | 16 KB |
| L2/卡 | 8 MB | 8 MB |
| Cache Line | 64 B | 64 B |
| 显存 | 约 16 GiB | 约 64 GiB |
| 最大报告频率 | 1319 MHz | 1400 MHz |
| 专用 Target Feature | gfx906 | gfx928-insts |
| AI LLVM MMOP 定义 | 未确认 | 后端可识别 |
| INT8 DOT 定义 | 需检查 | 后端可识别 |
| FP8/HCU Packed 定义 | 需检查 | 后端可识别 |

gfx928 相比 gfx906 的重点不仅是 CU 数量增加，还可能包含 INT8/INT4 DOT、mAI/MMOP、LDS Matrix Load、HCU Packed BF16/FP32、FP8/BF8 与 Global/Buffer → LDS Load。

---

# 20. 当前知识边界

## 20.1 已确认

- 4 张 K500SM_AI；
- gfx928；
- 120 CU；
- 4 SIMD/CU；
- 8 Shader Engines；
- Wave64；
- 64 GiB 显存；
- 64 KB LDS/CU；
- 16 KB L1/CU；
- 8 MB L2/卡；
- 64B Cache Line；
- 1400 MHz 最大报告频率；
- 40 Waves/CU；
- 2560 Work-items/CU；
- 1024 Threads/Workgroup；
- HIP 6.2；
- DCC 25.10；
- Clang 17；
- AI LLVM 识别 gfx928；
- AI LLVM 注册了 gfx928 专用 Feature。

## 20.2 后端可识别，但未证明 gfx928 默认启用

- INT8/INT4 DOT；
- FP16/BF16 DOT；
- mAI；
- MMOP/MMAC；
- INT8/UINT8 Matrix MMAC；
- FP8/BF8；
- HCU Packed BF16；
- HCU Packed FP32；
- DS Matrix Read；
- Matrix Format Read/Write；
- Packed B8 LDS Read；
- Global/Buffer → LDS x2/x4；
- DPP/DPP8/64-bit DPP；
- Packed FP16/BF16 Atomic；
- Scalar Atomic；
- MXFP Conversion。

## 20.3 仍需补充

- gfx928 默认 Feature 列表；
- DCC 与 AI LLVM 的默认 Feature 差异；
- W8A8 Kernel 反汇编；
- `v_mmac` Fragment Layout；
- LDS Bank 数；
- VGPR/SGPR 容量；
- 指令延迟与吞吐；
- 显存带宽；
- PCIe/NUMA 映射；
- Profiler Counter；
- BLAS/MIOpen/通信库版本；
- `hy-smi` 完整信息。

---

# 21. 最重要的工程结论

1. **gfx928 是 Wave64，不是 NVIDIA Warp32。**

2. **每 CU 仍只有 64 KB LDS。** 120 CU 不代表可以随意扩大单 Block Tile。

3. **Cache Line 是 64B。**

4. **W8A8 第一优先级应验证 `v_dot4_i32_i8`。**

5. **高性能矩阵 Kernel 应重点验证 `v_mmac_16x16x4_i8`。**

6. **后端中存在 DS Matrix Read 与 Matrix Format 指令定义。** 它们可能用于为 MMOP 准备 Fragment。

7. **后端中存在 Global/Buffer → LDS x2/x4 Feature。** 但它不是 Hopper TMA，且是否默认启用仍需反汇编确认。

8. **`-mattr=help` 不是启用列表。** 不应把所有 Feature 直接写成 gfx928 硬件已支持。

9. **最终结论以实际生成 ISA 为准。**

10. **W8A8 GEMM 应同时维护 DOT 版本与 MMOP 版本：**

```text
DOT 版本：
更容易开发、验证和理解

MMOP 版本：
潜在峰值更高，但 Fragment、Layout 和寄存器映射更复杂
```
