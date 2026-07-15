# 海光 DTK / HIP 后端契约

> 权威级别：选择 Hygon DTK/HIP，或在海光设备上选择通用 HIP 时的强制契约。

## 1. 工具链选择

编译器、SDK、Library 和 HIP Architecture 必须来自 `hardware_profile.json`。`/opt/dtk` 只是常见路径，不是可以无条件写死的规范。

Configure 日志至少记录：

```text
CMake version / C++ standard
C++ compiler / HIP compiler path and version
CMAKE_HIP_ARCHITECTURES
HIP Runtime / BLAS / Collective resolved library path
enabled/disabled capabilities and probe result
```

推荐 CMake 入口：

```cmake
cmake_minimum_required(VERSION 3.21)
project(native_qwen3 LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_HIP_ARCHITECTURES)
  message(FATAL_ERROR
    "CMAKE_HIP_ARCHITECTURES must come from hardware_profile.json/rocminfo")
endif()

add_library(metainfer_hip
  src/backend/hip_backend.cpp
  src/operators/rms_norm.hip
  src/operators/rope.hip)
target_include_directories(metainfer_hip PUBLIC include)
```

具体最低 CMake 版本必须在目标 DTK 环境验证，不得因为此示例而强制使用服务器不支持的版本。

## 2. 统一错误处理

每个 HIP Runtime、BLAS、Collective 返回值都必须检查：

```cpp
#define HIP_RETURN_IF_ERROR(expr)                                      \
  do {                                                                 \
    const hipError_t _status = (expr);                                  \
    if (_status != hipSuccess) {                                        \
      return BackendError(#expr, hipGetErrorString(_status),             \
                          __FILE__, __LINE__, CurrentRankAndDevice());   \
    }                                                                   \
  } while (false)
```

宏只是示意。项目可以使用函数/`Result<T>`，但错误必须包含 Operation、Rank、Device、Source Location 和 SDK Error。

Kernel Launch 后必须及时读取 Launch Error，异步错误在明确的 Event/Step Boundary 观察：

```cpp
kernel<<<grid, block, shared_bytes, stream>>>(args...);
HIP_RETURN_IF_ERROR(hipGetLastError());
// 不在每个生产算子后 hipDeviceSynchronize()；在 Step/Event 边界检查异步状态。
```

## 3. Dispatch 层级

1. Dense GEMM 优先使用“目标 DTK 已提供且 Probe 通过”的公开 BLAS API；
2. RMSNorm、RoPE、KV 搬运等使用可移植 HIP Kernel 基线；
3. 架构专用 Kernel 只能在能力检查、正确性和性能测试后启用；
4. 缺少必须 Library 时，选择已验证的 Native HIP 替代或明确失败。

禁止：

- 把上游最新 ROCm API 当作 DTK HIP 4.4 必然支持；
- 复制 CUDA Warp Size、SM 数量、Shared Memory 限制；
- Accelerator 错误后静默执行 CPU 算子；
- 服务启动时下载/编译第三方依赖。

## 4. Backend 能力结构

```cpp
struct HipCapabilities {
  int runtime_version = 0;
  int driver_version = 0;
  std::string architecture;
  int max_threads_per_block = 0;
  std::size_t shared_memory_per_block = 0;
  bool bf16_kernels = false;
  bool bf16_blas = false;
  bool peer_access = false;
  bool collectives = false;
  bool roctx = false;
};

Result<HipCapabilities> ProbeHipCapabilities(int logical_device);
```

所有优化分派必须依赖这些“已验证能力”，不能只写 `if (gpu_name == "Z200SM")`。

## 5. 必需 Probe

在框架实现前先完成：

```text
hipGetDeviceCount / hipGetDeviceProperties
hipMalloc -> hipMemcpy -> 自定义 Kernel -> 校验 -> hipFree
指定 DType 的 BLAS GEMM 与 Host Reference 对比
TP 需要的 Broadcast/AllReduce
ROCTX/Profiler（如果声明支持）
```

Probe 的 Source、Build Command、Exit Code、Output 和设备映射必须存档。

## 6. 性能可移植性

Generic Kernel 的 Workgroup Size、Vector Width、Shared Memory 等从 Runtime Properties 和实测中选择。每个 Specialized/Fused Kernel 必须记录支持 Shape、DType、Architecture、Tolerance、Benchmark Range 和 Fallback ID。
