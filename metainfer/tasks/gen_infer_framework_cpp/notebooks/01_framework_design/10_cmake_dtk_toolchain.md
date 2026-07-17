# CMake 与海光 DTK/HIP 工具链

先读：`00_contracts/hardware_profile_contracts.md`、`03_operators/05_hip_blas_backend.md`。

## 1. 从硬件画像配置，而不是写死路径

典型命令结构：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_HIP_COMPILER="<hardware-profile-hipcc>" \
  -DCMAKE_HIP_ARCHITECTURES="<rocminfo-architecture>"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`<...>` 是必须替换的探测值，不能原样复制。禁止根据 `Z200SM` 猜测架构。

如果 `hardware_profile.json` 中没有 HIP Architecture：

1. 查阅 `rocminfo` 原始证据；
2. 运行原生 `hipGetDeviceProperties` Probe；
3. 仍无法确认则报告 blocker；
4. 只有用户显式给出 Override 时才允许继续，并写入 Build Manifest。

## 2. 推荐顶层 CMake

```cmake
cmake_minimum_required(VERSION 3.21)
project(metainfer_native_qwen3 VERSION 0.1.0 LANGUAGES CXX)

option(METAINFER_ENABLE_HIP "Build HIP backend" ON)
option(METAINFER_ENABLE_TESTS "Build tests" ON)
option(METAINFER_ENABLE_PROFILING "Enable native tracing hooks" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(METAINFER_ENABLE_HIP)
  enable_language(HIP)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR "HIP architecture must be provided by hardware discovery")
  endif()
endif()

add_library(metainfer_core
  src/common/status.cpp
  src/common/tensor.cpp
  src/model/qwen3/config.cpp
  src/model/qwen3/weight_loader.cpp
  src/engine/scheduler.cpp
  src/engine/paged_kv_cache.cpp)
target_include_directories(metainfer_core PUBLIC include)
target_compile_features(metainfer_core PUBLIC cxx_std_17)

if(METAINFER_ENABLE_HIP)
  add_library(metainfer_hip
    src/backend/hip/hip_backend.cpp
    src/operators/rms_norm.hip
    src/operators/rope.hip
    src/operators/kv_cache.hip)
  target_include_directories(metainfer_hip PUBLIC include)
  target_link_libraries(metainfer_hip PUBLIC metainfer_core)
endif()

add_executable(metainfer_server
  src/main.cpp
  src/service/http_server.cpp)
target_link_libraries(metainfer_server PRIVATE metainfer_core metainfer_hip)

if(METAINFER_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
```

该示例展示 Target 边界，不代表所有 DTK 版本都接受同一最低 CMake 版本或 `.hip` 行为。Agent 必须在目标服务器验证。

## 3. 依赖探测与能力编译

不要仅根据 Header 存在就启用能力。推荐“Find + Try Compile + Runtime Probe”三层判断。

```cmake
find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
  HINTS "${METAINFER_DTK_ROOT}/include")
find_library(HIP_RUNTIME_LIBRARY
  NAMES amdhip64 hip_hcc
  HINTS "${METAINFER_DTK_ROOT}/lib" "${METAINFER_DTK_ROOT}/lib64")

if(NOT HIP_INCLUDE_DIR OR NOT HIP_RUNTIME_LIBRARY)
  message(FATAL_ERROR "Detected DTK/HIP runtime headers and library are required")
endif()

target_include_directories(metainfer_hip PRIVATE "${HIP_INCLUDE_DIR}")
target_link_libraries(metainfer_hip PRIVATE "${HIP_RUNTIME_LIBRARY}")
```

具体 Library 名称由 `hardware_profile.json`/目标 DTK 决定，示例中的候选不能视为完整列表。

能力探针可以使用 `check_cxx_source_compiles` 或独立 `try_compile`：

```cmake
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${HIP_INCLUDE_DIR}")
check_cxx_source_compiles([[
  #include <hip/hip_runtime.h>
  int main() {
    hipDeviceProp_t p{};
    return hipGetDeviceProperties(&p, 0) == hipSuccess ? 0 : 1;
  }
]] METAINFER_HIP_HEADERS_COMPILE)
```

编译成功仍不代表当前用户有 `/dev/kfd` 权限，所以 Runtime Probe 不能省略。

## 4. 生成统一能力 Header

```cmake
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/metainfer/build_config.h"
  @ONLY)
target_include_directories(metainfer_core PUBLIC
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
```

`build_config.h.in`：

```cpp
#pragma once
#cmakedefine01 METAINFER_HAS_HIP
#cmakedefine01 METAINFER_HAS_HIPBLAS
#cmakedefine01 METAINFER_HAS_COLLECTIVES
#cmakedefine01 METAINFER_HAS_ROCTX
#define METAINFER_BUILD_HIP_ARCH "@CMAKE_HIP_ARCHITECTURES@"
```

这些宏只能表示“已完成配置/编译验证”，不能用于偷偷改变模型语义。

## 5. Target 划分与编译器边界

```text
metainfer_core       纯 C++：状态、Parser、Tensor 元数据、Engine 接口
metainfer_weights    JSON/Safetensors/Tokenizer Adapter
metainfer_hip        HIP Runtime、BLAS Adapter、HIP Kernel
metainfer_model      Qwen3 Layer/Weight Mapping
metainfer_engine     KV/Scheduler/Runner
metainfer_server     HTTP/SSE/Rank-0 服务
unit_*               不依赖 GPU 的快速测试
device_*             小型 HIP/BLAS/Collective Probe
```

不要把所有 `.cpp` 都交给 `hipcc`。Host-only Target 使用 C++ Compiler，只有 HIP Translation Unit 使用 HIP Compiler，减少 ABI 和构建时间问题。

## 6. Build Manifest

每次成功 Configure 后生成 `build/build-manifest.json`：

```json
{
  "build_type": "RelWithDebInfo",
  "cxx_standard": 17,
  "cxx_compiler": "...",
  "hip_compiler": "...",
  "hip_architectures": ["..."],
  "libraries": {"hip_runtime": "...", "blas": "...", "collective": "..."},
  "features": {"paged_kv": true, "tp": true, "profiling": true}
}
```

`serve.sh` 应验证现有 Build Manifest 与当前源码/配置是否匹配；不匹配则重新 Configure，不得静默运行陈旧 Binary。

## 7. Warning 与测试策略

Host C++ 建议至少启用：

```cmake
target_compile_options(metainfer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Wshadow>)
```

不同 Compiler 对 Warning Flag 支持不同，应先探测。Lifetime、Narrowing、Ignored Return、UB 相关警告必须处理。

测试注册示例：

```cmake
add_executable(test_config unit/test_config.cpp)
target_link_libraries(test_config PRIVATE metainfer_core)
add_test(NAME config COMMAND test_config)

add_executable(device_probe device/device_probe.hip)
target_link_libraries(device_probe PRIVATE metainfer_hip)
add_test(NAME hip_device_probe COMMAND device_probe)
set_tests_properties(hip_device_probe PROPERTIES LABELS "device")
```

## 8. 常见失败模式

- 写死 `/opt/dtk`，换安装路径后找错 Header/Library；
- 用产品名称作为 `CMAKE_HIP_ARCHITECTURES`；
- 找到 Header 但 Link 到另一个 SDK 的 `.so`；
- 所有文件都通过 HIP Compiler；
- Release 下 `assert` 消失导致测试没有校验；
- `serve.sh` 每次启动都联网下载或全量重编；
- Build Binary 与当前 Model/Feature/DType 配置不一致。
