# C++ 推理框架知识库总览

本知识库为`gen-infer-framework-cpp`卡片提供自包含的原生C++实现知识。目录的
权威关系是：

```text
00_contracts/        强制接口、不变量和失败条件
01-05/               设计、模型、算子、并行和服务实现指南
06_experience/       已验证的C++故障与修复证据
06_profiling/        可重复的原生性能观测方法
07_improvementPlan/  已接受但尚未完成的改进计划
08_issues/           已观察但尚未解决的问题
```

如果指南示例与 Contract 冲突，以 Contract 为准。代码块主要用于确定接口、数据流和错误边界，不应未经目标 DTK/HIP 环境验证就直接作为生产代码。

## 1. 目标范围

第一目标是 Qwen3 原生 C++ Serving Runtime，支持：

- Qwen3 Dense；
- 用户明确选择时的 Qwen3 MoE；
- 海光 DTK/HIP、NVIDIA CUDA 或已探测 Vendor Backend；
- Paged KV Cache、Continuous Batching、SSE；
- 可选 TP，多卡环境首先验证 TP=1，再扩展到 TP=N；
- OpenAI-compatible HTTP API；
- 原生 Profiling 和 Immutable Oracle。

平台适配必须以 `hardware_profile.json` 为事实来源。产品型号表示用户目标，HIP Architecture、Library 和能力由探测结果决定。

## 2. 统一目录建议

```text
CMakeLists.txt
include/metainfer/
  status.h
  dtype.h
  tensor.h
  backend.h
  hardware_profile.h
  model_config.h
  weight_loader.h
  kv_cache.h
  scheduler.h
  model_runner.h
  engine.h
  tokenizer.h
  http_server.h
src/
  common/
  backend/hip/
  operators/
  model/qwen3/
  engine/
  service/
  main.cpp
tests/
  unit/
  device/
  integration/
serve.sh
LANGUAGE_BOUNDARY.md
```

文件名可调整，但依赖方向应保持单向：

```text
HTTP/Tokenizer
      |
      v
InferenceEngine -> Scheduler -> StepPlan
      |                          |
      v                          v
PagedKvCache <-------------- ModelRunner
                                 |
                                 v
Qwen3 Model -> Operator Registry -> HIP/BLAS/Collective Backend
```

Backend 不应依赖 HTTP；ModelRunner 不应拥有 Scheduler；HTTP 不应直接访问 DeviceBuffer。

## 3. 全项目统一的基础类型

建议先实现最小错误模型，后续文档中的 `RETURN_IF_ERROR`/`ASSIGN_OR_RETURN` 都建立在这一层：

```cpp
enum class StatusCode {
  kOk,
  kInvalidArgument,
  kNotFound,
  kUnsupported,
  kUnavailable,
  kDataLoss,
  kResourceExhausted,
  kBackendError,
  kInternal,
  kCancelled,
};

class Status {
 public:
  static Status Ok();
  static Status Error(StatusCode code, std::string message);
  bool ok() const noexcept;
  StatusCode code() const noexcept;
  const std::string& message() const noexcept;
};

template <typename T>
class Result {
 public:
  Result(T value);
  Result(Status status);
  bool ok() const noexcept;
  const Status& status() const noexcept;
  T& value() &;
  T&& value() &&;
};
```

不要在核心 Runtime 中混用异常、整数返回码、`bool` 和 `nullptr` 表示错误。Destructor 不抛异常；跨线程错误通过明确的 Event/Future/Status 传播。

常用元数据类型：

```cpp
enum class DType { kFp32, kFp16, kBf16, kInt32, kInt64, kUnknown };

struct BackendStream {
  void* native = nullptr;
  int device = -1;
};

using RequestId = std::uint64_t;
using RankId = std::int32_t;
```

## 4. 运行时完整流程

```text
serve.sh
  -> CMake Configure/Build 或验证已有 Build Manifest
  -> 原生 main() 读取参数与 hardware_profile.json
  -> 校验设备可见性、权限、HIP Architecture 和依赖能力
  -> 加载 Qwen3 config/tokenizer/safetensors index
  -> 初始化 Rank/Backend/Collective
  -> 计算每 Rank 内存预算并上传权重
  -> 创建 Workspace/KV Block Pool
  -> Model Warmup/Probe
  -> Rank 0 监听 HTTP
  -> Tokenize -> Admit -> Schedule -> Prefill/Decode
  -> Sample -> Detokenize -> JSON/SSE
  -> SIGTERM: Stop admission -> Drain/Cancel -> Flush profile -> Release
```

## 5. 推荐实现顺序与每步证据

| 顺序 | 实现目标 | 必须留下的可执行证据 |
|---|---|---|
| 1 | CMake + Hardware Probe | `device_probe` 输出 JSON |
| 2 | Status、DType、Shape、RAII | Host Unit Test + HIP alloc/copy test |
| 3 | Config/Safetensors Parser | Tiny Fixture Test |
| 4 | BLAS Adapter + 基础 HIP Kernel | 与 Host Reference 对比 |
| 5 | Qwen3 一个 Layer | Intermediate Tensor 对比 |
| 6 | 单请求 Prefill/Decode | 固定 Logits/Greedy Token |
| 7 | Paged KV | Block Boundary/Exhaustion Test |
| 8 | Scheduler/Continuous Batching | Fake Runner Lifecycle Test |
| 9 | Native HTTP/SSE | Schema/Disconnect/SIGTERM Test |
| 10 | TP | Collective + TP=1/TP=N 等价 |
| 11 | Profiling/Optimization | Trace + Before/After Benchmark |

不要从 Fused Attention 开始。可调试的 Unfused Reference Path 是后续每个优化的正确性基准。

## 6. ABCDEGF 阶段阅读方式

- **A Planner**：本总览、硬件/框架/模型 Contract，以及本次 Feature 对应指南。
- **B Implementer**：先读 `plan.md`，再读将要修改组件的接口与测试章节。
- **C Debugger**：只读失败组件 Contract、实现指南、测试指南和日志。
- **D Reviewer**：以 Contract 为主，检查边界、所有权、Fallback、错误路径。
- **E/G/F Performance**：先测量，再读Profiling、Operator、KV、Batching和TP。

Prompt 已根据阶段和卡片 Feature 给出精确文件清单，不需要 Glob/读取全部 Notebook。

## 7. 示例代码的使用规则

1. 示例中的类名应优先保持一致，减少跨文档 API 漂移。
2. `HIP_CHECK`、`RETURN_IF_ERROR` 等宏需要在项目中统一实现。
3. `std::span` 仅适用于 C++20；本文统一用项目自定义 `Span<T>` 表示 C++17 非拥有连续 View。
4. 示例中的 `SmallVector` 一律可以用 `std::vector` 替代；本知识库默认使用标准容器，避免引入未声明依赖。
5. 任何 `hip*`、BLAS、Collective API 都必须在目标 DTK 编译验证。
6. 示例没有包含全部并发、OOM、Signal 和安全检查时，不得直接标为完成。

有价值的迭代必须产生一个更小、更清楚的可运行证明；只有新增源码行数而没有测试/Probe，不算有效进展。
