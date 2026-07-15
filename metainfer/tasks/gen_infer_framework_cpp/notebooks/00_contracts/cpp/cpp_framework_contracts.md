# C++ 推理框架总体契约

> 权威级别：`gen-infer-framework-cpp` 的强制契约。

交付物必须是一个真正的原生 C++ 推理服务，而不是“Python HTTP Server + C++ 扩展”。即使 HTTP 返回格式正确，只要响应来自 Mock、固定文本、CPU 静默回退、Python 模型进程或第三方完整推理引擎，也应判定失败。

## 1. 必需目录与交付物

```text
CMakeLists.txt
include/
  metainfer/
src/
tests/
serve.sh
LANGUAGE_BOUNDARY.md
```

构建产物统一放到 `build/`。`serve.sh PORT` 必须在前台启动服务，并最终通过 `exec` 交出进程控制权：

```bash
#!/usr/bin/env bash
set -euo pipefail

port="${1:?usage: serve.sh PORT}"
cmake -S . -B build -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
cmake --build build --parallel
exec ./build/metainfer_server --port "$port" --model "${MODEL_DIR:?MODEL_DIR is required}"
```

该脚本只是接口示意。实际 target 名称必须与 `CMakeLists.txt` 一致，且不得在线下载依赖。

## 2. 核心组件必须由 C++/HIP 实现

- 配置解析、Tokenizer 接入、权重索引与加载；
- Tensor 元数据、设备内存分配、生命周期管理；
- Qwen3 Dense/MoE 模型执行和算子分派；
- Scheduler、Request Queue、Continuous Batching、取消；
- Paged KV Cache 分配、Block Table 和回收；
- Sampling、Greedy/随机解码；
- TP Rank 生命周期、权重切分、Collective；
- HTTP JSON、OpenAI Response、SSE Streaming；
- 指标、Tracing、Signal 和优雅退出。

允许 Python 的范围仅限：离线格式转换、测试客户端、测试数据生成、Profiling 后处理。所有 Python 文件必须记录在 `LANGUAGE_BOUNDARY.md`，并说明为什么不能合理地用 C++ 实现。

## 3. 推荐的顶层接口

不同实现可以调整文件和类名，但职责边界必须等价：

```cpp
class InferenceEngine {
 public:
  virtual ~InferenceEngine() = default;

  virtual Result<RequestId> Submit(GenerateRequest request) = 0;
  virtual Status Cancel(RequestId id) = 0;
  virtual Result<EngineEvent> NextEvent(RequestId id) = 0;
  virtual Status Shutdown(std::chrono::milliseconds timeout) = 0;
};

class ModelRunner {
 public:
  virtual ~ModelRunner() = default;
  virtual Result<StepOutput> Execute(const StepPlan& plan,
                                     const KvBatchView& kv,
                                     BackendStream stream) = 0;
};
```

HTTP 层只能调用 `Submit/Cancel/NextEvent`，不得直接操作 KV Cache 或执行模型层。Scheduler 只生成 `StepPlan`，ModelRunner 不得在执行一半时自行重新调度。

## 4. 运行时强制规则

- **CPP-FW-001**：必须从 `MODEL_DIR` 加载真实模型；文件缺失或模型不兼容时，在接收推理请求前失败。
- **CPP-FW-002**：禁止返回 Canned Response，禁止找不到模型时自动进入 Mock 模式。
- **CPP-FW-003**：保持任务指定的 Device Visibility、Weight DType、KV DType 和 TP Size。
- **CPP-FW-004**：每个被接收的请求只能进入一次终态：Finished/Cancelled/Rejected/Failed。
- **CPP-FW-005**：任意 TP Rank 发生不可恢复错误后，所有 Rank 必须协调退出，禁止状态分叉。
- **CPP-FW-006**：正常退出、加载失败、Kernel 失败和客户端断开都必须释放所有权资源。
- **CPP-FW-007**：公共头文件不得暴露没有生命周期约束的 owning raw device pointer。
- **CPP-FW-008**：服务进程树中不得存在长期运行的 Python 模型 Worker。

## 5. 依赖边界

允许使用叶子能力库：

```text
C++ 标准库
小型 HTTP Transport / JSON Parser
Tokenizer / Safetensors Parser
HIP Runtime / 厂商 BLAS / Collective Library
```

这些依赖可以提供 Primitive，但不得提供完整 Scheduler、KV Allocator、Serving Loop 或 Model Engine。依赖名称、版本、License 和链接路径必须写入 Build Manifest。

## 6. 可观测性最低要求

启动日志至少输出一条结构化摘要：

```json
{
  "model": "/models/qwen3",
  "backend": "hygon-dtk-hip",
  "tp_size": 4,
  "weight_dtype": "bf16",
  "kv_dtype": "bf16",
  "native_server": true
}
```

日志不得包含完整 Prompt 或模型敏感路径之外的任意文件内容。

## 7. 验收门槛

必须依次通过：

1. CMake Configure/Build；
2. Host Unit Tests；
3. HIP/BLAS Device Probe；
4. Model Config/Weight Load；
5. 固定 Prompt 的 Deterministic 1-token Request；
6. Cancellation 和 SIGTERM；
7. 原生进程树检查；
8. Immutable HTTP Correctness Oracle；
9. GPU Activity 和 TP Rank Activity Oracle。
