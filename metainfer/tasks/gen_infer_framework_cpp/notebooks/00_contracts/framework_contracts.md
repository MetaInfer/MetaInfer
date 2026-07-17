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
- **CPP-FW-009**：非 CPU Backend 的健康服务必须持有加速器 Device FD，
  权重驻留设备内存，并由设备 Kernel/BLAS 执行所有模型热路径；仅链接
  HIP/CUDA Library 或运行独立 Device Probe 不构成通过证据。
- **CPP-FW-010**：CPU Reference Kernel 只能用于小型单元测试，禁止完整
  Checkpoint 展开为主机 FP32 或在生产请求路径执行标量 CPU GEMM。
- **CPP-FW-011**：Chat 请求必须使用 Checkpoint Tokenizer 的 Chat
  Template 和 Special Token ID，禁止手工拼接 Role Prefix。
- **CPP-FW-012**：`temperature`、`top_p`、`seed`、`stream` 等已接受字段
  必须进入 Engine/Sampler 并改变对应语义；禁止解析后忽略。框架必须同时
  支持 `temperature=0` 的确定性 Greedy 和 `temperature>0` 的有 Seed 随机采样。
- **CPP-FW-013**：B 阶段必须生成 `implementation_report.json`，逐项记录
  Plan Item、实际文件和通过的测试。Prefill/Decode 差分测试必须有非零退出
  的阈值断言，禁止只打印 Cosine 后返回成功。

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
6. 同 Seed 随机采样复现、不同 Seed 输出差异和 `top_p` 生效；
7. Prefill/Decode Logits 与 Token 的阈值断言；
8. Cancellation 和 SIGTERM；
9. 原生进程树检查；
10. Immutable HTTP Correctness Oracle；
11. GPU Activity 和 TP Rank Activity Oracle。

## 8. 分阶段 Plan Gate

Plan Gate 必须区分首次完整交付与已验证基线上的增量工作：

- 第 1 轮、上一轮 C 未通过、上一轮中断、完成标记缺失或规划证据无效时，
  使用 **Full Gate**。Planner 必须给出完整架构、最小真实端到端路径、全部
  核心能力文件和完整 C Oracle 计划。
- 只有上一轮状态为 `success`、C 结果为 `ok`、未中断、完成标记存在且
  `plan_manifest.json` 与冻结需求一致时，才可使用 **Incremental Gate**。
- Incremental Gate 继承上一轮已验证核心能力。Planner 只列本轮目标、修改
  文件、受影响能力和专项测试，不得为了通过 Gate 重复枚举未修改文件。
- 增量实现仍必须运行 Immutable Full C Oracle 作为回归门槛。继承规划证据
  只减少重复规划，不能跳过运行时正确性验证。
- 失败或无法验证的基线必须自动降级到 Full Gate，禁止 Agent 自行声称继承。
- A Gate 通过后，B 仍必须通过独立 Implementation Gate。计划文件存在不代表
  实现完成；Implementation Gate 必须核对实际文件、增量 Diff、测试证据和
  `prefill_decode_cosine >= 0.95`。
