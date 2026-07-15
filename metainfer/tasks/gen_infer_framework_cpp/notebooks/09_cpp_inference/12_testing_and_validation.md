# 原生 C++ 测试与验收阶梯

测试必须在昂贵的完整模型 Oracle 之前定位错误。每个失败信息应包含 Component、Rank/Device、Expected/Actual Shape 或值，以及 Backend Status。

## 1. 测试目录

```text
tests/
  fixtures/
    tiny_config.json
    tiny_model.safetensors
    tokenizer_cases.json
  unit/
    test_checked_math.cpp
    test_config.cpp
    test_safetensors.cpp
    test_tensor.cpp
    test_scheduler.cpp
    test_kv_allocator.cpp
    test_http_json.cpp
  device/
    test_hip_runtime.hip
    test_blas.cpp
    test_rms_norm.hip
    test_rope.hip
    test_paged_kv.hip
    test_collective.cpp
  integration/
    test_qwen3_layer.cpp
    test_engine_lifecycle.cpp
    test_native_server.cpp
    test_tp_equivalence.cpp
```

## 2. 最小 Test Harness

如果不引入测试框架，也必须提供清楚的断言和退出码：

```cpp
#define EXPECT_TRUE(condition)                                      \
  do {                                                              \
    if (!(condition)) {                                             \
      std::cerr << __FILE__ << ':' << __LINE__                      \
                << " EXPECT_TRUE failed: " #condition << '\n';     \
      return 1;                                                     \
    }                                                               \
  } while (false)

int main() {
  auto result = RunFocusedCase();
  EXPECT_TRUE(result.ok());
  return 0;
}
```

更推荐使用仓库允许的成熟 C++ Test Framework，但不要在服务启动时下载。

## 3. Layer 1：Host-only Unit Test

无需 GPU，必须快速运行：

```text
JSON Config：字段缺失/类型/范围/Dense-MoE
Safetensors：Header/Offset/Shape/DType/Index/Path
Tokenizer Golden Vector/Chat Template
Checked Arithmetic/TensorView/RAII Move
Scheduler：FIFO/Budget/Cancel/Backpressure/Fake Runner Failure
KV：Reserve/Commit/Rollback/Generation/Exhaustion
HTTP JSON：Schema/错误映射/Response Shape
```

示例：

```cpp
TEST(SafeTensorParser, RejectsOutOfBoundsDataRange) {
  auto fixture = BuildSafeTensorFixture(
      /*shape=*/{2, 2}, /*dtype=*/DType::kFp32,
      /*declared_offsets=*/{0, 1024}, /*actual_data_bytes=*/16);
  auto parsed = ParseSafeTensor(fixture.path());
  ASSERT_FALSE(parsed.ok());
  EXPECT_EQ(parsed.status().code(), StatusCode::kDataLoss);
}
```

## 4. Layer 2：Device Probe

```cpp
TEST(HipRuntime, AllocCopyKernelCopyBack) {
  ASSERT_DEVICE_AVAILABLE();
  auto result = RunAddOneProbe(/*logical_device=*/0);
  ASSERT_TRUE(result.ok()) << result.status().message();
}

TEST(Blas, Bf16GemmMatchesFloatReference) {
  ASSERT_CAPABILITY("bf16_blas");
  auto actual = RunSmallBf16Gemm(/*m=*/7, /*n=*/11, /*k=*/5);
  auto expected = HostFloatGemm(...);
  EXPECT_ALLCLOSE(actual, expected, /*atol=*/..., /*rtol=*/...);
}
```

跳过必须附带明确原因。部署必需能力在设备不可访问时不能报告 Passed。

Device Matrix：

```text
枚举确切分配设备
Allocate/Copy/Kernel/Event/Free
Weight/KV DType 的 BLAS GEMM
RMSNorm、RoPE、SiLU、Argmax
Paged KV Block Boundary
TP Size 对应 Broadcast/AllReduce
```

## 5. Layer 3：模型组件 Reference 对比

为 Tiny Deterministic Fixture 保存 Host/Offline Reference：

```cpp
struct TensorExpectation {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<float> values;
  double atol;
  double rtol;
};

Status CompareTensor(const TensorView& device,
                     const TensorExpectation& expected);
```

比较：Embedding、RMSNorm、Q/K Norm、RoPE、Projection、Tiny Attention、MLP、一个 Decoder Layer、Prefill Logits、Decode Logits。

覆盖 Tail Dimension、变长序列、GQA Head Mapping、Page Boundary 和 DType。

## 6. Layer 4：Engine 生命周期

```cpp
TEST(Engine, PrefillThenCachedDecode) {
  FakeOrTinyNativeModel model;
  InferenceEngine engine(...);
  RequestId id = SubmitFixedPrompt(engine);
  EXPECT_TOKEN_SEQUENCE(engine, id, {/*golden tokens*/});
  EXPECT_EQ(engine.DebugFreeKvBlocks(), engine.DebugInitialKvBlocks());
}

TEST(Engine, CancellationReleasesKvExactlyOnce) { /* ... */ }
TEST(Engine, BackendFailureDoesNotCommitPartialStep) { /* ... */ }
TEST(Engine, RepeatedCreateDestroyHasStableMemory) { /* ... */ }
```

还要覆盖多请求 Continuous Admission、EOS、Max Token、KV Exhaustion、Queue Full、慢 Event Consumer、Shutdown。

## 7. Layer 5：Native Server 与进程真实性

Integration Test：

```bash
set -euo pipefail
port="$(pick_free_port)"
bash serve.sh "$port" >server.stdout.log 2>server.stderr.log &
server_pid=$!
trap 'kill -TERM "$server_pid" 2>/dev/null || true' EXIT

wait_until_ready "http://127.0.0.1:${port}/v1/models"
send_deterministic_chat_request "$port"
assert_native_process_tree "$server_pid"
kill -TERM "$server_pid"
wait "$server_pid"
```

脚本中的 Helper 需由项目实现。只能终止自己记录的 `server_pid`，端口占用时换端口。

检查：

```text
CMakeLists/include/src/tests/LANGUAGE_BOUNDARY 存在
serve.sh 最终启动迭代目录内原生 Binary
进程树无 Python 模型 Worker
原生进程打开 GPU Device FD 或映射 Accelerator Runtime
真实 MODEL_DIR，非 Mock/Canned Response
SIGTERM 有界退出并 Flush Profile
```

## 8. Layer 6：TP 与性能

```text
TP=1/TP=N Layer Output 与 Greedy Token
每 Rank Weight Shape/Bytes
Collective 顺序/Count/DType
注入一个 Rank 失败
期望设备 Activity
Correctness Oracle 先通过
固定 Prompt/Concurrency/Warmup 的 Perf Oracle
```

性能提高前后都要记录 Build Manifest、Hardware Profile、DType、TP Size、Context、Concurrency。

## 9. Failure Injection

建议给核心依赖留测试注入点：

```cpp
struct FailureInjection {
  std::optional<std::size_t> fail_allocation_number;
  std::optional<std::uint64_t> fail_step_id;
  std::optional<int> fail_rank;
  std::optional<OperatorKind> fail_operator;
};
```

该配置只在 Test Build 启用，生产服务拒绝相关环境变量/接口。

## 10. CMake/CTest 标签

```cmake
add_test(NAME unit_scheduler COMMAND test_scheduler)
set_tests_properties(unit_scheduler PROPERTIES LABELS "unit")

add_test(NAME device_hip COMMAND test_hip_runtime)
set_tests_properties(device_hip PROPERTIES LABELS "device")

add_test(NAME integration_server COMMAND test_native_server)
set_tests_properties(integration_server PROPERTIES LABELS "integration")
```

快速开发循环：

```bash
ctest --test-dir build -L unit --output-on-failure
ctest --test-dir build -L device --output-on-failure
ctest --test-dir build -L integration --output-on-failure
```

## 11. Acceptance Matrix

```text
model_family | backend | weight_dtype | kv_dtype | tp | feature | result
Qwen3 Dense  | DTK/HIP | BF16         | BF16     | 1  | base    | pass/fail
Qwen3 Dense  | DTK/HIP | BF16         | BF16     | 4  | TP      | pass/fail
Qwen3 MoE    | DTK/HIP | ...          | ...      | ...| MoE     | unsupported/pass
```

`Unsupported` 只有在服务运行前明确检测并拒绝时才诚实；不能用于掩盖已进入 Runtime 后失败的路径。
