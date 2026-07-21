# Qwen3-8B Z200 C0.1 快速算子数值测试契约

> 用途：指导实现 Agent 为生成的 C++/HIP 框架增加一个不加载真实模型、只使用确定性小张量的快速数值测试程序，并指导 MetaInfer 的 C 阶段在编译成功后、启动 HTTP server 前执行它。
>
> 本文只定义 **C0.1 快速算子数值测试**。固定 GGUF logits golden、完整模型数值 reference 和 HTTP 语义测试不属于本文范围。
>
> 多 slot / continuous batching 的 KV 隔离、batched logits 与 HTTP 并发测试不属于 C0.1；这些实现和验收以 `09_continuous_batching_contract.md` 为准。

参考路径：

```text
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/notebooks/qwen3_z200_kernels.hip.cpp
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/notebooks/04_qwen3_z200_operator_contract.md
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/notebooks/06_qwen3_runtime_notes.md
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/notebooks/09_continuous_batching_contract.md
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/orchestrator/hardware.py
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/orchestrator/oracles/correctness.py
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/orchestrator/pipeline.py
```

## 1. C0.1 的目标和边界

C0.1 要证明：生成框架实际链接的 Z200 HIP/hipBLAS 算子，在确定性小张量上与独立 CPU 公式一致。

C0.1 必须满足：

- 不加载 Qwen3-8B GGUF；
- 不依赖 tokenizer、HTTP server 或 `serve.sh`；
- 不分配完整模型权重和 KV cache；
- 不访问网络、不下载 gtest 等依赖；
- 使用固定输入、固定 seed、固定 shape；
- 直接运行在目标 Z200/gfx906 上；
- 与服务器链接同一个核心库和同一份 kernel object；
- 全部通过时返回 `0`，任一失败时返回非零；
- 正常目标是在几十秒内完成，C 阶段默认超时建议 120 秒。

C0.1 能发现：

- Q8_0 block layout、scale、行布局或 FP16 round 错误；
- hipBLAS `M/N/K`、转置、leading dimension 错误；
- RMSNorm reduction、Q/K per-head stride 错误；
- Qwen3 NEOX RoPE 配对或 position 错误；
- GQA 的 Q-head 到 KV-head 映射错误；
- KV cache `start_pos + local_t` 写入错误；
- prefill/decode attention 的 causal 范围或 softmax 错误；
- SwiGLU、residual、greedy tie-break 和尾部元素错误。

C0.1 不能单独证明：

- GGUF metadata/tensor offset 加载正确；
- 36 层完整 Qwen3-8B 的最终 logits 与可信 reference 一致；
- tokenizer/chat template 正确；
- HTTP JSON、`serve.sh` 和生成文本正确。

因此完整 C 流程应是：

```text
B writes CMake + qwen3_core + qwen3_numeric_tests + server + serve.sh
  -> C regenerates system-owned build.sh
  -> C runs bash build.sh
  -> C0.1 runs build/qwen3_numeric_tests
       fail -> do not load model or start server
       pass -> continue
  -> C starts serve.sh with the real model
  -> C runs existing OpenAI HTTP correctness cases
```

## 2. 生成框架必须提供的文件

建议生成目录至少包含：

```text
CMakeLists.txt
include/
  qwen3_z200_kernels.h
  qwen3_runtime.h
src/
  qwen3_z200_kernels.hip.cpp
  qwen3_runtime.cpp
  model_loader.cpp
  engine.cpp
  http_server.cpp
tests/
  qwen3_numeric_tests.cpp
  cpu_reference.h
  numeric_test_utils.h
serve.sh
build.sh                         # MetaInfer system-owned, Agent 不得修改
```

构建后必须存在：

```text
build/metainfer_cpp_server
build/qwen3_numeric_tests
```

`qwen3_numeric_tests` 必须链接服务器使用的同一个 `qwen3_core`。禁止：

- 在 `tests/` 再复制一份修改过的 kernel；
- 测试直接编译 notebooks 里的参考源码，而服务器使用另一份生成源码；
- 在测试中重写一个更简单的“假 GPU 实现”；
- 只检查 wrapper 返回 success，不读取和比较输出；
- 用当前 GPU 输出动态生成 expected value 再与自己比较。

推荐 CMake 结构：

```cmake
add_library(qwen3_core STATIC
    src/qwen3_z200_kernels.hip.cpp
    src/qwen3_runtime.cpp
    src/model_loader.cpp
)

# qwen3_core 在这里统一链接 HIP runtime/hipBLAS，并使用系统 build.sh
# 注入的 gfx906、Release 和编译器设置。

add_executable(metainfer_cpp_server
    src/main.cpp
    src/engine.cpp
    src/http_server.cpp
    src/openai_api.cpp
)
target_link_libraries(metainfer_cpp_server PRIVATE qwen3_core)

add_executable(qwen3_numeric_tests
    tests/qwen3_numeric_tests.cpp
)
target_link_libraries(qwen3_numeric_tests PRIVATE qwen3_core)

enable_testing()
add_test(NAME qwen3_numeric COMMAND qwen3_numeric_tests)
```

`qwen3_numeric_tests` 不能使用 `EXCLUDE_FROM_ALL`，因为 system-owned `build.sh` 只执行默认的：

```bash
cmake --build "$ROOT/build"
```

## 3. 测试程序 CLI 和返回契约

最低要求：

```bash
./build/qwen3_numeric_tests
```

建议同时支持：

```bash
./build/qwen3_numeric_tests --list
./build/qwen3_numeric_tests --filter q8_linear
./build/qwen3_numeric_tests --report /path/to/numeric-test-report.json
```

返回码：

| 返回码 | 含义 | C 阶段分类 |
| ---: | --- | --- |
| `0` | 所有数值测试通过 | 继续启动 server |
| `1` | GPU 输出与 CPU reference 不一致 | `LOGIC_FAIL`，进入同迭代修复 |
| `2` | CLI、缺少测试 binary、内部测试配置错误 | `LOGIC_FAIL`，生成物不满足契约 |
| `3` | HIP device/stream/hipBLAS 初始化失败 | `INFRA_FAIL` |

stdout 应保持可读：

```text
[PASS] cast_fp32_to_fp16       elements=257 exact=true
[PASS] dequant_q8_0            elements=64 exact=true
[PASS] q8_linear               M=2 N=3 K=32 max_abs=0.0021 rel_l2=0.00018
[PASS] rms_norm                rows=2 dim=4096 max_abs=0.000006
[PASS] rope_neox               T=2 H=4 D=128 start_pos=3 max_abs=0.000011
[PASS] kv_cache_write          T=3 start_pos=2 exact=true
[PASS] prefill_gqa_attention   T=3 Nq=4 Nkv=2 D=128 max_abs=0.000083
[PASS] decode_gqa_attention    pos=2 Nq=4 Nkv=2 D=128 max_abs=0.000091
[PASS] swiglu_tail             elements=257 max_abs=0.000004
[PASS] greedy_tie              expected=17 actual=17
SUMMARY passed=10 failed=0 skipped=0 elapsed_ms=...
```

stderr 只打印失败详情和 HIP/hipBLAS 错误。不要打印数千个元素。

## 4. 测试 harness 的公共规则

### 4.1 初始化和同步

测试进程只创建：

```text
one HIP device
one HIP stream
one hipBLAS handle
small reusable device buffers
```

并执行：

```cpp
hipSetDevice(0);
hipStreamCreate(&stream);
hipblasCreate(&handle);
hipblasSetStream(handle, stream);
```

每个测试的正确顺序：

```text
prepare deterministic host input/reference
  -> allocate/reuse small device buffers
  -> H2D copy on the test stream
  -> call the real exported wrapper
  -> check immediate hipError_t/hipblasStatus_t
  -> D2H copy on the same stream
  -> hipStreamSynchronize once at the test boundary
  -> finite check
  -> numeric comparison
  -> record metrics
```

不要在一个 kernel 尚未完成时从 host 读取输出。也不要依赖 `hipGetLastError()` 捕获异步执行错误。

### 4.2 确定性输入

优先使用显式公式：

```cpp
x[i] = 0.01f * static_cast<float>((i * 17) % 101 - 50);
w[i] = 0.5f + 0.001f * static_cast<float>(i % 37);
```

若使用伪随机数，只允许固定 seed，例如：

```cpp
std::mt19937 rng(0x514E3308u);
```

禁止 `std::random_device`、当前时间 seed 或未初始化 device memory。

输入必须包含：

- 正数、负数、零和小数；
- 不同 row/head/token 的不同值；
- 非 block 整数倍的元素总数，用于检测尾部；
- 非零 `start_pos`，用于检测 position/stride；
- 相同最大 logits，用于检测 greedy tie-break。

### 4.3 比较指标

对 FP32 数组至少计算：

```cpp
max_abs = max(abs(actual[i] - reference[i]));

max_rel = max(
    abs(actual[i] - reference[i])
    / max(abs(reference[i]), 1e-6f));

rel_l2 = sqrt(sum((actual[i] - reference[i])^2))
       / max(sqrt(sum(reference[i]^2)), 1e-12);
```

每次比较前必须检查：

```text
actual has no NaN
actual has no Inf
reference has no NaN/Inf
element count matches
```

失败时记录最大误差元素：

```text
test=q8_linear
shape=M2_N3_K32
index=4
reference=-1.28452
actual=-1.30117
abs_error=0.01665
max_abs_limit=0.01
```

只有 FP16 cast/dequant 等明确要求 bit-exact 的测试可以比较 half bits。普通 FP32 运算禁止用 `memcmp`。

### 4.4 初始容差

| 测试 | `atol` | `rtol` / 额外要求 |
| --- | ---: | ---: |
| FP32 -> FP16 cast | exact half bits | 无 |
| Q8_0 -> FP16 dequant | exact half bits | 无 |
| Q8_0 embedding -> FP32 | `1e-6` | `1e-6` |
| Q8 linear | `1e-2` | `1e-2`，并检查 `rel_l2 <= 1e-3` |
| hidden/per-head RMSNorm | `1e-4` | `1e-4` |
| NEOX RoPE | `1e-4` | `1e-4` |
| KV cache write | exact FP32 values | 未写区域保持 canary |
| prefill/decode attention | `1e-3` | `1e-3` |
| SwiGLU | `1e-4` | `1e-4` |
| add/add_inplace | `1e-6` | `1e-6` |
| greedy argmax | token id exact | 相同值取更小 id |

这些是 bring-up 上限，不是永远固定的宽松阈值。目标机建立稳定 baseline 后应收紧，不能为了让错误实现通过而扩大容差。

## 5. 必须实现的快速测试

### 5.1 FP32 到 FP16 cast

入口：

```cpp
qwen3_z200_launch_cast_fp32_to_fp16(...);
```

测试：

```text
n_elements = 257
```

输入覆盖：

```text
0
-0
1
-1
small fractions
large finite FP16 values
positive and negative non-integers
```

CPU 使用与 IEEE FP16 一致的转换函数。比较每个 half 的 16-bit payload，并确认第 257 个尾部元素被写入。

### 5.2 Q8_0 单 block 和多 block 反量化

入口：

```cpp
qwen3_z200_launch_dequant_q8_0_to_fp16(...);
```

构造两个 block：

```cpp
BlockQ8_0 blocks[2];
blocks[0].d = float_to_half(0.25f);
blocks[1].d = float_to_half(0.03125f);

for (int i = 0; i < 32; ++i) {
    blocks[0].qs[i] = static_cast<int8_t>(i - 16);
    blocks[1].qs[i] = static_cast<int8_t>(15 - i);
}
```

CPU reference：

```cpp
for (size_t i = 0; i < 64; ++i) {
    const BlockQ8_0& b = blocks[i / 32];
    const float v = half_to_float(b.d)
                  * static_cast<float>(b.qs[i % 32]);
    expected[i] = float_to_half(v);
}
```

检查：

- `sizeof(BlockQ8_0) == 34`；
- 64 个 half bit-exact；
- `n_elements=63` 返回非法参数；
- null pointer 和零元素返回非法参数。

### 5.3 Q8_0 Embedding

入口：

```cpp
qwen3_z200_launch_embedding_lookup_q8_0(...);
```

使用：

```text
vocab_size = 4
hidden_dim = 64
token_ids = [3, 1, 3]
```

每个 token row 使用不同 scale/quant pattern。CPU reference 按：

```text
block_idx = token_id * (hidden_dim / 32) + dim / 32
value = fp16_scale * int8_quant
```

检查：

- 重复 token 3 的两个输出 row 完全一致；
- token 1 与 token 3 不同；
- row-major `[T,H]` 顺序正确；
- runtime 层应提前拒绝非法 token id。当前 kernel 的零填充行为不能替代 runtime 边界检查。

### 5.4 Q8 Linear + hipBLAS

入口：

```cpp
qwen3_z200_q8_linear_fp32(...);
```

固定 shape：

```text
M = 2
N = 3
K = 32
X = [2,32] FP32
W = [3,32] Q8_0
Y = [2,3] FP32
```

CPU reference 必须复现真实 dtype 边界：

```cpp
x_half = fp32_to_fp16(x);
w_half = q8_0_to_fp16(weight);

for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += half_to_float(x_half[m*K + k])
                 * half_to_float(w_half[n*K + k]);
        }
        expected[m*N + n] = acc;
    }
}
```

不能用原始 FP32 `X` 或未 round 到 FP16 的反量化权重计算 reference，否则会把 dtype 差异误判为 GEMM 错误。

额外检查：

- `K=31` 被拒绝；
- `x_workspace_elements < M*K` 被拒绝；
- `weight_workspace_elements < N*K` 被拒绝；
- 输出布局确实是 row-major `[M,N]`，不是转置的 `[N,M]`；
- `beta=0`，预填充的输出 canary 不应参与结果。

### 5.5 Hidden RMSNorm

入口：

```cpp
qwen3_z200_launch_rms_norm(...);
```

固定 shape：

```text
rows = 2
dim = 4096
eps = 1e-6
```

CPU reference：

```cpp
double sum_sq = 0.0;
for (int d = 0; d < dim; ++d) {
    const double v = static_cast<double>(x[row*dim + d]);
    sum_sq += v * v;
}
const float inv = 1.0f / sqrtf(
    static_cast<float>(sum_sq / dim) + eps);
expected[row*dim + d] = x[row*dim + d] * inv * weight[d];
```

CPU 使用 double 累加只是为了得到稳定 reference；比较仍允许 GPU FP32 reduction 顺序带来的小误差。两个 row 必须使用不同输入。

### 5.6 Q/K per-head RMSNorm

入口：

```cpp
qwen3_z200_launch_per_head_rms_norm(...);
```

至少两组：

```text
Q-like: T=2, n_heads=4, head_dim=128, row_stride=512
K-like: T=2, n_heads=2, head_dim=128, row_stride=256
```

每个 `(token,head)` 单独计算 RMS，weight shape 是 `[128]`。不同 token/head 使用不同值，防止错误实现把整个 row 或所有 heads 一起归一化仍然通过。

### 5.7 Qwen3 NEOX RoPE

入口：

```cpp
qwen3_z200_launch_rope(...);
```

使用：

```text
T = 2
n_heads = 4
head_dim = 128
row_stride = 512
start_pos = 3
max_position = 16
rope_mode = 1  // ROPE_NEOX
```

CPU reference：

```cpp
const int half = head_dim / 2;
const int i0 = pair;
const int i1 = pair + half;
const int pos = start_pos + token;

expected[i0] = x0 * cos[pos][pair] - x1 * sin[pos][pair];
expected[i1] = x0 * sin[pos][pair] + x1 * cos[pos][pair];
```

必须检查：

- NEOX 是前后半区配对，不是 `(0,1),(2,3)`；
- token 0 使用 position 3，token 1 使用 position 4；
- Q-like `row_stride=512` 和 K-like stride 都测试；
- `start_pos=0` 再增加一个基础 case，但不能只有 position 0。

### 5.8 KV cache write

入口：

```cpp
qwen3_z200_launch_kv_cache_write(...);
```

固定 shape：

```text
T = 3
n_kv_heads = 2
head_dim = 8
max_seq_len = 8
start_pos = 2
```

构造：

```cpp
src[token][head][dim] = token * 10000 + head * 100 + dim;
```

cache 初始化为可识别 canary。调用后要求：

```text
cache[2] == src[0]
cache[3] == src[1]
cache[4] == src[2]
cache[0], cache[1], cache[5..7] remain canary
K and V use independent patterns and are not swapped
```

### 5.9 Prefill 和 Decode GQA Attention

入口：

```cpp
qwen3_z200_launch_prefill_gqa_attention(...);
qwen3_z200_launch_decode_gqa_attention(...);
```

基础 CPU-reference case：

```text
T = 3
n_heads = 4
n_kv_heads = 2
head_dim = 128
max_seq_len = 8
scale = 1 / sqrt(128)
```

GQA 映射：

```cpp
q_per_kv = n_heads / n_kv_heads; // 2
kv_head = q_head / q_per_kv;
```

对每个 query token/head：

```cpp
for (int pos = 0; pos <= current_pos; ++pos) {
    score[pos] = dot(q, k_cache[pos][kv_head]) * scale;
}
prob = stable_softmax(score);
expected = sum(prob[pos] * v_cache[pos][kv_head]);
```

CPU softmax 必须先减最大值。检查：

- prefill token `t` 只读取 `0..t`；
- q heads `0,1` 映射 KV head 0，q heads `2,3` 映射 KV head 1；
- prefill 最后 token 的输出与相同 Q/K/V、`current_pos=2` 的 decode 输出接近；
- decode scores workspace 至少是 `[n_heads,max_seq_len]`；
- 再增加一个 contract-shaped case：`n_heads=32,n_kv_heads=8,head_dim=128,T=2`，防止仅小 shape 通过而真实配置硬编码错误。

### 5.10 SwiGLU 和 residual

入口：

```cpp
qwen3_z200_launch_swiglu(...);
qwen3_z200_launch_add(...);
qwen3_z200_launch_add_inplace(...);
```

使用 `n_elements=257`，CPU：

```cpp
silu = gate / (1.0f + expf(-gate));
expected_swiglu = silu * up;
expected_add = a + b;
```

第 257 个元素必须参与计算。`add_inplace` 调用前保留输入副本，不能用已经修改的 `dst` 计算 expected。

### 5.11 Greedy argmax

入口：

```cpp
qwen3_z200_launch_greedy_sample(...);
```

使用：

```text
vocab_size = 513
all logits initialized to a deterministic finite pattern
logits[17] = 10
logits[300] = 10
```

期望 token 是 17。再测试最大值位于：

- id 0；
- id 512；
- 单一负数最大值；
- vocab size 1。

`out_token` 是初始化时分配的小 device buffer，不要在每个 case 中重复创建复杂 sampler 状态。

## 6. CPU reference 的独立性

CPU reference 只能实现数学公式和 layout，不能调用被测 GPU wrapper，也不能复用 GPU 输出作为中间 reference。

允许共享：

- `BlockQ8_0` 的明确 ABI 定义；
- shape 常量；
- 只读输入数据；
- 通用的 half bit conversion helper。

禁止共享：

- GPU kernel 的实现循环；
- device 输出；
- 被测 wrapper 的索引 helper；
- 为了迎合 GPU 错误而复制同一个错误 stride。

建议 CPU reference 使用清晰的多维索引，而 GPU 输入仍是线性内存。例如 KV reference 写成：

```cpp
auto cache_index = [=](int pos, int head, int dim) {
    return (static_cast<size_t>(pos) * n_kv_heads + head) * head_dim + dim;
};
```

测试输入的不同 row/head/token 必须有不同值，才能让错误 stride 显现。

## 7. 数值测试报告

建议 JSON：

```json
{
  "schema_version": 1,
  "suite": "qwen3-z200-c0.1",
  "device": "Z200SM_80",
  "arch": "gfx906",
  "passed": false,
  "elapsed_ms": 24.8,
  "summary": {
    "passed": 9,
    "failed": 1,
    "skipped": 0
  },
  "cases": [
    {
      "name": "q8_linear",
      "status": "fail",
      "shape": "M=2,N=3,K=32",
      "max_abs": 0.01665,
      "max_rel": 0.01296,
      "rel_l2": 0.00421,
      "worst_index": 4,
      "reference": -1.28452,
      "actual": -1.30117,
      "message": "max_abs 0.01665 exceeds 0.01"
    }
  ]
}
```

报告中禁止写入模型权重或巨大 tensor dump。C0.1 不加载模型，报告通常应小于几百 KiB。

## 8. B 阶段的上下游对接

B 实现 Agent 的顺序：

```text
read 01/03/04/06/08 contracts
  -> create qwen3_core
  -> create server target
  -> create qwen3_numeric_tests target linked to qwen3_core
  -> implement independent CPU references
  -> run bash build.sh
  -> run ./build/qwen3_numeric_tests
  -> only after C0.1 passes, boot serve.sh for HTTP smoke test
```

B 不得修改 system-owned `build.sh` 来绕过测试 target。若 numeric binary 没有被默认 build，应该修 `CMakeLists.txt`。

B 完成前最低自测：

```bash
bash build.sh
./build/qwen3_numeric_tests \
  --report numeric-test-report.local.json
./build/metainfer_cpp_server --help
```

如果快速测试失败，B 应先修具体 operator，不要启动并加载 8B 模型浪费时间。

## 9. C 阶段的准确插入点

当前 immutable C oracle 的主路径是：

```text
materialize_hardware_binding
  -> _run_build_check(build.sh)
  -> _start_server(serve.sh)
  -> _wait_healthy
  -> HTTP cases + judge
```

当前代码尚未自动运行 C0.1。接入后应变为：

```text
materialize_hardware_binding
  -> _run_build_check(build.sh)
  -> preflight_gpu(label="c0.1-numeric")
  -> _run_numeric_check(build/qwen3_numeric_tests)   # new C0.1
  -> _start_server(serve.sh)
  -> _wait_healthy
  -> HTTP cases + judge
```

建议在：

```text
MetaInfer/metainfer/tasks/gen_cpp_infer_framework/orchestrator/oracles/correctness.py
```

中 `_run_build_check()` 成功之后、`_start_server()` 之前调用：

```python
numeric_bin = iter_dir / "build" / "qwen3_numeric_tests"
numeric_report = report_dir / "numeric-test-report.json"

preflight = preflight_gpu(label="c0.1-numeric")
if preflight.kill_errors:
    raise NumericTestInfraError(
        f"GPU preflight failed: {preflight.kill_errors}"
    )

ok, numeric_kind, numeric_err = _run_numeric_check(
    numeric_bin,
    numeric_report,
    iter_dir=iter_dir,
    report_dir=report_dir,
    extra_env=hardware_env,
    timeout_s=120,
)
if not ok:
    if numeric_kind == "infra":
        raise NumericTestInfraError(
            numeric_err or "C0.1 numeric test infrastructure failed"
        )
    return self._fail(
        report_dir,
        numeric_err or "C0.1 numeric tests failed",
    )
```

`_run_numeric_check()` 建议执行：

```python
stdout_path = report_dir / "numeric-test.stdout.log"
stderr_path = report_dir / "numeric-test.stderr.log"
with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
    completed = subprocess.run(
        [str(numeric_bin), "--report", str(numeric_report)],
        cwd=str(iter_dir),
        env=env,
        stdout=stdout_fp,
        stderr=stderr_fp,
        timeout=120,
        check=False,
    )
```

C 必须：

- 每次 C attempt 都重新运行 numeric binary；
- 检查 binary 存在且是普通文件；
- 继承 hardware profile 的执行环境；
- 保留 stdout、stderr 和 JSON report 到本 iteration 的 logs；
- numeric fail 时不调用 `_start_server()`，不加载 GGUF；
- GPU preflight/device 初始化/进程启动失败标记 infra failure；
- 数值 mismatch、缺少 binary、非法 CLI/报告、程序崩溃和测试超时标记 logic failure，交给 C repair Agent；
- 不允许实现 Agent编辑 immutable oracle 或伪造 report 代替执行 binary。

当前 `pipeline._run_oracle_once()` 会把普通 `OracleResult(passed=false)` 映射为 `LOGIC_FAIL`，把 oracle 抛出的异常映射为 `INFRA_FAIL`。因此上面的 `NumericTestInfraError` 必须仅用于真实环境/设备失败；kernel hang、测试程序崩溃、binary 缺失和数值不匹配都应返回普通失败，进入修复流程，不能借异常绕过修复。

## 10. C repair Agent 的输入

C0.1 失败传给 repair Agent 的摘要至少包含：

```text
C0.1 numeric test failed
binary: <iter_dir>/build/qwen3_numeric_tests
case: q8_linear
shape: M=2,N=3,K=32
max_abs: 0.01665 limit=0.01
rel_l2: 0.00421 limit=0.001
worst_index: 4
reference: -1.28452
actual: -1.30117
stdout: <logs>/numeric-test.stdout.log
stderr: <logs>/numeric-test.stderr.log
report: <logs>/numeric-test-report.json
```

推荐定位映射：

| 失败 case | 优先检查 |
| --- | --- |
| cast | FP16 conversion、元素数、尾部 grid |
| dequant | `BlockQ8_0` 34-byte ABI、scale、block/offset |
| embedding | token row、blocks-per-row、重复 token |
| q8_linear | FP16 workspace、M/N/K、hipBLAS transpose/ld、row-major output |
| RMSNorm | reduction、eps、weight、row/head stride |
| RoPE | NEOX half split、position、row stride、sin/cos order |
| KV write | `start_pos+token`、KV head、K/V 指针 |
| attention | causal range、GQA mapping、scale、stable softmax |
| SwiGLU | gate/up 配对、尾部元素、`silu(gate)*up` |
| greedy | 全 vocab 扫描、reduction、相同值取较小 id |

## 11. 通过条件

C0.1 只有同时满足以下条件才算通过：

```text
1. build/qwen3_numeric_tests exists and starts.
2. It uses a real HIP device and the selected hardware profile.
3. All required cases execute; no required case is silently skipped.
4. Every wrapper status is success for valid inputs.
5. Invalid-input cases return the expected error.
6. All actual FP32 outputs are finite.
7. Exact tests match exactly.
8. Approximate tests satisfy their declared tolerances.
9. Test process exits 0.
10. JSON summary says failed=0 and agrees with the process exit code.
```

若 JSON 写入失败但所有数值测试通过，首版可以把它视为 reporting infra failure；不能仅凭 exit code 0 丢失所有诊断信息后继续。

## 12. 明确不做的内容

C0.1 不做：

- 不加载 GGUF；
- 不比较真实 Qwen3-8B `[151936]` logits；
- 不跑 36 层完整模型；
- 不比较生成文本；
- 不测试 tokenizer/chat template；
- 不启动 HTTP server；
- 不测吞吐或延迟优化；
- 不使用 llama.cpp、vLLM 或 GGML 作为运行时依赖；
- 不修改 `build.sh`、hardware profile 或 profiler contract。

这些边界使 C0.1 保持快速、确定、易定位。完整模型和 HTTP 验收继续由 runtime contract 与现有 C oracle 的后续阶段负责。

## 13. 最小实现顺序

实现 Agent 建议按以下顺序增加 case：

1. test harness、HIP stream、report、compare helper；
2. cast + Q8_0 dequant；
3. Q8 embedding；
4. Q8 linear + hipBLAS；
5. hidden/per-head RMSNorm；
6. NEOX RoPE；
7. KV cache write；
8. prefill/decode GQA attention；
9. SwiGLU + add；
10. greedy argmax；
11. CMake default target 和 `ctest` 注册；
12. B 本地运行 `bash build.sh && ./build/qwen3_numeric_tests`；
13. MetaInfer C oracle 接入 `_run_numeric_check()`。

不要先写 HTTP 测试来替代这里的数值比较。C0.1 的价值正是：在加载 8B 模型之前，用一个明确 case 和一个明确误差位置把底层算子问题暴露出来。
