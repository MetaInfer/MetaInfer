# 原生推理引擎契约（Native Engine Contract）

> 权威级别：Scheduler、Request Lifecycle、ModelRunner 边界的强制契约。

## 1. 请求状态机

```text
Waiting -> Prefill -> Decode -> Finished
                   \-> Cancelled
Waiting/Prefill/Decode -> Failed
Waiting -> Rejected
```

状态转换只能由 Engine Control Plane 提交。HTTP 线程不能直接修改状态。请求在进入终态之前拥有 Token State 和逻辑 Block Table；终态清理必须幂等。

```cpp
enum class RequestStatus : std::uint8_t {
  kWaiting,
  kPrefill,
  kDecode,
  kFinished,
  kCancelled,
  kRejected,
  kFailed,
};

struct SamplingParams {
  std::uint32_t max_new_tokens = 1;
  float temperature = 0.0f;
  float top_p = 1.0f;
  std::uint64_t seed = 0;
};

struct RequestState {
  std::uint64_t id = 0;
  std::vector<std::int32_t> prompt_tokens;
  std::vector<std::int32_t> output_tokens;
  SamplingParams sampling;
  RequestStatus status = RequestStatus::kWaiting;
  std::uint32_t kv_length = 0;
  BlockTable blocks;
};
```

`RequestState` 的具体字段可以扩展，但不得依赖悬空的 HTTP Request 引用。

## 2. StepPlan 契约

Scheduler 每一步生成一个不可变的执行计划：

```cpp
struct SequenceStep {
  std::uint64_t request_id;
  std::uint32_t token_offset;
  std::uint32_t token_count;
  std::uint32_t logical_position;
  Span<const std::int32_t> block_table;
};

struct StepPlan {
  std::uint64_t step_id;
  StepKind kind;  // kPrefill / kDecode / kMixed（若明确支持）
  std::vector<SequenceStep> sequences;
  std::uint32_t total_tokens;
};
```

`Span<T>` 是项目自定义的 C++17 非拥有连续 View。`StepPlan` 创建完成后，在 ModelRunner 返回之前不得改变序列顺序、位置和 Block Table。

## 3. Prefill 契约

- **ENG-001**：Prefill 消费本次计划中的全部有效 Prompt Token；Padding 不计入逻辑长度。
- **ENG-002**：每层 K/V 只写入一次，写入位置由逻辑位置和 Block Table 决定。
- **ENG-003**：Prefill 结束后只为需要的最后有效位置产生下一 Token Logits。
- **ENG-004**：所有层成功完成之前，不得提交新的 `kv_length`。

## 4. Decode 契约

- **ENG-005**：普通 Decode 中，每个 Active Sequence 每步最多提交一个新 Token。
- **ENG-006**：RoPE Position 是逻辑 Token Position，不是 Batch Index 或物理 Block Offset。
- **ENG-007**：Decode 必须直接访问 Paged KV，禁止每步拼接完整连续 KV 临时副本。
- **ENG-008**：采样结果只提交一次；提交后立即检查 EOS、Stop 和 Max Tokens。

推荐使用显式的提交阶段：

```cpp
Result<void> Engine::CommitStep(const StepPlan& plan,
                                const StepOutput& output) {
  RETURN_IF_ERROR(output.backend_status);
  RETURN_IF_ERROR(ValidateStepOutput(plan, output));
  for (std::size_t i = 0; i < plan.sequences.size(); ++i) {
    RequestState& request = FindOwnedRequest(plan.sequences[i].request_id);
    request.kv_length += plan.sequences[i].token_count;
    CommitSampledToken(request, output.sampled_tokens[i]);
    AdvanceTerminalState(request);
  }
  return {};
}
```

以上代码是事务边界示意，错误类型和宏需要项目统一定义。

## 5. 调度与资源规则

调度输入至少包括：

- 最大 Active Sequence 数；
- Prefill Token Budget、Decode Sequence Budget；
- 每个 TP Rank 的可用 KV Block；
- Context/Generation Limit；
- Request Priority/Arrival Order；
- Cancellation State；
- Backend Workspace 上限。

Scheduler 不得把多张设备显存相加后当作单一分配池。跨 Rank 分配必须使用所有 Rank 可满足的最小容量。

## 6. 并发规则

推荐单一 Engine Thread 拥有 Scheduler 和请求状态，其他线程通过有界队列发送命令：

```cpp
using EngineCommand = std::variant<SubmitCommand,
                                   CancelCommand,
                                   ShutdownCommand>;

BoundedQueue<EngineCommand> command_queue;
```

如果采用多锁设计，必须记录 Lock Order，并满足：

- 不得持有 Mutex 等待 Collective；
- 不得持有 Engine Lock 写网络响应；
- 不得在 Signal Handler 中获取普通 Mutex；
- 慢客户端不能阻塞模型执行线程。

## 7. 失败行为

输入错误只影响对应请求；设备错误、Kernel 错误或 Collective 错误在可能造成 Rank 状态分叉时必须触发协调关闭。部分初始化失败必须利用 RAII 逆序释放资源，不得留下后台线程或 GPU Context。
