# Continuous Batching 与请求生命周期

先读：`00_contracts/engine_contracts.md`和`01_framework_design/07_request_lifecycle.md`。

第一版推荐一个 Engine Thread 独占 Waiting/Running Queue、Request State、KV 决策和 Model Step 提交。HTTP 线程只发送命令和接收 Event，避免 Scheduler 上出现复杂锁。

## 1. Engine Command 与 Event

```cpp
struct SubmitCommand {
  GenerateRequest request;
  std::shared_ptr<EventSink> sink;
};

struct CancelCommand {
  RequestId request_id;
  CancelReason reason;
};

struct ShutdownCommand {
  std::chrono::milliseconds deadline;
};

using EngineCommand = std::variant<SubmitCommand,
                                   CancelCommand,
                                   ShutdownCommand>;

struct TokenEvent {
  RequestId request_id;
  std::int32_t token_id;
  std::string text_delta;
};

using EngineEvent = std::variant<TokenEvent, FinishedEvent, ErrorEvent>;
```

Command Queue 和每客户端 Event Queue 都必须有界。

## 2. Scheduler 配置

```cpp
struct SchedulerLimits {
  std::uint32_t max_running_sequences;
  std::uint32_t max_waiting_requests;
  std::uint32_t max_prefill_tokens_per_step;
  std::uint32_t max_decode_sequences_per_step;
  std::uint32_t max_context_tokens;
};

class Scheduler {
 public:
  Result<RequestId> Admit(GenerateRequest request,
                          std::shared_ptr<EventSink> sink);
  Status Cancel(RequestId id, CancelReason reason);
  Result<StepPlan> BuildNextStep(const SchedulerResources& resources);
  Status CommitStep(const StepPlan& plan, const StepOutput& output);
  Status FailStep(const StepPlan& plan, const Status& error);
};
```

`BuildNextStep` 只创建不可变计划，不执行模型或写网络。

## 3. Admission

```cpp
Result<RequestId> Scheduler::Admit(GenerateRequest request,
                                   std::shared_ptr<EventSink> sink) {
  RETURN_IF_ERROR(ValidateGenerateRequest(request, limits_));
  if (waiting_.size() >= limits_.max_waiting_requests) {
    return ResourceExhausted("request queue is full");
  }

  RequestState state;
  state.id = next_request_id_++;
  state.prompt_tokens = std::move(request.prompt_tokens);
  state.sampling = request.sampling;
  state.status = RequestStatus::kWaiting;
  state.sink = std::move(sink);

  const RequestId id = state.id;
  requests_.emplace(id, std::move(state));
  waiting_.push_back(id);
  return id;
}
```

Prompt Token 数超过 Context 上限、Max Tokens 非法、Queue 满时应在 Admission 阶段拒绝，不能永久留在 Waiting。

## 4. 基线调度策略

可靠的第一版：

1. Waiting 按 FIFO 选择可满足 KV/Token Budget 的 Prefill；
2. Running 中所有可运行请求各安排一个 Decode Token，受 Decode Sequence Budget 限制；
3. 相同 Priority 按 Arrival/Request ID 确定性排序；
4. 资源不足时保留 Waiting 或明确拒绝，禁止部分分配。

```cpp
Result<StepPlan> Scheduler::BuildNextStep(
    const SchedulerResources& resources) {
  StepPlan plan;
  plan.step_id = next_step_id_++;

  std::uint32_t prefill_budget = limits_.max_prefill_tokens_per_step;
  for (RequestId id : waiting_) {
    RequestState& request = requests_.at(id);
    const std::uint32_t tokens = request.prompt_tokens.size();
    const std::uint32_t blocks = RequiredBlocks(tokens, resources.block_size);
    if (tokens > prefill_budget || blocks > resources.free_blocks) continue;
    plan.sequences.push_back(BuildPrefillSequence(request));
    prefill_budget -= tokens;
    if (plan.sequences.size() >= limits_.max_running_sequences) break;
  }

  std::uint32_t decode_budget = limits_.max_decode_sequences_per_step;
  for (RequestId id : running_) {
    if (decode_budget == 0) break;
    RequestState& request = requests_.at(id);
    if (request.cancel_requested) continue;
    plan.sequences.push_back(BuildDecodeSequence(request));
    --decode_budget;
  }

  if (plan.sequences.empty()) return Unavailable("no runnable request");
  RETURN_IF_ERROR(ValidateStepPlan(plan, resources));
  return plan;
}
```

若 Backend 暂不支持 Mixed Prefill/Decode，计划必须只包含一种 `StepKind`；不能把两种 Shape 强行交给同一 Kernel。

## 5. 执行与提交

```cpp
void EngineLoop::RunOneIteration() {
  DrainCommands();

  auto plan = scheduler_.BuildNextStep(CurrentResources());
  if (!plan.ok()) {
    WaitForCommandOrResource();
    return;
  }

  auto kv_transaction = kv_.Prepare(plan.value());
  if (!kv_transaction.ok()) {
    scheduler_.FailStep(plan.value(), kv_transaction.status());
    return;
  }

  auto output = runner_.Execute(plan.value(), kv_transaction.value().views(), stream_);
  if (!output.ok()) {
    kv_transaction.value().Rollback();
    scheduler_.FailStep(plan.value(), output.status());
    return;
  }

  Status commit = kv_transaction.value().Commit();
  if (commit.ok()) commit = scheduler_.CommitStep(plan.value(), output.value());
  if (!commit.ok()) BeginCoordinatedFailure(commit);
}
```

Model Step 失败时不得部分推进 Token/KV。TP 模式中 Commit 前需确认所有 Rank 成功。

## 6. CommitStep

```cpp
Status Scheduler::CommitStep(const StepPlan& plan,
                             const StepOutput& output) {
  RETURN_IF_ERROR(ValidateStepOutput(plan, output));
  for (std::size_t i = 0; i < plan.sequences.size(); ++i) {
    RequestState& request = requests_.at(plan.sequences[i].request_id);
    request.kv_length += plan.sequences[i].token_count;
    request.output_tokens.push_back(output.sampled_tokens[i]);

    if (ShouldFinish(request, output.sampled_tokens[i])) {
      TransitionToFinished(request);
      EmitFinished(request);
      ReleaseRequestResources(request);
    } else {
      request.status = RequestStatus::kDecode;
      EnsureInRunningQueue(request.id);
    }
  }
  return Status::Ok();
}
```

Prefill 产生的第一个 Sampled Token 与写入的 Prompt KV 长度要区分，避免 Off-by-one。

## 7. Backpressure

有界对象：

```text
HTTP Body Bytes
Waiting Requests
Prompt Tokens
Active Sequences
Per-client Pending Events/Bytes
Workspace Bytes
KV Blocks
```

Event Sink 示例：

```cpp
Status BoundedEventSink::Push(EngineEvent event) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (closed_) return Cancelled("client disconnected");
  if (pending_bytes_ + EstimateBytes(event) > max_pending_bytes_) {
    return ResourceExhausted("client output buffer is full");
  }
  pending_bytes_ += EstimateBytes(event);
  events_.push_back(std::move(event));
  cv_.notify_one();
  return Status::Ok();
}
```

慢客户端不得卡住 Engine Thread；Queue 满时应取消/错误或按有界策略等待。

## 8. Cancellation

```cpp
Status Scheduler::Cancel(RequestId id, CancelReason reason) {
  auto it = requests_.find(id);
  if (it == requests_.end()) return Status::Ok();  // 幂等
  RequestState& request = it->second;
  if (IsTerminal(request.status)) return Status::Ok();
  request.cancel_requested = true;
  request.cancel_reason = reason;
  return Status::Ok();
}
```

Waiting 请求可立即移除。正在执行的请求在当前 Step 安全边界生效，之后禁止发送 Token，并释放 KV。客户端 Disconnect 走同一路径。

## 9. 关闭流程

```text
Signal Thread 写入 ShutdownCommand
-> Stop Admission
-> 等待当前 Step 或触发有界 Cancel
-> Flush Finished/Error Event
-> Free KV/Workspace
-> Shutdown TP Ranks
-> Flush Profile
-> Join Owned Threads
```

Signal Handler 本身不能调用 `Join`、普通 Mutex 或复杂日志。

## 10. Fake Runner 测试

```cpp
class FakeModelRunner final : public ModelRunner {
 public:
  Result<StepOutput> Execute(const StepPlan& plan,
                             const KvBatchView&,
                             BackendStream) override {
    if (fail_step_id_ == plan.step_id) return BackendError("injected failure");
    StepOutput output;
    for (const auto& sequence : plan.sequences) {
      output.sampled_tokens.push_back(DeterministicToken(sequence.request_id));
    }
    return output;
  }
};
```

Host-only 测试覆盖 FIFO、Budget、Queue Full、KV Exhaustion、同时完成、取消、慢消费者、Step Failure、Shutdown；再用真实 Backend 重复关键生命周期。

## 11. 指标

记录 Queue Delay、Prefill Time、TTFT、Decode Step、TPOT、Active Sequences、Tokens/Step、KV Occupancy、Reject/Cancel/Failure Count。仅看总 Tokens/s 无法判断 Scheduler 是否饥饿或牺牲尾延迟。
