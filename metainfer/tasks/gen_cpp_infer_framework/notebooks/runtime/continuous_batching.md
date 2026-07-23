# Continuous Batching：C++/HIP 推理服务生成指南

> 本文用于指导 Agent 从零实现 Continuous Batching，不依赖任何现有推理框架源码。
> 标准文件名是 `continuous_batching.md`，标准术语是 **Continuous Batching**。
> 文中的“必须”表示正确性或生命周期契约，“建议”表示适合第一版实现的
> 工程选择。
>
> **适用边界：** Continuous Batching 与 Paged KV Cache 是独立能力。本文的调度、
> packed row、状态提交和并发证据适用于所有 Continuous Batching 任务；只有冻结能力同时
> 包含 `paged_kv_cache` 时，才启用本文标注为“Paged 组合”的 block table 和
> Reserve/Commit/Rollback 规则。Continuous-only 使用每 sequence 独立的 contiguous
> FP16 KV allocation，不能读取 Paged 组合专属接口。

相关主题：[Paged KV Cache](paged_kv_cache.md) ·
[Tensor Parallel](../distributed/tensor_parallel.md)

## 1. 目标与范围

Continuous Batching 要解决的是在线生成服务中的动态请求问题：请求到达时间、Prompt
长度、生成长度和结束时间都不同，GPU 每完成一个调度 step，都应允许请求加入、退出或
改变执行阶段。

第一版应实现：

- 有界请求队列和明确的 admission backpressure；
- 单一逻辑 Scheduler 所有者；
- 每个 tick 动态重建 batch membership；
- 多请求 Packed Decode；
- 多请求 Ragged Prefill；
- Chunked Prefill 和基本公平性；
- token budget、sequence budget 与 KV budget 联合约束；
- 请求级 Sampling、停止条件、取消和错误隔离；
- 与当前 KV backend 的有界容量事务；Paged 组合使用 block 级
  Reserve/Commit/Rollback；
- JSON/SSE 等协议层与推理层解耦；
- 可证明 GPU 确实合批的 metrics 和测试。

第一版不必实现：

- Prefix Cache；
- CPU KV Swap 和请求抢占；
- Speculative Decoding；
- 多优先级、deadline 或 SLO-aware 调度；
- Prefill/Decode 计算与通信的多 stream 重叠；
- 跨节点 Scheduler；
- 任意模型结构的通用图编译器。

这些能力可以后续扩展，但不能模糊基础版本的状态、资源和提交边界。

## 2. Static Batching 与 Continuous Batching

Static Batching 先收集一组请求，然后固定 batch membership，直到整组执行结束：

```text
batch = [A, B, C]
A finishes ─┐
B finishes  ├─ wait until C finishes
C finishes ─┘
next batch = [D, E]
```

问题是短请求完成后留下空行，而新请求必须等待最长请求。

Continuous Batching 在 token 或 chunk 边界重新选择下一步工作：

```text
tick 0: [A prefill, B prefill]
tick 1: [A decode, B decode, C prefill]
tick 2: [A decode, C prefill]             // B 已完成
tick 3: [A decode, C decode, D prefill]   // D 复用资源
```

“Continuous”描述的是 **batch 成员连续变化**，不是 Kernel 持续运行，也不等同于：

- 多个 HTTP 连接；
- 一个线程处理一个请求；
- 一次把多个完整请求串行跑完；
- 固定 batch 中用 mask 隐藏已结束的行；
- 只把请求放入同一个队列。

必须在执行层观察到“一次模型调用包含多条活跃序列”，才能证明真正合批。

## 3. 三个互补概念

### 3.1 Continuous Batching：选择谁执行

Scheduler 每个 tick 根据请求状态和资源预算选择 sequence。它解决 membership 和公平性。

### 3.2 Chunked Prefill：一条长 Prompt 本轮执行多少

如果一次处理完整长 Prompt，Decode 请求可能在数百毫秒甚至更久内得不到执行。Chunked
Prefill 把 Prompt 切成多个有上限的 chunk，在 chunk 边界重新调度。

### 3.3 Ragged Packed Batch：选中的 token 如何表示

不同 sequence 本轮可以执行不同 token 数。Ragged Packing 只存储真实 token，并用 offsets
和 row mapping 表示边界，避免 padding 或逐请求模型调用。

三者的关系是：

```text
Continuous Batching
  选择 A、B、C
        ↓
Chunked Prefill
  A: 1 个 Decode token
  B: 96 个 Prefill token
  C: 32 个 Prefill token
        ↓
Ragged Packed Batch
  用 T = 1 + 96 + 32 个实际 token 表示本轮工作
```

它们不冲突。只有动态选择而没有 Ragged Packing，执行层仍可能逐请求启动图；只有 Packing
而没有动态调度，batch membership 仍然是静态的。

## 4. 实现前提

Continuous Batching 依赖以下基础能力：

1. 模型能接收 batch size 大于 1 的 Decode 输入；
2. Attention 能为每个 row 使用独立 `past_length` 和 KV 地址；
3. KV Cache 不与固定 batch row 绑定；
4. Runner 能返回每条 sequence 对应的 logits 或 sampled token；
5. 所有动态 workspace 在服务启动后可复用；
6. Host 可以识别 GPU step 完成，才能安全提交状态和回收资源。

Continuous-only 的完成定义允许使用 `[sequence_slot][max_context]` contiguous KV：每个
admitted sequence 独占一个 slot，slot 只在 terminal 且 GPU work 完成后复用。它的容量利用率
低于 Paged KV，但只要满足冻结的 `max_concurrency`、每 row 地址隔离和真实 packed forward，
就是完整的 Continuous Batching 实现。选择 Paged KV 后才把 slot view 替换为 block table。

### 4.1 Continuous-only FP16 KV 合同

冻结 layout：

```text
K/V[layer][slot][position][kv_head][head_dim]  // FP16
slot_count = max_concurrency
slot_capacity = max_context_length
```

Host store 至少提供：

```cpp
struct DenseSequenceKvView {
  SequenceId sequence_id;
  int32_t slot;
  int32_t committed_tokens;
  int32_t capacity;
  uint64_t generation;
};

PrepareBatch(sequence_ids, q_lens);  // all-or-nothing capacity validation
AdvanceBatch(sequence_ids, q_lens);  // only after Forward completion
Release(sequence_id, generation);    // only at a GPU-safe terminal point
```

KV writer 把 FP32 Q/K/V projection output 在写入边界转换为 FP16；Attention 从每 row 的
`slot` 和 `past_length` 读取 FP16 K/V，以 FP32 计算 score、softmax 和 V accumulator。索引为：

```text
((((layer * slot_count + slot) * max_context + position)
   * num_kv_heads + kv_head) * head_dim + d)
```

`packed_sequence_isolation` 必须使用至少两个 slot、不同 past lengths 和不同 K/V pattern，
证明 writer/attention 没有串 slot，并按 FP16 boundary 后的 CPU reference 比较。选择 Paged
后由 `paged_attention` 和联合状态机替换该地址合同。

## 5. 推荐分层

```text
HTTP / RPC producer threads
        │ Submit / Cancel / Disconnect
        ▼
Bounded command queue
        │
        ▼
Single service worker
        │ owns Scheduler and request state
        ▼
Scheduler::BuildNext()
        │ StepPlan
        ▼
BatchAssembler
        │ PackedSequenceBatch (dense view or paged view)
        ▼
ModelRunner::Forward()
        │ per-row ModelResult
        ▼
Scheduler::Apply()
        │ token / terminal events
        ▼
Per-request output queues
        │
        ▼
JSON response / SSE stream
```

职责必须分离：

- 协议层解析输入、鉴权、SSE 和断连，不做 GPU 调度；
- Service Runtime 管理线程、队列、句柄和 shutdown；
- Scheduler 只决定状态转换和下一步工作；
- Batch Assembler 只构造并验证执行元数据；
- Runner 只执行模型，不决定谁先运行；
- Paged KV Cache 管理物理 blocks 和逻辑长度；
- Sampling 模块只按每行配置选 token。

## 6. 线程所有权模型

建议使用一个 Scheduler worker，所有请求状态只由该线程修改。HTTP 线程通过有界 MPSC
command queue 发送命令：

```cpp
struct SubmitCommand {
  uint64_t request_id;
  RequestSpec spec;
};

struct CancelCommand {
  uint64_t request_id;
};

using ServiceCommand = std::variant<SubmitCommand, CancelCommand>;
```

这样有三个好处：

- 不需要在每个 `SequenceState` 上加锁；
- Scheduler、KV Cache 和 batch 生命周期可在同一线程内保持一致；
- Submit/Cancel 与 GPU step 的先后关系有明确序列。

不要让 HTTP worker 直接调用 `Scheduler::Cancel()` 或释放 KV。GPU 完成事件、请求状态和
物理 block 生命周期如果由不同线程无序修改，很容易产生 use-after-free。

## 7. 请求接口

### 7.1 Sampling 配置

```cpp
struct SamplingConfig {
  float temperature = 1.0f;
  int32_t top_k = 0;              // 0 表示不限制
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 0;
};
```

必须验证：

- `temperature >= 0`；
- `top_k >= 0`；
- `0 < top_p <= 1`；
- `repetition_penalty > 0`；
- Greedy 语义要明确，例如 `temperature == 0` 时忽略 `top_k/top_p`。

### 7.2 请求规格

```cpp
struct RequestSpec {
  std::vector<int32_t> prompt_token_ids;
  int64_t max_new_tokens = 1;
  SamplingConfig sampling;
  std::vector<int32_t> stop_token_ids;
  std::vector<std::vector<int32_t>> stop_token_sequences;
  bool ignore_eos = false;
};
```

协议层应在 Submit 前完成 tokenizer 和 chat template。Scheduler 处理 token ids，不依赖
JSON，也不应在调度热路径执行文本分词。

输入验证至少包含：

```text
prompt_token_ids 非空
max_new_tokens > 0
prompt_length <= max_model_len
prompt_length + max_new_tokens <= max_model_len
所有 token id 在 [0, vocab_size)
stop sequence 非空且长度受限
```

### 7.3 请求状态机

```cpp
enum class RequestState {
  kQueued,
  kPrefilling,
  kDecoding,
  kFinished,
  kCancelled,
  kFailed,
};

enum class FinishReason {
  kNone,
  kEos,
  kStop,
  kLength,
  kCancelled,
  kError,
};
```

主路径：

```text
Queued
  ├─ admission failure stays Queued
  ├─ cancel ───────────────────────────────► Cancelled
  └─ admit ─► Prefilling
                ├─ more prompt chunks ─────► Prefilling
                ├─ final prompt chunk ─────► Decoding 或直接 Finished
                ├─ cancel ─────────────────► Cancelled
                └─ error ──────────────────► Failed
                              Decoding
                                ├─ next token ─► Decoding
                                ├─ EOS/stop ───► Finished
                                ├─ max length ─► Finished
                                ├─ cancel ─────► Cancelled
                                └─ error ──────► Failed
```

Terminal state 只能进入一次；进入后必须最终释放 KV 和请求级资源。

## 8. SequenceState 与关键长度

```cpp
using SequenceId = uint64_t;  // 0 is invalid/reserved

struct SequenceState {
  uint64_t request_id = 0;
  SequenceId sequence_id = 0;          // stable KV/runtime identity
  RequestState state = RequestState::kQueued;
  RequestSpec spec;

  int64_t prefill_cursor = 0;
  std::vector<int32_t> generated_tokens;
  std::vector<int32_t> unpublished_tokens;

  bool in_flight = false;
  bool cancel_pending = false;
  uint64_t step_generation = 0;

  FinishReason finish_reason = FinishReason::kNone;
  std::string error_message;
};
```

`request_id` 是协议层句柄；`sequence_id` 是 Scheduler、KV backend 和 TP Ranks 共享的
稳定身份。两者可以数值相同，但不能用 batch row 或可复用 slot 代替 `sequence_id`。

KV 逻辑长度只由当前 `SequenceKvStore` 的 sequence state 持有。Scheduler 需要该值时
通过 `ViewSequence(sequence_id)` 读取，不在 `SequenceState` 中保存第二份可独立修改的
副本。Continuous-only 的 view 包含 slot/base pointer；Paged 组合的 view 包含 block table。
以下文字中的 `kv_committed_tokens` 都表示这个只读查询值。

几个长度不能混用：

```text
prompt_length         = spec.prompt_token_ids.size()
prefill_cursor        = 已成功 Prefill 的 Prompt token 数
kv_committed_tokens   = 已写入并提交的 Prompt + Decode 输入 token 数
generated_tokens.size = 已采样的输出 token 数
```

在 final Prefill 完成并采样首个输出 token `g0` 后：

```text
prefill_cursor        = prompt_length
kv_committed_tokens   = prompt_length
generated_tokens      = [g0]
```

此时 `g0` 已生成，但还没有作为模型输入写入 KV。下一次 Decode 输入是 `g0`，位置是
`prompt_length`；该 step 成功后才有：

```text
kv_committed_tokens   = prompt_length + 1
generated_tokens      = [g0, g1]
```

这是最常见的 off-by-one 来源。不要强行令 `kv_committed_tokens == prompt + generated`。

## 9. 生命周期不变量

任何实现都应持续满足：

1. 一个 request id 最多对应一个非终止 sequence；
2. 一个 sequence 同时最多有一个 in-flight step；
3. `0 <= prefill_cursor <= prompt_length`；
4. Paged KV 查询到的 `committed_tokens == prefill_cursor` 在 Prefill 阶段成立；
5. Decode 阶段 Paged KV `committed_tokens` 只在模型成功后增加；
6. Scheduler 不能读取或调度 terminal sequence；
7. batch 中 request id 不重复；
8. 本 tick token 总数不超过预算；
9. 每个 scheduled row 在执行前已获得足够 KV reservation；
10. KV view 的 generation 与 sequence generation 一致；
11. GPU 仍可能访问资源时不得释放；
12. 每个 terminal request 恰好发布一个 terminal event。

建议在 Debug 构建中每个 `BuildNext/Apply` 后运行 `CheckInvariants()`。

## 10. Scheduler 配置

```cpp
struct SchedulerConfig {
  int64_t max_model_len = 4096;
  int64_t max_active_sequences = 32;
  int64_t max_batched_tokens = 512;
  int64_t max_queue_size = 256;
  int64_t prefill_chunk_tokens = 256;
  int64_t max_stop_sequence_tokens = 16;
  int64_t max_commands_per_tick = 128;
  int64_t max_prefill_sequences_per_tick = 8;
};
```

三个主要约束分别是：

```text
sequence budget: active_sequences <= max_active_sequences
token budget:    sum(scheduled input tokens) <= max_batched_tokens
KV budget:       本 step 所需 blocks 必须能 Reserve
```

`max_batched_tokens` 是一个 tick 的真实输入 token 预算，不是 padded shape，也不是单个请求
最大长度。

为了让所有活跃 Decode 请求每 tick 都能前进一步，建议要求：

```text
max_batched_tokens >= max_active_sequences
```

若还要求最坏情况下至少安排一个最小 Prefill chunk，可进一步要求：

```text
max_batched_tokens >= max_active_sequences - 1 + min_prefill_quantum
```

`min_prefill_quantum` 可以是 1，也可以为便于页和 Kernel shape 而取一个小 block。不要把
配置验证写死到某个模型尺寸。

## 11. 调度输出 StepPlan

```cpp
enum class StepKind {
  kPrefill,
  kDecode,
};

struct TokenRange {
  const int32_t* data = nullptr;
  size_t size = 0;
};

struct ScheduledSequence {
  SequenceId sequence_id;
  StepKind kind;
  TokenRange input_token_ids;  // C++17 non-owning {pointer, size}
  int64_t start_position;
  int64_t past_length;
  bool samples_token;
  uint64_t sequence_generation;
  SamplingConfig sampling;
};

struct StepPlan {
  uint64_t plan_id;
  std::vector<ScheduledSequence> rows;
  int64_t scheduled_tokens;
};
```

`TokenRange` 必须指向在整个 in-flight step 期间保持稳定的 token 存储，并用
`const int32_t* data + size_t size` 表示。C++20 构建可以用 `std::span<const int32_t>`
实现同一 view，默认 C++17 契约不能在公开签名中直接使用 `std::span`。

对于 Prefill：

```text
input_token_ids = prompt[prefill_cursor : prefill_cursor + chunk]
start_position  = prefill_cursor
past_length     = prefill_cursor
samples_token   = chunk reaches end of prompt
```

对于 Decode：

```text
input_token_ids = [generated_tokens.back()]
start_position  = kv_committed_tokens
past_length     = kv_committed_tokens
samples_token   = true
```

`BuildNext()` 只能产生计划和 reservations，不能提前推进 `prefill_cursor`、KV logical length
或 generated history。状态在模型成功并 `Apply()` 后提交。

## 12. 推荐调度算法

一次 tick 建议按以下顺序执行：

```text
1. Drain bounded commands
2. Observe idle cancellations
3. Cleanup terminal sequences whose GPU work is complete
4. Create token/sequence/KV budgets
5. Schedule active Decode rows
6. Schedule active Prefill chunks round-robin
7. Admit queued requests and schedule first chunks
8. Prepare KV capacity for all selected rows
9. Commit the capacity transaction, freeze StepPlan and mark rows in-flight
```

伪代码：

```cpp
Result<StepPlan> Scheduler::BuildNext() {
  CHECK(!has_in_flight_plan_);
  DrainCommands(config_.max_commands_per_tick);
  ApplyIdleCancellations();
  ReclaimTerminalResources();

  TokenBudget budget(config_.max_batched_tokens);
  StepPlan plan{NextPlanId()};

  for (SequenceId id : decode_round_robin_) {
    if (!budget.CanConsume(1)) break;
    TryAppendDecode(id, budget, plan);
  }

  RotateAndAppendPrefill(prefill_round_robin_, budget, plan);
  AdmitAndAppendQueued(budget, plan);

  RETURN_IF_ERROR(ReserveAndCommitCapacityBatch(plan));
  MarkInFlight(plan);
  has_in_flight_plan_ = !plan.rows.empty();
  return plan;
}
```

### 12.1 Decode 优先但不能让 Prefill 永久饥饿

Decode-first 通常能保护正在流式输出请求的 TPOT；但如果 Decode 数量长期耗尽全部 token
预算，新 Prompt 永远无法 Prefill。

可以选择以下策略之一，并把它写成配置契约：

- 保证 `max_batched_tokens > max_active_sequences`，剩余预算给 Prefill；
- 每隔 `N` 个 tick 预留 `prefill_token_quota`；
- 为 Decode 和 Prefill 设置最小/最大配额；
- 使用带 age 的公平权重，等待越久优先级越高。

第一版推荐“所有 Decode 每 tick 一 token + 剩余预算 round-robin Prefill”，简单且可预测。

### 12.2 Prefill round-robin

维护一个 sequence id deque。每次从头部取出并选择：

```text
chunk = min(prompt_remaining,
            prefill_chunk_tokens,
            token_budget_remaining)
```

若本轮未完成 Prompt，将 sequence 放回尾部。`max_prefill_sequences_per_tick` 可限制一个
tick 内小 Prompt 数量，避免 Host 元数据膨胀。

chunk 不必与 KV block 完全对齐。为简化第一版 Kernel 可以优先选择块边界，但 final chunk
必须允许非整块长度。

### 12.3 Admission

Queued 请求只有同时满足以下条件才能进入 active set：

```text
active sequence slot available
本轮至少可调度一个 Prefill token
prompt + max_new_tokens <= max_model_len
KV Pool 能满足采用的 reservation policy
```

KV reservation 有两种合理策略：

- eager：接纳时预留整个 `prompt + max_new_tokens`，不会运行中途因容量失败，但并发低；
- incremental：每个 step 只预留近期需要的 blocks，并发高，但需要明确 OOM/backpressure
  或 preemption 行为。

第一版若不实现抢占，推荐至少保证已接纳 sequence 能完成下一个 step；不能在 `BuildNext`
选中后才发现无 KV 并部分推进其他行。

## 13. 一个混合 Tick 示例

配置：

```text
max_batched_tokens = 12
prefill_chunk_tokens = 8
max_active_sequences = 4
```

活跃请求：

```text
A: Decoding
B: Decoding
C: Prefilling, prompt_remaining = 20
D: Queued, prompt_length = 6
```

一个合法计划：

```text
A Decode  1 token     used = 1
B Decode  1 token     used = 2
C Prefill 8 tokens    used = 10
D Prefill 2 tokens    used = 12, admission succeeds
```

下一 tick，C 和 D 在 Prefill deque 中轮转。D 不需要等待 A、B、C 完成，只需等待资源可用
和下一个调度边界。

## 14. Ragged Packed Batch 契约

逻辑 rows 长度为 `q_lens = [q_0, ..., q_{B-1}]`，总 token 数：

```text
T = sum(q_lens)
```

推荐 Host/Device 元数据：

```cpp
enum class KvAddressKind { kDenseSlot, kPagedBlocks };

struct PackedSequenceBatch {
  std::vector<SequenceId> sequence_ids;    // [B]
  std::vector<int32_t> token_ids;          // [T]
  std::vector<int32_t> positions;          // [T]
  std::vector<int32_t> sequence_offsets;   // [B + 1]
  std::vector<int32_t> token_rows;         // [T]
  std::vector<int32_t> past_lengths;       // [B]
  KvAddressKind kv_address_kind;
  std::vector<int32_t> dense_slots;        // [B], Continuous-only
  std::vector<int32_t> block_tables;       // [B, stride], Paged combination
  int32_t block_table_stride = 0;
};
```

三条 row 的长度为 `[3, 1, 2]` 时：

```text
token_ids          = [A0 A1 A2 | B0 | C0 C1]
sequence_offsets   = [0, 3, 4, 6]
token_rows         = [0, 0, 0, 1, 2, 2]
positions          = [a, a+1, a+2, b, c, c+1]
```

必须验证：

```text
sequence_offsets.size == B + 1
sequence_offsets[0] == 0
sequence_offsets 单调不减
sequence_offsets[B] == T
token_rows[t] 指向包含 t 的 row
positions 在每条 row 内连续且从 past_lengths[row] 开始
sequence_ids 不重复
每条 KV view 覆盖 past_length + q_len；Paged 组合的 block table 必须完整
所有 token id 合法
```

不要把最大 token 预算 `N` 误解为 `[B, N]` padded 输入。Packed Batch 的设备计算规模是
实际 `T`，预算只限制 `T` 的上界以便预分配 workspace。

## 15. BatchAssembler

BatchAssembler 应是无状态或只持有可复用 buffer：

```cpp
class BatchAssembler {
 public:
  Result<PackedSequenceBatch> Assemble(
      const StepPlan& plan,
      const SequenceKvStore& kv_store);

  Status Validate(const PackedSequenceBatch& batch) const;
};
```

组装步骤：

1. 保持 `StepPlan.rows` 稳定顺序；
2. 计算每行 `q_len` 和 exclusive prefix sum；
3. 写入紧密 token ids、positions 和 row mapping；
4. 查询每条 sequence 的只读 KV view；
5. 复制 `past_length` 和地址描述；Continuous-only 复制 slot/base，Paged 组合复制 block table；
6. 完整验证后才上传设备；
7. 复用 pinned Host buffer 与 Device metadata buffer。

稳定 row 顺序很重要：Runner 的第 `i` 个输出必须能无歧义地映射回第 `i` 个请求。

## 16. Prefill 与 Decode 的执行图选择

### 16.1 两图 MVP

第一版可以让同一调度 tick 产生两个子 batch：

```text
Decode rows  ─► Decode graph
Prefill rows ─► Prefill graph
```

它仍然是 Continuous Batching，只要每轮 membership 动态变化，且每个子图内部确实执行多条
sequence。为了保护流式延迟，通常先运行 Decode 子图，再运行 Prefill 子图。

优点：

- Kernel 契约简单；
- Decode 的 `q_len=1` 可以使用专用 Attention；
- Prefill 可以使用高吞吐 Causal Attention；
- 错误定位容易。

代价是混合 tick 需要两次模型图和两套中间 workspace。

### 16.2 统一 token-level execution

更进一步可把 Prefill 与 Decode rows 放入一次模型调用：

```text
row A: q_len = 1,   past = 200    // Decode
row B: q_len = 64,  past = 128    // Prefill chunk
row C: q_len = 16,  past = 0      // New Prefill
```

这要求每层 Attention 根据每行 `q_len/past_length` 执行不同 causal 范围，并只为需要采样的
row 收集最后一个 token 的 hidden/logits。统一图减少模型入口、同步和 launch，但 Kernel
分支、workspace 和失败域更复杂。

两图与统一图是执行层选择，不改变 Continuous Batching 的定义。应先完成正确的两图版本，
再用 timeline 判断统一图是否值得实现。

## 17. 模型 step 的提交协议

建议把一次执行分为四个阶段：

```text
Prepare  -> Reserve/attach KV capacity, assemble batch, mark in-flight
Execute  -> run model on owning stream
Commit   -> after completion fence, advance KV logical lengths atomically
Apply    -> sample/result state transition and publish events
```

这里存在两个不同的“提交”，不能混淆：

```text
capacity Commit = 把新物理 blocks 挂到 sequence，但不改变可读历史长度
logical Advance = 模型成功后增加 committed_tokens
```

Prepare 必须通过 `SequenceKvStore::PrepareBatch()` 验证本 step 所有 row 的容量；任一行
失败时整个 transaction 回滚。Continuous-only 在这里原子占用/验证 sequence slots；Paged
组合将该接口映射到 `ReserveBatch()` 和 `CommitBatch()` 并挂接 blocks。Capacity commit
不推进逻辑长度；只有模型成功后的 `AdvanceBatch()` 才推进 `committed_tokens`。

### 17.1 Prepare 失败

如果任一 row 无法 Reserve 或元数据验证失败：

- 不启动 GPU；
- 回滚所有尚未 capacity Commit 的 reservations；
- 不推进任何 sequence；
- 清除 in-flight 标记；
- 根据错误类型选择保留 queued 或标记失败。

若 capacity 已挂接后才发现 Host 元数据错误，新增 blocks 可以安全留作该 sequence 的未来
容量，因为 logical length 尚未改变；也可以终止请求并按正常 terminal 路径释放。不能把
已消费的 reservation 再执行一次 Rollback。

### 17.2 Model Forward 失败

Kernel 可能已经写入部分层的 KV，因此不要原地重试同一个 sequence。应：

- 不提交 logical length；
- 等待 owning stream 到达安全点；
- 将本 batch 受影响请求标记 Failed；
- 回收 reservations 和全部 sequence blocks；
- 若是设备级错误，考虑让整个 runtime 进入 failed 状态。

### 17.3 Model Forward 成功

先对整个 batch 原子提交 KV logical lengths，再按稳定 row 顺序处理 logits/sampled tokens。
Sampling 或文本后处理若只对一行失败，可以只终止该请求；其他已成功 row 继续 Apply。

### 17.4 Apply 防重

`StepResult` 必须携带 `plan_id` 和 `sequence_generation`。以下情况必须拒绝：

- 同一 plan 重复 Apply；
- result row 数或 request id 顺序不一致；
- sequence 已被回收并重建；
- result kind 与 scheduled kind 不一致。

## 18. Sampling

每条 row 使用独立 Sampling 配置和 RNG 逻辑状态：

```text
rng_key     = request_seed
rng_counter = generated_token_index
```

不要使用“batch row index”作为唯一随机状态，因为 row membership 每个 tick 会改变。相同请求
在不同并发环境中是否要求完全确定，应由产品契约明确；若要求，RNG 只能依赖请求自身历史。

Device Sampling 可避免完整 logits D2H：

```text
hidden -> LM Head -> logits
       -> penalties -> top-k/top-p -> sample token
       -> copy B token ids to Host
```

第一版可以先实现 Host Greedy oracle，再实现 Device Greedy 和随机采样。无论采样在哪执行，
Scheduler 只消费每行一个明确 token 和状态。

## 19. 结束条件与输出发布

采样新 token 后按明确顺序检查：

```text
1. append generated token to internal history
2. update stop matcher
3. determine EOS / stop sequence / max_new_tokens
4. publish tokens that can no longer belong to a stop suffix
5. publish terminal event if finished
```

优先级必须固定。例如推荐：

```text
stop sequence > EOS > max_new_tokens
```

也可以采用其他顺序，但测试和 API `finish_reason` 必须一致。

### 19.1 Stop token

若生成 token 等于 EOS 或用户 stop token：

- 是否把该 token 返回客户端必须明确；
- 通常内部 history 保留，文本输出不发布；
- 若 `ignore_eos=true`，EOS 按普通 token 处理。

### 19.2 多 token stop sequence

不能在每个 token 产生后立即无条件发 SSE，因为最后若匹配 stop sequence，已发送字节无法
撤回。应保留最长 `max_stop_len - 1` 个潜在前缀 token：

```text
generated history -> incremental stop matcher
                  -> unpublished suffix buffer
                  -> confirmed-safe prefix -> SSE
```

若 tokenizer 的 token-to-text 边界不能直接表示 stop string，协议层还需增量 detokenizer，
并在 UTF-8 边界上发布。

## 20. 取消语义

取消分三种时机：

```text
Queued:    从队列移除，直接 Cancelled
Idle active: 不再调度，释放 KV，进入 Cancelled
In-flight: 设置 cancel_pending，GPU step 安全完成后终止
```

不要试图在共享 Kernel 执行中途释放单条 sequence 的 blocks。客户端断连应转换为普通
Cancel 命令；协议线程不能直接销毁执行资源。

取消与正常完成同时到达时，需要确定线性化点。推荐以 Scheduler 消费命令和 Apply result
的顺序为准：先观察到 terminal 则取消无效，先观察到 cancel 则不再发布新 token。

## 21. Backpressure

至少存在三层 backpressure：

### 21.1 Command queue

有界 MPSC queue 满时，Submit 应立即返回 busy/overloaded，不能无限占用 Host 内存。

### 21.2 Waiting queue

`max_queue_size` 满时拒绝新请求。排队超时应由 service worker 转换为 terminal event。

### 21.3 Active/KV capacity

无 active slot 或 KV blocks 时，请求留在 Queue。不要通过隐式 `hipMalloc` 绕过池上限；
也不要接纳后让它永久占有少量 KV 却无法继续。

建议 metrics 区分：

```text
rejected_command_queue_full
rejected_waiting_queue_full
waiting_for_sequence_slot
waiting_for_kv_blocks
waiting_for_token_budget
```

## 22. Service Runtime API

```cpp
struct TokenEvent {
  uint64_t request_id;
  std::vector<int32_t> token_ids;
};

struct TerminalEvent {
  uint64_t request_id;
  FinishReason reason;
  Status status;
};

using EngineEvent = std::variant<TokenEvent, TerminalEvent>;

class InferenceService {
 public:
  Result<uint64_t> Submit(RequestSpec spec);
  Status Cancel(uint64_t request_id);
  Result<std::vector<EngineEvent>> Poll(
      std::chrono::milliseconds timeout);
  Status Shutdown();
};
```

实现上可以为每个 request handle 提供独立 condition variable/output queue，或由全局事件
路由器分发。必须保证慢客户端不会阻塞 Scheduler worker；每请求输出队列也应有界。

## 23. Scheduler 与 Engine API

```cpp
class Scheduler {
 public:
  Status Enqueue(uint64_t request_id, RequestSpec spec);
  Status RequestCancel(uint64_t request_id);

  Result<StepPlan> BuildNext(SequenceKvStore& kv_store);
  Result<std::vector<EngineEvent>> Apply(
      const StepPlan& plan,
      const ModelBatchResult& result,
      SequenceKvStore& kv_store);
  Result<std::vector<EngineEvent>> Fail(
      const StepPlan& plan,
      Status error,
      SequenceKvStore& kv_store);

  SchedulerStats Stats() const;
  bool HasWork() const;
};

class InferenceEngine {
 public:
  Result<std::vector<EngineEvent>> Tick();
};
```

`Tick()` 的空闲行为不要 busy-spin。没有可执行工作时，worker 应等待 command、最近 deadline
或 shutdown signal；有 Queue 但暂时无 KV 时，可在 block release 时唤醒。

## 24. 与 Paged KV Cache 集成

本节只在 `paged_kv_cache` 与 `continuous_batching` 同时启用时生效。Continuous-only
不得为了复用这里的 API 而偷偷启用 Paged KV。

两者通过稳定 sequence id 连接，而不是通过 batch row：

```text
Scheduler 选择 sequence ids
        ↓
KV Cache ReserveBatch + CommitBatch capacity
        ↓
BatchAssembler 获取 per-sequence block table + past_length
        ↓
Attention 按 row 访问本序列历史
        ↓
模型成功后 AdvanceBatch
        ↓
terminal 后在 owning stream 安全点释放 blocks
```

关键规则：

- BuildNext/Prepare 可以 Reserve/Commit capacity，但不提前增加 committed length；
- 同一 plan 的多个 sequence 必须全体 Reserve/Commit capacity 成功或全体回滚；
- Attention 只读取每行声明的历史与本 step token；
- sequence 完成或取消后不再出现在新 batch；
- block 回收与 GPU completion 同步；
- batch row 可以每 tick 改变，block table 仍属于 sequence。

## 25. 与 Tensor Parallel 集成

本节只在 `tensor_parallelism` 与 `continuous_batching` 同时启用时生效。推荐只有一个逻辑
Scheduler，它生成一次 `StepPlan/PackedSequenceBatch`，再把相同逻辑元数据广播到
所有 TP ranks：

```text
one logical Scheduler
        ↓ identical token ids / positions / membership
Rank 0 Runner ─┐
Rank 1 Runner ─┼─ layer collectives
...           ─┘
        ↓
one logical sampling result
        ↓
Scheduler Apply once
```

每个 Rank 的 KV Pool 只保存本 Rank 的 KV heads，因此物理 block 地址不同，但逻辑 sequence
长度和 membership 必须一致。

若每个 Rank 都复制 Scheduler，则必须逐 tick 校验计划、状态和错误完全一致，复杂度更高。
第一版优先中央逻辑调度、Rank-local 执行状态。

LM Head/Sampling 的选择也要明确：

- 每 Rank replicated LM Head：只采用一个 Rank 的 token，并校验必要的一致性；
- Rank 0-only LM Head：其他 Rank 把所需 hidden 交给 Rank 0，再广播 token；
- Vocab Parallel：分片 logits 后执行分布式 top-k/sample。

不论哪种方案，每步只能向 Scheduler 提交一个逻辑 token。

## 26. Shutdown

推荐顺序：

```text
1. stop accepting new protocol requests
2. enqueue shutdown command
3. cancel queued requests
4. allow or cancel in-flight GPU step at safe boundary
5. publish all terminal events
6. synchronize owning streams
7. release KV and workspaces
8. stop event routing
9. join worker threads
10. destroy device/runtime objects
```

不得使用 detached GPU worker。Shutdown 必须幂等；设备错误或部分初始化失败也要能走同一
资源回收路径。

## 27. Metrics

### 27.1 Scheduler 指标

```text
queued_requests
active_prefill_requests
active_decode_requests
in_flight_requests
finished/cancelled/failed totals
admission_wait_seconds
scheduler_tick_seconds
scheduled_tokens_per_tick
prefill_tokens_per_tick
decode_rows_per_tick
```

### 27.2 执行合批指标

```text
model_batch_calls
prefill_batch_calls
decode_batch_calls
mixed_batch_calls
batch_sequences histogram
batch_tokens histogram
max_batch_sequences
max_batch_tokens
```

`max_decode_batch_size > 1` 或某次 trace 中一个模型调用包含多个 request id，才是 Packed
Decode 的直接证据。HTTP 并发数不是证据。

### 27.3 服务性能指标

```text
TTFT = first token emitted time - request accepted time
TPOT = 相邻输出 token 时间差的统计
ITL  = inter-token latency，通常按每请求观察分布
request latency
prompt throughput tokens/s
generation throughput tokens/s
goodput under an SLO
queueing time
```

必须同时记录场景：Prompt/Output 长度分布、并发、到达模式、采样策略、模型、精度、设备、
KV 配置和 warmup。只给一个 tokens/s 无法判断调度质量。

## 28. 性能测量方法

按层次测量，避免把协议或模型瓶颈归因于 Scheduler：

1. Host-only Scheduler benchmark：`BuildNext/Apply` 延迟和 allocations；
2. BatchAssembler benchmark：不同 B/T/blocks 的组装与 H2D；
3. Runner direct benchmark：固定 packed shapes 的 Kernel 时间；
4. Engine benchmark：真实 KV、sampling 和动态 membership；
5. HTTP load test：Poisson/固定速率到达、SSE 和断连；
6. Kernel timeline：确认 launch 数、空隙、同步和 H2D/D2H。

对比优化前后必须保持相同输入 workload 和测量边界。吞吐提高但 p99 TTFT 恶化可能只是把
Prefill 排得更激进，不能只看平均值。

## 29. 分阶段实施顺序

### 阶段 A：Host Scheduler oracle

- RequestSpec 和输入验证；
- 状态机与不变量；
- FIFO queue、active set；
- Decode-first 和 Chunked Prefill；
- 纯 Host fake runner；
- 取消、失败和 shutdown。

### 阶段 B：单请求真实模型

- 一条 sequence 的 Prefill/Decode；
- 明确首 token 和 KV 长度语义；
- Sampling 和停止条件；
- 当前 KV backend 的 sequence 生命周期；

### 阶段 C：Packed Decode

- 多条 `q_len=1` rows 一次 Runner；
- per-row past length 和 dense slot/paged view；
- 稳定 row-to-result 映射；
- batch size 1 与多行数值对齐。

### 阶段 D：Ragged Chunked Prefill

- offsets/token-to-sequence；
- 多 Prompt chunks 一次 Runner；
- 非整 block final chunk；
- Decode 与 Prefill 同 tick 的两图执行。

### 阶段 E：服务化

- bounded command/output queues；
- JSON/SSE；
- disconnect cancel；
- metrics、trace 和完整 shutdown。

### 阶段 F：可选统一图与高级策略

- unified token-level execution；
- priority/SLO-aware scheduling；
- preemption/swap/prefix cache；
- multi-stream overlap；
- speculative decoding。

每个阶段先完成可执行正确性测试，再进入性能优化。

## 30. 测试矩阵

### 30.1 状态机单元测试

- 空 Prompt、非法 token、超长上下文被拒绝；
- Queue -> Prefill -> Decode -> Finished；
- final Prefill 采样首 token的 off-by-one；
- EOS、stop token、multi-token stop、length；
- queued/idle/in-flight 三种取消；
- 重复 Apply、stale generation、重复 terminal 被拒绝。

### 30.2 调度算法测试

- token/sequence budget 永不超限；
- Decode rows 每 tick 至多一次；
- 多条长 Prompt round-robin 前进；
- Decode-first 不破坏 Prefill 最低公平性；
- Queue FIFO 或声明的优先级稳定；
- KV 不足时保留 Queue 且不部分接纳；
- 空 tick 不产生模型调用。

### 30.3 Packed metadata 测试

- B=1、B>1；
- ragged q lengths 包含 1 和非整块；
- 不同 past lengths；
- Continuous-only 的不同 dense slots；Paged 组合的 block table 跨多个物理页；
- offsets 非单调、row mapping 错误、重复 id 被拒绝；
- 所有 positions 与 causal 范围正确。

### 30.4 数值测试

- Packed Decode 与逐请求 CPU/单行 oracle 对齐；
- Ragged Prefill 与逐请求 Prefill 对齐；
- Chunked Prefill 与整段 Prefill 最终 logits/KV 对齐；
- 混合长短请求不相互污染；
- Greedy token 在 batch membership 改变时保持一致。

### 30.5 生命周期与故障测试

- Prepare 中任一 Reserve 失败时全体回滚；
- Kernel/Runner 失败后不提交 logical length；
- in-flight 取消不提前释放 block；
- 客户端断连最终回收请求；
- queue 满、KV 满、output queue 满均有明确行为；
- shutdown 在 idle、queued、in-flight、device error 时都能结束。

### 30.6 压力测试

- 数千个短请求反复加入退出；
- 长 Prompt 与短 Decode 混合；
- 随机 cancel 和 stop；
- 长时间运行无 Host/Device 内存增长；
- 请求 id/generation 复用不接受 stale result；
- 每个已接受请求最终收到且只收到一个 terminal event。

## 31. 常见错误

### 错误 1：HTTP 并发被称为 Continuous Batching

多个请求可能仍在 GPU 上串行执行。必须观察 Runner batch。
在 `/v1/models` 中暴露单调的 `max_observed_batch_size`，并且只在 Runner
接收到的实际多序列 `StepPlan`/packed batch 上更新它；HTTP worker 数量、
排队请求数和配置中的 `max_concurrency` 都不能作为这个计数器的数据源。

### 错误 2：Scheduler 动态，Runner 逐请求

动态 membership 没有转换成一次多行模型执行，Decode 性能不会得到核心收益。

### 错误 3：Ragged Batch 做成 padded `[B, max_len]`

这会让 token budget 与真实计算量脱节，并浪费 Prefill FLOPs。

### 错误 4：BuildNext 提前推进状态

GPU 失败后无法回滚，KV logical length 与 token history 分裂。

### 错误 5：生成 token 数等于 KV committed 数

刚采样出的 token 尚未作为下一步输入写入 KV，会产生位置偏移。

### 错误 6：每个 Rank 独立做随机采样

随机状态或微小数值差异会让 TP ranks 从下一 token 开始永久分叉。

### 错误 7：取消立即释放 in-flight KV

共享 GPU Kernel 仍可能读取该 sequence 的 blocks。

### 错误 8：无限队列

过载时延迟和内存无限增长，服务没有可控 backpressure。

### 错误 9：Decode 永久占满预算

新请求永远拿不到 Prefill 机会，TTFT 无上界。

### 错误 10：每 tick 做 allocation 和全量同步

Host 调度正确但 launch 间出现大量空隙，吞吐和 TPOT 仍然很差。

## 32. 完成定义

只有同时满足以下条件，才能声明 Continuous Batching 已实现：

1. 每个调度 tick 都能重新决定请求加入、退出和阶段转换；
2. 多条 Decode sequence 能进入一次多行模型调用；
3. 多条不同长度 Prefill chunk 能以 ragged 形式执行；
4. Chunked Prefill 有明确上限和不会永久饥饿的公平策略；
5. token、sequence、KV 三类预算均在执行前验证；
6. BuildNext/Execute/Commit/Apply 边界能处理失败且不部分推进；
7. 请求拥有独立 KV、Sampling、RNG、停止条件和输出状态；
8. 完成、取消、错误、断连和 shutdown 都能安全回收资源；
9. metrics 能证明 GPU batch size 大于 1，而非只有协议并发；
10. packed、chunked 与逐请求 oracle 数值一致；
11. 长时间动态压力测试没有泄漏、stale result 或重复 terminal event；
12. 文档明确两图执行还是统一 token-level execution，不能混淆能力边界。
