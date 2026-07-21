# Qwen3 HIP Runtime Continuous Batching 实现契约

## 1. 目的与边界

本文是给后续实现 agent 的工程契约。目标是在不引入 ggml、不替换现有 GGUF loader、保留 Q8_0/HIP 路径的前提下，将当前“多 HTTP 连接、单请求串行推理”改成 **continuous batching**。

本文中的 *batch* 指一次 GPU forward 同时推进多条独立序列；不是等待多个完整请求凑齐后再整体执行的 static batching。

本阶段必须实现：

- 多个请求可以独立进入、取消、完成；
- 所有活跃请求在每个 decode tick 各前进一步，并在同一次 GPU batch forward 中执行；
- 每条序列有独立 KV cache、position、sampler、生成结果；
- 模型权重仍只加载一份并全局只读；
- GPU API 只由一个 scheduler 线程调用。

本阶段不要求：

- 引入 ggml、llama.cpp 源码或通用多模型支持；
- continuous prefill（首版可先做 batch decode）；
- paged KV cache、prefix cache、FlashAttention、speculative decoding；
- OpenAI streaming（但接口和状态机应留出空间）。

## 2. 当前实现与并发瓶颈

当前 `Qwen3Engine::generate()` 以 `std::lock_guard<std::mutex> lock(mutex_)` 包住整个 prompt encode、prefill、逐 token decode 和采样过程。因此 HTTP server 虽然每连接一个线程，GPU inference 仍严格串行。

现有 `Qwen3Runtime` 是**单序列状态机**：

- `state_.current_pos`、`state_.n_prompt`、`state_.has_logits` 只有一份；
- KV cache 为 `[layer][position][kv_head][head_dim]`；
- `d_logits_`、`d_scores_`、Q/K/V/FFN scratch 都只有一份；
- `reset()` 只把 position 等状态归零，不分配、不清空 KV cache；下一个请求从 position 0 覆盖旧 cache。

因此不得删除 engine mutex 后直接并行调用当前 runtime：这样会同时覆盖 KV cache、scratch、logits 和 sampler 状态，产生数据竞争与错误输出。

当前模型配置为 36 层、8 KV heads、head_dim 128、最大 context 4096，单条序列的 FP16 K/V cache 约为：

```text
36 × 2(K,V) × 4096 × 8 × 128 × 2 bytes
= 603,979,776 bytes ≈ 0.56 GiB
```

这决定了并发 slot 数必须是显存预算的一部分，不能按请求即时无限创建。

## 3. 参考 llama.cpp 的原则，但不引入 ggml

llama.cpp 的核心设计是 `slot + sequence id + batch + KV memory`：一批 token 中每一项都有 token、position、sequence id；server scheduler 动态维护活跃 slot，然后调用一次 decode。

本项目应复制这个**架构思想**，而不是复制 llama.cpp 或接入 ggml：

```text
llama.cpp 的 seq_id        → 本项目的 slot_id
llama.cpp 的 llama_batch    → 本项目的 RuntimeBatch
llama.cpp 的 server slot    → 本项目的 SequenceState
llama.cpp 的 queue/loop     → 本项目的 Qwen3BatchScheduler
```

现有自定义 HIP runtime 与 kernel 可以继续使用；需要的是给 runtime 和 kernel 增加 batch/slot 维度。

## 4. 目标线程模型（强制）

```text
多个 HTTP worker 线程
  └─ Engine::generate() / submit()
      └─ 短时持锁：向有界 pending queue 入队
          └─ Qwen3BatchScheduler 的唯一 GPU worker 线程
              ├─ 分配/释放 sequence slot
              ├─ 合并 prefill 或 decode batch
              ├─ 调用 runtime 的 GPU forward
              ├─ 为每个 slot 独立采样、完成或继续
              └─ 通过 promise/callback 通知 HTTP worker
```

规则：

1. `Qwen3Runtime`、`hipStream_t`、`hipblasHandle_t`、GPU scratch 和 GPU KV pool 只能由 scheduler 的一个线程访问。
2. HTTP 线程不得直接调用 `runtime.prefill()`、`runtime.decode()` 或 sampler。
3. mutex 仍会存在，但只保护 host-side queue、slot 元数据、结果状态和 shutdown 状态；不得再保护整个 token generation loop。
4. 首版使用一个 HIP stream 和一个 scheduler worker；多 stream/多 GPU 是后续独立课题。

这既避免 CUDA/HIP 状态竞争，也让多请求在一次计算中获得 batching 效益。

## 5. 推荐文件职责

新增文件：

```text
src/qwen3_batch_scheduler.h
src/qwen3_batch_scheduler.cpp
```

修改文件：

```text
src/engine.h / src/engine.cpp            // 对外提交请求、生命周期
include/qwen3_runtime.h
src/qwen3_runtime.cpp                    // RuntimeBatch、batch forward、GPU pool
include/qwen3_z200_kernels.h
src/qwen3_z200_kernels.hip.cpp           // slot_id/position-aware KV 和 attention
src/qwen3_sampler.h / .cpp               // 每 sequence 一个 sampler state 或无状态采样 API
src/main.cpp                              // 可选 CLI: --max-sequences / --max-batch-size
CMakeLists.txt                            // 加入 scheduler source
tests/qwen3_numeric_tests.cpp             // 多 slot kernel 正确性
test.sh / test_spec.md                    // HTTP 并发与关闭测试
```

不要把 scheduler、HTTP socket 逻辑或 `std::thread` 放进 `qwen3_runtime.cpp`。runtime 是 GPU 执行器；scheduler 是主机端状态机；HTTP API 只是请求生产者。

## 6. 公共接口契约

### 6.1 保留的 HTTP/Engine 同步接口

为避免先改 `openai_api.cpp`，保留现有同步 API：

```cpp
bool Qwen3Engine::generate(
    const GenerateRequest &req,
    GenerateResult *res,
    std::string *error);
```

语义改为：

1. 仅在 CPU 上完成 prompt format/tokenize；
2. 建立 `SequenceRequest` 并投递给 scheduler；
3. 等待该请求的 `std::future`；
4. 从 scheduler 返回 `GenerateResult` 或错误。

HTTP worker 因而可以阻塞等待自己的 future；它不占用 GPU，也不持有 runtime mutex。后续支持 streaming 时新增异步接口，不破坏上述同步接口。

### 6.2 新增的 Engine API（建议）

```cpp
using GenerationFuture = std::future<GenerateResult>;

GenerationFuture submit(const GenerateRequest &req, std::string *error);
bool cancel(uint64_t request_id);
void shutdown();
```

`generate()` 可以内部调用 `submit()` 再 `future.get()`。如果队列满，`submit()` 立即失败并设置错误；不要让无限请求堆积在 detached HTTP thread 中。

### 6.3 scheduler API

```cpp
struct SchedulerConfig {
    int32_t max_sequences = 4;       // KV slot 数，必须经显存预算确认
    int32_t max_batch_size = 4;      // 单次 decode 最多 active slots
    int32_t prefill_chunk_size = 128;
    int32_t max_pending_requests = 64;
};

class Qwen3BatchScheduler {
public:
    Qwen3BatchScheduler(Qwen3Runtime &runtime,
                         const Qwen3Tokenizer &tokenizer,
                         SchedulerConfig cfg);
    ~Qwen3BatchScheduler();

    bool start(std::string *error);
    std::future<GenerateResult> enqueue(SequenceRequest request,
                                        std::string *error);
    bool cancel(uint64_t request_id);
    void stop_and_drain();
};
```

不要求把 tokenizer 传给 scheduler：也可在 `Engine::submit()` 先 tokenize 并把 token id 传入。首版建议后者，使 scheduler 只管理已经 tokenized 的请求。

## 7. sequence slot 与状态

一个活跃请求必须对应一个独立的 `SequenceState`。建议的最小结构：

```cpp
enum class SequencePhase { Pending, Prefill, Decode, Finished, Failed, Cancelled };

struct SamplerState {
    std::mt19937 rng;
    bool seeded = false;
    uint64_t seed = 0;
};

struct SequenceState {
    uint64_t request_id = 0;
    int32_t slot_id = -1;             // [0, max_sequences)，独占 KV 区域
    SequencePhase phase = SequencePhase::Pending;

    std::vector<int32_t> prompt_tokens;
    size_t prefill_cursor = 0;
    int32_t position = 0;             // 当前已经写入 KV 的 token 数
    int32_t next_input_token = -1;    // decode 下一轮要送入 runtime 的 token
    std::vector<int32_t> generated_tokens;

    SamplingParams sampling;
    SamplerState sampler_state;        // 每请求 RNG/采样历史；禁止跨 request 共享
    std::vector<int32_t> stop_token_ids;
    int32_t max_new_tokens = 0;

    std::promise<GenerateResult> completion;
    std::atomic<bool> cancel_requested{false};
    std::string error;
};
```

约束：

- `slot_id` 在 `Pending` 时为 -1；进入 GPU 执行前从空闲池取得；结束后才归还；
- 每条序列的 `position` 独立，不能继续使用 runtime 全局 `state_.current_pos`；
- 每条序列必须有独立 RNG。当前 `Qwen3Sampler` 将 RNG 存为成员，不可由所有 slot 共用；应改为 `SamplerState` 作为 sequence 成员，或把 sampler 改成显式接收 state；
- 达到 EOS、stop token、`max_new_tokens`、context 上限、取消或 GPU error 时，必须只结束当前 slot，不能影响其他 slot。

## 8. GPU 内存布局与容量

### 8.1 权重（保持不变）

`Qwen3GgufModel` 在启动时为每个 tensor 上传一份 GPU 内存，权重只读，全部 slot 共享。不要因为并发而复制权重。

### 8.2 KV cache（必须重构）

当前：

```text
K/V[layer][position][kv_head][head_dim]
```

目标：

```text
K/V[layer][slot][position][kv_head][head_dim]
```

推荐线性 offset：

```cpp
size_t kv_offset(
    int layer, int slot, int pos, int kv_head, int dim) {
    return (((((size_t) layer * max_sequences + slot) * max_seq_len + pos)
             * n_kv_heads + kv_head) * head_dim + dim);
}
```

实现可保留 `std::vector<KVCacheLayer>`，但每层的 `k`、`v` allocation 大小必须乘 `max_sequences`：

```cpp
sizeof(__half) * max_sequences * max_seq_len * n_kv_head * head_dim
```

为可控显存，启动时检查：

```text
required_kv_bytes = one_slot_kv_bytes × max_sequences
```

`hipMemGetInfo()` 后要确保还留有 weight、scratch 和安全余量；不足时 `initialize()` 明确报错，不能运行到 OOM。

### 8.3 scratch 和 logits

首版令 batch 上限 `B = max_batch_size`，将所有 token-row scratch 扩为 `[B, ...]`，包括：

- `d_token_ids_`、`d_residual_`、`d_xb_`；
- Q/K/V、norm、attention、FFN intermediate；
- `d_scores_`：需至少 `[B, n_head, max_seq_len]`，或者按 sequence 轮流使用并确保 kernel 不重叠；
- `d_logits_`：必须能够保存每条 batch row 的 vocab logits，即 `[B, vocab_size]`，或使用“forward 后立即逐 row sample”的等价安全设计。

注意：仅把 KV cache 加 slot 维度还不够；当前 `d_logits_` 只有一行，batch 下会互相覆盖。

## 9. RuntimeBatch 与 runtime 接口

删除“runtime 自己保存全局请求 position”的假设。用显式 batch 描述输入：

```cpp
struct RuntimeBatch {
    const int32_t *token_ids;      // [n_tokens]
    const int32_t *positions;      // [n_tokens]，每行所属 sequence 的绝对 position
    const int32_t *slot_ids;       // [n_tokens]，每行所属 KV slot
    int32_t n_tokens = 0;          // 首版 decode: 每 sequence 一行
    bool produce_logits = true;
};

bool Qwen3Runtime::forward_batch(
    const RuntimeBatch &batch,
    std::string *error);

const float *Qwen3Runtime::device_logits_row(int32_t row) const;
```

首版限定：`n_tokens <= max_batch_size`，一个 slot 在一个 decode batch 中最多出现一次。这样最易实现与测试。

原 `prefill()`/`decode()` 可暂时保留为 `slot=0` 的 compatibility wrapper，但 scheduler 不得调用它们。待新路径稳定后，删除 `Qwen3RuntimeState state_` 或降级为仅用于单请求测试。

## 10. scheduler 状态机与 tick 算法

### 10.1 入队和 slot 分配

1. HTTP/Engine 完成 tokenize，构造 `SequenceRequest`；
2. 在短锁下检查 `pending + active < max_pending_requests + max_sequences`；
3. queue 满：立即错误（HTTP 应为 429 或 503），不创建 detached worker；
4. scheduler 醒来，从 pending FIFO 取任务，分配空闲 slot；
5. 设置 phase 为 `Prefill`。

### 10.2 首版：prefill 不混合，decode 连续 batching

为降低第一版难度，可采用：

- 有空闲 slot 时，一个新请求单独或按同长度请求做 prompt prefill；
- 已处于 Decode 的请求始终每 tick batch；
- prompt 过长时按 `prefill_chunk_size` 分块，防止长请求垄断 GPU；
- 新请求的 prefill 不得抢占无限多个 decode tick；建议每 N 个 decode tick 至少处理一个 prefill chunk。

这已经是实用 continuous decode batching；后续再将 prefill chunk 与 decode token 混进同一个 batch。

### 10.3 decode tick（必须满足的语义）

伪代码：

```cpp
while (!stopping) {
    admit_pending_requests_to_free_slots();
    process_prefill_budgeted();

    auto active = collect_slots(SequencePhase::Decode, cfg.max_batch_size);
    if (active.empty()) {
        wait_for_work_or_shutdown();
        continue;
    }

    // one next-token input per active sequence
    RuntimeBatch batch = make_decode_batch(active);
    runtime.forward_batch(batch, &error);

    for (int row = 0; row < batch.n_tokens; ++row) {
        SequenceState &seq = active[row];
        int token = sample_logits_for_row(seq, runtime.device_logits_row(row));
        if (is_finished(seq, token)) {
            complete_and_release(seq);
        } else {
            seq.generated_tokens.push_back(token);
            seq.next_input_token = token;
            ++seq.position;
            // seq remains Decode and is automatically eligible next tick
        }
    }
}
```

“continuous”的必要条件是：每轮重新收集 active slot；新到请求能在后续 tick 加入，结束请求能立即释放 slot，绝不能等待同一批全部生成完才接受新请求。

### 10.4 cancel、错误、关闭

- `cancel()` 只设置 `cancel_requested`；scheduler 在每个 tick 的 batch 构建前检查并结束它；
- GPU forward 失败时，将本次 batch 中所有尚未完成的请求设为失败，并保留错误字符串；
- scheduler 不得在持有 queue mutex 时运行 GPU、采样或调用 promise；
- `stop_and_drain()`：停止接收新任务，唤醒 scheduler，允许当前已提交 GPU 工作完成或安全取消，然后为所有未完成 promise 设置异常/错误，再 join scheduler 线程；
- 禁止 detached scheduler 线程。它必须是可 join 的成员线程，先 join 后析构 runtime/model。

## 11. kernel 改造契约

### 11.1 KV write

旧函数依赖 `start_pos + t` 写入单个 cache。新接口至少需要每行 slot 和 position：

```cpp
hipError_t qwen3_z200_launch_kv_cache_write_fp16_batched(
    const float *k_src, const float *v_src,
    __half *k_cache, __half *v_cache,
    const int32_t *slot_ids, const int32_t *positions,
    int n_rows, int max_sequences, int max_seq_len,
    int n_kv_heads, int head_dim, hipStream_t stream);
```

目标地址必须使用 `slot_ids[row]`、`positions[row]`。不得假定 row i 的位置等于 `start_pos + i`。

### 11.2 RoPE

旧 RoPE 使用连续 `start_pos + token_index`。batch decode 中每行的 position 不同，所以把 `positions` 传入 kernel：

```text
rope position = positions[row]
```

### 11.3 attention

decode attention 要按 `row × q_head` 建 grid，每个 row：

```text
slot = slot_ids[row]
seq_len = positions[row] + 1
K/V base = cache[layer][slot][0]
```

每个 row 只能读自己的 slot，绝不能跨 slot。`d_scores_` 要按 row 切片。

### 11.4 linear / LM head

009 的 fused Q8_0 GEMV 只适合 `M==1`。目标选择策略：

```text
batch rows == 1 → 继续用 fused Q8_0 GEMV
batch rows > 1  → 使用已有 q8_linear + hipBLAS GEMM 路径
```

不要为了第一版强行把 GEMV 扩成复杂 batched kernel。先确保多行路径数值正确、吞吐可测；batch GEMM 或 Q8 GEMM 可以后续单独优化。

LM head 要为每 row 输出 logits。若临时用 `[B, vocab]` logits 内存，必须计入显存预算；若需降低内存，可后续实现 top-k/logit streaming，但不能让不同 row 复用同一 logits buffer 后再异步采样。

## 12. sampler 改造契约

现有 `Qwen3Sampler` 含 mutable RNG、host logits buffer 和 device next-token buffer，不能被多个 sequence 共享。首版可选两条路线：

1. 每个 slot 一个 `Qwen3Sampler`（实现简单，注意各自 GPU buffer）；
2. 一个无状态 `Qwen3Sampler` 服务对象 + `SamplerState`/RNG/host logits 作为每 sequence 参数（更节省，接口改动更大）。

无论哪条路线，必须保证：

- sequence A/B 的 seed 和随机数序列独立；
- 同一个 request 相同 seed 的结果可复现（明确 re-seed 时机）；
- `sample_logits_for_row()` 只读取该 row 的 logits；
- temperature=0 的 greedy path 也支持 batch 行。

## 13. 锁、所有权与禁止事项

允许：

- `std::mutex + std::condition_variable` 保护 pending queue、free slot list、shutdown flag；
- `std::atomic<bool>` 作为 cancellation flag；
- `std::promise/std::future` 将最终结果交回 HTTP thread；
- RAII 管理 scheduler thread、GPU buffer 和 slot 释放。

禁止：

- 用全局 engine mutex 包住 scheduler 的整个运行周期；
- 每个请求创建一个 `Qwen3Runtime` 或一套模型权重；
- 每请求 `hipMalloc` / `hipFree` KV cache；
- 多个线程直接调用同一个 runtime/stream/handle；
- detached scheduler thread；
- 在持有 queue mutex 时等待 GPU 或 future；
- 只改 HTTP server 并认为实现了 batching。

## 14. 实施阶段与验收

### Phase A：可安全排队（不计作 batching）

- 增加 scheduler thread 和有界队列；
- `Engine::generate()` 改为 enqueue + wait；
- runtime 仍单请求执行；
- 目标：替代全局长 mutex、关闭安全、过载可控。

### Phase B：多 slot KV + batch decode（continuous batching MVP）

- KV cache 加 slot 维度；
- scratch/logits 支持 `max_batch_size` 行；
- 增加 `RuntimeBatch` 和 batched decode kernel；
- 每 tick 从活跃 slot 取一个 token；
- 目标：2+ 请求同时活跃时，实际单次 `forward_batch.n_tokens > 1`。

### Phase C：公平性与 prefill 分块

- prompt 分块；
- 限制单次 prefill 预算，避免长 prompt 饿死 decode；
- 新请求能够在旧请求生成期间进入；
- 目标：长/短请求混合时短请求不会等待旧请求全部结束。

### Phase D：性能优化

- profile batch 1 与 batch >1；
- 调整 GEMV/GEMM 分界；
- 考虑 batched LM head、fused attention、paged KV 等。

## 15. 测试矩阵（实现完成前不得声称支持并发）

必须新增或更新测试：

1. **KV slot 隔离**：两个 slot 写入不同 K/V 常量，attention 结果只依赖自己的 slot。
2. **batched kernel vs 单请求**：相同的两个输入分别单独运行和 batch 运行，逐 row logits 在设定 FP16 容差内一致。
3. **相同 seed 可复现**：同一请求单独执行与与其他请求并发执行，输出应一致（若产品定义不同，文档必须明确）。
4. **2/4/最大 slot HTTP 并发**：每个响应 JSON 有效、非空、对应各自 prompt；记录总吞吐和每请求延迟。
5. **slot 回收**：短请求结束后，后到请求可获得其 slot；不得出现 stale KV 泄漏。
6. **queue 满**：返回清晰的过载错误，不崩溃、不无限创建线程。
7. **cancel**：取消一个生成中请求不影响同 batch 的其他请求。
8. **shutdown**：有 pending/active 请求时关闭，scheduler 被 join，所有调用者得到完成或受控错误，无 use-after-free。
9. **单请求回归**：batch size 1 输出仍通过现有 numeric/HTTP 测试。

性能验收至少记录：batch=1、2、4 的 aggregate tok/s、每请求 tok/s、TTFT、p50/p99、GPU 显存；不要仅报 aggregate tok/s。

## 16. Agent 实施前检查清单

- [ ] 阅读本文件以及 `src/engine.*`、`include/qwen3_runtime.h`、`src/qwen3_runtime.cpp`、`src/qwen3_z200_kernels.hip.cpp`。
- [ ] 计算目标 `max_sequences` 的 KV、scratch、logits 显存，不凭感觉设置。
- [ ] 先定义 `SequenceState`、`RuntimeBatch`、slot 生命周期与错误语义，再修改 kernel。
- [ ] 保证模型权重只读共享、KV/scratch/logits 有正确 batch/slot 所有权。
- [ ] 保证 scheduler 是唯一 GPU 调用者，删除旧的全程 engine mutex。
- [ ] 首版只承诺 batch decode；未做 mixed prefill 前，不得宣称完整 vLLM 式调度。
- [ ] 通过第 15 节测试后再调优 kernel。

## 17. 完成定义

满足以下全部条件，才可称为“已实现 continuous batching”：

1. 同时到达的请求不会互相覆盖 KV cache；
2. 两条及以上活跃 sequence 能出现在同一次 runtime batch forward；
3. 新请求能在旧请求未结束时加入后续 batch；
4. 已结束请求的 slot 被立即安全回收并复用；
5. batch=1 行为正确且无回归；
6. HTTP 并发、取消、过载、shutdown 均有自动化测试；
7. 代码和 metrics 能证明 GPU 确实执行过 `batch_size > 1`，而不只是 HTTP 层并发。
