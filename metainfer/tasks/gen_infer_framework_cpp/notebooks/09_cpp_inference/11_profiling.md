# 原生 Profiling 与性能证据

先读 `00_contracts/cpp/cpp_profiling_contracts.md`。该契约定义原生 ROCTX/NVTX、内部 Trace、GPU Event、外部工具、Signal 和 Artifact 行为。

## 1. 稳定的环境接口

生成 Runtime 必须识别：

```text
METAINFER_PROFILE=1
METAINFER_PROFILE_OUTDIR=<directory>
METAINFER_PROFILE_DURATION_S=<seconds>
```

启动时只输出一条摘要：

```text
[metainfer-profile] enabled=1 backend=roctx outdir=/path/to/profile
```

多 Rank 文件名包含 Rank，避免覆盖：

```text
metainfer-profile-rank0-<timestamp>.json.gz
metainfer-profile-rank1-<timestamp>.json.gz
```

## 2. Trace Event 模型

如果平台 Profiler 不可用，实现有界的内部 Chrome Trace：

```cpp
struct TraceEvent {
  std::uint64_t timestamp_ns;
  std::uint64_t duration_ns;
  std::uint64_t correlation_id;
  std::uint32_t thread_id;
  std::int32_t rank;
  TraceCategory category;
  FixedString<48> name;
};

class TraceBuffer {
 public:
  explicit TraceBuffer(std::size_t capacity);
  void Record(TraceEvent event) noexcept;
  Status FlushChromeTrace(const std::filesystem::path& output);

 private:
  std::vector<TraceEvent> events_;
  std::atomic<std::uint64_t> write_index_{0};
  std::atomic<std::uint64_t> dropped_events_{0};
};
```

Buffer 固定容量，满时记录 Dropped Count，禁止 Profiling 导致无限内存增长。

## 3. RAII Scope

```cpp
class TraceScope {
 public:
  TraceScope(TraceBuffer* buffer,
             TraceCategory category,
             std::string_view name,
             std::uint64_t correlation_id,
             int rank)
      : buffer_(buffer), category_(category), name_(name),
        correlation_id_(correlation_id), rank_(rank),
        start_ns_(MonotonicNowNs()) {}

  ~TraceScope() {
    if (buffer_ == nullptr) return;
    const std::uint64_t end = MonotonicNowNs();
    buffer_->Record(TraceEvent{
        start_ns_, end - start_ns_, correlation_id_, CurrentThreadId(),
        rank_, category_, FixedString<48>(name_)});
  }

 private:
  TraceBuffer* buffer_;
  TraceCategory category_;
  std::string_view name_;
  std::uint64_t correlation_id_;
  int rank_;
  std::uint64_t start_ns_;
};
```

`name_` 必须指向静态字符串或在构造时复制到固定缓冲，避免悬空 `string_view`。生产实现应修正此示例的生命周期细节。

使用：

```cpp
Result<StepOutput> ModelRunner::Execute(...) {
  TraceScope step(trace_, TraceCategory::kEngine, "model_step",
                  plan.step_id, rank_);
  for (int layer = 0; layer < layer_count_; ++layer) {
    TraceScope layer_scope(trace_, TraceCategory::kModel, "decoder_layer",
                           ComposeCorrelation(plan.step_id, layer), rank_);
    RETURN_IF_ERROR(layers_[layer].Forward(...));
  }
  return output;
}
```

## 4. Platform Range

如果探针确认 ROCTX/厂商 Range API 可用，可在同一 Scope 中发 Range：

```cpp
class PlatformRange {
 public:
  explicit PlatformRange(const char* name) {
#if METAINFER_HAS_ROCTX
    roctxRangePush(name);
#endif
  }
  ~PlatformRange() {
#if METAINFER_HAS_ROCTX
    roctxRangePop();
#endif
  }
};
```

Header/API 名称必须在 DTK 环境编译验证。缺少可选 Platform Profiler 时内部 Trace 仍需工作。

## 5. 必需 Region

```text
request_parse / tokenize / admission / queue_wait
schedule / kv_reserve / metadata_upload
prefill / decode_step
embedding / decoder_layer / attention / mlp / lm_head
kv_write / paged_attention
collective / sample / detokenize
response_write / profile_flush
```

Region 使用 Request ID、Step ID、Layer、Rank 等 Correlation，不依赖 Prompt 内容。

## 6. Metrics

```cpp
struct RuntimeMetrics {
  Histogram queue_delay_ms;
  Histogram ttft_ms;
  Histogram tpot_ms;
  Histogram decode_step_ms;
  Counter input_tokens;
  Counter output_tokens;
  Gauge active_sequences;
  Gauge kv_blocks_used;
  Gauge kv_blocks_free;
  Counter allocation_failures;
  Counter cancelled_requests;
};
```

还要记录启动/模型加载时间、每 Rank Weight Bytes、Collective Time/Bytes、主要 Operator Time。

## 7. GPU Telemetry

Immutable Perf Oracle 会在 Benchmark Window 使用 `nvidia-smi` 或 DTK `rocm-smi` 采集：

```text
utilization_gpu
utilization_memory
memory_used_mib
power（工具支持时）
per-device peak / active device indices
```

共享服务器上的其他任务会污染数据。Telemetry 只观察，不清理。记录预先 Occupancy、Visible Devices 和污染标记。

## 8. Benchmark Manifest

每份结果必须附带：

```json
{
  "model": "Qwen3-8B",
  "weight_dtype": "bf16",
  "kv_dtype": "bf16",
  "tp_size": 4,
  "prompt_tokens": 128,
  "max_new_tokens": 32,
  "concurrency": 16,
  "build_manifest": "...",
  "hardware_profile": "..."
}
```

缺少这些条件的 Tokens/s 不能跨迭代直接比较。

## 9. 优化闭环

```text
固定 Benchmark 和 Warmup
-> Capture Trace/Telemetry
-> 找到最大实测 Region
-> 只做一个可归因改动
-> 先跑 Correctness
-> 比较 Median/Tail/Memory
-> 记录支持 Shape 和 Fallback
```

不要只看平均 Tokens/s；优化可能损害 TTFT、P99、显存或 Rank Balance。

## 10. Flush 与 Signal

SIGTERM 时：

```cpp
Status Profiler::StopAndFlush() {
  enabled_.store(false, std::memory_order_release);
  RETURN_IF_ERROR(trace_buffer_.FlushChromeTrace(output_path_));
  RETURN_IF_ERROR(AtomicRename(temp_path_, output_path_));
  return Status::Ok();
}
```

先写临时文件，再 Atomic Rename。Oracle 会扫描非空 `.json/.json.gz/.csv/.txt` 等 Artifact。多 Rank 必须各自 Flush。

## 11. 测试

```text
Profiling Disabled 低开销且不写文件
固定容量满后 dropped_events 增加、不 OOM
Nested Scope 时间顺序正确
多线程/多 Rank 文件不覆盖
SIGTERM 能 Flush 非空文件
输出 JSON 可被标准 Parser 读取
Correlation ID 能连接 Request -> Step -> Layer -> Collective
```
