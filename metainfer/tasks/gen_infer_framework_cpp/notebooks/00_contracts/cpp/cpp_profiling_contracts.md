# C++ 原生 Profiling 与性能证据强制契约

> 权威级别：`gen-infer-framework-cpp` 的强制契约。
>
> 实现指南：`09_cpp_inference/11_profiling.md`。

本契约规定生成框架如何在不引入 Python Serving Runtime 的前提下，暴露可重复、可关联、可安全 Flush 的性能证据。Profiling 默认关闭，关闭时不得改变执行语义。

## 1. 环境接口

原生 Server 必须识别：

```text
METAINFER_PROFILE=0|1
METAINFER_PROFILE_OUTDIR=<existing-or-creatable-directory>
METAINFER_PROFILE_DURATION_S=<positive-seconds>
```

可选接口：

```text
METAINFER_PROFILE_BACKEND=auto|internal|roctx|nvtx|rocprof|nsys
METAINFER_PROFILE_BUFFER_EVENTS=<bounded-positive-count>
METAINFER_PROFILE_RANK=<integer>
```

非法值必须记录 Warning 并回退到安全默认值；输出目录不可写时必须明确报告，不能静默声称已启用。

启动日志只能输出不含敏感 Prompt/Token 的摘要：

```text
[metainfer-profile] enabled=1 backend=roctx outdir=/path rank=0 duration_s=120
```

## 2. 禁止项

- 不得导入或启动 Python Profiler；
- 不得用 Python Wrapper 启动长期运行的 C++ Server；
- 不得把 Prompt 文本、Authorization Header 或用户隐私写入 Trace；
- 不得在关闭 Profiling 时创建后台线程、文件或大 Buffer；
- 不得让 Trace Buffer 无界增长；
- 不得在 POSIX Signal Handler 中分配内存、加锁、写 JSON 或调用非 Async-signal-safe API。

## 3. 原生抽象

```cpp
enum class ProfileCategory : std::uint16_t {
  kRequest,
  kScheduler,
  kMemory,
  kModel,
  kOperator,
  kCollective,
  kServer,
};

struct ProfileEvent {
  std::uint64_t begin_ns = 0;
  std::uint64_t duration_ns = 0;
  std::uint64_t correlation_id = 0;
  std::uint32_t thread_id = 0;
  std::int32_t rank = 0;
  ProfileCategory category = ProfileCategory::kRequest;
  FixedString<48> name;
};

class Profiler {
 public:
  virtual ~Profiler() = default;
  virtual bool enabled() const noexcept = 0;
  virtual void Record(ProfileEvent event) noexcept = 0;
  virtual Status Flush() = 0;
};
```

关闭路径使用 Null Object，避免热路径反复读取环境变量：

```cpp
class DisabledProfiler final : public Profiler {
 public:
  bool enabled() const noexcept override { return false; }
  void Record(ProfileEvent) noexcept override {}
  Status Flush() override { return Status::Ok(); }
};
```

Profiler 在 Server 启动时构造一次，通过只读指针/引用传入 Engine；不得每个 Request 动态创建。

## 4. RAII Range

```cpp
class ProfileScope {
 public:
  ProfileScope(Profiler* profiler,
               ProfileCategory category,
               StaticString name,
               std::uint64_t correlation_id,
               std::int32_t rank) noexcept
      : profiler_(profiler),
        category_(category),
        name_(name),
        correlation_id_(correlation_id),
        rank_(rank),
        begin_ns_(profiler != nullptr && profiler->enabled()
                      ? MonotonicNowNs()
                      : 0) {}

  ~ProfileScope() {
    if (begin_ns_ == 0) return;
    const auto end_ns = MonotonicNowNs();
    profiler_->Record(ProfileEvent{begin_ns_, end_ns - begin_ns_,
                                   correlation_id_, CurrentThreadId(),
                                   rank_, category_, name_});
  }

 private:
  Profiler* profiler_ = nullptr;
  ProfileCategory category_;
  StaticString name_;
  std::uint64_t correlation_id_ = 0;
  std::int32_t rank_ = 0;
  std::uint64_t begin_ns_ = 0;
};
```

`name` 必须是静态字符串或已复制到固定容量 Buffer，不能保存临时 `std::string_view`。析构函数不得抛异常。

## 5. Platform Range Backend

构建时通过 CMake Probe 检测 ROCTX/NVTX Header 与 Library，不得仅按 Vendor 名称假定可用。

```cpp
class PlatformRange {
 public:
  explicit PlatformRange(const char* name) noexcept {
#if defined(METAINFER_HAS_ROCTX)
    roctxRangePush(name);
#elif defined(METAINFER_HAS_NVTX)
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }

  ~PlatformRange() {
#if defined(METAINFER_HAS_ROCTX)
    roctxRangePop();
#elif defined(METAINFER_HAS_NVTX)
    nvtxRangePop();
#endif
  }
};
```

DTK 上的实际 Header、符号和链接参数必须通过 `try_compile` 或小型 Probe 验证。Platform Range 不可用时，内部 Trace 仍需工作。

## 6. 内部 Trace Buffer

内部 Trace 用于生成 Chrome Trace JSON，必须满足：

- 固定容量或显式上限；
- 多线程写入不会破坏内存；
- Buffer 满时丢弃事件并增加 `dropped_events`；
- Flush 使用快照，不长时间阻塞推理线程；
- 先写临时文件，再 `fsync`（需要时）并 Atomic Rename；
- 每个 Rank 输出独立文件。

```cpp
class BoundedTraceBuffer final : public Profiler {
 public:
  explicit BoundedTraceBuffer(std::size_t capacity,
                              std::filesystem::path output);

  bool enabled() const noexcept override;
  void Record(ProfileEvent event) noexcept override;
  Status Flush() override;

 private:
  std::vector<ProfileEvent> events_;
  std::atomic<std::uint64_t> next_{0};
  std::atomic<std::uint64_t> dropped_{0};
  std::atomic<bool> enabled_{true};
  std::filesystem::path output_;
};
```

文件命名必须避免多 Rank/多次运行覆盖：

```text
metainfer-profile-rank<rank>-pid<pid>-<monotonic-or-wall-ts>.json
```

## 7. 必需 Region

至少提供以下稳定名称：

```text
request_parse
tokenize
request_admission
queue_wait
schedule
kv_reserve
metadata_upload
prefill
decode_step
embedding
decoder_layer
attention
kv_write
paged_attention
mlp_or_experts
lm_head
collective
sample
detokenize
response_write
profile_flush
```

事件通过 Request ID、Step ID、Layer ID 和 Rank 组成的 Correlation ID 关联。不得把完整 Prompt 拼入 Event Name。

## 8. GPU 时间

Host Scope 只能表示 CPU 提交耗时，不能冒充 GPU Kernel Duration。需要 GPU 时间时，应使用 Backend Event：

```cpp
class GpuInterval {
 public:
  static Result<GpuInterval> Begin(Backend& backend,
                                   BackendStream stream,
                                   StaticString name,
                                   std::uint64_t correlation_id);
  Status End(BackendStream stream);
  Result<float> ElapsedMilliseconds() const;

 private:
  BackendEvent begin_;
  BackendEvent end_;
};
```

Event Query/Elapsed 应在安全的收集点执行，禁止每个 Kernel 后 Device-wide Synchronize。

## 9. Runtime Metrics

Trace 之外必须记录聚合指标：

```cpp
struct RuntimeMetrics {
  Histogram queue_delay_ms;
  Histogram ttft_ms;
  Histogram tpot_ms;
  Histogram decode_step_ms;
  Counter input_tokens;
  Counter output_tokens;
  Counter cancelled_requests;
  Counter allocation_failures;
  Gauge active_sequences;
  Gauge kv_blocks_used;
  Gauge kv_blocks_free;
};
```

TP 还需记录每 Rank Weight Bytes、KV Bytes、Collective Calls/Bytes/Duration 和最后完成的 Collective Sequence。

## 10. 外部工具启动

`serve.sh` 可以在 Profiling 开启时使用原生工具包裹二进制，但必须保持前台 PID 可跟踪，并最终执行原生 Server：

```bash
case "${METAINFER_PROFILE_BACKEND:-auto}" in
  rocprof)
    exec rocprof --output-file "${METAINFER_PROFILE_OUTDIR}/rocprof.csv" \
      ./build/native_server "$@"
    ;;
  nsys)
    exec nsys profile --output "${METAINFER_PROFILE_OUTDIR}/nsys" \
      ./build/native_server "$@"
    ;;
  *)
    exec ./build/native_server "$@"
    ;;
esac
```

具体命令必须先运行 `--help/--version` Probe，并适配服务器实际工具版本。不得硬编码假定 `rocprof` 或 `nsys` 一定存在。

Perf Oracle 可收集：

```text
.json / .json.gz / .nsys-rep / .qdrep / .sqlite / .csv / .txt
```

文件存在不代表有效；必须非空、可解析，并与当前 PID/Rank/Benchmark 对应。

## 11. Signal 与 Flush

Signal Handler 只设置 `sig_atomic_t` 标志或写 Self-pipe：

```cpp
#include <csignal>

volatile std::sig_atomic_t stop_requested = 0;

extern "C" void HandleTerminationSignal(int) noexcept {
  stop_requested = 1;
}
```

主事件循环观察标志后执行：

```text
停止接收请求
→ 完成本轮或有界取消
→ 停止 Trace 写入
→ Flush 各 Rank 文件
→ 关闭 Collective/Stream/Device
→ 退出
```

如果收到第二次终止信号，可进入快速退出，但必须记录第一次 Flush 未完成。禁止在 Handler 中直接调用 `Profiler::Flush()`。

## 12. Benchmark Manifest

每份性能结果必须携带：

```json
{
  "model": "Qwen3-8B",
  "weight_dtype": "bf16",
  "kv_dtype": "bf16",
  "tp_size": 4,
  "build_type": "Release",
  "git_or_build_id": "...",
  "hardware_profile": ".../hardware_profile.json",
  "prompt_tokens": 128,
  "max_new_tokens": 32,
  "concurrency": 16,
  "warmup_requests": 8,
  "measured_requests": 64
}
```

缺少这些条件的 Tokens/s、TTFT、TPOT 不得跨迭代直接比较。

## 13. 必测矩阵

```text
Profiling 关闭时不创建文件/线程且低开销
Buffer 满后 dropped_events 增加而进程不 OOM
多线程 Record 不产生损坏事件
Nested Scope 时间范围正确
多 Rank 文件互不覆盖
SIGTERM 通过主循环 Flush 非空文件
第二次 SIGTERM 有界退出
输出 JSON 可由标准 Parser 读取
ROCTX/NVTX 缺失时构建和内部 Trace 正常
rocprof/nsys 不存在时返回明确 Fallback
Correlation 能连接 Request -> Step -> Layer -> Collective
```

Acceptance 要求：无 Python Profiler、无敏感数据、关闭路径无稳定开销、Trace/Metric 可关联到当前运行，并能在 Oracle 发送 SIGTERM 后得到完整原生 Artifact。
