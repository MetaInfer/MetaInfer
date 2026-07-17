# C++ Tensor Parallel 通信强制契约

> 权威级别：`gen-infer-framework-cpp` 的强制契约。
>
> 实现指南：`04_parallel_strategies/02_qwen_dense_tp.md`和`04_parallel_strategies/04_rccl_collectives.md`。
>
> 权重切分语义：`04_parallel_strategies/01_tensor_parallel.md`。

本文规定 C++ Runtime 的 Rank 生命周期、Collective 接口、Stream 顺序、失败传播和验收方式。生产实现不得依赖 Python、PyTorch Distributed、Gloo ProcessGroup 或 Python Custom Op。

## 1. Backend 原则

目标 Backend：

```text
Hygon DTK / HIP → 厂商提供并经 Probe 验证的 RCCL/NCCL-compatible Library
AMD ROCm        → RCCL
NVIDIA CUDA     → NCCL
TP=1            → Native No-op Backend
```

实际 Header、Library、Symbol 和 DType 支持必须通过 CMake/Runtime Probe 验证。不得只根据 Vendor 名称假定 `librccl.so`、`libnccl.so` 或某个函数签名存在。

自定义 P2P/IPC AllReduce 只能作为验证后的优化路径；第一版必须先有正确、可诊断的标准 Collective Fallback。

## 2. Rank 与 Device 映射

每个 Rank 必须绑定一个明确的逻辑可见设备。禁止绕过 `HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES` 重新访问未分配设备。

```cpp
struct RankConfig {
  std::int32_t rank = 0;
  std::int32_t world_size = 1;
  std::int32_t logical_device = 0;
  std::string rendezvous_endpoint;
  std::string run_id;
};

Status ValidateRankConfig(const RankConfig& cfg,
                          const HardwareProfile& hardware) {
  if (cfg.world_size <= 0 || cfg.rank < 0 || cfg.rank >= cfg.world_size) {
    return InvalidArgument("invalid rank/world_size");
  }
  if (cfg.world_size > hardware.visible_device_count) {
    return FailedPrecondition("TP exceeds assigned visible devices");
  }
  return Status::Ok();
}
```

推荐一设备一进程：

```text
Rank 0: HTTP Server + Engine Control + Local Model Worker
Rank 1..N-1: Native Model Worker Command Loop
```

Launcher 必须记录 Child PID、Rank、Device、启动时间和退出状态；只能终止自己创建并仍能确认身份的 PID。

## 3. 初始化顺序

强制顺序：

```text
解析/验证 RankConfig
→ 设置逻辑设备
→ 创建 Compute/Communication Stream
→ 创建 BLAS Handle 并绑定 Stream
→ Rendezvous/交换 Unique ID
→ 创建 Collective Communicator
→ 运行小规模 Collective Smoke Test
→ 分配模型权重和 KV Cache
→ 进入 Worker Loop
```

```cpp
Result<RankContext> InitializeRank(const RankConfig& cfg,
                                   Backend& backend,
                                   Rendezvous& rendezvous) {
  RETURN_IF_ERROR(backend.SetDevice(cfg.logical_device));

  RankContext ctx;
  ctx.config = cfg;
  ASSIGN_OR_RETURN(ctx.compute_stream, backend.CreateStream());
  ASSIGN_OR_RETURN(ctx.communication_stream, backend.CreateStream());
  ASSIGN_OR_RETURN(ctx.collective,
                   CreateCollective(cfg, rendezvous,
                                    ctx.communication_stream));
  RETURN_IF_ERROR(RunCollectiveSmokeTest(ctx));
  return ctx;
}
```

不得在 `SetDevice` 之前创建 Stream、BLAS Handle、Communicator 或 Device Allocation。

## 4. Native Rendezvous

Rendezvous 可以使用本机 TCP、Unix Domain Socket 或 Task 私有状态目录中的原子文件协议，但必须满足：

- Run ID 防止连接到旧任务；
- Protocol Version 明确；
- Unique ID/控制消息有长度上限；
- 有界超时和取消；
- 只有当前用户可读写；
- 不在日志输出 Secret/Raw Unique ID；
- Rank 0 崩溃时 Worker 能有界退出。

```cpp
struct RendezvousHeader {
  std::uint32_t magic = 0;
  std::uint16_t version = 1;
  std::uint16_t message_kind = 0;
  std::uint32_t payload_bytes = 0;
  std::uint32_t rank = 0;
  std::uint32_t world_size = 0;
  std::uint64_t run_hash = 0;
};
```

禁止直接发送包含 Pointer、`std::string`、VTable 或 Host ABI Padding 的 C++ 对象；Wire Format 必须定长或显式序列化。

## 5. Collective 抽象

```cpp
enum class ReduceOp { kSum, kMax, kMin };

class Collective {
 public:
  virtual ~Collective() = default;

  virtual Status AllReduceInPlace(MutableTensorView tensor,
                                  ReduceOp op,
                                  BackendStream stream) = 0;
  virtual Result<TensorStorage> AllGatherLastDim(
      TensorView local,
      BackendStream stream) = 0;
  virtual Status Broadcast(MutableTensorView tensor,
                           std::int32_t root,
                           BackendStream stream) = 0;
  virtual Status Barrier(BackendStream stream) = 0;
  virtual Status Abort() noexcept = 0;
};
```

接口入口必须验证：

```text
Tensor 位于当前 Rank 的设备
DType 可映射到 Collective Backend
Element Count 无溢出
Buffer 对齐满足 Backend 要求
Root Rank 合法
所有 Rank 的 Kind/Count/DType/Sequence 一致
```

TP=1 使用 Native No-op：In-place Collective 直接成功；需要新 Storage 的 AllGather 返回显式 Copy/Owned Result，调用方不能依赖 Python Tensor Alias 语义。

## 6. DType 映射

```cpp
Result<CollectiveDType> ToCollectiveDType(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return CollectiveDType::kFloat32;
    case DType::kFloat16: return CollectiveDType::kFloat16;
    case DType::kBFloat16:
      return ProbeBfloat16CollectiveSupport()
          ? Result<CollectiveDType>(CollectiveDType::kBFloat16)
          : Unsupported("BF16 collective is unavailable");
    default:
      return Unsupported("dtype is unavailable for collective");
  }
}
```

不得在未记录的情况下把 BF16 转成 FP16。需要 FP32 Accumulation/Conversion 时必须在计划、Manifest 和误差测试中明确。

## 7. 全局 Sequence 与死锁诊断

每个 Rank 对每个 Collective 分配单调递增 Sequence：

```cpp
struct CollectiveTraceRecord {
  std::uint64_t sequence = 0;
  CollectiveKind kind = CollectiveKind::kAllReduce;
  std::uint64_t elements = 0;
  DType dtype = DType::kUnknown;
  std::int32_t rank = 0;
  std::uint64_t step_id = 0;
};
```

调用前记录 Begin，完成后记录 End/Status。Hang 报告必须包含所有 Rank 最后一条 Sequence/Kind/Count/DType/Step，便于定位第一个分叉点。

禁止某个 Rank 在条件分支中跳过 Collective。需要分支时，Rank 0 必须把决定作为 `StepPlan` 广播给全部 Rank。

## 8. Stream/Event 顺序

正确顺序示例：

```cpp
RETURN_IF_ERROR(backend.RecordEvent(compute_done, compute_stream));
RETURN_IF_ERROR(backend.WaitEvent(communication_stream, compute_done));
RETURN_IF_ERROR(collective.AllReduceInPlace(
    output, ReduceOp::kSum, communication_stream));
RETURN_IF_ERROR(backend.RecordEvent(communication_done,
                                    communication_stream));
RETURN_IF_ERROR(backend.WaitEvent(compute_stream, communication_done));
```

强制规则：

- Producer Stream 完成后 Communication Stream 才能读取 Buffer；
- Consumer Stream 等待 Collective 完成后才能复用 Buffer；
- Host Buffer 生命周期覆盖异步调用；
- 不得用 Device-wide Synchronize 代替依赖设计；
- Backend 返回异步错误时必须在有界检查点传播。

## 9. Linear 切分语义

Column Parallel：

```text
Global weight [out_features, in_features]
Local weight  [local_out_features, in_features]
Local output  [tokens, local_out_features]
通常无需立即 Gather
```

Row Parallel：

```text
Global weight [out_features, in_features]
Local weight  [out_features, local_in_features]
Partial       [tokens, out_features]
AllReduce SUM 得到完整 Output
```

Bias 只能加一次。若每个 Rank 在 AllReduce 前都加完整 Bias，会得到 `tp_size * bias`，属于硬错误。

## 10. Qwen3 GQA 与 KV Head

```cpp
struct HeadPartition {
  std::int32_t local_q_heads = 0;
  std::int32_t local_kv_heads = 0;
  std::int32_t q_head_begin = 0;
  std::int32_t kv_head_begin = 0;
  bool replicate_kv = false;
};

Result<HeadPartition> PartitionQwen3Heads(const Qwen3Config& model,
                                         std::int32_t rank,
                                         std::int32_t world_size);
```

如果 KV Head 少于 TP Rank 或不能整除，必须选择并验证 Replication/Grouping。每 Rank Weight Loader、RoPE、KV Cache Layout 和 Attention Kernel 必须使用同一 Head Partition。

## 11. StepPlan 与 Worker Loop

Rank 0 是唯一调度决策源。Worker 接收版本化 StepPlan：

```cpp
struct StepHeaderWire {
  std::uint32_t protocol_version = 1;
  std::uint32_t step_kind = 0;
  std::uint64_t step_id = 0;
  std::uint32_t sequence_count = 0;
  std::uint32_t total_tokens = 0;
  std::uint32_t metadata_bytes = 0;
};
```

Worker 必须先验证所有长度、枚举、Token/Block 上限，再执行本地 Model Step。未知版本或非法长度必须全 Rank 失败，不能尝试猜测。

## 12. KV Cache 跨 Rank 事务

逻辑 Block 数和请求 Context Length 必须跨 Rank 一致：

```text
Rank 0 广播 block_count/context_length
→ 每 Rank 本地 Reserve
→ AllReduce(can_reserve, MIN)
→ 全部成功 Commit
→ 任意失败 Rollback + 请求失败
```

TP 容量由最小 Rank 决定。不得让显存较大的 Rank 继续推进而较小 Rank 跳过本步。

## 13. Sampling 与输出

第一版建议由 Rank 0 收集/持有完整 Logits 并采样，然后广播 Token ID：

```cpp
Status BroadcastSampledToken(std::int32_t* token,
                             RankContext& ctx) {
  MutableTensorView token_view = DeviceOrPinnedScalarView(token, ctx);
  return ctx.collective->Broadcast(token_view, 0,
                                   ctx.communication_stream);
}
```

所有 Rank 独立采样即使 Seed 相同也可能因数值差异分叉，禁止作为默认路径。分布式 Top-K 只能在 TP=1/Reference 对齐后实现。

## 14. 失败传播与退出

任意 Rank 检测到 Backend、Shape、OOM、Protocol 或 Collective 错误时：

```text
记录本地 Failure + Collective Sequence
→ 通过 Native 控制通道通知 Rank 0（尽力而为）
→ Abort Communicator
→ 停止新请求 Admission
→ 标记受影响请求失败
→ 终止 Worker Loop
→ Rank 0 回收自己启动的所有 Child PID
```

所有等待必须有界。不能让某个 Rank 无限等待已退出 Rank，也不能通过杀死设备上未知进程来恢复。

## 15. CMake 与运行时发现

CMake 必须检查 Header、Library 和符号，并把结果写入 Build Manifest：

```cmake
find_path(COLLECTIVE_INCLUDE_DIR NAMES rccl/rccl.h nccl.h)
find_library(COLLECTIVE_LIBRARY NAMES rccl nccl)

if(NOT COLLECTIVE_INCLUDE_DIR OR NOT COLLECTIVE_LIBRARY)
  message(FATAL_ERROR "TP requested but native collective library is missing")
endif()
```

实际 DTK 安装可能使用不同目录/名称，应允许 Toolchain File 或 Cache Variable 覆盖。禁止在 TP 请求下静默编译成单卡。

## 16. 必测矩阵

```text
TP=1 Native No-op 语义
TP=2/4 小 Tensor Broadcast/AllReduce/AllGather
FP32/FP16/BF16 支持与误差
Count/DType/Sequence 不一致时有界失败
Rank/Device 唯一映射
Column/Row Linear 对齐单卡 Reference
Qwen3 GQA Head Partition
Embedding/LM Head 与真实 vocab_size
KV Reserve 某 Rank OOM 时全局 Rollback
Rank 2 注入退出时其余 Rank 有界 Abort
多轮 Prefill/Decode Collective 顺序一致
Telemetry 中全部分配设备有实际活动
```

Acceptance 要求：TP=N Greedy Token/Logits 与 TP=1 在 DType 阈值内对齐；无 Python 通信 Runtime；无未分配设备访问；失败不会永久 Hang；全部 Rank 的 Collective Trace 可重建同一全局顺序。
