# 原生 Tensor Parallel 实现指南

先读：`00_contracts/cpp/tensor_parallel_contracts.md` 和共享的 `00_contracts/tp_*_contracts.md`。

TP 改变的是权重存储、算子输入输出和通信拓扑，不改变模型语义。必须先得到稳定的 TP=1 Logits/Token，TP=N 才有 Reference。

## 1. Rank 配置与 Launcher

```cpp
struct RankLaunchConfig {
  int rank = 0;
  int world_size = 1;
  int logical_device = 0;
  std::string rendezvous_id;
  std::string rendezvous_endpoint;
};

Result<RankLaunchConfig> ParseRankConfig(int argc, char** argv);
Result<void> RunWorkerRank(const RankLaunchConfig& config);
Result<void> RunRankZeroServer(const RankLaunchConfig& config);
```

推荐一设备一进程。Launcher 必须记录确切 Child PID、Rank、逻辑设备和启动时间，Signal 时只处理自己创建的 PID。

```text
rank0: HTTP + Engine Control + Model Worker
rank1..N-1: Model Worker Command Loop
```

Rank 0 广播 `StepPlan`/Token Metadata；Worker 不独立调度。

## 2. 初始化顺序

```cpp
Result<RankContext> InitializeRank(const RankLaunchConfig& launch,
                                   const HardwareProfile& hardware) {
  RETURN_IF_ERROR(ValidateWorldSize(launch.world_size, hardware));
  HIP_RETURN_IF_ERROR(hipSetDevice(launch.logical_device));

  RankContext context;
  context.rank = launch.rank;
  context.world_size = launch.world_size;
  context.logical_device = launch.logical_device;
  ASSIGN_OR_RETURN(context.compute_stream, CreateStream());
  ASSIGN_OR_RETURN(context.communication_stream, CreateStream());
  ASSIGN_OR_RETURN(context.collective,
                   CreateCollective(launch, context.communication_stream));
  RETURN_IF_ERROR(RunCollectiveSmokeTest(context));
  return context;
}
```

必须先 `hipSetDevice`，再创建 Stream/BLAS/Collective/Allocation。

## 3. Collective Sequencing

为每次 Collective 分配全局 Sequence：

```cpp
struct CollectiveRecord {
  std::uint64_t sequence;
  CollectiveKind kind;
  std::uint64_t element_count;
  DType dtype;
  int rank;
};

class SequencedCollective {
 public:
  Status AllReduce(TensorView buffer, ReduceOp op, BackendStream stream) {
    const std::uint64_t sequence = next_sequence_++;
    Trace(CollectiveRecord{sequence, CollectiveKind::kAllReduce,
                           NumElements(buffer), buffer.dtype, rank_});
    return impl_->AllReduce(buffer, op, stream);
  }
 private:
  std::uint64_t next_sequence_ = 0;
};
```

Hang 调试时比较各 Rank 最后一条 Record，即可定位第一处 Kind/Count/DType/Sequence 不一致。

## 4. Column Parallel Linear

Global Weight（以语义 Shape `[out_features, in_features]` 为例）沿 Output Axis 切分：

```cpp
class ColumnParallelLinear {
 public:
  Result<TensorStorage> Forward(const TensorView& input,
                                BackendStream stream) const {
    // local_weight: [out_features / tp, in_features]
    const std::int64_t tokens = input.shape[0];
    ASSIGN_OR_RETURN(auto local_output,
      TensorStorage::Allocate({tokens, local_out_features_},
                              activation_dtype_, input.device));
    RETURN_IF_ERROR(gemm_->Run(
        BuildLinearProblem(tokens, local_out_features_, in_features_),
        input.data, local_weight_.data(), local_output.data(), stream));
    return local_output;
  }
};
```

Q/K/V、Gate/Up 常采用此模式。是否 Gather Output 由下游决定；Attention Head Local Path 通常无需立即 Gather。

## 5. Row Parallel Linear

Weight 沿 Input Axis 切分，每 Rank 产生完整 Output Shape 的 Partial Sum，然后 AllReduce：

```cpp
Result<TensorStorage> RowParallelLinear::Forward(
    const TensorView& local_input,
    BackendStream stream) const {
  const std::int64_t tokens = local_input.shape[0];
  ASSIGN_OR_RETURN(auto output,
    TensorStorage::Allocate({tokens, out_features_},
                            activation_dtype_, local_input.device));
  RETURN_IF_ERROR(gemm_->Run(
      BuildLinearProblem(tokens, out_features_, local_in_features_),
      local_input.data, local_weight_.data(), output.data(), stream));
  RETURN_IF_ERROR(collective_->AllReduce(output.view(), ReduceOp::kSum,
                                         communication_stream_));
  RETURN_IF_ERROR(backend_->WaitStream(stream, communication_stream_));
  return output;
}
```

真实 Stream/Event 依赖方向需要 Backend API 明确定义。不得复用 Output 直到 Collective 完成。

## 6. Qwen3 GQA KV Head 策略

```cpp
struct HeadPartition {
  int local_q_heads;
  int local_kv_heads;
  bool kv_replicated;
  int q_head_begin;
  int kv_head_begin;
};

Result<HeadPartition> PartitionHeads(const Qwen3Config& cfg,
                                     int rank,
                                     int world_size);
```

如果 `num_key_value_heads % tp_size != 0`，必须选择并记录经过验证的 Replication/Grouping 策略，不能整数除法丢 Head。

## 7. Embedding 与 LM Head

Vocab Parallel Embedding：

```cpp
const bool owned = token_id >= vocab_begin && token_id < vocab_end;
if (owned) local_output = local_embedding[token_id - vocab_begin];
else       local_output = 0;
AllReduce(local_output, SUM);
```

LM Head 可以 Gather 全量 Logits 到 Rank 0，或实现 Distributed Sampling。第一版推荐 Gather/明确内存预算，减少分布式 Top-k 错误。

Vocab Padding 只影响内部 Shard，返回 Token ID 时必须裁剪到真实 `vocab_size`。

## 8. StepPlan 广播

不要广播含有 Host Pointer/`std::string` 内存布局的 C++ Struct。定义稳定 Wire Metadata：

```cpp
struct StepHeaderWire {
  std::uint64_t step_id;
  std::uint32_t kind;
  std::uint32_t sequence_count;
  std::uint32_t total_tokens;
};

struct SequenceWire {
  std::uint64_t request_id;
  std::uint32_t token_count;
  std::uint32_t logical_position;
  std::uint32_t block_count;
};
```

Rank 0 先广播 Header，再广播定长数组和 Token/Block 数据。所有长度 Checked，Worker 验证后才执行。

## 9. Token Sampling

```cpp
Result<std::int32_t> SampleAndBroadcast(const TensorView& logits,
                                        const SamplingParams& params,
                                        RankContext& rank) {
  std::int32_t token = 0;
  if (rank.rank == 0) {
    ASSIGN_OR_RETURN(token, sampler_.SampleOne(logits, params));
  }
  TensorView token_view = HostOrDeviceScalarView(token, rank);
  RETURN_IF_ERROR(rank.collective->Broadcast(token_view, 0,
                                              rank.communication_stream));
  return token;
}
```

实际 Collective 对 Host/Device Buffer 的要求需要 Probe。所有 Rank 独立 Sampling 即使 Seed 相同也可能分叉，禁止。

## 10. 失败传播

```text
某 Rank 检测 Backend/Shape/Collective Error
-> 写入本地 Failure State
-> 尽可能通过控制通道通知 Rank 0/其他 Rank
-> Collective Abort
-> Engine 标记受影响请求 Failed
-> Rank 0 停止 Admission
-> 所有 Rank 有界退出
```

不能让某 Rank 跳过一个 Collective 后继续下一步。

## 11. Overlap

先使用清晰的 Compute -> Collective -> Consumer 顺序。Profile 确认通信是瓶颈后，再用双 Stream/Event：

```cpp
RETURN_IF_ERROR(backend.RecordEvent(compute_done, compute_stream));
RETURN_IF_ERROR(backend.WaitEvent(communication_stream, compute_done));
RETURN_IF_ERROR(collective.AllReduce(buffer, ReduceOp::kSum,
                                     communication_stream));
RETURN_IF_ERROR(backend.RecordEvent(communication_done, communication_stream));
RETURN_IF_ERROR(backend.WaitEvent(compute_stream, communication_done));
```

禁止使用 Device-wide Synchronize“实现 overlap”。

## 12. TP=4 验收矩阵

```text
四个可见设备、四个 Rank、唯一映射
Broadcast/AllReduce 小数组正确
Dimension/Head/Vocab 可切分
每 Rank Weight Key/Shape/Bytes
Column/Row Linear 对比 Reference
一个 Qwen3 Layer 对比
TP=1/TP=4 Greedy Token
Rank 2 注入失败后全局退出
Telemetry 中期望设备 0/1/2/3（或分配列表）均活动
```
