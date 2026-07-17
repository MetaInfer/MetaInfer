# 原生 Tensor Parallel 契约

> 权威级别：多设备 C++ 执行的强制契约。

## 1. Rank 模型

默认可移植设计为“一设备一原生进程”。每个进程必须先设置本地设备，再创建 Stream、分配 Tensor、创建 BLAS Handle 和初始化 Collective。

```cpp
struct RankContext {
  int rank = 0;
  int world_size = 1;
  int logical_device = 0;
  std::string device_uuid;
  BackendStream compute_stream;
  BackendStream communication_stream;
};

Result<RankContext> InitializeRank(const LaunchConfig& launch);
```

Rank 0 负责外部 HTTP I/O 和确定性调度元数据。其他 Rank 不得自行接受请求或独立采样。

## 2. 初始化顺序

```text
所有 Rank 解析并校验同一模型/Build 配置
-> 设置本地逻辑设备
-> 创建 Backend/Stream/Handle
-> 初始化 Collective Communicator
-> 执行小型 Broadcast/AllReduce Probe
-> 加载本 Rank 权重分片
-> Barrier/Ready
-> Rank 0 开始监听 HTTP
```

TP Size 必须在加载权重前验证：可见设备数、模型维度整除、通信库和拓扑都必须满足要求。

## 3. Collective 接口

```cpp
class Collective {
 public:
  virtual ~Collective() = default;
  virtual Status Broadcast(TensorView buffer,
                           int root,
                           BackendStream stream) = 0;
  virtual Status AllReduce(TensorView input_output,
                           ReduceOp op,
                           BackendStream stream) = 0;
  virtual Status AllGather(TensorView input,
                           TensorView output,
                           BackendStream stream) = 0;
  virtual Status Abort() noexcept = 0;
};
```

每次 Collective 必须在各 Rank 上保持相同的全局顺序、Count、DType 和 Communicator。

## 4. 权重切分规则

- Q/K/V、Gate/Up 通常沿 Output Feature 做 Column Parallel；
- Attention Output、MLP Down 通常沿 Input Feature 做 Row Parallel 并归约 Partial Output；
- Embedding/LM Head 必须明确 Vocab Shard、Mask、Gather/Distributed Sampling；
- GQA KV Head 必须明确 Partition 或 Replication；
- Norm 参数和 Residual 必须明确复制/切分策略。

每个 Rank 在 Upload 前校验 Local Tensor Shape：

```cpp
Result<TensorSlice> ComputeShard(const TensorSpec& global,
                                 int rank,
                                 int world_size,
                                 ShardAxis axis);
```

## 5. 强制规则

- **TP-001**：所有 Rank Collective 顺序一致，可用递增 Sequence ID 记录。
- **TP-002**：Count、DType、Communicator 必须一致。
- **TP-003**：由 Rank 0 采样并 Broadcast Token，禁止各 Rank 独立采样。
- **TP-004**：任意 Rank 失败后，必须在有限时间内传播错误并协调退出。
- **TP-005**：通信完成之前不得复用依赖 Buffer。
- **TP-006**：Compute/Communication Overlap 必须使用显式 Stream/Event，在正确性建立后再启用。
- **TP-007**：Scheduler 决策和逻辑 Block Table 在所有 Rank 完全一致。
- **TP-008**：只使用 Visibility 暴露的设备，禁止重置或清理其他用户进程。

## 6. 四卡验收

TP=4 至少验证：

1. 恰好四个 Rank 映射到四个已分配设备；
2. Collective Smoke Test；
3. 每个 Rank 权重字节和 Local Shape；
4. 一个 Layer 与 Reference 对比；
5. TP=1 与 TP=4 的 Greedy Token 等价；
6. 注入单 Rank 失败后全局退出；
7. Telemetry 显示四个期望设备均有实际活动。

四张 16 GiB 卡始终是四个地址空间，不得把它们作为一个 64 GiB Allocation 使用。
