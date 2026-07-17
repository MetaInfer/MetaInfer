# 改进计划：Tensor Parallel与RCCL闭环

状态：proposed  
来源：SRC-RCCL、SRC-RCCL-OLD、SRC-ROCM-EX、SRC-TRT、SRC-VLLM。  
前置Contract：`00_contracts/tp_communication_contracts.md`、
`04_parallel_strategies/01_tensor_parallel.md`、
`04_parallel_strategies/04_rccl_collectives.md`。

## 1. 当前边界

单卡C++框架通过并不证明TP存在。当前计划首先完成单机多卡TP，PP、DP、EP和跨节点
通信是后续独立能力。独立`ROCm/rccl`仓库已经retired，当前上游入口在
`ROCm/rocm-systems`；海光DTK可能保留更早的NCCL-compatible接口，因此必须Probe。

## 2. 目标

- 一个Rank对应一个可见Device和独立HIP Context/Stream；
- Rank 0负责HTTP、Tokenize、Scheduling和Sampling决策；
- Qwen3 Dense按列并行/行并行规则加载Local Shard；
- AllReduce、AllGather、ReduceScatter和Broadcast经统一Collective Adapter；
- TP=1与TP=N在可解释容差内等价；
- 任一Rank失败时全组有界退出，不留下Server或设备资源。

非目标：第一阶段不做自定义P2P AllReduce，不做跨节点，不让多个Rank各自接受HTTP。

## 3. 实施阶段

### T0：硬件和Library Probe

- 保留Orchestrator分配的设备可见性，枚举每Rank PCI/Render Node/HIP Device；
- Probe Header、Library路径、SONAME、Version、Symbol和Datatype；
- 在实际Rank数运行Unique ID、Communicator Init、Broadcast和AllReduce Tiny Test；
- 记录P2P/Topology仅作为调优输入，不能代替Collective Smoke Test；
- 不支持时明确失败，禁止静默TP=1。

### T1：进程与控制面

- 明确使用一进程一Device或单进程多线程，第一版推荐一进程一Device；
- Rank 0生成版本化StepPlan Header，广播长度后再广播Payload；
- Worker校验Version、Shape、Token、Block和Sequence上限后执行；
- 所有Rank按同一Collective Sequence Number调用，避免条件分支死锁；
- Startup和Shutdown采用全Rank状态机，错误携带Rank、Device和Collective Name。

### T2：权重分片

- Q/K/V、O Projection、Gate/Up/Down和LM Head逐Tensor声明Shard Axis；
- Safetensors Slice使用Checked Offset，Local Shape写入Manifest；
- GQA的KV Head不能整除TP时选择Replication或Grouping并显式记录；
- 不先把完整模型加载到每Rank再丢弃大部分权重；
- TP=1走同一Shard Planner，减少两套语义漂移。

### T3：模型执行

- Column Parallel输出保持分片，Row Parallel输入保持分片并在边界Reduce；
- RMSNorm、RoPE、KV Layout和Attention使用一致的Local Head Mapping；
- Sampling基线Gather完整Logits到Rank 0，采样后Broadcast Token；
- KV Block分配决策由Rank 0协调，各Rank本地事务同时Commit或Rollback；
- Collective绑定当前Backend Stream并明确Completion Event。

### T4：性能与故障

- 分别测量Collective Payload、调用次数、GPU时间和等待时间；
- 小Collective可在API明确支持后Group，不能擅自改变顺序；
- 注入Rank提前退出、Datatype错误、超时和不同StepPlan；
- 失败时Rank 0停止Admission，全RankAbort/Finalize并由父进程回收；
- 只有标准Collective成为正确基线后才评估自定义P2P优化。

## 4. API范围

```text
RankConfig
CollectiveCapabilities
CollectiveGroup::Init/AllReduce/AllGather/ReduceScatter/Broadcast/Abort
TensorShardSpec
StepPlanWireHeader
RankError
TpRuntimeManifest
```

Model Layer不直接包含RCCL/NCCL兼容Header。Adapter负责Datatype映射、Count单位、Stream、
错误字符串和版本差异。

## 5. 本地验收

正确性：

- TP=1和TP=2/4的Tiny GEMM、单Layer Logits和完整Greedy Token对比；
- 每Rank实际Device FD、显存和Kernel Activity非零；
- Local Weight总元素与Global Tensor Shard守恒，无重叠或缺口；
- GQA不可整除配置选择正确策略或明确拒绝；
- Collective输入输出、In-place/Out-of-place、非默认Stream和多次调用通过；
- 任一Rank错误、SIGTERM和客户端取消不会导致其他Rank永久挂起；
- 所有Worker不监听公共HTTP端口，不独立采样。

性能：报告每RankToken/s、端到端Token/s、计算/通信重叠、Collective时间占比和负载差异。
TP提高可运行模型容量不等同于吞吐提升；若小模型TP变慢，应如实记录。

## 6. 风险与阻塞

- DTK只提供部分兼容API时，以本地Header和运行Probe为事实，不按最新RCCL文档猜测。
- Collective没有可靠超时时，父进程必须提供有界监控和精确PID清理。
- 多Rank数值顺序变化可能放大FP16差异，需要单Layer与Token两级容差。
- 设备数、权限或通信Library不足属于externally_blocked，不允许CPU或单卡冒充TP成功。
