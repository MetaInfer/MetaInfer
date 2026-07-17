# 改进计划：Continuous Batching与专用Engine Loop

状态：proposed  
来源：旧任务Iteration 6/7；SRC-VLLM、SRC-TRT、SRC-SGLANG。  
前置Contract：`00_contracts/engine_contracts.md`、
`01_framework_design/02_scheduler.md`、`01_framework_design/07_request_lifecycle.md`。

## 1. 当前证据

旧任务Iteration 7把最多4个Decode请求合成一个Step，在Concurrency 16下从19.88提升
到50.64 token/s；但平均Batch约2.6，HTTP Worker仍竞争`step_mutex_`，Admission、
Model Step和Response写出没有形成单Owner Engine Loop。这说明Batching有效，但并发
线程不等于Continuous Batching。

## 2. 目标与非目标

目标：一个专用Engine线程拥有Scheduler和ModelRunner状态，请求可在每个Step边界
加入、完成或取消；Prefill和Decode共享可解释的Token/Block预算。

非目标：第一阶段不做跨节点PD解耦、Speculative Scheduling或无锁Scheduler。正确的
单Owner状态机优先于复杂并发结构。

## 3. 目标数据流

```text
HTTP workers -> bounded command queue -> Engine loop
                                     -> admission/token budget
                                     -> StepPlan
                                     -> one batched ModelRunner call
                                     -> transactional commit
                                     -> per-request event queues
SSE writers <- bounded response queues <- Engine loop
```

只有Engine线程修改Request State、KV Block Table和Sampling Counter。Tokenizer可在
CPU池运行，但结果携带Generation ID，取消后返回的旧结果必须丢弃。

## 4. 实施阶段

### B0：可观测基线

- 记录每Step的Prefill Token、Decode Sequence、空闲Block、Queue Age和Batch原因；
- 统计实际Batch分布，而不是只记录配置的`max_batch_size`；
- 分离Queue Wait、Tokenize、Model Step、Sample和Write耗时；
- 保存Concurrency 1/4/16的吞吐、P50、P99、错误率和GPU Activity。

### B1：专用Engine线程

- HTTP只调用Submit/Cancel/NextEvent，不直接调用`Step()`或读取内部Request对象；
- 使用有界MPSC Command Queue和每请求有界Event Queue；
- Engine在命令到达、设备Event完成或短Deadline时唤醒，禁止Busy Spin；
- Shutdown按Stop Admission、Cancel/Drain、Flush、Join的顺序执行。

### B2：Token-budget Scheduler

- Prefill预算使用Token数和所需KV Block，不只使用请求数；
- Decode预算使用Active Sequence、Workspace和Backend Batch Shape；
- 长Prompt使用Chunked Prefill，Chunk边界不改变Position和Causal Mask；
- Aging或Deficit策略防止短请求永久抢占长请求。

### B3：Persistent Batch Metadata

- 为Active Slot预分配Input Token、Position、Block Table、Sequence Length和Output；
- 请求加入/退出只更新受影响Slot和Mapping，不重建全部Host/Device Metadata；
- ModelRunner接受一个不可变StepPlan，失败时Engine统一Rollback；
- Batch Shape变化必须走已验证的Operator Dispatch，不读取陈旧Slot。

### B4：Prefill/Decode策略优化

- 先比较Prefill优先、Decode优先和Mixed策略的真实SLO；
- 只有Trace证明大Prefill阻塞Decode时才启用Chunk和独立预算；
- 可保留容易验证的“同一Step只做一种Phase”基线；
- PD解耦只作为后续独立计划，不能用多线程伪装。

## 5. 文件和API范围

```text
Engine::RunLoop
EngineCommand { Submit, Cancel, Shutdown }
Scheduler::BuildNextStep(resources, now)
StepPlan { phase, slots, token_budget, kv_transaction }
ModelRunner::Execute(step_plan, stream)
RequestEventQueue
SchedulerMetrics
```

HTTP、Scheduler、KV Pool和ModelRunner之间只通过这些结构通信。禁止HTTP持有可变
Scheduler指针。

## 6. 本地验收

正确性：

- 单请求输出与原始Greedy基线一致；
- Batch顺序变化不改变请求级Seed序列；
- 请求在Waiting、Prefill、Decode、Response Backpressure时均可取消；
- KV不足、Kernel失败和客户端断开均事务回滚；
- Chunked和Unchunked Prefill的Logits/Token在容差内一致；
- 并发请求不会混用Block Table、Token或Response事件；
- 进程SIGTERM后线程全部Join且设备内存回到基线。

性能：

- 使用固定Prompt集合和输出长度报告实际Batch Size直方图；
- Concurrency 1不允许出现无法解释的明显回退；
- Concurrency 4/16吞吐提升必须同时报告P50/P99和完成率；
- 改动收益必须能由GEMM调用次数、CPU Dispatch或Queue Wait下降解释；
- 任何吞吐提升不能来自减少生成Token或放宽错误判定。

## 7. 风险与回滚

- Dedicated Loop可能暴露已有线程所有权错误，先以单Stream单Owner落地。
- Batch增大可能使小M GEMM更快但KV/Workspace超预算，Scheduler必须使用真实资源快照。
- 如果DTK BLAS某些M Shape数值或性能异常，保留Shape Capability表并缩小Batch，不得
  静默改用CPU。
- 达不到性能目标时回滚优化策略，不回滚Engine所有权和正确性状态机。
