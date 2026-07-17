# 原生C++推理框架总体架构

先读：`00_overview/README.md`、`00_contracts/framework_contracts.md`和
`00_contracts/engine_contracts.md`。

## 1. 分层与依赖

推荐依赖方向：

```text
Native HTTP / OpenAI Schema
          |
          v
Tokenizer + Chat Template
          |
          v
InferenceEngine ----> Scheduler ----> StepPlan
          |                |              |
          v                v              v
      ModelRunner <---- PagedKvCache / Workspace
          |
          v
Model(Qwen3) -> Operators -> Backend(HIP/BLAS/Collective)
```

下层不得依赖上层：Backend不知道Request，Model不知道HTTP，KV Cache不知道
Tokenizer，HTTP不能直接操作Device Pointer。跨层只传递有生命周期约束的值、
View、Handle或Event。

## 2. 进程模型

TP=1使用一个长驻原生进程。TP=N默认一设备一进程：Rank 0拥有HTTP、全局
Scheduler和Sampler；其他Rank只执行已广播的StepPlan。每个进程只管理自己的
设备、Stream、权重和KV分片。Launcher记录并只终止自己创建的PID。

线程建议最小化：

- 一个I/O线程或事件循环接收请求；
- 一个Engine线程串行提交调度状态变化；
- 有界Worker用于JSON/Tokenization等Host工作；
- GPU工作通过Stream/Event异步提交，不以每请求一个线程替代Scheduler。

## 3. 初始化状态机

```text
Created
 -> HardwareValidated
 -> BackendReady
 -> ModelMetadataReady
 -> WeightsReady
 -> RuntimeAllocated
 -> WarmedUp
 -> Serving
 -> Draining
 -> Stopped
```

每次转换只有在当前阶段全部成功后提交。失败时按相反顺序回滚。`/v1/models`
只有在`Serving`后返回Ready；加载期间可返回结构化503，但不能把固定503作为
最终实现。

## 4. 请求数据流

请求先验证OpenAI Schema和采样参数，再应用Checkpoint Chat Template和
Tokenizer。Engine创建RequestState，Scheduler分配Prefill资源并产生StepPlan；
ModelRunner执行真实Checkpoint，提交KV和Logits；Sampler按请求状态选择Token；
Engine提交状态后才向HTTP发布结果。Streaming只改变输出方式，不改变生成语义。

## 5. 所有权边界

- Engine拥有RequestState和Scheduler；
- Model拥有只读Weight Storage；
- KV Manager拥有KV Block；Request只持Block ID；
- Workspace Pool拥有临时Buffer；Step持有Lease；
- Backend拥有Device、Stream、Event和Library Handle；
- HTTP Connection拥有取消令牌和响应Sink，不拥有模型资源。

## 6. 最小端到端交付

第一轮B结束时必须存在一条连续真实路径：

```text
serve.sh -> native binary -> checkpoint config/weights/tokenizer
-> one real accelerator prefill -> decode -> sampler
-> OpenAI response -> SIGTERM cleanup
```

Paged KV、Continuous Batching等被用户选择的能力必须在完整架构中有位置；核心
路径不能以Mock、Python Worker或CPU全模型替代。性能优化可以后续迭代，但真实
模型和正确服务不能后置。

## 7. 可扩展点

ModelFamily、Backend、Operator和Collective使用显式Registry或Factory。扩展点
返回Capabilities，不通过设备名称猜测。每个扩展都必须有Reference Path、
Unsupported错误和独立测试，禁止运行时静默切换到不同Backend。

## 8. 架构验收

- 组件依赖无环，公共头文件不暴露Owning Raw Pointer；
- 一个请求可从HTTP追踪到每个Device Step；
- 取消、OOM、Kernel失败和SIGTERM均有有限时间清理路径；
- TP Rank的调度、Collective和退出顺序明确；
- 采样参数、Streaming和用户Feature都能映射到具体组件与测试。

