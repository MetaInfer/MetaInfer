# 008 稳定候选：TP2 + Paged KV + Continuous Batching

本文记录一个已经通过真实验证的实现案例，用于理解模块怎样组合以及哪些失败经验值得复用。
它不是 Binding Contract，也不是可直接复制的 Golden Repository。冻结需求、不可变 Oracle 和
专题合同始终具有更高优先级。

## 1. 已验证范围

候选来源：

```text
nodes/worker6/workspaces/
  stage7-qwen3-8b-tp2-full-validat-f310c8e2/008-layered-validation
```

验证配置：

| 项目 | 008 值 |
|---|---|
| Model | Qwen3-8B F16 GGUF |
| Hardware | 2 x Hygon Z200 |
| Backend | HIP/hipBLAS, gfx906 |
| Tensor Parallel | TP=2 |
| Context | 4096 |
| Paged KV | block size 16, rank-local pool/table |
| Continuous Batching | max concurrency 4 |
| API | OpenAI-compatible chat completions |

最终分层验证证据：

- Numeric：17/17；
- Hard cases：12/12；
- Acceptance Suite：16/16；
- C4 真实模型阶段约 142.3 秒；
- 人工验证期间观测到 TP2 每卡约 10.1 GB；
- 验证完成后 Server 进程与显存正常释放。

证据文件位于：

```text
nodes/worker6/.metainfer/tasks/
  stage7-qwen3-8b-tp2-full-validat-f310c8e2/logs/
    manual-layered-validation-008-after-numeric-repair/
      oracle-stages.json
      oracle-report.json
```

这里的“已验证”只覆盖上述组合，不证明 TP4、Q8_0 TP、其他模型、其他 GPU 或任意并发参数。

## 2. 通过实现的模块边界

008 使用以下结构完成真实请求：

```text
GGUF Model
  -> Model Loader + Weight Materializer
  -> per-rank Runtime/Stream/BLAS/Weight/KV
  -> one logical Scheduler
  -> per-rank BatchAssembler snapshot
  -> TP-coordinated Transformer layer loop
  -> rank-0 LM Head/Sampler
  -> Scheduler Apply
  -> Engine completion state
  -> OpenAI HTTP response
```

模块对应关系：

| 责任 | 008 模块 |
|---|---|
| Model metadata and shard materialization | `model_loader.cpp` |
| Rank-local forward/workspace | `runtime.cpp` |
| Rank-local physical KV | `kv_cache.cpp` |
| Logical request state and StepPlan | `scheduler.cpp` |
| Per-rank immutable batch snapshot | `batch_assembler.cpp` |
| Rank init, layer order and collectives | `tp_coordinator.cpp` |
| Request wait/notify and scheduler worker | `engine.cpp` |
| JSON and transport | `openai_api.cpp`, `http_server.cpp` |

成功的关键不是文件名，而是 Scheduler 不操作 Device Pointer、BatchAssembler 不拥有长期请求、
每个 Rank 拥有自己的 Weight/KV/Stream，以及 HTTP 线程不直接执行 Scheduler Tick。

## 3. 已验证的数据流

一次 Tick 的逻辑路径：

```text
waiting request
  -> admission by sequence/token/KV budgets
  -> reserve capacity on every participating rank
  -> freeze one logical StepPlan
  -> assemble two rank-local PackedPagedBatch snapshots
  -> embedding on each rank
  -> for each layer:
       local attention projections + paged attention
       row-parallel O partial -> AllReduce -> residual
       local Gate/Up/SwiGLU/Down partial -> AllReduce -> residual
  -> final norm + LM Head + greedy sample on rank 0
  -> Apply sampled token and advance KV lengths on every rank
  -> release finished sequence on every rank
```

这证明了 `Scheduler -> StepPlan -> rank-local snapshot -> Execute -> Apply` 是可行的集成边界。
通用模板应保留这条边界，但 Collective 实现、LM Head 分片策略和 Rank 数必须由当前任务决定。

## 4. 最有价值的成功模式

### 4.1 一个 Scheduler，多个 Rank-local KV Owner

Scheduler 产生一次逻辑 Plan。每个 Rank 使用相同 Sequence ID、Position 和 Token Row，但从本地
KV Manager 读取 Block Table。不能让 Rank 0 的 Physical Block ID 直接成为 Rank 1 Device 地址。

### 4.2 Loader 只读取一次，Shard 成功后释放完整 Host Tensor

008 在 Materialization 前保留 Model/Tokenizer Metadata，在所有 Rank Shard 建立后释放原始完整
Tensor。这个模式减少 Host Memory 峰值，但必须保证所有 `TensorView` 已重新绑定到拥有生命周期的
Shard Storage，不能留下指向已释放 Buffer 的 View。

### 4.3 Collective 启动自测

在 HTTP Ready 前，用已知小向量验证 Peer Access 和 AllReduce。这样 Collective 错误不会等到
36 层真实 Forward 中才表现为坏 Logits。

### 4.4 单一 Scheduler Owner

HTTP 线程只 Enqueue 并等待自己的 Sequence Completion。专用 Worker 串行拥有 `BuildNext`、
`Execute`、`Apply`，避免多个 HTTP 线程同时修改 Scheduler/KV 状态。

### 4.5 Numeric 先于真实模型 Server

最终稳定验证先运行 Reduced Numeric，再运行真实模型阶段。C3 能复用 C2 Numeric 证据，因此没有
重复加载模型。这个顺序直接减少了无效的长时间 Server Boot。

## 5. 008 暴露出的返工来源

首轮从空目录创建完整框架时，单次 Implementer 尝试出现约 65 次 Read、56 次 Write、72 次 Edit
和 62 次 Bash。主要返工来自：

- 同时创建几十个 Header/Source，接口在实现过程中反复变化；
- 直到真实 Server Boot 才发现 Loader、Shape 或 Numeric 问题；
- Scheduler、KV 和 TP 在不同文件重复使用固定参数；
- 手工 Smoke Test 重复启动大模型；
- 测试脚本的非权威警告诱发额外 Server 生命周期操作。

因此知识库不应只增加更多算法说明，而应提供编译通过的公共合同、精确实现顺序和分层验证梯子。

## 6. 不得复制的 008 选择

以下是案例限制，不是推荐默认值：

- `world_size != 2` 直接失败；
- Device Ordinal 固定为 `[0, 1]`；
- Scheduler 固定 `4/512/256` 等参数；
- KV Pool 固定 256 Blocks；
- CMake 固定 `--offload-arch=gfx906`；
- 只接受 Qwen3-8B F16 TP；
- Layer Loop 中存在用于定位数值问题的 D2H 和 Debug Print；
- TP2 Collective 使用特定的 P2P Copy/Sum/Copy 实现；
- LM Head 和 Sampling 放在 Rank 0 的具体策略。

生成新任务时，以上每一项都必须重新由冻结需求、模型 Metadata、硬件能力和 TP 合同决定。

## 7. 从案例到通用实现的映射

| 008 观察 | 通用资产 |
|---|---|
| 大量文件从零创建导致接口返工 | `implementation_sequence.md` 先冻结接口和配置 |
| Scheduler/Batch/TP 接线最终稳定 | `framework_wiring_template.hpp` 固化逻辑边界 |
| Numeric Repair 后才可信 | `numeric_harness_template.hpp` 强制精确 case set |
| TP/Paged/Batching 组合成功 | 联合状态机和 TP rank-local KV 合同 |
| 固定参数散落 | 所有参数集中到 `FrameworkConfig` |
| 大模型 Smoke 重复 | Build -> Numeric -> Loader -> Forward -> Server 验证梯子 |

## 8. Agent 使用规则

1. 先读取冻结需求和 Binding Contract；
2. 只提取模块边界、事务顺序和验证方法；
3. 不复制 008 常量、设备编号、模型尺寸或 Backend 假设；
4. 不把 008 的 Debug 路径放进正式 Request Path；
5. 当前 Oracle 与案例冲突时，以当前 Oracle 为准；
6. 当前能力组合不是 TP+Paged KV+Batching 时，不读取或套用本案例。
