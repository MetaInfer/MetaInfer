# 从空目录到可信框架的分层实现路径

本文把专题合同转换为 Implementer 可以逐层执行的工程顺序。它提炼自已经通过真实
Qwen3-8B F16、TP=2、Paged KV、Continuous Batching 验证的 008 候选，但所有参数
必须来自当前任务的冻结配置。008 只提供经过验证的结构经验，不是可直接复制的产品代码。

## 1. 使用边界

开始编辑前，先读取 `resolved_requirements`、`plan_manifest.json` 和当前能力路由，并生成
一份只读 `FrameworkConfig`。后续模块只能从这份配置或真实 GGUF Metadata 获取参数。

禁止从 008 继承以下常量：

- `tp_size=2` 或固定设备 `[0, 1]`；
- `max_concurrency=4`、`max_batched_tokens=512`；
- `kv_block_size=16`、`kv_total_blocks=256`；
- Qwen3-8B 固定 Layer/Head/Hidden/Vocab；
- F16、gfx906 或 HIP P2P 是所有任务的唯一实现。

只有当前任务明确选择相同条件时，这些值才能作为冻结配置的结果出现。当知识路由显式
注入案例文档时，再读取其中的 008 证据、成功模式和不可泛化部分：
[008 验证案例](../case_studies/008_tp2_paged_continuous.md)。

## 2. 先冻结配置，不要边写边猜

`FrameworkConfig` 至少应包含：

| 域 | 字段 |
|---|---|
| Model | model path、weight format、max context |
| Hardware | backend、device ordinals、architecture |
| TP | enabled、world size、rank-local head/shard contract |
| KV | layout、dtype、block size、capacity policy、total capacity |
| Scheduler | max active、queue size、max batched tokens、prefill chunk |
| Serving | host/port、model id、request limits |
| Validation | required capability IDs、Numeric case IDs、active suites |

配置编译必须在任何大块 Device Allocation 之前完成，并执行以下检查：

1. GGUF Metadata 与冻结模型要求一致；
2. 所有 Head、Intermediate 和 Shard 维度可被 `tp_size` 合法切分；
3. KV 合同至少容纳一个完整 Context；
4. `full_context_per_request` 能同时容纳承诺的全部 Active Request；
5. 每个启用能力都有模块、测试和 Runtime Metadata 所有者；
6. 每个禁用能力都不会因为模板默认值而被偷偷启用。

失败时立即返回配置错误，不要在 Loader、Runtime 和 HTTP 层分别使用不同默认值补救。

## 3. 第一批文件：接口先于实现

先创建能够独立编译的公共接口，再填充后端实现。推荐所有权如下：

```text
include/
  framework_config.h   frozen task/model/runtime configuration
  model_config.h       GGUF-derived model dimensions
  tensor.h             dtype, shape, storage and shard views
  gguf_loader.h        metadata and weight materialization
  tokenizer.h          tokenizer and chat-template boundary
  runtime.h            one-rank forward API and workspaces
  kv_cache.h           dense or paged KV ownership
  scheduler.h          only Continuous: request state and logical StepPlan
  batch_assembler.h    only Continuous: immutable per-tick device snapshot
  tp_coordinator.h     only TP: rank lifecycle and collective order
  engine.h             lifecycle and request completion
  http_server.h        transport only
```

不是所有任务都必须使用这些文件名，但所有状态必须只有一个明确所有者。接口中不得先写
`TP=2`、`block_size=16` 等默认值再等待后续覆盖。

模板 [framework_wiring_template.hpp](framework_wiring_template.hpp) 提供参数化配置、
`StepPlan` 不变量、资源初始化账本和一次 Tick 的事务外壳。它是接口参考，不负责模型数学。

## 4. Shape 与状态账本

实现前在 `plan.md` 或代码常量附近明确以下符号，所有 Kernel 和 GEMM 都复用同一含义：

| 符号 | 含义 |
|---|---|
| `T` | 当前 Tick 的 packed token 总数 |
| `R` | 当前 Tick 的 sequence row 数 |
| `S` | 需要采样的 row 数 |
| `H` | global hidden size |
| `Nq/Nkv` | global query/KV head 数 |
| `D` | head dimension |
| `I` | global intermediate size |
| `P` | TP world size |
| `Nq_r/Nkv_r/I_r` | rank-local shard size |

必须显式验证：

```text
Nq_r  = Nq / P
Nkv_r = Nkv / P
I_r   = I / P
Q rank shape = [T, Nq_r, D]
K/V rank shape = [T, Nkv_r, D]
attention partial = [T, H]
FFN partial = [T, H]
```

Column Parallel 的 Q/K/V/Gate/Up 切输出维；Row Parallel 的 O/Down 切输入维并在
Residual Add 前 AllReduce。禁用 TP 时 `P=1`，仍走相同 Shape 验证，不维护第二套数学。

Packed Batch 至少满足：

- `token_ids.size == positions.size == token_rows.size == T`；
- 每个 `token_rows[i]` 位于 `[0, R)`；
- `sample_rows` 位于 `[0, T)` 且无重复；
- 每个 Sequence Slice 完全落在 `[0, T)`；
- `past_length + token_count <= max_context`；
- Paged KV 的 Block Table 是 Commit 后的 rank-local snapshot；
- 相同逻辑 `plan_id` 发送给所有 Rank，但物理 Block ID 不要求跨 Rank 相等。

## 5. 实现层级与局部完成条件

### L0：构建和公共合同

实现 CMake target、公共类型、错误返回和配置校验。此时不加载模型。

完成条件：

- `bash build.sh` 能编译 Server 空壳和 `qwen3_numeric_tests`；
- `--help`、`--version` 不初始化 GPU；
- Host-only 参考模板测试通过；
- 不存在 Mock Response 或固定 Token。

### L1：GGUF 与 Tokenizer

顺序必须是 Header、Metadata、TensorInfo、对齐后的 data base、Checked Range、Weight
Materialization。先保留 Tokenizer Metadata，再释放不再需要的 Host Model Blob。

完成条件：

- 真实 Metadata 编译出 ModelConfig；
- Tensor range 和 fingerprint 可重复；
- Tokenizer round-trip、Special Token 和 Chat Template 通过；
- Loader 失败不会留下 Device Allocation。

### L2：Reduced Numeric Vertical Slice

先实现当前能力要求的所有 Numeric case，再尝试完整模型。使用
[numeric_harness_template.hpp](numeric_harness_template.hpp) 生成精确 Required Case Set，
缺失 Case 必须失败，不能 skip 或只输出同名 PASS。

完成条件：

- CPU Reference 与 Device Kernel 使用独立实现；
- 所有必需 case ID 均执行；
- F16/Q8_0 和选中能力只激活对应 Case；
- 报告失败时包含具体 Case 和误差，而不是只有总状态。

### L3：单 Rank 真实 Forward

即使最终选择 TP，也先让同一个 Rank Runtime API 能执行 rank-local Forward。TP 任务的
Rank Runtime 从一开始就持有 shard，不要先加载完整权重再在最后一轮伪装分片。

推荐验证顺序：

1. Embedding 非零、有限且输入相关；
2. 第一层 RMSNorm、Q/K/V、RoPE、Attention 中间值有限；
3. 单层 Residual 路径正确；
4. 完整层循环后 Logits 有限且输入相关；
5. Greedy Token 在 Vocabulary 范围内；
6. 两个不同 Prompt 不产生固定输出。

只有 Debug 开关启用时才允许 D2H 中间值采样，正式请求路径不得每层同步和打印 Tensor。

### L4：Engine 和单请求闭环

Engine 负责 Tokenizer、Runtime、Sampler、请求完成状态和线程生命周期。HTTP 层只做协议
转换。单请求闭环必须与选中能力共享同一 Runtime/KV 语义，但未选择 Continuous Batching
时不创建 Scheduler：

```text
Baseline:   Tokenize -> Direct Execute -> Sample/Decode -> Response -> Release
Continuous: Submit -> Enqueue -> BuildNext -> Execute -> Apply -> Response
```

完成条件：真实模型生成非空、有限、输入相关内容，Usage 和 Finish Reason 正确。

### L5：按选择接入 Optional Capability

能力开关彼此独立：

- Paged KV：替换物理 KV Addressing 和 Attention lookup，不自动启用并发；
- Continuous Batching：增加队列、Admission、Chunked Prefill、Packed Decode，不自动启用 Paged KV；
- TP：增加 Rank-local Shard、Collective 和 Group Failure，不自动启用 Paged KV/Batching。

如果 Paged KV 与 Batching 同时启用，使用联合 Reserve/Commit/Apply 状态机；再加入 TP 时，
每个 Rank 本地 Prepare，全部成功后才进入 Group Commit。Iteration 1 仍必须包含所有选中
能力的可执行 Vertical Slice，这里的层级不是延期许可。

### L6：Serving 和真实证据

最后接入 HTTP Listener、OpenAI JSON 和 `/v1/models`。Listener 必须在模型、Runtime 和
所有已选 worker 都 Ready 后启动；未选择 Continuous 时没有 Scheduler worker。Shutdown
顺序相反。

`/v1/models` 只能报告真实配置与观测值。`max_observed_batch_size` 来自真正执行过的
`StepPlan`，不能来自 HTTP 连接数。

## 6. 一次 Scheduler Tick 的事务边界（仅 Continuous Batching）

推荐只有 Scheduler 线程可以调用 `BuildNext` 和 `Apply`：

```text
Drain terminal requests
  -> Admit within sequence/token/KV budgets
  -> Reserve/Prepare every selected rank
  -> Commit capacity and freeze StepPlan
  -> Assemble rank-local batch snapshots
  -> Execute every rank in identical collective order
  -> Sample/Broadcast
  -> Apply token and advance committed KV length
  -> Release terminal sequences
```

关键规则：

- Prepare 失败：回滚本次 Prepare，不产生可见 Plan；
- Execute 失败：不推进 Logical Length，Group 进入一致失败状态；
- Apply 失败：停止继续调度并保留明确错误，不能悄悄跳过 Token；
- Cancel in-flight：只标记，等拥有该 Tick 的线程完成/失败后统一释放；
- 任一路径都必须有有界完成，并且资源只释放一次。

## 7. 初始化与反向清理

推荐初始化顺序：

```text
Validate frozen config
  -> Parse GGUF metadata and tokenizer data
  -> Materialize local weights, or rank-local shards when TP is selected
  -> Create local runtime/KV; add rank collective resources only for TP
  -> Run local self-tests; add rank/collective self-tests only for TP
  -> Create scheduler/engine worker only for Continuous Batching
  -> Start HTTP listener
```

每成功获得一个资源，立即登记到初始化账本。任一步失败或正常 Shutdown 时按严格反序释放：

```text
Stop accepting -> cancel/wake waiters -> join worker -> quiesce ranks
-> release scheduler requests/KV -> runtime/workspace -> collective/BLAS/stream
-> host weights/metadata
```

不要只在 `initialized=true` 后才允许清理；初始化中途失败同样必须释放已经成功创建的部分。

## 8. B 阶段的有界验证梯子

一次改动只运行能证明该层的最小验证：

1. Host-only compile/contract test；
2. `bash build.sh`；
3. `qwen3_numeric_tests --report ...`；
4. Loader-only 或 tokenizer-only test mode；
5. 单个真实 Forward probe；
6. 一次有生命周期所有权的 Server smoke；
7. 完整 Oracle 由编排器运行。

Numeric、Loader 或编译错误不要反复启动 16 GB 模型 Server。完整 smoke 已通过后也不要因为
非权威警告继续启动临时 Server。每次本地 Server 必须在同一 Shell 调用内捕获 `$!`、探测、
发送 TERM 并 `wait`。

## 9. 实现完成前的快速核对

- 冻结配置没有被模块默认值覆盖；
- 每个选中能力都有代码路径、Numeric/行为测试和 Metadata 证据；
- 每个禁用能力保持关闭；
- 所有 Rank 使用相同逻辑 Plan 和 Collective 顺序；
- KV Addressing 使用 rank-local KV Head 和本地 Block Table；
- 只有 Execute 成功才 Apply/Advance；
- 初始化失败、取消、OOM、Shutdown 都能反向清理；
- 不存在 Mock、固定 Token、CPU LM Head 或 TP1 回退；
- 只运行与当前故障层级相符的本地验证。
