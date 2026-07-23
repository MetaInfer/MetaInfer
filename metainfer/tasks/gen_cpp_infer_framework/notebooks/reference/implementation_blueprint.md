# C++ 推理框架实现蓝图

本文是任务的模块索引和集成合同，不替代各专题文档。Agent 应先用本文确定所有权、
接口和数据流，再按当前能力或失败 Route 阅读详细合同。

## 1. 真相来源

优先级从高到低：

1. 冻结的 `resolved_requirements` 与 `plan_manifest.json`；
2. 不可变 Oracle 的用例和 Numeric case ID；
3. 本知识库中的合同与 reference-source；
4. 当前迭代的真实构建、Numeric、Server 和 HTTP 证据；
5. Review、Retrospective 和 Agent 推测。

不得用后一级推测覆盖前一级合同。所有 Optional Capability 都是独立开关；只实现
任务选中的能力及其激活的组合合同。

## 2. 模块所有权

| 模块 | 必须拥有的状态 | 禁止拥有的状态 |
|---|---|---|
| GGUF Loader | Metadata、TensorInfo、对齐后的 data base、Host/Device Weight | Scheduler、请求状态 |
| Tokenizer | Vocabulary、Merge Rank、Special Token、Chat Template | Model Weight、KV Block |
| Model Runtime | Rank-local Weight、Workspace、Stream、Forward Kernel | HTTP 连接、全局请求队列 |
| KV Manager | Physical Block、Generation、Sequence Block Table、Committed Length | Batch row 顺序、采样状态 |
| Scheduler | Sequence 状态、Admission、Fairness、StepPlan | Physical Device Pointer、Kernel 临时区 |
| Batch Assembler | 当前 Step 的 row、token、position、past length、block table snapshot | 长生命周期请求所有权 |
| TP Coordinator | Rank、Device、Shard、Collective 顺序、Group Failure | HTTP JSON、跨 Rank 物理 Block ID |
| Engine | 上述模块的生命周期和事务边界 | 协议字符串拼接 |
| OpenAI API | JSON 校验、请求映射、响应和 `/v1/models` 证据 | Model 数值状态、KV 分配 |
| `serve.sh` | Port、Model Path 传递、前台进程生命周期 | Mock 自动回退、后台 daemon |

## 3. 必须存在的接口边界

### 3.1 Loader

```text
ReadHeader -> ReadMetadata -> ReadTensorInfo
           -> data_base = align_up(tensor_info_end, general.alignment)
           -> absolute_offset = checked_add(data_base, tensor.offset)
           -> ValidateRange -> Load/Shard Tensor
```

每个加法和乘法都必须检查溢出与文件边界。Tensor offset 相对 data blob，不相对
文件开头。Model Config 从真实 Metadata 编译，不能用 Qwen3-8B 常量掩盖解析失败。

### 3.2 KV 与 Scheduler

```text
Submit -> Queued -> Admitted -> Prefill/Decode -> Finished/Cancelled/Failed
                    |               |
                    +-- Reserve -----+
                         Commit capacity before Forward
                         Advance committed length after Forward succeeds
```

Scheduler 只产生逻辑 `StepPlan`。KV Manager 拥有物理容量。失败的 Reserve/Prepare
不能改变任何可观察状态；Forward 失败不能提前推进 committed length。

### 3.3 TP

Rank 共享逻辑 Batch membership、Token rows、Collective 顺序和最终 Token。Weight、
Workspace、Stream、KV Pool、Physical Block ID 和 Block Table 均为 Rank-local。
Rank 0 采样后广播 Token；任一 Rank 失败时整个 group 进入一致的失败/停止状态。

### 3.4 HTTP

HTTP worker 将已验证请求提交给 Engine，并等待有界结果。取消只发送事件，不直接释放
in-flight KV。`GET /v1/models` 的能力字段必须来自实际 Runtime 配置和观测计数。

## 4. 一次请求的数据流

```text
HTTP JSON
  -> Tokenizer + Chat Template
  -> [Continuous?] Scheduler admission + packed StepPlan
     [otherwise] direct single-request Generate plan
  -> KV capacity/view
     [Paged?] transactional blocks + block-table snapshot
     [otherwise] dense sequence slot/contiguous view
  -> [TP?] per-rank local snapshot
     [otherwise] one local Runtime view
  -> Embedding -> 36 Transformer Layers -> Final Norm -> LM Head
  -> [TP?] Rank 0 Sample -> Token Broadcast
     [otherwise] local Sample
  -> KV committed-length Advance
  -> [Continuous?] Scheduler Apply
  -> Decode/Stop loop
  -> Detokenize -> OpenAI response
  -> release every active local/rank-local resource exactly once
```

Prefill 和 Decode 必须走同一套 Weight、Tokenizer、KV 和 Sampling 语义。测试专用路径
不能成为 HTTP 路径的另一份实现。

## 5. Capability 分支

### Baseline

单请求也必须包含真实 GGUF、Tokenizer、Forward、KV、Generation 和 C++ HTTP Server。

### Paged KV

将 dense position 改为 `sequence block table + position`。增加 Block generation、
Reserve/Commit/Rollback/Release 和 Paged Attention；不能只暴露 Metadata。

### Continuous Batching

增加有界请求队列、单一 Scheduler owner、packed/chunked StepPlan、per-row 状态和
`max_observed_batch_size`。并发 HTTP socket 不等于 Continuous Batching。

### Tensor Parallelism

根据冻结 `tp_size` 分片 Weight 和 KV Head，固定 Collective 顺序。真实目标模型不得
降级为 TP1；只允许 reduced synthetic operator 使用单设备参考。

### 组合能力

Paged KV + Continuous Batching 使用联合事务状态机。TP 组合还要求每 Rank 对同一逻辑
Step 独立 Reserve 本地 Block，并在 group barrier 后一起 Commit。

## 6. 建议实现顺序

1. 建立 CMake target、公共类型和错误模型；
2. 完成 GGUF Metadata/Tensor range 验证和 Tokenizer；
3. 完成 reduced Numeric target 及全部当前必需 case ID；
4. 完成基础单序列 Prefill/Decode/Generation；
5. 仅在选择 TP 时完成冻结拓扑下的 Rank-local Weight 初始化和 Collective；
6. 接入其余已选 KV/Batching 能力和被激活的组合事务；
7. 接入 OpenAI API、Runtime Metadata 和 owned process lifecycle；
8. 只运行 bounded build/Numeric/boot smoke，完整 Oracle 由编排器执行。

这只是依赖顺序，不是延期许可。Iteration 1 的最终产物仍必须包含所有选中能力的可执行
vertical slice。

## 7. 完成定义

- `build.sh` 产生 Server 和 `qwen3_numeric_tests`；
- Numeric Report 包含当前能力要求的所有精确 case ID，无 skip；
- Server 使用冻结 Model Path、Hardware 和 TP topology；
- `/v1/models` 字段能追溯到真实 Runtime 状态；
- Paged KV、Batching、TP 及激活组合均有行为证据；
- 无 Mock、固定答案、CPU LM-head、TP1 或禁用能力回退；
- 所有失败路径有界返回，并释放或回滚其拥有的资源。

详细验收项见 [能力实现检查表](../validation/capability_checklists.md)。

## 8. Implementer 落地资产

本文确定模块所有权和集成边界；真正开始 B 阶段实现时继续使用：

- [分层实现路径](implementation_sequence.md)：从冻结配置到 L0-L6 Vertical Slice、
  Shape/状态账本、初始化反向清理和有界验证梯子；
- [Framework Wiring Template](framework_wiring_template.hpp)：可编译的参数配置、
  Logical StepPlan、Rank-local Snapshot、Init Journal 和 Tick Transaction；
- [Numeric Harness Template](numeric_harness_template.hpp)：根据 Weight/Capability 生成
  精确 Required Case Set，缺失 Case 直接失败；
- [008 验证案例](../case_studies/008_tp2_paged_continuous.md)：仅当路由显式注入完整
  TP + Paged KV + Continuous Batching 组合时读取。

模板只固定接口不变量，不固定 `tp_size`、并发数、Block Size、模型尺寸、Device ID 或
Backend。专题合同与当前冻结任务决定这些参数。
