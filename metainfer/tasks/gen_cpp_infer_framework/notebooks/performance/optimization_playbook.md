# C++ 推理框架测量驱动优化手册

本文只服务 Perf Tester 和 Perf Planner。正确性阶段不得为了采用这里的优化而改变冻结的
weight format、KV layout、TP topology 或能力开关。任何优化都必须保留相同 Numeric case、
真实模型输出合同和 HTTP 生命周期。

## 1. 优化输入

开始前固定并记录：

```text
model fingerprint + weight format
context/output lengths
selected capabilities and combination contracts
tp_size/device ids
max_concurrency/batch-token budget
KV dtype/layout/capacity policy
build profile and commit/candidate id
```

没有这些字段的性能数字不能跨迭代比较。008 的 TP2/F16/Paged/Continuous 结果只能用于
说明测量方法，不能成为 Q8、单卡或 Continuous-only 任务的目标值。

## 2. 先分解时间

至少把一次请求或一次 Scheduler tick 分成：

```text
queue/admission
tokenization
prefill model time
decode model time
sampling/detokenization
HTTP serialization
```

模型时间继续按 operator family 分解：

```text
linear GEMM / dequant
attention + KV read/write
collective
host gap / synchronization
```

先用 wall-clock 和现有 runtime counters 建立可重复 baseline，再对最大的区段运行 rocprof。
不要在完整 8B 模型上反复采集所有 counter；先用 `--stats` 或同等摘要定位，再做单项 trace。

## 3. 按冻结权重格式选择入口

### F16

- Weight 已常驻 F16，不存在 Q8 dequant workspace；不要实现量化优化。
- Decode 的 local GEMM 常为 `M=1`，重点观察 GEMV-like shape、launch 数和 Host gap。
- Prefill 重点观察 `T`、矩阵尺寸、GEMM 时间和是否被不必要的逐层同步打断。
- TP 后 local GEMM 变小；必须同时比较 GEMM 收益与 Collective 占比。

### Q8_0

- 若每个 decode linear 都把整个矩阵解到 FP16，首先测 dequant bytes/time 与 GEMM time。
- `M=1` 优先评估直接读取 Q8_0 block 的 fused GEMV；Prefill 再评估 tiled dequant GEMM。
- LM Head 可按 vocab tile 分块，但必须保持完整 global argmax/sampling 语义。
- 不得把 GGUF Q8_0 bytes 当普通 INT8 matrix 交给 hipBLAS。

一次实验只替换一个 linear family 或一个 shape，Numeric 与固定 logits 通过后才扩大范围。

## 4. KV 与 Attention 分支

### Dense 单序列或 TP-only

- 先测实际 context 范围内的 KV read bandwidth，不按训练最大 context 推算性能。
- FP32 dense reference 可以优化访问和向量化，但不能按 FP16 bytes 报告。
- Decode Attention 的工作量随 committed length 线性增长，报告应按 context bucket 分组。

### Continuous-only contiguous slots

- 每 sequence slot 使用冻结 FP16 KV；测 active slots、实际 batch size 和 context 分布。
- 固定 slot 的容量浪费是已知取舍，不能通过暗中启用 Paged KV 改变任务能力。
- Packed Runner 必须真正执行多 row；HTTP 并发数不是 GPU batch size。

### Paged KV

- 同时记录 block utilization、最后 block 浪费、block-table upload 和 paged attention 时间。
- 优先合并/复用 metadata upload，再评估 Paged Attention 的向量化和 online softmax。
- 不能为追求命中率破坏 generation、stale-view 拒绝或 batch transaction 原子性。

## 5. Continuous Batching

吞吐优化必须同时报告：

```text
request throughput / generated tokens per second
TTFT and TPOT percentiles
observed Runner batch histogram
queue wait and active/queued counts
prefill/decode token mix
error/cancel rate
```

优先级通常是：消除串行 engine mutex、形成真实 packed decode、限制 chunked prefill 尾延迟、
减少每 tick Host allocation/copy，最后才考虑统一 Prefill/Decode 图。提高
`max_concurrency` 但 `max_observed_batch_size` 仍为 1，不算优化成功。

## 6. Tensor Parallel

每层至少测 Attention O 与 MLP Down 两个 Collective slot，并报告：

```text
local GEMM time
collective time and bytes
barrier/host wait
peer-access or collective backend
per-rank memory
```

若 Collective 占比高，先检查 count、重复同步和 P2P 依赖；若 local GEMM 效率下降，检查 TP
后 shape 是否过小。Replicated LM Head/Embedding 可能成为显存或计算瓶颈，但切换 Vocab
Parallel 会改变通信和 sampling 合同，必须作为独立、完整验证的候选。

当前只验证 TP2。不得为了得到更好数字切到 TP1，也不得在没有 TP4 合同和设备验证时改变
`tp_size`。

## 7. 实验和晋升条件

每个候选记录：

```text
hypothesis
one changed mechanism
baseline/candidate command and workload
median plus dispersion
correctness evidence
memory delta
decision: keep / revert / inconclusive
```

候选只有同时满足以下条件才可晋升：

1. 全部冻结 Numeric case 无 skip；
2. 固定真实模型输出仍有限且满足确定性合同；
3. Server 能正常 SIGTERM、无残留进程/显存；
4. 目标指标提升超过噪声，非目标延迟和错误率没有越界；
5. `/v1/models` 仍反映真实能力与资源配置；
6. 没有通过禁用能力、缩短工作量或 TP1 fallback 获得收益。

不满足时回退该候选，不把多个不确定改动叠到下一轮。

## 8. 信号到动作

| 主要信号 | 首个检查 | 首选候选 |
|---|---|---|
| Q8 decode dequant 时间最高 | 每 linear dequant bytes/time | fused Q8 GEMV |
| Prefill GEMM 之间 Host gap 大 | stream sync、allocation、launch timeline | 移除逐层同步/分配 |
| Attention 随 context 急剧恶化 | causal range、KV dtype、memory loads | online softmax/向量化 KV load |
| Runner batch 始终为 1 | queue ownership、mutex、tick membership | packed decode 接线 |
| TTFT 被长 Prompt 拖高 | prefill chunk 和 decode fairness | 调整 chunk/token budget |
| TP Collective 占比高 | count、barrier、peer topology | 合并同步/优化 Collective |
| TP local GEMM 效率低 | shard 后 M/N/K | shape-specific kernel/策略 |
| 显存接近上限 | weight/KV/workspace 实测账本 | 先处理最大真实占用项 |

表中动作只是候选生成规则，不能替代 profiler 证据。
