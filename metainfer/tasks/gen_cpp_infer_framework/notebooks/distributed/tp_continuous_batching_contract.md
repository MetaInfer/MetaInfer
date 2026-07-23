# Tensor Parallel 与 Continuous Batching 合同

本文只在 `tensor_parallelism + continuous_batching` 同时启用时生效。Paged KV 是独立
开关：未选择 Paged 时，每个 Rank 使用本地 contiguous sequence slots；选择 Paged 后，
地址与容量事务再由 `tp_paged_kv_contract.md` 和联合状态机覆盖。

## 1. 共享与本地状态

| 所有 Rank 逻辑一致 | 每 Rank 本地 |
|---|---|
| SequenceId、row order、token ids、position | device、stream、workspace |
| q_len、past length、sampling generation | weight shard、local KV heads |
| cancel/terminal decision、Collective sequence | dense slot/base pointer 或 block table |

一个逻辑 Scheduler 是唯一请求状态 owner。Rank worker 不按本地时间重新选 batch，也不独立
推进 sequence 状态。

## 2. Continuous-only 的 KV 视图

未选择 Paged KV 时，Admission 为每个 sequence 在每个 Rank 占用一个本地 slot：

```text
K/V[layer][slot][position][local_kv_head][head_dim]
```

各 Rank 的 slot number 可以不同，但 `SequenceId`、past length 和容量必须一致。冻结资源
合同要求 `max_concurrency` 个 slot，每个都可达到 `max_context_length`。KV dtype 为 FP16，
地址只通过 Rank-local `SequenceKvView` 暴露，不能广播 device pointer。

## 3. 一次 Step

```text
Scheduler freezes logical StepPlan
  -> broadcast/verify row descriptors
  -> each Rank resolves local sequence KV views
  -> all-rank prepare status exchange
  -> concurrent rank-local Forward with identical Collective order
  -> all-rank completion/status exchange
  -> on success, advance committed length on every Rank
  -> Rank 0 samples and broadcasts token
  -> Scheduler Apply exactly once
```

任何 Rank 在 view、Forward 或 Collective 上失败都使 group 失败。失败时不推进任何 Rank 的
committed length，也不允许其余 Rank 单独 Apply。

## 4. Paged 覆盖

同时选择 Paged KV 时，第 3 节的“resolve local view”替换为每 Rank
`ReserveBatch -> group prepare -> CommitBatch -> local block-table snapshot`。Forward 成功后
才执行 `AdvanceBatch`。不能把 Rank 0 的 physical block id 或 pointer 广播给其他 Rank。

## 5. Sampling 与发布

只有 Rank 0（或冻结的 sampling rank）读取最终 logits 并产生 token。Token broadcast 成功
后 Scheduler 才发布输出。HTTP worker 只等待逻辑 request result，不感知 Rank worker。

Cancel 在 in-flight step 中只设置 `CancelPending`；必须等所有 Rank 的 completion fence 安全
后再释放 dense slots 或 Paged blocks。

## 6. 最小证据

1. `tp_collective` 和 `tp_sharded_linear` 使用真实两个 device；
2. `packed_sequence_isolation` 在 TP2 下执行至少两条 sequence；
3. 两个不同 Prompt 的并发确定性输出等于各自 TP2 顺序 baseline；
4. `/v1/models` 同时暴露真实 TP topology、`max_concurrency` 和
   `max_observed_batch_size >= 2`；
5. 一个 Rank 注入失败时所有 Rank 不 Advance，Scheduler 只产生一个 terminal error；
6. shutdown 唤醒 Scheduler 与所有 Rank workers，并 join 后释放每 Rank KV。
