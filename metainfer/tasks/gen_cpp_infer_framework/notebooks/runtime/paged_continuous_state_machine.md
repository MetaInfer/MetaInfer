# Paged KV 与 Continuous Batching 联合状态机

本文只描述两项能力同时启用时的跨模块事务。各自数据结构见专题合同。

## 1. 请求状态

```text
Created -> Queued -> Admitted -> Prefill <-> Decode -> Finishing -> Terminal
             |          |           |          |
             +-------> CancelPending <---------+
                                      -> Failed
```

`Terminal` 只进入一次。只有 Scheduler owner 可以修改请求状态；HTTP worker 只能发送
Submit/Cancel 事件。

## 2. Tick 事务

一次 Tick 必须按以下顺序执行：

1. Snapshot 当前 Active Sequence；
2. 根据 fairness、token budget 和 max concurrency 选择 membership；
3. 为每行计算 `past_length + q_len` 所需 capacity；
4. `ReserveBatch` 暂存所有新增 Block；
5. 任一行失败则 `RollbackBatch`，状态和 free count 完全不变；
6. 全部成功后 `CommitBatch`，再冻结 `StepPlan`；
7. BatchAssembler 读取每 Sequence 的 committed capacity snapshot；
8. 上传 token、position、past length、Block Table；
9. 执行 Paged KV Write、Attention 和 Model Forward；
10. Forward 全部成功后原子 `AdvanceBatch` committed length；
11. Sampling、Stop、Output 发布；
12. Terminal Sequence Release，重新进入下一 Tick。

Capacity Commit 只让物理 Block 对该 Sequence 可见，不等于写入 token 成功。逻辑长度只在
第 10 步推进。

## 3. 必须保持的不变量

- 同一 Sequence 的物理 Block 可以变化，SequenceId 不能随 batch row 变化；
- `committed_length <= reserved_blocks * block_size`；
- 每个 row 的 Block Table 只包含该 Sequence 的有效 generation；
- `free + reserved/attached == total`，且失败路径不制造重复 free；
- BatchAssembler 不持有跨 Tick 的可修改 Block Table 引用；
- in-flight Step 的 KV 在设备完成前不能释放；
- Scheduler 的 `max_observed_batch_size` 来自执行过的 StepPlan。

## 4. Exhaustion

单请求的 `prompt + max_new_tokens` 永远超过 Pool 时立即返回 ResourceExhausted。暂时容量不足
但请求理论可满足时可以留在有界队列；必须有 timeout/cancel，不能永久等待。

`full_context_per_request` 要保证 `max_concurrency` 个完整 Context 同时 reservation。
`shared_token_budget` 只承诺共享 Token 上限，并如实暴露较小的 full-context guarantee。

## 5. Cancel 与错误

- Queued cancel：移出队列，不触碰 KV；
- Admitted 但未 Commit：Rollback reservation；
- Capacity 已 Commit、未 Forward：可以保留给同 Sequence 下一 Tick，或在终止时 Release；
- Forward in-flight：标记 CancelPending，等待设备完成后 Release；
- Device/Collective error：Runtime Failed，停止新 admission，清理所有 Sequence。

## 6. 最小测试

1. 两请求 Reserve，其中第二个导致 exhaustion，验证第一请求也未部分 Commit；
2. Commit 后构造 Host metadata 错误，验证 committed length 未推进；
3. 两个不同 Prompt 并发，输出等于各自顺序 baseline；
4. Cancel in-flight 请求，验证其他 Sequence 的 Block Table 不变；
5. 全部请求结束后 free blocks 回到初始值；
6. `max_observed_batch_size >= 2`。
