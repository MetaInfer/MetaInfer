# Tensor Parallel 与 Rank-local Paged KV 合同

本文只在 TP 与 Paged KV 同时启用时生效，定义逻辑共享状态和每卡物理状态的边界。
它不要求 Continuous Batching；TP+Paged 的单请求也使用相同 Rank-local Pool 和 group
Prepare/Commit。只有再选择 Continuous Batching 时才增加 packed rows 和动态 membership。

## 1. 状态划分

| 逻辑上所有 Rank 一致 | 每 Rank 本地 |
|---|---|
| SequenceId、Batch membership、row order | Device、Stream、Workspace |
| token id、position、q_len、past length | Weight shard |
| committed token count、Stop/Cancel | KV Pool、Physical Block ID、Generation |
| Collective 序号和 tensor shape | Block Table device pointer |

禁止广播 Rank 0 的 Physical Block ID 或 Block Table pointer。相同物理编号在另一张卡上没有
共享语义。

## 2. Head 与容量关系

对 `tp_size=P`：

```text
local_q_heads  = num_attention_heads / P
local_kv_heads = num_kv_heads / P
kv_bytes_per_token_per_rank
  = num_layers * 2(K,V) * local_kv_heads * head_dim * dtype_bytes
```

Head 数必须整除 P。每 Rank 的 Block 数和 Capacity Policy 必须支持同一逻辑请求承诺；
Physical Block 选择可以不同。

## 3. Group Prepare/Commit

```text
Rank 0 Scheduler builds logical StepPlan
  -> broadcast/verify membership and row shapes
  -> every Rank ReserveBatch in its local KV Manager
  -> all-rank prepare status exchange
  -> if any failure: rollback every successful Rank reservation
  -> barrier
  -> every Rank CommitBatch capacity
  -> each Rank assembles its local Block Table
  -> Forward with identical Collective order
  -> all-rank success exchange
  -> Advance logical committed length
```

不能在知道其他 Rank Prepare 成功前单独 Commit；不能因一个 Rank exhaustion 而让其他 Rank
继续 Forward。

## 4. Weight Sharding

- Q/K/V、Gate/Up 通常按 output dimension 做 column shard；
- O、Down 通常按 input dimension 做 row shard，局部结果后 AllReduce；
- Embedding/LM Head 的复制或分片策略必须明确，并满足内存预算；
- Tensor range 在 Host 解析后再取 Rank slice，检查 offset、shape、dtype 和 byte range；
- Rank 0 Sampling 后广播 token，禁止各 Rank 独立 argmax 后假设结果相同。

## 5. Collective 顺序

每层至少明确 Attention Output 和 MLP Down 的 Collective slot。所有 Rank 即使遇到空 row、
Cancel 或局部错误，也必须通过统一 group 状态退出，不能让一个 Rank 跳过 Collective、另一个
Rank 进入等待。

## 6. 最小测试

- `tp_sharded_linear`：reduced column/row shard 与 CPU Reference；
- `tp_collective`：sum、shape、rank 和错误传播；
- `kv_capacity_contract`：每 Rank 使用 local KV heads 计算 Pool；
- 同一逻辑 Sequence 的两个 Rank Block Table 可以物理不同但 past length 相同；
- 一个 Rank 模拟 exhaustion，验证所有 Rank rollback；
- TP + Paged KV 下完成跨 block 长上下文；再选择 Continuous Batching 时增加两个并发请求；
- `/v1/models` 的 tp_size/world_size/device_ids 与真实初始化一致。
