# Qwen MoE的TP与Expert放置

先读：`02_model_specifics/02_qwen3/02_moe.md`、`01_tensor_parallel.md`和
`04_rccl_collectives.md`。

## 1. 并行维度

MoE包含两种独立选择：Dense/Attention使用Tensor Parallel；Expert可以复制、
继续按TP切分，或使用Expert Parallel。第一版应选择一种并写入Runtime Manifest，
不能把TP和EP术语混用。

## 2. Router

Router输入在各TP Rank上语义一致时，Router Weight通常复制并得到一致Top-K。
Tie规则和归一化固定，必要时由Rank 0广播Assignment摘要。所有Rank必须验证每个
Token的Expert ID、Weight和顺序一致，否则后续Collective可能挂起或产生错值。

## 3. TP-sharded Expert

每个Expert的Gate/Up沿Output切分，Down沿Input切分并AllReduce。每个Rank都持有
所有Expert的Local Shard，内存预算按`num_experts * local_expert_bytes`计算。Shared
Expert遵循同一规则并在合并前明确Scale。

## 4. Expert Parallel

EP只在Collective和Token Dispatch能力验证后启用：

```text
router assignment
-> count per destination rank
-> bounded all-to-all dispatch
-> local expert grouped GEMM
-> reverse all-to-all
-> weighted scatter-add in original token order
```

必须有容量上限、零Token Rank、重复Expert、失败传播和原顺序恢复。缺少AllToAll
能力时明确Unsupported，不得静默退化为单Rank执行全部Expert。

## 5. 权重加载

Weight Report按Expert、Rank和Shard列出Shape/DType/Bytes。未知Expert Key、缺失
Shared Expert或Double Shard必须失败。量化Expert还需独立Scale/Zero Point契约。

## 6. 测试

- Tiny 4-Expert Router与Reference Assignment一致；
- 每Rank零/多Token和同Token多Expert；
- TP=1/TP=N的单MoE层输出与Top-K一致；
- EP Dispatch往返恢复原Token顺序；
- 单Rank失败触发全局Abort；
- 每个分配设备有权重、Device FD和活动遥测。

