# DeepSeek V3 — 混合专家（MoE）

## 结构

DeepSeek V3 使用**细粒度 MoE** 并带**共享专家**，与 Mixtral 等更简单的 MoE 不同。

### 对比

| 特性 | Mixtral | DeepSeek V3 |
|--------|---------|-------------|
| 专家总数 | 8 | 256 |
| 每 token 激活 | 2 | 8（从 256 中路由） |
| 共享专家 | 无 | 1–2（始终参与） |
| 专家粒度 | 粗（一专家 = 整段 FFN） | 细（一专家 = 小 FFN） |
| 路由 | 简单 top-k | 分组 top-k |

## 数据流

```
输入 hidden_state [B, H]
    ↓
[Router / Gate: Linear(H → N_experts)]  → 路由分数
    ↓
[分组 Top-K]  → 从 N 里选 K 个专家
    ↓
┌──────────────────┐
│ 共享专家         │ → 必算，不经路由
│ gate_up → SiLU×x │
│ down → 输出     │
└──────────────────┘
        +
┌──────────────────┐
│ 路由专家         │ → 仅选中的 K 个参与计算
│ 对每个激活专家:  │
│   gate_up → SiLU │
│   down → 输出   │
│ 按权重加和      │
└──────────────────┘
    ↓
最终输出 = 共享支路 + 路由支路
```

## 分组 Top-K 路由

不是对全体专家做简单 top-k，而是**先分组再在组内/跨组**选择，例如：

```python
def grouped_topk(scores, n_groups, topk_group, top_k):
    # 1. 将分数 reshape 为组
    grouped = scores.reshape(batch, n_groups, experts_per_group)

    # 2. 先选「重要组」
    group_scores = grouped.max(dim=-1).values
    top_groups = group_scores.topk(topk_group).indices

    # 3. 屏蔽未选组
    mask = zeros_like(scores)
    mask[top_groups] = 1
    scores = scores * mask

    # 4. 在保留分数上做 top-k
    topk_indices = scores.topk(top_k).indices
    topk_weights = softmax(scores[topk_indices])

    return topk_indices, topk_weights
```

有利于让不同组都有专家被用到，减少「扎堆」在相似专家上。

## 融合 MoE 内核

为效率常把 MoE 算子融合在单个内核中：
1. token → 专家 的路由
2. 各激活专家上的 gate+up
3. SiLU
4. down
5. 加权累加

### Triton 融合 MoE 示意（mini-sglang）
```python
def fused_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """
    w1: [num_experts, 2*intermediate_size, hidden_size]  (gate+up 合并)
    w2: [num_experts, hidden_size, intermediate_size]   (down)
    """
    sorted_token_ids, expert_ids, num_tokens_per_expert = sort_by_expert(topk_ids)
    fused_moe_kernel[grid](
        hidden_states, w1, w2,
        sorted_token_ids, expert_ids, num_tokens_per_expert,
        topk_weights, ...
    )
```

## 专家并行（EP）

对超大 MoE，专家可跨 GPU 放置：

### 方式一：专家内部 TP
与稠密模型类似，单专家内权重在 TP 各卡上分片。

### 方式二：专家间 EP
不同专家在不同 GPU 上，例如：
```
Rank 0: Expert[0..63]
Rank 1: Expert[64..127]
...
```
需要 all-to-all 等通信把 token 送到持有对应专家的设备。

## 对推理框架的影响

1. **显存**：专家参数量大（256×FFN），需规划 GPU 显存
2. **算力**：每 token 只激活 K 个专家，等效算力可接近同规模稠密
3. **通信**：EP 需 all-to-all；TP+MoE 需整体设计
4. **内核**：融合 MoE 对性能关键；按专家逐次算会极慢
5. **负载均衡**：token 在各专家上的分布影响吞吐
