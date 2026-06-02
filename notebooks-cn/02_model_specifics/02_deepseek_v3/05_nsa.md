# DeepSeek V3 — 原生稀疏注意力（NSA）

## 核心概念

对超长序列，全量 attention 的平方复杂度代价过高。NSA 在保留局部滑窗的同时，**只对部分「重要」token 做全连接 attention**，并可用压缩后的全局表示，从而在长度上更接近次线性。

## 结构

```
位置 t 的 Query
    ↓
三路 attention 分支:
    ├── [滑窗]  → 始终 attend 最近 W 个 token
    ├── [稀疏选择] → attend 分数最高的若干 token 块
    └── [压缩全局] → attend 对全序列的压缩 K/V
    ↓
三路输出加权组合
```

## 稀疏选择机制

### 索引子模块（Indexer）
用轻量网络估计哪些 token「重要」：

```python
class NSAIndexer:
    def __init__(self):
        self.proj = Linear(hidden_size, index_dim)

    def compute_importance(self, hidden_states):
        index_features = self.proj(hidden_states)
        scores = query_features @ index_features.T
        block_scores = scores.reshape(-1, block_size).mean(dim=-1)
        top_k_blocks = block_scores.topk(K).indices
        return top_k_blocks
```

### Attention 计算
```python
def nsa_attention(q, k, v, sliding_window, selected_blocks, compressed_kv):
    local_out = sliding_window_attention(q, k_local, v_local, window_size)
    sparse_k, sparse_v = gather_blocks(k, v, selected_blocks)
    sparse_out = attention(q, sparse_k, sparse_v)
    compressed_out = attention(q, compressed_k, compressed_v)
    output = gate_local * local_out + gate_sparse * sparse_out + gate_global * compressed_out
    return output
```

## 对推理框架的影响

1. **KV 索引**：需要能同时支持稠密（滑窗）与稀疏（按块选读）的 `TokenToKV` 等池
2. **Top-K 开销**：在真正做 attention 前多一步索引/打分
3. **多模式合一**：同一层要支持多种 attention 模式
4. **块粒度**：稀疏选择在「块」上选，不是逐 token
5. **显存**：全量 attention 显存可降，但会增 indexer 等参数

## 何时适合

- **适合**：极长上下文（如 >32K）且全量 attention 过慢/过大
- **不太适合**：短序列本身已很快
- **取舍**：结构更复杂，依赖精确长程依赖的任务可能略损质量
