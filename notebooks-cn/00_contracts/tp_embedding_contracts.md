# TP Embedding & LM Head — API 契约

> 关联 notebooks: —

## 概述

VocabParallelEmbedding (mask+masked_fill) + ParallelLMHead (all_gather 拼接)。
源实现文件: `engine/tp_layers/embedding.py`

---

## 接口签名

### VocabParallelEmbedding

- **input_ids**: `[B, T]` int64
- **local_weight**: `[vocab_size/tp, hidden_size]`
- **输出**: `[B, T, hidden_size]` (after all_reduce_sum)

```python
def forward(self, input_ids):  # [B,T] int64
    mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
    local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)
    out = F.embedding(local_ids, self.weight)  # [B,T,embedding_dim]
    out = out.masked_fill((~mask).unsqueeze(-1), 0)
    return all_reduce_sum(out)

# vocab_start = tp_rank * (vocab_size // tp_size)
# vocab_end = vocab_start + local_vocab_size
```

**关键语义**:
- 每个 rank 持有一部分词表权重
- 超出本地词表范围的 token → mask 置 0
- 最后 `all_reduce_sum` 将各 rank 贡献求和 (某个 rank 的非零值 + 其他 rank 的 0 = 完整 embedding)

### ParallelLMHead

- **input_hidden**: `[B, T, hidden_size]`
- **local_logits**: `[B, T, vocab_size/tp]`
- **output logits**: `[B, T, vocab_size]` (after all_gather_last_dim)

```python
def forward(self, x):  # x: [B, T, hidden_size]
    local_logits = F.linear(x, self.weight)  # [B, T, vocab_size/tp]
    if self.gather_output:
        return all_gather_last_dim(local_logits)  # [B, T, vocab_size]
    return local_logits
```

---

## 数据流约束

- **VocabParallelEmbedding**: mask → masked_fill → all_reduce_sum (非 all_gather)
- **ParallelLMHead**: F.linear → all_gather_last_dim
- **权重加载防双切片**: 同 tp_linear_contracts.md 规则

---

## 陷阱与反模式

- **FM-001**: TP Embedding 双重切片 — `_load_tensor(split_dim=0)` 已按 tp_rank 取本地分片，load_weight_shard 不得二次切片
- **EMBED-001**: VocabParallelEmbedding 最后是 `all_reduce_sum`，非 `all_gather` (语义不同)
- **EMBED-002**: mask+masked_fill 是必需的 — 直接 F.embedding(global_input_ids, local_weight) 会导致越界
