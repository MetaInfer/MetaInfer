# TP 线性层 — API 契约

> 关联 notebooks: `04_parallel_strategies/qwen_dense_tp_implementation_guide.md`

## 概述

4 种 TP 线性层: ColumnParallelLinear, RowParallelLinear, QKVColumnParallelLinear, MergedColumnParallelLinear。
源实现文件: `engine/tp_layers/linear.py`

---

## 接口签名

### ColumnParallelLinear

- **构造**: `ColumnParallelLinear(hidden_size, output_size_per_partition, bias=False, gather_output=True)`
- **weight shape**: `[out/tp, in]`
- **输入**: `[B, T, in]`
- **输出 (no gather)**: `[B, T, out/tp]`
- **输出 (gather)**: `[B, T, out]`

### RowParallelLinear

- **构造**: `RowParallelLinear(input_size, output_size, bias=False)`
- **weight shape**: `[out, in/tp]`
- **输入**: `[B, T, in/tp]`
- **partial output**: `[B, T, out]`
- **output after all_reduce**: `[B, T, out]`

### QKVColumnParallelLinear

- **构造**: `QKVColumnParallelLinear(hidden_size, head_dim, total_num_heads, total_num_kv_heads, bias=False, gather_output=False)`
- **weight shape**: `[q_size + 2*kv_size, hidden_size]` per rank (前 Q 中 K 后 V)
- **head_dim**: `cfg.head_dim` (128 for Qwen3-8B)，不能用 `head_size`
- **q_size**: `num_heads * head_dim // tp_size`
- **kv_size**: `num_kv_heads_local * head_dim` (where num_kv_heads_local = max(1, total_num_kv_heads // tp_size))

```python
def forward(self, x):  # x: [B, T, hidden_size] e.g. [1,1,4096]
    y = F.linear(x, self.weight)  # [1,1, q_size+2*kv_size]
    if self.gather_output and self.tp_size > 1:
        y = all_gather_last_dim(y)
    q, k, v = y.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    return q, k, v

# Caller-side reshape (in QwenAttentionTP.forward):
# q = q.view(B, T, self.num_heads, self.head_dim)       # [1,1,8,128]
# k = k.view(B, T, self.num_kv_heads, self.head_dim)    # [1,1,2,128]
# v = v.view(B, T, self.num_kv_heads, self.head_dim)    # [1,1,2,128]
# WARNING: K/V reshape MUST use self.num_kv_heads (per-rank local), NOT self.num_heads
```

### MergedColumnParallelLinear

- **构造**: `MergedColumnParallelLinear(hidden_size, intermediate_size, bias=False, gather_output=False)`
- **gate_up weight**: `[2*intermediate/tp, hidden_size]` per rank
- **输入**: `[B, T, hidden_size]`
- **输出**: `[B, T, 2*intermediate/tp]` (前 gate 后 up)

---

## 权重加载 (防双切片)

```python
def load_weight_shard(self, weight_tensor):
    if weight_tensor.shape == self.weight.shape:
        self.weight.copy_(weight_tensor)  # 已预分片 → 直接复制
    else:
        # 按全量权重切片 → 根据 tp_rank 取对应分片
        ...
```

---

## KV Head Replication

当 `tp_size > num_kv_heads` 时 (如 Qwen3-8B tp=4, kv_heads=8 → 不触发):
```python
if cfg.num_key_value_heads >= tp_size:
    self.num_kv_heads = cfg.num_key_value_heads // tp_size  # 2
else:
    self.num_kv_heads = 1  # replicated
    self.kv_head_replica = tp_size // cfg.num_key_value_heads
```

---

## 维度验证 (示例: Qwen3-8B TP=4)

> **⚠️ 这是特定模型的验证示例，不是通用规范。** 不同模型/TP size 会产生不同的 per-rank 维度。
> 表中数值 = 模型全量值 / tp_size 的计算结果。

| 参数 | 计算方式 | 示例值 (Qwen3-8B, TP=4) |
|------|---------|----------------------|
| hidden_size | config.json → hidden_size | 4096 |
| intermediate_size | config.json → intermediate_size | 12288 |
| num_attention_heads | config.json → num_attention_heads | 32 |
| num_key_value_heads | config.json → num_key_value_heads | 8 |
| head_dim | config.json → head_dim | 128 |
| q_size per rank | num_heads * head_dim | 1024 (8×128) |
| kv_size per rank | num_kv_heads_local * head_dim | 256 (2×128) |
| qkv_weight per rank | [q_size+2*kv_size, hidden_size] | [1536, 4096] |
| gate_up_weight per rank | [2*intermediate/tp, hidden_size] | [6144, 4096] |
| intermediate per rank | intermediate_size / tp_size | 3072 |

---

## 陷阱与反模式

- **FM-001**: TP Embedding 双重切片 — load_weight_shard 必须先检查 shape
- **FM-006**: QKV weight 拼接索引 — 按 `[0:q_size]`, `[q_size:q_size+kv_size]`, `[q_size+kv_size:]` 三段复制
- **FM-018**: RowParallel o_proj `_row_slice` 的 size 参数必须除以 tp_size
- **LINEAR-001**: head_dim = 128 (Qwen3-8B)，不能用 head_size
- **LINEAR-002**: K/V reshape 用 `self.num_kv_heads` (per-rank local = 2)，不能用 `self.num_heads` (= 8)
- **LINEAR-003**: 维度值来自 config.json 动态读取，禁止硬编码
