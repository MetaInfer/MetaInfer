# 权重加载 — API 契约

> 关联 notebooks: `07_improvementPlan/task10_tp_qwen_debug_experience.md`

## 概述

HF safetensors 权重 → TP 模型属性的 14 键映射 + 拼接顺序 + 构造链。
源实现文件: `engine/models/qwen.py` (load_weights 方法)

---

## HF Key → Attr 映射 (14 个键)

| HF Key Pattern | 模型属性 | 线性层类型 | 拼接规则 |
|---------------|---------|-----------|---------|
| `model.embed_tokens.weight` | `embed_tokens.weight` | VocabParallelEmbedding | dim=0 切片 |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers[i].self_attn.qkv_proj.weight` | QKVColumnParallelLinear | **Q 位前**：`[0:q_size]` |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers[i].self_attn.qkv_proj.weight` | QKVColumnParallelLinear | **K 位中**：`[q_size:q_size+kv_size]` |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers[i].self_attn.qkv_proj.weight` | QKVColumnParallelLinear | **V 位后**：`[q_size+kv_size:]` |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers[i].self_attn.o_proj.weight` | RowParallelLinear | dim=1 切片 |
| `model.layers.{i}.self_attn.q_norm.weight` | `layers[i].self_attn.q_norm.weight` | — | dim=0 切片(per-head) — **Qwen3 特有** |
| `model.layers.{i}.self_attn.k_norm.weight` | `layers[i].self_attn.k_norm.weight` | — | dim=0 切片(per-head) — **Qwen3 特有** |
| `model.layers.{i}.mlp.gate_proj.weight` | `layers[i].mlp.gate_up_proj.weight` | MergedColumnParallelLinear | **Gate 位前**：`[0:half]` |
| `model.layers.{i}.mlp.up_proj.weight` | `layers[i].mlp.gate_up_proj.weight` | MergedColumnParallelLinear | **Up 位后**：`[half:]` |
| `model.layers.{i}.mlp.down_proj.weight` | `layers[i].mlp.down_proj.weight` | RowParallelLinear | dim=1 切片 |
| `model.layers.{i}.input_layernorm.weight` | `layers[i].input_layernorm.weight` | — | 直接复制 (dim=0) |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers[i].post_attention_layernorm.weight` | — | 直接复制 (dim=0) |
| `model.norm.weight` | `norm.weight` | — | 直接复制 |
| `lm_head.weight` | `lm_head.weight` | ParallelLMHead | dim=0 切片 (tied embedding 时 = embed_tokens.weight) |

---

## 关键拼接顺序

### QKV 拼接 (Q-K-V 顺序)
```
weight_chunks = []
weight_chunks.append(_shard(q_weight, 0))    # Q: [q_size, hidden]
weight_chunks.append(_shard(k_weight, 0))    # K: [kv_size, hidden]
weight_chunks.append(_shard(v_weight, 0))    # V: [kv_size, hidden]
qkv_weight = torch.cat(weight_chunks, dim=0)  # [q_size+2*kv_size, hidden]
```

### Gate-Up 拼接 (gate-up 顺序，非 up-gate)
```
gate_up = torch.cat([gate_shard, up_shard], dim=0)  # gate 在前
```

---

## 防双切片 Guard (强制)

```python
def load_weight_shard(self, weight_tensor):
    if weight_tensor.shape == self.weight.shape:
        self.weight.copy_(weight_tensor.to(self.weight.dtype))
        return  # 已预分片，直接复制
    # 仅当传入权重为全量时才切片
    chunk = _slice(weight_tensor, dim, tp_rank, tp_size)
    self.weight.copy_(chunk.to(self.weight.dtype))
```

---

## 构造链 (5 步)

```python
cfg = QwenTPConfig.from_model_dir(model_dir, tp_size, tp_rank)
model = QwenForCausalLMTP(cfg)
model = model.to(torch.bfloat16).cuda()
model.load_weights(model_dir)
model.eval()
```

---

## 陷阱与反模式

- **Bug 6 (Phase 9)**: `_dispatch_weight` 遗漏 q_norm.weight / k_norm.weight → 模型静默产生错误 logits
- **FM-001**: 双重切片 — safetensors `get_slice` 可能已预分片
- **FM-006**: QKV 拼接索引偏移 — Q `[0:q_size]`, K `[q_size:q_size+kv_size]`, V `[q_size+kv_size:]`
- **FM-018**: RowParallel o_proj `_row_slice` size 必须除以 tp_size
- **LOAD-001**: Qwen3 特有 q_norm/k_norm (per-head RMSNorm) — HF 权重中有对应 key
- **LOAD-002**: tied embedding — `lm_head.weight` = `model.embed_tokens.weight`
- **LOAD-003**: gate-up 顺序为 gate 前 up 后 (非 up-gate)

---

## Qwen3.5/3.6 HF Key Mapping (Hybrid Model)

<!-- source: evolution/evo-001, timestamp=2026-07-03 -->

### Key Prefix 差异

Qwen3.5/3.6 的 HF key 前缀为 `model.language_model.*`（而非 Qwen3-8B 的 `model.*`）。
因为 Qwen3_5ForConditionalGeneration 包装了 Qwen3_5Model 的 `.language_model` 子模块。
Weight loading 的正则 pattern 必须匹配 `model\.language_model\.layers\.(\d+)\.(.+)`.

### Full Attention 层 Key 映射 (16 layers)

| HF Key Pattern | 模型属性 | 线性层类型 | 拼接规则 |
|---------------|---------|-----------|---------|
| `model.language_model.layers.{i}.self_attn.q_proj.weight` | `layers[i].self_attn.q_proj` | ColumnParallelLinear | 双倍输出: Q + gate, q_size*2 per rank |
| `model.language_model.layers.{i}.self_attn.k_proj.weight` | `layers[i].self_attn.k_proj` | ColumnParallelLinear | dim=0 切片 |
| `model.language_model.layers.{i}.self_attn.v_proj.weight` | `layers[i].self_attn.v_proj` | ColumnParallelLinear | dim=0 切片 |
| `model.language_model.layers.{i}.self_attn.o_proj.weight` | `layers[i].self_attn.o_proj` | RowParallelLinear | dim=1 切片 |
| `model.language_model.layers.{i}.self_attn.q_norm.weight` | `layers[i].self_attn.q_norm` | — | 直接复制 (dim=0, standard RMSNorm) |
| `model.language_model.layers.{i}.self_attn.k_norm.weight` | `layers[i].self_attn.k_norm` | — | 直接复制 (dim=0, standard RMSNorm) |

**注意**: FullAttention 使用分离的 q_proj/k_proj/v_proj（非 Qwen3-8B 的合并 qkv_proj）。q_proj 输出为双倍大小：前半为 Q，后半为 output gate。

### Linear Attention 层 Key 映射 (48 layers)

| HF Key Pattern | 模型属性 | 线性层类型 | 拼接规则 |
|---------------|---------|-----------|---------|
| `model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight` | `layers[i].linear_attn.in_proj_qkv` | _DeltaNetQKVColumnParallelWeight | Head-aware 3段非连续切片: Q[0:512]+K[2048:2560]+V[4096:5632] per rank |
| `model.language_model.layers.{i}.linear_attn.in_proj_z.weight` | `layers[i].linear_attn.in_proj_z` | ColumnParallelLinear | dim=0 切片 |
| `model.language_model.layers.{i}.linear_attn.in_proj_a.weight` | `layers[i].linear_attn.in_proj_a` | nn.Linear (Replicate) | 直接复制 (全量 48 heads) |
| `model.language_model.layers.{i}.linear_attn.in_proj_b.weight` | `layers[i].linear_attn.in_proj_b` | nn.Linear (Replicate) | 直接复制 (全量 48 heads) |
| `model.language_model.layers.{i}.linear_attn.A_log` | `layers[i].linear_attn.A_log` | — | dim=0 切片: 48 → 12 per rank |
| `model.language_model.layers.{i}.linear_attn.dt_bias` | `layers[i].linear_attn.dt_bias` | — | dim=0 切片: 48 → 12 per rank |
| `model.language_model.layers.{i}.linear_attn.conv1d.weight` | `layers[i].linear_attn.conv1d_weight` | — | dim=0 切片: [10240,1,4] → [2560,1,4] per rank |
| `model.language_model.layers.{i}.linear_attn.norm.weight` | `layers[i].linear_attn.norm` | Qwen3_5RMSNormGated | 直接复制 (Replicate) |
| `model.language_model.layers.{i}.linear_attn.out_proj.weight` | `layers[i].linear_attn.out_proj` | RowParallelLinear | dim=1 切片 |

### 共享 Key (所有 64 layers)

| HF Key Pattern | 模型属性 | 注意事项 |
|---------------|---------|---------|
| `model.language_model.layers.{i}.input_layernorm.weight` | `layers[i].input_layernorm` (Qwen3_5RMSNorm) | zeros-init, 直接复制 |
| `model.language_model.layers.{i}.post_attention_layernorm.weight` | `layers[i].post_attention_layernorm` (Qwen3_5RMSNorm) | zeros-init, 直接复制 |
| `model.language_model.layers.{i}.mlp.gate_proj.weight` | `layers[i].mlp.gate_up_proj` | gate 前 + up 后，需 buffer 组装 |
| `model.language_model.layers.{i}.mlp.up_proj.weight` | `layers[i].mlp.gate_up_proj` | gate 前 + up 后，需 buffer 组装 |
| `model.language_model.layers.{i}.mlp.down_proj.weight` | `layers[i].mlp.down_proj` | RowParallelLinear dim=1 切片 |

### 顶层 Key

| HF Key Pattern | 模型属性 | 注意事项 |
|---------------|---------|---------|
| `model.language_model.embed_tokens.weight` | `embed_tokens` | VocabParallelEmbedding dim=0 切片 |
| `model.language_model.norm.weight` | `norm` (Qwen3_5RMSNorm) | zeros-init, 直接复制 |
| `lm_head.weight` | `lm_head` | ParallelLMHead dim=0 切片 |

### 新增陷阱

- **LOAD-Q35-001**: 正则 pattern 必须匹配 `model\.language_model\.layers\.(\d+)\.(.+)`。若沿用 Qwen3-8B 的 `model\.layers\.` → 所有层权重静默跳过
- **LOAD-Q35-002**: in_proj_qkv 的 shard 不是连续切片，而是 Q/K/V 三段非连续区域的各 per-rank 拼接: Q[rank*512:(rank+1)*512] + K[2048+rank*512:2048+(rank+1)*512] + V[4096+rank*1536:4096+(rank+1)*1536]
- **LOAD-Q35-003**: A_log 和 dt_bias 需要 TP 切分: 48 → 12 per rank
- **LOAD-Q35-004**: Qwen3_5RMSNorm weight 为 zeros-init，但加载时直接 copy——不做任何变换。公式 (1+w) 在 forward 中处理
