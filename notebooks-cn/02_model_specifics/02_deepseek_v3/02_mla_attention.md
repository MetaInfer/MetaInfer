# 多潜注意力（MLA）

## 核心思想

MLA 通过低秩瓶颈压缩 Key 与 Value 的投影，在保持接近全 MHA 表达能力的同时，大幅减小 KV cache 规模。

## 与标准 Attention 的对比

### 标准 MHA / GQA
```
hidden_state [B, H]
    ↓
Q_proj: [H → N_q × D_h]     → Q [B, N_q, D_h]
K_proj: [H → N_kv × D_h]    → K [B, N_kv, D_h]
V_proj: [H → N_kv × D_h]    → V [B, N_kv, D_h]
    ↓
KV Cache：每层每 token 存 K [N_kv × D_h] + V [N_kv × D_h]
```

### MLA
```
hidden_state [B, H]
    ↓
kv_a_proj: [H → kv_lora_rank + qk_rope_head_dim]  → compressed_kv [B, R+D_rope]
    ↓
切分:  c_kv [B, R]  (潜向量)
      k_rope [B, D_rope]  (Key 的 RoPE 部分)
    ↓
kv_b_proj: [R → N_kv × (D_nope + D_v)]  → K_nope, V  (上投影)
    ↓
K = concat(K_nope, RoPE(k_rope))
    ↓
KV Cache：每层每 token 存 c_kv [R] + k_rope [D_rope]
```

## 关键维度

```python
kv_lora_rank = 512         # 瓶颈维 R
qk_nope_head_dim = 128     # Q/K 中不含位置部分
qk_rope_head_dim = 64      # Q/K 的 RoPE 维
v_head_dim = 128            # V 头维
num_heads = 128             # attention 头数
```

## KV 占用对比

| 方法 | 每层每 token cache | 例：128 头 × 128 维 |
|--------|----------------------|----------------------|
| MHA | `2 × N_h × D_h` | 2 × 128 × 128 = 32,768 |
| GQA（8 组） | `2 × N_kv × D_h` | 2 × 8 × 128 = 2,048 |
| MLA | `R + D_rope` | 512 + 64 = 576 |

与 MHA 相比，MLA 可约 **56 倍** 压缩；与 GQA-8 相比约 **3.6 倍**，质量仍可接受。

## 推理模式

### 「吸收」（Absorb）模式（面向推理优化）
关键观察：`kv_b_proj` 在数学上可吸收进 Q 投影与输出投影，使得 attention 可直接在压缩后的潜变量上计算。

```
标准: Q × K^T = Q × (W_UK × c_kv)^T = (Q × W_UK^T) × c_kv^T
吸收: Q_absorbed = Q × W_UK^T，再算 Q_absorbed × c_kv^T
```

因此：
1. **Prefill**：cache 中只存 `c_kv` 与 `k_rope`（不存完整 K、V）
2. **Decode**：用压缩后的 cache 直接做 attention
3. **`kv_b_proj` 的权重** 会折叠进 Q 与 O 的投影

### 解耦 RoPE
仅 Q、K 的一小部分应用 RoPE：
```python
q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
k_nope, k_rope = k.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

# 仅对 rope 段应用 RoPE
q_rope = apply_rope(q_rope, positions)
k_rope = apply_rope(k_rope, positions)

# 再拼接
q = concat(q_nope, q_rope)
k = concat(k_nope, k_rope)
```

## 对推理框架的影响

### KV Cache 结构
```python
# 标准: [2, layers, blocks, block_size, kv_heads, head_dim]
# MLA: 更小的每 token 占用
kv_cache_shape = [layers, blocks, block_size, kv_lora_rank + qk_rope_head_dim]
```

### Attention 内核
标准 FlashAttention 无法直接套用在 MLA 上，因为 Q·K 的维数与全维 MHA 不同。需要专用内核，例如：
- **FlashMLA**：面向 MLA 的定制内核
- **吸收后 attention**：先把 Q 左乘 `W_UK^T` 再调用类标准 attention

### 权重命名
MLA 的投影与标准 `q_proj, k_proj, v_proj, o_proj` 不同，例如：
```python
# 常见为: q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj
# 可能还有 q_a_layernorm, kv_a_layernorm
```
