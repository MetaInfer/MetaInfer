# Native Sparse Attention (NSA)

## 概述

Native Sparse Attention (NSA) 是 DeepSeek V3.2 引入的稀疏注意力机制，通过智能索引和块级注意力实现长序列的高效处理。

## 核心思想

### 问题背景

传统注意力在长序列上的复杂度：
```
复杂度 = O(seq_len²)
显存 = seq_len × seq_len × num_heads × head_dim
```

对于 128K 上下文：
- 显存需求：~40GB（仅注意力矩阵）
- 计算量：巨大

### NSA 解决方案

通过块级索引和稀疏注意力：
1. **块级选择**：将序列分块，只选择重要块
2. **Top-K 索引**：预先计算哪些块需要完整注意力
3. **压缩 KV Cache**：对不重要的块使用压缩表示

## 架构设计

### NSA 结构

```
DeepseekV2MLAAttention (with NSA)
├── indexer (Indexer)                    # 稀疏索引器
│   ├── wq_b (Linear)                    # Query 投影
│   ├── wk_weights_proj (Linear)         # Key + Weight 投影 (融合)
│   ├── k_norm (LayerNorm)               # Key 归一化
│   └── k_cache (DeepseekV32IndexerCache)
│
├── MLA 组件
│   ├── fused_qkv_a_proj
│   ├── kv_b_proj
│   └── ...
│
└── mla_attn (MLAAttention)
```

### 索引器流程

```
输入 hidden_states [num_tokens, hidden_size]
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
MLA QKV 投影                          Indexer:
    │                                 wq_b(q_c) → Query for index
    │                                     │
    │                                 wk(hidden) → Key for index
    │                                     │
    │                                 weights_proj → 重要性权重
    │                                     │
    │                                 k_norm → 归一化
    │                                     │
    │                                 RoPE → 位置编码
    │                                     │
    │                                 FP8 量化
    │                                     │
    │                                     ▼
    │                                 Top-K 选择 → topk_indices
    │                                     │
    └─────────────────────────────────────┘
                                          │
                                          ▼
                               Sparse Attention
                              (只在选中的块上计算)
```

## 代码实现

### Indexer 结构

```python
class Indexer(nn.Module):
    def __init__(self, vllm_config, config, hidden_size, q_lora_rank, ...):
        super().__init__()
        
        # NSA 参数
        self.topk_tokens = config.index_topk      # 32
        self.n_head = config.index_n_heads        # 64
        self.head_dim = config.index_head_dim     # 128
        self.rope_dim = config.qk_rope_head_dim   # 64
        
        # Q 投影
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
        )
        
        # 融合的 K + Weights 投影
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],  # K + importance weights
            bias=False,
        )
        
        # Key 归一化
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        
        # 索引缓存
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // 128 * 4,
            dtype=torch.uint8,  # FP8
        )
        
        # 稀疏注意力索引操作
        self.indexer_op = SparseAttnIndexer(...)
```

### 索引前向传播

```python
def forward(self, hidden_states, qr, positions, rotary_emb):
    """
    计算稀疏注意力索引
    
    Args:
        hidden_states: [num_tokens, hidden_size]
        qr: Q 的 latent 表示
        positions: 位置编码
        rotary_emb: RoPE 实现
    
    Returns:
        topk_indices: 选中的块索引
    """
    # 1. 计算 Query for index
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)
    q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
    
    # 2. 融合计算 Key + Importance Weights
    kw, _ = self.wk_weights_proj(hidden_states)
    k = kw[:, :self.head_dim]
    weights = kw[:, self.head_dim:]
    
    # 3. 归一化并应用 RoPE
    k = self.k_norm(k)
    k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
    
    # 4. 组装完整 Q, K
    q = torch.cat([q_pe, q_nope], dim=-1)
    k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)
    
    # 5. FP8 量化
    q_fp8, q_scale = per_token_group_quant_fp8(
        q.view(-1, self.head_dim),
        self.quant_block_size,
    )
    
    # 6. 计算重要性权重
    weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
    weights = weights.squeeze(-1)
    
    # 7. 执行稀疏索引操作
    return self.indexer_op(hidden_states, q_fp8, k, weights)
```

### SGLang NSA Indexer

```python
class Indexer(nn.Module):
    """SGLang NSA Indexer 实现"""
    
    def __init__(self, hidden_size, index_n_heads, index_head_dim, 
                 rope_head_dim, index_topk, q_lora_rank, ...):
        # Q 投影
        self.wq_b = ColumnParallelLinear(
            q_lora_rank,
            index_n_heads * index_head_dim,
            bias=False,
        )
        
        # K 投影
        self.wk = ReplicatedLinear(
            hidden_size,
            index_head_dim,
            bias=False,
        )
        
        # 重要性权重投影
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            index_n_heads,
            bias=False,
        )
```

## Skip Top-K 优化

### 跨层复用索引

DeepSeek V3.2 支持在连续层间复用索引结果：

```python
# 配置
index_topk_freq = 2  # 每2层重新计算索引

# 层配置
layer_idx = 10
skip_topk = (layer_idx - 1) % index_topk_freq != 0  # True: 复用上层索引
next_skip_topk = layer_idx % index_topk_freq != 0   # 下层是否复用
```

### 实现

```python
def forward(self, positions, hidden_states, forward_batch, ..., prev_topk_indices=None):
    # 判断是否复用上层索引
    if self.skip_topk and prev_topk_indices is not None:
        # 复用上层的 topk 索引
        topk_indices = prev_topk_indices
    else:
        # 计算新的 topk 索引
        topk_indices = self.indexer(hidden_states, q_c, positions, rotary_emb)
    
    # 执行稀疏注意力
    attn_output = self.mla_attn(q, kv_c_normed, k_pe, topk_indices)
    
    return attn_output, topk_indices if self.next_skip_topk else None
```

## KV Cache 格式

### 索引器 KV Cache

```python
# FP8 索引器缓存
k_cache_shape = (
    max_seq_len,
    head_dim + head_dim // block_size * 4  # 数据 + scale
)

# 存储
dtype = torch.uint8  # FP8
```

### 主 KV Cache

主 KV Cache 使用 MLA 的压缩格式：
```python
# MLA KV Cache
kv_cache_shape = (max_seq_len, kv_lora_rank + qk_rope_head_dim)
```

## Context Parallel 支持

### CP + NSA 配置

```python
# 启用 Context Parallel + NSA
enable_nsa_prefill_cp = True

# CP 分割
cp_size = get_attention_cp_size()  # 例如 4

# 序列分割
seq_len = 128000
chunk_len = seq_len // cp_size  # 32000
```

### CP 前向传播

```python
def forward_with_cp(hidden_states, positions, forward_batch):
    if nsa_use_prefill_cp(forward_batch):
        # 1. 分割输入
        hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
        positions = cp_split_and_rebuild_position(forward_batch, positions)
    
    # 2. 正常前向传播
    output = self.forward(hidden_states, positions, ...)
    
    # 3. 合并输出
    if nsa_use_prefill_cp(forward_batch):
        output = cp_all_gather_rerange_output(output, self.cp_size, forward_batch)
    
    return output
```

## 性能特点

### 内存节省

| 序列长度 | 标准 Attention | NSA | 节省 |
|----------|----------------|-----|------|
| 32K | 16 GB | 2 GB | 87.5% |
| 64K | 64 GB | 4 GB | 93.7% |
| 128K | 256 GB | 8 GB | 96.9% |

### 计算加速

| 序列长度 | 标准 Attention | NSA | 加速比 |
|----------|----------------|-----|--------|
| 32K | 1.0x | 4x | 4x |
| 64K | 1.0x | 8x | 8x |
| 128K | 1.0x | 16x | 16x |

## 配置参数

### V3.2 新增参数

```python
@dataclass
class DeepseekV32Config:
    # NSA 参数
    index_topk: int = 32              # Top-K 块数
    index_n_heads: int = 64           # 索引头数
    index_head_dim: int = 128         # 索引头维度
    index_topk_freq: int = 1          # 索引计算频率
    
    # 可选的索引模式配置
    index_topk_pattern: List[str] = None  # 例如 ["C", "S", "S", "C", ...]
    # C = Compute, S = Skip (复用上层)
```

### 使用示例

```bash
# 启用 NSA (V3.2)
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3.2 \
    --tp 8 \
    --context-length 128000
```

## 实现差异

| 特性 | vLLM | SGLang |
|------|------|--------|
| Indexer 类 | Indexer | Indexer |
| 融合 wk + weights | ✅ (FP4 模式) | 分离 |
| FP8 Cache | ✅ | ✅ |
| Skip Top-K | ✅ | ✅ |
| CP 支持 | ✅ | ✅ |
| NPU 支持 | - | ✅ |
