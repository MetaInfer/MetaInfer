# Multi-Head Latent Attention (MLA)

## 概述

MLA (Multi-Head Latent Attention) 是 DeepSeek V2/V3 的核心创新之一。通过低秩压缩技术，MLA 大幅减少了 KV Cache 的存储需求，同时保持模型性能。

## 核心思想

### 传统 Attention vs MLA

**传统 Multi-Head Attention**:
```
KV Cache 大小 = 2 × num_layers × batch_size × seq_len × num_heads × head_dim
```

**MLA**:
```
KV Cache 大小 = 2 × num_layers × batch_size × seq_len × (kv_lora_rank + qk_rope_head_dim)
```

### 参数对比

| 参数 | 传统 MHA | MLA | 压缩比 |
|------|----------|-----|--------|
| K 维度 | 128 × 128 = 16384 | 512 + 64 = 576 | 28.4x |
| V 维度 | 128 × 128 = 16384 | 512 (latent) | 32x |
| **总计** | 32768 | 576 | **56.9x** |

## 架构设计

### QKV 投影结构

```
输入 hidden_states (B, S, H=7168)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
q_a_proj (H → q_lora_rank=1536)    kv_a_proj_with_mqa (H → kv_lora_rank + qk_rope_dim = 576)
    │                                     │
    ▼                                     ├──────────────────┐
q_a_layernorm                              ▼                  ▼
    │                               kv_c (512)           k_pe (64)
    ▼                                     │                  │
q_b_proj (1536 → 128 × 192)               ▼                  │
    │                               kv_a_layernorm          │
    │                                     │                  │
    ▼                                     ▼                  │
q (B, S, 128, 192)                   kv_b_proj              │
    │                          (512 → 128 × (128+128))      │
    ├───────────────┐                        │              │
    ▼               ▼                        ▼              │
q_nope (128)   q_pe (64)              k_nope, v             │
    │               │                        │              │
    │               ◄────────────────────────┼──────────────┘
    │                       RoPE             │
    ▼               ▼                        ▼
    └───────────────┴────────────────────────┘
                          │
                          ▼
                     Attention
                          │
                          ▼
                      o_proj (128×128 → 7168)
```

### 维度分解

```python
# Q 维度分解
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 128 + 64 = 192
# Q: [batch, seq, num_heads, qk_head_dim]
# Q_nope: [..., :128]  # 非旋转部分
# Q_pe:   [..., 128:]  # 旋转位置编码部分

# K 维度分解
# K_c: 压缩后的 KV latent, [batch, seq, kv_lora_rank=512]
# K_pe: 旋转位置编码, [batch, seq, qk_rope_head_dim=64]

# V 维度
# V_c: 与 K_c 共享压缩表示, kv_lora_rank=512
```

## 推理实现

### Prefill 阶段：MQA 模式

在 Prefill 阶段，使用 Multi-Query Attention 模式：

```python
def forward_absorb_prefill(q, kv_c_normed, k_pe):
    """
    Prefill 使用 MQA 模式：
    - Q: [num_tokens, num_heads, qk_head_dim]
    - K: 从 kv_c_normed 恢复，共享 1 个 KV head
    - V: 从 kv_c_normed 恢复
    """
    # 1. 从 latent 恢复 KV
    # kv_b_proj 输出: [num_tokens, num_heads, qk_nope_head_dim + v_head_dim]
    kv = kv_b_proj(kv_c_normed)  # [B, num_heads, 256]
    k_nope, v = kv.split([qk_nope_head_dim, v_head_dim], dim=-1)
    
    # 2. 组装完整的 K
    k = torch.cat([k_nope, k_pe.expand(-1, num_heads, -1)], dim=-1)
    
    # 3. 执行注意力计算
    attn_output = flash_attention(q, k, v)
    
    return attn_output
```

### Decode 阶段：Weight Absorption

在 Decode 阶段，使用权重吸收优化：

```python
def forward_absorb_decode(q_nope, kv_c_normed, k_pe):
    """
    Decode 使用权重吸收：
    - 将 kv_b_proj 融合到 attention 计算中
    - 避免显式恢复完整的 KV
    """
    # 1. Q_nope 与 W_kv 相乘
    # q_nope: [1, num_heads, qk_nope_head_dim]
    # w_kc: [num_heads, qk_nope_head_dim, kv_lora_rank]
    q_nope_out = torch.bmm(q_nope, w_kc)  # [1, num_heads, kv_lora_rank]
    
    # 2. 与压缩的 KV latent 做注意力
    # kv_c_cache: [seq_len, kv_lora_rank]
    attn_weights = torch.matmul(q_nope_out, kv_c_cache.T)
    
    # 3. 加上 RoPE 部分
    q_pe_out = torch.bmm(q_pe, w_kr)
    attn_weights += torch.matmul(q_pe_out, k_pe_cache.T)
    
    # 4. Softmax 并与 V 权重结合
    attn_output = torch.matmul(attn_weights, kv_c_cache)
    attn_output = torch.bmm(attn_output, w_vc)  # 恢复 V
    
    return attn_output
```

### 权重吸收原理

```
传统方式:
  K = W_k @ kv_c    →    Attention(Q, K, V)
  V = W_v @ kv_c

权重吸收后:
  Q' = Q @ W_k      →    Attention(Q', kv_c, kv_c) @ W_v
```

通过矩阵乘法结合律，将 K/V 的恢复与 Attention 计算融合，减少内存访问。

## 代码实现

### vLLM MLA Modules 结构

```python
@dataclass
class MLAModules:
    """MLA 所需的模块集合"""
    kv_a_layernorm: nn.Module      # KV 压缩后的 LayerNorm
    kv_b_proj: nn.Module           # KV 上投影
    rotary_emb: nn.Module          # 旋转位置编码
    o_proj: nn.Module              # 输出投影
    fused_qkv_a_proj: nn.Module    # 融合的 Q/KV 下投影
    kv_a_proj_with_mqa: nn.Module  # 非融合时的 KV 下投影
    q_a_layernorm: nn.Module       # Q 压缩后的 LayerNorm
    q_b_proj: nn.Module            # Q 上投影
    q_proj: nn.Module              # 非 LoRA 时的 Q 投影
    indexer: nn.Module             # NSA 索引器 (V3.2)
    is_sparse: bool                # 是否使用稀疏注意力
```

### 完整前向传播

```python
class MultiHeadLatentAttentionWrapper(nn.Module):
    def forward(self, positions, hidden_states):
        # 1. QKV 投影
        if self.q_lora_rank is not None:
            # 融合投影
            qkv_lora = self.fused_qkv_a_proj(hidden_states)
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim]
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)
        else:
            # 非融合投影
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)
            q = self.q_proj(hidden_states)
        
        # 2. 分离 KV latent 和 RoPE
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim])
        kv_c_normed = self.kv_a_layernorm(kv_c)
        
        # 3. 应用 RoPE
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe
        )
        
        # 4. 执行注意力
        attn_out = self.mla_attn(q, kv_c_normed, k_pe)
        
        # 5. 输出投影
        return self.o_proj(attn_out)
```

## KV Cache 格式

### 存储

```python
# MLA KV Cache 只存储压缩的 latent
kv_cache_shape = (max_seq_len, kv_lora_rank + qk_rope_head_dim)
# = (max_seq_len, 512 + 64) = (max_seq_len, 576)
```

### FP8 KV Cache

DeepSeek V3 支持FP8 KV Cache以进一步减少显存：

```python
# FP8 KV Cache 配置
kv_cache_dtype = torch.uint8  # FP8 存储
scale_format = "ue8m0"        # Block-wise scale
block_size = 128              # 量化块大小
```

## Attention Backend 支持

### vLLM Backend 选择

| Backend | 适用场景 | 特点 |
|---------|----------|------|
| FlashMLA | Decode | DeepSeek 官方优化 |
| FlashInfer MLA | 通用 | FlashInfer 库实现 |
| Triton MLA | 兼容性 | 纯 Triton 实现 |
| ROCm AITER | AMD GPU | AMD 平台优化 |

### SGLang Backend 选择

| Backend | Prefill | Decode | 特点 |
|---------|---------|--------|------|
| FlashAttention3 | ✅ | ✅ | 默认，广泛优化 |
| FlashInfer MLA | ✅ | ✅ | FlashInfer 实现 |
| FlashMLA | - | ✅ | DeepSeek 官方 |
| CutlassMLA | ✅ | ✅ | NVIDIA Cutlass |
| TRTLLM MLA | ✅ | ✅ | Blackwell 优化 |
| Triton | ✅ | ✅ | 兼容性好 |

## DP Attention

Data Parallel Attention 是 SGLang 的优化特性：

```python
# 启用 DP Attention
# --enable-dp-attention --tp 8 --dp 8

# 效果：
# - KV Cache 只存储一份（而非 TP 份）
# - Attention 独立计算
# - MoE 层前同步
```

### 实现原理

```
TP=8 时 KV Cache:
  传统: 每个 TP rank 存储完整 KV → 8x 冗余
  DP Attention: 只在 DP rank 0 存储 → 1x 存储

计算流程:
  [DP rank 0] Prefill/Decode → KV Cache 更新
  [DP rank 1-7] 独立计算 Attention（无 KV Cache）
  [同步] AllReduce 后进入 MoE
```

## 性能优化

### Chunked Prefix Cache

对于长序列 Prefill，使用分块处理：

```python
# 将长序列分割成块
chunk_size = 8192
for chunk_start in range(0, seq_len, chunk_size):
    chunk_end = min(chunk_start + chunk_size, seq_len)
    chunk_hidden = hidden_states[chunk_start:chunk_end]
    # 处理每个 chunk
    ...
```

### CUDA Graph 兼容

MLA 支持 CUDA Graph 以减少启动开销：

```python
# 条件判断
use_cuda_graph = (
    batch_size <= max_cuda_graph_batch_size and
    not dynamic_sequence_length
)
```

## YARN RoPE 扩展

DeepSeek V3 使用 YARN 扩展支持长上下文：

```python
def yarn_get_mscale(scale=1.0, mscale=1.0):
    """计算 YARN 缩放因子"""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

# 应用缩放
scaling_factor = 40  # 支持到 128K 上下文
mscale = yarn_get_mscale(scaling_factor)
self.scaling = self.qk_head_dim ** -0.5 * mscale * mscale
```
