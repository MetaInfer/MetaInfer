# 注意力算子

## 1. 注意力算子概述

Attention是Transformer的核心算子，其计算复杂度直接影响推理性能。

```
标准Attention:
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

优化目标:
1. 减少内存访问 (Flash Attention)
2. 支持变长序列
3. 支持KV Cache
4. 支持Paged Attention
```

## 2. Flash Attention

### 2.1 核心思想

Flash Attention通过分块计算减少内存访问：

```
传统方式:
Q @ K^T → [batch, heads, seq_len, seq_len]  (大量内存)
softmax → [batch, heads, seq_len, seq_len]
@ V → [batch, heads, seq_len, head_dim]

Flash Attention:
分块计算，不完整存储attention matrix
```

### 2.2 变长序列Flash Attention

```python
import flash_attn

def flash_attn_varlen(
    q: torch.Tensor,           # [total_tokens, heads, head_dim]
    k: torch.Tensor,           # [total_kv_tokens, kv_heads, head_dim]
    v: torch.Tensor,           # [total_kv_tokens, kv_heads, head_dim]
    cu_seqlens_q: torch.Tensor, # [batch + 1], query累积长度
    cu_seqlens_k: torch.Tensor, # [batch + 1], key累积长度
    max_seqlen_q: int,         # query最大序列长度
    max_seqlen_k: int,         # key最大序列长度
    softmax_scale: float,      # 1/sqrt(head_dim)
    causal: bool = True,       # 是否causal mask
) -> torch.Tensor:
    """
    变长序列Flash Attention
    
    输入是拼接在一起的变长序列，通过cu_seqlens标识每个序列的边界
    """
    output = flash_attn.flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return output  # [total_tokens, heads, head_dim]
```

### 2.3 使用示例

```python
# Prefill阶段
def attention_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    """
    q: [total_q_tokens, num_heads, head_dim]
    k: [total_kv_tokens, num_kv_heads, head_dim]
    """
    # GQA支持：k、v的head数可能少于q
    output = flash_attn_varlen(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
    )
    return output
```

## 3. KV Cache Attention

### 3.1 Decode阶段Attention

```python
def flash_attn_with_kvcache(
    q: torch.Tensor,           # [batch, 1, heads, head_dim]
    k_cache: torch.Tensor,     # [num_blocks, block_size, kv_heads, head_dim]
    v_cache: torch.Tensor,     # [num_blocks, block_size, kv_heads, head_dim]
    block_tables: torch.Tensor, # [batch, max_blocks]
    context_lens: torch.Tensor, # [batch]
    softmax_scale: float,
) -> torch.Tensor:
    """
    使用KV Cache的Decode Attention
    
    k_cache, v_cache: Paged KV Cache存储
    block_tables: 每个序列使用的block id列表
    context_lens: 每个序列的当前长度
    """
    # 这是flash_attn库的接口
    output = flash_attn.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        softmax_scale=softmax_scale,
    )
    return output  # [batch, heads, head_dim]
```

### 3.2 Block-based存储

```python
# KV Cache存储格式
# Shape: [num_blocks, block_size, num_kv_heads, head_dim]
k_cache = torch.empty(num_blocks, block_size, num_kv_heads, head_dim)
v_cache = torch.empty(num_blocks, block_size, num_kv_heads, head_dim)

# Block Table
# 序列的token分散存储在不同的block中
block_tables = torch.tensor([
    [5, 12, 3, -1],   # 序列0使用block 5, 12, 3
    [8, 7, -1, -1],   # 序列1使用block 8, 7
    [2, 9, 15, 1],    # 序列2使用block 2, 9, 15, 1
], dtype=torch.int32)
```

## 4. KV Cache存储内核

### 4.1 Triton实现

```python
import triton
import triton.language as tl

@triton.jit
def store_kvcache_kernel(
    # 输入
    key_ptr,      # K张量指针
    value_ptr,    # V张量指针
    # KV Cache
    k_cache_ptr,  # K Cache指针
    v_cache_ptr,  # V Cache指针
    # 映射
    slot_mapping_ptr,  # 槽位映射
    # 维度
    HEAD_DIM: tl.constexpr,
):
    """
    将计算出的K、V存储到KV Cache
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    
    if slot == -1:
        # 前缀缓存命中，跳过存储
        return
    
    # 存储K
    offsets = tl.arange(0, HEAD_DIM)
    k = tl.load(key_ptr + idx * HEAD_DIM + offsets)
    tl.store(k_cache_ptr + slot * HEAD_DIM + offsets, k)
    
    # 存储V
    v = tl.load(value_ptr + idx * HEAD_DIM + offsets)
    tl.store(v_cache_ptr + slot * HEAD_DIM + offsets, v)

def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """
    调用Triton内核存储KV Cache
    """
    num_tokens = key.shape[0]
    head_dim = key.shape[-1]
    
    grid = (num_tokens,)
    store_kvcache_kernel[grid](
        key, value,
        k_cache, v_cache,
        slot_mapping,
        HEAD_DIM=head_dim,
    )
```

## 5. 前缀缓存Attention

### 5.1 两阶段Attention

当有前缀缓存时，Attention分两阶段：

```python
def attention_with_prefix_cache(q, k_new, v_new, k_cache, v_cache, prefix_len):
    """
    带前缀缓存的Attention
    
    q: 当前query
    k_new, v_new: 新计算的K、V
    k_cache, v_cache: 缓存的K、V
    prefix_len: 缓存的前缀长度
    """
    # 阶段1: 与前缀缓存的attention（非causal）
    prefix_output = flash_attn_func(
        q[:, :prefix_len],
        k_cache,
        v_cache,
        causal=False,  # 非causal
    )
    
    # 阶段2: 与新token的attention（causal）
    new_output = flash_attn_func(
        q[:, prefix_len:],
        k_new,
        v_new,
        causal=True,
    )
    
    # 合并结果
    output = torch.cat([prefix_output, new_output], dim=1)
    return output
```

## 6. GQA支持

### 6.1 Grouped Query Attention

```python
def expand_kv_heads(k, v, num_heads, num_kv_heads):
    """
    GQA: 将KV head扩展到与Q head数相同
    
    k: [tokens, num_kv_heads, head_dim]
    v: [tokens, num_kv_heads, head_dim]
    
    返回: [tokens, num_heads, head_dim]
    """
    if num_heads == num_kv_heads:
        return k, v
    
    # 计算每个KV head对应的Q head数
    n_rep = num_heads // num_kv_heads
    
    # 扩展
    k = k[:, None, :, :].expand(tokens, n_rep, num_kv_heads, head_dim)
    k = k.reshape(tokens, num_heads, head_dim)
    
    v = v[:, None, :, :].expand(tokens, n_rep, num_kv_heads, head_dim)
    v = v.reshape(tokens, num_heads, head_dim)
    
    return k, v
```

### 6.2 Flash Attention GQA

Flash Attention原生支持GQA：

```python
# flash_attn_varlen_func 自动处理GQA
# 当 num_kv_heads < num_heads 时自动扩展
output = flash_attn_varlen_func(
    q,  # [tokens, num_heads, head_dim]
    k,  # [tokens, num_kv_heads, head_dim]
    v,  # [tokens, num_kv_heads, head_dim]
    ...
)
```

## 7. Attention后端选择

### 7.1 可用后端

| 后端 | 特点 | 适用场景 |
|------|------|----------|
| FlashAttention 2 | NVIDIA优化 | NVIDIA GPU |
| FlashAttention 3 | Hopper优化 | H100/H200 |
| FlashInfer | 高度优化 | 生产部署 |
| Triton | 灵活可定制 | 研究/开发 |
| PyTorch Native | 兼容性好 | CPU/其他 |

### 7.2 精简框架推荐

```python
# 推荐使用FlashAttention或FlashInfer
import flash_attn

def attention(q, k, v, is_prefill, **kwargs):
    if is_prefill:
        return flash_attn.flash_attn_varlen_func(q, k, v, **kwargs)
    else:
        return flash_attn.flash_attn_with_kvcache(q, k, v, **kwargs)
```

## 8. 性能优化要点

### 8.1 内存布局

```python
# 推荐: 最后一维连续
# [tokens, heads, head_dim] ✓
# [heads, tokens, head_dim] ✗ (需要transpose)
```

### 8.2 精度选择

```python
# FP16/BF16: 标准精度
# FP8: 需要特殊支持 (Hopper+)
dtype = torch.float16  # 推荐
```

### 8.3 Kernel选择

```python
# 小batch: flash_attn_with_kvcache
# 大batch/prefill: flash_attn_varlen_func
```
