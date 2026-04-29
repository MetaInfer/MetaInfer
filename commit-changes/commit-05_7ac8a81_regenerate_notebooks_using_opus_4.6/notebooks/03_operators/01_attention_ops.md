# Attention Operators

## Overview

Attention is the most performance-critical operator in LLM inference. Different phases (prefill vs decode) and different KV cache layouts require different kernel implementations. This document covers the kernel-level details across the three main approaches.

## Attention Kernel Taxonomy

```
Attention Kernels
├── Prefill / Extend
│   ├── Flash Attention varlen    (nano-vllm)
│   ├── Triton extend_attention   (nano-sglang)
│   └── Flash Attention / FlashInfer wrappers (mini-sglang)
├── Decode
│   ├── Flash Attention with KV cache (nano-vllm)
│   ├── Triton token_attention    (nano-sglang)
│   └── FlashInfer paged decode   (mini-sglang)
└── KV Cache Store
    ├── Triton store_kvcache      (nano-vllm)
    ├── Python-level slicing      (nano-sglang)
    └── Custom store_cache kernel (mini-sglang)
```

## Prefill Attention

### Flash Attention varlen (nano-vllm)
For processing variable-length sequences in a single batch:

```python
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,                  # [total_q_tokens, num_heads, head_dim]
    k,                  # [total_kv_tokens, num_kv_heads, head_dim]
    v,                  # [total_kv_tokens, num_kv_heads, head_dim]
    cu_seqlens_q,       # [batch_size + 1], cumulative Q sequence lengths
    cu_seqlens_k,       # [batch_size + 1], cumulative KV sequence lengths
    max_seqlen_q,       # int, maximum Q sequence length in batch
    max_seqlen_k,       # int, maximum KV sequence length in batch
    causal=True,        # Apply causal mask
)
# output: [total_q_tokens, num_heads, head_dim]
```

When prefix caching is active, the function also accepts `block_table` to read cached KV from paged memory.

### Triton Extend Attention (nano-sglang)
A custom 2-stage Triton kernel that handles prefix (cached) + extend (new) tokens:

```python
def extend_attention_fwd(
    Q_Extend,        # [total_extend_tokens, num_heads, head_dim]
    K_Extend,        # [total_extend_tokens, num_kv_heads, head_dim]
    V_Extend,        # [total_extend_tokens, num_kv_heads, head_dim]
    O_Extend,        # Output: [total_extend_tokens, num_heads, head_dim]
    K_Buffer,        # KV pool: [pool_size, num_kv_heads, head_dim]
    V_Buffer,        # KV pool: [pool_size, num_kv_heads, head_dim]
    Req_to_tokens,   # [max_batch, max_ctx_len] - maps to physical indices
    B_req_idx,       # [batch_size] - request indices in Req_to_tokens
    B_Seq_Len,       # [batch_size] - total sequence lengths (prefix + extend)
    B_Start_Loc_Extend,  # [batch_size] - start offsets in Q_Extend
    B_Seq_Len_Extend,    # [batch_size] - extend lengths
    sm_scale,            # 1/sqrt(head_dim)
    kv_group_num,        # num_q_heads // num_kv_heads
):
```

**Kernel logic**:
1. **Stage 1 (Prefix)**: For each query token, compute attention scores against all cached prefix tokens by reading from `K_Buffer/V_Buffer` via `Req_to_tokens` indirection
2. **Stage 2 (Extend)**: Compute attention scores against new (extend) tokens with causal masking
3. **Combine**: Online softmax to merge the two partial attention results

**Tiling**: `BLOCK_M=128, BLOCK_N=128, num_warps=8` (for head_dim > 64)

### Flash Attention Backend (mini-sglang)
Wraps the `sgl-kernel` Flash Attention implementation:

```python
class FlashAttentionBackend(BaseAttnBackend):
    def forward(self, q, k, v, layer):
        output = flash_attn_with_kvcache(
            q=q,
            k_cache=self.kv_cache.k_buffer(layer),
            v_cache=self.kv_cache.v_buffer(layer),
            page_table=self.page_table,
            cache_seqlens=self.cache_seqlens,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k_new,
            max_seqlen_q=self.max_seqlen_q,
            causal=True,
            softmax_scale=self.sm_scale,
        )
        return output
```

## Decode Attention

### Flash Attention with KV Cache (nano-vllm)
```python
from flash_attn import flash_attn_with_kvcache

output = flash_attn_with_kvcache(
    q,                   # [batch_size, 1, num_heads, head_dim] (one token per request)
    k_cache,             # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,             # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table=block_table,  # [batch_size, max_blocks] - physical block indices
    cache_seqlens=context_lens,  # [batch_size] - actual sequence lengths
    causal=True,
)
```

### Triton Token Attention (nano-sglang)
A 2-stage Triton kernel optimized for single-token queries:

```python
# Stage 1: Compute attention logits
# Grid: (batch, num_heads, num_blocks_per_seq)
def _fwd_kernel_stage1(Q, K_Buffer, Req_to_tokens, Att_Out, ...):
    """
    For each query token, compute Q·K^T against a block of KV cache tokens.
    Output: partial attention logits [batch, num_heads, num_blocks, BLOCK_N]
    """

# Stage 2: Softmax and reduce with V
# Grid: (batch, num_heads, 1)
def _fwd_kernel_stage2(Att_Out, V_Buffer, Req_to_tokens, Out, ...):
    """
    Apply softmax across all blocks, then compute weighted sum with V.
    Output: attention output [batch, num_heads, head_dim]
    """
```

**Why 2 stages?** Single-token queries have very long KV sequences. Splitting into blocks allows parallel computation across KV blocks, then a reduction stage combines them.

### FlashInfer Backend (mini-sglang)
```python
class FlashInferBackend(BaseAttnBackend):
    def __init__(self):
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer,  # 128MB pre-allocated buffer
            kv_layout="NHD",
            use_tensor_cores=(gqa_group_size >= 4),
        )

    def forward_decode(self, q, layer):
        return self.decode_wrapper.forward(
            q, paged_kv_cache=(k_buffer, v_buffer),
        )
```

## KV Cache Store Kernels

### Triton Store KV Cache (nano-vllm)
```python
@triton.jit
def store_kvcache_kernel(K, V, KVCache, SlotMapping, ...):
    """
    K: [num_new_tokens, num_kv_heads, head_dim]
    V: [num_new_tokens, num_kv_heads, head_dim]
    KVCache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    SlotMapping: [num_new_tokens] → physical slot index

    For each new token i:
        slot = SlotMapping[i]
        block_id = slot // block_size
        offset = slot % block_size
        KVCache[0, layer, block_id, offset, :, :] = K[i]
        KVCache[1, layer, block_id, offset, :, :] = V[i]
    """
```

### Direct Slicing (nano-sglang)
```python
def store_kv_cache(self, k, v, out_cache_loc, layer_id):
    """Simple Python-level indexing into the pre-allocated pool"""
    self.token_to_kv_pool.kv_data[layer_id][out_cache_loc] = torch.stack([k, v], dim=1)
```

## Attention Backend Abstraction (mini-sglang)

Mini-sglang provides the cleanest abstraction for swapping backends:

```python
class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(self, q, k, v, layer_id) -> Tensor:
        """Execute attention for current batch"""
        pass

    @abstractmethod
    def prepare_metadata(self, batch) -> None:
        """Prepare metadata (page tables, sequence lengths) before forward"""
        pass

    @abstractmethod
    def init_capture_graph(self, batch_size) -> None:
        """Prepare for CUDA graph capture"""
        pass

class HybridBackend(BaseAttnBackend):
    """Use different backends for prefill vs decode"""
    def __init__(self, prefill_backend, decode_backend):
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id):
        if self.current_phase == "prefill":
            return self.prefill_backend.forward(q, k, v, layer_id)
        else:
            return self.decode_backend.forward(q, k, v, layer_id)
```

## Performance Considerations

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Bottleneck | Compute-bound | Memory-bandwidth-bound |
| Q length | Many tokens | 1 token |
| KV length | Prompt length | Full sequence length |
| Best kernel | Flash Attention (varlen) | FlashInfer or 2-stage Triton |
| Block size | Large (128-256) | Small (32-64) |
| Warps | 8 | 2-4 |
| CUDA graph | Usually not (variable shape) | Yes (fixed batch size) |

## Design Template

For generating attention operators:
1. **Choose library**: Flash Attention (most portable) or FlashInfer (best decode performance)
2. **Implement two paths**: Prefill (varlen) and Decode (paged)
3. **Add KV store kernel**: Triton kernel or direct indexing
4. **Abstract behind interface**: `BaseAttnBackend` pattern allows future swaps
5. **Support CUDA graph**: Decode backend must support graph capture
