# DeepSeek V3 - Native Sparse Attention (NSA)

## Core Concept

NSA addresses the quadratic complexity of full attention for very long sequences by selecting only the most "important" tokens to attend to, while maintaining a local sliding window for recent context.

## Architecture

```
Query token at position t
    ↓
Three attention components:
    ├── [Sliding Window]  → Attend to recent W tokens (always)
    ├── [Sparse Selection] → Attend to top-K important token blocks
    └── [Compressed Global] → Attend to a compressed summary of all tokens
    ↓
Weighted combination of three attention outputs
```

## Sparse Selection Mechanism

### Indexer Module
A lightweight module that determines which tokens are "important":

```python
class NSAIndexer:
    def __init__(self):
        self.proj = Linear(hidden_size, index_dim)  # Compress to lower dim

    def compute_importance(self, hidden_states):
        # Project to lower dimension
        index_features = self.proj(hidden_states)

        # Compute importance scores between current query and all keys
        scores = query_features @ index_features.T

        # Select top-K blocks
        block_scores = scores.reshape(-1, block_size).mean(dim=-1)
        top_k_blocks = block_scores.topk(K).indices

        return top_k_blocks
```

### Attention Computation
```python
def nsa_attention(q, k, v, sliding_window, selected_blocks, compressed_kv):
    # 1. Sliding window attention (local)
    local_out = sliding_window_attention(q, k_local, v_local, window_size)

    # 2. Sparse attention (selected important blocks)
    sparse_k, sparse_v = gather_blocks(k, v, selected_blocks)
    sparse_out = attention(q, sparse_k, sparse_v)

    # 3. Compressed global attention
    compressed_out = attention(q, compressed_k, compressed_v)

    # 4. Weighted combination
    output = gate_local * local_out + gate_sparse * sparse_out + gate_global * compressed_out
    return output
```

## Impact on Inference Framework

1. **KV Cache Indexing**: Need specialized `NSATokenToKVPool` that supports both dense (sliding window) and sparse (block selection) access patterns
2. **Top-K Computation**: Adds an indexing step before attention
3. **Multiple Attention Types**: Must support three different attention patterns in a single layer
4. **Block Granularity**: Sparse selection operates on blocks of tokens, not individual tokens
5. **Memory**: Reduced attention memory but adds indexer parameters

## When to Use

- **Beneficial**: Very long context (>32K tokens) where full attention is prohibitively expensive
- **Less beneficial**: Short sequences where full attention is already fast
- **Trade-off**: Adds model complexity; may reduce quality on tasks that require precise long-range attention
