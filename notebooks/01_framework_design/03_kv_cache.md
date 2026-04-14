# KV Cache Management

## Core Problem

During autoregressive generation, each token's attention requires access to all previous tokens' key and value vectors. Storing and managing this KV cache is the primary memory bottleneck of LLM inference. A model with 32 layers, 32 heads, and 128-dim head on 1000 tokens uses:
```
2 (K+V) × 32 layers × 32 heads × 128 dim × 1000 tokens × 2 bytes (fp16) ≈ 500 MB
```

## Three Approaches (Increasing Sophistication)

### 1. Contiguous Allocation (Naive)
Each request gets a contiguous chunk of memory sized to `max_seq_len`.

```
Request A: [████████████░░░░░░░░]  (12 tokens used, 8 wasted)
Request B: [██████░░░░░░░░░░░░░░]  (6 tokens used, 14 wasted)
```

**Problem**: Massive internal fragmentation. A 4096-token budget wastes ~50% memory on average.

### 2. Paged Attention (nano-vllm style)
Borrowed from OS virtual memory: divide KV cache into fixed-size **blocks** (pages). Each request maintains a **block table** mapping logical positions to physical blocks.

```
Physical blocks: [B0][B1][B2][B3][B4][B5][B6][B7]...

Request A block_table: [B0, B3, B5]  → tokens 0-767
Request B block_table: [B1, B4]      → tokens 0-511
Free blocks: [B2, B6, B7]
```

**Key data structures**:
```python
class Block:
    block_id: int
    ref_count: int      # For prefix sharing
    hash: Optional[int] # For prefix caching lookup

class BlockManager:
    blocks: List[Block]
    free_block_ids: Deque[int]
    hash_to_block_id: Dict[int, int]  # Prefix cache lookup

    def allocate(seq) → List[int]:
        """Assign physical blocks to sequence. Check hash for cache hits."""

    def may_append(seq) → bool:
        """Allocate new block when sequence crosses boundary."""

    def deallocate(seq):
        """Return blocks to free list (decrement ref_count first)."""
```

**Block size trade-off**:
- Large blocks (256 tokens): Less metadata overhead, fewer allocations, but more internal fragmentation
- Small blocks (16 tokens): Less fragmentation, but more metadata, more scattered memory access

### 3. Token-Level Pool with Radix Cache (nano-sglang / mini-sglang style)
Allocate individual token slots from a pre-allocated pool. Combined with a Radix Tree for prefix caching.

**Two-level pool**:
```python
class ReqToTokenPool:
    """Maps (request_id, position) → physical token index"""
    req_to_token: Tensor  # shape: [max_reqs, max_context_len]
    # req_to_token[table_idx, position] = physical_token_idx

class TokenToKVPool:
    """Physical KV storage indexed by token slot"""
    kv_data: List[Tensor]  # Per-layer, shape: [pool_size, 2, heads, dim]
    mem_state: Tensor      # Reference count per slot
```

**Allocation flow**:
```
1. Request arrives with prompt tokens [t0, t1, t2, ..., tn]
2. Query radix cache: match_prefix([t0..tn]) → cached 0..k
3. Allocate (n-k) new token slots from TokenToKVPool
4. Write slot indices into ReqToTokenPool[req_table_idx, k..n]
5. During forward: use slot indices to scatter K/V into pool
```

## Prefix Caching with Radix Tree

### Concept
Many requests share common prefixes (system prompts, few-shot examples). A Radix Tree indexes these shared prefixes so KV data can be reused.

### Data Structure (nano-sglang)
```python
class TreeNode:
    children: Dict[int, TreeNode]  # token_id → child
    parent: TreeNode
    value: List[int]              # token IDs in this edge
    ref_counter: int              # Active users (prevents eviction)
    last_access_time: float       # For LRU eviction

class RadixCache:
    root_node: TreeNode
    evictable_size: int           # Tokens with ref_counter == 0
```

### Data Structure (mini-sglang)
```python
class RadixTreeNode:
    _key: Tensor      # Token IDs for this edge
    _value: Tensor    # Physical cache indices
    _children: Dict
    ref_count: int
    timestamp: int    # For LRU ordering
```

### Core Operations

**match_prefix(tokens)**:
```
Walk tree from root, consuming tokens that match edges.
Return: (matched_length, last_matching_node)
Example:
  Tree: root → "Hello world" → "How are" → "you?"
  Query: "Hello world, How is"
  Match: "Hello world" (11 tokens matched)
```

**insert(tokens, cache_indices)**:
```
After request finishes, insert its full token sequence into tree.
If partial match at node, split the node:
  Before: root → "Hello world How"
  Insert: "Hello world Where"
  After:  root → "Hello world " → "How" (existing)
                                → "Where" (new)
```

**evict(num_tokens)**:
```
While need to free more tokens:
  Pop leaf node with lowest last_access_time (and ref_count == 0)
  Free its physical cache indices
  If parent becomes leaf, add parent to eviction candidates
```

### Reference Counting
- **Increment** when a request starts using cached tokens (during scheduling)
- **Decrement** when the request batch finishes or is preempted
- Nodes with `ref_count > 0` are **pinned** and cannot be evicted

## Physical KV Buffer Layout

### Per-Layer Storage (nano-sglang)
```python
# One tensor per layer
kv_data[layer_idx].shape = [pool_size, 2, num_heads, head_dim]
# Access: kv_data[layer][token_slot, 0] = K, kv_data[layer][token_slot, 1] = V
```

### Monolithic Buffer (nano-vllm / mini-sglang)
```python
# Single large tensor
kv_cache.shape = [2, num_layers, num_blocks, block_size, num_heads, head_dim]
# Access: kv_cache[0, layer, block_id, offset] = K
#         kv_cache[1, layer, block_id, offset] = V
```

### Writing KV Data
A specialized kernel maps new tokens to their physical locations:
```python
def store_kvcache(k, v, kv_cache, slot_mapping):
    """
    slot_mapping: [num_new_tokens] → physical slot indices
    Scatter k, v into kv_cache at the specified slots
    """
    for i, slot in enumerate(slot_mapping):
        block_id = slot // block_size
        offset = slot % block_size
        kv_cache[0, :, block_id, offset] = k[i]  # All layers
        kv_cache[1, :, block_id, offset] = v[i]
```

In practice this is a Triton or CUDA kernel for performance.

## Memory Capacity Estimation

All frameworks use a similar approach to determine how much KV cache to allocate:

```python
def estimate_kv_cache_blocks(config, gpu_memory_utilization=0.9):
    # 1. Profile model memory usage
    model_memory = measure_model_memory()  # Load model, run dummy forward

    # 2. Calculate available memory for KV cache
    total_gpu_memory = torch.cuda.get_device_properties(0).total_mem
    available = total_gpu_memory * gpu_memory_utilization - model_memory

    # 3. Calculate per-block memory
    kv_per_token = 2 * num_layers * num_heads * head_dim * dtype_size
    kv_per_block = kv_per_token * block_size

    # 4. Determine block count
    num_blocks = available // kv_per_block
    return num_blocks
```

## Design Template

For a minimal KV cache system:
1. **Choose granularity**: Blocks (simpler, works with PagedAttention) or tokens (more flexible, works with FlashInfer)
2. **Allocate pool**: One large pre-allocated tensor
3. **Track free slots**: A stack or deque of available indices
4. **Provide scatter-write kernel**: Map logical positions to physical locations
5. **Manage lifecycle**: Allocate on schedule, free on completion

Add prefix caching only when:
- Requests share common prefixes (system prompts, templates)
- The deployment serves many similar requests
- Memory is constrained enough that reuse matters
