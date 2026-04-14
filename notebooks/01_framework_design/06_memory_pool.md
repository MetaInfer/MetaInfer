# Memory Pool - GPU Memory Management

## Core Problem

LLM inference must pre-allocate a large contiguous GPU memory buffer for KV cache at startup, then manage sub-allocations within it at runtime. Unlike CPU memory, GPU memory cannot be easily allocated/freed dynamically without performance penalties.

## Memory Layout Strategies

### Strategy 1: Monolithic Block Buffer (nano-vllm)
One large tensor for the entire KV cache:

```python
# Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
kv_cache = torch.zeros(
    2,              # K and V
    num_layers,     # e.g., 32
    num_blocks,     # e.g., 1000 (calculated based on available memory)
    block_size,     # e.g., 256
    num_kv_heads,   # e.g., 8 (GQA)
    head_dim,       # e.g., 128
    dtype=torch.float16,
    device='cuda'
)
```

**Free block tracking**: Simple deque of available block IDs.
```python
free_block_ids = deque(range(num_blocks))  # O(1) push/pop
```

### Strategy 2: Two-Level Pool (nano-sglang)
Separates request-level and token-level management:

```python
class ReqToTokenPool:
    """Level 1: Maps (request, position) → token_slot"""
    req_to_token: Tensor  # [max_reqs, max_context_len], dtype=int32
    mem_state: Tensor     # [max_reqs], boolean: is slot in use?

class TokenToKVPool:
    """Level 2: Physical KV storage"""
    kv_data: List[Tensor]  # Per layer: [pool_size, 2, num_heads, head_dim]
    mem_state: Tensor      # [pool_size], int: reference count per slot
```

**Allocation**:
```python
def alloc(self, need_size):
    # Find slots where mem_state == 0 (free)
    free_indices = (self.mem_state == 0).nonzero().flatten()
    if len(free_indices) < need_size:
        return None  # OOM
    selected = free_indices[:need_size]
    self.mem_state[selected] = 1
    return selected
```

**Contiguous allocation** (preferred by some kernels):
```python
def alloc_contiguous(self, need_size):
    """Find a contiguous run of free slots"""
    # Scan for consecutive zeros in mem_state
    # Returns (start_index, indices) or raises if fragmented
```

### Strategy 3: Page-Based Pool (mini-sglang)
```python
class MHAKVCache:
    """Physical KV buffer with page-granularity allocation"""
    _kv_buffer: Tensor  # [2, layers, num_pages, page_size, heads, head_dim]
    free_slots: List[int]  # Stack of free page starts

    def alloc_pages(self, num_pages):
        return [self.free_slots.pop() for _ in range(num_pages)]

    def free_pages(self, page_indices):
        self.free_slots.extend(page_indices)
```

## Memory Capacity Estimation

### Profiling Approach (all frameworks)
```python
def profile_and_allocate():
    # Step 1: Load model onto GPU
    model = load_model()

    # Step 2: Run a dummy forward pass to capture peak memory
    dummy_input = torch.zeros(max_batch_tokens, ...)
    with torch.no_grad():
        model(dummy_input)

    # Step 3: Measure remaining memory
    peak_memory = torch.cuda.max_memory_allocated()
    total_memory = torch.cuda.get_device_properties(0).total_mem
    available = total_memory * gpu_memory_utilization - peak_memory

    # Step 4: Calculate KV cache capacity
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    if using_blocks:
        bytes_per_block = bytes_per_token * block_size
        num_blocks = available // bytes_per_block
    else:
        num_tokens = available // bytes_per_token
```

### GQA Consideration
With Grouped-Query Attention (GQA), KV heads < Q heads:
```python
# Q heads: 32, KV heads: 8 (GQA ratio = 4)
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
# 4x less KV memory than full MHA → 4x more tokens can be cached
```

## Reference Counting for Prefix Sharing

When using prefix caching, multiple requests may share the same KV data:

```python
class TokenToKVPool:
    mem_state: Tensor  # Reference counts, not boolean

    def add_refs(self, indices):
        self.mem_state[indices] += 1

    def decrease_refs(self, indices):
        self.mem_state[indices] -= 1
        # Slot is only truly free when ref_count reaches 0
```

**Lifecycle**:
1. Request A allocates tokens, `ref_count = 1`
2. Request A finishes, tokens inserted into radix cache, `ref_count` stays 1
3. Request B matches prefix, `ref_count` incremented to 2
4. Request A's cache entry evicted, `ref_count` drops to 1
5. Request B finishes and entry evicted, `ref_count` drops to 0 → slot is free

## Design Template

A minimal memory pool needs:
1. **Pre-allocation**: Single large tensor allocated at startup
2. **Free list**: Track available slots/blocks (deque or stack)
3. **Alloc/free**: O(1) allocation and deallocation
4. **Capacity estimation**: Profile model memory to determine pool size

Optional enhancements:
- Reference counting (for prefix caching)
- Contiguous allocation support (for certain kernels)
- Two-level pool (request-level + token-level, for more flexible management)
- Memory defragmentation (for long-running servers)
