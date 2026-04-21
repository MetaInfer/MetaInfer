# 内存池 - GPU 内存管理

## 核心问题

LLM 推理需要在启动时为 KV 缓存预分配一块大的连续 GPU 内存缓冲区，并在运行时管理其内部子分配。与 CPU 内存不同，GPU 内存无法在不引入性能损失的情况下频繁动态分配/释放。

## 内存布局策略

### 策略 1：单体块缓冲区（nano-vllm）
整份 KV 缓存使用一个大张量：

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

**空闲块跟踪**：使用简单的 deque 存放可用 block ID。
```python
free_block_ids = deque(range(num_blocks))  # O(1) push/pop
```

### 策略 2：两级内存池（nano-sglang）
将请求级管理与词元级管理拆分：

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

**分配**：
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

**连续分配**（某些 kernel 更偏好）：
```python
def alloc_contiguous(self, need_size):
    """Find a contiguous run of free slots"""
    # Scan for consecutive zeros in mem_state
    # Returns (start_index, indices) or raises if fragmented
```

### 策略 3：按页内存池（mini-sglang）
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

## 内存容量估算

### Profiling 方式（所有框架通用）
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

### GQA 注意事项
在 Grouped-Query Attention (GQA) 中，KV heads < Q heads：
```python
# Q heads: 32, KV heads: 8 (GQA ratio = 4)
bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
# 4x less KV memory than full MHA → 4x more tokens can be cached
```

## 前缀共享的引用计数

使用前缀缓存时，多个请求可能共享同一份 KV 数据：

```python
class TokenToKVPool:
    mem_state: Tensor  # Reference counts, not boolean

    def add_refs(self, indices):
        self.mem_state[indices] += 1

    def decrease_refs(self, indices):
        self.mem_state[indices] -= 1
        # Slot is only truly free when ref_count reaches 0
```

**生命周期**：
1. 请求 A 分配词元，`ref_count = 1`
2. 请求 A 结束，词元插入 radix 缓存，`ref_count` 仍为 1
3. 请求 B 命中同前缀，`ref_count` 增加到 2
4. 请求 A 的缓存项被驱逐，`ref_count` 降到 1
5. 请求 B 结束且其缓存项被驱逐，`ref_count` 降到 0 → 槽位可复用

## 设计模板

最小内存池需要：
1. **预分配**：启动时分配一个大张量
2. **空闲列表**：跟踪可用槽位/块（deque 或 stack）
3. **分配/释放**：O(1) 复杂度分配与回收
4. **容量估算**：通过模型 profiling 确定池大小

可选增强：
- 引用计数（用于前缀缓存）
- 连续分配支持（适配特定 kernel）
- 两级池（请求级 + 词元级，管理更灵活）
- 内存碎片整理（长时运行服务）
