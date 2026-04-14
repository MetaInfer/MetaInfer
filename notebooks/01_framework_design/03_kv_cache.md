# KV Cache管理

## 1. KV Cache核心概念

### 1.1 什么是KV Cache

在Transformer的自回归生成过程中，每生成一个新token都需要重新计算所有历史token的Key和Value。KV Cache通过缓存这些中间结果来避免重复计算。

```
生成过程:
Step 1: [T1] → 生成 T2
Step 2: [T1, T2] → 生成 T3  (需要重新计算T1的K,V)
Step 3: [T1, T2, T3] → 生成 T4  (需要重新计算T1,T2的K,V)

使用KV Cache:
Step 1: [T1] → K1,V1 → 生成 T2
Step 2: [T2] + [K1,V1] → K2,V2 → 生成 T3  (复用K1,V1)
Step 3: [T3] + [K1,V1,K2,V2] → K3,V3 → 生成 T4  (复用K1-K2,V1-V2)
```

### 1.2 内存计算

单个序列的KV Cache大小：

```
KV Cache Size = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size

示例（Llama-7B, seq_len=4096, FP16）:
= 2 × 32 × 4096 × 32 × 128 × 2 bytes
= 2.1 GB
```

## 2. Paged Attention设计

### 2.1 Block-Based内存管理

将KV Cache切分为固定大小的Block（页）：

```python
# Block配置
block_size = 256  # 每个block存储256个token的K,V

# Block结构
class Block:
    block_id: int        # Block唯一标识
    ref_count: int       # 引用计数（支持共享）
    hash: int            # 内容哈希（用于前缀缓存）
    token_ids: list[int] # Block内存储的token
```

### 2.2 Block Table

每个序列维护一个Block Table，记录其KV Cache块的映射：

```python
class Sequence:
    block_table: list[int]  # [block_id_0, block_id_1, ...]
    
    # Block Table示例
    # token序列: [0-255] [256-511] [512-767] ...
    # block_table: [5, 12, 3, ...]
```

### 2.3 内存分配策略

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        
    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块"""
        return len(self.free_block_ids) >= seq.num_blocks
        
    def allocate(self, seq: Sequence):
        """为序列分配Block"""
        for i in range(seq.num_blocks):
            block_id = self.free_block_ids.popleft()
            block = self.blocks[block_id]
            block.reset()
            self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
            
    def deallocate(self, seq: Sequence):
        """释放序列的Block"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)
        seq.block_table.clear()
```

## 3. 前缀缓存（Prefix Caching）

### 3.1 概念

多个请求可能共享相同的前缀（如相同的系统提示词），前缀缓存允许复用已计算的KV Cache。

### 3.2 基于哈希的前缀缓存

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.hash_to_block_id: dict[int, int] = {}  # 哈希到Block映射
        
    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        """链式哈希：当前块哈希依赖于前缀哈希"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
        
    def allocate(self, seq: Sequence):
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 计算链式哈希
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # 查找缓存
            block_id = self.hash_to_block_id.get(h, -1)
            
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                # 缓存命中！
                cache_miss = False
                seq.num_cached_tokens += self.block_size
                
                # 增加引用计数
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            else:
                # 缓存未命中，分配新Block
                cache_miss = True
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
                
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
                
            seq.block_table.append(block_id)
```

### 3.3 Radix Tree缓存（nano-sglang方式）

更灵活的前缀缓存，支持任意长度前缀共享：

```python
class TreeNode:
    def __init__(self):
        self.children: dict = {}      # 子节点
        self.parent: TreeNode = None  # 父节点
        self.value: list = None       # 存储的token
        self.ref_counter: int = 0     # 引用计数
        self.last_access_time: float  # LRU驱逐

class RadixCache:
    def match_prefix(self, key):
        """前缀匹配，返回匹配的KV Cache索引"""
        value = []
        last_node = self.root_node
        self._match_prefix_helper(self.root_node, key, value, last_node)
        return torch.concat(value), last_node
        
    def insert(self, key, value=None):
        """插入新的token序列到Radix Tree"""
        return self._insert_helper(self.root_node, key, value)
        
    def evict(self, num_tokens, evict_callback):
        """基于LRU驱逐缓存"""
        leaves = self._collect_leaves()
        heapq.heapify(leaves)  # 按访问时间排序
        
        while num_evicted < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_counter == 0:  # 未被使用
                num_evicted += evict_callback(node.value)
                self._delete_leaf(node)
```

**Radix Tree vs Hash Block**:

| 特性 | Hash Block | Radix Tree |
|------|------------|------------|
| 共享粒度 | 固定block大小 | 任意长度 |
| 内存效率 | 一般 | 更高 |
| 实现复杂度 | 简单 | 复杂 |
| 查找速度 | O(1) 哈希 | O(n) 遍历 |

## 4. KV Cache存储格式

### 4.1 内存布局

```python
# 方式1：连续张量 [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
kv_cache = torch.empty(
    2,  # [K, V]
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads,
    head_dim
)

# 方式2：分离存储每层
kv_cache = [
    torch.empty(num_blocks, block_size, num_kv_heads, head_dim)
    for _ in range(2 * num_layers)  # K和V分开
]
```

### 4.2 与Attention层绑定

```python
class ModelRunner:
    def allocate_kv_cache(self):
        # 计算可用内存
        free, total = torch.cuda.mem_get_info()
        block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
        num_blocks = (available_memory) // block_bytes
        
        # 分配KV Cache
        self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        
        # 绑定到各Attention层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```

## 5. 两级内存池设计

nano-sglang采用的两级内存池架构：

### 5.1 ReqToTokenPool

管理请求到Token索引的映射：

```python
class ReqToTokenPool:
    def __init__(self, size, max_context_len):
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self.req_to_token = torch.empty(
            (size, max_context_len), 
            dtype=torch.int32, 
            device="cuda"
        )
        
    def alloc(self, need_size):
        """分配请求槽位"""
        selected = torch.nonzero(self.mem_state).squeeze(1)[:need_size]
        self.mem_state[selected] = False
        return selected
        
    def free(self, free_index):
        """释放请求槽位"""
        self.mem_state[free_index] = True
```

### 5.2 TokenToKVPool

管理实际的KV Cache存储：

```python
class TokenToKVPool:
    def __init__(self, size, dtype, head_num, head_dim, layer_num):
        self.mem_state = torch.zeros((size,), dtype=torch.int16, device="cuda")
        self.kv_data = [
            torch.empty((size, 2, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]
        
    def alloc(self, need_size):
        """分配分散的Token槽位"""
        available = torch.nonzero(self.mem_state == 0).squeeze(1)[:need_size]
        self.mem_state[available] = 1
        return available
        
    def alloc_contiguous(self, need_size):
        """分配连续的Token槽位（Decode优化）"""
        # 查找连续空闲区域
        ...
```

## 6. KV Cache操作

### 6.1 存储KV Cache（Triton Kernel）

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, value_ptr,      # 输入K, V
    k_cache_ptr, v_cache_ptr, # KV Cache存储
    slot_mapping_ptr,         # 槽位映射
    D: tl.constexpr           # head_dim
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    
    if slot == -1:
        return  # 前缀缓存命中，跳过存储
        
    # 存储K
    k = tl.load(key_ptr + idx * D + tl.arange(0, D))
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), k)
    
    # 存储V
    v = tl.load(value_ptr + idx * D + tl.arange(0, D))
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), v)
```

### 6.2 读取KV Cache

在Attention计算中读取缓存的K、V：

```python
def attention_with_kvcache(q, k_cache, v_cache, block_tables, context_lens):
    """
    Decode阶段的Attention计算
    """
    # 根据block_tables和context_lens组装K、V
    # 使用Flash Attention的kvcache变体
    output = flash_attn_with_kvcache(
        q.unsqueeze(1),  # [batch, 1, heads, head_dim]
        k_cache,
        v_cache,
        block_tables,
        context_lens,
    )
    return output
```

## 7. 内存优化策略

### 7.1 动态内存分配

```python
def allocate_kv_cache(self):
    # Warmup确定模型内存占用
    torch.cuda.reset_peak_memory_stats()
    self.warmup_model()
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    
    # 计算可用KV Cache空间
    free, total = torch.cuda.mem_get_info()
    available = total * gpu_memory_utilization - peak
    
    # 分配KV Cache
    block_bytes = compute_block_size()
    num_blocks = available // block_bytes
    self.kv_cache = torch.empty(...)
```

### 7.2 内存碎片管理

```python
# 定期整理内存碎片
def defragment(self):
    # 收集所有活跃的Block
    active_blocks = []
    for seq in self.running:
        active_blocks.extend(seq.block_table)
    
    # 重新分配连续内存
    new_kv_cache = torch.empty_like(self.kv_cache)
    for i, block_id in enumerate(active_blocks):
        new_kv_cache[:, i] = self.kv_cache[:, block_id]
        # 更新Block Table
        ...
```

## 8. 设计选择建议

### 8.1 精简框架推荐

| 功能 | 推荐实现 | 原因 |
|------|----------|------|
| 内存管理 | Block-based | 简单高效 |
| 前缀缓存 | Hash Block | 实现简单 |
| 引用计数 | 支持 | 必要功能 |
| 驱逐策略 | 简单LRU | 足够使用 |

### 8.2 生产级框架考虑

| 功能 | 实现 | 适用场景 |
|------|------|----------|
| Radix Tree | nano-sglang | 高缓存复用场景 |
| 两级内存池 | nano-sglang | 灵活内存管理 |
| 换出/换入 | vLLM | 低内存环境 |
| 跨节点传输 | vLLM/SGLang | 分布式推理 |
