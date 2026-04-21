# KV 缓存管理

## 核心问题

在自回归生成中，每个词元的注意力都需要访问此前所有词元的 Key 与 Value 向量。存储并管理这份 **KV 缓存** 是 LLM 推理的主要内存瓶颈。例如 32 层、32 个注意力头、头维 128、序列长度 1000 的模型：

```
2 (K+V) × 32 layers × 32 heads × 128 dim × 1000 tokens × 2 bytes (fp16) ≈ 500 MB
```

## 三种做法（由简到繁）

### 1. 连续分配（朴素做法）

每个请求获得一块大小为 `max_seq_len` 的连续内存。

```
Request A: [████████████░░░░░░░░]  (已用 12 个词元，浪费 8)
Request B: [██████░░░░░░░░░░░░░░]  (已用 6 个词元，浪费 14)
```

**问题**：内部碎片严重。4096 词元预算下平均约浪费 ~50% 内存。

### 2. 分页注意力（nano-vllm 风格）

借鉴操作系统虚拟内存：把 KV 缓存划成固定大小的 **块（页）**。每个请求维护一张 **块表（block table）**，把逻辑位置映射到物理块。

```
Physical blocks: [B0][B1][B2][B3][B4][B5][B6][B7]...

Request A block_table: [B0, B3, B5]  → tokens 0-767
Request B block_table: [B1, B4]      → tokens 0-511
Free blocks: [B2, B6, B7]
```

**关键数据结构**：

```python
class Block:
    block_id: int  # 块的引用计数
    ref_count: int      # 前缀共享用
    hash: Optional[int] # 前缀缓存查找
    def update(self, hash: int, token_ids: list[int]) # 更新块的哈希值和token序列
    def reset(self) #重置块的状态，块重新分配时初始化

class BlockManager: #显存管理
    blocks: List[Block]
    free_block_ids: deque[int] #空闲块ID的双端队列
    hash_to_block_id: Dict[int, int]  #  哈希值到块ID的映射字典，前缀缓存查找
    used_block_ids: set[int] # 已使用块ID的集合

    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算token序列的哈希值,支持前缀哈希，保证连续块的哈希唯一性
        token_ids: 要计算哈希的token ID列表
        prefix: 前缀哈希值（前一个块的哈希，用于链式哈希），-1表示无前缀
        return: token序列的xxh64哈希值（整数形式）
        """

    def allocate(seq) → List[int]:
        """为序列分配物理块；通过 hash 检查缓存命中。"""

        for i in range(seq.num_blocks):
            """遍历序列需要的每个块索引"""

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: 
            """块ID无效或缓存块的token与当前不匹配，则分配新块"""

            else:  
            """缓存命中：复用已有块，块引用计数+1"""

    def may_append(seq) → bool:
        """
        序列跨块边界时分配新块。分三种场景处理
        """
        if len(seq) % self.block_size == 1:  # 场景1：新增token后需要新开块
        elif len(seq) % self.block_size == 0:  #场景2：最后块刚被填满，更新最后块的哈希映射
        else:  # 场景3：最后一个块未填满（无需操作）

    def deallocate(seq):
        """释放指定序列占用的所有块
        按引用计数递减，无引用则释放。"""
```

**块大小的权衡**：

- **大块（如 256 词元）**：元数据少、分配次数少，但内部碎片更大,nano_vllm默认设置  
- **小块（如 16 词元）**：碎片小，但元数据多、访存更散

### 3. 词元级内存池 + Radix 缓存（nano-sglang / mini-sglang 风格）

从预先分配好的池中按 **词元槽位** 分配；配合 **Radix 树** 做前缀缓存。

**两级池**：

```python
class ReqToTokenPool:
    """将 (request_id, position) 映射到物理词元下标"""
    req_to_token: Tensor  # shape: [max_reqs, max_context_len]
    # req_to_token[table_idx, position] = physical_token_idx

class TokenToKVPool:
    """按词元槽位索引的物理 KV 存储"""
    kv_data: List[Tensor]  # 每层: [pool_size, 2, heads, dim]
    mem_state: Tensor      # 每个槽位的引用计数
```

**分配流程**：

```
1. 请求到达，携带 prompt 词元 [t0, t1, t2, ..., tn]
2. 查询 radix 缓存：match_prefix([t0..tn]) → 已缓存 0..k
3. 从 TokenToKVPool 分配 (n-k) 个新词元槽位
4. 将槽位下标写入 ReqToTokenPool[req_table_idx, k..n]
5. 前向时：用槽位下标将 K/V scatter 写入池中
```

## 基于 Radix 树的前缀缓存

### 概念

许多请求共享相同前缀（系统提示、few-shot 示例等）。Radix 树索引这些共享前缀，从而 **复用** 已有 KV。

### 数据结构（nano-sglang）

```python
class TreeNode:
    children: Dict[int, TreeNode]  # token_id → child
    parent: TreeNode
    value: List[int]              # 边上对应的 token ID
    ref_counter: int              # 活跃引用（防止被驱逐）
    last_access_time: float       # LRU 驱逐用

class RadixCache:
    root_node: TreeNode
    evictable_size: int           # ref_counter == 0 的可驱逐词元量
```

### 数据结构（mini-sglang）

```python
class RadixTreeNode:
    _key: Tensor      # 边上的 Token ID
    _value: Tensor    # 物理缓存下标
    _children: Dict
    ref_count: int
    timestamp: int    # LRU 排序
```

### 核心操作

**match_prefix(tokens)**：

```
从根沿树行走，消费与边匹配的 token。
返回: (matched_length, last_matching_node)
示例:
  树: root → "Hello world" → "How are" → "you?"
  查询: "Hello world, How is"
  匹配: "Hello world"（11 个 token 命中）
```

**insert(tokens, cache_indices)**：

```
请求结束后，将其完整 token 序列插入树。
若在节点处仅部分匹配，则分裂该节点:
  之前: root → "Hello world How"
  插入: "Hello world Where"
  之后: root → "Hello world " → "How"（原有）
                                → "Where"（新建）
```

**evict(num_tokens)**：

```
仍需释放更多词元时:
  弹出 last_access_time 最小（且 ref_count == 0）的叶节点
  释放其物理缓存下标
  若父节点变为叶节点，将父节点加入驱逐候选
```

### 引用计数

- **增加**：调度阶段，请求开始使用已缓存词元时  
- **减少**：该请求的 batch 结束或被抢占时  
- `ref_count > 0` 的节点为 **固定（pinned）**，不可驱逐

## 物理 KV 缓冲区布局

### 按层存储（nano-sglang）

```python
# 每层一个张量
kv_data[layer_idx].shape = [pool_size, 2, num_heads, head_dim]
# 访问: kv_data[layer][token_slot, 0] = K, kv_data[layer][token_slot, 1] = V
```

### 整块缓冲区（nano-vllm / mini-sglang）

```python
# 单一大张量
kv_cache.shape = [2, num_layers, num_blocks, block_size, num_heads, head_dim]
# 访问: kv_cache[0, layer, block_id, offset] = K
#       kv_cache[1, layer, block_id, offset] = V
```

### 写入 KV

通常用专用内核把新词元映射到物理位置：

```python
def store_kvcache(k, v, kv_cache, slot_mapping):
    """
    slot_mapping: [num_new_tokens] → 物理槽位下标
    将 k、v scatter 到 kv_cache 的指定槽位
    """
    for i, slot in enumerate(slot_mapping):
        block_id = slot // block_size
        offset = slot % block_size
        kv_cache[0, :, block_id, offset] = k[i]  # 所有层
        kv_cache[1, :, block_id, offset] = v[i]
```

实际部署中多为 Triton 或 CUDA 内核以保证性能。

## 内存容量估算

各框架确定 KV 缓存总量的思路类似：

```python
def estimate_kv_cache_blocks(config, gpu_memory_utilization=0.9):
    # 1. 统计模型占用显存
    model_memory = measure_model_memory()  # 加载模型，跑一遍 dummy forward

    # 2. 计算可用于 KV 的显存
    total_gpu_memory = torch.cuda.get_device_properties(0).total_mem
    available = total_gpu_memory * gpu_memory_utilization - model_memory

    # 3. 计算每块内存
    kv_per_token = 2 * num_layers * num_heads * head_dim * dtype_size
    kv_per_block = kv_per_token * block_size

    # 4. 确定块数
    num_blocks = available // kv_per_block
    return num_blocks
```

## 设计模板

最小 KV 缓存系统需要：

1. **选择粒度**：块（更简单，配合 PagedAttention）或词元（更灵活，配合 FlashInfer）
2. **分配池**：启动时预分配一张大张量
3. **跟踪空闲槽**：栈或 deque 保存可用下标
4. **提供 scatter 写内核**：逻辑位置 → 物理位置
5. **管理生命周期**：调度时分配，结束时释放

仅在以下情况再加前缀缓存：

- 请求共享长前缀（系统提示、模板等）  
- 部署场景下大量相似请求  
- 内存紧张，复用能带来明显收益

