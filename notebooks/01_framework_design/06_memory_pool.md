# 内存池管理

## 1. 内存管理概述

LLM推理框架需要高效管理GPU内存，主要包括：

1. **模型权重**：固定占用，推理过程中不变
2. **KV Cache**：动态增长，随序列长度增加
3. **中间激活**：前向传播过程中的临时内存
4. **CUDA Graph Buffer**：如果使用CUDA Graph优化

## 2. GPU内存计算

### 2.1 模型权重内存

```python
def compute_model_memory(config):
    """
    计算模型权重大小
    
    示例：Llama-7B, FP16
    = 7B parameters × 2 bytes = 14 GB
    """
    num_params = sum(p.numel() for p in model.parameters())
    memory = num_params * dtype_size
    return memory
```

### 2.2 KV Cache内存

```python
def compute_kvcache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    dtype_size: int = 2,  # FP16
):
    """
    计算单个序列的KV Cache大小
    
    KV Cache = 2 × num_layers × max_seq_len × num_kv_heads × head_dim × dtype_size
    """
    return 2 * num_layers * max_seq_len * num_kv_heads * head_dim * dtype_size
```

### 2.3 Block内存

```python
def compute_block_memory(
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_size: int = 2,
):
    """
    计算单个Block的内存大小
    """
    return 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
```

## 3. 动态内存分配

### 3.1 内存分配流程

```python
class ModelRunner:
    def allocate_kv_cache(self):
        # 1. 获取GPU总内存
        free, total = torch.cuda.mem_get_info()
        
        # 2. 计算已使用内存（模型权重 + 激活）
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 3. 计算可用于KV Cache的内存
        available = total * config.gpu_memory_utilization - used - peak + current
        
        # 4. 计算Block数量
        block_bytes = compute_block_memory(...)
        num_blocks = int(available // block_bytes)
        
        # 5. 分配KV Cache
        self.kv_cache = torch.empty(
            2,  # K, V
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda"
        )
```

### 3.2 内存利用率配置

```python
# 配置参数
gpu_memory_utilization: float = 0.9  # 使用90%的GPU内存

# 内存计算
available_for_kvcache = total_gpu_memory * gpu_memory_utilization - model_memory - activation_memory
```

## 4. Block Manager

### 4.1 Block状态管理

```python
class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0      # 引用计数
        self.hash = -1          # 内容哈希
        self.token_ids = []     # 存储的token

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
```

### 4.2 Block分配与释放

```python
def allocate_block(self) -> Block:
    """分配一个空闲Block"""
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    block.reset()
    self.used_block_ids.add(block_id)
    return block

def deallocate_block(self, block_id: int):
    """释放一个Block"""
    block = self.blocks[block_id]
    assert block.ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

### 4.3 引用计数管理

```python
def inc_ref(self, block_id: int):
    """增加引用计数"""
    self.blocks[block_id].ref_count += 1

def dec_ref(self, block_id: int) -> bool:
    """
    减少引用计数
    返回：是否可以释放
    """
    block = self.blocks[block_id]
    block.ref_count -= 1
    return block.ref_count == 0
```

## 5. 两级内存池（nano-sglang方式）

### 5.1 ReqToTokenPool

管理请求到Token的映射：

```python
class ReqToTokenPool:
    """
    维护每个请求的Token到KV Cache索引的映射
    req_to_token[req_id, token_pos] = kv_cache_index
    """
    def __init__(self, size: int, max_context_len: int):
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self.req_to_token = torch.empty(
            (size, max_context_len),
            dtype=torch.int32,
            device="cuda"
        )
    
    def alloc(self, need_size: int) -> torch.Tensor:
        """分配请求槽位"""
        available = torch.nonzero(self.mem_state).squeeze(1)
        selected = available[:need_size]
        self.mem_state[selected] = False
        return selected
    
    def free(self, free_indices: torch.Tensor):
        """释放请求槽位"""
        self.mem_state[free_indices] = True
```

### 5.2 TokenToKVPool

管理实际的KV Cache存储：

```python
class TokenToKVPool:
    """
    管理Token级别的KV Cache存储
    每个Token对应一个唯一的索引
    """
    def __init__(self, size: int, dtype, head_num: int, head_dim: int, layer_num: int):
        self.mem_state = torch.zeros((size,), dtype=torch.int16, device="cuda")
        self.kv_data = [
            torch.empty((size, 2, head_num, head_dim), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]
    
    def alloc(self, need_size: int) -> torch.Tensor:
        """分配分散的Token槽位"""
        available = torch.nonzero(self.mem_state == 0).squeeze(1)
        selected = available[:need_size]
        self.mem_state[selected] = 1
        return selected
    
    def alloc_contiguous(self, need_size: int) -> torch.Tensor:
        """
        分配连续的Token槽位
        用于Decode阶段优化
        """
        # 查找连续空闲区域
        available = (self.mem_state == 0).int()
        # 卷积查找连续区域
        ...
```

### 5.3 两级池的优势

| 特性 | 单级Block池 | 两级内存池 |
|------|-------------|------------|
| 内存粒度 | Block级别 | Token级别 |
| 灵活性 | 一般 | 高 |
| 实现复杂度 | 简单 | 复杂 |
| 内存利用率 | 中 | 高 |

## 6. 内存优化策略

### 6.1 内存预热

```python
def warmup_memory(self):
    """
    预热内存，确定峰值使用
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 执行一次最大batch的推理
    self.warmup_model()
    
    # 记录峰值内存
    self.peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
```

### 6.2 内存碎片整理

```python
def defragment_memory(self):
    """
    整理内存碎片
    将活跃的KV Cache移动到连续内存区域
    """
    # 收集活跃Block
    active_blocks = self.collect_active_blocks()
    
    # 分配新的连续内存
    new_cache = torch.empty_like(self.kv_cache)
    
    # 复制数据
    new_idx = 0
    for block_id in active_blocks:
        new_cache[:, new_idx] = self.kv_cache[:, block_id]
        self.update_block_table(block_id, new_idx)
        new_idx += 1
    
    # 替换缓存
    self.kv_cache = new_cache
```

### 6.3 懒分配

```python
def lazy_allocate(self, seq: Sequence):
    """
    懒分配：只在需要时分配Block
    """
    needed_blocks = (len(seq) + self.block_size - 1) // self.block_size
    current_blocks = len(seq.block_table)
    
    if needed_blocks > current_blocks:
        new_blocks = needed_blocks - current_blocks
        for _ in range(new_blocks):
            if self.free_block_ids:
                seq.block_table.append(self.allocate_block())
            else:
                raise OutOfMemoryError
```

## 7. 内存监控

### 7.1 内存统计

```python
def get_memory_stats(self):
    """获取内存使用统计"""
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    
    return {
        "total": total,
        "free": free,
        "used": total - free,
        "peak": stats["allocated_bytes.all.peak"],
        "current": stats["allocated_bytes.all.current"],
        "num_blocks": len(self.blocks),
        "num_free_blocks": len(self.free_block_ids),
        "num_used_blocks": len(self.used_block_ids),
    }
```

### 7.2 内存预警

```python
def check_memory_pressure(self):
    """检查内存压力"""
    free_ratio = len(self.free_block_ids) / len(self.blocks)
    
    if free_ratio < 0.1:
        return "critical"
    elif free_ratio < 0.3:
        return "warning"
    else:
        return "ok"
```

## 8. 设计选择建议

### 8.1 精简框架推荐

| 功能 | 推荐 | 原因 |
|------|------|------|
| 内存池类型 | 单级Block池 | 实现简单 |
| 引用计数 | 支持 | 必要功能 |
| 前缀缓存 | Hash Block | 平衡复杂度和效率 |
| 碎片整理 | 不实现 | 增加复杂度 |

### 8.2 Block Size选择

```python
# Block Size权衡
small_block_size = 16   # 更灵活，但管理开销大
medium_block_size = 256 # 平衡选择（vLLM默认）
large_block_size = 1024 # 管理简单，但可能浪费内存

# 推荐使用256
```
