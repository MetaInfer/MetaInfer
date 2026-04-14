# 调度器设计

## 1. 调度器核心职责

调度器是推理框架的核心组件，负责：

1. **请求排队**：管理等待中的请求
2. **资源分配**：为请求分配KV Cache内存
3. **批处理决策**：决定哪些请求参与当前推理步骤
4. **抢占处理**：当内存不足时选择请求抢占

## 2. 基础调度器实现

### 2.1 核心数据结构

```python
class Scheduler:
    def __init__(self, config: Config):
        # 调度约束
        self.max_num_seqs = config.max_num_seqs              # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 最大批处理token数
        
        # 请求队列
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列
        
        # KV Cache管理
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, 
            config.kvcache_block_size
        )
```

### 2.2 调度核心逻辑

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    """
    核心调度逻辑
    
    返回:
        - scheduled_seqs: 本轮要处理的序列列表
        - is_prefill: 是否为prefill阶段
    """
    scheduled_seqs = []
    num_batched_tokens = 0

    # Phase 1: Prefill（处理新请求）
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]
        
        # 计算需要处理的token数
        num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
        remaining = self.max_num_batched_tokens - num_batched_tokens
        
        # 检查预算和资源约束
        if remaining == 0:
            break  # token预算耗尽
        if not seq.block_table and not self.block_manager.can_allocate(seq):
            break  # KV Cache不足
            
        # Chunked Prefill: 只允许第一个序列分块
        if remaining < num_tokens and scheduled_seqs:
            break
            
        # 分配KV Cache
        if not seq.block_table:
            self.block_manager.allocate(seq)
            
        # 设置本轮调度的token数
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        
        # 更新序列状态
        if seq.num_scheduled_tokens == num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            
        scheduled_seqs.append(seq)
        num_batched_tokens += seq.num_scheduled_tokens
        
    if scheduled_seqs:
        return scheduled_seqs, True  # is_prefill=True

    # Phase 2: Decode（生成阶段）
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.running.popleft()
        
        # 检查KV Cache空间
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())  # 抢占其他序列
            else:
                self.preempt(seq)
                break
        else:
            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False  # is_prefill=False
```

## 3. 调度策略

### 3.1 FCFS（先到先服务）

最简单的调度策略，按请求到达顺序处理：

```python
def schedule_fcfs(self):
    # 按队列顺序处理，不做额外排序
    return self.waiting
```

**特点**：
- 实现简单
- 公平性好
- 不考虑缓存复用

### 3.2 LPM（最长前缀匹配优先）

优先调度与前缀缓存匹配最长的请求：

```python
def schedule_lpm(self, forward_queue):
    # 按前缀匹配长度降序排序
    forward_queue.sort(key=lambda x: -len(x.prefix_indices))
    return forward_queue
```

**优点**：
- 最大化缓存复用
- 减少重复计算

**缺点**：
- 可能导致短请求饥饿
- 需要前缀匹配信息

### 3.3 Weight-Based（权重调度）

基于Radix Tree权重决定调度优先级：

```python
def get_priority_queue(self, forward_queue):
    # 计算每个节点的权重
    node_to_weight = defaultdict(int)
    self._calc_weight_recursive(self.tree_cache.root_node, node_to_weight)
    
    # 按权重遍历生成优先级队列
    tmp_queue = []
    self._get_weight_priority_recursive(
        self.tree_cache.root_node, 
        node_to_weight, 
        tmp_queue
    )
    return tmp_queue

def _calc_weight_recursive(self, node, node_to_weight):
    # 权重 = 1 + 子节点权重之和 + 该节点上的请求数
    node_to_weight[node] = 1
    if node in last_node_to_reqs:
        node_to_weight[node] += len(last_node_to_reqs[node])
    for child in node.children.values():
        self._calc_weight_recursive(child, node_to_weight)
        node_to_weight[node] += node_to_weight[child]
```

**优点**：
- 智能最大化缓存复用
- 考虑整体缓存树结构

### 3.4 策略选择建议

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 无前缀缓存 | FCFS | 简单高效 |
| 有前缀缓存 | LPM/Weight | 最大化缓存复用 |
| 高吞吐优先 | Weight | 智能调度 |
| 低延迟优先 | FCFS | 公平响应 |

## 4. 抢占机制

### 4.1 抢占触发条件

当以下情况发生时需要抢占：

1. **KV Cache不足**：新请求需要更多内存
2. **优先级调度**：高优先级请求需要资源

### 4.2 抢占实现

```python
def preempt(self, seq: Sequence):
    """
    抢占序列：释放资源，放回等待队列
    """
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)  # 释放KV Cache
    self.waiting.appendleft(seq)        # 放回等待队列头部
```

### 4.3 抢占策略

**Preemption Recomputation（重计算）**：
- 完全释放KV Cache
- 请求重新排队
- 下次调度时重新计算prefill

**Swap to CPU（换出到CPU）**：
- 将KV Cache换出到CPU内存
- 需要时换回GPU
- vLLM默认实现（复杂框架）

**精简框架建议**：使用重计算策略，实现简单。

## 5. Chunked Prefill

### 5.1 概念

Chunked Prefill允许将长prompt分成多个chunk处理：

```
长Prompt: [token_0, token_1, ..., token_10000]
                │
                ▼ 分成多个chunk
Chunk 1: [token_0, ..., token_1023]
Chunk 2: [token_1024, ..., token_2047]
...
```

### 5.2 实现要点

```python
# 在schedule()中
if remaining < num_tokens:
    if scheduled_seqs:
        break  # 只有第一个序列可以分块
    # 允许第一个序列部分处理
    seq.num_scheduled_tokens = remaining
```

### 5.3 优缺点

**优点**：
- 避免长prompt阻塞短请求
- 提高系统响应性
- 更好的GPU利用率

**缺点**：
- 实现复杂度增加
- 可能增加总延迟

## 6. 后处理逻辑

### 6.1 Prefill后处理

```python
def postprocess_prefill(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        # 更新已缓存token数
        seq.num_cached_tokens = min(
            seq.num_cached_tokens + seq.num_scheduled_tokens, 
            seq.num_tokens
        )
        
        # 检查是否完成prefill
        if seq.num_cached_tokens < seq.num_tokens:
            # Chunked prefill，等待下一轮
            seq.num_scheduled_tokens = 0
            continue
            
        # Prefill完成，生成第一个token
        seq.append_token(token_id)
        seq.num_cached_tokens += 1
        seq.num_scheduled_tokens = 0
```

### 6.2 Decode后处理

```python
def postprocess_decode(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        seq.num_scheduled_tokens = 0
        
        # 检查终止条件
        if self._should_stop(seq, token_id):
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)

def _should_stop(self, seq, token_id):
    # EOS终止
    if not seq.ignore_eos and token_id == self.eos:
        return True
    # 长度限制
    if seq.num_completion_tokens == seq.max_tokens:
        return True
    return False
```

## 7. 调度器设计原则

### 7.1 精简原则

1. **单一策略**：只实现一种调度策略（FCFS）
2. **无优先级**：不区分请求优先级
3. **重计算抢占**：使用简单的重计算而非换出
4. **固定预算**：使用固定的批处理token预算

### 7.2 扩展点

如果需要扩展调度能力：

```python
class Scheduler:
    def __init__(self, config, strategy="fcfs"):
        self.strategy = strategy
        
    def schedule(self):
        if self.strategy == "fcfs":
            return self._schedule_fcfs()
        elif self.strategy == "lpm":
            return self._schedule_lpm()
        # ...
```

## 8. 性能优化点

### 8.1 避免频繁排序

```python
# 差：每次调度都排序
def schedule_lpm(self):
    self.waiting.sort(key=lambda x: -len(x.prefix_indices))
    
# 好：插入时维护有序性
def add(self, seq):
    bisect.insort(self.waiting, seq, key=lambda x: -len(x.prefix_indices))
```

### 8.2 批量操作

```python
# 差：逐个检查
for seq in seqs:
    if self.block_manager.can_allocate(seq):
        self.block_manager.allocate(seq)

# 好：批量检查和分配
allocatable = [seq for seq in seqs if self.block_manager.can_allocate(seq)]
for seq in allocatable:
    self.block_manager.allocate(seq)
```

### 8.3 预分配数据结构

```python
# 预分配足够大的队列
self.waiting = deque(maxlen=config.max_num_seqs * 2)
```
