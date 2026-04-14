# 模型执行器

## 1. 核心职责

ModelRunner负责：

1. **模型加载**：加载模型权重到GPU
2. **KV Cache分配**：分配和管理KV Cache内存
3. **输入准备**：将调度结果转换为模型输入张量
4. **前向传播**：执行模型推理
5. **采样**：从logits生成token

## 2. 初始化流程

### 2.1 完整初始化流程

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        
        # 1. 初始化分布式
        dist.init_process_group("nccl", ...)
        torch.cuda.set_device(rank)
        
        # 2. 创建模型
        self.model = ModelClass(config.hf_config)
        
        # 3. 加载权重
        load_model(self.model, config.model)
        
        # 4. 创建采样器
        self.sampler = Sampler()
        
        # 5. Warmup（确定峰值内存）
        self.warmup_model()
        
        # 6. 分配KV Cache
        self.allocate_kv_cache()
        
        # 7. 捕获CUDA Graph（可选）
        if not config.enforce_eager:
            self.capture_cudagraph()
```

### 2.2 Warmup流程

```python
def warmup_model(self):
    """
    Warmup模型以确定峰值内存使用
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 模拟最大batch的prefill
    max_tokens = min(self.config.max_num_batched_tokens, self.config.max_model_len)
    num_seqs = min(self.config.max_num_seqs, max_tokens // self.config.max_model_len)
    
    seqs = [Sequence([0] * self.config.max_model_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = self.config.max_model_len
        
    self.run(seqs, is_prefill=True)
    torch.cuda.empty_cache()
```

## 3. 输入准备

### 3.1 Prefill输入准备

```python
def prepare_prefill(self, seqs: list[Sequence]):
    """
    准备Prefill阶段的输入张量
    """
    input_ids = []
    positions = []
    cu_seqlens_q = [0]  # Query累积序列长度
    cu_seqlens_k = [0]  # Key累积序列长度
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []    # KV Cache槽位映射
    
    for seq in seqs:
        seqlen = len(seq)
        start = min(seq.num_cached_tokens, seqlen - 1)  # 前缀缓存跳过已缓存部分
        seqlen_q = seq.num_scheduled_tokens
        seqlen_k = seqlen
        end = start + seqlen_q
        
        # 输入token和位置
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        
        # 累积序列长度（用于变长Flash Attention）
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        
        # 计算KV Cache槽位
        slot_mapping.extend(self._compute_slot_mapping(seq, start, end))
    
    # 转换为GPU张量
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    return input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping
```

### 3.2 Decode输入准备

```python
def prepare_decode(self, seqs: list[Sequence]):
    """
    准备Decode阶段的输入张量
    """
    input_ids = []      # 只需要最后一个token
    positions = []      # 当前位置
    slot_mapping = []   # 新token的KV Cache槽位
    context_lens = []   # 上下文长度
    
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        
        # 新token存储在最后一个block的末尾
        slot = seq.block_table[-1] * self.block_size + (len(seq) - 1) % self.block_size
        slot_mapping.append(slot)
    
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    block_tables = self._prepare_block_tables(seqs)
    
    return input_ids, positions, slot_mapping, context_lens, block_tables
```

### 3.3 Block Tables准备

```python
def prepare_block_tables(self, seqs: list[Sequence]):
    """
    准备Block Tables张量
    """
    max_len = max(len(seq.block_table) for seq in seqs)
    
    # 填充对齐
    block_tables = [
        seq.block_table + [-1] * (max_len - len(seq.block_table))
        for seq in seqs
    ]
    
    return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
```

## 4. 上下文传递

### 4.1 全局上下文模式

避免在模型各层之间传递大量参数，使用全局上下文：

```python
@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor = None    # Decode用
    block_tables: torch.Tensor = None    # Decode用

# 全局变量
_CONTEXT = Context()

def get_context(): return _CONTEXT
def set_context(**kwargs): 
    for k, v in kwargs.items():
        setattr(_CONTEXT, k, v)
def reset_context():
    # 重置上下文
    ...
```

### 4.2 使用方式

```python
# 在ModelRunner中设置上下文
def run(self, seqs, is_prefill):
    if is_prefill:
        input_ids, positions, ... = self.prepare_prefill(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, ...)
    else:
        input_ids, positions, ... = self.prepare_decode(seqs)
        set_context(False, slot_mapping=slot_mapping, ...)
    
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures)
    
    reset_context()
    return token_ids

# 在Attention层中获取上下文
def forward(self, x):
    context = get_context()
    if context.is_prefill:
        # Prefill路径
        flash_attn_varlen_func(...)
    else:
        # Decode路径
        flash_attn_with_kvcache(...)
```

## 5. 模型执行

### 5.1 标准执行

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    """
    标准模型前向传播
    """
    hidden_states = self.model.embed_tokens(input_ids)
    
    for layer in self.model.layers:
        hidden_states = layer(hidden_states, positions)
    
    hidden_states = self.model.norm(hidden_states)
    logits = self.model.lm_head(hidden_states)
    
    return logits
```

### 5.2 CUDA Graph优化执行

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    """
    使用CUDA Graph加速Decode阶段
    """
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # Prefill或大batch：常规执行
        return self.model(self.model(input_ids, positions))
    
    # Decode小batch：使用CUDA Graph
    bs = input_ids.size(0)
    context = get_context()
    
    # 选择合适的graph（batch size >= 实际bs）
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    
    # 填充输入buffer
    self.graph_vars["input_ids"][:bs] = input_ids
    self.graph_vars["positions"][:bs] = positions
    self.graph_vars["slot_mapping"][:bs] = context.slot_mapping
    self.graph_vars["context_lens"][:bs] = context.context_lens
    self.graph_vars["block_tables"][:bs] = context.block_tables
    
    # 重放graph
    graph.replay()
    
    return self.model.compute_logits(self.graph_vars["outputs"][:bs])
```

### 5.3 CUDA Graph捕获

```python
@torch.inference_mode()
def capture_cudagraph(self):
    """
    为不同batch size捕获CUDA Graph
    """
    max_bs = min(self.config.max_num_seqs, 512)
    
    # 预分配buffer
    input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
    positions = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
    outputs = torch.zeros(max_bs, self.config.hidden_size, device="cuda")
    
    # 捕获不同batch size的graph
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None
    
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        
        # Warmup
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        # 捕获
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
            
        self.graphs[bs] = graph
    
    self.graph_vars = {
        "input_ids": input_ids,
        "positions": positions,
        "outputs": outputs,
        ...
    }
```

## 6. 张量并行

### 6.1 多进程架构

```python
class ModelRunner:
    def __init__(self, config, rank, events):
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        
        # 初始化NCCL
        dist.init_process_group(
            "nccl",
            init_method="tcp://localhost:2333",
            world_size=self.world_size,
            rank=rank
        )
        
        if rank == 0:
            # 主进程：接收请求，分发任务
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        else:
            # Worker进程：监听执行
            self.shm = SharedMemory(name="nanovllm")
            self.loop()
```

### 6.2 进程间通信

```python
def call(self, method_name, *args):
    """主进程调用，分发到所有Worker"""
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)
    method = getattr(self, method_name)
    return method(*args)

def write_shm(self, method_name, *args):
    """写入共享内存，通知Worker"""
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.events:
        event.set()  # 唤醒Worker

def loop(self):
    """Worker进程主循环"""
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

## 7. 采样准备

```python
def prepare_sample(self, seqs: list[Sequence]):
    """准备采样参数"""
    temperatures = [seq.temperature for seq in seqs]
    return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

def run(self, seqs, is_prefill):
    # 准备输入
    if is_prefill:
        input_ids, positions = self.prepare_prefill(seqs)
    else:
        input_ids, positions = self.prepare_decode(seqs)
    
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    
    # 执行模型
    logits = self.run_model(input_ids, positions, is_prefill)
    
    # 采样（只在rank 0执行）
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    
    return token_ids
```

## 8. 设计选择建议

### 8.1 精简框架推荐

| 功能 | 推荐实现 | 原因 |
|------|----------|------|
| 分布式 | 无 | 单GPU足够学习 |
| CUDA Graph | 可选 | 有明显加速但增加复杂度 |
| 全局上下文 | 使用 | 简化参数传递 |
| 异步传输 | 使用 | pin_memory + non_blocking |

### 8.2 性能优化点

1. **Pinned Memory**：使用`pin_memory=True`加速CPU到GPU传输
2. **Non-blocking Transfer**：使用`.cuda(non_blocking=True)`异步传输
3. **CUDA Graph**：Decode阶段使用CUDA Graph减少kernel launch开销
4. **KV Cache复用**：前缀缓存避免重复计算
