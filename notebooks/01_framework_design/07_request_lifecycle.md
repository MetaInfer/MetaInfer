# 请求生命周期

## 1. 请求处理完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              请求生命周期                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  用户输入
     │
     ▼
┌─────────────────┐
│ 1. Tokenization │  文本 → Token IDs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 创建Sequence │  封装请求状态
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 加入调度队列 │  Scheduler.add(seq)
└────────┬────────┘
         │
    ┌────┴────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
┌─────────────┐                      ┌─────────────┐
│   PREFILL   │                      │   DECODE    │
│  (首次计算)  │                      │  (逐token生成)│
└──────┬──────┘                      └──────┬──────┘
       │                                    │
       │  ┌─────────────────────────────────┘
       │  │
       ▼  ▼
┌─────────────────┐
│ 4. 推理循环     │  while not finished: step()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. 检查终止条件 │  EOS / max_tokens / stop_str
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Detokenize   │  Token IDs → 文本
└────────┬────────┘
         │
         ▼
    返回结果
```

## 2. Sequence数据结构

### 2.1 Sequence状态定义

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 等待调度
    RUNNING = auto()    # 正在推理
    FINISHED = auto()   # 已完成

class Sequence:
    def __init__(self, prompt: list[int], sampling_params: SamplingParams):
        # 基本信息
        self.seq_id: int = next_seq_id()
        self.status: SequenceStatus = SequenceStatus.WAITING
        
        # Token数据
        self.token_ids: list[int] = list(prompt)  # 所有token（prompt + completion）
        self.num_prompt_tokens: int = len(prompt)
        self.num_completion_tokens: int = 0
        
        # 调度相关
        self.num_cached_tokens: int = 0      # 已缓存的token数（前缀缓存）
        self.num_scheduled_tokens: int = 0   # 本次调度的token数
        
        # KV Cache
        self.block_table: list[int] = []     # KV Cache块映射
        
        # 采样参数
        self.temperature: float = sampling_params.temperature
        self.max_tokens: int = sampling_params.max_tokens
        self.ignore_eos: bool = sampling_params.ignore_eos
        
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)
    
    @property
    def last_token(self) -> int:
        return self.token_ids[-1]
    
    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
```

### 2.2 Sequence操作

```python
def append_token(self, token_id: int):
    """添加生成的token"""
    self.token_ids.append(token_id)
    self.num_completion_tokens += 1

def block(self, block_idx: int) -> list[int]:
    """获取指定block的token"""
    start = block_idx * self.block_size
    end = min(start + self.block_size, self.num_tokens)
    return self.token_ids[start:end]
```

## 3. 推理步骤详解

### 3.1 Step函数

```python
def step(self) -> list[tuple[int, list[int]]]:
    """
    执行一个推理步骤
    
    Returns:
        完成的序列列表: [(seq_id, completion_token_ids), ...]
    """
    # 1. 调度
    seqs, is_prefill = self.scheduler.schedule()
    
    # 2. 执行模型
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    
    # 3. 后处理
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    
    # 4. 收集完成的序列
    outputs = [
        (seq.seq_id, seq.completion_token_ids) 
        for seq in seqs 
        if seq.is_finished()
    ]
    
    return outputs, num_tokens
```

### 3.2 Generate函数

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
) -> list[dict]:
    """
    批量生成接口
    """
    # 标准化参数
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    
    # 添加所有请求
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    
    # 推理循环
    outputs = {}
    while not self.is_finished():
        finished, _ = self.step()
        for seq_id, token_ids in finished:
            outputs[seq_id] = token_ids
    
    # 按顺序整理输出
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [
        {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
        for token_ids in outputs
    ]
    
    return outputs
```

## 4. Prefill阶段

### 4.1 Prefill流程

```python
def prefill_phase(self, seq: Sequence):
    """
    Prefill阶段：处理prompt，计算初始KV Cache
    """
    # 1. 前缀缓存检查
    if self.block_manager.has_prefix_cache(seq):
        seq.num_cached_tokens = self.block_manager.get_cached_tokens(seq)
    
    # 2. 分配KV Cache块
    if not seq.block_table:
        self.block_manager.allocate(seq)
    
    # 3. 确定需要计算的token范围
    start = seq.num_cached_tokens
    end = seq.num_tokens
    seq.num_scheduled_tokens = end - start
    
    # 4. 准备输入
    input_ids = seq.token_ids[start:end]
    positions = list(range(start, end))
    
    # 5. 执行模型
    logits = self.model_runner.forward(input_ids, positions)
    
    # 6. 采样最后一个token
    next_token = self.sampler(logits[-1:], seq.temperature)
    seq.append_token(next_token)
    
    # 7. 更新状态
    seq.status = SequenceStatus.RUNNING
```

### 4.2 Chunked Prefill

```python
def chunked_prefill(self, seq: Sequence, max_tokens: int):
    """
    分块Prefill：长prompt分成多个chunk处理
    """
    total_tokens = seq.num_tokens - seq.num_cached_tokens
    
    if total_tokens <= max_tokens:
        # 可以一次处理完
        return self.prefill_phase(seq)
    
    # 分块处理
    seq.num_scheduled_tokens = max_tokens
    
    # 执行部分prefill
    start = seq.num_cached_tokens
    end = start + max_tokens
    
    input_ids = seq.token_ids[start:end]
    positions = list(range(start, end))
    
    # 只存储KV Cache，不采样
    self.model_runner.forward(input_ids, positions, store_kv_cache=True)
    
    # 更新缓存状态
    seq.num_cached_tokens = end
    seq.num_scheduled_tokens = 0
    # 保持WAITING状态，等待下一轮
```

## 5. Decode阶段

### 5.1 Decode流程

```python
def decode_phase(self, seq: Sequence):
    """
    Decode阶段：逐token生成
    """
    # 1. 准备输入（只需要最后一个token）
    input_id = seq.last_token
    position = len(seq) - 1
    
    # 2. 执行模型
    logits = self.model_runner.forward([input_id], [position])
    
    # 3. 采样
    next_token = self.sampler(logits, seq.temperature)
    seq.append_token(next_token)
    
    # 4. 更新KV Cache
    self.block_manager.may_append(seq)
    
    # 5. 检查终止条件
    if self._should_stop(seq, next_token):
        self._finish_sequence(seq)

def _should_stop(self, seq: Sequence, token_id: int) -> bool:
    # EOS检查
    if not seq.ignore_eos and token_id == self.eos_token_id:
        return True
    
    # 长度检查
    if seq.num_completion_tokens >= seq.max_tokens:
        return True
    
    # Stop string检查
    if seq.stop_str and self._match_stop_str(seq, seq.stop_str):
        return True
    
    return False
```

### 5.2 批量Decode

```python
def batch_decode(self, seqs: list[Sequence]):
    """
    批量Decode：多个序列同时生成
    """
    # 1. 准备批量输入
    input_ids = [seq.last_token for seq in seqs]
    positions = [len(seq) - 1 for seq in seqs]
    temperatures = [seq.temperature for seq in seqs]
    
    # 2. 批量执行
    logits = self.model_runner.forward(input_ids, positions)
    
    # 3. 批量采样
    next_tokens = self.sampler(logits, temperatures)
    
    # 4. 更新每个序列
    for seq, token_id in zip(seqs, next_tokens):
        seq.append_token(token_id)
        self.block_manager.may_append(seq)
        
        if self._should_stop(seq, token_id):
            self._finish_sequence(seq)
```

## 6. 抢占与恢复

### 6.1 抢占触发

```python
def check_preemption(self):
    """
    检查是否需要抢占
    """
    while not self.block_manager.can_allocate_new_request():
        if not self.running:
            break
        
        # 选择要抢占的序列（通常选择最后加入的）
        victim = self.running.pop()
        self.preempt(victim)

def preempt(self, seq: Sequence):
    """
    抢占序列：释放资源，放回等待队列
    """
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

### 6.2 恢复执行

```python
def resume_sequence(self, seq: Sequence):
    """
    恢复被抢占的序列
    """
    # 重新分配KV Cache
    self.block_manager.allocate(seq)
    
    # 需要重新计算prefill（因为KV Cache已释放）
    seq.num_cached_tokens = 0
    seq.status = SequenceStatus.WAITING
```

## 7. 终止处理

### 7.1 正常终止

```python
def _finish_sequence(self, seq: Sequence):
    """正常完成序列"""
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
```

### 7.2 异常终止

```python
def abort_sequence(self, seq_id: int, reason: str = ""):
    """异常终止序列"""
    # 查找序列
    seq = self._find_sequence(seq_id)
    if seq is None:
        return
    
    # 释放资源
    seq.status = SequenceStatus.FINISHED
    if seq.block_table:
        self.block_manager.deallocate(seq)
    
    # 从队列中移除
    self._remove_from_queues(seq)
```

## 8. 性能指标

### 8.1 吞吐量计算

```python
def compute_throughput(self, num_tokens: int, elapsed_time: float):
    """
    计算吞吐量
    
    Args:
        num_tokens: 正数表示prefill token数，负数表示decode token数
        elapsed_time: 耗时（秒）
    """
    if num_tokens > 0:
        prefill_throughput = num_tokens / elapsed_time
    else:
        decode_throughput = -num_tokens / elapsed_time
    
    return prefill_throughput, decode_throughput
```

### 8.2 延迟统计

```python
class SequenceMetrics:
    def __init__(self):
        self.arrival_time: float      # 请求到达时间
        self.first_token_time: float  # 首token时间
        self.completion_time: float   # 完成时间
        
    @property
    def ttft(self) -> float:
        """Time to First Token"""
        return self.first_token_time - self.arrival_time
    
    @property
    def latency(self) -> float:
        """总延迟"""
        return self.completion_time - self.arrival_time
    
    @property
    def tpot(self) -> float:
        """Time Per Output Token"""
        return (self.completion_time - self.first_token_time) / self.num_completion_tokens
```
