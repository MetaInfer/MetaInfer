# 反模式（应避免）

## 1. 架构反模式

### 1.1 过度抽象

```python
# ❌ 反模式：多层抽象
class BaseEngine(ABC):
    @abstractmethod
    def step(self): ...

class BaseScheduler(ABC):
    @abstractmethod
    def schedule(self): ...

class BaseBlockManager(ABC):
    @abstractmethod
    def allocate(self): ...

class LLMEngine(BaseEngine):
    def __init__(self):
        self.scheduler = SchedulerFactory.create()
        self.block_manager = BlockManagerFactory.create()
    # ...

# ✅ 推荐：直接实现
class LLMEngine:
    def __init__(self, config):
        self.scheduler = Scheduler(config)
        self.block_manager = BlockManager(config.num_blocks, config.block_size)
    
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        ...
```

### 1.2 过度模块化

```python
# ❌ 反模式：过度拆分
# file: engine/scheduler/base.py
class BaseScheduler: ...

# file: engine/scheduler/policy/fcfs.py
class FCFSPolicy: ...

# file: engine/scheduler/policy/lpm.py
class LPMPolicy: ...

# file: engine/scheduler/policy/weight.py
class WeightPolicy: ...

# file: engine/scheduler/factory.py
class SchedulerFactory: ...

# ✅ 推荐：必要模块
# file: scheduler.py
class Scheduler:
    def __init__(self, config):
        self.waiting = deque()
        self.running = deque()
    
    def schedule(self): ...
```

### 1.3 注册表滥用

```python
# ❌ 反模式：不必要的注册表
MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

@register_model("llama")
class LlamaModel: ...

@register_model("mistral")
class MistralModel: ...

def get_model(name):
    return MODEL_REGISTRY[name]()

# ✅ 推荐：直接定义
class Model:  # 只支持一种模型
    ...
```

## 2. 配置反模式

### 2.1 参数爆炸

```python
# ❌ 反模式：过多参数
@dataclass
class ServerArgs:
    model_path: str
    tokenizer_path: str = None
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384
    max_model_len: int = 4096
    block_size: int = 256
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    attention_backend: str = "flash"
    sampling_backend: str = "pytorch"
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = False
    # ... 200+ 参数

# ✅ 推荐：最小参数
@dataclass
class Config:
    model_path: str
    max_model_len: int = 4096
    # 其他参数自动推导或固定
```

### 2.2 深层嵌套配置

```python
# ❌ 反模式：嵌套配置
@dataclass
class ParallelConfig:
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1

@dataclass
class AttentionConfig:
    backend: str = "flash"
    block_size: int = 256

@dataclass
class SchedulerConfig:
    policy: str = "fcfs"
    max_seqs: int = 512

@dataclass
class Config:
    parallel: ParallelConfig
    attention: AttentionConfig
    scheduler: SchedulerConfig
    # 访问: config.parallel.tp_size

# ✅ 推荐：扁平配置
@dataclass
class Config:
    model_path: str
    max_seqs: int = 512
    block_size: int = 256
    # 访问: config.max_seqs
```

## 3. 运行时反模式

### 3.1 过多条件分支

```python
# ❌ 反模式：条件分支爆炸
def forward(self, x):
    if self.config.model_type == "llama":
        if self.config.attention_backend == "flash":
            x = self.flash_attention(x)
        elif self.config.attention_backend == "triton":
            x = self.triton_attention(x)
        else:
            x = self.native_attention(x)
    elif self.config.model_type == "mistral":
        if self.config.use_sliding_window:
            x = self.sliding_attention(x)
        else:
            x = self.normal_attention(x)
    # ...

# ✅ 推荐：固定实现
def forward(self, x):
    x = self.attention(x)  # 只有Flash Attention
    ...
```

### 3.2 动态导入

```python
# ❌ 反模式：运行时动态导入
def get_attention_backend(name):
    if name == "flash":
        from .flash_attn import FlashAttention
        return FlashAttention
    elif name == "triton":
        from .triton_attn import TritonAttention
        return TritonAttention
    # ...

# ✅ 推荐：静态导入
from .flash_attn import FlashAttention

class Model:
    def __init__(self):
        self.attention = FlashAttention()
```

### 3.3 反射滥用

```python
# ❌ 反模式：通过字符串调用方法
def execute(self, method_name, *args):
    method = getattr(self, method_name)
    return method(*args)

result = self.execute("forward", input_ids)

# ✅ 推荐：直接调用
result = self.forward(input_ids)
```

## 4. 设计模式反模式

### 4.1 工厂模式滥用

```python
# ❌ 反模式：不必要的工厂
class SchedulerFactory:
    @staticmethod
    def create(policy, config):
        if policy == "fcfs":
            return FCFSScheduler(config)
        elif policy == "lpm":
            return LPMScheduler(config)
        # ...

scheduler = SchedulerFactory.create("fcfs", config)

# ✅ 推荐：直接实例化
scheduler = Scheduler(config)  # 只有一种调度策略
```

### 4.2 策略模式滥用

```python
# ❌ 反模式：过多的策略选择
class AllocationStrategy(ABC):
    @abstractmethod
    def allocate(self, seq): ...

class ContiguousAllocation(AllocationStrategy):
    ...

class PagedAllocation(AllocationStrategy):
    ...

class BlockManager:
    def __init__(self, strategy: AllocationStrategy):
        self.strategy = strategy
    
    def allocate(self, seq):
        return self.strategy.allocate(seq)

# ✅ 推荐：固定策略
class BlockManager:
    def allocate(self, seq):
        # 固定使用Paged Attention
        ...
```

### 4.3 观察者模式滥用

```python
# ❌ 反模式：不必要的事件系统
class Event:
    pass

class EventBus:
    def subscribe(self, event_type, handler): ...
    def publish(self, event): ...

event_bus = EventBus()
event_bus.subscribe("sequence_created", on_sequence_created)
event_bus.subscribe("token_generated", on_token_generated)
event_bus.publish(SequenceCreatedEvent(seq))

# ✅ 推荐：直接调用
def create_sequence(self, prompt):
    seq = Sequence(prompt)
    self.on_sequence_created(seq)
    return seq
```

## 5. 性能反模式

### 5.1 过度优化

```python
# ❌ 反模式：过早优化
@triton.jit
def custom_kernel(...):  # 手写复杂内核
    ...

# 在简单场景下反而更慢

# ✅ 推荐：先用标准实现
output = torch.matmul(a, b)  # PyTorch已优化
```

### 5.2 不必要的缓存

```python
# ❌ 反模式：缓存不需要的东西
class Model:
    def __init__(self):
        self._logits_cache = {}
    
    def forward(self, input_ids):
        key = tuple(input_ids.tolist())
        if key in self._logits_cache:
            return self._logits_cache[key]
        logits = self._forward(input_ids)
        self._logits_cache[key] = logits
        return logits

# ✅ 推荐：让KV Cache处理缓存
# KV Cache是正确的缓存位置
```

### 5.3 复杂的异步处理

```python
# ❌ 反模式：复杂的异步流水线
async def complex_pipeline(self):
    task1 = asyncio.create_task(self.stage1())
    task2 = asyncio.create_task(self.stage2())
    task3 = asyncio.create_task(self.stage3())
    # 复杂的同步逻辑...

# ✅ 推荐：简单同步流程
def simple_forward(self):
    x = self.stage1()
    x = self.stage2(x)
    x = self.stage3(x)
    return x
```

## 6. 代码风格反模式

### 6.1 过度注释

```python
# ❌ 反模式：注释过多
def forward(self, x):
    # 首先进行输入归一化
    # 使用RMSNorm
    # eps设置为1e-5
    x = self.norm(x)  # 归一化
    
    # 然后进行注意力计算
    # 使用Flash Attention
    # ...
    x = self.attention(x)  # 注意力
    
    # ✅ 推荐：代码即注释
def forward(self, x):
    x = self.norm(x)
    x = self.attention(x)
    x = self.mlp(x)
    return x
```

### 6.2 类型标注过度

```python
# ❌ 反模式：过度类型标注
from typing import Dict, List, Tuple, Optional, Union, Callable, TypeVar, Generic

T = TypeVar('T')

class BlockManager(Generic[T]):
    def allocate(
        self,
        seq: Sequence,
        num_blocks: Optional[int] = None,
    ) -> Tuple[List[int], Dict[str, Union[int, float]]]:
        ...

# ✅ 推荐：必要类型标注
class BlockManager:
    def allocate(self, seq: Sequence) -> list[int]:
        ...
```

## 7. 总结

### 7.1 反模式识别

| 反模式 | 表现 | 解决方案 |
|--------|------|----------|
| 过度抽象 | 多层继承/接口 | 直接实现 |
| 参数爆炸 | 配置项过多 | 固定配置 |
| 条件分支 | 大量if-else | 单一路径 |
| 工厂滥用 | 不必要的工厂 | 直接实例化 |
| 动态导入 | 运行时导入 | 静态导入 |

### 7.2 精简原则

1. **能用一行代码解决的不要用十行**
2. **能直接调用的不要动态派发**
3. **能固定的配置不要暴露**
4. **能省略的抽象不要添加**
