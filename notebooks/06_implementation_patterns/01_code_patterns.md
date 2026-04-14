# 代码组织模式

## 1. 精简框架的代码原则

### 1.1 核心原则

1. **单一路径**：避免if-else分支，每个功能只有一条执行路径
2. **直接调用**：避免动态派发，直接调用具体实现
3. **固定配置**：避免可配置性，固定最优配置
4. **最小抽象**：只在必要时引入抽象层

### 1.2 反模式识别

```python
# ❌ 反模式：动态派发
def get_attention(backend):
    if backend == "flash":
        return FlashAttention()
    elif backend == "triton":
        return TritonAttention()
    # ...

# ✅ 推荐：直接实例化
attention = FlashAttention()
```

## 2. 模块组织模式

### 2.1 扁平结构

```python
# 精简框架目录结构
nano_llm/
├── config.py          # 配置
├── engine.py          # 引擎（包含Scheduler, BlockManager）
├── model.py           # 模型（包含所有层）
├── attention.py       # Attention实现
├── sampler.py         # 采样器
└── utils.py           # 工具函数

# 而非：
complex_llm/
├── engine/
│   ├── scheduler/
│   ├── block_manager/
│   ├── sequence/
│   └── ...
├── model/
│   ├── layers/
│   ├── attention/
│   └── ...
└── ...
```

### 2.2 内聚组织

```python
# 相关功能放在同一文件
# engine.py
class Scheduler:
    ...

class BlockManager:
    ...

class Sequence:
    ...

# 而非分散在多个文件
```

## 3. 配置模式

### 3.1 固定配置

```python
# ❌ 反模式：大量可配置项
@dataclass
class Config:
    model_path: str
    max_seqs: int = 512
    max_tokens: int = 16384
    block_size: int = 256
    attention_backend: str = "flash"
    scheduler_policy: str = "fcfs"
    quantization: str = "fp16"
    # ... 200+ 参数

# ✅ 推荐：最小配置
@dataclass
class Config:
    model_path: str
    # 其他参数自动推导或固定
```

### 3.2 自动推导

```python
def create_config(model_path: str) -> Config:
    """从模型自动推导配置"""
    hf_config = AutoConfig.from_pretrained(model_path)
    
    return Config(
        model_path=model_path,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        # 自动计算，不暴露给用户
    )
```

## 4. 接口设计模式

### 4.1 简单接口

```python
# ❌ 反模式：复杂接口
class LLM:
    def generate(
        self,
        prompts,
        sampling_params=None,
        use_tqdm=True,
        prefix_cache=True,
        attention_backend=None,
        stream=False,
        # ... 更多参数
    ): ...

# ✅ 推荐：最小接口
class LLM:
    def generate(self, prompts, temperature=1.0, max_tokens=16):
        ...
```

### 4.2 数据类封装

```python
# 使用数据类封装参数
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 16

class LLM:
    def generate(self, prompts, params=None):
        params = params or SamplingParams()
        # 使用params
```

## 5. 错误处理模式

### 5.1 快速失败

```python
# ❌ 反模式：复杂错误处理
def allocate_memory(size):
    try:
        return torch.empty(size, device="cuda")
    except RuntimeError as e:
        if "out of memory" in str(e):
            # 尝试清理
            torch.cuda.empty_cache()
            try:
                return torch.empty(size, device="cuda")
            except:
                # 尝试更小的size
                ...
        else:
            raise

# ✅ 推荐：快速失败
def allocate_memory(size):
    return torch.empty(size, device="cuda")
    # 让错误直接抛出，由调用者处理
```

### 5.2 断言优先

```python
def forward(self, input_ids):
    # 使用断言检查前置条件
    assert input_ids.dim() == 2
    assert input_ids.max() < self.vocab_size
    
    # 正常逻辑
    ...
```

## 6. 状态管理模式

### 6.1 全局上下文

```python
# 使用全局变量避免参数传递
_CONTEXT = None

def get_context():
    return _CONTEXT

def set_context(**kwargs):
    global _CONTEXT
    _CONTEXT = Context(**kwargs)

# 在模型各层中使用
context = get_context()
```

### 6.2 状态封装

```python
# 将相关状态封装在一起
class Sequence:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.block_table = []
        self.status = Status.WAITING
        # 所有状态集中管理
```

## 7. 性能优化模式

### 7.1 避免过度优化

```python
# ❌ 反模式：过早优化
def complex_optimized_function():
    # 复杂的手动优化
    ...

# ✅ 推荐：先简洁，后优化
def simple_function():
    # 清晰的实现
    ...

# 使用torch.compile自动优化
@torch.compile
def optimized_function():
    ...
```

### 7.2 内置优化

```python
# 利用PyTorch内置优化
# Pinned memory
tensor = torch.tensor(data, pin_memory=True)

# Non-blocking transfer
tensor = tensor.cuda(non_blocking=True)

# torch.compile
model = torch.compile(model)
```

## 8. 代码复用模式

### 8.1 避免过度抽象

```python
# ❌ 反模式：过度抽象
class BaseAttention(ABC):
    @abstractmethod
    def forward(self, q, k, v): ...

class FlashAttention(BaseAttention):
    def forward(self, q, k, v): ...

class TritonAttention(BaseAttention):
    def forward(self, q, k, v): ...

# ✅ 推荐：直接实现
class Attention:
    def forward(self, q, k, v):
        # 直接使用Flash Attention
        return flash_attn_func(q, k, v)
```

### 8.2 组合优于继承

```python
# 使用组合而非继承
class Model:
    def __init__(self, config):
        self.embed = Embedding(config)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config)
        self.head = Linear(config)
```

## 9. 测试模式

### 9.1 简单测试

```python
# 直接测试核心功能
def test_generate():
    llm = LLM("path/to/model")
    outputs = llm.generate(["Hello, world!"])
    assert len(outputs) == 1
    assert len(outputs[0]["text"]) > 0
```

### 9.2 回归测试

```python
# 与参考实现对比
def test_correctness():
    llm = LLM("path/to/model")
    output = llm.generate(["Test prompt"], temperature=0)
    
    # 与HuggingFace对比
    hf_output = hf_generate("Test prompt", temperature=0)
    
    assert output == hf_output
```

## 10. 总结

### 10.1 精简框架代码量对比

| 框架 | 代码行数 | 模式复杂度 |
|------|----------|------------|
| nano-vllm | ~1.2K | 简单直接 |
| nano-sglang | ~3K | 简单直接 |
| vLLM | ~500K | 复杂抽象 |
| SGLang | ~200K | 复杂抽象 |

### 10.2 核心原则

1. **简洁 > 灵活**：宁可重复代码也不要过度抽象
2. **直接 > 间接**：直接调用比动态派发更好
3. **固定 > 可配置**：固定配置减少运行时判断
4. **正确 > 优化**：先保证正确，再考虑性能
