# 整体架构设计

## 1. 核心架构模式

LLM推理框架的核心架构可以抽象为以下组件协作模式：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Engine (引擎层)                                 │
│  职责：协调所有组件，对外提供统一接口                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐              │
│   │   Scheduler   │───▶│  ModelRunner  │───▶│    Sampler    │              │
│   │   (调度器)    │    │  (模型执行器)  │    │   (采样器)    │              │
│   └───────┬───────┘    └───────┬───────┘    └───────────────┘              │
│           │                    │                                            │
│           ▼                    ▼                                            │
│   ┌───────────────┐    ┌───────────────┐                                   │
│   │ BlockManager  │    │  KV Cache     │                                   │
│   │ (内存管理器)  │    │  (缓存存储)   │                                   │
│   └───────────────┘    └───────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 精简架构设计（nano-vllm模式）

### 2.1 单进程架构

```python
# 最简架构：单进程，所有组件在同一进程中
class LLMEngine:
    def __init__(self, model, **kwargs):
        self.config = Config(model, **kwargs)
        self.scheduler = Scheduler(self.config)
        self.model_runner = ModelRunner(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
    
    def step(self):
        # 核心推理循环
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

**优点**：
- 代码简洁，调试方便
- 组件间通信无开销
- 适合学习和小规模部署

**缺点**：
- 无法利用多GPU张量并行
- 难以处理高并发请求

### 2.2 多进程张量并行架构

```python
# 张量并行架构：spawn多进程，共享内存通信
class LLMEngine:
    def __init__(self, model, **kwargs):
        self.config = Config(model, **kwargs)
        
        # 启动多个GPU Worker进程
        ctx = mp.get_context("spawn")
        for rank in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, rank, event))
            process.start()
        
        # 主进程运行rank 0
        self.model_runner = ModelRunner(config, 0, events)
        self.scheduler = Scheduler(config)
```

**关键设计点**：
- 使用`multiprocessing.Event`同步
- 使用`SharedMemory`传递参数
- 使用NCCL进行分布式通信

## 3. 生产级架构设计（nano-sglang模式）

### 3.1 多进程服务架构

```
┌─────────────────┐    ZMQ     ┌─────────────────┐    ZMQ    ┌─────────────────┐
│ TokenizerManager │ ─────────▶ │  RouterManager  │ ────────▶ │ DetokenizerMgr  │
│   (主进程)       │            │   (路由进程)     │           │   (解码进程)    │
└─────────────────┘            └────────┬────────┘           └─────────────────┘
                                        │
                                        │ rpyc
                                        ▼
                           ┌───────────────────────────┐
                           │      Model Workers        │
                           │   (Tensor Parallel GPUs)  │
                           └───────────────────────────┘
```

**进程职责划分**：

| 进程 | 职责 | 通信方式 |
|------|------|----------|
| TokenizerManager | HTTP请求处理、文本编码 | ZMQ PUSH |
| RouterManager | 请求路由、调度协调 | ZMQ PUSH/PULL |
| DetokenizerManager | Token解码、结果返回 | ZMQ PUSH |
| ModelWorkers | 模型推理执行 | rpyc/NCCL |

### 3.2 架构设计决策

**何时使用单进程架构**：
- 单GPU部署
- 学习和研究目的
- 快速原型验证

**何时使用多进程架构**：
- 多GPU张量并行
- 高并发服务
- 需要异步处理

## 4. 组件接口设计

### 4.1 Scheduler接口

```python
class Scheduler:
    def __init__(self, config: Config):
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列
    
    def add(self, seq: Sequence) -> None:
        """添加请求到等待队列"""
        
    def schedule(self) -> tuple[list[Sequence], bool]:
        """返回：待处理序列列表 + 是否为prefill阶段"""
        
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool) -> None:
        """处理推理结果，更新序列状态"""
    
    def is_finished(self) -> bool:
        """检查是否所有请求都已完成"""
```

### 4.2 ModelRunner接口

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int):
        self.model = load_model(config.model)
        self.allocate_kv_cache()
        
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """执行模型推理，返回采样的token ids"""
        
    def allocate_kv_cache(self) -> None:
        """分配KV Cache内存"""
```

### 4.3 BlockManager接口

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.blocks: list[Block] = [...]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够空闲块"""
        
    def allocate(self, seq: Sequence) -> None:
        """为序列分配KV Cache块"""
        
    def deallocate(self, seq: Sequence) -> None:
        """释放序列的KV Cache块"""
```

## 5. 数据流设计

### 5.1 请求数据流

```
Request Input (prompt/sampling_params)
    │
    ▼
┌─────────────────┐
│   Tokenize      │  text → token_ids
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Sequence │  封装请求状态
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scheduler.add() │  加入等待队列
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Prefill    Decode
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Sequence完成    │  输出结果
└─────────────────┘
```

### 5.2 推理步骤数据流

```
step()
  │
  ├──▶ scheduler.schedule()
  │      │
  │      ├── 返回待处理序列
  │      └── 返回is_prefill标志
  │
  ├──▶ model_runner.run(seqs, is_prefill)
  │      │
  │      ├── 准备输入张量
  │      ├── 执行前向传播
  │      └── 采样生成token
  │
  └──▶ scheduler.postprocess(seqs, token_ids)
         │
         ├── 更新序列状态
         └── 检查完成条件
```

## 6. 配置设计

### 6.1 精简配置类

```python
@dataclass
class Config:
    # 模型配置
    model: str                    # 模型路径
    hf_config: AutoConfig = None  # HuggingFace配置
    
    # 推理配置
    max_num_seqs: int = 512              # 最大并发序列数
    max_num_batched_tokens: int = 16384  # 最大批处理token数
    max_model_len: int = 4096            # 最大模型长度
    
    # KV Cache配置
    kvcache_block_size: int = 256        # KV Cache块大小
    num_kvcache_blocks: int = -1         # KV Cache块数量（自动计算）
    gpu_memory_utilization: float = 0.9  # GPU内存利用率
    
    # 并行配置
    tensor_parallel_size: int = 1        # 张量并行大小
    
    # 优化配置
    enforce_eager: bool = False          # 是否禁用CUDA Graph
```

### 6.2 配置设计原则

1. **最小必要**：只保留必须的配置项
2. **自动推导**：能自动计算的参数不暴露给用户
3. **单一职责**：配置类只负责参数存储，不包含逻辑
4. **类型安全**：使用dataclass提供类型检查

## 7. 架构设计原则

### 7.1 精简原则

1. **单一路径**：避免if-else分支，使用独立函数
2. **直接调用**：避免动态派发，直接调用具体实现
3. **内联数据**：避免抽象层，数据直接存储在需要的地方
4. **固定配置**：避免可配置性，固定最优配置

### 7.2 扩展原则（如需扩展）

1. **接口隔离**：定义最小接口
2. **组合优于继承**：使用组合添加功能
3. **依赖注入**：通过构造函数传入依赖
4. **配置驱动**：通过配置切换行为而非代码分支

## 8. 代码组织结构

### 8.1 精简项目结构

```
nano_llm/
├── __init__.py          # 模块入口
├── config.py            # 配置类
├── llm.py               # 主入口类
├── sampling_params.py   # 采样参数
├── engine/
│   ├── llm_engine.py    # 引擎主类
│   ├── scheduler.py     # 调度器
│   ├── sequence.py      # 序列状态
│   ├── block_manager.py # KV Cache管理
│   └── model_runner.py  # 模型执行
├── layers/
│   ├── attention.py     # 注意力层
│   ├── linear.py        # 线性层
│   ├── sampler.py       # 采样器
│   └── ...              # 其他层
├── models/
│   └── llama.py         # 具体模型实现
└── utils/
    ├── context.py       # 上下文管理
    └── loader.py        # 权重加载
```

### 8.2 模块职责

| 模块 | 职责 | 依赖 |
|------|------|------|
| engine/ | 核心推理逻辑 | layers/, models/ |
| layers/ | 神经网络层实现 | 无 |
| models/ | 模型架构定义 | layers/ |
| utils/ | 辅助工具 | 无 |
