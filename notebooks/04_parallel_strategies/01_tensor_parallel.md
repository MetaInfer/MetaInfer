# 张量并行

## 1. 张量并行概述

张量并行（Tensor Parallelism, TP）将模型参数切分到多个GPU上，每个GPU持有部分参数，通过All-Reduce同步结果。

```
单GPU模型:
┌─────────────────────────────────────┐
│         Full Model Weights          │
│              [H, H]                 │
└─────────────────────────────────────┘

TP=2 切分:
┌───────────────────┐ ┌───────────────────┐
│  GPU 0: W[:, :H/2]│ │  GPU 1: W[:, H/2:]│
│   Col Parallel    │ │   Col Parallel    │
└───────────────────┘ └───────────────────┘
          All-Reduce
```

## 2. 并行线性层

### 2.1 Column Parallel Linear

```python
class ColumnParallelLinear(nn.Module):
    """
    列并行线性层：输出维度切分
    
    用途: QKV投影
    
    Y = XW, W: [in_features, out_features]
    切分: W = [W_0, W_1], 每个 GPU 持有 W_i: [in_features, out_features/tp_size]
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_size: int = 1,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = get_tensor_parallel_rank()
        
        assert out_features % tp_size == 0
        self.out_features_per_partition = out_features // tp_size
        
        # 只存储切分后的权重
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        每个 GPU 计算部分输出
        无需通信，输出已在各 GPU 分散
        """
        return F.linear(x, self.weight, self.bias)
```

### 2.2 Row Parallel Linear

```python
class RowParallelLinear(nn.Module):
    """
    行并行线性层：输入维度切分
    
    用途: O投影, Down投影
    
    Y = XW, W: [in_features, out_features]
    切分: X = [X_0, X_1], W = [W_0; W_1]
    每个 GPU 计算 Y_i = X_i W_i, 然后 All-Reduce
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_size: int = 1,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = get_tensor_parallel_rank()
        
        assert in_features % tp_size == 0
        self.in_features_per_partition = in_features // tp_size
        
        # 只存储切分后的权重
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        每个 GPU 计算部分结果，然后 All-Reduce
        """
        output = F.linear(x, self.weight)
        
        # All-Reduce 求和
        torch.distributed.all_reduce(output, group=self.tp_group)
        
        if self.bias is not None:
            output = output + self.bias  # bias 只在一个 rank 加
        
        return output
```

### 2.3 QKV Parallel Linear

```python
class QKVParallelLinear(nn.Module):
    """
    QKV并行线性层：同时处理Q、K、V投影
    
    支持 GQA: Q的head数可能与K、V不同
    """
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,        # Q的head数
        num_kv_heads: int,     # K、V的head数
        bias: bool = False,
        tp_size: int = 1,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = get_tensor_parallel_rank()
        
        # 计算每个partition的head数
        self.num_heads_per_partition = num_heads // tp_size
        self.num_kv_heads_per_partition = num_kv_heads // tp_size
        
        # 总输出维度
        q_dim = self.num_heads_per_partition * head_dim
        kv_dim = self.num_kv_heads_per_partition * head_dim
        total_dim = q_dim + 2 * kv_dim
        
        self.weight = nn.Parameter(
            torch.empty(total_dim, hidden_size)
        )
    
    def forward(self, x: torch.Tensor):
        """
        返回切分后的Q、K、V
        """
        qkv = F.linear(x, self.weight)
        
        # 分离Q、K、V
        q_dim = self.num_heads_per_partition * self.head_dim
        kv_dim = self.num_kv_heads_per_partition * self.head_dim
        
        q = qkv[..., :q_dim]
        k = qkv[..., q_dim:q_dim + kv_dim]
        v = qkv[..., q_dim + kv_dim:]
        
        return q, k, v
```

### 2.4 Merged Column Parallel Linear

```python
class MergedColumnParallelLinear(nn.Module):
    """
    合并列并行线性层：合并Gate和Up投影
    
    用途: MLP的gate_up_proj
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,  # gate + up 的总维度
        bias: bool = False,
        tp_size: int = 1,
    ):
        super().__init__()
        self.tp_size = tp_size
        
        assert out_features % (2 * tp_size) == 0
        self.out_features_per_partition = out_features // tp_size
        
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
    
    def forward(self, x: torch.Tensor):
        """
        返回gate和up两部分
        """
        output = F.linear(x, self.weight)
        
        # 分离gate和up
        half = self.out_features_per_partition // 2
        gate = output[..., :half]
        up = output[..., half:]
        
        return gate, up
```

## 3. 并行通信

### 3.1 进程组初始化

```python
def init_tensor_parallel(tp_size: int):
    """初始化张量并行进程组"""
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    assert world_size % tp_size == 0
    
    # 创建张量并行进程组
    tp_group = torch.distributed.new_group(
        ranks=list(range(tp_size))
    )
    
    return tp_group
```

### 3.2 All-Reduce

```python
def all_reduce(tensor: torch.Tensor, group=None):
    """All-Reduce求和"""
    torch.distributed.all_reduce(tensor, group=group)
    return tensor

def all_reduce_async(tensor: torch.Tensor, group=None):
    """异步All-Reduce"""
    handle = torch.distributed.all_reduce(
        tensor, 
        group=group, 
        async_op=True
    )
    return handle
```

## 4. 多进程架构

### 4.1 进程启动

```python
import torch.multiprocessing as mp

def launch_tensor_parallel(config: Config):
    """启动张量并行进程"""
    world_size = config.tensor_parallel_size
    
    # 使用spawn模式
    ctx = mp.get_context("spawn")
    
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=worker_main,
            args=(config, rank, world_size)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

def worker_main(config, rank, world_size):
    """Worker进程主函数"""
    # 设置设备
    torch.cuda.set_device(rank)
    
    # 初始化分布式
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:2333",
        world_size=world_size,
        rank=rank,
    )
    
    # 创建模型
    model = Model(config, rank)
    
    # 运行推理循环
    ...
```

### 4.2 主从通信

```python
class ModelRunner:
    def __init__(self, config, rank, events):
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        
        if rank == 0:
            # 主进程
            self.shm = SharedMemory(name="llm_cache", create=True, size=2**20)
            self.events = events
        else:
            # Worker进程
            self.shm = SharedMemory(name="llm_cache")
            self.loop()
    
    def call(self, method_name, *args):
        """主进程调用，广播到所有Worker"""
        if self.world_size > 1 and self.rank == 0:
            self._broadcast_call(method_name, *args)
        return getattr(self, method_name)(*args)
    
    def _broadcast_call(self, method_name, *args):
        """广播调用到Worker"""
        data = pickle.dumps([method_name, *args])
        self.shm.buf[:4] = len(data).to_bytes(4, "little")
        self.shm.buf[4:4+len(data)] = data
        for event in self.events:
            event.set()
    
    def loop(self):
        """Worker进程主循环"""
        while True:
            # 等待主进程通知
            self.event.wait()
            
            # 读取调用
            data_len = int.from_bytes(self.shm.buf[:4], "little")
            method_name, args = pickle.loads(self.shm.buf[4:4+data_len])
            
            # 执行
            getattr(self, method_name)(*args)
            
            # 清除事件
            self.event.clear()
```

## 5. 权重加载

### 5.1 切分权重加载

```python
def load_tensor_parallel_weights(model, state_dict, tp_rank, tp_size):
    """加载张量并行权重"""
    for name, param in model.named_parameters():
        if name not in state_dict:
            continue
        
        loaded_weight = state_dict[name]
        
        # 检查是否需要切分
        if hasattr(param, "weight_loader"):
            # 使用参数的weight_loader方法
            param.weight_loader(param, loaded_weight)
        else:
            # 直接复制
            param.data.copy_(loaded_weight)
```

### 5.2 Weight Loader

```python
class ColumnParallelLinear:
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """列并行权重加载"""
        # 按输出维度切分
        shard_size = loaded_weight.shape[0] // self.tp_size
        start = self.tp_rank * shard_size
        end = start + shard_size
        
        param.data.copy_(loaded_weight[start:end])

class RowParallelLinear:
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """行并行权重加载"""
        # 按输入维度切分
        shard_size = loaded_weight.shape[1] // self.tp_size
        start = self.tp_rank * shard_size
        end = start + shard_size
        
        param.data.copy_(loaded_weight[:, start:end])
```

## 6. 精简实现建议

### 6.1 最小化张量并行

```python
# 精简框架可以不实现张量并行
# 只需要单GPU版本
if config.tensor_parallel_size > 1:
    raise NotImplementedError("TP not supported in minimal implementation")
```

### 6.2 必要的张量并行

如果需要TP，推荐：

1. **固定TP size**: 只支持特定的TP配置
2. **简化通信**: 使用NCCL原生API
3. **同步模式**: 使用同步通信，避免复杂的异步处理

```python
# 简化的TP实现
class SimpleColumnParallel(nn.Module):
    def __init__(self, in_features, out_features, tp_rank, tp_size):
        super().__init__()
        self.out_features = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features))
    
    def forward(self, x):
        return F.linear(x, self.weight)
```
