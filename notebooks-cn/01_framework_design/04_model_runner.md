# Model Runner — 前向执行

## 核心职责

Model Runner 是 **调度器的逻辑决策** 与 **GPU 上的实际执行** 之间的桥梁。它负责：

1. 将调度好的 batch 转为 GPU 张量
2. 执行模型前向
3. 从 logits 中采样下一个词元
4. （可选）在 decode 阶段使用 CUDA Graph 加速

## 输入准备

### Prefill 输入准备

Prefill 阶段对每个请求 **同时** 处理多个词元：

```python
def prepare_prefill(sequences):
    # 将所有 prompt 词元拼成扁平 1D 张量
    input_ids = concat([seq.tokens[cached:total] for seq in sequences])

    # 构造累积序列长度，供变长 attention 使用
    cu_seqlens = [0]
    for seq in sequences:
        cu_seqlens.append(cu_seqlens[-1] + seq.num_new_tokens)

    # 将每个词元映射到 KV 缓存的物理位置
    slot_mapping = []
    for seq in sequences:
        for pos in range(seq.cached, seq.total):
            block_id = seq.block_table[pos // block_size]
            offset = pos % block_size
            slot_mapping.append(block_id * block_size + offset)

    # 块表：用于读取已缓存前缀的 KV
    block_tables = pad([seq.block_table for seq in sequences])

    return input_ids, cu_seqlens, slot_mapping, block_tables
```

### Decode 输入准备

Decode 阶段每个请求 **恰好一个词元**：

```python
def prepare_decode(sequences):
    # 每个序列一个词元（最后生成的那个）
    input_ids = [seq.tokens[-1] for seq in sequences]

    # 各词元的位置
    positions = [seq.num_tokens - 1 for seq in sequences]

    # 新 KV 条目的物理槽位
    slot_mapping = []
    for seq in sequences:
        pos = seq.num_tokens - 1
        block_id = seq.block_table[pos // block_size]
        offset = pos % block_size
        slot_mapping.append(block_id * block_size + offset)

    # 完整块表：对历史所有词元做 attention
    block_tables = pad([seq.block_table for seq in sequences])
    context_lens = [seq.num_tokens for seq in sequences]

    return input_ids, positions, slot_mapping, block_tables, context_lens
```

## 前向执行

### 基本流程

```python
def run(sequences, is_prefill):
    if is_prefill:
        inputs = prepare_prefill(sequences)
    else:
        inputs = prepare_decode(sequences)

    # 为注意力层设置全局上下文
    context.is_prefill = is_prefill
    context.slot_mapping = inputs.slot_mapping
    context.block_tables = inputs.block_tables

    # 执行模型
    logits = model(inputs.input_ids, inputs.positions)

    # 采样下一个词元
    if is_prefill:
        # 只需每条序列最后一个位置的 logits
        last_indices = cumsum(seq_lens) - 1
        logits = logits[last_indices]

    next_tokens = sampler(logits, temperatures)
    return next_tokens
```

### Context 对象模式

三个框架都用 **全局 / 线程局部上下文** 把 batch 元数据传给注意力层，避免在每一层模型里层层传参：

```python
# nano-vllm：模块级 context
class _Context:
    is_prefill: bool
    slot_mapping: Tensor
    block_tables: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    kvcache: Tensor

context = _Context()  # 全局单例

# mini-sglang：带上下文管理器的 Context 类
class Context:
    def forward_batch(self, batch):
        """上下文管理器：设置当前 batch 状态"""
        self._current_batch = batch
        yield
        self._current_batch = None
```

## CUDA Graph 捕获与回放

### 为何使用 CUDA Graph？

Decode batch 输入很小（每请求 1 个词元），但内核启动仍承受同等 Python 开销。CUDA Graph 预先录制整条执行序列，再 **一次 GPU 调用** 回放。

### 捕获流程（nano-vllm）

```python
def capture_cudagraph(max_batch_size):
    # 为固定 batch 大小捕获：1, 2, 4, 8, 16, 32, ...
    batch_sizes = [1, 2, 4, 8] + list(range(16, max_batch_size+1, 16))

    for bs in reversed(batch_sizes):  # 先大后小，便于共享内存
        # 固定形状的输入缓冲区
        input_ids = torch.zeros(bs, device='cuda')
        positions = torch.zeros(bs, device='cuda')
        # ... 其他固定缓冲区

        # Warmup（CUDA 要求）
        model(input_ids, positions)

        # 捕获
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):
            output = model(input_ids, positions)

        captured_graphs[bs] = (graph, input_ids, positions, output)
```

### 回放流程

```python
def replay_cudagraph(batch):
    # 取不小于实际 batch 的最小已捕获尺寸
    bs = next(s for s in captured_sizes if s >= len(batch))

    graph, input_buf, pos_buf, output_buf = captured_graphs[bs]

    # 把真实数据拷入捕获时的缓冲区
    input_buf[:len(batch)].copy_(batch.input_ids)
    pos_buf[:len(batch)].copy_(batch.positions)

    # 回放图（单次 GPU 调用）
    graph.replay()

    # 从捕获的输出缓冲区读取结果
    return output_buf[:len(batch)]
```

### 内存池共享（mini-sglang）

```python
# 使用共享内存池，使不同 batch 尺寸的图复用 GPU 内存
graph_pool = torch.cuda.graph_pool_handle()
for bs in batch_sizes:
    with torch.cuda.graph(graph, pool=graph_pool):
        ...
```

### CUDA Graph 的限制

- **张量形状固定**：输入/输出形状须与捕获时一致  
- **无输入相关的动态控制流**：不能存在依赖输入值的 if/else  
- **需要填充**：实际 batch 须 pad 到某个已捕获的尺寸  
- **一般不用于 prefill**：Prefill 长度多变，CUDA Graph 通常只用于 decode

## Model Runner 中的张量并行

### 多进程协同（nano-vllm）

```python
# Rank 0（主进程）:
def run(seqs, is_prefill):
    inputs = prepare(seqs, is_prefill)

    # 通过 SharedMemory 通知其他 rank
    shared_mem.write(inputs.serialized())
    event.set()  # 唤醒其他 rank

    # 在 rank 0 上执行
    output = model(inputs)
    return output

# Rank 1-N（子进程）:
def worker_loop():
    while True:
        event.wait()  # 等待 rank 0 信号
        inputs = shared_mem.read()
        model(inputs)  # 丢弃输出；all-reduce 已在内核路径中完成
```

### mini-sglang 做法

使用带 NCCL 后端的 `torch.distributed`，由 Engine 负责初始化：

```python
def init_distributed(tp_size, rank):
    torch.distributed.init_process_group(backend='nccl')
    # 或使用自定义 PyNCCL 封装以降低开销
```

## 权重加载

### 流式分片加载（mini-sglang）

为降低 CPU 内存占用：

```python
def load_weights(model, model_path, tp_rank, tp_size):
    for shard_file in sorted(safetensors_files):
        with safe_open(shard_file) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                param = model.get_parameter(name)
                # 用 param.weight_loader 按当前 TP rank 切分
                param.weight_loader(param, tensor, tp_rank, tp_size)
```

### Weight Loader 模式（nano-vllm）

每个参数自带「如何按 TP 切分」的逻辑：

```python
class ColumnParallelLinear:
    def __init__(self, ...):
        self.weight = Parameter(...)
        self.weight.weight_loader = self._column_shard_loader

    def _column_shard_loader(self, param, loaded_weight, tp_rank, tp_size):
        shard_size = loaded_weight.shape[0] // tp_size
        param.data.copy_(loaded_weight[tp_rank*shard_size:(tp_rank+1)*shard_size])
```

## 设计模板

最小 Model Runner 需要：

1. **输入准备**：把已调度的序列变成张量
2. **上下文传递**：把 batch 元数据交给注意力层
3. **前向执行**：跑模型
4. **采样**：logits → 词元

可选增强：

- CUDA Graph 捕获/回放（decode 明显加速，但实现更复杂）  
- GPU 执行与 CPU 准备重叠  
- 多 rank 间用共享内存传递元数据

