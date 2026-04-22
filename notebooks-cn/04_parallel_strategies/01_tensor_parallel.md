# 张量并行（Tensor Parallelism）

## 核心概念

**张量并行（TP）**把单层计算拆到多张 GPU 上，使**单卡放不下的模型**仍能推理。它是 LLM 推理里**最常见**的并行方式，因为能压低延迟（每张卡都参与**每一个 token** 的计算）。

下文路径均相对于仓库中的 **`meta-infer/ref_projects/`** 目录。

## 参考工程中的 TP 源码位置

以下列出三个精简参考实现中与**张量并行**直接相关的 `.py` 文件及**大致行号**（便于打开即跳转到类/函数；合并行号表示该文件内连续相关片段）。

### nano-vllm（`nano-vllm/`）

| 主题 | 文件 | 行号（约） |
|------|------|------------|
| `LinearBase`、`ReplicatedLinear` | `nanovllm/layers/linear.py` | 12–51 |
| `ColumnParallelLinear`（列切输出维） | 同上 | 54–73 |
| `MergedColumnParallelLinear`（多段输出合并列切） | 同上 | 76–93 |
| `QKVParallelLinear`（Q/K/V 分片加载） | 同上 | 96–128 |
| `RowParallelLinear`（行切 + `dist.all_reduce`） | 同上 | 131–156 |
| `VocabParallelEmbedding`、`ParallelLMHead`（词表切分 / gather logits） | `nanovllm/layers/embed_head.py` | 9–66 |
| 多进程 `dist.init_process_group`、`SharedMemory` 协同 worker | `nanovllm/engine/model_runner.py` | 22–59（含 NCCL、barrier、`tensor_parallel_size`） |

### nano-sglang（`nano-sglang/python/sglang/srt/`）

| 主题 | 文件 | 行号（约） |
|------|------|------------|
| `initialize_model_parallel`、TP/PP 进程组、`tensor_model_parallel_all_reduce` / `all_gather` | `parallel_utils/parallel_state.py` | 19–84（初始化组），95–123（rank/world_size），188–227（集合通信封装） |
| `ColumnParallelLinear`、`MergedColumnParallelLinear` | `layers/linear.py` | 138–232，235–333 |
| `QKVParallelLinear`（含 GQA/MQA 下 KV 复制等逻辑） | 同上 | 336–470 |
| `RowParallelLinear`（可选 `split_tensor_along_last_dim` + all_reduce） | 同上 | 473–595 |
| `VocabParallelEmbedding`、`ParallelLMHead` | `layers/vocab_parallel_embedding.py` | 35–109，112–150 |
| `torch.distributed.init_process_group` 后调用 `initialize_model_parallel` | `managers/router/model_runner.py` | 225–237 |

### mini-sglang（`mini-sglang/python/minisgl/`）

| 主题 | 文件 | 行号（约） |
|------|------|------------|
| `_LinearTPImpl` 基类；`LinearColParallelMerged`、`LinearQKVMerged`；`LinearOProj` / `LinearRowParallel`（`all_reduce`） | `layers/linear.py` | 13–127 |
| `VocabParallelEmbedding`（`indexing` + `all_reduce`） | `layers/embedding.py` | 14–42 |
| `ParallelLMHead`（`all_gather` 拼 logits 等） | 同上 | 45–110 |
| `DistributedCommunicator`、`TorchDistributedImpl` / `PyNCCLDistributedImpl`、`enable_pynccl_distributed` | `distributed/impl.py` | 14–97 |
| `Engine._init_communication`（`nccl`/`gloo` + 可选 PyNCCL） | `engine/engine.py` | 112–137 |
| `MoELayer` 末尾对隐藏状态 `all_reduce`（专家并行与 TP 同进程组下的归约） | `layers/moe.py` | 9–58 |

> **说明**：行号随上游仓库变更可能漂移；若不一致，以文件内类名 **`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` / `VocabParallelEmbedding`** 等为锚搜索即可。

## 切分策略

### 列并行线性层（Column Parallel Linear）

按**输出维度**切分权重。每张卡只算输出的一部分：

```
完整权重: [H_in, H_out]
Rank 0: W[H_in, 0:H_out/N]
Rank 1: W[H_in, H_out/N:2*H_out/N]
...

输入: [B, H_in]  （各 rank 上相同，复制）
输出: [B, H_out/N]  （部分结果，各 rank 不同）
```

**前向结束后不需要通信**——部分输出由下一层直接消费。

常用于：`QKV 投影`、MLP 中的 `Gate+Up 投影`

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_rank, tp_size):
        self.local_output_size = output_size // tp_size
        self.weight = Parameter(torch.empty(self.local_output_size, input_size))

    def forward(self, x):
        return F.linear(x, self.weight)

    def weight_loader(self, param, loaded_weight, tp_rank, tp_size):
        shard_size = loaded_weight.shape[0] // tp_size
        start = tp_rank * shard_size
        param.data.copy_(loaded_weight[start:start + shard_size])
```

### 行并行线性层（Row Parallel Linear）

按**输入维度**切分权重。每张卡只持有输入的一部分：

```
完整权重: [H_in, H_out]
Rank 0: W[0:H_in/N, H_out]
Rank 1: W[H_in/N:2*H_in/N, H_out]
...

输入: [B, H_in/N]  （部分，来自上一列并行层的输出）
输出: [B, H_out]  （部分和，需要 all-reduce）
```

**前向结束后需要 all-reduce**，对 partial 结果求和。

常用于：`O 投影`、MLP 的 `Down 投影`

```python
class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_rank, tp_size):
        self.local_input_size = input_size // tp_size
        self.weight = Parameter(torch.empty(output_size, self.local_input_size))

    def forward(self, x):
        y = F.linear(x, self.weight)
        # All-reduce：对各 rank 的部分和求和
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    def weight_loader(self, param, loaded_weight, tp_rank, tp_size):
        shard_size = loaded_weight.shape[1] // tp_size
        start = tp_rank * shard_size
        param.data.copy_(loaded_weight[:, start:start + shard_size])
```

### QKV 并行线性层（特例）

Q、K、V 的 head 数不同（尤其 GQA 时），切分需单独处理：

```python
class QKVParallelLinear(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, head_dim, tp_size):
        self.local_q_heads = num_q_heads // tp_size
        self.local_kv_heads = num_kv_heads // tp_size
        self.q_size = self.local_q_heads * head_dim
        self.kv_size = self.local_kv_heads * head_dim
        total_size = self.q_size + 2 * self.kv_size
        self.weight = Parameter(torch.empty(total_size, hidden_size))

    def forward(self, x):
        output = F.linear(x, self.weight)
        q, k, v = output.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v
```

## Transformer 块的 TP 排布

```
                    复制                 列并行                 行并行
                    ──────               ────────               ────────
输入隐状态          [B, H]               
    ↓
RMSNorm             [B, H]              （复制）
    ↓
QKV 投影                                 [B, H] → [B, (Q+K+V)/N]
    ↓
注意力                                  （本地计算，无通信）
    ↓
O 投影                                                         [B, H/N] → [B, H] + all_reduce
    ↓
残差                [B, H]              （all_reduce 后复制）
    ↓
RMSNorm             [B, H]              （复制）
    ↓
Gate+Up 投影                             [B, H] → [B, 2*FFN/N]
    ↓
SiLU + Mul                              （本地计算）
    ↓
Down 投影                                                      [B, FFN/N] → [B, H] + all_reduce
    ↓
残差                [B, H]              （all_reduce 后复制）
```

**通信**：每个 Transformer 层恰好 **2 次 all-reduce**（一次在 O 投影后，一次在 Down 投影后）。

## Embedding 与 LM Head 的 TP

### 并行 Embedding

```python
class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, tp_rank, tp_size):
        self.local_vocab_size = vocab_size // tp_size
        self.vocab_start = tp_rank * self.local_vocab_size
        self.weight = Parameter(torch.empty(self.local_vocab_size, hidden_size))

    def forward(self, input_ids):
        # 对本 rank 词表范围外的 token 置掩码
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_start + self.local_vocab_size)
        local_ids = input_ids - self.vocab_start
        local_ids[~mask] = 0
        output = F.embedding(local_ids, self.weight)
        output[~mask] = 0
        # All-reduce 合并各 rank 结果
        dist.all_reduce(output)
        return output
```

### LM Head（常与 embedding 权重绑定）

许多模型把 embedding 与 LM head 权重绑在一起。TP 下：

```python
class ParallelLMHead(nn.Module):
    def forward(self, hidden):
        # 每 rank 只算本词表分片的 logits
        local_logits = F.linear(hidden, self.weight)  # [B, local_vocab_size]
        # All-gather 得到完整词表 logits
        all_logits = all_gather(local_logits)  # [B, vocab_size]
        return all_logits
```

## NCCL 初始化

### 标准 PyTorch 分布式

```python
def init_tp(tp_size, rank):
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=tp_size,
        rank=rank,
    )
    tp_group = torch.distributed.new_group(list(range(tp_size)))
    return tp_group
```

### 自定义 NCCL 封装（mini-sglang）

降低开销时：

```python
class PyNCCLCommunicator:
    def __init__(self, rank, size, uid):
        # 通过 FFI 直接创建 NCCL communicator
        self.comm = nccl_module.create_comm(rank, size, uid)

    def all_reduce(self, tensor, stream=None):
        self.comm.all_reduce(tensor.data_ptr(), tensor.numel(), ...)
```

### 多进程协同（nano-vllm）

用 SharedMemory + Event 在 rank 间传递元数据：

```python
# Rank 0：准备输入，写入共享内存，通知其他 rank
shared_mem = SharedMemory(create=True, size=buffer_size)
event = multiprocessing.Event()

# Rank 0：发出就绪信号
shared_mem.buf[:] = serialized_inputs
event.set()

# Rank 1-N：等待信号，读取输入
event.wait()
inputs = deserialize(shared_mem.buf)
```

## 设计模板

实现 TP 时可按：

1. **进程组**：初始化 NCCL 进程组  
2. **ColumnParallelLinear**：切输出维，前向后**无**后通信  
3. **RowParallelLinear**：切输入维，前向后 **all-reduce**  
4. **QKVParallelLinear**：处理 Q/KV head 数不对称  
5. **权重加载器**：每个参数知道如何截取本 TP rank 的分片  
6. **Embedding / LMHead**：按 rank 切分词表  

**通信量**：每层 **2 次 all-reduce** × `hidden_size` × `batch_size` × `dtype` 字节数  

**何时用 TP**：单卡放不下模型，或延迟要求多卡分担（每 token 每卡工作量更小）

---

*译文对应英文原文：`notebooks/04_parallel_strategies/01_tensor_parallel.md`。*
