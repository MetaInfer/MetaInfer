# Tensor Parallelism

## Core Concept

Tensor Parallelism (TP) splits individual layers across multiple GPUs, enabling inference of models that don't fit on a single GPU. It is the most common parallelism strategy for LLM inference because it minimizes latency (all GPUs work on every token).

## Sharding Strategy

### Column Parallel Linear
Splits the **output dimension** across GPUs. Each GPU computes a portion of the output:

```
Full weight: [H_in, H_out]
Rank 0: W[H_in, 0:H_out/N]
Rank 1: W[H_in, H_out/N:2*H_out/N]
...

Input: [B, H_in]  (replicated on all ranks)
Output: [B, H_out/N]  (partial, different on each rank)
```

**No communication needed after the forward pass** — the partial outputs are consumed by the next layer.

Used for: `QKV projection`, `Gate+Up projection` in MLP

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

### Row Parallel Linear
Splits the **input dimension** across GPUs. Each GPU has a portion of the input:

```
Full weight: [H_in, H_out]
Rank 0: W[0:H_in/N, H_out]
Rank 1: W[H_in/N:2*H_in/N, H_out]
...

Input: [B, H_in/N]  (partial, from column parallel output)
Output: [B, H_out]  (partial sum, needs all-reduce)
```

**Requires all-reduce** after the forward pass to sum partial results.

Used for: `O projection`, `Down projection` in MLP

```python
class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_rank, tp_size):
        self.local_input_size = input_size // tp_size
        self.weight = Parameter(torch.empty(output_size, self.local_input_size))

    def forward(self, x):
        y = F.linear(x, self.weight)
        # All-reduce: sum partial results from all ranks
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    def weight_loader(self, param, loaded_weight, tp_rank, tp_size):
        shard_size = loaded_weight.shape[1] // tp_size
        start = tp_rank * shard_size
        param.data.copy_(loaded_weight[:, start:start + shard_size])
```

### QKV Parallel Linear (Special Case)
Q, K, V have different head counts (especially with GQA), requiring careful sharding:

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

## Transformer Block TP Layout

```
                    Replicated           Column Parallel       Row Parallel
                    ──────────           ──────────────        ────────────
Input hidden        [B, H]               
    ↓
RMSNorm             [B, H]              (replicated)
    ↓
QKV Proj                                 [B, H] → [B, (Q+K+V)/N]
    ↓
Attention                                (local computation, no comm)
    ↓
O Proj                                                         [B, H/N] → [B, H] + all_reduce
    ↓
Residual            [B, H]              (replicated after all_reduce)
    ↓
RMSNorm             [B, H]              (replicated)
    ↓
Gate+Up Proj                             [B, H] → [B, 2*FFN/N]
    ↓
SiLU + Mul                              (local computation)
    ↓
Down Proj                                                      [B, FFN/N] → [B, H] + all_reduce
    ↓
Residual            [B, H]              (replicated after all_reduce)
```

**Communication**: Exactly 2 all-reduce operations per transformer layer (one after O proj, one after Down proj).

## Embedding and LM Head TP

### Parallel Embedding
```python
class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, tp_rank, tp_size):
        self.local_vocab_size = vocab_size // tp_size
        self.vocab_start = tp_rank * self.local_vocab_size
        self.weight = Parameter(torch.empty(self.local_vocab_size, hidden_size))

    def forward(self, input_ids):
        # Mask out-of-range tokens for this rank
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_start + self.local_vocab_size)
        local_ids = input_ids - self.vocab_start
        local_ids[~mask] = 0
        output = F.embedding(local_ids, self.weight)
        output[~mask] = 0
        # All-reduce to combine results
        dist.all_reduce(output)
        return output
```

### LM Head (reuses embedding weight)
Many models tie embedding and LM head weights. With TP:
```python
class ParallelLMHead(nn.Module):
    def forward(self, hidden):
        # Each rank computes logits for its vocab shard
        local_logits = F.linear(hidden, self.weight)  # [B, local_vocab_size]
        # All-gather to get full vocabulary logits
        all_logits = all_gather(local_logits)  # [B, vocab_size]
        return all_logits
```

## NCCL Initialization

### Standard PyTorch Distributed
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

### Custom NCCL Wrapper (mini-sglang)
For lower overhead:
```python
class PyNCCLCommunicator:
    def __init__(self, rank, size, uid):
        # Directly initialize NCCL communicator via FFI
        self.comm = nccl_module.create_comm(rank, size, uid)

    def all_reduce(self, tensor, stream=None):
        self.comm.all_reduce(tensor.data_ptr(), tensor.numel(), ...)
```

### Multi-Process Coordination (nano-vllm)
Uses SharedMemory + Events for metadata passing between ranks:
```python
# Rank 0: Prepare inputs, write to shared memory, signal other ranks
shared_mem = SharedMemory(create=True, size=buffer_size)
event = multiprocessing.Event()

# Rank 0: Signal ready
shared_mem.buf[:] = serialized_inputs
event.set()

# Rank 1-N: Wait for signal, read inputs
event.wait()
inputs = deserialize(shared_mem.buf)
```

## Design Template

For implementing TP:
1. **Process group**: Initialize NCCL process group
2. **ColumnParallelLinear**: Shard output dim, no post-communication
3. **RowParallelLinear**: Shard input dim, all-reduce after forward
4. **QKVParallelLinear**: Handle asymmetric Q/KV head counts
5. **Weight loader**: Each parameter knows how to extract its TP shard
6. **Embedding/LMHead**: Split vocabulary across ranks

**Communication cost**: 2 all-reduce per layer × `hidden_size` × `batch_size` × `dtype_bytes`

**When to use TP**: Model doesn't fit on one GPU, or latency requirements demand multi-GPU (each GPU does less work per token)
