# Model Runner - Forward Pass Execution

## Core Responsibility

The Model Runner is the bridge between the scheduler's logical decisions and the GPU's physical execution. It:
1. Converts scheduled batches into GPU tensors
2. Executes the model forward pass
3. Samples next tokens from logits
4. Optionally uses CUDA graphs for decode acceleration

## Input Preparation

### Prefill Input Preparation

For prefill, multiple tokens per request are processed simultaneously:

```python
def prepare_prefill(sequences):
    # Concatenate all prompt tokens into a flat 1D tensor
    input_ids = concat([seq.tokens[cached:total] for seq in sequences])

    # Build cumulative sequence lengths for varlen attention
    cu_seqlens = [0]
    for seq in sequences:
        cu_seqlens.append(cu_seqlens[-1] + seq.num_new_tokens)

    # Map each token to its physical KV cache location
    slot_mapping = []
    for seq in sequences:
        for pos in range(seq.cached, seq.total):
            block_id = seq.block_table[pos // block_size]
            offset = pos % block_size
            slot_mapping.append(block_id * block_size + offset)

    # Block tables for accessing cached prefix KV data
    block_tables = pad([seq.block_table for seq in sequences])

    return input_ids, cu_seqlens, slot_mapping, block_tables
```

### Decode Input Preparation

For decode, exactly one token per request:

```python
def prepare_decode(sequences):
    # One token per sequence (the last generated token)
    input_ids = [seq.tokens[-1] for seq in sequences]

    # Position of each token
    positions = [seq.num_tokens - 1 for seq in sequences]

    # Physical slot for the new KV entry
    slot_mapping = []
    for seq in sequences:
        pos = seq.num_tokens - 1
        block_id = seq.block_table[pos // block_size]
        offset = pos % block_size
        slot_mapping.append(block_id * block_size + offset)

    # Full block tables for attention over all previous tokens
    block_tables = pad([seq.block_table for seq in sequences])
    context_lens = [seq.num_tokens for seq in sequences]

    return input_ids, positions, slot_mapping, block_tables, context_lens
```

## Forward Pass Execution

### Basic Flow
```python
def run(sequences, is_prefill):
    if is_prefill:
        inputs = prepare_prefill(sequences)
    else:
        inputs = prepare_decode(sequences)

    # Set global context for attention layers
    context.is_prefill = is_prefill
    context.slot_mapping = inputs.slot_mapping
    context.block_tables = inputs.block_tables

    # Execute model
    logits = model(inputs.input_ids, inputs.positions)

    # Sample next tokens
    if is_prefill:
        # Only need logits at the last position of each sequence
        last_indices = cumsum(seq_lens) - 1
        logits = logits[last_indices]

    next_tokens = sampler(logits, temperatures)
    return next_tokens
```

### Context Object Pattern
All three frameworks use a global/thread-local context to pass batch metadata to attention layers without threading it through every model layer:

```python
# nano-vllm: module-level context
class _Context:
    is_prefill: bool
    slot_mapping: Tensor
    block_tables: Tensor
    cu_seqlens: Tensor
    max_seqlen: int
    kvcache: Tensor

context = _Context()  # Global singleton

# mini-sglang: Context class with context manager
class Context:
    def forward_batch(self, batch):
        """Context manager that sets active batch state"""
        self._current_batch = batch
        yield
        self._current_batch = None
```

## CUDA Graph Capture and Replay

### Why CUDA Graphs?
Decode batches have very small inputs (1 token per request) but require the same Python overhead for kernel launches. CUDA graphs pre-record the entire execution sequence and replay it in a single GPU call.

### Capture Process (nano-vllm)
```python
def capture_cudagraph(max_batch_size):
    # Capture graphs for fixed batch sizes: 1, 2, 4, 8, 16, 32, ...
    batch_sizes = [1, 2, 4, 8] + list(range(16, max_batch_size+1, 16))

    for bs in reversed(batch_sizes):  # Largest first to share memory
        # Create fixed-size input buffers
        input_ids = torch.zeros(bs, device='cuda')
        positions = torch.zeros(bs, device='cuda')
        # ... other fixed buffers

        # Warmup run (required by CUDA)
        model(input_ids, positions)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):
            output = model(input_ids, positions)

        captured_graphs[bs] = (graph, input_ids, positions, output)
```

### Replay Process
```python
def replay_cudagraph(batch):
    # Find smallest captured size >= actual batch size
    bs = next(s for s in captured_sizes if s >= len(batch))

    graph, input_buf, pos_buf, output_buf = captured_graphs[bs]

    # Copy actual data into captured buffers
    input_buf[:len(batch)].copy_(batch.input_ids)
    pos_buf[:len(batch)].copy_(batch.positions)

    # Replay the graph (single GPU call)
    graph.replay()

    # Read results from captured output buffer
    return output_buf[:len(batch)]
```

### Memory Pool Sharing (mini-sglang)
```python
# Use a shared memory pool so different batch-size graphs reuse GPU memory
graph_pool = torch.cuda.graph_pool_handle()
for bs in batch_sizes:
    with torch.cuda.graph(graph, pool=graph_pool):
        ...
```

### CUDA Graph Limitations
- **Fixed tensor shapes**: All inputs/outputs must have the same shape as during capture
- **No dynamic control flow**: Cannot have if/else that depends on input values
- **Padding required**: Actual batch must be padded to a captured size
- **Not for prefill**: Prefill has variable-length inputs, so CUDA graphs are typically only used for decode

## Tensor Parallelism in Model Runner

### Multi-Process Coordination (nano-vllm)
```python
# Rank 0 (main process):
def run(seqs, is_prefill):
    inputs = prepare(seqs, is_prefill)

    # Signal other ranks via SharedMemory
    shared_mem.write(inputs.serialized())
    event.set()  # Wake up other ranks

    # Execute on rank 0
    output = model(inputs)
    return output

# Rank 1-N (sub-processes):
def worker_loop():
    while True:
        event.wait()  # Wait for signal from rank 0
        inputs = shared_mem.read()
        model(inputs)  # Output discarded; all-reduce already happened
```

### Mini-sglang Approach
Uses `torch.distributed` with NCCL backend. The Engine handles initialization:
```python
def init_distributed(tp_size, rank):
    torch.distributed.init_process_group(backend='nccl')
    # Or use custom PyNCCL wrapper for lower overhead
```

## Weight Loading

### Streaming Shard Loading (mini-sglang)
To minimize CPU memory usage:
```python
def load_weights(model, model_path, tp_rank, tp_size):
    for shard_file in sorted(safetensors_files):
        with safe_open(shard_file) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                param = model.get_parameter(name)
                # Use param's weight_loader to shard for this TP rank
                param.weight_loader(param, tensor, tp_rank, tp_size)
```

### Weight Loader Pattern (nano-vllm)
Each parameter knows how to shard itself:
```python
class ColumnParallelLinear:
    def __init__(self, ...):
        self.weight = Parameter(...)
        self.weight.weight_loader = self._column_shard_loader

    def _column_shard_loader(self, param, loaded_weight, tp_rank, tp_size):
        shard_size = loaded_weight.shape[0] // tp_size
        param.data.copy_(loaded_weight[tp_rank*shard_size:(tp_rank+1)*shard_size])
```

## Design Template

A minimal Model Runner needs:
1. **Input preparation**: Convert scheduled sequences into tensors
2. **Context propagation**: Pass batch metadata to attention layers
3. **Forward execution**: Run the model
4. **Sampling**: Convert logits to tokens

Optional enhancements:
- CUDA graph capture/replay (significant speedup for decode, but adds complexity)
- Overlap between GPU execution and CPU preparation
- Shared memory for multi-rank metadata passing
