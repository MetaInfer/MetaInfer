# Implementation Patterns

## Common Patterns Across LLM Inference Frameworks

These patterns appear consistently in nano-vllm, nano-sglang, and mini-sglang. They represent proven approaches for structuring inference code.

### 1. Global Context / Thread-Local State

**Problem**: Batch metadata (slot mappings, block tables, sequence lengths) must be accessible deep in the model's attention layers, but threading it through every layer's forward() signature is verbose and fragile.

**Solution**: Use a module-level or thread-local context object.

```python
# nano-vllm approach: module-level singleton
class _Context:
    is_prefill: bool
    slot_mapping: Tensor
    block_tables: Tensor
    kvcache: Tensor

context = _Context()

# Set before forward pass:
context.is_prefill = True
context.slot_mapping = computed_mapping

# Access in attention layer:
class Attention(nn.Module):
    def forward(self, q, k, v):
        slot_mapping = context.slot_mapping  # Access global state
```

```python
# mini-sglang approach: context manager
class Context:
    @contextmanager
    def forward_batch(self, batch):
        self._batch = batch
        try:
            yield
        finally:
            self._batch = None
```

**Rule**: Set context before the model forward pass, clear it after. Never leave stale state.

### 2. Merged Linear Projections

**Problem**: Q, K, V are three separate matrix multiplications with the same input, as are gate and up projections.

**Solution**: Merge into a single GEMM and split the output.

```python
# Instead of:
q = self.q_proj(x)  # [B, H] → [B, Q_dim]
k = self.k_proj(x)  # [B, H] → [B, K_dim]
v = self.v_proj(x)  # [B, H] → [B, V_dim]

# Use:
qkv = self.qkv_proj(x)  # [B, H] → [B, Q_dim + K_dim + V_dim]
q, k, v = qkv.split([Q_dim, K_dim, V_dim], dim=-1)
```

**Benefit**: One large GEMM is more efficient than three small GEMMs (better GPU utilization).

### 3. Fused Residual + Normalization

**Problem**: Residual addition and layer normalization are separate operations that read/write the same tensor.

**Solution**: Fuse them into a single kernel that avoids materializing the intermediate sum.

```python
# Instead of:
hidden = hidden + residual        # Write intermediate
residual = hidden                  # Copy for next residual
hidden = rms_norm(hidden)          # Read intermediate

# Use fused version:
class RMSNormFused:
    def forward(self, x, residual):
        # Single kernel: add residual, save for next layer, normalize
        x = x + residual
        residual = x  # Save before normalization
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps) * self.weight
        return x, residual
```

### 4. Weight Loader Attached to Parameters

**Problem**: TP-aware weight loading requires knowing how to shard each parameter, but the loading logic is separate from the parameter definition.

**Solution**: Attach a `weight_loader` callable to each Parameter.

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self):
        self.weight = Parameter(...)
        self.weight.weight_loader = self._load_column_shard

    def _load_column_shard(self, param, loaded_weight, tp_rank, tp_size):
        shard = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        param.data.copy_(shard)

# Generic loader just calls the attached method:
for name, param in model.named_parameters():
    if hasattr(param, 'weight_loader'):
        param.weight_loader(param, checkpoint[name], rank, size)
```

### 5. Two-Path Forward (Prefill + Decode)

**Problem**: Prefill and decode have fundamentally different input shapes and optimal kernels.

**Solution**: Explicit branching at the model runner level, not inside every layer.

```python
# Model Runner decides the path:
if is_prefill:
    inputs = prepare_prefill(sequences)
    context.is_prefill = True
else:
    inputs = prepare_decode(sequences)
    context.is_prefill = False

output = model(inputs)  # Model uses context to choose kernels

# Inside attention layer:
if context.is_prefill:
    return flash_attn_varlen_func(q, k, v, ...)
else:
    return flash_attn_with_kvcache(q, k_cache, v_cache, ...)
```

### 6. Streaming Weight Loading

**Problem**: Loading all weights into CPU memory before transferring to GPU can exceed CPU RAM for large models.

**Solution**: Stream weights file-by-file, shard-by-shard.

```python
for shard_file in safetensors_files:
    with safe_open(shard_file, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)  # Load one tensor at a time
            load_into_model(model, key, tensor)
            del tensor  # Free CPU memory immediately
```

### 7. CUDA Graph Batch Size Padding

**Problem**: CUDA graphs require fixed tensor shapes, but actual batch sizes vary.

**Solution**: Capture graphs for power-of-2 batch sizes and pad actual batches.

```python
captured_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def get_graph_batch_size(actual_size):
    for size in captured_sizes:
        if size >= actual_size:
            return size
    return actual_size  # Fall back to eager if too large

# Pad inputs before replay:
padded_size = get_graph_batch_size(len(batch))
input_buffer[:len(batch)].copy_(actual_inputs)
# Remaining positions already have dummy data from capture
```

### 8. ZMQ for Inter-Process Communication

**Problem**: Multi-process architectures need efficient, non-blocking communication.

**Solution**: Use ZeroMQ PUSH/PULL sockets for fire-and-forget message passing.

```python
# Process A (sender):
sender = zmq.Context().socket(zmq.PUSH)
sender.connect(f"tcp://127.0.0.1:{port}")
sender.send_pyobj(request)

# Process B (receiver):
receiver = zmq.Context().socket(zmq.PULL)
receiver.bind(f"tcp://127.0.0.1:{port}")
request = receiver.recv_pyobj()
```

**Why ZMQ over multiprocessing.Queue**: Higher throughput, supports TCP (cross-machine), doesn't require shared memory setup.

---

## Key Design Principles

1. **Minimize dynamic dispatch**: Hard-code paths for the target model/hardware
2. **Fuse operations**: Reduce memory bandwidth by combining adjacent operations
3. **Pre-allocate everything**: Avoid runtime GPU memory allocation
4. **Separate preparation from execution**: CPU work on one stream, GPU work on another
5. **Keep the hot loop minimal**: The engine step() should be as simple as possible
