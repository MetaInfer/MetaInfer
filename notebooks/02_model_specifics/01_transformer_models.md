# Transformer Models - General Dense Transformer Patterns

## Core Architecture

All dense transformer models in LLM inference follow the same high-level structure:

```
Input Token IDs
    ↓
[Embedding Layer]  → token_ids → hidden_states
    ↓
[Transformer Block × N]
    │
    ├── [RMSNorm] → normalized_hidden
    ├── [Attention] → attn_output
    ├── [Residual Add] → hidden + attn_output
    ├── [RMSNorm] → normalized_hidden
    ├── [MLP / MoE] → mlp_output
    └── [Residual Add] → hidden + mlp_output
    ↓
[RMSNorm]
    ↓
[LM Head]  → hidden_states → logits (vocab_size)
    ↓
[Sampler]  → logits → next_token_id
```

## Model Definition Pattern

### Standard Pattern (nano-vllm style)
```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelEmbedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        hidden = self.model(input_ids, positions)
        logits = self.lm_head(hidden)
        return logits


class Qwen3Model(nn.Module):
    def __init__(self, config):
        self.embed_tokens = ParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids, positions):
        hidden = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions)
        hidden = self.norm(hidden, residual)
        return hidden
```

### Context-Driven Pattern (mini-sglang style)
```python
class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([LlamaDecoderLayer(config) for _ in range(num_layers)])
        self.norm = RMSNormFused(config.hidden_size)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)

    def forward(self, batch):
        ctx = get_global_ctx()
        h = self.embed(ctx.input_ids)
        h = self.layers(h)      # OPList chains forward calls
        h = self.norm(h)
        h = self.lm_head(h)
        return h
```

## Attention Block

### Standard MHA/GQA Attention
```python
class Attention(nn.Module):
    def __init__(self, config):
        # Merged QKV projection for efficiency
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,  # GQA: fewer KV heads
            head_dim=config.head_dim,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size
        )
        self.rotary_emb = RotaryEmbedding(head_dim, max_position, base=rope_theta)

    def forward(self, hidden, positions):
        qkv = self.qkv_proj(hidden)
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

        q = q.view(-1, num_q_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)

        # Apply RoPE
        q, k = self.rotary_emb(q, k, positions)

        # Attention with KV cache (backend-specific)
        attn_output = attention_backend(q, k, v, kv_cache, ...)

        # Output projection (includes all-reduce for TP)
        return self.o_proj(attn_output)
```

### Grouped Query Attention (GQA)
GQA uses fewer KV heads than Q heads. The ratio `num_q_heads / num_kv_heads` determines the group size:
- **MHA**: ratio = 1 (every Q head has its own KV head)
- **GQA**: ratio > 1 (multiple Q heads share one KV head)
- **MQA**: ratio = num_q_heads (all Q heads share one KV head)

The attention kernel handles this by repeating KV heads or using optimized group-aware implementations.

## MLP Block

### Gated MLP (SwiGLU)
The standard MLP in modern LLMs uses a gated architecture:
```python
class GatedMLP(nn.Module):
    def __init__(self, config):
        # Merged gate + up projection
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 2 * intermediate_size
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size
        )
        self.act = SiluAndMul()  # SiLU(gate) * up

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act(gate_up)         # Split, SiLU(gate) * up
        x = self.down_proj(x)         # Includes all-reduce
        return x
```

### Activation: SiLU and Mul
```python
class SiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
```

## Normalization

### RMSNorm with Fused Residual
```python
class RMSNorm(nn.Module):
    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        # RMS normalization
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight
        return x, residual
```

The "fused residual add + norm" pattern is critical for performance: it avoids materializing an intermediate tensor.

## Rotary Positional Embedding (RoPE)

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=8192, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k, positions):
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        cos = freqs.cos()
        sin = freqs.sin()

        # Apply rotation
        q_rotated = rotate_half(q, cos, sin)
        k_rotated = rotate_half(k, cos, sin)
        return q_rotated, k_rotated

def rotate_half(x, cos, sin):
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

## Weight Loading Pattern

### packed_modules_mapping
When HuggingFace models have separate Q, K, V weight matrices but the inference framework uses a merged QKV layer:

```python
# Mapping: HF weight name → (merged layer name, position)
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),   # → stored in q portion of merged weight
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),  # → first half of merged weight
    "up_proj": ("gate_up_proj", 1),    # → second half of merged weight
}
```

### Streaming Weight Loader
```python
def load_weights(model, path, tp_rank, tp_size):
    for shard_file in safetensors_files:
        with safe_open(shard_file) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                param = find_parameter(model, name)  # May remap via packed_modules_mapping
                if hasattr(param, 'weight_loader'):
                    param.weight_loader(param, tensor, tp_rank, tp_size)
                else:
                    param.data.copy_(tensor)
```

## Model Configuration

Key parameters from HuggingFace config that affect inference:
```python
@dataclass
class ModelConfig:
    hidden_size: int           # e.g., 4096
    num_hidden_layers: int     # e.g., 32
    num_attention_heads: int   # Q heads, e.g., 32
    num_key_value_heads: int   # KV heads for GQA, e.g., 8
    intermediate_size: int     # MLP hidden dim, e.g., 11008
    vocab_size: int            # e.g., 32000
    max_position_embeddings: int  # e.g., 4096
    rope_theta: float          # RoPE base frequency, e.g., 10000.0
    head_dim: int              # hidden_size // num_attention_heads
    rms_norm_eps: float        # e.g., 1e-6
```

## Common Model Families and Their Differences

| Feature | Llama/Qwen2 | Qwen3 | Mistral | Mixtral |
|---------|-------------|-------|---------|---------|
| Attention | GQA | GQA + sliding window | GQA + sliding window | GQA |
| MLP | SwiGLU | SwiGLU | SwiGLU | MoE (SwiGLU per expert) |
| Norm | RMSNorm | RMSNorm (pre+post attn) | RMSNorm | RMSNorm |
| RoPE | Standard | Standard | Standard | Standard |
| Bias | No | No (QKV has no bias) | No | No |
| QKV merge | Yes | Yes | Yes | Yes |
