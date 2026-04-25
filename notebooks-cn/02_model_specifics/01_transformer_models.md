# Transformer 模型：通用稠密 Transformer 模式

## 核心架构

LLM 推理中的稠密 Transformer 模型在高层结构上是一致的：

```
输入 Token ID
    ↓
[Embedding 层]  → token_ids → hidden_states
    ↓
[Transformer 块 × N]
    │
    ├── [RMSNorm] → normalized_hidden
    ├── [Attention] → attn_output
    ├── [残差加] → hidden + attn_output
    ├── [RMSNorm] → normalized_hidden
    ├── [MLP / MoE] → mlp_output
    └── [残差加] → hidden + mlp_output
    ↓
[RMSNorm]
    ↓
[LM Head]  → hidden_states → logits (vocab_size)
    ↓
[Sampler]  → logits → next_token_id
```

## 模型定义模式

### 标准模式（nano-vllm 风格）
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

### 上下文驱动模式（mini-sglang 风格）
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
        h = self.layers(h)      # OPList 链式前向
        h = self.norm(h)
        h = self.lm_head(h)
        return h
```

## Attention 子模块

### 标准 MHA / GQA Attention
```python
class Attention(nn.Module):
    def __init__(self, config):
        # 合并 QKV 投影以提升效率
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,  # GQA：更少的 KV 头
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

        # 应用 RoPE
        q, k = self.rotary_emb(q, k, positions)

        # 带 KV cache 的 attention（与具体 backend 相关）
        attn_output = attention_backend(q, k, v, kv_cache, ...)

        # 输出投影（TP 时包含 all-reduce）
        return self.o_proj(attn_output)
```

### 分组查询注意力（GQA）
GQA 中 KV 头数少于 Q 头。比值 `num_q_heads / num_kv_heads` 决定「一组」里有多少个 Q 头共享一个 KV 头：

- **MHA**：比值为 1（每个 Q 头有独立 KV 头）
- **GQA**：比值 > 1（多个 Q 头共享一个 KV 头）
- **MQA**：比值为 `num_q_heads`（所有 Q 头共享一个 KV 头）

Attention 内核通过重复扩展 KV 头，或使用支持分组的实现来处理。

## MLP 子模块

### 门控 MLP（SwiGLU）
现代 LLM 的 FFN 通常采用门控结构：
```python
class GatedMLP(nn.Module):
    def __init__(self, config):
        # 合并 gate 与 up 投影
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 2 * intermediate_size
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size
        )
        self.act = SiluAndMul()  # SiLU(gate) * up

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act(gate_up)         # 切分、SiLU(gate) * up
        x = self.down_proj(x)         # 含 all-reduce
        return x
```

### 激活：SiLU 与逐元素乘
```python
class SiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
```

## 归一化

### 带融合残差的 RMSNorm
```python
class RMSNorm(nn.Module):
    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        # RMS 归一化
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight
        return x, residual
```

「融合：残差加 + 归一化」对性能很重要：避免再物化一整个中间张量。

## 旋转位置编码（RoPE）

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=8192, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k, positions):
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        cos = freqs.cos()
        sin = freqs.sin()

        # 应用旋转
        q_rotated = rotate_half(q, cos, sin)
        k_rotated = rotate_half(k, cos, sin)
        return q_rotated, k_rotated

def rotate_half(x, cos, sin):
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

## 权重加载模式

### packed_modules_mapping
当 HuggingFace 模型是分开的 Q、K、V 矩阵，而推理框架使用合并的 QKV 层时：

```python
# 映射：HF 权重名 → (合并层名, 位置/槽位)
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),   # 写入合并权重的 Q 段
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),  # 合并权重的第一半
    "up_proj": ("gate_up_proj", 1),   # 合并权重的第二半
}
```

### 流式权重加载
```python
def load_weights(model, path, tp_rank, tp_size):
    for shard_file in safetensors_files:
        with safe_open(shard_file) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                param = find_parameter(model, name)  # 可能经 packed_modules_mapping 重映射
                if hasattr(param, 'weight_loader'):
                    param.weight_loader(param, tensor, tp_rank, tp_size)
                else:
                    param.data.copy_(tensor)
```

## 模型配置

影响推理的 HuggingFace 配置关键项：
```python
@dataclass
class ModelConfig:
    hidden_size: int           # 如 4096
    num_hidden_layers: int     # 如 32
    num_attention_heads: int   # Q 头数，如 32
    num_key_value_heads: int   # GQA 的 KV 头，如 8
    intermediate_size: int     # MLP 隐层，如 11008
    vocab_size: int            # 如 32000
    max_position_embeddings: int  # 如 4096
    rope_theta: float          # RoPE 基频，如 10000.0
    head_dim: int              # hidden_size // num_attention_heads
    rms_norm_eps: float        # 如 1e-6
```

## 常见模型家族与差异

| 特性 | Llama/Qwen2 | Qwen3 | Mistral | Mixtral |
|------|-------------|-------|---------|---------|
| Attention | GQA | GQA + 滑窗 | GQA + 滑窗 | GQA |
| MLP | SwiGLU | SwiGLU | SwiGLU | MoE（每专家 SwiGLU） |
| Norm | RMSNorm | RMSNorm（attn 前后 pre+post） | RMSNorm | RMSNorm |
| RoPE | 标准 | 标准 | 标准 | 标准 |
| Bias | 无 | 无（QKV 无 bias） | 无 | 无 |
| QKV 合并 | 是 | 是 | 是 | 是 |
