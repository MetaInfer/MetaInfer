# Transformer模型实现

## 1. 模型架构概述

标准的Decoder-only Transformer模型结构：

```
Input Tokens
     │
     ▼
┌─────────────────┐
│ Embedding Layer │  [batch, seq_len] → [batch, seq_len, hidden_size]
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Block × N                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  Attention Layer                       │  │
│  │  Input → RMSNorm → QKV Proj → Attention → O Proj      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    MLP Layer                           │  │
│  │  Input → RMSNorm → Gate+Up Proj → SiLU → Down Proj    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Final Norm    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    LM Head      │  [batch, seq_len, hidden_size] → [batch, seq_len, vocab_size]
└─────────────────┘
```

## 2. 模型实现模板

### 2.1 模型类结构

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_id)
            for layer_id in range(config.num_hidden_layers)
        ])
        
        # Final Layer Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Transformer Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        
        # Final Norm
        hidden_states = self.norm(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        return logits
```

### 2.2 Decoder Layer

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        
        # Attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, layer_id)
        
        # MLP
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)
    
    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

## 3. Attention实现

### 3.1 Attention层结构

```python
class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = hidden_size // num_heads
        
        # QKV Projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
        )
        
        # QK Normalization (可选)
        self.q_norm = RMSNorm(head_dim) if hasattr(config, "use_qk_norm") else None
        self.k_norm = RMSNorm(head_dim) if hasattr(config, "use_qk_norm") else None
        
        # Output Projection
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )
        
        # KV Cache (运行时绑定)
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor):
        # QKV Projection
        q, k, v = self.qkv_proj(hidden_states)
        
        # QK Normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # RoPE
        q, k = self.rotary_emb(positions, q, k)
        
        # Attention
        context = get_context()
        if context.is_prefill:
            output = self._attention_prefill(q, k, v, context)
        else:
            output = self._attention_decode(q, k, v, context)
        
        # Output Projection
        output = self.o_proj(output)
        
        return output
```

### 3.2 Prefill Attention

```python
def _attention_prefill(self, q, k, v, context):
    """Prefill阶段Attention"""
    # 存储KV Cache
    if self.k_cache is not None:
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
    
    # 前缀缓存情况
    if context.block_tables is not None:
        # 使用缓存的K、V
        k = self.k_cache
        v = self.v_cache
    
    # Flash Attention (变长序列)
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        softmax_scale=self.scaling,
        causal=True,
    )
    
    return output
```

### 3.3 Decode Attention

```python
def _attention_decode(self, q, k, v, context):
    """Decode阶段Attention"""
    # 存储新token的KV Cache
    if self.k_cache is not None:
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
    
    # Flash Attention with KV Cache
    output = flash_attn_with_kvcache(
        q.unsqueeze(1),  # [batch, 1, heads, head_dim]
        self.k_cache,
        self.v_cache,
        block_tables=context.block_tables,
        context_lens=context.context_lens,
        softmax_scale=self.scaling,
    )
    
    return output.squeeze(1)
```

## 4. MLP实现

### 4.1 标准MLP

```python
class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        
        # Gate + Up Projection (合并)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        
        # Down Projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
    
    def forward(self, x):
        gate, up = self.gate_up_proj(x)
        x = silu_mul(gate, up)  # gate * silu(up)
        x = self.down_proj(x)
        return x
```

### 4.2 MoE MLP (可选)

```python
class MixtralMLP(nn.Module):
    """MoE MLP实现"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        # Router
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            LlamaMLP(config) for _ in range(self.num_experts)
        ])
    
    def forward(self, x):
        # Router输出
        router_logits = self.router(x)
        
        # Top-k选择
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(router_logits, dim=-1), self.top_k
        )
        
        # Expert计算
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]
            weight = top_k_weights[..., i]
            expert_output = self.experts[expert_idx](x)
            output = output + weight * expert_output
        
        return output
```

## 5. 权重加载

### 5.1 权重映射

```python
# Packed权重映射
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}

def load_weights(self, model_path: str):
    """加载模型权重"""
    # 加载safetensors
    state_dict = load_safetensors(model_path)
    
    for name, param in self.named_parameters():
        # 处理packed权重
        for orig_name, (packed_name, idx) in packed_modules_mapping.items():
            if orig_name in name:
                # 从packed权重中提取
                packed_weight = state_dict[name.replace(orig_name, packed_name)]
                if isinstance(idx, int):
                    # gate_up_proj情况
                    shard_size = packed_weight.shape[0] // 2
                    param.data.copy_(packed_weight[idx * shard_size:(idx + 1) * shard_size])
                else:
                    # qkv_proj情况
                    param.data.copy_(getattr(param, "weight_loader")(packed_weight))
                break
        else:
            # 普通权重
            param.data.copy_(state_dict[name])
```

### 5.2 张量并行权重加载

```python
class ColumnParallelLinear(nn.Module):
    def weight_loader(self, param, loaded_weight):
        """列并行权重加载：按rank切分"""
        tp_rank = get_tensor_parallel_rank()
        tp_size = get_tensor_parallel_world_size()
        
        shard_size = param.shape[self.tp_dim] // tp_size
        start = tp_rank * shard_size
        end = start + shard_size
        
        param.data.copy_(loaded_weight.narrow(self.tp_dim, start, shard_size))

class RowParallelLinear(nn.Module):
    def weight_loader(self, param, loaded_weight):
        """行并行权重加载：按输入维度切分"""
        tp_rank = get_tensor_parallel_rank()
        tp_size = get_tensor_parallel_world_size()
        
        shard_size = loaded_weight.shape[0] // tp_size
        start = tp_rank * shard_size
        end = start + shard_size
        
        param.data.copy_(loaded_weight[start:end])
```

## 6. 模型配置

### 6.1 配置参数

```python
@dataclass
class LlamaConfig:
    # 基本维度
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # GQA支持
    
    # 位置编码
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    
    # 归一化
    rms_norm_eps: float = 1e-5
    
    # 词表
    vocab_size: int = 32000
    
    # 计算衍生值
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
```

### 6.2 模型架构识别

```python
def get_model_class(config) -> type:
    """根据配置选择模型类"""
    architectures = getattr(config, "architectures", [])
    
    for arch in architectures:
        if "Llama" in arch:
            return LlamaForCausalLM
        elif "Mistral" in arch:
            return MistralForCausalLM
        elif "Qwen" in arch:
            return QwenForCausalLM
        # ...
    
    raise ValueError(f"Unknown architecture: {architectures}")
```

## 7. 不同模型变体

### 7.1 主流模型差异

| 模型 | 特殊之处 | 实现差异 |
|------|----------|----------|
| LLaMA | 基础架构 | 标准实现 |
| Mistral | Sliding Window | 滑动窗口注意力 |
| Qwen | RoPE变体 | 不同的旋转编码 |
| DeepSeek | MLA | 多头潜在注意力 |
| Mixtral | MoE | 专家混合架构 |
| Gemma | 特殊RoPE | 不同的位置编码 |

### 7.2 精简实现建议

对于精简框架，建议：

1. **只实现一种架构**：选择LLaMA作为基准
2. **固定配置**：不通过配置切换行为
3. **合并算子**：gate_up_proj, qkv_proj等
4. **简化权重加载**：直接映射，不做复杂转换
