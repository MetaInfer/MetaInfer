# Qwen3 Dense — 稠密模型结构

## 整体结构

```
Qwen3ForCausalLM
├── model: Qwen3Model
│   ├── embed_tokens: VocabParallelEmbedding(vocab_size, hidden_size)
│   ├── layers: ModuleList[Qwen3DecoderLayer × num_hidden_layers]
│   │   └── Qwen3DecoderLayer
│   │       ├── input_layernorm: RMSNorm(hidden_size, eps)
│   │       ├── self_attn: Qwen3Attention
│   │       │   ├── qkv_proj: QKVParallelLinear(hidden_size → q_size + 2*kv_size, bias=False)
│   │       │   ├── q_norm: RMSNorm(head_dim, eps)         ← Qwen3 特有
│   │       │   ├── k_norm: RMSNorm(head_dim, eps)         ← Qwen3 特有
│   │       │   ├── rotary_emb: RotaryEmbedding(head_dim, theta=1e6)
│   │       │   ├── attn: Attention(num_heads, head_dim, num_kv_heads)
│   │       │   └── o_proj: RowParallelLinear(num_heads*head_dim → hidden_size, bias=False)
│   │       ├── post_attention_layernorm: RMSNorm(hidden_size, eps)
│   │       └── mlp: Qwen3MLP
│   │           ├── gate_up_proj: MergedColumnParallelLinear(hidden_size → 2*intermediate_size, bias=False)
│   │           ├── act_fn: SiluAndMul
│   │           └── down_proj: RowParallelLinear(intermediate_size → hidden_size, bias=False)
│   └── norm: RMSNorm(hidden_size, eps)
└── lm_head: ParallelLMHead(hidden_size → vocab_size)
    └── (可选: tie_word_embeddings 与 embed_tokens 共享权重)
```

## Attention 前向传播详解

Qwen3 Attention 的核心区别在于 QK Norm 的引入：

```python
def forward(self, positions, hidden_states):
    # Step 1: QKV 投影（合并为单个 GEMM）
    qkv = self.qkv_proj(hidden_states)       # [B, hidden_size] → [B, q_size + 2*kv_size]
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # Step 2: Reshape 为多头格式
    q = q.view(-1, num_heads, head_dim)       # [B, num_q_heads, head_dim]
    k = k.view(-1, num_kv_heads, head_dim)    # [B, num_kv_heads, head_dim]
    v = v.view(-1, num_kv_heads, head_dim)    # [B, num_kv_heads, head_dim]

    # Step 3: QK Norm（Qwen3 特有！在 RoPE 之前）
    q = self.q_norm(q)                        # RMSNorm on head_dim
    k = self.k_norm(k)                        # RMSNorm on head_dim

    # Step 4: 旋转位置编码
    q, k = self.rotary_emb(positions, q, k)

    # Step 5: 注意力计算（含 KV cache 读写）
    o = self.attn(q, k, v)                    # [B, num_heads, head_dim]

    # Step 6: 输出投影（含 all-reduce）
    output = self.o_proj(o.flatten(1, -1))    # [B, num_heads*head_dim] → [B, hidden_size]
    return output
```

### QK Norm 的重要性

QK Norm 防止 Q 和 K 的点积在训练/推理中出现数值不稳定。它在 head_dim 维度上对每个 head 独立做 RMSNorm：

```python
class QKNorm(RMSNorm):
    """与标准 RMSNorm 相同，但作用在 head_dim 维度"""
    # weight shape: [head_dim]
    # input shape: [batch, num_heads, head_dim]
    # 对最后一个维度做归一化
```

**关键：QK Norm 在 RoPE 之前应用。** 顺序是：`QKV投影 → QK Norm → RoPE → Attention`。

### Qwen3 与 Qwen2 的区别

| 操作 | Qwen2 | Qwen3 |
|------|-------|-------|
| attention_bias | `True` | `False` |
| QK Norm | 无 | **有** |
| QKV → RoPE 流程 | QKV → RoPE → Attn | QKV → **QK Norm** → RoPE → Attn |

因此 Qwen3 的代码中有这样的判断：
```python
# nano-vllm 中：qkv_bias=False 时才有 QK Norm
if not self.qkv_bias:
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

## MLP 前向传播

```python
def forward(self, x):
    # Step 1: Gate + Up 合并投影
    gate_up = self.gate_up_proj(x)        # [B, hidden_size] → [B, 2*intermediate_size]

    # Step 2: SiLU 激活 + 门控
    x = self.act_fn(gate_up)              # SiLU(gate) * up → [B, intermediate_size]

    # Step 3: Down 投影（含 all-reduce）
    x = self.down_proj(x)                 # [B, intermediate_size] → [B, hidden_size]
    return x
```

## DecoderLayer 前向传播

```python
def forward(self, positions, hidden_states, residual):
    # Fused Residual + Norm (第一次)
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    # Attention
    hidden_states = self.self_attn(positions, hidden_states)

    # Fused Residual + Norm (第二次)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

    # MLP
    hidden_states = self.mlp(hidden_states)

    return hidden_states, residual
```

**关键模式**：residual 在层间传递，避免额外的内存分配。第一层的 residual 为 None，由 input_layernorm 初始化。

## HuggingFace Config 字段

```python
# 从 HF config.json 中需要读取的字段
config_fields = {
    "hidden_size": int,             # 模型隐藏维度，如 4096
    "num_hidden_layers": int,       # 层数，如 36
    "num_attention_heads": int,     # Q 头数，如 32
    "num_key_value_heads": int,     # KV 头数（GQA），如 8
    "head_dim": int,                # 每头维度，通常 128（可选，默认 hidden_size//num_heads）
    "intermediate_size": int,       # MLP 中间维度，如 12288
    "vocab_size": int,              # 词表大小，如 151936
    "max_position_embeddings": int, # 最大位置，如 40960
    "rms_norm_eps": float,          # RMSNorm epsilon，如 1e-6
    "rope_theta": float,            # RoPE 基频，如 1000000.0
    "rope_scaling": dict | None,    # RoPE 缩放配置
    "hidden_act": str,              # 激活函数，"silu"
    "attention_bias": bool,         # QKV 投影是否有 bias，Qwen3 为 False
    "tie_word_embeddings": bool,    # 是否共享 embedding 和 lm_head 权重
}
```

## 权重名称映射

### HuggingFace 权重 → 推理框架权重

```python
# packed_modules_mapping（nano-vllm 风格）
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),      # HF q_proj.weight → 合并到 qkv_proj 的 Q 部分
    "k_proj": ("qkv_proj", "k"),      # HF k_proj.weight → 合并到 qkv_proj 的 K 部分
    "v_proj": ("qkv_proj", "v"),      # HF v_proj.weight → 合并到 qkv_proj 的 V 部分
    "gate_proj": ("gate_up_proj", 0), # HF gate_proj.weight → 合并到 gate_up_proj 的前半
    "up_proj": ("gate_up_proj", 1),   # HF up_proj.weight → 合并到 gate_up_proj 的后半
}
```

### 完整权重列表

```
model.embed_tokens.weight                           [vocab_size, hidden_size]
model.layers.{i}.input_layernorm.weight              [hidden_size]
model.layers.{i}.self_attn.q_proj.weight             [num_heads * head_dim, hidden_size]
model.layers.{i}.self_attn.k_proj.weight             [num_kv_heads * head_dim, hidden_size]
model.layers.{i}.self_attn.v_proj.weight             [num_kv_heads * head_dim, hidden_size]
model.layers.{i}.self_attn.q_norm.weight             [head_dim]           ← Qwen3 特有
model.layers.{i}.self_attn.k_norm.weight             [head_dim]           ← Qwen3 特有
model.layers.{i}.self_attn.o_proj.weight             [hidden_size, num_heads * head_dim]
model.layers.{i}.post_attention_layernorm.weight     [hidden_size]
model.layers.{i}.mlp.gate_proj.weight                [intermediate_size, hidden_size]
model.layers.{i}.mlp.up_proj.weight                  [intermediate_size, hidden_size]
model.layers.{i}.mlp.down_proj.weight                [hidden_size, intermediate_size]
model.norm.weight                                    [hidden_size]
lm_head.weight                                      [vocab_size, hidden_size]
```

## TP 切分方式

| 权重 | 切分维度 | 切分方式 | 通信 |
|------|---------|---------|------|
| `embed_tokens` | vocab 维 | 每 rank 持有 vocab_size/tp_size | all-reduce |
| `qkv_proj` | 输出维（Q heads + KV heads） | 每 rank 持有 local_heads | 无 |
| `q_norm`, `k_norm` | 无 | 每 rank 完整复制 | 无 |
| `o_proj` | 输入维 | 每 rank 持有 num_heads/tp_size 的输出 | **all-reduce** |
| `gate_up_proj` | 输出维 | 每 rank 持有 intermediate_size/tp_size | 无 |
| `down_proj` | 输入维 | 每 rank 持有 intermediate_size/tp_size | **all-reduce** |
| `lm_head` | vocab 维 | 每 rank 持有 vocab_size/tp_size | all-gather |

## 代码生成模板

生成 Qwen3 Dense 推理代码时的最小必要组件：

```python
# 最小 Qwen3 Dense 实现骨架
class Qwen3Attention(nn.Module):
    def __init__(self, config):
        self.qkv_proj = QKVParallelLinear(...)
        self.q_norm = RMSNorm(head_dim)    # ← 必须有
        self.k_norm = RMSNorm(head_dim)    # ← 必须有
        self.rotary_emb = RotaryEmbedding(head_dim, theta=1e6)
        self.attn = AttentionBackend(...)
        self.o_proj = RowParallelLinear(...)

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        self.gate_up_proj = MergedColumnParallelLinear(...)
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(...)

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = Qwen3MLP(config)
```

## 源码参考

| 项目 | 文件 |
|------|------|
| nano-vllm | `nanovllm/models/qwen3.py` (最简实现，~217行) |
| mini-sglang | `minisgl/models/qwen3.py` (上下文驱动风格) |
| vllm | `vllm/model_executor/models/qwen3.py` (继承自 Qwen2) |
| sglang | `srt/models/qwen3.py` (含 JIT 融合 QK Norm 优化) |
