# DeepSeek V3 整体架构

## 模型结构概览

DeepSeek V3 采用 Decoder-only Transformer 架构，其核心结构如下：

```
DeepSeekV3ForCausalLM
├── model (DeepseekV2Model)
│   ├── embed_tokens (VocabParallelEmbedding)
│   ├── layers[0..60] (DeepseekV2DecoderLayer)
│   │   ├── self_attn (DeepseekV2MLAAttention)
│   │   ├── mlp (DeepseekV2MoE | DeepseekV2MLP)
│   │   ├── input_layernorm (RMSNorm)
│   │   └── post_attention_layernorm (RMSNorm)
│   └── norm (RMSNorm)
└── lm_head (ParallelLMHead)
```

## Decoder Layer 结构

### 层类型分布

DeepSeek V3 的 61 层分为两种类型：

```python
# 前 first_k_dense_replace 层为 Dense MLP
first_k_dense_replace = 1  # 第一层是 Dense MLP

# 其余层为 MoE MLP，按 moe_layer_freq 间隔
moe_layer_freq = 1  # 每1层一个MoE
```

**层分布**：
- Layer 0: Dense MLP
- Layer 1-60: MoE MLP

### DecoderLayer 前向传播

```python
def forward(positions, hidden_states, residual):
    # Self Attention
    if residual is None:
        residual = hidden_states.clone()
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    
    hidden_states = self.self_attn(positions, hidden_states)
    
    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    
    return hidden_states, residual
```

## 关键配置参数

### 完整配置示例

```python
@dataclass
class DeepseekV3Config:
    # 基础参数
    model_type: str = "deepseek_v3"
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    vocab_size: int = 129280
    
    # Attention 参数
    num_attention_heads: int = 128
    qk_nope_head_dim: int = 128      # QK 非RoPE部分
    qk_rope_head_dim: int = 64       # QK RoPE部分
    v_head_dim: int = 128            # V 头维度
    kv_lora_rank: int = 512          # KV 低秩压缩秩
    q_lora_rank: int = 1536          # Q 低秩压缩秩
    
    # MoE 参数
    n_routed_experts: int = 256      # 路由专家数
    n_shared_experts: int = 1        # 共享专家数
    num_experts_per_tok: int = 8     # 每token激活专家数
    moe_intermediate_size: int = 2048
    n_group: int = 8                 # 专家分组数
    topk_group: int = 4              # 每组选top-k
    first_k_dense_replace: int = 1   # 前几层Dense
    moe_layer_freq: int = 1          # MoE层频率
    routed_scaling_factor: float = 256.0  # 路由缩放因子
    scoring_func: str = "softmax"    # 评分函数
    topk_method: str = "noaux_tc"    # TopK方法
    
    # MTP 参数
    num_nextn_predict_layers: int = 1
    
    # Norm 参数
    rms_norm_eps: float = 1e-6
    
    # RoPE 参数
    rope_parameters: dict = {
        "rope_type": "deepseek_yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096
    }
```

## 两种注意力实现

DeepSeek V3 支持两种注意力模式：

### 1. MLA (Multi-Head Latent Attention) - 默认

用于 DeepSeek V2/V3 系列，通过低秩压缩减少 KV Cache。

### 2. MHA (Multi-Head Attention) - DeepSeek V1

用于原始 DeepSeek 模型，标准的注意力实现。

**判断逻辑**：

```python
# 判断是否使用 MHA
qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)

use_mha = config.model_type == "deepseek" or all(
    dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
)
```

## 权重映射

### 模块融合策略

vLLM 和 SGLang 都实现了权重融合以优化推理：

```python
# MLP 权重融合
packed_modules_mapping = {
    "gate_up_proj": ["gate_proj", "up_proj"],  # MLP gate 和 up 融合
}

# MLA 权重融合
packed_modules_mapping["fused_qkv_a_proj"] = [
    "q_a_proj",        # Q 低秩投影
    "kv_a_proj_with_mqa",  # KV 低秩投影 + RoPE
]
```

### 权重加载流程

```python
def load_weights(weights):
    for name, weight in weights:
        # 1. 跳过不需要的权重
        if "rotary_emb.inv_freq" in name:
            continue
        
        # 2. 处理 MTP 层权重
        spec_layer = get_spec_layer_idx(name)
        if spec_layer is not None:
            continue  # 主模型跳过 MTP 层
        
        # 3. 处理融合权重
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name:
                name = name.replace(weight_name, param_name)
                param.weight_loader(param, weight, shard_id)
                break
        
        # 4. 处理专家权重
        # ...
```

## FP16 溢出修复

DeepSeek V3 使用 `routed_scaling_factor` 缩放路由专家输出，在 FP16 下可能导致溢出：

```python
# 在 MLA Attention 和 MoE 中需要特殊处理
if hidden_states.dtype == torch.float16:
    # 缩放以避免溢出
    hidden_states *= 1.0 / self.routed_scaling_factor
    if layer_idx == 0:
        residual *= 1.0 / self.routed_scaling_factor
```

## 模型初始化流程

### vLLM 实现

```python
class DeepseekV3ForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        # 1. 初始化配置
        self.config = vllm_config.model_config.hf_config
        
        # 2. 创建模型主体
        self.model = DeepseekV2Model(vllm_config=vllm_config, prefix=prefix)
        
        # 3. 创建 LM Head
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 4. 设置 MoE 参数
        self.set_moe_parameters()
```

### SGLang 实现

```python
class DeepseekV3ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    def __init__(self, config: PretrainedConfig, quant_config: QuantizationConfig):
        # 1. 创建模型主体
        self.model = DeepseekV2Model(config, quant_config)
        
        # 2. 创建 LM Head
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 3. 确定共享专家融合策略
        self.determine_num_fused_shared_experts()
```

## 并行策略支持

DeepSeek V3 支持多种并行策略：

| 并行类型 | 说明 | 适用场景 |
|----------|------|----------|
| TP (Tensor Parallel) | 张量并行，分割权重 | 单节点多GPU |
| EP (Expert Parallel) | 专家并行，分割专家 | MoE 专用 |
| DP (Data Parallel) | 数据并行，复制模型 | 多节点部署 |
| PP (Pipeline Parallel) | 流水线并行，分割层 | 大模型部署 |
| CP (Context Parallel) | 上下文并行，分割序列 | 长序列处理 |
