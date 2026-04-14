# Multi-Token Prediction (MTP)

## 概述

Multi-Token Prediction (MTP) 是 DeepSeek V3 引入的投机解码机制，通过同时预测多个 token 来加速推理。该实现基于 EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 架构。

## 架构设计

### MTP 结构

```
DeepSeekMTP (投机解码模型)
├── model (DeepSeekMultiTokenPredictor)
│   ├── embed_tokens (VocabParallelEmbedding)    # 与主模型共享
│   └── layers (ModuleDict)
│       └── [61, 62, ...] (DeepSeekMultiTokenPredictorLayer)
│           ├── enorm (RMSNorm)                  # Embedding norm
│           ├── hnorm (RMSNorm)                  # Hidden states norm
│           ├── eh_proj (Linear)                 # Embedding + Hidden 投影
│           ├── mtp_block (DeepseekV2DecoderLayer)
│           │   ├── self_attn
│           │   └── mlp
│           └── shared_head
│               ├── norm (RMSNorm)
│               └── head (ParallelLMHead)
└── logits_processor
```

### 工作原理

```
主模型推理:
  Token t → [Main Model] → Hidden_t → LM_Head → Logits_t+1

MTP 推理:
  Hidden_t + Embed_{t+1} → [MTP Layer 1] → Hidden_t+1' → Head_1 → Logits_t+2
  Hidden_t+1' + Embed_{t+2} → [MTP Layer 2] → Hidden_t+2' → Head_2 → Logits_t+3
  ...

验证阶段:
  比较 MTP 预测的 Logits 与主模型实际计算的 Logits
  接受匹配的 tokens，拒绝不匹配的
```

## 代码实现

### MTP Layer 结构

```python
class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        
        # Norm 层
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 投影层：将 embedding 和 hidden states 融合
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        # Transformer 层（复用主模型结构）
        self.mtp_block = DeepseekV2DecoderLayer(vllm_config, prefix, config=config)
        
        # 共享的输出头
        self.shared_head = SharedHead(config, prefix)
```

### MTP 前向传播

```python
def forward(self, input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_index):
    """
    MTP 前向传播
    
    Args:
        input_ids: 当前输入 token ids
        positions: 位置编码
        previous_hidden_states: 主模型或上一 MTP 层的隐藏状态
        inputs_embeds: 输入 embedding
        spec_step_index: 投机步骤索引
    
    Returns:
        hidden_states: 当前层的隐藏状态
    """
    # 1. Mask 掉位置 0 的输入（MTP 不需要）
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    
    # 2. 归一化
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = self.hnorm(previous_hidden_states)
    
    # 3. 融合 embedding 和 hidden states
    hidden_states = self.eh_proj(
        torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
    )
    
    # 4. 通过 Transformer 层
    hidden_states, residual = self.mtp_block(
        positions=positions,
        hidden_states=hidden_states,
        residual=None
    )
    hidden_states = residual + hidden_states
    
    return hidden_states
```

### SGLang NextN 实现

```python
class DeepseekModelNextN(nn.Module):
    """SGLang 的 MTP 实现（称为 NextN）"""
    
    def __init__(self, config, quant_config, prefix):
        super().__init__()
        
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        
        self.decoder = DeepseekV2DecoderLayer(
            config, 0, quant_config=quant_config,
            is_nextn=True, prefix=prefix
        )
        
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

### NextN 前向传播

```python
def forward(self, input_ids, positions, forward_batch, input_embeds):
    if input_embeds is None:
        hidden_states = self.embed_tokens(input_ids)
    else:
        hidden_states = input_embeds
    
    if hidden_states.shape[0] > 0:
        # 融合当前 embedding 和主模型隐藏状态
        hidden_states = self.eh_proj(
            torch.cat(
                (
                    self.enorm(hidden_states),
                    self.hnorm(forward_batch.spec_info.hidden_states),
                ),
                dim=-1,
            )
        )
    
    # 通过 Transformer 层
    hidden_states, residual, topk_indices = self.decoder(
        positions, hidden_states, forward_batch, residual, zero_allocator
    )
    
    # 输出
    hidden_states = self.shared_head.norm(hidden_states, residual)
    
    return hidden_states
```

## EAGLE 投机解码

### 配置参数

```python
# EAGLE 投机解码配置
speculative_algorithm = "EAGLE"
speculative_num_steps = 3      # 投机步骤数
speculative_eagle_topk = 1     # 每步 top-k 选择
speculative_num_draft_tokens = 4  # 投机 token 数量
```

### 工作流程

```
1. 主模型推理 step:
   - 输入: [t0, t1, ..., tn]
   - 输出: Hidden_n, Logits_n+1
   - 选择: token_{n+1}

2. MTP 投机 step:
   - 输入: Hidden_n, Embed_{n+1}
   - 输出: Hidden_{n+1}', Logits_{n+2}
   - 预测: token_{n+2}, token_{n+3}, ...

3. 验证 step:
   - 主模型计算实际 Logits
   - 比较与投机 Logits
   - 接受/拒绝决策

4. 提交 step:
   - 接受的 tokens 加入序列
   - 拒绝的 tokens 丢弃
```

### 性能提升

| Batch Size | 加速比 |
|------------|--------|
| 1 | 1.8x |
| 32 | 1.5x |
| 64+ | 1.2x |

## 权重共享

### Embedding 共享

MTP 层与主模型共享 embedding：

```python
# MTP 使用主模型的 embedding
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    prefix=maybe_prefix(prefix, "embed_tokens"),
)

# 权重加载时只加载一次
if spec_layer != self.model.mtp_start_layer_idx:
    # 跳过后续层的 embedding
    continue
```

### LM Head 独立

每个 MTP 层有独立的 LM Head：

```python
class SharedHead(nn.Module):
    def __init__(self, config, prefix, quant_config):
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
```

## V3.2 特性：Index Top-K

DeepSeek V3.2 在 MTP 中引入了 Index Top-K 优化：

```python
# V3.2 特有参数
index_topk = 32  # 保留 top-32 候选

# 在 MTP 层中使用
if self.is_v32:
    topk_indices_buffer = torch.empty(
        max_num_batched_tokens,
        topk_tokens,
        dtype=torch.int32,
        device=device,
    )
```

## 使用方法

### 启用 MTP

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V3 \
    --speculative-model deepseek-ai/DeepSeek-V3 \
    --num-speculative-tokens 4

# SGLang
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --speculative-algorithm EAGLE \
    --tp 8
```

### 默认配置

```python
# DeepSeek V3 MTP 默认配置
default_config = {
    "speculative_num_steps": 3,
    "speculative_eagle_topk": 1,
    "speculative_num_draft_tokens": 4,
}

# 最小配置（资源受限场景）
min_config = {
    "speculative_num_steps": 1,
    "speculative_eagle_topk": 1,
    "speculative_num_draft_tokens": 2,
}
```

## 实现差异

### vLLM vs SGLang

| 特性 | vLLM | SGLang |
|------|------|--------|
| 类名 | DeepSeekMTP | DeepseekModelNextN |
| 层数 | 可配置 | 可配置 |
| Embedding 共享 | ✅ | ✅ |
| CUDA Graph | ✅ | ✅ |
| Overlap Scheduler | - | ✅ (SGLANG_ENABLE_SPEC_V2=1) |
| 大 Batch 支持 | 需调整参数 | 需调整参数 |

### 大 Batch 配置

```bash
# 对于 batch > 48，需要调整参数
--max-running-requests 128  # 增大
--cuda-graph-bs 1 2 4 8 16 32 64 128  # 捕获更多 batch size
```
