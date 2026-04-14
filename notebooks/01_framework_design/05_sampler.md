# 采样器

## 1. 采样器职责

采样器负责从模型输出的logits生成token：

1. **温度缩放**：控制输出的随机性
2. **Top-k/Top-p过滤**：限制候选token范围
3. **采样方法**：从概率分布中采样

## 2. 基础采样器实现

### 2.1 最简采样器

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        Args:
            logits: [batch_size, vocab_size]
            temperatures: [batch_size]
        Returns:
            token_ids: [batch_size]
        """
        # 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # Softmax
        probs = torch.softmax(logits, dim=-1)
        
        # Gumbel-Max采样（避免显式随机数生成）
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        
        return sample_tokens
```

### 2.2 Gumbel-Max采样原理

Gumbel-Max采样是一种等价于Multinomial采样但更高效的方法：

```python
# 标准Multinomial采样
probs = softmax(logits / temperature)
token = multinomial(probs, num_samples=1)

# Gumbel-Max等价实现
gumbel_noise = -log(-log(uniform()))
token = argmax(logits / temperature + gumbel_noise)
```

**优点**：
- 避免显式随机数生成
- 更好的CUDA并行性
- 支持torch.compile优化

## 3. Top-k和Top-p采样

### 3.1 Top-k采样

只从概率最高的k个token中采样：

```python
def top_k_sampling(logits, k):
    # 获取top-k的索引和值
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Softmax只对top-k计算
    probs = torch.softmax(top_k_values, dim=-1)
    
    # 从top-k中采样
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    # 映射回原始索引
    return torch.gather(top_k_indices, dim=-1, index=sampled_idx)
```

### 3.2 Top-p（Nucleus）采样

从累积概率达到p的最小token集合中采样：

```python
def top_p_sampling(logits, p):
    # 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 计算累积概率
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过p的位置
    sorted_indices_to_remove = cum_probs > p
    
    # 保留第一个超过p的token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 将被移除的token设为负无穷
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    # 采样
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1)
    
    return torch.gather(sorted_indices, dim=-1, index=sampled_idx)
```

### 3.3 组合Top-k和Top-p

```python
def top_k_top_p_sampling(logits, k, p):
    """
    先Top-k，再Top-p
    """
    # Top-k过滤
    if k > 0:
        top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
        top_k_logits = torch.full_like(logits, float('-inf'))
        top_k_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        logits = top_k_logits
    
    # Top-p过滤
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cum_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        logits = torch.gather(sorted_logits, dim=-1, index=sorted_indices.argsort())
    
    # 采样
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## 4. 批量采样参数

### 4.1 采样参数数据结构

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0          # 0表示不限制
    top_p: float = 1.0      # 1.0表示不限制
    max_tokens: int = 16
    ignore_eos: bool = False
    stop_str: list[str] = None
```

### 4.2 批量采样

```python
class Sampler(nn.Module):
    def forward(
        self, 
        logits: torch.Tensor,           # [batch, vocab_size]
        temperatures: torch.Tensor,     # [batch]
        top_ks: torch.Tensor = None,    # [batch]
        top_ps: torch.Tensor = None,    # [batch]
    ):
        batch_size = logits.size(0)
        
        # 温度缩放
        logits = logits.float()
        logits = logits / temperatures.unsqueeze(1)
        
        # Top-k过滤
        if top_ks is not None:
            for i in range(batch_size):
                if top_ks[i] > 0:
                    top_k = min(top_ks[i].item(), logits.size(-1))
                    top_k_values, _ = torch.topk(logits[i], top_k)
                    logits[i, logits[i] < top_k_values[-1]] = float('-inf')
        
        # Top-p过滤
        if top_ps is not None:
            for i in range(batch_size):
                if top_ps[i] < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=0)
                    sorted_indices_to_remove = cum_probs > top_ps[i]
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    sorted_logits[sorted_indices_to_remove] = float('-inf')
                    logits[i] = torch.gather(sorted_logits, dim=0, index=sorted_indices.argsort())
        
        # 采样
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

## 5. 特殊采样模式

### 5.1 Greedy（贪婪）采样

```python
def greedy_sampling(logits):
    """总是选择概率最高的token"""
    return torch.argmax(logits, dim=-1)
```

### 5.2 Beam Search

```python
def beam_search(logits, beam_width, num_beams):
    """
    Beam Search采样
    注意：需要维护多个候选序列，实现较复杂
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=-1)
    # ... 维护beam状态
```

### 5.3 约束生成

```python
def constrained_sampling(logits, allowed_token_ids):
    """
    只从允许的token中采样
    用于结构化输出（JSON、正则表达式等）
    """
    # 创建mask
    mask = torch.full_like(logits, float('-inf'))
    mask[:, allowed_token_ids] = 0
    
    # 应用mask
    logits = logits + mask
    
    # 正常采样
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## 6. Logits处理

### 6.1 Logit Bias

```python
def apply_logit_bias(logits, logit_bias):
    """
    应用logit偏置
    用于调整特定token的生成概率
    """
    if logit_bias is not None:
        # logit_bias: dict[int, float] = {token_id: bias}
        for token_id, bias in logit_bias.items():
            logits[:, token_id] += bias
    return logits
```

### 6.2 重复惩罚

```python
def apply_repetition_penalty(logits, generated_tokens, penalty):
    """
    重复惩罚：降低已生成token的概率
    """
    if penalty != 1.0:
        for token_id in generated_tokens:
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits
```

## 7. 采样器设计建议

### 7.1 精简框架推荐

| 功能 | 推荐 | 原因 |
|------|------|------|
| 采样方法 | Gumbel-Max | 简单高效，支持torch.compile |
| Top-k/Top-p | 可选 | 增加复杂度但提高输出质量 |
| 约束生成 | 不实现 | 属于非核心功能 |

### 7.2 性能优化

```python
# 使用torch.compile优化采样器
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        # 编译后的代码更高效
        ...
```

### 7.3 批量处理优化

```python
# 避免逐样本循环
# 差
for i in range(batch_size):
    logits[i] = top_k_filter(logits[i], top_ks[i])

# 好：使用矩阵操作
def batch_top_k_filter(logits, top_ks):
    # 使用torch.topk批量处理
    ...
```

## 8. 采样参数默认值

```python
# 常用采样参数配置
DEFAULT_SAMPLING_PARAMS = SamplingParams(
    temperature=1.0,    # 标准温度
    top_k=0,           # 不限制
    top_p=1.0,         # 不限制
    max_tokens=16,     # 默认生成长度
)

# 高质量生成配置
CREATIVE_SAMPLING_PARAMS = SamplingParams(
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    max_tokens=256,
)

# 确定性生成配置
DETERMINISTIC_PARAMS = SamplingParams(
    temperature=0.0,   # 等同于greedy
    max_tokens=64,
)
```
