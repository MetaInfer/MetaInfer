# Sampler — 词元采样策略

## 核心概念

模型输出 logits（词表中每个词元的未归一化分数）后，由 **Sampler** 选出下一个词元。这里是 **生成质量** 与 **生成可控性** 的交汇点。

## 采样流水线

```
logits → 温度缩放 → top-k 过滤 → top-p（核采样）过滤 → softmax → 采样
```

### 步骤 1：温度缩放

```python
logits = logits / temperature
# temperature < 1.0: 分布更尖（更确定）
# temperature = 1.0: 不变
# temperature > 1.0: 分布更平（更随机）
# temperature = 0.0: 贪心（argmax）
```

### 步骤 2：Top-K 过滤

```python
if top_k > 0:
    # 只保留 logits 最高的 top_k 个，其余置为 -inf
    top_k_values, _ = torch.topk(logits, top_k)
    threshold = top_k_values[:, -1]
    logits[logits < threshold] = -inf
```

### 步骤 3：Top-P（核采样）过滤

```python
if top_p < 1.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # 去掉累积概率超过阈值的词元
    mask = cumulative_probs - sorted_probs > top_p
    sorted_logits[mask] = -inf
    # 散射回原始位置
    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
```

### 步骤 4：采样

```python
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

## 各项目中的实现

### nano-vllm：编译后的 Sampler

使用 `@torch.compile` 做算子融合：

```python
@torch.compile(dynamic=True)
def sample(logits, temperatures):
    logits = logits / temperatures.unsqueeze(1)
    probs = torch.softmax(logits, dim=-1)
    # Gumbel-max 技巧，高效采样
    q = torch.empty_like(probs).exponential_(1)
    return (probs / q).argmax(dim=-1)
```

**Gumbel-max 技巧** 利用如下等价性，避免昂贵的 `torch.multinomial`：

```
argmax(log(probs) + gumbel_noise) == multinomial(probs)
```

其中 Gumbel 噪声为 `-log(-log(uniform))`，等价于从指数分布采样再相除。

### nano-sglang：带 FSM 的批量采样

支持 **按请求的采样参数** 与 **约束解码**：

```python
def sample(logits, batch):
    for i, req in enumerate(batch.reqs):
        # 约束解码：应用 FSM 掩码
        if req.regex_fsm:
            allowed = req.regex_fsm.allowed_token_ids(req.fsm_state)
            logits[i, ~allowed] = -inf

        # 每个请求各自的 temperature
        logits[i] /= req.sampling_params.temperature

    # 批量 top-p / top-k
    apply_top_k(logits, batch.top_k_values)
    apply_top_p(logits, batch.top_p_values)

    probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens
```

### mini-sglang：GPU 侧 Sampler

```python
class Sampler:
    def __call__(self, logits, batch):
        # 贪心（temperature = 0）
        if all_greedy:
            return logits.argmax(dim=-1)

        # 随机采样
        logits /= temperatures
        if has_top_k:
            apply_top_k(logits, top_k_values)
        if has_top_p:
            apply_top_p(logits, top_p_values)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
```

## 贪心解码捷径

当 `temperature == 0` 或显式要求贪心模式时：

```python
if temperature == 0:
    next_token = logits.argmax(dim=-1)
    # 跳过过滤与概率计算
```

比完整采样路径 **快得多**。

## 设计模板

最小 Sampler 需要：

1. **温度缩放**（始终要有）
2. `**temperature == 0` 时的贪心捷径**
3. 随机采样用 `**torch.multinomial`** 或 **Gumbel-max**

可选增强：

- Top-k / Top-p 过滤（常见用户需求）  
- 按请求的采样参数（服务场景必需）  
- `@torch.compile` 融合内核（可测得的加速）  
- 与约束解码集成（应用 FSM 掩码）  
- 重复惩罚、频率惩罚、存在惩罚  
- Min-p 采样

