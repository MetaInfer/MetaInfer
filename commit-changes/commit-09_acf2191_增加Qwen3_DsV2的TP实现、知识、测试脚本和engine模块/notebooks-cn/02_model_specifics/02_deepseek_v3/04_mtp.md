# DeepSeek V3 — 多 Token 预测（MTP）

## 核心概念

MTP 在「下一 token」之外增加若干**预测头**，可预测更远的未来 token。推理时这些头可充当**内建投机解码**。

## 结构

```
主模型前向:
    hidden_states[layer_L]  ← 主模型最后一层隐状态
        ↓
MTP 头 1（预测位置 t+2 的 token）:
    ├── eh_proj: concat(embed(token_{t+1}), hidden[L]) → 合并特征
    ├── RMSNorm
    ├── TransformerLayer → mtp_hidden_1
    └── LM Head → 位置 t+2 的 logits

MTP 头 2（预测 t+3）:
    ├── eh_proj: concat(embed(token_{t+2}), mtp_hidden_1)
    ...
```

## 设计要点

1. **Embedding–Hidden 投影（eh_proj）**：把「已预测 token 的 embedding」与上一段 hidden 拼在一起
2. **共享 LM Head**：MTP 头常复用主模型 LM head 权重
3. **轻量层数**：每个 MTP 头通常只含 1 个 Transformer 层（主模型可有 32+ 层）

## 与投机解码结合

### 草稿（draft）阶段
```python
def speculative_draft(main_hidden, last_token):
    draft_tokens = [last_token]
    h = main_hidden
    for mtp_head in mtp_heads:
        combined = mtp_head.eh_proj(concat(embed(draft_tokens[-1]), h))
        h = mtp_head.transformer_layer(combined)
        logits = mtp_head.lm_head(h)
        draft_token = sample(logits)
        draft_tokens.append(draft_token)
    return draft_tokens[1:]  # N 个草稿 token
```

### 验证（verify）阶段
```python
def speculative_verify(input_tokens, draft_tokens):
    all_tokens = concat(input_tokens, draft_tokens)
    all_logits = main_model.forward(all_tokens)
    accepted = []
    for i, draft in enumerate(draft_tokens):
        main_prediction = sample(all_logits[input_len + i])
        if main_prediction == draft:
            accepted.append(draft)
        else:
            accepted.append(main_prediction)
            break
    return accepted
```

## 对推理框架的影响

1. **隐状态出口**：主模型需能取出用于 MTP 的中间 hidden
2. **额外前向**：每步投机会多一次较轻的 MTP 前向（但比主模型便宜得多）
3. **验证批处理**：需把草稿与主模型一次前向做 batch 验证
4. **接受率**：接受率越高加速越明显；训练良好的 MTP 常可达约 70%–90%
5. **显存**：MTP 头额外显存相对不大（每头约 1 层 + 共享 LM head）

## 何时值得用

- **更适合**：单请求、低延迟、关心每 token 延时的场景
- **不太适合**：大 batch 高吞吐、GPU 已打满时
- **取舍**：推理循环更复杂；仅当接受率足够高时才值得
