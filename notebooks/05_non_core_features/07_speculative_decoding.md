# 投机解码（Speculative Decoding）

## 概述

投机解码通过一个快速的"草稿模型"预测多个 token，然后用目标模型一次性验证，从而在保持输出质量不变的前提下，将 decode 阶段加速 2-4 倍。

## 为什么是可选但重要的

- **可选**：会增加系统复杂度，在高吞吐批量场景下收益有限
- **重要**：在延迟敏感的单请求/少请求场景下，可显著降低 Time Per Output Token (TPOT)

## 核心算法：Draft-Verify Loop

```python
def speculative_decode_step(target_model, draft_model, prompt, num_spec_tokens=5):
    # === 阶段 1: Draft（草稿生成） ===
    draft_tokens = []
    draft_probs = []
    for _ in range(num_spec_tokens):
        logits = draft_model.forward(last_token)
        probs = softmax(logits)
        token = sample(probs)
        draft_tokens.append(token)
        draft_probs.append(probs)

    # === 阶段 2: Verify（目标模型验证） ===
    # 用目标模型一次 forward 处理所有 draft tokens
    all_tokens = prompt_tokens + draft_tokens
    target_logits = target_model.forward(all_tokens)  # 像 prefill 一样并行
    target_probs = softmax(target_logits)

    # === 阶段 3: Accept/Reject ===
    accepted_tokens = []
    for i, (draft_tok, draft_p, target_p) in enumerate(
        zip(draft_tokens, draft_probs, target_probs)
    ):
        if is_greedy:
            # Greedy: 只要目标模型的 argmax 和 draft 一致就接受
            if target_p.argmax() == draft_tok:
                accepted_tokens.append(draft_tok)
            else:
                accepted_tokens.append(target_p.argmax())
                break  # 第一个拒绝后停止
        else:
            # Stochastic: 按概率比接受
            r = random.uniform(0, 1)
            if r < min(1, target_p[draft_tok] / draft_p[draft_tok]):
                accepted_tokens.append(draft_tok)
            else:
                # 从修正分布中重采样
                corrected = max(0, target_p - draft_p)
                corrected = corrected / corrected.sum()
                accepted_tokens.append(sample(corrected))
                break

    # === Bonus Token ===
    # 如果所有 draft tokens 被接受，target_model 的最后一个位置还提供了一个额外 token
    if len(accepted_tokens) == num_spec_tokens:
        bonus = sample(target_probs[num_spec_tokens])
        accepted_tokens.append(bonus)

    return accepted_tokens  # 最多 num_spec_tokens + 1 个 token
```

## 三种投机方式

### 1. 外部 Draft Model
使用一个更小的同系列模型（如用 Llama-7B 草稿 + Llama-70B 验证）:

```python
class DraftModelSpeculator:
    def __init__(self, draft_model, target_model):
        self.draft_model = draft_model   # 小模型
        self.target_model = target_model  # 大模型
        # draft 和 target 共享 tokenizer

    def draft(self, input_ids, num_tokens):
        tokens = []
        for _ in range(num_tokens):
            logits = self.draft_model(input_ids[-1:])
            token = sample(logits)
            tokens.append(token)
            input_ids = torch.cat([input_ids, token.unsqueeze(0)])
        return tokens
```

**优点**：通用性强，任何 target 模型都可以配一个 draft 模型
**缺点**：需要额外的模型内存，draft 模型的词表必须和 target 一致

### 2. EAGLE（基于隐层状态的投机）
EAGLE 使用 target 模型的隐层状态作为额外输入，显著提高 draft 接受率：

```python
class EAGLESpeculator:
    def __init__(self, target_model, eagle_head):
        self.target_model = target_model
        self.eagle_head = eagle_head  # 轻量级模型（1层 transformer）

    def draft(self, hidden_states, last_token_embed, num_tokens):
        """
        hidden_states: target 模型最后一层的隐层输出
        last_token_embed: 最后一个 token 的 embedding
        """
        tokens = []
        h = hidden_states

        for _ in range(num_tokens):
            # 合并 embedding 和 hidden state
            combined = self.eagle_head.embed_proj(
                torch.cat([last_token_embed, h], dim=-1)
            )
            h = self.eagle_head.transformer_layer(combined)
            logits = self.eagle_head.lm_head(h)
            token = sample(logits)
            tokens.append(token)
            last_token_embed = self.target_model.embed(token)

        return tokens
```

**优点**：接受率高（70-90%），内存开销小
**缺点**：需要专门训练的 EAGLE head，且需要捕获 target 的 hidden states

### 3. MTP/NextN（内置投机头）
DeepSeek V3 等模型自带多 token 预测头（详见 `02_model_specifics/02_deepseek_v3/04_mtp.md`）：

```python
class MTPSpeculator:
    def __init__(self, target_model):
        self.target_model = target_model
        self.mtp_heads = target_model.mtp_heads  # 模型自带

    def draft(self, hidden_states, last_token_embed):
        tokens = []
        h = hidden_states

        for head in self.mtp_heads:
            combined = head.eh_proj(torch.cat([last_token_embed, h], dim=-1))
            h = head.norm(combined)
            h = head.transformer_layer(h)
            logits = head.lm_head(h)
            token = sample(logits)
            tokens.append(token)
            last_token_embed = self.target_model.embed(token)

        return tokens
```

**优点**：无需额外模型，与 target 紧密集成
**缺点**：仅特定模型支持（DeepSeek V3 系列）

## Tree Attention（树状投机）

高级优化：不是线性生成 draft token，而是生成一棵 token 树，一次验证多条候选路径：

```python
# 树结构示例（EAGLE v2）
#         root
#        /    \
#     tok_a   tok_b
#    /  \       |
# tok_c tok_d  tok_e

# 构建 tree attention mask
def build_tree_mask(tree_structure):
    """构建因果注意力掩码，允许同层 token 间不互相注意"""
    n = tree_structure.num_nodes
    mask = torch.zeros(n, n, dtype=torch.bool)
    for node in tree_structure:
        # 每个节点只能看到自己和祖先节点
        for ancestor in node.ancestors():
            mask[node.id, ancestor.id] = True
    return mask
```

## KV Cache 管理

### Draft Token 的 KV 处理

```python
def manage_spec_kv_cache(accepted_tokens, draft_kv_entries, target_kv_cache):
    # 接受的 token：保留其 KV cache 条目
    for i, token in enumerate(accepted_tokens):
        target_kv_cache.keep(draft_kv_entries[i])

    # 被拒绝的 token 及其后续：释放 KV cache
    for i in range(len(accepted_tokens), len(draft_kv_entries)):
        target_kv_cache.free(draft_kv_entries[i])
```

### 关键问题：KV 不一致
Draft 模型和 target 模型的 KV cache 是分开的。验证后：
- Target 模型需要用自己的 KV cache，而不是 draft 的
- 接受的 token 的 KV 来自 target 的 verify forward pass
- 需要确保 target 的 KV cache 正确更新

## 调度器变化

```python
class SpeculativeScheduler(BaseScheduler):
    def schedule(self):
        batch = super().schedule()
        if batch.is_decode:
            # 为每个 decode 请求预留 N 个额外 token 的 KV 空间
            for req in batch.reqs:
                self.reserve_kv_slots(req, num_spec_tokens)
        return batch

    def postprocess(self, batch, accepted_tokens_list):
        for req, accepted in zip(batch.reqs, accepted_tokens_list):
            # 一次性添加多个 token（而不是通常的 1 个）
            for token in accepted:
                req.append_token(token)
            # 释放未使用的预留空间
            unused = self.num_spec_tokens - len(accepted) + 1
            self.release_kv_slots(req, unused)
```

## 集成到生成代码的方式

### 修改推理主循环

标准推理循环：
```python
while not finished:
    batch = scheduler.schedule()
    tokens = model_runner.run(batch)        # 1 token per request
    scheduler.postprocess(batch, tokens)
```

投机解码循环：
```python
while not finished:
    batch = scheduler.schedule()
    if batch.is_decode:
        # Draft 阶段
        draft_tokens = speculator.draft(batch, num_spec_tokens)
        # Verify 阶段（将 draft tokens 作为 prefill 输入）
        verify_logits = model_runner.run_verify(batch, draft_tokens)
        # Accept/Reject
        accepted = accept_reject(draft_tokens, verify_logits)
        scheduler.postprocess_speculative(batch, accepted)
    else:
        # Prefill 正常处理
        tokens = model_runner.run(batch)
        scheduler.postprocess(batch, tokens)
```

### 需要修改的组件

| 组件 | 需要的修改 |
|------|-----------|
| Model Runner | 添加 `run_verify()` 方法，支持多 token 验证 forward |
| Scheduler | 预留额外 KV 空间，支持批量 append token |
| KV Cache | 支持批量分配/释放，回滚未接受的 token |
| Model | 暴露 hidden states（EAGLE），或加载 MTP head |
| Sampler | 实现 acceptance criterion（greedy 或 rejection sampling） |

### 配置参数
```python
@dataclass
class SpeculativeConfig:
    method: str = "none"           # "none" | "eagle" | "draft_model" | "mtp"
    num_spec_tokens: int = 5       # 每步投机的 token 数
    draft_model_path: str = ""     # 外部 draft 模型路径
    eagle_head_path: str = ""      # EAGLE head 权重路径
    use_tree_attention: bool = False  # 是否使用树状投机
```

## 源码参考

| 项目 | 关键文件 |
|------|---------|
| sglang | `srt/speculative/eagle_worker.py` |
| sglang | `srt/speculative/eagle_utils.py` |
| sglang | `srt/models/deepseek_nextn.py` |
| vllm | `v1/spec_decode/` |
