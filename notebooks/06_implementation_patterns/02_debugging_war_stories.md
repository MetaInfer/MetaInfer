# 调试战报：Qwen3.5 混合注意力引擎实现中的 15 个 Bug

本文档记录了从零实现 Qwen3.5 Dense (QwenPaw-Flash-2B) 推理引擎过程中遇到的全部 bug。模型架构为 24 层混合注意力（18 层 Gated DeltaNet 线性注意力 + 6 层 softmax 全注意力），目标平台为 Apple Silicon MPS。

这些 bug 按类别可分为：权重加载错误、算子实现偏差、采样逻辑错误、平台特定问题、以及架构理解偏差。每个 bug 都包含症状、根因和修复方法，作为未来实现类似模型时的检查清单。

---

## 一、权重加载错误（Bug #7, #8, #9）

### Bug #7: linear_attn 权重前缀错误

**症状**：线性注意力层的权重全部未加载（random init），模型输出完全随机。

**根因**：safetensors 文件中线性注意力层使用 `linear_attn` 前缀（如 `linear_attn.in_proj_qkv.weight`），而代码中假设所有注意力层都用 `self_attn` 前缀。

**修复**：权重加载映射表增加 `linear_attn` 前缀分支，根据 `layer_types[i]` 选择正确前缀。

**教训**：不同类型的注意力层在 checkpoint 中可能使用不同的命名前缀。加载权重前先用 `safe_open` 检查实际的 key 名称。

### Bug #8: 单 shard 加载失败

**症状**：当模型只有单个 `model.safetensors`（无 `model.safetensors.index.json`）时，权重加载崩溃。

**根因**：代码假设必然存在 `index.json` 文件并从中构建 `weight_map`，小模型通常没有这个文件。

**修复**：当 `index.json` 不存在时，直接从 `model.safetensors` 的 keys 构建 `weight_map`。

**教训**：永远处理"单 shard"和"多 shard"两种情况。小模型只有单 shard，大模型有 index 文件。

### Bug #9: 权重前缀检测不全

**症状**：模型是 ForConditionalGeneration 包装的（视觉+语言模型），checkpoint 中语言模型权重带有 `model.language_model.` 前缀，导致加载失败。

**根因**：`_detect_prefix` 函数只检查了 `""` 前缀，没有检查 `model.language_model.` 和 `model.` 两个常见前缀。

**修复**：`_detect_prefix` 增加 `model.` 前缀检测；`_w` 函数增加 fallback 逻辑。

**教训**：视觉语言模型的文本权重通常嵌套在 `model.language_model.` 下。加载前探测前缀，而非硬编码。

---

## 二、算子实现偏差（Bug #1, #4, #5, #6, #15）

### Bug #15: RMSNormGated 使用了错误的 weight 格式（关键修复）

**症状**：prefill logits 余弦相似度只有 0.845，逐层 hidden norm 比对发现 linear attention 层输出幅度偏差约 2 倍。

**根因**：`RMSNormGated` 错误地使用了 `(1 + weight) * x` 格式（与普通 layer RMSNorm 一致），但 HF transformers 中 `Qwen3_5RMSNormGated` 使用的是 `weight * x`（不加 1）。

**修复**：改为 `self.weight * x.to(input_dtype)`。

**影响**：修复后 prefill logits cosine similarity 从 0.845 提升到 >0.9999，所有 4 个测试 prompt 的 top-1 token 完全匹配。

**教训**：**同一模型中不同位置的 RMSNorm 可能使用不同的 weight 格式**。必须逐个检查 HF 源码中每种 norm 的实现，不能假设所有 norm 格式一致。这个 bug 影响范围极大——它导致所有 linear attention 层的输出幅度偏差 2 倍，进而导致所有后续层的 hidden states 发散。

### Bug #1: q_proj view 形状错误

**症状**：full attention 层 crash，维度不匹配。

**根因**：`attn_output_gate=True` 时，q_proj 输出 2x 维度（包含 gate 分支），view 时需要用 `head_dim * 2` 而非 `head_dim`。

**修复**：根据 `attn_output_gate` 标志调整 view 形状。

**教训**：带输出门控的注意力层，query 投影的输出维度是标准值的 2 倍。检查 config 中的 gate 标志。

### Bug #4: Conv1d 缺少 SiLU 激活

**症状**：线性注意力层输出异常，prefill logits 与 HF 不匹配。

**根因**：Gated DeltaNet 的因果卷积后需要 `F.silu()` 激活，代码中遗漏。

**修复**：在 conv1d 输出后添加 `F.silu()` 调用。

**教训**：Gated DeltaNet 的因果卷积不是简单的线性变换，参考实现中包含激活函数。逐行对照 HF 实现检查。

### Bug #5: QK L2 norm 缺失

**症状**：线性注意力输出幅度偏大。

**根因**：线性注意力需要对 Q、K 做 L2 归一化（`F.normalize`），HF 使用 `use_qk_l2norm_in_kernel=True`。

**修复**：在 kernel 内对 Q、K 做 L2 normalize。

**教训**：DeltaNet 系列架构通常对 QK 做归一化。检查 config 中的 `use_qk_l2norm_in_kernel` 标志。

### Bug #6: Query 缺少 `1/sqrt(d_k)` 缩放

**症状**：Gated DeltaNet 输出幅度偏大。

**根因**：需要在 query 上乘以 `1 / sqrt(key_head_dim)`，与 HF 的 `scale = 1 / (k_head_dim ** 0.5)` 对齐。

**修复**：添加 `q = q * self.scale`。

**教训**：注意力机制中的缩放因子容易被遗漏。标准 softmax attention 自动包含 `1/sqrt(d)` 缩放，但线性注意力的 kernel 需要手动添加。

---

## 三、状态管理错误（Bug #2, #3）

### Bug #2: kwargs 解包导致状态丢失

**症状**：decode 循环中 recurrent state 不更新，模型输出退化。

**根因**：`**kwargs` 解包创建新字典，layer 内对 `kwargs["recurrent_state"]` 的赋值写入的是新字典，而非原始 states 列表中的引用。

**修复**：改为直接传递 dict 引用，不解包。layer 内直接操作 `kw["recurrent_state"]`。

**教训**：**永远不要用 `**kwargs` 解包传递可变状态**。Python 的 `**` 解包会创建新 dict，内部赋值不会传回。这是 Python 陷阱，不是模型 bug。

### Bug #3: torch.empty() 全零

**症状**：模型输出在 Mac 上异常，在某些随机种子下退化严重。

**根因**：Mac 上 `torch.empty()` 产生全零张量（而非随机值），导致初始 recurrent state 为零矩阵，特定输入下发散。

**修复**：添加 `_init_weights()` 方法，用正态分布初始化所有参数。

**教训**：**`torch.empty()` 不保证产生非零值**。在 Mac 上它通常产生全零。需要随机初始化的参数必须显式初始化。

---

## 四、采样逻辑错误（Bug #10）

### Bug #10: top_p_sample 返回排序索引而非 token ID

**症状**：模型输出乱码，生成的 token 完全不对。

**根因**：`torch.multinomial` 返回的是 sorted 数组中的位置索引，不是原始 vocab 中的 token ID。需要用 `sorted_indices.gather()` 映射回原始索引。当添加 top-k 预筛选后，还需要额外的 `top_k_indices.gather()` 映射。

**修复**：
```python
if top_k_indices is not None:
    local_idx = sorted_idx.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)
    return top_k_indices.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
return sorted_idx.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)
```

**教训**：采样函数中的索引映射是高频出错区域。top-k 预筛选引入了额外的映射层：原始 vocab → top-k 子集 → sorted 子集 → multinomial 采样 → 反向映射。每一层都必须正确反向映射。**采样函数的 bug 不会影响 logits 正确性验证（余弦相似度），但会导致生成结果完全错误。**

---

## 五、Tokenizer/API 错误（Bug #11）

### Bug #11: apply_chat_template 返回 BatchEncoding

**症状**：tokenizer 的 `apply_chat_template` 返回值传给模型时 crash。

**根因**：`apply_chat_template(tokenize=True)` 返回 `BatchEncoding` 对象，不是 `list[int]`。需要用 `result["input_ids"]` 提取 token ID 列表。

**修复**：`token_ids = result["input_ids"]`。

**教训**：HuggingFace tokenizer API 的返回类型不直观。`tokenize=True` 时不返回 `list[int]`，而是 `BatchEncoding`。检查返回类型。

---

## 六、Thinking Token 处理（Bug #13, #14 — 负优化，已回退）

### Bug #13: 截断 thinking tokens 失败

**尝试**：在 chat template 末尾注入 `<think\>\n\n</think\>\n\n` 让模型跳过思考。

**失败原因**：模型在 prompt 中看到 `<think\></think\>` 后会跳过思考直接回答，但如果去掉 prompt 中的 thinking tokens，模型会自己生成 `<think\>` 进入 thinking mode，导致输出全是思考文本。

### Bug #14: skip_thinking 过滤失败

**尝试**：在 decode 循环中添加 skip_thinking 过滤，等看到 `</think\>` token 后才开始输出。

**失败原因**：
1. 模型在 prompt 中看到 `<think\></think\>` 时跳过思考直接回答，第一个 token 就不是 `</think\>`，thinking_done 永远为 False
2. 去掉 prompt 中的 thinking tokens 后，模型会自己生成 thinking 内容，但 max_tokens 在 thinking 阶段也被消耗完，导致输出为空

**结论**：保留原始 chat template，不做任何 thinking token 相关的特殊处理。模型的 thinking 行为由 chat template 和采样参数自然控制。

**教训**：**不要试图 hack 模型的思考过程**。chat template 是模型训练时固定的输入格式，修改它会破坏模型的预期行为。

---

## 七、精度问题（Bug #12）

### Bug #12: Gated DeltaNet recurrent state float32 计算

**症状**：MPS float16 下 decode 超过约 30 步后输出退化（重复 `**`、循环文本、乱码）。

**根因**：Gated DeltaNet 的 recurrent state 在 float16 下累积误差。每步的递归更新涉及多次乘法（衰减门控、beta、kv 外积），float16 的精度不足以支撑长序列。

**修复**：`torch_recurrent_gated_delta_rule` 内部升级为 float32 计算，recurrent state 初始分配为 float32，输出前转回模型 dtype (fp16)。

**影响**：长文本输入用例质量明显提升（从完全循环变成有意义的摘要）。

**教训**：**线性注意力的 recurrent state 必须使用 float32**。这与 KV cache 不同——KV cache 存储的是原始 K/V 向量（不涉及累积乘法），而 recurrent state 是累积的矩阵乘积，对精度敏感。这是线性注意力与 softmax 注意力在工程实现上的关键区别。

---

## 检查清单：实现新模型时的必检项

1. **权重前缀探测**：用 `safe_open` 检查实际 key 名称，处理 `model.language_model.`、`model.`、`""` 三种前缀
2. **RMSNorm 格式**：逐个检查每种 norm 的 weight 格式（`weight * x` vs `(1 + weight) * x`）
3. **单 shard / 多 shard**：两种情况都要处理
4. **注意力缩放**：检查是否需要 `1/sqrt(d_k)` 和 QK L2 norm
5. **门控维度**：带 output gate 的注意力层，Q 投影输出 2x 维度
6. **卷积激活**：检查因果卷积后是否有激活函数
7. **状态传递**：不要用 `**kwargs` 解包传递可变状态
8. **张量初始化**：`torch.empty()` 在 Mac 上产生全零，需要显式初始化
9. **采样索引映射**：top-k + top-p 采样需要三层反向映射
10. **Tokenizer 返回类型**：`apply_chat_template(tokenize=True)` 返回 `BatchEncoding`
11. **Recurrent state 精度**：线性注意力的 recurrent state 使用 float32
12. **Chat template 完整性**：不要修改模型训练时的 chat template
