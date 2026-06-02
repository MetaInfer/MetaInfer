# Task10 TP 调试经验（Qwen TP=4）

本文记录本轮 `Qwen TP` 真实联调中出现的问题、根因、修复方式，供后续 Agent 生成 TP 组件时复用。其中 **§3.2/§3.3**（RoPE `rotate_half`、RMSNorm fp32）是本轮**短句级重复/文本退化**在权重与分片都正确后仍与 HF 不一致的关键；**§6** 为迁移到 **DeepSeek 类 MoE** 与「一次生成全框架」时的检查清单。

## 1. 错误一：Embedding 分片加载 shape=0

- **现象**
  - 报错：`The size of tensor a (...) must match the size of tensor b (0)`。
  - 发生在 `VocabParallelEmbedding.load_weight_shard`。
- **根因**
  - 在 `qwen.py` 的 `_load_tensor(..., split_dim=0)` 已经按 `tp_rank` 取了本地词表 shard。
  - `load_weight_shard` 又按 `vocab_start:vocab_end` 二次切片，rank>0 会切成空张量。
- **修复**
  - `load_weight_shard` 增加分支：
    - 若传入张量首维已等于 `local_vocab_size`，直接 copy；
    - 否则再按全量词表范围切片。
- **经验**
  - 对 TP 参数加载，必须明确“传入的是全量权重还是本地分片”；不要在两个层次重复切分。

## 2. 错误二：Qwen 配置触发 DeepSeek MLA 字段访问异常

- **现象**
  - 报错：`AttributeError: 'QwenTPConfig' object has no attribute 'qk_nope_head_dim'`。
  - 出现在 KV 预算估算路径。
- **根因**
  - `hf_deepseek_v2_kv_bytes_per_token` 早期只按 DeepSeek MLA 公式写死字段。
  - Qwen 是 Dense/GQA 配置，不具备 `qk_nope_head_dim/qk_rope_head_dim/v_head_dim`。
- **修复**
  - 在 `engine/kv_specs.py` 增加 Dense/GQA fallback：
    - 使用 `num_key_value_heads` + `head_dim` 估算 K/V bytes。
  - 在 `RealModelRunner` 打印配置时，区分 MLA 与 Dense/GQA 两种日志格式。
- **经验**
  - KV 估算函数应做“模型族分支”而非“单模型硬编码”。

## 3. TP 输出短句/片段重复、文本退化

短句反复、同义碎片刷屏，在 TP 上常见且**多源**：既有数据路径（mask/position）问题，也有**自研前向与 HuggingFace 同一模型族在“局部算子”上的细微不一致**。下面分两条子经验记录。

### 3.1 左 padding 批 + 与 HF 不一致的 position/mask 语义

- **现象**
  - TP 输出出现大量重复短语（如“苏州园林的特色”反复）。
- **根因**
  - 早期 TP runner 用左 padding 批处理，但自定义 Qwen forward 没有完整对齐 HF 的 `attention_mask/position_ids` 语义。
  - 导致注意力上下文被污染，表现为重复、退化文本。
- **修复**
  - 在 `QwenTPModelRunner.run` 中改为**逐序列前向**（每条样本单独输入，不做左 padding 批拼接），再取最后位 logits 采样。
  - 虽然更慢，但可先保证与 HF baseline 的语义一致性。
- **经验**
  - TP 正确性阶段优先“逐序列无 padding”的保守路径，性能优化（batched + mask）放在后续阶段。

### 3.2 RoPE 的 `rotate_half` 与 HF 不一致（Qwen3 / Llama 系「前后半维」 vs 错用「奇偶维」）

- **现象**
  - 在 mask/已对齐、权重加载无二次切片后，**TP=4 与 HF 单卡**仍出现明显差异：同 prompt 下输出含**短句级重复、无意义排比**、与单卡通顺回答不一致。
- **根因**
  - `transformers` 中 **Qwen3** 的 `rotate_half` 为：最后一维**从中间对半**拆成 `x1, x2`，再 `cat((-x2, x1))`（与 Llama/多数开源实现一致）。
  - 自研版若错写成**奇偶维交错**（`x[...,::2]`, `x[...,1::2]` 再 stack/flatten），与 Qwen3 的 `cos/sin` 展布 `cat([freqs, freqs], dim=-1)` **不配套**，等效于整段注意力**相位/位置编码整层错位**，不会简单表现为「整体 logits 一个常数差」，而常表现为**退化分布 → 重复抽局部高频片段**。
- **修复**
  - 在 `engine/models/qwen.py` 中将 `_rotate_half` 与 HF `modeling_qwen3` 逐行对齐；`_apply_rope` 在关闭 autocast 的 float 精度路径上计算 `emb`，再 `to` 回激活 dtype；若存在二维 `position_ids` 需与 `Qwen3RotaryEmbedding` 的 `inv_freq.expand` + matmul 行为一致（见同文件内实现）。
- **经验**
  - **自研前向时，RoPE 必须与目标 checkpoint 的 HF 类实现一致**：`rotate_half` 是「对半转」还是「交错转」依模型族而异；不能从别的仓库抄一段 RoPE 就当通用。
  - 调试时优先对照 **`transformers.models.<model_name>.modeling_***` 的 `apply_rotary_pos_emb` 与 `RotaryEmbedding.forward`，用同一随机 `q` 在 CPU 上对比 `apply_rope` 后张量，而不是只盯 TP all_reduce。

### 3.3 RMSNorm 在 bf16 下用全半精度做 variance（与 Qwen3RMSNorm 不一致）

- **现象**
  - 与 3.2 叠加时进一步放大分布偏移；单独也可能导致 logits 漂移。
- **根因**
  - HF 的 `Qwen3RMSNorm` 在 **float32** 中算 `variance` 和 `rsqrt`，再与 `weight` 相乘，最后转回 `hidden_states` 的 dtype。
- **修复**
  - 自研 `RMSNorm` 与 HF 相同： `input_dtype` → `float32` → 归一化 → `* weight` → `to(input_dtype)`。
- **经验**
  - 对「已有权重」的推理路径，**Norm 的数值路径尽量与官方推理一致**；TP 只切矩阵，不擅自改 Norm 的精度假设，除非有完整基准测试。

## 4. NCCL error code 2/3 的解释与处理建议

- **现象**
  - 进程结束时看到 RCCL/NCCL `socket... error code 2/3`、`Abort COMPLETE`、`Proxy Service` 等日志。
- **解释**
  - 多数是测试进程退出时 communicator 异步 abort 的尾日志，不一定代表计算失败。
  - 若测试摘要为 `PASSED`，通常可视为退出期噪声。
- **建议**
  - 优先看 pytest 结果与主异常栈，不要仅凭 NCCL 尾日志判定失败。
  - 若需要减少噪声，可在测试尾部统一做进程组清理；但要避免在参数化多用例中反复销毁/重建导致额外网络不稳定。

## 5. 本轮验证结论

- `torchrun --nproc_per_node=4 -m pytest tests/test_qwen_tp_real.py -v -s` 在 Dense Qwen3 上可通过（历史上曾报 `2 passed`；以当前用例与模型路径为准）。
- 说明：
  - TP 分片加载链路可用；
  - 关键探针（权重 shape/device、logits device/shape、每 rank 显存）正常输出；
  - **HF 单卡**仍应作为**真值参考**，逐层或至少首层/首 token logits 做 diff，再开 batched+mask 等优化。

## 6. 可迁移：DeepSeek 类 MoE 的 TP 与「一次性拉满推理框架」的清单

本节的读者是**后续接手的 Agent/人**：在 Dense Qwen3 上踩过的坑，在 **MoE（如 DeepSeek）** 上会以「更多分支 + 更多权重名 + 每 token 多专家」形式再现；若用 Cursor 一次生成**含 TP 的整栈推理框架**，请把下面当作**必答检查项**，避免只实现 Column/Row 线性但忘记模型族特化算子。

### 6.1 从 Qwen3 Dense 直接继承的习惯

- **以 HF/官方 `modeling_*` 为单一真源**
  - 对每一种 attention、RoPE、RMS/LayerNorm、expert 路由、KV cache 布局，在仓库里**点名文件与函数**（如 `Qwen3Attention`、`apply_rotary_pos_emb`），自研前向**逐行对照**，而不是凭记忆从「通用 Llama」文章抄 RoPE。
- **Norm / softmax / scale 的 dtype 策略**
  - 与预训练/推理权重的**官方实现**一致（例如 Norm 在 fp32、attn_softmax 在 fp32 再 cast 回）；TP 不替代这份契约。
- **先跑通无 padding 单条序列**，再开 batch 与长上下文优化。

### 6.2 在 MoE 上**额外**必须想清楚的点

- **专家并行（EP）与张量并行（TP）的维度**
  - Dense 只有「列并 / 行并」切 `hidden` 与 `ffn`；MoE 还有 **router、gate、expert 权重**按专家维或隐藏维分片。需要明确：哪些算子**跟 TP rank**、哪些**跟 EP rank**、是否同一批进程复用多组 `process_group`。
- **路由与 all_to_all / all_gather**
  - 错误实现常表现为**部分 token 永远不走专家**或**专家输出重复累加**；比 Dense 的 `all_reduce` 更难肉眼验，要准备**小 batch、固定随机种子、对比单卡逐 token 路由索引**。
- **同一类「短句重复/乱码」症状**
  - 在 Dense 上根因常是 **RoPE/Norm/position**；在 MoE 上还要加查：**shared expert、device expert、是否误在推理中引入训练项（如 aux loss）** 等。仍可能表现为重复文本，要**按子系统排除**。

### 6.3 给「Cursor 一次生成完整 TP 框架」的提示词/结构建议

若希望少返工、一次生成**可联调**的骨架，建议显式要求生成物包含（可直接写进 `AGENTS.md` 或设计文档）：

1. **模型族真源**：`transformers` 中对应类的路径 + 须对齐的类列表（Attention、RMSNorm、Rotary、MoE 层等）。
2. **精度约定**：每类算子输入/中间/输出的 dtype（与 HF `forward` 中 `autocast` 行为一致，或标出**禁止 autocast 的区段**）。
3. **TP/EP 分组与通信原语表**：每模块后使用 `all_reduce` / `all_gather` / `reduce_scatter` / `all_to_all` 的哪一种、张量 shape 与行/列主序。
4. **权重加载契约**：`safetensors` 键名、是否已分片、**禁止重复切片**的约定（见本文 §1）。
5. **KV 与多模型族**：`kv_specs` 或等价层里对 **MLA / GQA / MHA 分支** 的表驱动（见本文 §2），避免一个公式套所有 checkpoint。
6. **最小验收**：`tp=1` 自研前向与 HF **同 prompt 下首 token 或整段 logits 对齐**；再 `tp=n` 与 `tp=1` **文本或 logits 一致**；最后才做吞吐。

> **一句话备忘**：TP 分片没写错、通信没死锁，仍可能「看起来能跑、生成像坏模型」——**RoPE/Norm/position 与 HF 差一个约定，在输出上会像灾难**；到 MoE 再叠**路由与 expert 通信**，没有单卡/单进程真值对齐习惯，调试成本会**指数级**上升。
