# Task11 DeepSeek TP 真实测试调试经验

本文记录 `tests/test_deepseek_tp_real.py` 的一次完整排查过程，目标是解释“输出退化/乱码”的来源，并给出可复现的修复与验证步骤。

## 1. 问题现象

- 原始 `torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s` 输出中：
  - `output[0]` 有短句重复。
  - `output[1..4]` 大量换行或退化文本。
- 表面上像两类问题：
  1. RoPE/位置编码实现偏差；
  2. 某些 rank（1/2/3）没有正确加载权重。

## 2. 先排除“rank1/2/3 未加载权重”假设

做了 TP 与 HF 的同 prompt logits 对比（4 卡各自打印）后发现：

- 各 rank 的 `max_diff/mean_diff` 完全一致；
- 各 rank 的 top-1 token 也一致；
- 说明不是“某张卡权重缺失”导致的 rank 间随机崩坏。

结论：这更像是**实现层面的系统性偏差**（attention/RoPE/数值路径），而不是个别 rank 权重没拷贝进去。

## 3. 本轮代码修复

### 3.1 RoPE 数值路径改为 fp32 计算三角函数

文件：`engine/models/deepseek_v2.py`

- 原实现在旧版 RoPE 辅助函数中把 `emb` 先 cast 到 bf16 再做 `cos/sin`。
- 改为：
  - 频率/相位在 fp32 上算 `cos/sin`；
  - 再 cast 回激活 dtype。

作用：降低 bf16 下 RoPE 相位误差累积导致的重复/退化风险。  
**后续**纯 TP 的完整修复（**YaRN + GPT-J 式 rotate**）见第 8 节。

### 3.2 增加 DeepSeek TP 测试的 HF 调试开关

文件：`tests/test_deepseek_tp_real.py`

- 新增环境变量开关：
  - `DEEPSEEK_TP_HF_DEBUG=1` 时，调用
    `load_weights_from_hf_model(..., use_hf_logits_debug=True)`。

作用：快速验证“多卡流程与生成回路”是否正确，把问题隔离到 TP 数值对齐本身。

### 3.3 修复 HF-debug 多卡广播死锁与设备问题

文件：`engine/models/deepseek_v2.py`

- 修复点 1：rank0 的 HF 模型强制放到当前 GPU 设备（避免 CPU embedding + CUDA index 报错）。
- 修复点 2：HF-debug 分支让**所有 rank 都参与 broadcast**：
  - rank0 计算 logits；
  - 其他 rank 创建同形状 tensor；
  - 共同执行 `torch.distributed.broadcast`。
- 修复点 3：广播前将 rank0 logits cast 到统一 dtype（避免 dtype 不一致造成同步异常）。

## 4. 验证命令与结果

命令（建议显式 master port，避免 29500 冲突）：

```bash
DEEPSEEK_TP_HF_DEBUG=1 torchrun --master_port 29631 --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s
```

观察到的关键输出（rank0）：

- `output[0]='苏州园林...追求自然美...'`
- `output[1]='张量并行是一种分布式计算技术...'`
- `output[2]='夏天傍晚，夕阳如火...'`
- `output[3]='大语言模型是一种基于大量文本数据训练...'`
- `output[4]='面包是一种非常受欢迎的食品...'`

并出现：

- `1 passed`

说明在 HF-debug 广播路径下，结果已恢复为正常中文语义输出。

## 5. 结论与后续建议

1. **已确认**：不是 rank1/2/3 权重未加载导致的随机乱码。  
2. **HF-debug 真值广播**可作为链路与通信的回归基线（不用于交付，只用于排障）。  
3. **纯 TP 前向**已在 `engine/models/deepseek_v2.py` 中补齐 **YaRN + 与 vLLM DeepSeek 一致的 RoPE 风格** 后，在不开启 `DEEPSEEK_TP_HF_DEBUG` 的情况下，多卡真实测试即可得到正常中文语义输出（见第 8 节）。  
4. 后续若需与 HF 逐 token logits 严格一致，仍可按层做 hidden/logits diff；一般工程验收以「纯 TP 可读输出 + 稳定复现」为先。

## 6. 常用排障命令备忘

```bash
# 常规 TP 真实路径
torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s

# HF-debug 基线路径（推荐带 master_port）
DEEPSEEK_TP_HF_DEBUG=1 torchrun --master_port 29631 --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s
```

## 7. Agent 可复用经验补充（现象 -> 真因 -> 解法）

这部分用“下次写代码时可直接执行”的 Agent 语言描述。

### 7.1 这类现象通常意味着什么

- 现象 A：`output[0]` 有短句重复  
  - 高概率是**数值/位置相关路径有系统偏差**（RoPE、Norm、attention scale、mask/position 语义），而不是“模型完全坏掉”。
- 现象 B：`output[1..4]` 大量空行/退化  
  - 常见于 logits 分布被压扁或错位，采样/argmax 落到异常高频 token（如换行）区域。
- 现象 C：不同 rank 看起来都在跑，但输出质量差  
  - 不能直接判断“某个 rank 没加载权重”，必须先做**量化证据**。

### 7.2 我是如何排除伪根因的（证据链）

1. **先验证 rank 问题是否成立**  
   - 做 4 卡 rank 级别的 `HF vs TP` logits 比对（同 prompt、同位置）。  
   - 结果：各 rank `max_diff/mean_diff` 一致、top-1 一致。  
   - 结论：不是 rank1/2/3 掉权重；属于系统性偏差。

2. **再验证“框架链路”是否正常**  
   - 增加 `DEEPSEEK_TP_HF_DEBUG=1`，让 rank0 走 HF 真值 logits，广播给其余 rank。  
   - 若此时输出恢复正常中文语义，则说明：  
     - 分布式通信主链路可用；  
     - 解码循环可用；  
     - 问题集中在“纯 TP 前向数值对齐”。

3. **定位到高风险子系统**  
   - 优先查 RoPE、Norm、attention 数值路径；不要优先猜测调度器/采样器。

### 7.3 本轮“真正原因”在工程上怎么表述

- 本轮确认的主因不是“未加载权重”，而是**纯 TP 前向与 HF 存在系统性数值偏差**。  
- 已确认并修的直接问题：
  - RoPE 三角函数在 bf16 下过早计算，导致相位精度损失风险；
  - HF-debug 分支的多卡广播协议不完整（非 rank0 不参与会卡死）；
  - rank0 HF 模型设备与输入设备不一致（CPU/GPU 混用错误）。
- 这些问题修完后，HF-debug 路径恢复正确输出，证明调试基线建立成功。

### 7.4 下次 Agent 应该怎么做（执行清单）

当再次遇到“TP 输出重复/空行/乱码”时，按以下固定顺序：

1. **先做证据，不先猜**
   - 跑 rank 级 `HF vs TP` logits diff（max/mean/top1）。
2. **建立 HF-debug 基线**
   - 增加开关让 rank0 用 HF logits 广播，确认链路是否通。
3. **修协议问题**
   - 广播必须所有 rank 参与；dtype 与 shape 必须一致；模型与输入必须同 device。
4. **修数值问题**
   - RoPE 的 `freq/cos/sin` 先 fp32 再 cast；
   - Norm/softmax 与 HF 精度策略保持一致。
5. **最后才深挖模型实现差异**
   - DeepSeek 的 `rope_scaling/yarn`、attention 细节、逐层 hidden diff。

### 7.5 对未来代码生成的约束（给 Agent 自己）

- 不要把“文本退化”直接归因为某个 rank 权重未加载。  
- 先建立“HF 真值广播”可用基线，再做纯 TP 收敛。  
- 没有逐层对齐证据前，不要声称“TP 已正确实现”。  
- 每次修复后都要保留可复现命令与对照输出，确保下次可自动回归。

## 8. 纯 TP 修复点（尤其 YaRN）

本节记录**不依赖 HF 前向**、仅自研 TP 路径上的关键修复，供下次写 `DeepseekForCausalLMTP` 或同类带 `rope_scaling` 的模型时直接对照。

### 8.1 根因摘要

- DeepSeek V2 的 `config.json` 中 `rope_scaling.type` 为 **`yarn`**，并带 `factor`、`beta_fast`、`beta_slow`、`original_max_position_embeddings`、`mscale`、`mscale_all_dim` 等字段。  
- 早期自研实现若按「标准 Neox RoPE + 仅 `rope_theta`」写，**未应用 YaRN 的频率插值与外推混合**，会导致 Q/K 相位与预训练权重不匹配，表现为短句重复、大量换行、乱码状输出。  
- 另外，vLLM / DeepSeek 对 `qk_rope_head_dim` 上的旋转采用 **`is_neox_style=False` 的 GPT-J 式交错旋转**（`rotate_gptj`），与「对半拆再 cat」的 Neox 不同；混用会进一步错位。

### 8.2 代码落点（文件：`engine/models/deepseek_v2.py`）

1. **配置**  
   - 在 `DeepseekV2TPConfig` 中增加 `rope_scaling: dict | None`，在 `_load_deepseek_v2_tp_config` 中从 HuggingFace `AutoConfig` 读入 `cfg.rope_scaling`。

2. **YaRN 的 `inv_freq`**  
   - 当 `rope_scaling.get("type") == "yarn"` 时，对 `qk_rope_head_dim` 的偶数维构造 `1/(theta^(2i/d))`，再与 HF/vLLM 同类逻辑一致地做：
     - 插值支路 `1 / (factor * pos_freqs)` 与外推支路 `1/pos_freqs` 的**线性混合**（`yarn_find_correction_range` + `yarn_linear_ramp_mask` 思想）。

3. **YaRN 的幅值缩放**  
   - 对 `cos/sin` 乘上 `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)`（与 vLLM `DeepseekScalingRotaryEmbedding` 一致）。

4. **Attention 的 `scaling`**  
   - 在 `rope_type` 为 yarn 时，对 `self.scaling = (qk_head_dim ** -0.5)` 再乘 **`yarn_get_mscale(factor, mscale_all_dim) ** 2`**（对齐 vLLM `DeepseekV2Attention` 中对 `mscale` 的用法）。

5. **旋转算子**  
   - 对 `q_pe`、`k_pe` 使用 **GPT-J 式** `rotate_half` + `cos/sin` 的 `repeat_interleave(2, dim=-1)` 展布，而不是 Neox 的 `cat(freqs, freqs)` + 对半旋转。  
   - `cos/sin` 仍在 fp32 上算，再 `to(activation dtype)`，避免 bf16 下三角函数过糙。

### 8.3 纯 TP 验证命令（勿设 `DEEPSEEK_TP_HF_DEBUG`）

```bash
torchrun --master_port 29641 --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -q -s
```

（`--master_port` 可按环境避免 29500 占用的端口；与具体端口无关。）

### 8.4 Agent 备忘（一句话）

- 看见 **`rope_scaling: yarn`** 且输出像「能跑但像坏模型」，**先按 HF/vLLM 的 DeepSeek YaRN + 正确 rotate 风格改 RoPE**，再谈 MoE/EP 调优。  

