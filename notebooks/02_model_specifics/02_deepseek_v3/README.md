# DeepSeek V3 模型核心形态分析

## 概述

DeepSeek V3 是 DeepSeek 团队发布的大型语言模型，其在推理框架中的实现涉及多项创新技术。本文档基于 vLLM 和 SGLang 两个推理框架的代码实现，抽象出该模型的核心形态。

## 模型版本

| 版本 | 特点 | 发布时间 |
|------|------|----------|
| DeepSeek V2 | 首次引入 MLA 和 MoE | 2024.05 |
| DeepSeek V3 | 优化 MLA，增强 MoE | 2024.12 |
| DeepSeek V3.1 | 增加推理内容分离 | 2025.03 |
| DeepSeek V3.2 | 引入 NSA (Native Sparse Attention) | 2025.06 |
| DeepSeek R1 | 推理模型，基于 V3 | 2025.01 |

## 核心架构特点

DeepSeek V3 的核心创新包括：

1. **Multi-Head Latent Attention (MLA)**：通过低秩压缩减少 KV Cache 大小
2. **DeepSeek MoE**：混合专家架构，包含共享专家和路由专家
3. **Multi-Token Prediction (MTP)**：多token预测，用于投机解码加速
4. **Native Sparse Attention (NSA)**：V3.2 引入的稀疏注意力机制

## 文档目录

- [01_architecture.md](./01_architecture.md) - 整体架构设计
- [02_mla_attention.md](./02_mla_attention.md) - MLA 注意力机制
- [03_moe.md](./03_moe.md) - MoE 混合专家实现
- [04_mtp.md](./04_mtp.md) - MTP 投机解码
- [05_nsa.md](./05_nsa.md) - NSA 稀疏注意力（V3.2）
- [06_optimization_patterns.md](./06_optimization_patterns.md) - 推理优化模式

## 模型参数规模

```python
# DeepSeek V3 关键参数
hidden_size = 7168          # 隐藏层维度
num_layers = 61             # 层数
vocab_size = 129280         # 词表大小

# MLA 参数
num_attention_heads = 128   # 注意力头数
qk_nope_head_dim = 128      # QK 非旋转部分维度
qk_rope_head_dim = 64       # QK 旋转位置编码维度
v_head_dim = 128            # V 头维度
kv_lora_rank = 512          # KV 低秩压缩维度
q_lora_rank = 1536          # Q 低秩压缩维度

# MoE 参数
n_routed_experts = 256      # 路由专家数量
n_shared_experts = 1        # 共享专家数量
num_experts_per_tok = 8     # 每个token激活的专家数
moe_intermediate_size = 2048  # 专家中间层维度
n_group = 8                 # 专家分组数
topk_group = 4              # 每组选择的top-k

# MTP 参数 (DeepSeek V3)
num_nextn_predict_layers = 1  # MTP层数
```

## 框架实现对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| MLA 后端 | FlashMLA, FlashInfer, Triton | FA3, FlashInfer, FlashMLA, CutlassMLA, TRTLLM |
| MoE 后端 | DeepGEMM, Cutlass, Triton | DeepGEMM, DeepEP, FlashInfer |
| MTP 支持 | ✅ | ✅ |
| NSA 支持 | ✅ (V3.2) | ✅ (V3.2) |
| EP 支持 | ✅ | ✅ |
| DP Attention | ✅ | ✅ |

## 硬件需求

| 配置 | GPU 数量 | 说明 |
|------|----------|------|
| FP8 | 8×H200 / 8×B200 / 8×MI300X | 推荐 |
| BF16 | 16×H200 / 16×MI300X | 需要更多显存 |
| INT8 | 16×A100 / 32×L40S | 量化版本 |
| W4A8 | 8×H20 / 4×H200 | 高度量化 |
