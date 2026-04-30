# DeepSeek V3 — 架构总览

## 与常规模型有何不同

DeepSeek V3 是面向生产的大语言模型，在标准 Transformer 之上引入多项结构创新。这些创新会显著影响推理框架的设计，要生成正确的推理代码就必须理解它们。

## 主要创新

| 创新点 | 常见做法 | DeepSeek V3 做法 | 对推理的影响 |
|--------|----------|------------------|--------------|
| Attention | 多头（MHA）或 GQA | 多潜注意力（MLA） | KV cache 更小，attention 内核不同 |
| MLP | 稠密 FFN 或简单 MoE | 细粒度 MoE + 共享专家 | 专家路由、all-to-all 通信 |
| 预测 | 单步下一 token | 多 token 预测（MTP） | 内建投机解码 |
| 长上下文 | 全量 attention | 原生稀疏 attention（NSA） | 稀疏选 token、次线性 attention |

## 结构示意

```
输入 Token
    ↓
[Embedding]
    ↓
[DecoderLayer × N]
    │
    ├── [RMSNorm]
    ├── [MLA Attention]  ← 替代标准 MHA/GQA
    ├── [残差]
    ├── [RMSNorm]
    ├── [MoE MLP]        ← 替代稠密 MLP
    │   ├── [共享专家]   ← 始终计算
    │   └── [路由专家，从 N 里选 K]  ← Top-K
    └── [残差]
    ↓
[RMSNorm]
    ↓
[LM Head]
    ↓
[MTP 头 × M]  ← 预测未来 token（可选，用于投机解码）
```

## 子文档

- [02_mla_attention.md](02_mla_attention.md) — 多潜注意力（MLA）机制
- [03_moe.md](03_moe.md) — 带共享专家的细粒度混合专家
- [04_mtp.md](04_mtp.md) — 多 token 预测与投机解码
- [05_nsa.md](05_nsa.md) — 长上下文下的原生稀疏注意力
- [06_optimization_patterns.md](06_optimization_patterns.md) — FP8 等优化与内核模式
