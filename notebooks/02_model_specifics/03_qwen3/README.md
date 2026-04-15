# Qwen3 模型架构知识

## 概述

Qwen3 是通义千问系列的第三代大语言模型，包含 **Dense**（稠密）和 **MoE**（混合专家）两种变体。Qwen3 在 Qwen2 基础上引入了 **QK Norm**（对 Query 和 Key 施加 RMSNorm），这是其与 Llama/Qwen2 系列的核心区别。

## 模型家族

| 模型 | 类型 | 参数量 | Layers | Heads | KV Heads | Hidden | Intermediate | 备注 |
|------|------|--------|--------|-------|----------|--------|-------------|------|
| Qwen3-0.6B | Dense | 0.6B | 28 | 16 | 8 | 1024 | 3072 | GQA |
| Qwen3-1.7B | Dense | 1.7B | 28 | 16 | 8 | 2048 | 6144 | GQA |
| Qwen3-4B | Dense | 4B | 36 | 32 | 8 | 2560 | 9728 | GQA |
| Qwen3-8B | Dense | 8B | 36 | 32 | 8 | 4096 | 12288 | GQA |
| Qwen3-14B | Dense | 14B | 40 | 40 | 8 | 5120 | 17408 | GQA |
| Qwen3-32B | Dense | 32B | 64 | 64 | 8 | 5120 | 27648 | GQA |
| Qwen3-30B-A3B | MoE | 30B/3B active | 48 | 32 | 4 | 2048 | - | 128 experts, top-8 |
| Qwen3-235B-A22B | MoE | 235B/22B active | 94 | 64 | 4 | 5120 | - | 128 experts, top-8 |

## 核心架构差异

| 特性 | Llama 3 | Qwen2 | Qwen3 Dense | Qwen3 MoE |
|------|---------|-------|-------------|-----------|
| QK Norm | 无 | 无 | **有** (RMSNorm) | **有** (RMSNorm) |
| Attention Bias | 无 | 有 | **无** | **无** |
| MLP | SwiGLU | SwiGLU | SwiGLU | MoE + SwiGLU per expert |
| RoPE theta | 500,000 | 1,000,000 | **1,000,000** | **1,000,000** |
| Norm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| Shared Experts | N/A | N/A | N/A | 可选（部分模型有） |

## 子文档

- [Dense 变体](01_dense.md) — 稠密模型的完整结构、权重映射和代码生成模板
- [MoE 变体](02_moe.md) — 混合专家模型的专家路由、权重结构和集成要点
