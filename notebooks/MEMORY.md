# 知识点索引

## 概述

本目录包含了LLM推理框架的设计方法论和知识点总结，用于指导AI Agent生成精简的推理框架。

## 目录结构

```
notebooks/
├── 00_overview/                    # 总体概述
│   └── README.md                   # 项目背景和知识点组织
│
├── 01_framework_design/            # 推理框架设计方法论（核心）
│   ├── 01_architecture.md          # 整体架构设计
│   ├── 02_scheduler.md             # 调度器设计
│   ├── 03_kv_cache.md              # KV Cache管理
│   ├── 04_model_runner.md          # 模型执行器
│   ├── 05_sampler.md               # 采样器
│   ├── 06_memory_pool.md           # 内存池管理
│   └── 07_request_lifecycle.md     # 请求生命周期
│
├── 02_model_specifics/             # 模型特定方法论
│   └── 01_transformer_models.md    # Transformer模型实现
│
├── 03_operators/                   # 算子设计
│   └── 01_attention_ops.md         # 注意力算子
│
├── 04_parallel_strategies/         # 并行策略
│   └── 01_tensor_parallel.md       # 张量并行
│
├── 05_non_core_features/           # 非核心功能（可抽离）
│   ├── 01_multi_model_support.md   # 多模型支持
│   ├── 02_multi_quantization.md    # 多量化支持
│   └── 03_platform_abstraction.md  # 平台抽象层
│
└── 06_implementation_patterns/     # 实现模式
    ├── 01_code_patterns.md         # 代码组织模式
    └── 03_anti_patterns.md         # 反模式（应避免）
```

## 核心发现

### 推理框架的本质

LLM推理框架的核心功能非常简单：

1. **请求管理**：接收请求，维护请求状态
2. **调度**：决定哪些请求参与当前推理步骤
3. **内存管理**：KV Cache的分配、共享、释放
4. **模型执行**：前向传播计算
5. **采样**：从logits生成token

### 复杂性来源

| 来源 | vLLM | SGLang | 影响 |
|------|------|--------|------|
| 模型支持 | 274+ | 90+ | 大量条件分支 |
| 量化方法 | 30+ | 25+ | 不同算子实现 |
| Attention后端 | 10+ | 15+ | 动态派发 |
| 并行策略 | TP/PP/DP/EP/CP | TP/PP/DP/EP | 分布式复杂性 |
| 平台支持 | CUDA/ROCm/TPU/XPU/CPU | CUDA/ROCm/NPU | 平台抽象层 |

### 精简原则

1. **单一模型**：只支持一种模型架构
2. **单一后端**：只使用一种Attention实现
3. **无平台抽象**：针对特定硬件直接实现
4. **最小配置**：只保留必要参数
5. **可选功能外置**：量化、LoRA等作为独立模块

## 使用指南

### 为Agent提供上下文

在让AI Agent生成推理框架时，可以提供以下上下文：

```
请参考以下知识点生成推理框架：
- 架构设计：notebooks/01_framework_design/01_architecture.md
- 调度器：notebooks/01_framework_design/02_scheduler.md
- KV Cache：notebooks/01_framework_design/03_kv_cache.md
- ...

用户需求：
- 模型：LLaMA-7B
- 硬件：NVIDIA GPU
- 特性：Paged Attention + 前缀缓存
```

### 知识点优先级

1. **必须阅读**（核心功能）：
   - 架构设计
   - 调度器
   - KV Cache管理
   - 模型执行器

2. **按需阅读**（特定需求）：
   - 张量并行（如果需要多GPU）
   - 模型实现（如果支持特定模型）

3. **参考阅读**（避免复杂性）：
   - 非核心功能（了解应该避免什么）
   - 反模式（了解不应该怎么做）

## 下一步工作

1. 基于知识点构建Agent提示模板
2. 开发自动生成推理框架的Agent系统
3. 持续完善和更新知识点
