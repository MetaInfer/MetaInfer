# 大语言模型推理框架知识库

## 目的

本知识库提炼自多个 LLM 推理框架（vLLM、SGLang、nano-vllm、nano-sglang、mini-sglang）中的共性模式与设计原则，作为方法论参考，供 AI Agent 系统根据具体部署需求生成**定制化、精简**的推理框架。

## 知识结构

### 核心框架设计（`01_framework_design/`）

构建任意 LLM 推理框架的通用方法论。无论具体模型或部署场景如何，每个推理系统都需要这些基本构件。

- **架构（Architecture）** — 系统拓扑、进程模型、组件关系
- **调度器（Scheduler）** — 连续批处理、prefill/decode 调度、抢占
- **KV 缓存（KV Cache）** — 分页注意力、radix/前缀缓存、内存块管理
- **模型执行器（Model Runner）** — 前向执行、CUDA Graph 捕获/回放
- **采样器（Sampler）** — 词元采样策略（贪心、top-k、top-p、temperature）
- **内存池（Memory Pool）** — GPU 内存分配、池管理、利用率估算
- **请求生命周期（Request Lifecycle）** — 从 HTTP 请求到生成响应的端到端流程

### 模型相关知识（`02_model_specifics/`）

不同模型家族的架构模式与实现细节。为特定模型生成代码时，请查阅对应子章节。

- **Transformer 模型** — 通用稠密 Transformer 模式（Llama、Qwen、Mistral）
- **DeepSeek V3** — MLA 注意力、MoE 路由、MTP（多词元预测）、NSA

### 算子（`03_operators/`）

性能关键路径上的内核级知识。

- **注意力算子** — Flash Attention、FlashInfer、Triton 注意力、分页 KV 访问
- （可扩展：RoPE、MoE 内核、归一化等）

### 并行策略（`04_parallel_strategies/`）

如何在多块 GPU 上划分计算。

- **张量并行（Tensor Parallelism）** — 列/行切分、all-reduce 模式、权重分片

### 非核心特性（`05_non_core_features/`）

生产级框架中常见、但可按部署需求**独立增删**的能力，分为两类：

**可剥离的复杂度**（极简框架可完全省略）：

- **多模型支持** — 模型注册表、动态派发
- **多种量化方式** — AWQ、GPTQ、FP8 等
- **平台抽象** — 多硬件（CUDA、ROCm、XPU、TPU）

**生产可选特性**（真实部署中很重要，各文档含集成指引）：

- **PD 分离** — Prefill 与 Decode 分离，便于独立扩缩容
- **KVCache Connector** — 跨节点 KV 缓存传输接口（NCCL、RDMA、Mooncake、NIXL）
- **后处理** — 增量 detokenization、停止串检测、流式输出、结果格式化
- **投机解码** — EAGLE、草稿模型、MTP/NextN 以降低延迟
- **约束解码** — JSON Schema、正则、语法约束生成（XGrammar、Outlines）

### 实现模式（`06_implementation_patterns/`）

各项目中的代码级模式与反模式。

- **代码模式** — 经实践验证、效果良好的实现方式
- **反模式** — 生成代码时应避免的复杂度陷阱

## 如何使用本知识库

1. **生成推理框架**：从 `01_framework_design/` 入手，理解核心架构；生成的框架都应包含这些组件。
2. **编写模型相关代码**：针对目标模型架构查阅 `02_model_specifics/`。
3. **追求性能**：参考 `03_operators/` 与 `04_parallel_strategies/` 做 GPU 侧优化实现。
4. **增加特性**：查看 `05_non_core_features/`。每个「生产可选特性」文档会说明在何处加钩子、改哪些组件、以及配置项；可按用户需求接入到已生成的框架中。
5. **保证代码质量**：遵循 `06_implementation_patterns/`，生成清晰、可维护的代码。

## 源码参考项目


| 项目          | 角色     | 说明                                                |
| ----------- | ------ | ------------------------------------------------- |
| nano-vllm   | 正向示例   | 约 20 个文件，实现面向 Qwen3 的 PagedAttention + 连续批处理      |
| nano-sglang | 正向示例   | 约 35 个文件，实现 RadixCache + 面向 Llama/Mixtral 的多进程服务  |
| mini-sglang | 正向示例   | 约 60 个文件，SGLang 官方精简版，含 CUDA 内核与 TP               |
| vllm        | 参考（复杂） | 270+ 模型、20+ 种量化、15+ 种注意力后端；PD 分离、KV 连接器等生产特性的重要来源 |
| sglang      | 参考（复杂） | 100+ 模型、投机解码、约束解码、多模态；EAGLE、XGrammar 等集成的重要来源     |
