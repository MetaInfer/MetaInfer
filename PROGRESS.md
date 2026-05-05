# 项目进度追踪

> 最后更新：2026-05-05

## 总览

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一：知识库建设 | 已完成 | 32 篇文档，覆盖框架设计、模型、算子、并行策略 |
| 阶段二：推理引擎开发 | 进行中 | CUDA TP 引擎已完成；Mac MPS 引擎基础可用 |
| 阶段三：Agent 自动生成系统 | 未开始 | 核心目标 |
| 阶段四：生成框架验证 | 未开始 | 依赖阶段三 |

---

## 阶段一：知识库建设

从参考项目中提取推理框架的共性知识，存入 `notebooks/`。

- [x] 框架设计方法论（架构、调度器、KV Cache、Model Runner、Sampler、内存池、请求生命周期）
- [x] 模型专项知识（DeepSeek V3、Qwen3）
- [x] 算子知识（Flash Attention、FlashInfer、Triton）
- [x] 并行策略（Tensor Parallelism）
- [x] 可选功能知识（PD 分离、投机解码等）
- [x] 实现模式（Code Patterns、Anti-Patterns）

---

## 阶段二：推理引擎开发

### 已完成

- [x] 核心组件（`engine/`）
  - [x] LLMEngine 入口（`llm_engine.py`）— HF 和 TP 双后端
  - [x] Scheduler — prefill 优先 + continuous batching
  - [x] Memory Pool — 分页 KV Cache
  - [x] Block Manager — 基于哈希的 prefix caching
  - [x] Sampler — greedy / top-p / temperature
  - [x] Model Runner + Structs
- [x] 张量并行（`engine/tp_layers/`）
  - [x] 列/行并行线性层、Vocab 并行 Embedding/LM Head
  - [x] MoE 专家并行
  - [x] 分布式通信原语
- [x] 模型实现（`engine/models/`）
  - [x] DeepSeek V2 — MLA + MoE TP
  - [x] Qwen3 — Dense + MoE TP
- [x] 部署与测试
  - [x] OpenAI 兼容 HTTP 服务
  - [x] 4 GPU 实际推理验证
- [x] Mac GPU (MPS) 引擎（`engine/mac_gpu/`）

### 待完成

- [ ] 性能优化
  - [ ] KV Cache 接入模型 forward pass（当前 use_cache=False 全量重计算）
  - [ ] Mac GPU 路径的推理性能优化
- [ ] 调度增强
  - [ ] 抢占（preemption）机制
  - [ ] 优先级调度

---

## 阶段三：Agent 自动生成系统

核心目标：基于知识库构建 Agent，根据用户需求自动生成精简的推理框架代码。

- [ ] 构建 Agent 提示模板（基于 notebooks 知识点）
- [ ] 开发 Agent 系统（接收用户需求 → 生成推理框架代码）
- [ ] 定义用户需求输入格式（目标模型、并行策略、部署方式等）
- [ ] 定义生成框架的输出规范（目录结构、代码风格）

### 已有基础

- `.claude/agents/` — 5 个 sub-agent（contract-checker、impl-coder、integration-verifier、ref-tracer、tdd-test-writer）
- `.claude/skills/inference-sop/` — 架构知识图谱 + Agent 执行 SOP

---

## 阶段四：生成框架验证

用 Agent 自动生成的框架跑通实际推理，验证正确性。

- [ ] 使用 Agent 生成推理框架
- [ ] 跑通推理流程，对比输出正确性
- [ ] 与手动编写的 engine 对比代码质量
