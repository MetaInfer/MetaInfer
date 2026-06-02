# MetaInfer — Claude Code 工作指南

AI 驱动的 LLM 推理框架生成工具。从成熟框架（vLLM、SGLang）及其精简实现中提取模式，构建知识库，指导生成精简推理框架。

## 当前分支

**`feature/mac-impl`** — Apple Silicon (MPS/Metal) 平台推理引擎实现。

上游分支: `origin/feature/tp-implementation`（NVIDIA CUDA TP 优化，55.7 tok/s @ Qwen3-8B）。

## 目录结构

- `notebooks/` — 知识库文档（框架设计、模型专项、算子、并行、实现模式）
- `ref_projects/` — 参考项目子模块（vllm, sglang, nano-vllm, nano-sglang, mini-sglang）
- `src/meta_infer/` — Python 包（Agent 系统）
- `subprojects/` — 平台推理引擎构建项目，每个有独立 `docs/` 记录工作过程
  - `subprojects/mac-engine/` — Mac Apple Silicon 推理引擎

## 常用命令

```bash
# 初始化参考项目子模块
git submodule update --init --recursive

# 安装开发环境
pip install -e ".[dev]"
pre-commit install

# 代码检查与格式化
ruff check src/ subprojects/
ruff format src/ subprojects/
```

## 子项目工作流

1. 从 `notebooks/` 提取目标推理模式
2. 在 `subprojects/` 创建/进入平台构建项目
3. 按阶段在 `docs/` 记录：规划 → 设计 → 实现 → 测试
4. 阶段结论同步回 `notebooks/` 知识库

## 开发约定

- 知识文档中文撰写，代码标识符和术语英文
- Python 代码遵循 ruff 规范（line-length=100, double-quote）
- 文档结构参照 `notebooks/MEMORY.md` 索引层次
- 参考 `notebooks/06_implementation_patterns/` 中的模式和反模式
- ref_projects 子模块仅供阅读分析，不直接修改

## mac-engine 约束

**mac-engine 目标**: 取代 `mlx_lm`，从头实现推理引擎。可用的底层与禁止的上层：

| 层级 | 可用 ✅ | 禁止 ❌ |
|------|--------|--------|
| MLX Core | `mlx.core` (mx), `mlx.nn` | — |
| 模型实现 | 自己实现 Transformer/Attention/MLP/RMSNorm 等 | `mlx_lm.models.*` |
| 模型加载 | 自己读 safetensors + config.json | `mlx_lm.load()` |
| Tokenizer | 自己读 tokenizer.json + 实现 encode/decode | `mlx_lm` 的 tokenizer wrapper |
| KV Cache | 自己实现 | `mlx_lm.models.cache.make_prompt_cache` |
| 采样 | 自己实现 | `mlx_lm.sample_utils.make_sampler` |
| Server | 自己实现 OpenAI API | — |

**可用参考**（只读）:
- `mlx_lm/models/qwen3.py` — 模型架构参考
- `mlx_lm/models/cache.py` — KV cache 设计参考
- `mlx_lm/generate.py` — 生成流程参考
- `notebooks/` — 设计模式参考

**基准测试串行约束**: 所有 benchmark 必须串行执行。
- ❌ 禁止多个 bench 并行 (MLX GPU 资源竞争导致结果失真)
- ✅ 使用 `&&` 链式串行执行
- ✅ 每个 bench 完成后 `mx.clear_cache()` 再跑下一个

## 上游分支参考

`origin/feature/tp-implementation` 包含 NVIDIA CUDA TP=4 的完整优化链路：

| 阶段 | 内容 | 吞吐 (tok/s) |
|------|------|-------------|
| Baseline | 全量重算，无 KV cache | 2.15 |
| P0 | 增量 KV Cache 解码 | 8.49 |
| P2 | torch.compile + 固定形状 | 12.75 |
| P3-FA | Flash Attention 集成 | 正确性通过 |
| P3-Triton | Triton MLA Decode Kernel | 13.08 |
| P5 | TP 通信优化 | 8.87 |
| Stage 1-7 | kernel 替换 | 55.7 |

上游模型文件: `engine/models/qwen.py`, `engine/models/deepseek_v2.py`, `engine/scheduler.py`, `engine/memory_pool.py`, `engine/tp_layers/`, `llm_engine.py`。
