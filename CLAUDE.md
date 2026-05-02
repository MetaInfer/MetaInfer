# meta-infer

AI 驱动的 LLM 推理框架生成工具。通过分析成熟推理框架（vLLM、SGLang）和精简参考实现（nano-vllm、nano-sglang、mini-sglang）的共同模式，构建知识库，指导 Agent 系统按需生成精简的推理框架。

## 目录结构

- `notebooks/` — 知识库文档，按层次组织：框架设计、模型专项、算子、并行策略、可选功能、实现模式
- `ref_projects/` — 参考项目（git 子模块）
  - 正向案例（精简）：nano-vllm, nano-sglang, mini-sglang
  - 参考案例（完整）：vllm, sglang
- `src/meta_infer/` — Python 包，后续开发的 Agent 系统

## 开发约定

- 知识文档使用中文撰写，代码标识符和术语保持英文
- 文档结构遵循 `notebooks/MEMORY.md` 中的索引层次
- Python 代码遵循 ruff 格式化规范
- 参考 `notebooks/06_implementation_patterns/` 中的模式和反模式

## 参考项目说明

ref_projects 中的子模块仅供阅读和分析，不直接修改。知识从参考项目中提取后存入 notebooks。
