# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

meta-infer 是一个 AI 驱动的 LLM 推理框架生成工具。核心思路是从成熟推理框架（vLLM、SGLang）和精简参考实现中提取共性知识，构建知识库，最终让 Agent 系统按需生成精简推理框架。

## Commands

```bash
# Install (dev mode)
uv sync --extra dev

# Lint & Format
ruff check . --fix
ruff format .

# Run unit tests (no GPU required)
pytest tests/test_scheduler.py tests/test_sequence.py tests/test_memory.py tests/test_prefix_cache.py -v

# Run TP integration tests (4x GPU required)
torchrun --nproc_per_node=4 -m pytest tests/test_qwen_tp_real.py -v -s
torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -v -s

# Run single test file
pytest tests/test_scheduler.py -v

# Pre-commit
pre-commit run --all-files
```

## Architecture

推理引擎实现在 `engine/` 目录（非 `src/meta_infer/`），入口为根目录 `llm_engine.py`。

### 数据流

Request → **Scheduler**（连续批调度）→ **ModelRunner**（前向计算）→ **Sampler**（采样策略）→ Response

### 核心组件 (`engine/`)

- **structs.py** — `Sequence` / `SequenceStatus` 等基础数据结构，贯穿所有组件
- **scheduler.py** — 连续批调度器，管理请求的 prefill/decode 阶段切换
- **block_manager.py** — Paged KV Cache 块管理，支持 radix/prefix caching
- **memory_pool.py** — GPU 显存池管理
- **model_runner.py** — 模型前向执行
- **sampler.py** — greedy / top-p / temperature 采样
- **kv_specs.py** — KV cache 显存规格计算

### Tensor Parallelism (`engine/tp_layers/`)

- **distributed.py** — 通信原语（all-reduce 等）
- **linear.py** — ColumnParallel / RowParallel 线性层
- **embedding.py** — 词表并行 embedding
- **moe.py** — MoE expert 并行化

### 模型实现 (`engine/models/`)

- **deepseek_v2.py** — DeepSeek V2 TP 实现（MLA + MoE）
- **qwen.py** — Qwen3 TP 实现（Dense + MoE）

### 辅助文件

- **tp_distributed.py** — TP 分布式初始化工具
- **openai_tp_server.py** — OpenAI 兼容 HTTP 服务，用于 vllm bench 压测
- **scripts/start_tp_infer_service.sh** — 服务启动脚本
- **scripts/run_*.sh** — 各类基准测试脚本

### Mac GPU 引擎 (`engine/mac_gpu/`)

基于 Apple Silicon MPS 后端的最小推理引擎，结构与 CUDA 引擎类似但面向 Mac GPU。

## Knowledge Base (`notebooks/`)

知识库按层次组织，索引入口为 `notebooks/MEMORY.md`：

- `01_framework_design/` — 调度器、KV Cache、模型执行器等核心设计
- `02_model_specifics/` — DeepSeek V3（MLA/MoE/MTP/NSA）、Qwen3 等模型专项
- `03_operators/` — Flash Attention、FlashInfer、Triton kernel
- `04_parallel_strategies/` — Tensor Parallelism 实现
- `05_non_core_features/` — 量化、PD 分离、投机解码等可选功能
- `06_implementation_patterns/` — 编码模式与反模式

## Development Conventions

- 知识文档使用中文撰写，代码标识符和术语保持英文
- Python 代码遵循 ruff 规范：100 字符行宽，双引号，target Python 3.10+
- `ref_projects/` 中的子模块只读不改，知识提取后存入 notebooks
- 参考 `notebooks/06_implementation_patterns/` 中的模式和反模式进行开发
- 文档结构变更需同步更新 `notebooks/MEMORY.md` 索引

## Project Structure Conventions

### 目录职责

- `engine/` — 推理引擎核心代码（CUDA 主路径）
  - `engine/mac_gpu/` — Mac GPU (MPS) 引擎，仅存放 Mac 独有实现（memory_pool、model_runner、scheduler、engine、main），共用代码（structs、sampler、block_manager）直接引用父目录
  - `engine/models/` — 模型 TP 实现，每个模型一个文件
  - `engine/tp_layers/` — TP 通信与并行算子
- `scripts/` — 所有 shell 脚本（启动服务、基准测试、TP 测试），脚本开头用 `cd "$(dirname "${BASH_SOURCE[0]}")/.."` 定位项目根目录
- `tests/` — 测试文件，命名 `test_<模块>.py`
- `notebooks/` — 知识库文档，中文撰写
- `docs/` — 设计文档
- `ref_projects/` — 参考项目（git 子模块，只读）

### 根目录文件

- `llm_engine.py` — 引擎入口，被 tests 和 openai_tp_server 直接 import，不要移动
- `openai_tp_server.py` — OpenAI 兼容 HTTP 服务入口，被 scripts/ 中的 shell 脚本调用，不要移动
- `CLAUDE.md` / `README.md` / `PROGRESS.md` / `TODO` — 项目文档
- `pyproject.toml` — 包配置，packages 指向 `engine`

### 代码规范

- 引擎入口文件（`llm_engine.py`、`openai_tp_server.py`）留在根目录，因为 tests 和 torchrun 直接 import 它们
- shell 脚本放在 `scripts/`，不在根目录散落 `.sh` 文件
- 新增 Mac GPU 代码时优先复用 `engine/` 的共用组件，避免复制
- `engine/structs.py` 是全局数据结构，所有子模块（包括 mac_gpu）共享
- 删除文件前确认没有其他模块 import 它

## Reference Projects (`ref_projects/`)

| 项目 | 类型 | 用途 |
|------|------|------|
| nano-vllm, nano-sglang, mini-sglang | 正向案例（精简） | 提取精简实现模式 |
| vllm, sglang | 参考案例（完整） | 提取优化策略和架构设计 |

`ref_projects/` 为 git 子模块，初始化需 `git submodule update --init`。
