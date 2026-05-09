# meta-infer

AI 驱动的 LLM 推理框架生成工具。从成熟推理框架（vLLM、SGLang）和精简参考实现中提取共性知识，构建知识库，最终让 Agent 系统按需生成精简推理框架。

## 项目结构

```
meta-infer/
├── engine/              # 自研推理引擎核心组件
│   ├── structs.py       # Sequence / SequenceStatus 数据结构
│   ├── scheduler.py     # 连续批调度器
│   ├── block_manager.py # Paged KV Cache + prefix caching
│   ├── memory_pool.py   # GPU 显存池管理
│   ├── model_runner.py  # 模型前向执行
│   ├── sampler.py       # greedy / top-p / temperature 采样
│   ├── models/          # 模型实现（Qwen3、DeepSeek V2）
│   ├── tp_layers/       # Tensor Parallelism 算子
│   └── mac_gpu/         # Mac GPU (MPS) 推理引擎
├── tests/               # 测试用例
├── notebooks/           # 推理框架知识库
├── docs/                # 文档
├── scripts/             # 部署与基准测试脚本
├── ref_projects/        # 参考项目（git 子模块，只读）
├── llm_engine.py        # 引擎入口
└── openai_tp_server.py  # OpenAI 兼容 HTTP 服务
```

## 快速开始

```bash
# 安装
uv sync --extra dev

# 运行单元测试（无需 GPU）
pytest tests/test_scheduler.py tests/test_sequence.py tests/test_memory.py tests/test_prefix_cache.py -v

# 运行 TP 推理测试（4x GPU）
torchrun --nproc_per_node=4 -m pytest tests/test_qwen_tp_real.py -v -s
torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -v -s
```

## 知识库

知识库位于 `notebooks/`，按层次组织：

- `01_framework_design/` — 调度器、KV Cache、模型执行器等核心设计
- `02_model_specifics/` — DeepSeek V3（MLA/MoE）、Qwen3 等模型专项
- `03_operators/` — Flash Attention、FlashInfer、Triton kernel
- `04_parallel_strategies/` — Tensor Parallelism 实现
- `05_non_core_features/` — 量化、PD 分离、投机解码等
- `06_implementation_patterns/` — 编码模式与反模式

索引入口：`notebooks/MEMORY.md`
