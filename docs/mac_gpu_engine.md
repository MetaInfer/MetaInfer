# Mac GPU (MPS) 推理引擎实现总结

## 项目背景

meta-infer 项目的现有推理引擎（`engine/`）面向 CUDA + Tensor Parallelism 设计，无法在 Apple Silicon Mac 上运行。目标是基于项目已有的知识库和架构经验，构建一个能跑在 Mac MPS 后端的最小推理引擎。

## 架构决策

### 核心思路：复用设备无关组件，替换 CUDA 相关部分

现有引擎的组件并非全部依赖 CUDA。通过分析，将组件分为三类：

| 类别 | 组件 | 处理方式 |
|------|------|----------|
| 设备无关 | structs, sampler, block_manager, scheduler | 直接复制，改 import 路径 |
| CUDA 绑定 | 内存估算（torch.cuda.mem_get_info）、TP 通信层 | 新写 MPS 适配版本 |
| 不需要 | TP layers、Flash Attention、CUDA Graph | 直接跳过 |

### 关键技术选型

1. **`use_cache=False`**：与现有 `RealModelRunner`（`llm_engine.py:97-179`）保持一致。每次 decode 步骤全量重计算，避免在 MPS 上管理 `past_key_values` 的复杂性。代价是 decode 速度 O(n^2)，但最小引擎优先验证正确性。

2. **`torch.float16`**：MPS 对 bfloat16 硬件加速支持有限（截至 2026 年 Apple Silicon 无 bf16 计算单元），float16 是 MPS 上性能最好的选择。

3. **HuggingFace AutoModelForCausalLM**：直接用 HF 的模型实现，不写自定义 attention kernel。HF 内部在 PyTorch 2.0+ 自动使用 `F.scaled_dot_product_attention`，MPS 后端原生支持。

4. **psutil 估算统一内存**：Apple Silicon 使用统一内存架构，没有独立的 GPU VRAM。用 `psutil.virtual_memory().available` 减去模型权重和 2GB 系统预留来估算可用 KV 块数量。

## 实现过程

### 文件结构

```
mac_gpu/
  __init__.py           # 包标记
  structs.py            # Sequence/SequenceStatus 数据结构（复制自 engine/）
  sampler.py            # greedy/top-p 采样器（复制自 engine/）
  block_manager.py      # 分页块管理 + 前缀哈希（复制自 engine/，改 import）
  scheduler.py          # 连续批调度器（复制自 engine/，改 import）
  memory_pool.py        # MPS 内存池（新写，psutil 估算）
  model_runner.py       # MPS 模型运行器（新写，HF + SDPA）
  engine.py             # MacGPUEngine 入口（新写，简化版 LLMEngine）
  main.py               # CLI 入口
chat/
  server.py             # Web 服务（Python stdlib http.server）
  index.html            # 聊天界面
tests/
  test_mac_gpu.py       # 7 个测试用例
```

### 复制 vs 新写的判断标准

判断一个文件是否可以直接复制的标准：**是否包含 `torch.cuda` 调用**。用 grep 搜索 `torch.cuda|\.cuda\(|cuda:|mem_get_info` 即可快速定位 CUDA 绑定点。

- `structs.py`、`sampler.py`：纯 Python / 纯 torch 操作，零 CUDA 依赖 → 直接复制
- `block_manager.py`：纯 Python 逻辑，只有 import 引用 `engine.structs` → 复制 + 改 import
- `scheduler.py`：引用 `engine.memory_pool.KVMemoryPool` → 复制 + 改 import 指向新写的 `MPSMemoryPool`
- `memory_pool.py`：有 `torch.cuda.mem_get_info` 和 `kv_specs` 依赖 → 新写
- `model_runner.py`：原版是 TP 测试用假模型 → 新写 HF 版本

### 遇到的问题

1. **psutil 未安装**：`memory_pool.py` 的 `estimate_num_blocks` 用 `import psutil` 延迟导入，运行时才报错。解决方案：`uv add psutil` 加入依赖。

2. **HuggingFace 下载慢**：国内网络访问 HuggingFace 需要设置 `HF_ENDPOINT=https://hf-mirror.com` 环境变量。

3. **transformers 弃用警告**：`torch_dtype` 参数已弃用，应改用 `dtype`。对运行无影响但日志有噪音。

## 使用方式

### 启动推理

```bash
# CLI 单次推理
HF_ENDPOINT=https://hf-mirror.com python -m mac_gpu.main \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Hello" \
  --max-tokens 64

# Web 服务（浏览器访问 http://localhost:8765）
HF_ENDPOINT=https://hf-mirror.com python chat/server.py --port 8765
```

### 运行测试

```bash
# 基础组件测试（无需 GPU 和模型）
pytest tests/test_mac_gpu.py::test_structs_and_block_manager tests/test_mac_gpu.py::test_memory_pool tests/test_mac_gpu.py::test_scheduler -v

# MPS 设备和采样器测试
pytest tests/test_mac_gpu.py::test_mps_device_available tests/test_mac_gpu.py::test_sampler -v

# 模型加载和端到端测试（需要联网下载模型）
HF_ENDPOINT=https://hf-mirror.com pytest tests/test_mac_gpu.py::test_model_load_and_prefill tests/test_mac_gpu.py::test_end_to_end -v
```

## 性能参考

在 Apple Silicon Mac 上使用 Qwen2.5-0.5B（494M 参数，float16）：

- 模型加载：约 15 秒（本地缓存后）
- 模型权重：约 0.92GB
- Prefill 单条：秒级
- Decode 单步：因 `use_cache=False` 需要全量重计算，随序列增长变慢

## 后续优化方向

1. **启用 KV Cache**：将 `use_cache` 改为 `True`，decode 步骤只传入新 token + past_key_values，可将 decode 从 O(n^2) 降到 O(n)。需要管理 MPS 设备上的 KV cache tensor 生命周期。

2. **流式输出**：当前生成完所有 token 才返回。可以在 engine 中逐步 yield token，配合 SSE (Server-Sent Events) 实现打字机效果。

3. **Per-sequence 采样参数**：当前 decode 批处理时统一使用第一个序列的采样参数，应改为逐序列采样。

4. **支持更多模型**：当前依赖 HF AutoModelForCausalLM 自动加载，理论上支持所有 HF 兼容模型。大模型受限于 Mac 统一内存容量。

## 经验总结

1. **先判断设备依赖再做复制/改写**：用 grep 搜索 CUDA 相关调用可以快速分类，避免不必要的重写。

2. **最小引擎优先验证端到端**：`use_cache=False` 虽然慢，但能最快跑通完整链路（加载→prefill→decode→采样→输出），确认架构正确后再做性能优化。

3. **统一内存需要保守估算**：Apple Silicon 的 CPU 和 GPU 共享内存，不能把所有可用内存都给 KV cache。预留 2GB 系统开销 + 模型权重是必要的。

4. **Python stdlib 够用就不引入框架**：Web 服务用 `http.server` 而非 Flask/FastAPI，减少依赖，零安装即可运行。
