---
name: infer-bench
description: >
  对自研推理引擎（meta-infer / mac-engine）与成熟框架（vLLM、SGLang、MLX、llama.cpp）
  进行系统化的三层性能对比（算子 → Profiling → 端到端 Serving），产出结构化对比报告。
  当用户说"对比一下"、"跑 benchmark"、"基准测试"、"性能对比"、"和 vLLM 比一下"、
  "跟 llama.cpp 对比"、"评估优化效果"、"看看差距多大"时立即触发。
  支持 NVIDIA CUDA 和 Apple Silicon 双平台。
---

# Infer-Bench: 推理引擎性能对抗规范

对自研推理引擎与成熟框架进行三层对比，产出量化差距报告与可操作的优化建议。

---

## 前置条件

在执行任何对比之前，**必须**先读取对规范文档：

```
notebooks-cn/07_improvementPlan/benchmark_spec.md
```

该文档定义了完整的三层对比模型、环境规范、报告模板和优化优先级排序方法。本 skill 是规范的执行器，所有方法论细节以该文档为准。

---

## 第一步：解析用户意图

从用户输入中提取以下参数（未指定的使用默认值）：

| 参数 | 默认值 | 可选值 |
|------|--------|--------|
| **自研引擎** | 根据当前分支推断 | `meta-infer` (NVIDIA CUDA), `mac-engine` (Apple Silicon) |
| **成熟框架** | 单平台选 vLLM，双平台选 MLX | `vllm`, `sglang`, `mlx`, `llama.cpp` |
| **模型** | Qwen3-8B (NVIDIA) / Qwen2.5-7B (Apple) | 任何已下载的模型 |
| **对比层级** | 全部三层 | `op` (算子), `profile` (Profiling), `serve` (Serving), `all` |
| **ROUNDS** | 5 (快速) / 25 (正式) | 整数 |
| **STEPS** | 8 (快速) / 1,2,4,8,16,32 (正式) | 逗号分隔 |
| **模式** | quick (快速验证) | `quick` (ROUNDS=5,STEPS=8), `full` (ROUNDS=25,STEPS=1,2,4,8,16,32) |

**询问规则**：关键参数（自研引擎、模型）缺失且无法推断时，向用户确认。非关键参数使用默认值，在报告中标注。

**平台推断**：
- 用户提到 mac-engine → Apple Silicon 平台（成熟框架默认 MLX + llama.cpp）
- 用户提到 meta-infer → NVIDIA CUDA 平台（成熟框架默认 vLLM + SGLang）
- 无法推断 → 询问用户当前在哪个平台

---

## 第二步：环境检查

### 2.1 检查模型是否可用

```bash
# 根据模型名在常见路径查找
find /data/models /data/xinference/cache ~/.cache/huggingface -maxdepth 3 -type d -name "*<model_name>*" 2>/dev/null
```

若模型不存在，告知用户并停止。

### 2.2 检查成熟框架是否可用

```bash
# NVIDIA
python -c "import vllm; print(vllm.__version__)" 2>/dev/null
# Apple
python -c "import mlx; print(mlx.__version__)" 2>/dev/null
python -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null
```

若目标框架未安装，提供安装命令，询问是否继续或更换对比对象。

### 2.3 检查 GPU / 计算资源

**NVIDIA**:
```bash
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
```

**Apple Silicon**:
```bash
sysctl -n machdep.cpu.brand_string && sysctl hw.memsize
```

记录硬件配置在报告中。

---

## 第三步：执行对比

按用户指定的层级执行。**同一层级的双方对比必须使用相同的参数和硬件**。
**正确性验证（Layer 0）必须在任何性能对比之前通过——输出不对，性能数据无意义。**

### Layer 0: 正确性验证（Greedy Decode 逐 token 对齐）

> **优先级最高。** Layer 0 不通过，Layer 1/2/3 结果无效。

#### 验证方法

通过 OpenAI-compatible API 调用两个引擎，对比 greedy decode (temperature=0.0) 输出：

```bash
# 两个 engine 分别启动 server
# 参考引擎 (已由 infer-ref-bench 采集 golden outputs)
python -m mlx_lm.server --model <model_path> --port 8080 &

# 自研引擎
python -m mac_engine.server --model <model_path> --port 8081 &
```

运行验证脚本（与 infer-ref-bench 共用 `scripts/verify_correctness.py`）：

```bash
python scripts/verify_correctness.py \
  --ref-url http://localhost:8080/v1 \
  --target-url http://localhost:8081/v1 \
  --golden tests/golden_outputs/golden_outputs.json
```

#### 验证矩阵

| 维度 | 检查方法 | 通过条件 |
|------|---------|---------|
| 逐 token 对齐 | 对比 golden vs target 完整输出文本 | 字符级完全一致 |
| 特殊 token 处理 | 空 prompt / chat template 用例 | 输出一致 |
| 长上下文 | 长 prompt 用例 | 输出一致 |
| 终止条件 | finish_reason 一致 | `stop` / `length` 一致 |

#### 输出格式

```
Layer 0 正确性验证:
| 测试用例 | Golden Hash | Target Hash | 状态 |
|---------|------------|-------------|------|
| basic_en | a1b2c3d4 | a1b2c3d4 | ✅ |
| basic_zh | e5f6g7h8 | e5f6g7h8 | ✅ |
| edge_empty | i9j0k1l2 | i9j0k1l2 | ✅ |

Result: 7/7 PASS
```

**FAIL 处理**: 输出不一致时，先定位差异位置 → 对比两者 tokenizer 输出 → 对比两者模型 logits → 定位根因后再跑性能。

### Layer 1: 算子级对比

**适用范围**: 仅当自研引擎有可运行的算子实现时。

对每个关键算子（RMSNorm、RotaryEmbedding、Attention、SiluAndMul 等），执行：

1. 从成熟框架提取同义算子的 CUDA/Metal kernel
2. 用相同输入分别在双方上运行
3. `torch.testing.assert_close` 验证数值精度（rtol=1e-2）
4. 用 CUDA/Metal event timer 记录单次耗时

输出格式：
```
| 算子 | 自研耗时 | 成熟框架耗时 | 差异 | 数值精度 |
|------|---------|-------------|------|---------|
| rms_norm | 12μs | 8μs | +50% | ✅ PASS |
```

### Layer 2: Profiling Trace 对比

**NVIDIA 平台**:
```python
# 使用 torch.profiler 抓取双方 trace
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(STEPS):
        model.forward(...)  # 或 engine.step()
```

**Apple Silicon 平台**:
使用 PyTorch MPS profiler 或 Instruments 抓取 Metal trace。

从 trace 中提取并输出：
1. **GPU Compute / Comm / Other 三分法分解表**
2. **CPU GEMM dispatch / 通信 dispatch / Kernel launch / Tensor 管理分解表**
3. **Top 8 GPU Kernel 对比表**

对比分析要点：
- Compute 差距 > 20% → 输入 shape 不一致或未使用标品 kernel
- Kernel 数量差距 > 2x → 逐元素算子未融合
- CPU Self 差距 > 3x → CUDA Graph 缺失或 Python 控制流过多

### Layer 3: Serving Benchmark

**NVIDIA 平台**:
- 使用 vLLM 官方 benchmark 脚本 `ref_projects/vllm/benchmarks/benchmark_serving_structured_output.py`
- 对双方使用完全相同的参数跑 benchmark
- 提取 request_throughput, output_throughput, TTFT, TPOT, E2EL

**Apple Silicon 平台**:
- 对 MLX: 使用其自带的 benchmark 脚本或写简单的 HTTP server + wrk/httpx 压测
- 对 llama.cpp: 使用 `llama-bench` 工具或 server 模式 + HTTP 压测

**关键**: 多 STEPS 梯度跑（1, 2, 4, 8, 16, 32），输出对比表：
```
| STEPS | 自研引擎 | 成熟框架 | 差距 |
|-------|---------|---------|------|
| 1     | X tok/s | Y tok/s | ...  |
| 4     | ...     | ...     | ...  |
| 32    | ...     | ...     | ...  |
```

---

## 第四步：生成对比报告

报告必须包含以下章节，参考 `benchmark_spec.md` 中的模板：

### 报告结构

```markdown
# [自研引擎] vs [成熟框架] 性能对比报告

## 1. 测试环境
- 硬件: ...
- 软件: PyTorch X.X, 成熟框架版本, ...
- 模型: ...
- 参数: ROUNDS=X, STEPS=[...], ...

## 2. 正确性验证
- [ ] L1: 所有算子数值验证通过
- [ ] L2: 端到端 greedy decode 字字对齐

## 3. GPU 时间分解 (Layer 2)
| 类别 | 自研 | 成熟框架 | 差距 |
|------|------|---------|------|
| Compute | ... | ... | ... |
| Comm    | ... | ... | ... |
| Other   | ... | ... | ... |
| **Total** | ... | ... | ... |

## 4. CPU 时间分解 (Layer 2)
| 类别 | 自研 | 成熟框架 | 差距 |
|------|------|---------|------|

## 5. Top GPU Kernel 对比 (Layer 2)
| 排名 | 自研 Kernel | 耗时 | 成熟框架 Kernel | 耗时 |
|------|-----------|------|----------------|------|

## 6. 端到端 Serving 对比 (Layer 3)
| STEPS | 自研吞吐 | 成熟框架吞吐 | 差距 | 自研 TTFT | 成熟框架 TTFT |
|-------|---------|------------|------|----------|-------------|

## 7. 差距分析与优化建议
| 优先级 | 瓶颈 | 当前耗时 | 目标耗时 | 预期提升 | 改进方案 |
|--------|------|---------|---------|---------|---------|

## 8. 结论
- 总差距: X.Xx
- 主要瓶颈: [Compute / Comm / CPU dispatch]
- 下一步: [具体优化方向]
```

### 报告文件命名

```
notebooks-cn/07_improvementPlan/compare_<engine>_<framework>_<model>_<YYYYMMDD>.md
```

例如: `compare_mac-engine_mlx_Qwen2.5-7B_20260602.md`

---

## 第五步：提交分析结论

在终端输出简洁摘要：

```
✅ 对比完成 → notebooks-cn/07_improvementPlan/compare_mac-engine_mlx_Qwen2.5-7B_20260602.md

自研引擎 vs MLX:
- Layer 2 GPU: 45ms vs 38ms (1.18x)
- Layer 2 CPU: 120ms vs 45ms (2.67x) ← 主要瓶颈
- Layer 3 STEPS=32: 24.5 tok/s vs 35.2 tok/s (0.70x)

关键发现:
- GPU Compute 差距小 (1.05x) — 使用了相同的 Metal kernel
- CPU dispatch 差距大 (2.67x) — 缺少 CUDA/Graph 或 Metal 等价物
- 下一步: P0 = 减少 kernel launch 次数 (逐个算子融合)
```

---

## 快速模式 vs 完整模式

**快速模式** (`quick`, 默认):
- ROUNDS=5, STEPS=8
- 仅跑 Layer 3 (Serving Benchmark) — 最直接反映差距
- 如果 Layer 3 结果异常，再回过头跑 Layer 2
- 报告精简为 1-5 节 + 结论

**完整模式** (`full`):
- ROUNDS=25, STEPS=1,2,4,8,16,32
- 跑全部三层对比
- 完整报告含所有章节

---

## 平台特定说明

### NVIDIA CUDA (meta-infer)

- **对比对象**: vLLM (首选), SGLang (次选)
- **Benchmark 工具**: `run_compare_metainfer_vllm.sh` (一键对比脚本，优先使用)
- **Profiling**: `torch.profiler` + CUDA events
- **参考文档**: `notebooks-cn/07_improvementPlan/stage0_2_vs_vllm.md`

### Apple Silicon (mac-engine)

- **对比对象**: MLX (首选, 同质 Metal 后端), llama.cpp (次选, C++ Metal)
- **Benchmark 工具**: 需要自行编写简单的 HTTP server + 压测脚本（mac-engine 目前为骨架，暂无一键脚本）
- **Profiling**: PyTorch MPS profiler 或 Instruments (Metal System Trace)
- **注意**: 与 NVIDIA 平台的 vLLM 数据不可直接对比（硬件不同），但可以作为目标线参考

---

## 关键约束

1. **正确性优先**：Layer 0 不通过，禁止进入 Layer 1/2/3。
2. **同一硬件**: 双方必须在同一台机器、同一 GPU/芯片上测试。严禁跨机器对比。
3. **同一模型**: 双方使用同一 checkpoint 目录和 tokenizer 配置。
4. **温度归零**: `temperature=0.0` 保证 greedy decode 可复现。
5. **先验证后测速**: 正确性不通过，性能数据无效。
6. **差距必须量化**: 不说"差不多"，只说"1.18x"或"快/慢 X%"。
7. **OpenAI API 协议对比**：自研引擎与参考引擎均通过 `/v1/completions` 调用，避免 tokenizer 实现差异。
