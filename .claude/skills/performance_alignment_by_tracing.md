# 技能：基于 Tracing 的推理框架性能对齐

## 概述

当你有两个实现相同推理任务的框架（一个为性能基准，另一个为待优化目标），本 skill 提供一套完整的、可复现的迭代优化方法论。核心思路是：**benchmark 量化差距 → profiler 定位瓶颈 → 针对性修改 → 验证正确性 → 记录沉淀 → 循环迭代**，直到目标框架性能达到或超越基准。

## 适用场景

- 两个推理框架实现同一模型架构，目标框架吞吐/延迟劣于基准
- 重写/迁移推理引擎后需要性能对齐
- AI 生成的推理代码需要与人工优化的原型进行性能 PK
- 任意两个可独立运行、可 profile 的同功能系统之间的性能对比

## 前置条件

1. **两个框架均可独立运行**：能接受相同输入、产生可比输出（不要求 bit-exact，但逻辑正确性必须一致）
2. **有统一的 benchmark 脚本**：相同的 prompt、相同的 token 数、相同的 batch size、相同的硬件环境
3. **支持 profiler 工具**：PyTorch 用 `torch.profiler`，其他框架用对应工具（Nsight Systems、perf、vtune 等）
4. **输出正确性可验证**：每轮修改后必须确认输出结果不变（temperature=0 时结果应严格一致）
5. **版本控制**：目标框架代码在 git 管理下，每轮修改可追溯、可还原

## 强制执行铁律

| 规则 | 说明 |
|------|------|
| **绝不修改基准框架** | 基准框架的代码一行都不改。只读对比，只改目标框架 |
| **每轮验证正确性** | temperature=0 时输出必须与基线一致。性能提升但结果错误 = 本轮作废 |
| **每次只改一类瓶颈** | 不要同时改 MLP 缓冲区和 attention 算子。改完一类 → benchmark → 确认有效 → 再改下一类 |
| **改动失败立即还原** | benchmark 回退或正确性失败 → `git checkout` 还原，记录错误原因，尝试下一个方向 |
| **记录每一次尝试** | 包括失败的。失败的教训往往比成功的更有价值 |

## 目录结构约定

在目标框架仓库下建立独立的迭代记录目录：

```
<target-project>/
└── perf_iteration/                  # 性能迭代专用目录
    ├── profiler_compare.py           # 统一的 profiler 脚本（可切换两个框架）
    ├── ROUND_0_BASELINE.md           # 第0轮：基线基准
    ├── ROUND_1_BOTTLENECK_FIXES.md   # 第1轮：瓶颈修复
    ├── ROUND_2_*.md                  # 后续轮次...
    ├── trace_baseline/               # 基准框架的 profiler trace
    │   ├── trace_rank0.json
    │   └── key_avg.txt
    └── trace_target/                 # 目标框架的 profiler trace（每轮更新）
        ├── trace_rank0.json
        └── key_avg.txt
```

技能沉淀目录：
```
<target-project>/.claude/skills/      # 从迭代中提炼的可复用 issue/skill
```

## 详细操作步骤

### 步骤 1：建立统一 benchmark 脚本

两个框架使用**完全相同的测试条件**运行。编写或确认一个 benchmark 脚本，满足：

- 相同的 prompt 输入
- 相同的 max_tokens（建议32~128，太长浪费时间，太短看不出差距）
- temperature=0（确保确定性输出，同时验证正确性）
- 相同的硬件（GPU 数量、型号、CUDA 版本）
- 输出包含：耗时（秒）、吞吐率（tok/s）、生成的文本（用于正确性对比）

**命令示例（PyTorch 多卡 TP）：**
```bash
# 基准框架
cd /path/to/baseline && PYTHONPATH="$(pwd):$PYTHONPATH" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 benchmark.py

# 目标框架
cd /path/to/target && PYTHONPATH="$(pwd):$PYTHONPATH" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 benchmark.py
```

记录结果到 `ROUND_0_BASELINE.md`：

```markdown
| 框架 | 耗时 | 吞吐率 | 差距 |
|------|------|--------|------|
| 基准框架 | X.XXXs | XX.X tok/s | — |
| 目标框架 | Y.YYYs | YY.Y tok/s | -Z% |
```

#### 以 vLLM Eager 模式为基准（Phase 11 场景）

当基准框架为 vLLM Eager 模式（无 CUDA Graph）时，使用 `torch.profiler.ProfilerConfig` 建立 profiler 基线：

```python
import torch
import os
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

MODEL_DIR = os.environ.get("MODEL_DIR", "")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
PROMPT = "苏州园林的特点是"
MAX_TOKENS = 32

# vLLM Eager 模式（enforce_eager=True 禁用 CUDA Graph）
llm = LLM(
    model=MODEL_DIR,
    tensor_parallel_size=TP_SIZE,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    profiler_config=ProfilerConfig(
        enabled=True,
        trace_path="./perf_iteration/trace_baseline",
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ),
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
outputs = llm.generate([PROMPT], sampling_params)
print(f"vLLM eager output: {outputs[0].outputs[0].text}")
```

如果 `ref_projects/vllm/examples/offline_inference/simple_profiling.py` 存在，可直接复用；否则使用上述代码（Python 环境中的 vLLM 包即可，无需 ref_projects）。

### 步骤 2：编写统一 profiler 脚本

编写一个可切换两个框架的 profiler 脚本（参考 `perf_iteration/profiler_compare.py`），关键要素：

```python
import torch.profiler as profiler

activities = [
    profiler.ProfilerActivity.CPU,
    profiler.ProfilerActivity.CUDA,
]

with profiler.profile(
    activities=activities,
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    with profiler.record_function("generate_full"):
        engine.generate(PROMPT, max_new_tokens=MAX_TOKENS, temperature=0.0)

# 输出 key averages 表
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
# 导出 Chrome trace（用 chrome://tracing 可视化）
prof.export_chrome_trace("trace_rank0.json")
```

**运行：**
```bash
# 基准框架
cd /path/to/baseline && python profiler_compare.py baseline

# 目标框架
cd /path/to/target && python profiler_compare.py target
```

### 步骤 3：逐算子对比，识别瓶颈

将两个框架的 `key_avg.txt` 并排对比。重点关注以下维度的差异：

#### 3.1 计算 kernel（GEMM、attention）

| 信号 | 含义 |
|------|------|
| `aten::mm` / `aten::addmm` CUDA 时间相近（±5%） | 矩阵乘正常，GPU 算力利用一致 |
| `flash_fwd_*` 时间差异很大（>2x） | 可能是 page_block_size、KV head 数、或序列长度参数不同 |
| `cutlass` / `ampere_*gemm*` 时间一致 | 底层 GEMM kernel 没问题 |

**判断准则**：如果所有计算 kernel 的 CUDA 时间都相近，说明瓶颈不在 GPU 计算，而在 CPU 调度、内存分配、或同步开销。

#### 3.2 CPU 侧开销

| 信号 | 含义 |
|------|------|
| `cudaLaunchKernel` 大量 CPU 时间 | 过多的 kernel 启动次数或 autograd 包装开销 |
| `cudaDeviceGetAttribute` 出现在前列 | `torch.empty_like()` 等操作在热路径上频繁查询设备属性 |
| `cudaFuncSetAttribute` | 每次调用都在动态设置 kernel 属性，应预编译/缓存 |
| `aten::clone` 大量调用 | autograd 追踪导致中间张量被克隆，或 `.contiguous()` 调用过多 |
| `GeneratedBackwardFor*` | autograd 在构建反向图——推理不需要！加上 `@torch.inference_mode()` |
| CPU 总时间 >> 2× wall clock | 大量 CPU 簿记工作，GPU 在等待 CPU 提交 kernel |

#### 3.3 内存操作

| 信号 | 含义 |
|------|------|
| `Memcpy DtoD` 调用次数差距 >50% | 目标框架有不必要的设备内存拷贝 |
| `Memset (Device)` 大量调用 | 过多的张量初始化 |
| `aten::copy_` 调用过多 | 张量赋值频繁，可能是 KV cache 写入方式不同 |

#### 3.4 同步点

| 信号 | 含义 |
|------|------|
| `aten::item` / `_local_scalar_dense` | GPU→CPU 同步点，会阻塞 CUDA 流水线 |
| `Memcpy DtoH (Device → Pinned)` | 从 GPU 读数据到 CPU，同样会同步 |
| `cudaStreamSynchronize` | 显式的流同步 |

**组织瓶颈清单**：将发现的问题按 GPU 时间贡献排序，写入当轮文档的"已识别问题"表。

### 步骤 4：设计修改方案（只改目标框架）

对每个瓶颈，自顶向下排优先级：

1. **架构级**：是否缺少关键模式？（如 `inference_mode`、CUDA Graph、编译缓存）——影响最大
2. **同步点**：是否有不必要的 GPU→CPU 同步？——消除后通常有显著提升
3. **内存分配**：热路径上是否每步都在 `torch.empty()` / `torch.empty_like()`？——预分配可消除
4. **计算 kernel**：算子选择是否最优？（attention 实现、GEMM 精度、kernel fusion）——通常是最后才需要调的
5. **微优化**：`split()` vs 切片、`index_copy_` vs 直接赋值——影响最小

**每次只改一类瓶颈**，改完立刻 benchmark 验证。不要贪多。

### 步骤 5：实施修改并验证正确性

```
修改 → benchmark → 正确性检查 → 记录结果
  │                    │
  │         ┌──────────┘
  │         ▼
  │   结果正确 + 性能提升 → 保留
  │   结果正确 + 性能不变 → 保留（至少没引入回归），记录原因
  │   结果正确 + 性能回退 → git checkout 还原，记录错误
  │   结果错误（无论性能） → git checkout 还原，记录错误
  └── 继续下一个瓶颈
```

**正确性验证**：temperature=0 时，目标框架的输出必须与基准框架完全一致。如果不一致，说明修改引入了逻辑错误。

### 步骤 6：对比 profiler，确认瓶颈消除

每次成功的修改后，重新运行 profiler，确认对应的异常指标已经消失或改善：

```bash
cd /path/to/target && python perf_iteration/profiler_compare.py target
# 对比 key_avg.txt 中前一轮识别的瓶颈条目
```

### 步骤 7：输出结构化迭代文档

每轮（包含多步修改）写入一个 `ROUND_N_*.md` 文档，必须包含：

```markdown
# 第N轮 — <简短描述>

## 吞吐率对比

| 框架 | 耗时 | 吞吐率 | 差距 |
|------|------|--------|------|
| 基准 | X.XXXs | XX.X tok/s | 目标 |
| 目标（基线） | Y.YYYs | YY.Y tok/s | -Z% |
| 目标（本轮后） | Z.ZZZs | ZZ.Z tok/s | -Z'% 或 +Z'% |

## 优化过程（每步记录）

| 步骤 | 改动 | 耗时 | 吞吐率 |
|------|------|------|--------|
| 0 | 本轮起始 | — | — |
| 1 | <改动描述> | — | — |
| ... | ... | ... | ... |

## 已应用的修复

### 修复1：<标题>
**文件**：`path/to/file:line`
**改动**：`before` → `after`
**原因**：<为什么这样改，为什么有效>

## Profiler 对比

| 指标 | 本轮前 | 本轮后 | 变化 |
|------|--------|--------|------|
| <关键指标> | — | — | — |

## 遇到的错误

1. **<错误描述>**：<现象> → <原因> → <处理方式>
```

### 步骤 8：沉淀有效经验为 skill

将每轮发现的**通用模式**（不限于当前框架）沉淀为 skill 文件。

**什么样的发现值得沉淀为 skill：**
- 适用于任意 PyTorch / JAX / TensorFlow 推理框架的通用模式
- 在 profiler 中有明确识别信号的性能反模式
- 一行/少量代码即可消除显著性能损失的模式
- 容易在 AI 生成代码中遗漏但人工专家从不会忘的模式

**skill 文件结构：**
```markdown
# 技能：<简短名称>

## 模式
<核心做法，含代码示例>

## 为什么重要
<原理 + 实际数据>

## 如何识别
<profiler 信号、benchmark 症状>

## 何时应用 / 何时不用

## 验证方法

## 反模式
<错误做法及为什么错>
```

将 skill 文件保存到 `.claude/skills/` 目录，命名遵循 `<关键词>.md` 格式。

**skill 的作用**：在下一次 agent 从零构建推理框架时，这些 skill 会在相应阶段自动匹配并被加载到 agent 上下文中，帮助 agent 在第一版代码中就做对，而非事后修复。

### 步骤 9：持续迭代直到对齐

```
while 目标吞吐率 < 基准吞吐率:
    1. 运行 profiler_compare.py target    # 获取最新 trace
    2. 对比 key_avg.txt 与 baseline       # 识别剩余瓶颈
    3. 按优先级排序瓶颈                   # 架构 > 同步 > 分配 > 计算
    4. 选最高优先级瓶颈，设计方案         # 只改目标框架
    5. 修改代码，benchmark，验证正确性    # 严格单步
    6. 记录到 ROUND_N_*.md               # 结构化文档
    7. 如有通用发现，新建 skill           # 经验沉淀
    8. N += 1

print("性能对齐完成，输出最终摘要")
```

**终止条件**：目标框架吞吐率达到基准的 95% 以上，或所有可识别瓶颈已消除。

**如果差距无法合拢**：列出剩余的"不可消除差异"（如基准使用了目标框架无法使用的闭源组件、不同的硬件拓扑等），作为最终结论的一部分。

## 输出产物规范

| 产物 | 位置 | 格式 | 内容要求 |
|------|------|------|---------|
| 基线 benchmark | `perf_iteration/ROUND_0_BASELINE.md` | Markdown 表格 | 两框架首次对比 + profiler 差异表 + 问题清单 |
| 每轮迭代记录 | `perf_iteration/ROUND_N_<主题>.md` | Markdown | 优化过程表 + 修复详情 + profiler 对比 + 错误记录 |
| Profiler trace | `perf_iteration/trace_<name>/` | JSON + txt | Chrome trace + key_avg.txt |
| 通用 issue/skill | `.claude/skills/<关键词>.md` | Markdown | 模式 + 原理 + 识别 + 验证 + 反模式 |
| Profiler 脚本 | `perf_iteration/profiler_compare.py` | Python | 可切换两个框架，可复用 |

## 注意事项

1. **绝不要修改基准框架代码**。即使发现基准框架有 bug 也不改——那会让你失去参照。在文档中记录差异即可。
2. **每轮必须验证正确性**。temperature=0 时输出应与基准一致。性能提升但逻辑错误 = 伪提升。
3. **profiler 本身有开销**。profiler 会扭曲通信操作的 CUDA 时间（特别是 NCCL all_reduce、custom_ar）。端到端耗时（无 profiler 的 benchmark）才是真正的性能指标。profiler 只用来对比算子级别的差异，不用来测量绝对性能。
4. **benchmark 有波动**。±2% 的波动是正常的，不要反复在两三个百分点之间纠结。看到 ≥3% 的变化才视为有效。
5. **优先消除新增项**。profiler 中目标框架有而基准没有的条目（如 `cudaDeviceGetAttribute`），往往是最容易消除且收益最大的瓶颈。
6. **谨防 profiler 失真**。某些 kernel 在 profiler 下会显示异常高的时间（如 `flash_fwd_splitkv_combine` 被报告 5.8x 慢于基准，但实际两个框架使用完全相同的 CUDA kernel）。当所有计算 kernel 时间都"一致"但总体仍有较大差距时，问题大概率在 CPU 侧（同步点、内存分配、autograd 追踪）。
7. **失败也是产出**。记录每一条失败的尝试和原因，这是未来避免走弯路的宝贵知识。
8. **对 AI 生成的代码格外注意**。AI 容易正确实现模型架构但遗漏关键的性能模式（如 `torch.inference_mode()`、预分配缓冲区、消除 GPU→CPU 同步）。在 profiler 对比时优先排查这些"AI 盲区"。

## 快速检查清单

在每轮开始前确认：

- [ ] 基准框架 benchmark 稳定（连续 3 次波动 < 2%）
- [ ] 目标框架输出正确（与基准对比，temperature=0）
- [ ] profiler 脚本对两个框架都能正常运行
- [ ] 上一轮的文档已写完并保存
- [ ] 上一轮的有效发现已沉淀为 skill（如果够通用）

每轮结束后确认：

- [ ] 本轮 benchmark 数据已记录
- [ ] profiler trace 已保存
- [ ] 所有改动（包括失败的）已记录到文档
- [ ] 代码改动已 commit（保持可追溯）
- [ ] 正确性已验证

## 抽象过程说明（本次的具体案例 → 通用方法论）

本次 skill 是从以下具体案例中抽象而来：

**具体案例**：
- 基准框架：人工编写的推理引擎
- 目标框架：AI 生成 + agent 迭代的同架构推理引擎
- 初始差距：目标框架 vs 基准（存在显著性能差距）
- 最终结果：经过多轮迭代后目标框架超越基准

**迭代过程**：
- 第0轮：建立基线，识别8个 profiler 差异项
- 第1轮：应用9项修复，每步 benchmark 验证，记录4个失败尝试

**抽象出的通用模式**：
1. 具体的 `torch.empty_like` → GPU 同步 → 热路径内存分配 → 通用的"CPU 侧开销"类别
2. 具体的 `aten::clone` 2504次 → autograd 追踪 → 通用的"推理模式下禁用梯度追踪"模式
3. 具体的 Qwen3-8B → 通用的"同模型、同输入、不同实现"场景
4. 具体的 `perf_iteration/ROUND_N_*.md` → 通用的结构化文档模板
5. 具体的 `torch-inference-mode.md` skill → 通用的 skill 沉淀规则

这样，当下次面对任意两个推理框架（不论模型架构、框架语言、硬件平台）的性能对齐任务时，本 skill 提供的方法论、目录结构、文档模板和检查清单均可直接复用。
