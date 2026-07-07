---
name: fix-paper
description: Iteratively fix MetaInfer paper §6.2 K100 DCU issues. Reads Metainfer-review.md, re-measures profiler data (12-token, aligned with A800), fixes LaTeX, and self-verifies with ${CLAUDE_CLI} -p. Use when user says /fix-paper, "fix paper", "修改论文", or working through review feedback. K100 DCU only.
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
  - TaskCreate
  - TaskUpdate
  - TaskList
  - Agent
---

# /fix-paper — MetaInfer §6.2 K100 DCU 章节迭代修改

**机器**: K100 DCU（海光 K100, 4× GPU PCIe 4.0, ROCm 6.3.3）
**负责范围**: 论文 §6.2（海光 DCU 适配）及关联的 §6.3/§7.1 中引用 DCU 数据的部分。**不负责** §5（A800 性能验证）、§6.1（Apple Silicon）。
**Profiler 配置**: **12 tokens 输出**，对齐 A800 §5.3 的 12 output tokens 配置。审稿人 P66 指出原 24-token（23 decode steps）步数奇怪，改为与 A800 一致。

## 路径映射

| 用途 | 路径 |
|------|------|
| 项目根目录 | `$AGENT_INFER_ROOT`（由 `.env_agent_infer` 动态设置） |
| 模型 | `$MODEL_DIR`（由 `.env_agent_infer` 设置，如 `/data/xinference/cache/Qwen3-8B`） |
| LaTeX 论文 | `$AGENT_INFER_ROOT/paperdoc/MetaInfer.tex` |
| 技术文档 | `$AGENT_INFER_ROOT/paperdoc/MetaInfer技术文档（修订）.md` |
| 审阅意见 | `$AGENT_INFER_ROOT/paperdoc/Metainfer-review.md` |
| 数据日志 | `$AGENT_INFER_ROOT/paperdoc/Metainfer-DATALOG.md` |
| 审计日志 | `$AGENT_INFER_ROOT/paperdoc/Metainfer-AUDIT.md` |
| 修改计划 | `$AGENT_INFER_ROOT/paperdoc/Metainfer-PLAN.md` |
| 矩阵 benchmark（已有） | `$AGENT_INFER_ROOT/paperdoc/K100测量数据/benchmark_matrix_3x3.md` |
| Profiler trace（归档） | `$AGENT_INFER_ROOT/paperdoc/K100测量数据/profiler_trace/` |
| 环境变量 | `source $AGENT_INFER_ROOT/.env_agent_infer` |

## 已有数据资产（无需重测）

**矩阵 benchmark（2026-06-10）**: 3×3 矩阵（3 prompts × 3 gen_lens, 3 rounds, TP=4, Qwen3-8B），完整 9 cell 数据已在 `paperdoc/K100测量数据/benchmark_matrix_3x3.md`。

| 平台 | 脚本 | 结果 |
|------|------|------|
| MetaInfer DCU-engine | `bench_matrix_direct.py` (torchrun) | 25.0 tok/s avg |
| vLLM Eager | `bench_vllm_eager_matrix.sh` (HTTP API) | 21.1 tok/s avg |

**矩阵 benchmark 数据已确认正确**，直接用于论文表 22-23 的数值。不再重跑。

## 需要重测的数据：Profiler Tracing

以下 6 项审阅问题需要新 profiler 数据才能修复：

| 编号 | 问题 | 需要的 profiler 数据 |
|------|------|---------------------|
| P66 | 原采集 23 decode steps（奇怪数字）→ 改为 12 token，对齐 A800 | 新 12-token profiler trace |
| P69 | `hipGetDeviceProperties` 每个 decode step 出现 20.4ms → 疑似 profiler artifact 或代码 bug | 确认调用频率和真实耗时 |
| P119 | DCU-engine TPOT 异常稳定（39.6-39.8ms, CV=0.25%）→ 需 profiler 验证 decode kernel 耗时方差 | Decode kernel 耗时分布 |
| P120 | TTFT 随生成长度变化 → 可能含 KV cache 分配开销 | Profiler 单独计时 prefill vs KV alloc |
| P160 | 表 25 各组件 GPU 时间之和 > Self CUDA Total → 算术矛盾 | 正确拆分各组件 GPU 时间，互斥计时 |
| P148 | TPOT +17% 优势无法仅用 CPU 调度差异解释 → 需 GPU 端定位 | GPU kernel 层面根因分析 |

### Profiler 重采流程

**Step 1: 修改 profiler 脚本（12 tokens）**

```bash
cd $AGENT_INFER_ROOT
source .env_agent_infer

# 检查并修改脚本中的 MAX_TOKENS 为 13（= 12 decode steps + 1 first token）
# 脚本位置: scripts/test_phase11_202_profiler.sh
# 将 MAX_TOKENS = 24 → MAX_TOKENS = 13
```

**Step 2: 运行 profiler**

```bash
cd $AGENT_INFER_ROOT
source .env_agent_infer

# 设置 NCCL/RCCL 日志级别
export NCCL_DEBUG=ERROR
export VLLM_LOGGING_LEVEL=ERROR

# 运行 profiler（torchrun TP=4）
torchrun --nproc_per_node=4 python -c "
import os, sys
sys.path.insert(0, '$AGENT_INFER_ROOT')
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['NCCL_DEBUG'] = 'ERROR'
# ... 使用 torch.profiler.profile() 采集 12 decode steps
"
```

**Step 3: 验证数据自洽性**

运行后立即检查：
1. 子项之和 = 总项（误差 < 1%）— 每个组件 GPU 时间是互斥的
2. TPOT 方差分析 — CV 是否真的 <1%
3. hipGetDeviceProperties 调用次数 — 是否 >1
4. Prefill 与 KV cache 分配时间可分离

**Step 4: 归档原始 trace**

```bash
mkdir -p $AGENT_INFER_ROOT/paperdoc/K100测量数据/profiler_trace/
cp trace_rank0.json $AGENT_INFER_ROOT/paperdoc/K100测量数据/profiler_trace/
cp key_avg.txt $AGENT_INFER_ROOT/paperdoc/K100测量数据/profiler_trace/
# 保存运行日期和 git commit
date +%Y-%m-%d_%H:%M:%S > $AGENT_INFER_ROOT/paperdoc/K100测量数据/profiler_trace/TIMESTAMP
git rev-parse HEAD >> $AGENT_INFER_ROOT/paperdoc/K100测量数据/profiler_trace/TIMESTAMP
```

**Step 5: 写入 DATALOG**

```bash
# 追加到 paperdoc/Metainfer-DATALOG.md，按以下格式
# ## [日期时间] [K100 DCU] [Profiler 重采 12-token]
# **运行命令**: ...
# **环境**: ROCm 6.3.3, PyTorch 2.9.0, git commit, ...
# **原始输出**: ...
# **数据映射**: 表 25/26/27 的新旧值对比
```

## 每轮迭代工作流

### Step 0: 从 review 文件选问题

读取 `paperdoc/Metainfer-review.md`，筛选 DCU §6.2 相关且未标记 `<!-- FIXED -->` 的问题。

**优先级**:
1. 数据类（P66/P69/P119/P120/P148/P160）— 需要 profiler 重测
2. 术语/逻辑（P67/P68/P73/P106/P107/P109/P127/P129/P131/P151/P154/P156/P164/P167）
3. 方法论/跨平台（G18/G19/P166）

每次迭代处理 **1 个数据类问题** 或 **2-3 个同类文本问题**。

### Step 1: 分析问题

确认：精确位置（.tex 行号）、根因、是否需要数据、影响哪些文件。

### Step 2: 判断是否需要重跑

- **需要 profiler**: P66/P69/P119/P120/P148/P160
- **不需要**: 其余纯文本/逻辑问题

**矩阵 benchmark 数据已确认，不需要重跑。**

### Step 3: 执行修改

**必须同时修改**:
- `paperdoc/MetaInfer.tex` (§6.2 及相关 §6.3/§7.1)
- `paperdoc/MetaInfer技术文档（修订）.md` (对应章节)

**修改原则**:
- 数据修改：标注来源（profiler trace 文件名、时间戳）
- 措辞修改：学术、准确、不夸大
- 结构修改：保持 LaTeX begin/end 平衡

**注意**: DCU 机器无 xelatex，无法编译 PDF。修改后以 `.tex` 源码和 `.md` 预览为验证目标。

### Step 4: ${CLAUDE_CLI} -p 独立审稿

**这是关键步骤**。每次修改后，启动独立 agent 以审稿人视角验证：

```
你是严谨的学术审稿人。我刚修改了 MetaInfer.tex §6.2（K100 DCU 适配）的以下问题：
[问题编号和描述]

请审核：
1. 修改是否正确解决了问题？
2. 是否引入了新的矛盾或错误？
3. 修改后的文本是否学术、准确、自洽？
4. 如果涉及数据修改，数字来源是否可追溯？
5. 同一个数据是否在 §6.2、§6.3、§7.1 中保持一致？

只审核本次修改涉及的内容，不审核全文。

输出格式：
PASS: 修改正确
FAIL: [具体问题描述]
NEW: [新发现的问题，编号自动递增]
```

### Step 5: 写入 AUDIT

每次 ${CLAUDE_CLI} -p 审核后（PASS 或 FAIL），追加到 `paperdoc/Metainfer-AUDIT.md`：

```markdown
## [日期时间] [K100 DCU] [问题编号] [PASS/FAIL]

**问题原文**: > ...
**修改前状态**: ```latex ... ```
**修改思路**: 为什么选择此方案
**修改后状态**: ```latex ... ```
**${CLAUDE_CLI} -p 审核结论**: PASS/FAIL
**审核原文**: > ...
**残留风险**: ...
```

### Step 6: 标记 review 文件

在已修复问题行后添加 `<!-- FIXED: YYYY-MM-DD, K100 DCU, [说明] -->`。保留原条目。

### Step 7: 输出本轮摘要

记录本轮做了什么、验证结果。

## 数据采集 SOP

### B. K100 DCU Profiler Tracing

```bash
cd $AGENT_INFER_ROOT
source .env_agent_infer

# === Profiler Tracing（12 tokens，对齐 A800 §5.3） ===

# 1. 修改 test_phase11_202_profiler.sh 中的 MAX_TOKENS
#    MAX_TOKENS = 24 → MAX_TOKENS = 13（12 decode steps）

# 2. 运行 profiler
export NCCL_DEBUG=ERROR
export VLLM_LOGGING_LEVEL=ERROR
bash scripts/test_phase11_202_profiler.sh

# 3. 获取 GPU kernel 平均耗时
cat perf_iteration/trace_target/key_avg.txt

# 4. 归档
mkdir -p paperdoc/K100测量数据/profiler_trace/
cp perf_iteration/trace_target/trace_rank0.json paperdoc/K100测量数据/profiler_trace/
cp perf_iteration/trace_target/key_avg.txt paperdoc/K100测量数据/profiler_trace/
```

### C. 数据验证清单

重跑后逐项检查：
1. 子项之和 = 总项（误差 < 1%）
2. GPU 组件时间互斥（无重叠计时）
3. hipGetDeviceProperties 调用次数 ≤ 1
4. TPOT 方差合理（CV >0.5%，不异常稳定）
5. Prefill 和 KV cache 分配时间可区分

## 第 3 批：纯文本/逻辑修复清单

以下问题无需数据重测，按类型分组批量修改：

**术语统一**:
- P67: `glen=256` → `g=256`
- P68: `CustomAR 约 5× 于 RCCL` → `CustomAR 耗时约为 RCCL 的 5 倍`
- P107/P129: CustomAR 归属 → MetaInfer 专有 P2P kernel，vLLM 用 RCCL all-reduce
- P151: `CUDA JIT` → `GPU kernel 即时编译`（ROCm 平台术语）

**逻辑修正**:
- P73: 补充 vLLM torch.compile inductor 后端配置说明
- P106: 删除 "Phase 11 Stage 2"，改为 "三轮基于 profiler trace 比对的迭代优化"
- P109: 区分预分配 vs 实际使用显存，补充显存效率分析
- P127: 统一双协议叙述 — 矩阵测试为正式评估（+19%），24-token 为快速验证（+14.4%）
- P131: 注明 prompt 构造方法（中文文本块重复填充）
- P154: 修正核心发现 1 — 删除"融合 kernel 三种平台功能一致"（被 CustomAR DCU 失败证伪）
- P156: 统一两套测试 prompt 内容来源说明
- P164/P167: 报告 TTFT 测量精度（均值 ± 标准差或 95% CI），增加 round 数或在正文中讨论精度限制

**跨平台方法论**:
- G18: 解释 DCU 双协议关系（24-token 为快速验证预实验，矩阵为正式评估）
- G19: 在 §6.2 和 §6.3 描述 DCU benchmark 协议（矩阵 3×3, rounds=3, warmup=1）
- P166: 在 DCU 表注中标注 tok/s 为 E2E 吞吐（含 prefill），与 A800 的 Decode 吞吐定义不同

## /loop 迭代审核机制

当用户触发 `/loop /fix-paper ch6` 时：

1. **第 1 轮**: 用 Agent spawn ${CLAUDE_CLI} -p 以审稿人角色审核 §6.1--§6.3（DCU 部分）
2. **收集新问题**: 将审稿人发现的新问题写入临时文件 `paperdoc/K100测量数据/review_new_findings.md`
3. **逐条修复**: 按问题顺序逐条修改 .tex + .md
4. **第 2 轮**: 重新启动审稿 agent，审核修复后的内容
5. **退出条件**: 连续 2 轮审核无新增问题

每轮审核的 agent 提示词：
```
你是严谨的学术审稿人，正在审核一篇中文论文 MetaInfer.tex 的 §6.2（K100 DCU 适配）及关联的 §6.3/§7.1。

请逐段阅读，找出以下类别的问题：
1. 数据自洽性：子项之和是否等于总项？不同表中的同一数字是否一致？
2. 定义准确性：每个缩写是否在首次出现处定义？术语使用是否前后一致？
3. 逻辑严密性：结论是否被数据支持？是否存在过度声称？
4. 跨平台一致性：DCU 数据的定义、测试协议、报告格式是否与方法论描述一致？
5. 措辞学术性：是否有口语化、营销化、情绪化表达？

对每个发现的问题，给出：
- 编号（NEW-1, NEW-2, ...）
- 位置（行号或章节）
- 问题描述
- 修改建议

不要提出与前 14 轮审阅已发现并修复的问题重复的建议。
```

## 环境信息

### K100 DCU 机器（当前 — 环境快照，不同机器可能不同）
- **硬件**: 海光 K100 DCU (gfx928), 4× GPU PCIe 4.0 P2P, 65GB VRAM
- **软件**: ROCm 6.3.3, RCCL 2.22.3, Python 3.10.12, PyTorch 2.9.0（版本号从环境检测获取，此处为记录快照）
- **引擎**: `$AGENT_INFER_ROOT` (git `feature/refactor-konwledge`, commit `f7961d5`)
- **vLLM**: `/workspace/vllm-v0.15.1-dev` (editable install，机器特定路径)
- **模型**: Qwen3-8B Dense (`$MODEL_DIR`)
- **编译**: **无 xelatex**（DCU 机器无 LaTeX 发行版），以 .tex 源码 + .md 预览为验证目标

### 通用
- Python: 系统 Python 3.10.12（路径取决于系统配置，此处为 `/usr/bin/python`）
- 环境变量: `source $AGENT_INFER_ROOT/.env_agent_infer`
- torchrun: TP=4, MASTER_PORT 避免冲突

## 常见问题速查（DCU §6.2 专用）

| 编号 | 类型 | 简述 | 需要重测 |
|------|------|------|---------|
| P66 | 数据 | Profiler 23 steps → 12 token 对齐 A800 | ✅ |
| P69 | 数据 | hipGetDeviceProperties 每 step 20.4ms | ✅ |
| P119 | 数据 | TPOT 异常稳定 CV=0.25% | ✅ |
| P120 | 数据 | TTFT 随 g 变化不符合定义 | ✅ |
| P160 | 数据 | 表25 组件之和 > Self CUDA Total | ✅ |
| P148 | 数据 | TPOT +17% 根因不完整 | ✅ |
| P67 | 文本 | glen=256 → g=256 | ❌ |
| P68 | 文本 | CustomAR 约 5× 于 RCCL 语法错误 | ❌ |
| P73 | 文本 | GEMM 差距与 torch.compile 脱节 | ❌ |
| P106 | 文本 | Phase 11 Stage 2 无前文定义 | ❌ |
| P107 | 文本 | 表25 vLLM 列出现 CustomAR | ❌ |
| P109 | 文本 | 显存差 10× 解释不足 | ❌ |
| P127 | 文本 | 两个吞吐优势数字未区分 | ❌ |
| P129 | 文本 | §6.2 配置描述 CustomAR 归因错误 | ❌ |
| P131 | 文本 | prompt 构造方法引偏置 | ❌ |
| P154 | 逻辑 | 核心发现 1 被 CustomAR 失败证伪 | ❌ |
| P156 | 文本 | 两套测试 prompt 内容不一致 | ❌ |
| P164 | 数据 | TTFT 非单调（DCU-engine） | ❌(协议层) |
| P167 | 数据 | TTFT 非单调（vLLM 基线）同证 | ❌(协议层) |
| P166 | 逻辑 | tok/s 定义跨平台不一致 | ❌ |
| G18 | 方法论 | DCU 双协议未说明关系 | ❌ |
| G19 | 方法论 | 三平台协议互不兼容 | ❌(部分) |

## 四类文件协同

```
MetaInfer.tex                  ← 论文本体（修改目标）
MetaInfer技术文档（修订）.md   ← 技术文档（同步修改）
Metainfer-review.md            ← 导师/审稿人问题清单（标记 FIXED）
Metainfer-DATALOG.md           ← 实验原始数据（profiler 重测写入）
Metainfer-AUDIT.md             ← 修改决策记录（每次 ${CLAUDE_CLI} -p 审核后写入）
Metainfer-PLAN.md              ← 总修改计划与进度跟踪
K100测量数据/
├── benchmark_matrix_3x3.md    ← 矩阵数据（已有）
├── profiler_trace/            ← 新 profiler trace 归档
└── review_new_findings.md     ← /loop 迭代审稿新发现
```

## 注意事项

1. **矩阵 benchmark 已确认**：3×3 矩阵数据在 `K100测量数据/benchmark_matrix_3x3.md`，直接使用。
2. **Profiler 对齐 A800**：12 tokens 输出，12 decode steps。
3. **原始 trace 必须归档**：每次 profiler 运行后 trace 文件存入 `K100测量数据/profiler_trace/`。
4. **同时修改 .tex 和 .md**：两个文件的数据、措辞、结构必须同步。
5. **无法编译 PDF**：DCU 无 xelatex，验证以 ${CLAUDE_CLI} -p 纯文本审核为准。
6. **一次修一类问题**：数据类单独修，文本类可 2-3 个同类一起修。
7. **DATALOG 和 AUDIT 是给导师看的**：完整记录每一次数据采集和修改决策。
8. **诚实优于完美**：宁可承认精度限制（如 TTFT 测量精度 ±3-7ms），也不抹平数据呈现。
