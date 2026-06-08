# Phase 11：性能优化（两阶段）

## 触发词

`/phase11`

## 角色

你是主 Agent。按 CLAUDE.md 的 spawn 协议执行。本次是性能优化 Phase——**不改变功能行为，只改实现方式**。

Phase 11 分两阶段：先应用知识规则（快速见效），再启动 tracing 对齐流程（多轮迭代）。

---

## 阶段一：知识规则优化（O1-O9 审计）

应用 O1-O9 性能规则（详见 AGENT_SKILL.md §I），按优先级分层：

### 审计门禁（O1-O6，CRITICAL/HIGH，必须全部 PASS）

| 规则 | 内容 | 审计方式 |
|------|------|---------|
| O1 | `@torch.inference_mode()` 装饰 forward + forward_decode | grep 应有 2 匹配 |
| O2 | 零 `.item()` GPU 同步 | grep 应为零匹配 |
| O3 | 预分配 buffer（`_q_norm_out`, `_k_norm_out`, `_silu_out`） | grep 应有匹配；`empty_like` 零匹配 |
| O4 | block_table arange 初始化 | grep 确认 arange |
| O5 | prefill KV 直接赋值（非 index_copy_） | grep index_copy_ 仅在 decode |
| O6 | register_buffer 完整声明 | grep 应 ≥ 6 |

### 补充优化（O7-O9，LOW）

O7 懒 contiguous、O8 view 非 reshape、O9 消除中间 tensor。

### 阶段一执行步骤

1. **STEP-AUDIT**：逐条执行 O1-O6 审计检查，记录 PASS/FAIL
2. **STEP-FIX**：每条 FAIL 定位源码修复
3. **STEP-REAUDIT**：重新审计直到 O1-O6 全部 PASS
4. **STEP-BENCHMARK**：跑 `test_phase11_throughput.py` 验证吞吐

修复时改动小（≤10 行）可走快速修复路径跳过 spec-reviewer。

---

## 阶段二：Tracing 对齐优化（多轮迭代）

阶段一完成后，仅凭知识规则通常无法达到最优性能。阶段二启动 `.claude/roles/performance_alignment_by_tracing.md` 方法论进行多轮 profiling→fix→verify 迭代。

### 基线目标（Target Baseline）

**目标框架**：vLLM Eager 模式（无 CUDA Graph，与当前 nocompile 模式对齐）。

**参考 profiling 脚本**：`ref_projects/vllm/examples/offline_inference/simple_profiling.py`

若 `ref_projects/` 不存在或该脚本缺失，使用 Python 环境中已安装的 vLLM 包直接 profiling：

```python
import torch
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

MODEL_DIR = os.environ.get("MODEL_DIR", "")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))

# vLLM Eager 模式 profiling（无 CUDA Graph）
llm = LLM(
    model=MODEL_DIR,
    tensor_parallel_size=TP_SIZE,
    enforce_eager=True,          # 禁用 CUDA Graph
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

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
prompt = "苏州园林的特点是"
outputs = llm.generate([prompt], sampling_params)
print(f"vLLM eager output: {outputs[0].outputs[0].text}")
```

### 迭代流程

按 `performance_alignment_by_tracing.md` 的完整方法论执行：

```
while 目标吞吐率 < vLLM Eager 基线:
    1. 运行 profiler — 获取目标框架最新 trace
    2. 对比 key_avg.txt 与 vLLM 基线 — 识别剩余瓶颈
    3. 按优先级排序瓶颈 — 架构 > 同步 > 分配 > 计算
    4. 选最高优先级瓶颈，设计方案 — 只改目标框架
    5. 修改代码 → benchmark → 验证正确性 → 记录
    6. 写入 ./phase_report/PHASE11_ROUND<N>_*.md
    N += 1
```

profiler trace 目录结构：
```
perf_iteration/
├── profiler_compare.py           # 统一 profiler 脚本（可切换两个框架）
├── ROUND_0_BASELINE.md           # 基线：目标框架 vs vLLM Eager
├── ROUND_1_*.md                  # 后续轮次
├── trace_baseline/               # vLLM 的 profiler trace
└── trace_target/                 # 目标框架的 profiler trace（每轮更新）
```

### 阶段二终止条件

- 目标框架吞吐率达到 vLLM Eager 模式的 95% 以上
- 或所有可识别瓶颈已消除
- 剩余差距记录为"不可消除差异"并说明原因

---

## Phase 11 Scripts

| 脚本 | 门禁 |
|------|------|
| `test_phase11_throughput.py` | 吞吐对齐 baseline |
| `test_phase11_profiler.sh` | 稳态 decode 无 cudaMalloc |

verif L2：**重跑 Phase 1-10 全部 26 个脚本**——性能优化不能引入正确性回归。

## 知识映射

- Blueprint：`todo_generation_playbook.phase_11_performance`（O1-O9 分层规则）→ `performance_gate`
- 阶段一实测验证：`notebooks-cn/07_improvementPlan/ROUND_1_BOTTLENECK_FIXES.md`
- 阶段二方法论：`.claude/roles/performance_alignment_by_tracing.md`
- 阶段二辅助：`.claude/roles/torch-inference-mode.md`
- vLLM 参考：`ref_projects/vllm/examples/offline_inference/simple_profiling.py`（或环境中的 vllm 包）

## 关键约束

- 不改变功能行为，只改实现方式
- 阶段一每应用一个 O 就跑 tp=4 多卡端到端正确性验证 + 测吞吐（增量验证）
- 如果某个 O 引入正确性回归 → 回滚 → 标记 BLOCKED → 继续其他 O
- 阶段二每次只改一类瓶颈，改完 benchmark + 验证正确性 + 记录
- 阶段二改前必须确认 baseline 正确性（temperature=0 输出字字对齐）
- 改动失败立即 `git checkout` 还原，记录错误原因
