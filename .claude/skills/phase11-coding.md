# Skill: Phase 11 编码（性能优化）

## 触发词

`/inference:phase11` 或 `/phase11`

## 概述

Phase 11 性能优化——不改变功能行为，只改实现方式。通过 spawn phase-runner 子代理执行审计闭环，主 Agent 只做调度和 benchmark 抽查。

---

Phase 1-10 已完成，TP=4 推理正确性字字对齐。当前吞吐仅 ~10 tok/s，目标 > 50 tok/s。

## 环境

- **必须先执行** `source .env_agent_infer` 加载 MODEL_DIR 和 PYTHON_PATH

## 你的角色

你是**主 Agent**——只做高层调度和 benchmark 抽查。通过 spawn phase-runner 子代理（审计模式）执行 Phase 11 的 audit → fix → reaudit → benchmark 闭环，你只看到结构化摘要，保持上下文轻量。

## 强制审计闭环（不可跳过）

Phase 11 不按传统 implementer→spec→verif 流程，而是**审计-修复-再审计**循环：

### STEP-AUDIT

逐条执行 `inference_blueprint.json > performance_optimization` 中 O1-O6 的 `audit_check` 命令，记录每条 PASS/FAIL。

```bash
# O1: @torch.inference_mode() — 贡献 82%
grep -c '@torch.inference_mode' engine/models/qwen.py  # 应 =2

# O2: 全文件零 .item()
grep '\.item()' llm_engine.py engine/models/qwen.py  # 仅注释中允许

# O3: 预分配 buffer + 禁止 empty_like
grep '_q_norm_out\|_k_norm_out\|_silu_out' engine/models/qwen.py  # 应有匹配
grep 'empty_like' engine/models/qwen.py  # 应为零

# O4: block_table arange
grep '_block_table.*arange' engine/models/qwen.py

# O5: prefill 直接赋值, index_copy_ 仅在 decode
grep 'index_copy_' engine/models/qwen.py  # 应仅在 forward_decode 出现

# O6: register_buffer 数量
grep -c 'register_buffer' engine/models/qwen.py  # 应 ≥8
```

O7-O9 为 LOW（补充优化，正确但非致命），仅记录不阻塞。

### STEP-FIX

每条 FAIL 的项目，定位到对应 Phase 构建的源码文件（见 `apply_to` 字段），修改代码使 audit 通过。

### STEP-REAUDIT

修复后重新跑全部 O1-O6 audit_check，直到全部 PASS。

### STEP-BENCHMARK

全部 audit 通过后，跑 benchmark 验证吞吐：

```bash
bash scripts/test_phase11_throughput.py
bash scripts/test_phase11_profiler.sh
```

不达标（≤ 50 tok/s）则回 STEP-FIX 做性能诊断（profiler 查 cudaMalloc、aten::item、aten::clone）。

### STEP-DONE

O1-O6 全部 PASS + throughput > 50 tok/s → Phase 11 完成。写 ./phase_report/PHASE11_SUMMARY.md。

## O1-O9 优化规则（优先级排序）

**审计门禁（O1-O6，必须 PASS）：**

| 规则 | 优先级 | 描述 |
|------|--------|------|
| **O1** @torch.inference_mode() | CRITICAL | forward() 和 forward_decode() 加装饰器 — 占提升 82% |
| **O2** 零 .item() GPU sync | CRITICAL | get_num_free_blocks() 返回常量、s.kv_len += 1、slot_mapping arange、forward() decode 用 past_key_values[0] + 1 |
| **O3** 预分配 buffer | CRITICAL | _q_norm_out/_k_norm_out/_silu_out register_buffer；通用路径禁止 empty_like 改用 torch.empty(显式参数) |
| **O4** block_table arange | HIGH | torch.arange(max_blocks).unsqueeze(0) 非 zeros |
| **O5** prefill KV 直接赋值 | HIGH | kc_flat[slot_mapping] = k_flat；index_copy_ 仅在 decode |
| **O6** register_buffer 完整声明 | HIGH | persistent=False, dtype=...，≥8 个 |

**补充优化（O7-O9，仅记录不阻塞）：**

| 规则 | 优先级 | 描述 |
|------|--------|------|
| **O7** 懒 contiguous | LOW | .contiguous() 仅在内核输入前，删除 vLLM kernel 会报错 |
| **O8** view 非 reshape | LOW | .view() 零拷贝，reshape 可能隐式 contiguous |
| **O9** 消除中间 tensor | LOW | O3 预分配 buffer 是其具体实现 |

详细规则见 `inference_blueprint.json > performance_optimization` 和 `AGENT_SKILL.md §I`。

## Phase 11 Scripts

| 脚本 | 内容 |
|------|------|
| test_phase11_throughput.py | TP=4 nocompile 吞吐 > 50 tok/s |
| test_phase11_profiler.sh | aten::item < 10ms + 无 cudaMalloc |

## verification 要求

- L1: 运行 Phase 11 scripts（throughput + profiler）
- L2: 跨 Phase 回归——**重跑 Phase 1-10 全部 26 个 scripts/**！性能优化不能引入正确性回归

## 执行方式

### 步骤 1：spawn phase-runner（审计模式）

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase 11 runner (audit)",
  prompt: """
Phase 11: 性能优化（审计模式）。
读取 .claude/skills/phase-runner.md 了解你的角色边界（使用模式 C：审计闭环）。
读取 .claude/skills/phase11-coding.md 了解审计规则和 O1-O9 优化清单。
执行 STEP-AUDIT → STEP-FIX → STEP-REAUDIT → STEP-BENCHMARK 完整闭环。
"""
)
```

phase-runner 返回结构化摘要后，进入步骤 2。

### 步骤 2：主 Agent 防假 PASS 抽查

从 Phase 11 scripts 中随机抽 1 个（或两个都跑），亲自验证：

```bash
# 抽查 benchmark（phase-runner 的吞吐数据）
python scripts/test_phase11_throughput.py 2>&1
# 或抽查 profiler
bash scripts/test_phase11_profiler.sh 2>&1
```

读取 `./phase_report/PHASE11_IMPLEMENTER_REPORT.md` 中 phase-runner 记录的 benchmark 数据，与亲自跑的结果比对：
- **一致**（吞吐差距 ≤ 2%）✅ → Phase 11 交付，写 `./phase_report/PHASE11_SUMMARY.md`
- **不一致** ❌ → 写 `./phase_report/PHASE11_SPOT_CHECK_FAIL.md` → 回到步骤 1（重试模式）：

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase 11 runner (RETRY audit)",
  prompt: """
Phase 11 RETRY。
读取 ./phase_report/PHASE11_SPOT_CHECK_FAIL.md 了解失败原因。
读取 .claude/skills/phase-runner.md 了解你的角色边界（使用模式 D：审计重试）。
读取 .claude/skills/phase11-coding.md 了解审计规则。
从 STEP-AUDIT 重新开始完整审计闭环（不得跳过任何步骤）。
"""
)
```

重试后再次抽查。连续 5 次驳回 → 停止，向人类报告。

## 关键约束

- 主 Agent 只做调度 + 抽查，不亲自 execute 审计闭环
- STEP-AUDIT → STEP-FIX → STEP-REAUDIT → STEP-BENCHMARK → STEP-DONE 不可跳过
- O7 的 .contiguous() 不能删除——vLLM kernel 要求输入连续（ROUND_1 Error #2 已验证）
- 不改变功能行为，只改实现方式
- 如果某个 O 引入正确性回归 → 回滚 → 标记为 BLOCKED → 继续其他 O
- 连续 5 次驳回 → 停止，向人类报告全部 5 次驳回记录
