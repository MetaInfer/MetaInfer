# Phase All：全量一次性构建

## 触发词

`/phase-all`

## 概述

一次性从头构建整个推理框架，依次完成 Phase 1-4 → Phase 5 → Phase 6 → Phase 7-8 → Phase 9-10 → Phase 11。每一 Phase 严格按照 `CLAUDE.md` 的三代理对抗协作流执行（implementer → spec-reviewer → verification），主 Agent 只做调度和防假 PASS 抽查。

---

## 你的角色

你是**主 Agent**——只做高层调度和抽查。依次读取每个 Phase 的 SKILL.md 获取**构建目标**，然后按 `CLAUDE.md` 的 spawn 协议完成 implementer → spec → verif → 汇总。

## 执行顺序

| 顺序 | Phase | 触发词 | 任务卡 |
|------|-------|--------|--------|
| 1 | Phase 1-4 | `/phase1-4` | `.claude/skills/phase1-4/SKILL.md` |
| 2 | Phase 5 | `/phase5` | `.claude/skills/phase5/SKILL.md` |
| 3 | Phase 6 | `/phase6` | `.claude/skills/phase6/SKILL.md` |
| 4 | Phase 7-8 | `/phase7-8` | `.claude/skills/phase7-8/SKILL.md` |
| 5 | Phase 9-10 | `/phase9-10` | `.claude/skills/phase9-10/SKILL.md` |
| 6 | Phase 11 | `/phase11` | `.claude/skills/phase11/SKILL.md` |

## 执行方式

### 首次启动（Phase 1-4 前）

若 `.env_agent_infer` 不存在，按 `CLAUDE.md` 启动流程询问用户 `MODEL_DIR` 和 `PYTHON_PATH`，写入 `.env_agent_infer`。

### 每 Phase 执行步骤

1. 读取对应 Phase 的 SKILL.md 了解构建目标
2. 按 `CLAUDE.md` 的 spawn 协议执行：implementer → spec-reviewer → verification → 主 Agent 汇总
3. 防假 PASS 抽查通过后进入下一个 Phase
4. 写入 `phase_report/PHASE<N>_MEMORY.md`
5. Git commit 代码 + 文档

### 关键约束

- 每个 Phase 必须通过所有门禁后才能进入下一个
- 抽查连续 5 次驳回 → 停止并向人类报告
- 禁止跳过任何 Phase 或篡改执行流程
- 每个 Phase 结束后重新读取下一 Phase 的 SKILL.md（防上下文遗忘）
