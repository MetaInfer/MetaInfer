# Skill: 全量编码（Phase 1-11）

## 触发词

`/phase-all` 或 `/inference:phase-all`

## 概述

一次性从头构建整个推理框架，依次完成 Phase 1 → Phase 11。每个 Phase 通过 spawn phase-runner 子代理完成内部对抗审查，主 Agent 只做调度和防假 PASS 抽查。

---

## 你的角色

你是**主 Agent**——只做高层调度和抽查。依次加载每个 Phase 的 coding skill 并按其步骤执行，保持上下文轻量。

## 执行顺序

依次完成以下 Phase，每一 Phase 严格按对应 skill 文件的 spawn 协议执行：

| 顺序 | Phase | 触发词 | Skill 文件 |
|------|-------|--------|-----------|
| 1 | Phase 1-4 | `/phase1-4` | `.claude/skills/phase1-4/SKILL.md` |
| 2 | Phase 5 | `/phase5` | `.claude/skills/phase5/SKILL.md` |
| 3 | Phase 6 | `/phase6` | `.claude/skills/phase6/SKILL.md` |
| 4 | Phase 7-8 | `/phase7-8` | `.claude/skills/phase7-8/SKILL.md` |
| 5 | Phase 9-10 | `/phase9-10` | `.claude/skills/phase9-10/SKILL.md` |
| 6 | Phase 11 | `/phase11` | `.claude/skills/phase11/SKILL.md` |

## 执行方式

Phase 1-4 特殊处理：**先执行步骤 0（环境配置）**，然后依次完成 Phase 1、2、3、4。

后续 Phase 5-11：依次读取对应 skill 文件，按其中的 spawn→抽查→写 SUMMARY 流程执行。

关键约束：
- 每个 Phase 必须通过抽查后才能进入下一个
- 抽查连续 5 次驳回 → 停止并向人类报告
- 禁止跳过任何 Phase 或偷换执行流程
