# 推理框架编码 Skill 触发器

本文件只保留简短触发词，每个触发词对应 `.claude/skills/` 下的完整编码 prompt。

## 编码阶段

| 触发词 | 对应 Skill | Phase 范围 | 说明 |
|--------|-----------|-----------|------|
| `/phase1-4` | `phase1-4/SKILL.md` | Phase 1-4 | 数值基元 + TP通信 + TP线性层 + TP Embedding |
| `/phase5` | `phase5/SKILL.md` | Phase 5 | Attention + KV Cache（最高错误密度） |
| `/phase6` | `phase6/SKILL.md` | Phase 6 | MLP + Decoder Layer（最高错误密度） |
| `/phase7-8` | `phase7-8/SKILL.md` | Phase 7-8 | 权重加载 + 框架外壳 |
| `/phase9-10` | `phase9-10/SKILL.md` | Phase 9-10 | 引擎集成 + E2E 验收 |
| `/phase11` | `phase11/SKILL.md` | Phase 11 | 性能优化（审计-修复-再审计闭环） |
| `/phase-all` | `phase-all/SKILL.md` | Phase 1-11 | 全量一次性编码，依次执行所有 Phase |

## 使用方式

输入触发词，Agent 将自动加载对应的 skill 文件，获得完整的任务上下文、执行步骤、知识映射和高发错误清单。

例如：
```
/phase5
```

Agent 将读取 `.claude/skills/phase5/SKILL.md`，获得 Phase 5 的完整编码指导。

## 全量一次性编码

如需一次完成所有 Phase（从零构建整个推理框架），使用：

```
/phase-all
```

Agent 将依次执行 Phase 1-4 → Phase 5 → Phase 6 → Phase 7-8 → Phase 9-10 → Phase 11。
