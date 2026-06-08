# Phase 触发词速查

## 快速开始

1. `cd` 到本项目根目录
2. 让 agent **先阅读 `CLAUDE.md`** 理解三代理对抗协作流
3. 输入下方触发词即可启动构建

```
# 示例：在新会话中启动 Phase 1-4
cd /path/to/meta-infer
# 然后对 agent 说："请阅读 CLAUDE.md，然后执行 /phase1-4"
```

> **注意**：首次使用需要配置 Python 环境和模型路径。启动任一 Phase 后 agent 会自动询问 `MODEL_DIR` 和 `PYTHON_PATH`，写入 `.env_agent_infer` 供后续会话复用。

## 触发词

| 触发词 | Phase | 构建内容 |
|--------|-------|---------|
| `/phase1-4` | 1-4 | 数值基元 + TP 通信 + TP 线性层 + TP Embedding |
| `/phase5` | 5 | Attention + KV Cache（最高错误密度） |
| `/phase6` | 6 | MLP + Decoder Layer |
| `/phase7-8` | 7-8 | 权重加载 + 框架外壳（Scheduler/Sequence/Sampler/BlockManager） |
| `/phase9-10` | 9-10 | 引擎集成 + E2E 验收 |
| `/phase11` | 11 | 性能优化（知识规则 + Tracing 对齐） |

## Skill 文件位置

```
.claude/skills/phase1-4/SKILL.md
.claude/skills/phase5/SKILL.md
.claude/skills/phase6/SKILL.md
.claude/skills/phase7-8/SKILL.md
.claude/skills/phase9-10/SKILL.md
.claude/skills/phase11/SKILL.md
```

## 协作流角色

| 角色 | 职责 | 约束 |
|------|------|------|
| implementer | 写代码 | 不跑测试，只输出 SUBMITTED 报告 |
| spec-reviewer | 契约审查 | Shell `claude -p` 独立进程，对照 blueprint 逐条核验 |
| verification | 跑测试验收 | Shell `claude -p` 独立进程，L1+L2+L3 分层验收 |

红线：spec ❌ 时 verification 不启动；主 Agent 禁止降级子代理结论；三个子代理 PID 必须互不相同。
