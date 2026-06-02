# inference-agent-system

一个自包含的知识包（prior knowledge package），让**白板 Agent**（无任何推理框架背景知识）能够独立生成 Qwen3-8B TP=4 纯 Eager 模式推理引擎的完整代码。

Agent 仅凭本包内的蓝图、SOP、测试合约和参考文档，不需要访问原始 meta-infer 源码。

## 快速开始

### 1. 环境准备

```bash
# GPU 机器，至少 1 张 A800/A100（4 张用于 TP=4）
# conda 环境需包含：
pip install torch flash-attn>=2.8.0 vllm
```

### 2. 模型权重

```
${MODEL_DIR}    ← 用户指定的模型目录（如 /data/models/qwen/Qwen3-8B）
```

### 3. 喂给 Claude Code

将以下 prompt 输入一个干净的 Claude Code 会话：

```
你现在在 inference-agent-system 目录下工作。
请先读取本目录的 CLAUDE.md 文件，理解你的角色、知识体系和执行规则。
然后按照 CLAUDE.md 中的指引，逐步加载其他知识文件，开始构建推理框架。
```

Agent 会按 10 个 Phase 逐层构建，每个 Phase 必须通过 scripts/ 测试合约才能进入下一 Phase。

## 包结构

```
inference-agent-system/
├── README.md                  ← 本文件
├── CLAUDE.md                  ← Agent 入口（角色、知识体系、对抗子代理协作流）
├── AGENT_SKILL.md             ← 执行 SOP（10 Phase + 编码铁律 + Debug 指南）
├── inference_blueprint.json   ← 架构知识图谱（唯一契约来源）
├── test_prompt.md             ← 白板 Agent 验证 Prompt（附审阅清单）
│
├── .claude/skills/            ← 3 个对抗子代理角色定义
│   ├── implementer-inference.md    — 只写代码，不跑测试
│   ├── spec-reviewer-inference.md  — 对照蓝图逐条核验
│   └── verification-inference.md   — 唯一裁定者，L1/L2/L3 三层验收
│
├── scripts/                   ← 26 个固定测试合约（先验知识，不可修改）
├── notebooks-cn/              ← 中文知识文档
├── ref_projects/              ← 参考工程源码（nano-vllm, vllm, sglang）
├── engine/                    ← 推理框架代码（Agent 生成产物）
├── llm_engine.py              ← 引擎主循环（Agent 生成产物）
├── openai_tp_server.py        ← OpenAI API 服务（Agent 生成产物）
```

## 三层知识体系

```
第一层：先验知识（人类写，Agent 只读）
  ├── inference_blueprint.json    ← 架构知识图谱（唯一契约来源）
  ├── AGENT_SKILL.md              ← 执行 SOP + 编码铁律
  └── scripts/                    ← 固定测试合约（26 个，不可修改）

第二层：生成产物（Agent 写，受第一层约束）
  ├── engine/**/*.py
  ├── llm_engine.py
  └── openai_tp_server.py

第三层：验收证据（Agent 运行，不可伪造）
  ├── profiler trace
  ├── HCU/VRAM 监控
  └── benchmark JSON
```

## 10 Phase 构建流水线

```
Phase 1: 数值基元    → Phase 2: TP通信    → Phase 3: TP线性层
Phase 4: TP Embedding → Phase 5: Attention  → Phase 6: Decoder+MLP
Phase 7: 权重加载     → Phase 8: 框架外壳  → Phase 9: 引擎集成
                                                    ↓
                                            Phase 10: E2E验收
```

每个 Phase 对应固定的 scripts/ 测试合约（共 26 个，全部 PASS 才算交付）。

## Agent 如何工作

### 对抗子代理协作流

代码生成由三个独立角色完成，物理隔离，互不信任：

```
主 Agent（读蓝图 → 拆 Task）
  │
  ▼
implementer（Agent 工具 spawn）
  写代码 + 自读 diff → SUBMITTED
  │
  ▼
spec-reviewer（Shell claude -p 独立进程，新 PID）
  对照蓝图逐条核验契约
  ├── ❌ FAIL → 直接打回 implementer，verification 不启动
  └── ✅ PASS → 进入 verification
        │
        ▼
      verification（Shell claude -p 独立进程，新 PID）
        L1: 运行 scripts/ 全部测试
        L2: 跨 Phase 回归
        L3: profiler + HCU 证据
        ├── ❌ FAIL → 打回 implementer
        └── ✅ PASS → Phase 交付
```

### 核心铁律

| 铁律 | 内容 |
|------|------|
| implementer 不自证清白 | 只写代码 + 自读 diff，不跑测试，不判 PASS。状态是 SUBMITTED |
| 审查串行执行 | spec-reviewer 先，通过后才到 verification。不并行 |
| verification 是唯一裁定者 | 只有 verification 有权宣判 Phase 交付 |
| 主 Agent 是信使非裁判 | 禁止降级/修改/解释子代理的审查结论 |
| scripts/ 不可变 | 测试不过 → 改实现代码，不改脚本 |
| PID 互不相同 | 三个子代理的 PID 必须不同——物理隔离的硬证据 |

## 验证结果

本包已通过 Phase 1 白板 Agent 验证：

| 角色 | PID | 结论 |
|------|-----|------|
| implementer | 27224 | SUBMITTED |
| spec-reviewer | 29163 | ✅ PASS（11 个契约节点逐条核验） |
| verification | 30845 | ✅ PASS（2/2 scripts 全绿，8/8 KERNEL 测试通过） |

三个 PID 互不相同 → 子代理物理隔离已确认。Phase 1 交付完成。

## 构建范围

- 模型: Qwen3-8B
- 并行: TP=4
- 模式: 纯 Eager（nocompile, no CUDA Graph）
- 批次: B=1 单序列
- DeepSeek-V2 支持: 蓝图已包含 MLA+MoE 知识，独立扩展阶段
