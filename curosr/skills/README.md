# curosr/skills 使用说明

本目录用于固化「推理框架生成」的**契约**与**执行 SOP**，供后续 Agent / Sub-agents 复用。

## 文件说明

- `inference_blueprint.json`
  - 系统规格与接口契约（组件职责、参考文档/源码、Tensor shape/dtype/device、逻辑约束）。
  - 任何实现前，必须先读取并按契约开发。

- `AGENT_SKILL.md`
  - 子 Agent 执行手册（TDD、排障、完成标准、日志要求）。
  - 用于约束“先测后写、逐组件封口、失败不串联”。

## 快速使用（单 Agent）

1. 读取 `inference_blueprint.json`，锁定本次要实现的组件。
2. 读取 `AGENT_SKILL.md`，按 Workflow 执行。
3. 先写对应 `tests/test_xxx.py`，再写实现。
4. 每改一个组件立刻跑对应 pytest。
5. 最后跑：
   - `python -m pytest tests/test_real_inference.py -s -q`

## 多 Sub-agents 协作建议

建议按职责拆分并行执行：

- **agent-a（契约分析）**
  - 解析 `inference_blueprint.json`，输出本轮任务的 shape/接口检查清单。
- **agent-b（参考溯源）**
  - 对照 `ref_code` 抽取最小实现路径（不要搬运复杂分支）。
- **agent-c（测试先行）**
  - 先新增/更新 `tests/`，覆盖契约与边界条件。
- **agent-d（实现落地）**
  - 按测试实现组件代码，确保日志齐全。
- **agent-e（集成验收）**
  - 执行端到端测试，检查真实模型输出与性能日志。

并行原则：
- 可并行：契约分析、参考溯源、测试草案。
- 串行：最终实现合并、修复冲突、端到端验收。

## 推荐任务 Prompt（可直接复制）

```text
你是“LLM 推理框架工程师 + Agent 协作专家”。请在 meta-infer 下完成任务，并严格遵循：

1) 强制先读：
- curosr/skills/inference_blueprint.json
- curosr/skills/AGENT_SKILL.md
- notebooks/tasks.md
- notebooks/MEMORY.md
- notebooks/00_overview/README.md

2) 执行方式：
- 使用 sub-agents 并行：
  - A: 契约解析与检查清单
  - B: ref_code 最小实现路径溯源
  - C: 测试先行（TDD）
  - D: 实现与修复
  - E: 集成验收
- 每个组件必须“先测试、后实现、立即 pytest”。
- 未通过当前组件测试，禁止改动其他组件。

3) 实现要求：
- 严格按 inference_blueprint.json 的 data_flow_contracts 开发。
- 保留关键日志：shape、block 使用、调度信息、finish reason、输出文本。
- 若出现 shape mismatch/OOM，按 AGENT_SKILL.md 故障流程处理。

4) 完成标准：
- 运行 python -m pytest tests/test_real_inference.py -s -q 通过。
- 终端可见真实 16B 模型可读输出（例如 “The capital of France is ...”）。

请先给出你的执行清单（按组件和测试拆分），然后开始实施。
```

## 主控 Orchestrator Prompt（并行 + 串行门禁，可直接复制）

```text
你现在是“主控 Orchestrator Agent”。目标：在 meta-infer 下完成推理框架任务，并严格执行“并行分析 + 串行门禁”。

【强制读取】
1) curosr/skills/inference_blueprint.json
2) curosr/skills/AGENT_SKILL.md
3) notebooks/tasks.md
4) notebooks/MEMORY.md
5) notebooks/00_overview/README.md

【子代理编排】
第一阶段（并行，只产出计划与草案，不改核心实现）：
- 调用 /contract-checker：输出契约检查清单（shape/dtype/device/状态机/边界）
- 调用 /ref-tracer：输出最小参考实现路径与禁止迁移项
- 调用 /tdd-test-writer：输出测试设计与待新增 tests 文件草案

并行阶段结束后，你必须先汇总三方结果，形成“统一执行计划 V1”，包含：
- 组件拆分顺序
- 每个组件对应测试文件
- 每个组件的硬门禁契约

第二阶段（串行门禁，按组件逐个推进）：
对每个组件按以下顺序循环：
1) 先落测试（来自 tdd-test-writer 草案）
2) 调用 /impl-coder 实现该组件最小改动
3) 立即运行对应 pytest 单测
4) 若失败：仅允许修复当前组件，禁止改其他组件
5) 单测 100% 通过后，才可进入下一个组件

第三阶段（集成验收）：
- 调用 /integration-verifier 执行：
  - python -m pytest tests/test_real_inference.py -s -q
- 必须检查终端中出现真实 16B 模型可读输出（例如 “The capital of France is ...”）。

【硬性门禁】
- 任一组件单测失败：停止跨组件改动
- 未满足 inference_blueprint.json 契约：停止实现并回退修正
- 出现 OOM/shape mismatch：按 AGENT_SKILL.md Troubleshooting 流程处理

【输出格式要求】
每一轮仅输出以下四段：
1) 当前阶段（并行/串行/验收）
2) 本轮改动文件
3) 测试命令与结果
4) 是否通过门禁（通过/未通过 + 下一步）
```

