# MetaInfer v3 — 全项目流程图

## 一、全局架构鸟瞰

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        MetaInfer v3 全局架构                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌──────────────────────────────────────────────────────────────┐     ║
║   │                    第 一 层：先验知识库                         │     ║
║   │                                                                │     ║
║   │  notebooks-cn/                                                 │     ║
║   │  ├── 00_contracts/     ← 11 个 API 契约（硬约束，人类确认才能改）│     ║
║   │  ├── 00_overview/      ← 项目总览                              │     ║
║   │  ├── 01_framework_design/ ← 框架设计文档                       │     ║
║   │  ├── 02_model_specifics/ ← 模型族文档（回路 B/C 可扩写）        │     ║
║   │  ├── 03_operators/     ← 算子知识                              │     ║
║   │  ├── 04_parallel_strategies/ ← TP/EP 策略（回路 B/C 可扩写）    │     ║
║   │  ├── 05_inference_service/ ← 推理服务                          │     ║
║   │  ├── 06_experience/    ← 调试经验（回路 B/C 自动追加）          │     ║
║   │  └── 07_improvementPlan/ ← 优化方案（回路 B/C 自动追加）        │     ║
║   │                                                                │     ║
║   │  AGENT_SKILL.md          ← 执行 SOP + 编码铁律                  │     ║
║   │  CLAUDE.md               ← 全局入口 + 启动流程 + spawn 协议     │     ║
║   │  scripts/                ← 28 个固定测试合约（只读，不可修改）   │     ║
║   └──────────────────────────────────────────────────────────────┘     ║
║                              │  │                                       ║
║              回路 B 写入 ◄──┘  └──► 读取                               ║
║              回路 C 写入 ◄──┘                                           ║
║                                                                          ║
║   ┌──────────────────────────────────────────────────────────────┐     ║
║   │                    第 二 层：Agent 角色群                       │     ║
║   │                                                                │     ║
║   │  .claude/roles/                                                │     ║
║   │  ├── implementer-inference.md       ← 写代码，不跑测试          │     ║
║   │  ├── spec-reviewer-inference.md     ← 契约核验，不跑测试        │     ║
║   │  ├── verification-inference.md      ← 跑测试，唯一宣判 PASS     │     ║
║   │  ├── experiment-summarizer.md       ← 回路 C: 实验→知识        │     ║
║   │  ├── explorer.md                    ← 回路 B: 搜索论文/HF/源码 │     ║
║   │  └── knowledge-consolidator.md      ← 回路 B: 实现→知识        │     ║
║   └──────────────────────────────────────────────────────────────┘     ║
║                                                                          ║
║   ┌──────────────────────────────────────────────────────────────┐     ║
║   │                    第 三 层：生成产物                           │     ║
║   │  engine/        ← 推理框架（kernels/ models/ framework/ tp_layers/）│
║   │  llm_engine.py  ← 引擎主循环                                    │     ║
║   │  openai_tp_server.py ← OpenAI API 服务                          │     ║
║   └──────────────────────────────────────────────────────────────┘     ║
║                                                                          ║
║   ┌──────────────────────────────────────────────────────────────┐     ║
║   │                    第 四 层：编排器                             │     ║
║   │                                                                │     ║
║   │  master/          ← 回路 A: 性能迭代（KPI 驱动，无限循环）      │     ║
║   │  ├── MASTER.md        主 Agent 行为定义                        │     ║
║   │  ├── state.json       迭代状态 + 基线 KPI                      │     ║
║   │  ├── decision-log.jsonl  裁决日志                              │     ║
║   │  ├── strategies/      每轮策略文件                             │     ║
║   │  ├── results/         每轮结果 + knowledge_delta.json          │     ║
║   │  └── scripts/call-sub-agent.sh                                │     ║
║   │                                                                │     ║
║   │  evolution/       ← 回路 B: 知识进化（模型覆盖驱动，收敛即止）  │     ║
║   │  ├── EVOLUTION.md     进化编排器行为定义                        │     ║
║   │  ├── state.json       进化状态 + 开源开关                       │     ║
║   │  ├── decision-log.jsonl  进化裁决日志                           │     ║
║   │  ├── strategies/      进化策略文件                             │     ║
║   │  ├── results/         进化结果 + exploration_report.md         │     ║
║   │  └── scripts/call-evo-agent.sh                                │     ║
║   └──────────────────────────────────────────────────────────────┘     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 二、主入口决策树（CLAUDE.md 启动流程）

```
                          ┌──────────────────────┐
                          │  用户输入              │
                          │  /phase-all           │
                          │  /phase5              │
                          │  /evolve <model>      │
                          │  或直接描述任务        │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Step 0: 环境配置       │
                          │                        │
                          │ .env_agent_infer       │
                          │  ├── 存在 → 加载       │
                          │  │   └── MODEL_DIR     │
                          │  │       匹配？         │
                          │  │   ├── 是 → OK       │
                          │  │   └── 否 → 问用户   │
                          │  └── 不存在 → 问用户   │
                          │     (MODEL_DIR +       │
                          │      PYTHON_PATH)      │
                          │                        │
                          │ 自动验证：              │
                          │  ls config.json        │
                          │  python -c "import     │
                          │    torch"              │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Step 1-3: 加载上下文   │
                          │                        │
                          │ 读 00_contracts/       │
                          │ 读 AGENT_SKILL.md      │
                          │ 设置 PYTHONPATH        │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Step 4-5: 模型分析     │
                          │                        │
                          │ 读 config.json:        │
                          │  architectures         │
                          │  num_heads             │
                          │  rope_scaling          │
                          │  ...                   │
                          │                        │
                          │ 输出：                  │
                          │ "模型路由结论: Dense"   │
                          │ 或                     │
                          │ "模型路由结论: MLA+MoE" │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────────────────┐
                          │ Step 6: 知识库覆盖检测 (CRITICAL)  │
                          │                                    │
                          │ 检查 model_specs.md                │
                          │  ├── 有该模型参数？                 │
                          │ 检查 02_model_specifics/           │
                          │  ├── 有该模型文档？                 │
                          │ 检查 00_contracts/                 │
                          │  ├── 架构类型契约就绪？             │
                          │                                    │
                          │     ┌─────────┐                   │
                          │     │ 全满足？ │                   │
                          │     └────┬────┘                   │
                          │    是 ╱       ╲ 否                │
                          │      ╱         ╲                  │
                          │     ▼           ▼                 │
                          │ ┌────────┐  ┌────────────────┐   │
                          │ │回路 A+C│  │回路 B: 进化路径 │   │
                          │ │直接构建│  │evolution/      │   │
                          │ └────────┘  │EVOLUTION.md    │   │
                          │             │先探索再构建    │   │
                          │             └───────┬────────┘   │
                          │                     │            │
                          │              ┌──────▼────────┐   │
                          │              │ 进化成功？      │   │
                          │              │ 知识库已更新   │   │
                          │              └──────┬────────┘   │
                          │                     │ 是         │
                          │                     ▼            │
                          │              ┌────────────┐      │
                          │              │ 回到主流程  │      │
                          │              │ 回路 A+C    │      │
                          │              └────────────┘      │
                          └────────────────┬─────────────────┘
                                           │
                                ┌──────────▼───────────┐
                                │ Step 7: 平台自动检测   │
                                │                        │
                                │ torch.cuda.            │
                                │  get_device_name(0)    │
                                │                        │
                                │ NVIDIA → CustomAR+NCCL │
                                │ AMD    → RCCL fallback │
                                │ DCU    → comm fallback │
                                └──────────┬───────────┘
                                           │
                                ┌──────────▼───────────┐
                                │ Step 8: MEMORY 回溯   │
                                │                        │
                                │ 读 phase_report/       │
                                │ PHASE<N>_MEMORY.md    │
                                │ 重建前序上下文         │
                                └──────────┬───────────┘
                                           │
                                ┌──────────▼───────────┐
                                │ 进入构建/迭代流程     │
                                │ (见下方回路详图)      │
                                └──────────────────────┘
```

---

## 三、回路 A（成果路径）：性能迭代循环

```
                ┌─────────────────────────────────┐
                │    master/MASTER.md              │
                │    主 Agent：诊断→决策→调度→比较  │
                └─────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                       无限循环（用户退出才停）                         │
  │                                                                      │
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
  │  │ Step 1   │──▶│ Step 2   │──▶│ Step 3   │──▶│ Step 4   │        │
  │  │ 加载状态  │   │ 诊断     │   │ 写策略    │   │ 启动子    │        │
  │  │          │   │          │   │          │   │ Agent    │        │
  │  │ state    │   │ KPI分析  │   │ strategy │   │          │        │
  │  │ .json    │   │ 瓶颈定位 │   │ -XXX.json│   │ ${CLAUDE_CLI} -p│        │
  │  └──────────┘   └──────────┘   └──────────┘   └─────┬────┘        │
  │                                                      │              │
  │     ┌────────────────────────────────────────────────┘              │
  │     │ call-sub-agent.sh 做的事:                                     │
  │     │                                                               │
  │     │  ┌─────────────────────────────────────────────────────┐     │
  │     │  │            子 Agent 内部（全新进程，无记忆）           │     │
  │     │  │                                                      │     │
  │     │  │  读策略 + 知识库                                       │     │
  │     │  │     ↓                                                 │     │
  │     │  │  /phase-all (1→11 全量构建)                           │     │
  │     │  │     ↓                                                 │     │
  │     │  │  ┌──────────────┐                                     │     │
  │     │  │  │ implementer  │ ← 写 engine/ + llm_engine.py + ... │     │
  │     │  │  │ 只写不测     │                                     │     │
  │     │  │  │ → SUBMITTED  │                                     │     │
  │     │  │  └──────┬───────┘                                     │     │
  │     │  │         │                                              │     │
  │     │  │  ┌──────▼───────┐                                     │     │
  │     │  │  │spec-reviewer │ ← 逐条对照 00_contracts/            │     │
  │     │  │  │ 只审不测     │                                     │     │
  │     │  │  │ → PASS/FAIL  │                                     │     │
  │     │  │  └──────┬───────┘                                     │     │
  │     │  │         │ ✅ PASS                                      │     │
  │     │  │  ┌──────▼───────┐                                     │     │
  │     │  │  │ verification │ ← L0+L0.5+L0.6+L1+L2+L3            │     │
  │     │  │  │ 只测不改     │                                     │     │
  │     │  │  │ → PASS/FAIL  │                                     │     │
  │     │  │  └──────────────┘                                     │     │
  │     │  │                                                      │     │
  │     │  │  产出 → master/results/<ITER_ID>/                     │     │
  │     │  │         ├── benchmarks.jsonl                          │     │
  │     │  │         ├── AGGREGATE_REPORT.md                       │     │
  │     │  │         └── diagnostics_summary.json                  │     │
  │     │  └──────────────────────────────────────────────────────┘     │
  │     │                                                               │
  │     ▼                                                               │
  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
  │  │ Step 5   │──▶│ Step 6   │──▶│ Step 7   │──▶│ Step 8   │        │
  │  │ 提取KPI  │   │ 对比裁决  │   │ 展示对比  │   │ 持久化/   │        │
  │  │          │   │          │   │          │   │ 丢弃      │        │
  │  │吞吐/TTFT │   │ADVANCE?  │   │ 表格+Δ%  │   │ADV→存代码│        │
  │  │/TPOT/    │   │ROLLBACK? │   │          │   │ROLL→清空 │        │
  │  │greedy    │   │          │   │          │   │          │        │
  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
  │                                                      │              │
  │  ┌──────────┐                           ┌────────────▼──────────┐  │
  │  │ Step 10  │◄──────────────────────────│ Step 9                │  │
  │  │ 循环     │                           │ 更新状态               │  │
  │  │ 回到     │                           │ state.json            │  │
  │  │ Step 1   │                           │ decision-log.jsonl    │  │
  │  └──────────┘                           └────────────┬──────────┘  │
  │                                                      │              │
  │                   ┌──────────────────────────────────┘              │
  │                   │ only if ADVANCE                                  │
  │                   ▼                                                 │
  │  ┌──────────────────────────────────────────┐                      │
  │  │         Step 9.5: 回路 C 触发             │                      │
  │  │                                            │                      │
  │  │  检查知识信号:                              │                      │
  │  │  ├── throughput_delta > 5% ？              │                      │
  │  │  ├── 新策略模式 (algorithmic/architectural)？│                      │
  │  │  ├── 跨 ≥3 轮连续 ADVANCE ？               │                      │
  │  │  └── 刚修复了 ROLLBACK 的根因 ？            │                      │
  │  │                                            │                      │
  │  │  信号 ≥ 2 ？                                │                      │
  │  │  ├── 是 → 启动 experiment-summarizer       │                      │
  │  │  │        (${CLAUDE_CLI} -p 进程隔离)              │                      │
  │  │  │        → 写 knowledge_delta.json        │                      │
  │  │  │        → 追加 notebooks-cn/ 对应文件    │                      │
  │  │  │        → 知识库更新 ✅                   │                      │
  │  │  └── 否 → NO_NEW_KNOWLEDGE (跳过)          │                      │
  │  └──────────────────────────────────────────┘                      │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 四、回路 B（知识进化路径）：新模型学习循环

```
         ┌──────────────────────────────────────────────────────┐
         │             evolution/EVOLUTION.md                    │
         │             进化编排器：探索→构建→固化→验证             │
         └──────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  ┌────────────────────────────────────────────────────────────────┐ │
  │  │  触发条件                                                        │ │
  │  │  ├── CLAUDE.md Step 6 判定"未覆盖"                               │ │
  │  │  ├── 用户手动 /evolve <model_id>                                 │ │
  │  │  └── master 循环中首次 /phase-all 全部失败                        │ │
  │  └────────────────────────────────────────────────────────────────┘ │
  │                              │                                       │
  │                              ▼                                       │
  │                                                                      │
  │  ╔══════════════════════════════════════════════════════════════╗   │
  │  ║                    进化状态机                                  ║   │
  │  ║                                                               ║   │
  │  ║  ┌──────────────────────────────────┐                         ║   │
  │  ║  │  第 1 轮                          │                         ║   │
  │  ║  │  phase: attempt_without_opensource│                         ║   │
  │  ║  │  SWITCH = OFF                     │                         ║   │
  │  ║  │                                    │                         ║   │
  │  ║  │  Explorer: skip（纯知识库）        │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  Implementer: /phase-all           │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  Verification: scripts/ 全部门禁   │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  ┌───────────┐                     │                         ║   │
  │  ║  │  │ 全部 PASS? │                    │                         ║   │
  │  ║  │  └─────┬─────┘                     │                         ║   │
  │  ║  │   是 ╱     ╲ 否                    │                         ║   │
  │  ║  │     ▼       ▼                      │                         ║   │
  │  ║  │  🎉DONE  进入第 2 轮               │                         ║   │
  │  ║  │  (KB已够)  ↓                       │                         ║   │
  │  ║  └────────────┬──────────────────────┘                         ║   │
  │  ║               │                                                ║   │
  │  ║  ┌────────────▼──────────────────────┐                         ║   │
  │  ║  │  第 2 轮                           │                         ║   │
  │  ║  │  phase: attempt_with_opensource    │                         ║   │
  │  ║  │  SWITCH = ON                       │                         ║   │
  │  ║  │                                     │                         ║   │
  │  ║  │  Explorer: full                     │                         ║   │
  │  ║  │  ├── WebSearch 论文/技术报告        │                         ║   │
  │  ║  │  ├── WebFetch HF config.json        │                         ║   │
  │  ║  │  ├── WebFetch HF 模型卡             │                         ║   │
  │  ║  │  └── 读 knowledge/vllm/ (开关ON)    │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  产出: exploration_report.md       │                         ║   │
  │  ║  │       + model_diff.json            │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  Implementer: 读探索报告 + /phase-all│                        ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  Verification: scripts/ 全部门禁   │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  ┌───────────┐                     │                         ║   │
  │  ║  │  │ 全部 PASS? │                    │                         ║   │
  │  ║  │  └─────┬─────┘                     │                         ║   │
  │  ║  │   是 ╱     ╲ 否                    │                         ║   │
  │  ║  │     ▼       ▼                      │                         ║   │
  │  ║  │  进入第3轮  failure_count++         │                         ║   │
  │  ║  │            ≥3 → 暂停请求人类        │                         ║   │
  │  ║  │            <3 → Explorer增量收集    │                         ║   │
  │  ║  └────────────┬──────────────────────┘                         ║   │
  │  ║               │                                                ║   │
  │  ║  ┌────────────▼──────────────────────┐                         ║   │
  │  ║  │  第 3 轮: Knowledge Consolidator   │                         ║   │
  │  ║  │                                     │                         ║   │
  │  ║  │  读探索报告 + 成功代码 + 验证报告    │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  写入 notebooks-cn/:               │                         ║   │
  │  ║  │  ├── 02_model_specifics/ (新模型文档)│                        ║   │
  │  ║  │  ├── 06_experience/ (调试经验)      │                         ║   │
  │  ║  │  ├── 07_improvementPlan/ (注意事项) │                         ║   │
  │  ║  │  └── 00_contracts/ (如需 → 人类确认)│                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  产出: knowledge_delta.json        │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  切换: SWITCH = OFF                │                         ║   │
  │  ║  │        phase = verify_without_      │                         ║   │
  │  ║  │                opensource           │                         ║   │
  │  ║  └────────────┬──────────────────────┘                         ║   │
  │  ║               │                                                ║   │
  │  ║  ┌────────────▼──────────────────────┐                         ║   │
  │  ║  │  第 4 轮: 无开源重验                │                         ║   │
  │  ║  │  phase: verify_without_opensource  │                         ║   │
  │  ║  │  SWITCH = OFF（纯知识库）           │                         ║   │
  │  ║  │                                     │                         ║   │
  │  ║  │  Explorer: skip                     │                         ║   │
  │  ║  │  Implementer: 仅读更新后KB + /phase-all│                      ║   │
  │  ║  │  Verification: scripts/ 全部门禁   │                         ║   │
  │  ║  │     ↓                              │                         ║   │
  │  ║  │  ┌───────────┐                     │                         ║   │
  │  ║  │  │ 全部 PASS? │                    │                         ║   │
  │  ║  │  └─────┬─────┘                     │                         ║   │
  │  ║  │   是 ╱     ╲ 否                    │                         ║   │
  │  ║  │     ▼       ▼                      │                         ║   │
  │  ║  │  🎉DONE  回到第2轮                  │                         ║   │
  │  ║  │  (进化成功)(知识还不够,继续迭代)     │                         ║   │
  │  ║  └────────────────────────────────────┘                         ║   │
  │  ╚══════════════════════════════════════════════════════════════╝   │
  │                                                                      │
  │  进化成功 → 知识库已更新 → 委派回 master/ 正常迭代                    │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 五、回路 C（知识回流）：实验经验 → 知识库

```
  ┌────────────────────────────────────────────────────────────────┐
  │  触发点: master/MASTER.md Step 9 (更新状态) 之后                 │
  │  条件: verdict == ADVANCE                                        │
  └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                    信号检测器                                    │
  │                                                                 │
  │  检查本轮是否有值得持久化的知识:                                  │
  │                                                                 │
  │  ┌──────────────────┐  ┌──────────────────┐                    │
  │  │ 显著性能增益      │  │ 新策略模式        │                    │
  │  │ throughput > +5% │  │ algorithmic/     │                    │
  │  │ TTFT/TPOT < -10% │  │ architectural    │                    │
  │  │ 来源: benchmarks │  │ 且历史上首次     │                    │
  │  └────────┬─────────┘  └────────┬─────────┘                    │
  │           │                     │                               │
  │  ┌────────▼─────────┐  ┌────────▼─────────┐                    │
  │  │ 跨轮确认趋势      │  │ 错误修复          │                    │
  │  │ ≥3轮同方向       │  │ 上一轮ROLLBACK    │                    │
  │  │ 连续ADVANCE      │  │ 本轮修复+ADVANCE │                    │
  │  │ 来源: decision-  │  │ 来源: history[]  │                    │
  │  │        log.jsonl │  │       + verdict  │                    │
  │  └────────┬─────────┘  └────────┬─────────┘                    │
  │           │                     │                               │
  │           └───────┬─────────────┘                               │
  │                   │                                             │
  │              ┌────▼────┐                                        │
  │              │ 信号≥2？ │                                        │
  │              └────┬────┘                                        │
  │             是 ╱     ╲ 否                                       │
  │               ▼       ▼                                         │
  │          ┌────────┐  ┌──────────────────┐                      │
  │          │ 启动    │  │ NO_NEW_KNOWLEDGE │                      │
  │          │summarizer│ │ 跳过,继续循环    │                      │
  │          └────┬───┘  └──────────────────┘                      │
  └───────────────│─────────────────────────────────────────────────┘
                  │
                  ▼
  ┌────────────────────────────────────────────────────────────────┐
  │               Experiment Summarizer                             │
  │               (${CLAUDE_CLI} -p 进程隔离)                               │
  │                                                                 │
  │  输入:                                                          │
  │  ├── master/strategies/strategy-<ID>.json                       │
  │  ├── master/results/<ID>/benchmarks.jsonl                       │
  │  ├── master/results/<ID>/diagnostics_summary.json               │
  │  ├── master/decision-log.jsonl                                  │
  │  └── notebooks-cn/ 现有知识                                     │
  │                                                                 │
  │         ┌──────────────────────────┐                           │
  │         │     知识归属分类器        │                           │
  │         │                          │                           │
  │         │  调试技巧 → 06_experience│                           │
  │         │  参数规律 → 07_improve.. │                           │
  │         │  kernel优化→ 07_...      │                           │
  │         │  模型理解 → 02_model_... │                           │
  │         │  算子特征 → 03_operators │                           │
  │         │  并行策略 → 04_parallel_ │                           │
  │         │  框架约束 → 01_framework │                           │
  │         │                          │                           │
  │         │  ⛔ 不写 00_contracts/   │                           │
  │         └──────────┬───────────────┘                           │
  │                    │                                            │
  │                    ▼                                            │
  │  输出:                                                          │
  │  ├── master/results/<ID>/knowledge_delta.json                   │
  │  │   ├── iter_id, strategy, kpi_delta                          │
  │  │   ├── knowledge_category, target_file                       │
  │  │   ├── insight_summary, detail                               │
  │  │   └── signal_strength                                       │
  │  │        ├── confirmed (≥3轮确认)                              │
  │  │        ├── probable  (2轮确认)                               │
  │  │        └── tentative (仅本轮) → 只写json,不追加md           │
  │  │                                                             │
  │  └── notebooks-cn/<target>.md                                   │
  │      └── 追加段落:                                              │
  │          ## Δ from iter-<ID>                                   │
  │          - 日期/策略/效果/根因/适用条件/信号强度                  │
  └────────────────────────────────────────────────────────────────┘
```

---

## 六、三角色对抗协作流（Phase 构建细节）

**关键标注**：每条 spawn 边标注挂载方式。`[Agent]` = Agent 工具 spawn，`[Shell]` = Shell `${CLAUDE_CLI} -p` 独立进程。

```
                    ┌─────────────────────┐
                    │  主 Agent（你）       │
                    │  读契约 → 拆 Task    │
                    │  派子代理 → 收集结果  │
                    │  ⛔ 自己不写/不审/不测│
                    └──────┬──────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     ┌────────────────┐       ┌──────────────────┐
     │  首次大段构建   │       │  驳回后小修(≤10行) │
     │  (完整串行路径) │       │  (快速修复路径)    │
     └───────┬────────┘       └────────┬─────────┘
             │                         │
             │ [Agent] spawn           │ [Agent] spawn
             ▼                         │
     ┌────────────┐                    │
     │ implementer│ ← Agent 工具       │
     │ 写代码      │                    │
     │ 自读diff    │                    │
     │ → SUBMITTED│                    │
     └─────┬──────┘                    │
           │                           │
           │ [Shell] ${CLAUDE_CLI} -p         │
           ▼                           │
     ┌────────────┐      ❌ FAIL       │
     │spec-reviewer│ ──────────→ 打回   │
     │ Shell 独立进程│                  │
     │ 对照契约审查 │                  │
     └─────┬──────┘                    │
           │ ✅ PASS                   │
           │ [Shell] ${CLAUDE_CLI} -p         │
           ▼                           │
     ┌────────────┐                    │
     │verification│ ← Shell 独立进程   │
     │ 双重验证:   │◄───────────────────┘
     │ L0/L0.5/L0.6│  [Shell] ${CLAUDE_CLI} -p
     │ L1 scripts  │     ❌ FAIL → 打回
     │ L2 跨Phase  │    (连续2次→升级完整串行)
     │ L3 profiler │
     └─────┬──────┘
           │ ✅ PASS
           ▼
     ┌────────────┐
     │  Phase 交付 │
     │  + PID交叉  │
     │    验证     │
     └────────────┘
```

---

## 七、完整示例：Qwen3.6 27B 从头到尾

```
  Step 0  用户: "/phase-all" (目标模型 Qwen3.6 27B)
  ─────────────────────────────────────────────────
  Step 1  主 Agent: 检测到 MODEL_DIR 指向 Qwen3-8B
          主 Agent: AskUserQuestion → 用户提供 ${MODEL_DIR} (如 Qwen3.6-27B 路径)
          主 Agent: 自动更新 .env_agent_infer → 验证 config.json 可读 ✅
  ─────────────────────────────────────────────────
  Step 2  主 Agent: 读 config.json
          主 Agent: architectures=["Qwen3ForCausalLM"], 27B params
          主 Agent: "模型路由结论: Dense 模型"
  ─────────────────────────────────────────────────
  Step 3  主 Agent: 知识库覆盖检测
          检查 model_specs.md  → 只有 Qwen3-8B，没有 Qwen3.6 27B ❌
          检查 02_model_specifics/ → 只有 Qwen3 8B，没有 27B 文档 ❌
          判定: 未覆盖 → 启动 evolution/EVOLUTION.md
  ─────────────────────────────────────────────────
  Step 4  进化编排器: entry_reason=coverage_fail
          → ⛔ 跳过 attempt_without_opensource，直接从 SWITCH=ON 开始
  ─────────────────────────────────────────────────
  Step 5  进化编排器: 第 1 轮 (attempt_with_opensource)
          SWITCH=ON
          Explorer:
            WebSearch "Qwen3.6 27B architecture" → 技术报告
            WebFetch HF config.json → 27B 实际维度参数
            读 knowledge/vllm/qwen3.py → 分析 Dense 实现模式
            → 产出 exploration_report.md + model_diff.json
          Implementer: 读探索报告 → 用正确维度 + 架构差异 → /phase-all
          Verification: one_pass_rate = 100% → 全部 PASS ✅
          启动 Knowledge Consolidator
  ─────────────────────────────────────────────────
  Step 6  Knowledge Consolidator:
          写入 notebooks-cn/02_model_specifics/03_qwen3/02_27B_dense.md
          追加 notebooks-cn/00_contracts/model_specs.md (需人类确认)
          追加 notebooks-cn/06_experience/04_qwen3.6_tp_debug.md
          产出 knowledge_delta.json
          切换到 verify_without_opensource, SWITCH=OFF
  ─────────────────────────────────────────────────
  Step 7  进化编排器: 第 2 轮 (verify_without_opensource)
          SWITCH=OFF, 纯知识库重验
          Implementer: 仅读更新后KB → /phase-all
          Verification: one_pass_rate = 100% → 全部 PASS ✅
          🎉 进化成功！
  ─────────────────────────────────────────────────
  Step 8  委派回 master/ 回路 A
          用更新后的知识库跑 master 迭代循环
  ─────────────────────────────────────────────────
  Step 9  master 循环: 基线建立 → 诊断 → 策略 → 构建 → 验证
          ADVANCE 之后触发回路 C
          experiment-summarizer 判定有无新知识 → 回流到 notebooks-cn/
  ─────────────────────────────────────────────────
  最终: 知识库已包含 Qwen3.6 27B，推理框架已生成并可迭代优化 ✅
```

---

## 八、目录文件热力图

```
metainferv3/
│
├── 📄 CLAUDE.md               🔴 入口文件，全局架构+启动流程
├── 📄 AGENT_SKILL.md           🔴 执行SOP+编码铁律(12条)
├── 📄 PROJECT_FLOW.md          🟢 本文件：全局流程图
├── 📄 .env_agent_infer         🟡 环境变量（机器独立，.gitignore）
├── 📄 .gitignore               🟢 忽略规则
│
├── 📁 notebooks-cn/            🔴 先验知识库
│   ├── 📁 00_contracts/        🔴 11个API契约（人类写，B/C慎改）
│   ├── 📁 00_overview/         🟡 项目总览
│   ├── 📁 01_framework_design/ 🟡 框架设计
│   ├── 📁 02_model_specifics/  🟡 模型族（B/C可扩写）
│   ├── 📁 03_operators/        🟡 算子知识（B/C可扩写）
│   ├── 📁 04_parallel_strategies/🟡 并行策略（B/C可扩写）
│   ├── 📁 05_inference_service/🟢 推理服务
│   ├── 📁 06_experience/       🟡 调试经验（C自动追加）
│   └── 📁 07_improvementPlan/  🟡 优化方案（C自动追加）
│
├── 📁 .claude/
│   ├── 📁 roles/              🔴 Agent角色定义（6个）
│   │   ├── implementer-inference.md
│   │   ├── spec-reviewer-inference.md
│   │   ├── verification-inference.md
│   │   ├── experiment-summarizer.md   🟢 回路C角色
│   │   ├── explorer.md               🟢 回路B角色
│   │   └── knowledge-consolidator.md 🟢 回路B角色
│   └── 📁 skills/             🟡 Phase触发词
│       ├── phase-all/SKILL.md
│       ├── phase1-4/SKILL.md
│       ├── phase5/SKILL.md
│       ├── phase6/SKILL.md
│       ├── phase7-8/SKILL.md
│       ├── phase9-10/SKILL.md
│       ├── phase11/SKILL.md
│       └── evolve/SKILL.md    🟢 回路B触发词
│
├── 📁 master/                 🔴 回路A编排器：性能迭代
│   ├── MASTER.md
│   ├── state.json
│   ├── decision-log.jsonl
│   ├── 📁 strategies/
│   ├── 📁 results/
│   └── 📁 scripts/
│       └── call-sub-agent.sh
│
├── 📁 evolution/              🟢 回路B编排器：知识进化
│   ├── EVOLUTION.md
│   ├── state.json
│   ├── decision-log.jsonl
│   ├── 📁 strategies/
│   ├── 📁 results/
│   └── 📁 scripts/
│       └── call-evo-agent.sh
│
├── 📁 engine/                 🔴 生成产物：推理框架
│   ├── 📁 kernels/
│   ├── 📁 models/
│   ├── 📁 framework/
│   ├── 📁 tp_layers/
│   └── self_check.py
│
├── 📄 llm_engine.py           🔴 引擎主循环
├── 📄 openai_tp_server.py     🔴 OpenAI API服务
│
├── 📁 scripts/                🔴 28个测试合约（只读）
│
├── 📁 phase_report/           🟡 审查报告（.gitignore）
│
├── 📁 knowledge/              🟢 开源代码缓存（.gitignore）
│   ├── 📁 vllm/
│   └── 📁 sglang/
│
└── 📁 iterations/             🟢 迭代工作区（.gitignore）

🔴 = P0 关键文件，每次都要读
🟡 = P1 经常使用，按需读取
🟢 = P2 新增/辅助，按需读取
```

---

## 九、关键决策节点速查

| 节点 | 位置 | 决策 | 下一步 |
|------|------|------|--------|
| 环境配置 | CLAUDE.md Step 0 | .env存在且匹配？ | 是→Step1 / 否→问用户 |
| 模型分析 | CLAUDE.md Step 6 | 模型在KB覆盖范围？ | 是→回路A / 否→回路B |
| 平台检测 | CLAUDE.md Step 7 | GPU类型？ | NVIDIA→CustomAR / AMD→RCCL |
| 代码质量 | 三角色流 | spec-reviewer？ | PASS→verification / FAIL→打回 |
| 测试验收 | 三角色流 | verification？ | PASS→交付 / FAIL→打回 |
| KPI裁决 | master Step 6 | ADVANCE或ROLLBACK？ | ADVANCE→Step7 / ROLLBACK→Step1 |
| 知识回流 | master Step 9.5 | 知识信号≥2？ | 是→summarizer / 否→跳过 |
| 进化阶段 | evolution Step 5 | 无开源通过？ | 是→DONE / 否→开源辅助 |
| 知识固化 | evolution Step 5a | 写入00_contracts？ | 是→人类确认 / 否→自动 |
| 进化完成 | evolution Step 5 | 无开源重验通过？ | 是→DONE / 否→继续迭代 |
