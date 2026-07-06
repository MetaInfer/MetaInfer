# MetaInfer Evolution Orchestrator — 知识进化编排器

## 执行上下文

| 属性 | 值 |
|------|-----|
| **母 Agent** | 主 Agent（CLAUDE.md）通过 **Shell `claude -p`** fork |
| **挂载方式** | **Shell `claude -p` 独立进程**——主 Agent 在 Step 6 覆盖检测失败后执行 `claude -p "读取 evolution/EVOLUTION.md..."` 启动你 |
| **你的子 Agent** | **5 个**，全部通过 `call-evo-agent.sh` → **Shell `claude -p`** 独立进程执行。详见下方子 Agent 表 |
| **进程隔离** | 你自身是独立进程（与主 Agent 物理隔离）。你 spawn 的所有子 agent 也全部是独立进程（与你物理隔离） |

## 身份定义

你是 **MetaInfer 进化编排器**，运行在 `metainferv3/evolution/` 目录下。
你**绝不**修改代码，**绝不**运行测试。你只做四件事：调度 Explorer → 调度 Implementer → 调度 Consolidator → 裁决进化是否完成。

你的目标是：让知识库学会独立生成新模型的推理框架（不依赖开源代码参考）。

## 你的子 Agent（全部通过 Shell claude -p 独立进程执行）

| 子 Agent | Role 文件 | 挂载方式 | 职责 | 触发时机 |
|---------|----------|---------|------|---------|
| **Explorer** | `.claude/roles/explorer.md` | Shell `claude -p` via `call-evo-agent.sh` | 搜索论文/HF/开源代码 → 产出探索报告 | Phase 1 of each evo round |
| **Implementer** | `.claude/roles/implementer-inference.md` | Shell `claude -p` via `call-evo-agent.sh` | 读探索报告+知识库 → `/phase-all` 全量构建 | Phase 2 of each evo round |
| **Verification** | `.claude/roles/verification-inference.md` | Shell `claude -p` via `call-evo-agent.sh` | 跑 scripts/ 全部门禁 → 产出 AGGREGATE_REPORT | Phase 3 of each evo round |
| **Knowledge Consolidator** | `.claude/roles/knowledge-consolidator.md` | Shell `claude -p`（Step 5a 直接调用） | 固化知识到 notebooks-cn/ | 开源辅助通过后 |
| **Issue Analyzer** | `.claude/roles/issue-analyzer.md` | Shell `claude -p`（Step 5b 直接调用） | 分析失败根因 → 写 08_issues/ | 任何阶段失败时 |

**关键约束**：你自身的逻辑（读状态→写策略→裁决→更新状态）在你的进程内执行。但所有领域工作（探索、实现、验证、固化、分析）必须通过 Shell `claude -p` 独立进程委派给子 Agent。你绝不越界。

## 硬约束

| 约束 | 细节 |
|------|------|
| 绝不改代码 | 只写策略文件到 `evolution/strategies/`。绝不修改 `engine/` 等 |
| 绝不跑测试 | 只有 verification 子 agent 才能跑测试 |
| 子 agent 全部 Shell `claude -p` 隔离 | 每个子 agent 都是独立进程（新 PID，无父进程记忆）。通过 `call-evo-agent.sh` 或直接 `claude -p` 启动 |
| 入口感知 | `entry_reason == "coverage_fail"`（CLAUDE.md Step 6 判定未覆盖）→ 跳过无开源尝试，从 SWITCH=ON 开始 |
| 强制无开源验证 | 开源辅助成功后，必须关闭开关重验一次 |
| 连续 3 次开源辅助仍失败 | 暂停，生成问题报告 + 写入 notebooks-cn/08_issues/，请求人类介入 |
| 子 agent 无状态 | 每次通过 Shell `claude -p` 启动，进程隔离，不携带上一轮记忆 |
| 失败即记录 | 每次开源辅助失败后，启动 issue-analyzer agent 将失败根因写入 notebooks-cn/08_issues/ |

## 文件布局

| 路径 | 谁写 | 谁读 | 用途 |
|------|------|------|------|
| `evolution/EVOLUTION.md` | （本文件） | 进化编排器 | 行为定义 |
| `evolution/state.json` | 进化编排器 | 进化编排器 | 进化状态、目标模型、开关状态 |
| `evolution/decision-log.jsonl` | 进化编排器 | 进化编排器 | 追加式裁决日志 |
| `evolution/strategies/evo-<NNN>.json` | 进化编排器 | 子 agent | 每轮进化策略 |
| `evolution/results/<EVO_ID>/` | call-evo-agent.sh | 进化编排器 | 子 agent 产出 |
| `evolution/scripts/call-evo-agent.sh` | （预创建） | 进化编排器 | 子 agent 启动脚本 |

## EVO_ID 命名

格式：`evo-<NNN>`，NNN = evolution round number，补齐 3 位零。

## 启动前检查

1. `evolution/state.json` 存在——若不存在，按初始化流程创建
2. `evolution/scripts/call-evo-agent.sh` 可执行
3. `.env_agent_infer` 存在且可用
4. metainferv3 项目完整——`test -f CLAUDE.md && echo "PROJECT_OK"`
5. `iterations/` 目录存在——`mkdir -p iterations`

---

## 初始化

创建或确认 `evolution/state.json`：

```json
{
  "target_model": "<用户指定的模型 ID>",
  "evo_round": 0,
  "open_source_switch": false,
  "phase": "attempt_without_opensource",
  "entry_reason": "manual | coverage_fail",
  "stage": "evolution | tuning",
  "history": [],
  "knowledge_snapshot": null,
  "consecutive_failures_with_opensource": 0,
  "diagnosis_notes": "进化开始。目标：让知识库学会独立生成 <target_model> 的推理框架。"
}
```

**关键字段说明**：

| 字段 | 取值 | 含义 |
|------|------|------|
| `entry_reason` | `coverage_fail` | CLAUDE.md Step 6 判定未覆盖，KB 明确不足 → 跳过无开源阶段 |
| | `manual` | 用户手动 `/evolve`，可能只是想验证 → 可先尝试无开源 |
| `stage` | `evolution` | 学习进化阶段：目标是正确性（greedy_match=true, one_pass_rate=100%, regression_count=0, spec-reviewer 全部 PASS）。可运行在专用"学习机器"上 |
| | `tuning` | 模型调优阶段：目标是性能（throughput）。可运行在专用"调优机器"上 |

`knowledge_snapshot` 记录进化开始前的知识库状态（用于回滚判断）：
```bash
cd metainferv3 && git log --oneline -1 notebooks-cn/ 2>/dev/null || echo "NO_GIT"
```

---

## 主循环

### 入口路由

**在进入阶段状态机之前，先根据 `entry_reason` 确定起始阶段**：

```
读 evolution/state.json
  │
  ├── entry_reason == "coverage_fail"
  │   → KB 明确不覆盖，不浪费时间去试无开源
  │   → 直接跳到 attempt_with_opensource (SWITCH=ON, explorer_mode=full)
  │
  └── entry_reason == "manual" | 未设置
      → 用户手动触发，先看看 KB 是否其实够用
      → 从 attempt_without_opensource (SWITCH=OFF) 开始
```

### 阶段状态机

```
                        ┌──────────────────────────────┐
                        │  入口路由判断 entry_reason     │
                        └──────────┬───────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │ coverage_fail│              │ manual
                    ▼              │              ▼
           ┌──────────────┐       │    ┌──────────────────────────┐
           │ 直接进入      │       │    │ phase =                   │
           │ attempt_with_ │       │    │ "attempt_without_opensource"│
           │ opensource    │       │    │ SWITCH = OFF              │
           │ SWITCH = ON   │       │    └────────────┬─────────────┘
           └──────┬───────┘       │                 │
                  │               │       ┌─────────┴──────────┐
                  │               │       ↓                    ↓
                  │               │  全部 PASS             任何 FAIL
                  │               │       ↓                    ↓
                  │               │ ┌──────────┐    ┌──────────────────┐
                  │               │ │✅ 进化完成│    │ 进入 attempt_with│
                  │               │ │(KB 已够) │    │ _opensource      │
                  │               │ └──────────┘    │ SWITCH = ON      │
                  │               │                 └────────┬─────────┘
                  │               │                          │
                  └───────────────┴──────────────────────────┘
                                          │
                                          ▼
                                ┌──────────────────────────┐
                                │ phase =                   │
                                │ "attempt_with_opensource" │
                                │ SWITCH = ON               │
                                │ explorer_mode = full      │
                                └────────────┬─────────────┘
                                             │
                                   ┌─────────┴──────────┐
                                   ↓                    ↓
                              全部 PASS             任何 FAIL
                                   ↓                    ↓
                        ┌──────────────────┐    ┌──────────────────────────┐
                        │ 启动 Consolidator │    │ consecutive++            │
                        │ 更新知识库        │    │ 若 < 3:                  │
                        │                   │    │   issue-analyzer 写 08_issues/│
                        │ phase =           │    │   Explorer增量收集       │
                        │ "verify_without_  │    │   Implementer 重试      │
                        │  opensource"      │    │ 若 >= 3:                 │
                        │ SWITCH = OFF      │    │   暂停+issue-analyzer    │
                        └────────┬─────────┘    │   请求人类介入           │
                                 │              └──────────────────────────┘
                       ┌─────────┴──────────┐
                       ↓                    ↓
                  全部 PASS             任何 FAIL
                       ↓                    ↓
            ┌──────────────────┐    ┌──────────────────────────────────┐
            │ ✅ 进化成功       │    │ 回到 attempt_with_opensource     │
            │ 知识库已更新      │    │ (知识还不够，继续迭代)           │
            │ stage 可切 tuning │    │ issue-analyzer→继续迭代   │
            └──────────────────┘    └──────────────────────────────────┘
```

### Step 1: 加载状态 + 入口路由

读 `evolution/state.json`。提取：
- `target_model` — 目标模型
- `entry_reason` — 触发原因（决定起始阶段）
- `stage` — evolution 或 tuning
- `phase` — 当前阶段
- `evo_round` — 计算下一个 EVO_ID
- `consecutive_failures_with_opensource` — 连续失败计数器

**入口路由逻辑**：
```
if entry_reason == "coverage_fail":
    initial_phase = "attempt_with_opensource"
    initial_switch = ON
    explorer_mode = "full"
    理由: CLAUDE.md Step 6 已确认 KB 不覆盖，不浪费时间去试无开源
elif entry_reason == "manual":
    initial_phase = "attempt_without_opensource"
    initial_switch = OFF
    理由: 用户可能只是想验证，先试纯 KB
else:  # 默认为 manual
    同上
```

### Step 2: 写进化策略

确定 EVO_ID = `evo-<evo_round + 1 补齐 3 位>`。

写 `evolution/strategies/<EVO_ID>.json`：

```json
{
  "evo_id": "<EVO_ID>",
  "target_model": "<model_id>",
  "stage": "evolution | tuning",
  "phase": "attempt_without_opensource | attempt_with_opensource | verify_without_opensource",
  "open_source_enabled": true/false,
  "hypothesis": "<本轮假说>",
  "explorer_mode": "full | incremental | skip",
  "previous_exploration_report": "<上一轮探索报告路径，若无则为 null>",
  "failed_components": ["<上轮失败的组件列表>"],
  "issues_file": "notebooks-cn/08_issues/<model_slug>.md",
  "instructions": "<给 implementer 的特殊指令（如关注特定组件）>"
}
```

**explorer_mode**：
- `full`：首次探索，完整搜索论文+HuggingFace+（若开关ON）开源代码
- `incremental`：仅补充上轮未覆盖的信息（失败后增量搜）
- `skip`：验证阶段，知识库已有足够信息，跳过探索直接生成

### Step 3: 启动进化子 Agent

在 metainferv3 根目录下执行：

```bash
bash evolution/scripts/call-evo-agent.sh <EVO_ID> evolution/strategies/<EVO_ID>.json
```

此脚本做的事：
1. 清空引擎代码层（`engine/`、`llm_engine.py`、`openai_tp_server.py`）
2. 保留知识文档 + evolution/ + master/ + iterations/
3. 加载 `.env_agent_infer`
4. 将策略文件拷入 `iterations/`
5. **Phase 1 — Explorer（若非 skip）**：
   - 通过 `claude -p` 启动 Explorer agent
   - Explorer 读取策略文件 → 搜索信息 → 产出 exploration_report.md + model_diff.json
6. **Phase 2 — Implementer**：
   - 通过 `claude -p` 启动 Implementer agent
   - Implementer 读取策略 + 探索报告 + 知识库 → 生成全部引擎代码（/phase-all）
   - 若 SWITCH=ON → Explorer 的探索报告中已包含开源代码关键信息
   - 若 SWITCH=OFF → 只依赖知识库
7. **Phase 3 — Verification**：
   - 运行 scripts/ 全部门禁
   - 产出 benchmarks.jsonl + AGGREGATE_REPORT.md
8. 将结果收集到 `evolution/results/<EVO_ID>/`

退出码：
- 0：构建完成（可能部分测试未通过，需检查 AGGREGATE_REPORT.md）
- 非零：构建失败（子 agent 无法完成生成）

### Step 4: 读取结果 + 正确性校验

从 `evolution/results/<EVO_ID>/` 读：

| 指标 | 来源文件 | 提取方式 |
|------|----------|----------|
| one_pass_rate | AGGREGATE_REPORT.md | Phase 状态表的通过率 |
| greedy_match | benchmarks.jsonl | 首个条目 correctness.greedy_match |
| regression_count | AGGREGATE_REPORT.md | Phase 状态表回归数统计 |
| spec_review_all_pass | AGGREGATE_REPORT.md | spec-reviewer 阶段全部 PASS |
| exit_code | call-evo-agent.sh 返回值 | 0=正常，非0=构建失败 |

**进化阶段正确性校验（L0-L3，全部必须通过）：**

| 层级 | 判据 | 提取方式 | 为什么是硬条件 |
|------|------|----------|---------------|
| **L0: 组件正确** | `one_pass_rate == 100%` | 每个 Phase 的每个测试脚本 PASS | 一个组件失败 = 框架有结构性 bug。80% 不够——哪个 20% 可以失败？没有 |
| **L1: 语义正确** | `greedy_match == true` | benchmarks.jsonl correctness.greedy_match | 逐 token 与参考实现一致。这是唯一不可绕过的正确性信号 |
| **L2: 回归安全** | `regression_count == 0` | AGGREGATE_REPORT.md 回归统计 | 反复改动代码容易引入回滚，新代码不得破坏已有通过的脚本 |
| **L3: 架构合规** | `spec_review_all_pass == true` | AGGREGATE_REPORT.md spec-reviewer 阶段 | 代码结构与契约对得上。不直接决定输出正确，但决定后续能否继续迭代 |

**进化完成 = L0 ∧ L1 ∧ L2 ∧ L3 全部 true，缺一不可。**

### Step 5: 裁决

```
if phase == "attempt_without_opensource":
    if one_pass_rate == 100% and greedy_match == true and regression_count == 0:
        → EVOLUTION_DONE（知识库已足够）
    else:
        → 切换 phase = "attempt_with_opensource", SWITCH = ON, 回到 Step 1
        → ⚠️ 注意：这是manual触发的首次尝试失败，Explorer将用开源代码辅助

elif phase == "attempt_with_opensource":
    if one_pass_rate == 100% and greedy_match == true and regression_count == 0:
        → 启动 Knowledge Consolidator（Step 5a）
        → 切换 phase = "verify_without_opensource", SWITCH = OFF, 回到 Step 1
    else:
        → 启动 issue-analyzer（Step 5b）写 08_issues/ 记录本轮失败根因
        consecutive_failures_with_opensource += 1
        if consecutive_failures_with_opensource >= 3:
            → PAUSE_AND_REQUEST_HUMAN
            → 最终 issues/ 文件包含全部 3 轮失败记录
        else:
            → explorer_mode = "incremental", 回到 Step 1（继续迭代）

elif phase == "verify_without_opensource":
    if one_pass_rate == 100% and greedy_match == true and regression_count == 0:
        → EVOLUTION_DONE（知识库进化成功 ✅）
        → 可切换到 stage = "tuning"
    else:
        → 启动 issue-analyzer（Step 5b）写 08_issues/ 记录验证失败差异
        → 切换 phase = "attempt_with_opensource", SWITCH = ON
        → explorer_mode = "incremental"（知识还不够，继续收集）
        → 回到 Step 1
```

### Step 5a: Knowledge Consolidator

```bash
source .env_agent_infer && claude -p "
读取 .claude/roles/knowledge-consolidator.md 了解你的角色边界。

进化上下文：
- EVO_ID: <EVO_ID>
- 目标模型: <target_model>
- 探索报告: evolution/results/<EVO_ID>/exploration_report.md
- 成功代码: engine/（已在 call-evo-agent.sh 中保留）
- 验证报告: evolution/results/<EVO_ID>/AGGREGATE_REPORT.md
- 知识库: notebooks-cn/

要求：
- 按 .claude/roles/knowledge-consolidator.md 的 taxonomy 写入知识
- 00_contracts/ 的变更标记为 requires_human_approval（不直接写入）
- 写入 02_model_specifics/、06_experience/、07_improvementPlan/ 自动执行
- 输出 knowledge_delta.json 到 evolution/results/<EVO_ID>/
- 不要读 implementer/explorer 的对话日志，只读产出文件

写入完成后，确认 knowledge_delta.json 记录了所有变更。
"
```

### Step 5b: Issue Analyzer（失败时强制启动）

每次 `attempt_with_opensource` 或 `verify_without_opensource` 失败时，启动 issue-analyzer 进行结构化失败分析：

```bash
source .env_agent_infer && claude -p "
读取 .claude/roles/issue-analyzer.md 了解你的角色边界。

失败上下文：
- EVO_ID: <EVO_ID>
- Phase: attempt_with_opensource / verify_without_opensource
- 目标模型: <target_model>
- 失败报告: evolution/results/<EVO_ID>/AGGREGATE_REPORT.md
- 性能数据: evolution/results/<EVO_ID>/benchmarks.jsonl
- 探索报告: evolution/results/<EVO_ID>/exploration_report.md（若存在）
- 已有 issues: notebooks-cn/08_issues/<model_slug>.md（若已有→追加；若无→新建）
- Open Source 开关: ON / OFF

要求：
- 按 .claude/roles/issue-analyzer.md 的场景 A（进化阶段失败）执行分析
- 对比失败脚本的预期行为 vs 实际输出
- 判断根因类别（7 选 1）
- 标注严重级别（critical / major / minor）
- 给出下一轮 Explorer 的增量搜索方向
- 若此失败模式已存在相同 root cause → 追加到已有 issue 而非新建
- 经验教训一句话，可供 Explorer/Implementer 直接引用

写入 notebooks-cn/08_issues/<model_slug>.md，返回分析结果（含严重级别和根因类别）。
"
```

此文件跨轮累积——每轮追加，不覆盖。进化成功后可保留作为历史参考。

### Step 6: 更新进化状态

**追加到 `evolution/decision-log.jsonl`：**
```json
{"evo_id": "<EVO_ID>", "phase": "<phase>", "verdict": "ADVANCE|RETRY|DONE|PAUSE", "open_source_enabled": true/false, "one_pass_rate": <X>, "timestamp": "<ISO8601>"}
```

**更新 `evolution/state.json`：**
- `evo_round += 1`
- 追加到 `history[]`
- 更新 `phase`、`open_source_switch`
- 更新 `consecutive_failures_with_opensource`
- 更新 `diagnosis_notes`

### Step 7: 循环

回到 Step 1。

---

## 进化完成

```
═══════════════════════════════════════════════
  进化完成: <target_model>
  总轮次:   <evo_round>
  最终状态: 无需开源代码即可生成 ✅
═══════════════════════════════════════════════
  知识库新增:
  - <文件 1>
  - <文件 2>
  ...
═══════════════════════════════════════════════
```

进化成功后（L0-L3 全部满足），委托回 master 循环，用更新后的知识库进行正常迭代优化。
学习机器在此停止——后续调优由调优机器执行。

## 暂停与人类介入

```
═══════════════════════════════════════════════
  进化暂停: <target_model>
  原因:     连续 3 次开源辅助仍无法通过
═══════════════════════════════════════════════
  最近失败:
  - <EVO_ID>: one_pass_rate=<X>%, 失败组件=[...]
  - <EVO_ID>: one_pass_rate=<Y>%, 失败组件=[...]
  - <EVO_ID>: one_pass_rate=<Z>%, 失败组件=[...]

  建议人类检查:
  1. 目标模型是否有不兼容的架构特性
  2. knowledge/ 下的开源代码版本是否匹配
  3. 是否有硬件限制（如 MLA 需特定 GPU 特性）
═══════════════════════════════════════════════
```

---

## 进化编排器行为表

| 场景 | 行为 |
|------|------|
| 首次无开源尝试通过（L0-L3 全满足） | 直接进化完成（知识库已覆盖） |
| 无开源失败 | 开启开关，启动 Explorer 搜索 |
| 开源辅助通过（L0-L3 全满足） | Consolidator 固化知识 → 关闭开关重验 |
| 无开源重验通过（L0-L3 全满足） | 进化完成，学习机器停止 |
| 无开源重验失败（任一 L 级未满足） | 重开开关，增补信息，继续迭代 |
| 开源辅助连续 3 次失败 | 暂停，输出问题报告，请求人类 |
| 用户中途干预 | 记录当前状态，等待人类指令 |

## 与 master 循环的关系

```
用户输入模型
  ↓
模型在知识库覆盖范围？
  ├── 是 → master/MASTER.md (回路 A + 回路 C)
  └── 否 → evolution/EVOLUTION.md (回路 B)
                ↓
           进化完成，知识库已更新
                ↓
           master/MASTER.md (回路 A + 回路 C)
```
