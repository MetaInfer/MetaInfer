# MetaInfer v3 — 全项目流程图（Mermaid 版）

> 本文是 `PROJECT_FLOW.md` 的 Mermaid 等价转换。每个 Mermaid 图独立可渲染，LLM 可直接解析节点/边/子图关系，无需理解 ASCII art。

---

## 一、全局架构鸟瞰（四层 + 三条回路）

```mermaid
flowchart TB
    subgraph L1["第一层：先验知识库"]
        direction TB
        contracts["00_contracts/<br/>11个API契约<br/>⛔人类确认才能改"]
        overview["00_overview/"]
        framework_doc["01_framework_design/"]
        model_spec["02_model_specifics/<br/>回路B/C可扩写"]
        operators["03_operators/"]
        parallel["04_parallel_strategies/<br/>回路B/C可扩写"]
        service["05_inference_service/"]
        experience["06_experience/<br/>回路B/C自动追加"]
        improvement["07_improvementPlan/<br/>回路B/C自动追加"]
        agent_skill["AGENT_SKILL.md<br/>执行SOP+编码铁律"]
        claude_md["CLAUDE.md<br/>全局入口+spawn协议"]
        scripts["scripts/<br/>28个测试合约<br/>⛔只读不可修改"]
    end

    subgraph L2["第二层：Agent 角色群"]
        direction LR
        impl["implementer-inference.md<br/>写代码，不跑测试"]
        spec["spec-reviewer-inference.md<br/>契约核验，不跑测试"]
        verify["verification-inference.md<br/>跑测试，唯一宣判PASS"]
        summarizer["experiment-summarizer.md<br/>回路C: 实验→知识"]
        explorer["explorer.md<br/>回路B: 搜索论文/HF/源码"]
        consolidator["knowledge-consolidator.md<br/>回路B: 实现→知识"]
        issue_analyzer["issue-analyzer.md<br/>失败分析: 结构化写08_issues/"]
    end

    subgraph L3["第三层：生成产物"]
        direction LR
        engine["engine/<br/>kernels/models/framework/tp_layers"]
        llm_engine["llm_engine.py<br/>引擎主循环"]
        openai_server["openai_tp_server.py<br/>OpenAI API服务"]
    end

    subgraph L4["第四层：编排器"]
        direction LR
        subgraph master["master/ — 回路A: 性能迭代"]
            master_md["MASTER.md"]
            state_json["state.json"]
            decision_log["decision-log.jsonl"]
            strategies["strategies/"]
            results["results/"]
        end
        subgraph evolution["evolution/ — 回路B: 知识进化"]
            evo_md["EVOLUTION.md"]
            evo_state["state.json"]
            evo_log["decision-log.jsonl"]
            evo_strategies["strategies/"]
            evo_results["results/"]
        end
    end

    L1 -->|"回路B写入<br/>回路C写入"| L1
    L1 -->|"读取"| L2
    L2 -->|"生成"| L3
    L3 -->|"被编排"| L4
    L4 -->|"验证反馈"| L3
```

---

## 二、主入口决策树（CLAUDE.md 启动流程）

```mermaid
flowchart TD
    START(["用户输入<br/> /phase-all | /phase5 | /evolve model | 直接描述任务"])
    
    START --> S0["Step 0: 环境配置"]
    S0 --> S0_CHECK{".env_agent_infer<br/>存在？"}
    S0_CHECK -->|"存在"| S0_LOAD["加载环境变量"]
    S0_CHECK -->|"不存在"| S0_ASK["AskUserQuestion<br/>MODEL_DIR + PYTHON_PATH"]
    S0_LOAD --> S0_MATCH{"MODEL_DIR<br/>匹配用户目标？"}
    S0_MATCH -->|"是"| S0_VALIDATE
    S0_MATCH -->|"否"| S0_ASK
    S0_ASK --> S0_WRITE["写入 .env_agent_infer"]
    S0_WRITE --> S0_VALIDATE["验证:<br/>ls config.json<br/>python -c 'import torch'"]
    
    S0_VALIDATE --> S1["Step 1-3: 加载上下文<br/>读 00_contracts/<br/>读 AGENT_SKILL.md<br/>设置 PYTHONPATH"]
    
    S1 --> S4["Step 4-5: 模型分析<br/>读 config.json<br/>architectures/num_heads/rope_scaling"]
    S4 --> S4_OUT["输出模型路由结论<br/>Dense 或 MLA+MoE"]
    
    S4_OUT --> S6{"Step 6: 知识库覆盖检测<br/>━━━━━━━━━━━━━<br/>① model_specs.md 有该模型参数？<br/>② 02_model_specifics/ 有该模型文档？<br/>③ 00_contracts/ 架构契约就绪？"}
    
    S6 -->|"三项全满足 ✅"| ROUTE_AC["进入回路 A+C<br/>直接构建"]
    S6 -->|"任一不满足 ❌"| ROUTE_B["⛔ 进入回路 B: 进化路径<br/>entry_reason=coverage_fail<br/>跳过无开源尝试<br/>直接从 SWITCH=ON 开始"]
    
    ROUTE_B --> EVO_CHECK{"进化成功？<br/>知识库已更新"}
    EVO_CHECK -->|"是"| ROUTE_AC
    EVO_CHECK -->|"否"| ROUTE_B
    
    ROUTE_AC --> S7["Step 7: 平台自动检测<br/>torch.cuda.get_device_name(0)"]
    S7 --> S7_OUT["NVIDIA → CustomAR+NCCL<br/>AMD → RCCL fallback<br/>DCU → comm fallback"]
    
    S7_OUT --> S8["Step 8: MEMORY 回溯<br/>读 phase_report/PHASEN_MEMORY.md<br/>重建前序上下文"]
    
    S8 --> BUILD["进入构建/迭代流程<br/>见下方回路详图"]
```

---

## 三、回路 A（成果路径）：性能迭代循环

```mermaid
flowchart TD
    MASTER["master/MASTER.md<br/>主Agent: 诊断→决策→调度→比较"]
    
    subgraph LOOP["无限循环（用户退出才停）"]
        S1["Step1: 加载状态<br/>state.json"]
        S2["Step2: 诊断<br/>KPI分析 + 瓶颈定位"]
        S3["Step3: 写策略<br/>strategy-XXX.json"]
        S4["Step4: 启动子Agent<br/>claude -p 进程隔离"]
        
        subgraph SUBAGENT["子Agent内部（全新进程，无记忆）"]
            READ["读策略 + 知识库"]
            PHASE_ALL["/phase-all (1→11全量构建)"]
            
            subgraph TRIAD["三角色对抗协作"]
                IMPL["implementer<br/>写代码，只写不测<br/>→ SUBMITTED"]
                SPEC["spec-reviewer<br/>逐条对照00_contracts/<br/>只审不测<br/>→ PASS/FAIL"]
                VERIF["verification<br/>L0+L0.5+L0.6+L1+L2+L3<br/>只测不改<br/>→ PASS/FAIL"]
            end
            
            OUTPUT["产出 → master/results/ITER_ID/<br/>benchmarks.jsonl<br/>AGGREGATE_REPORT.md<br/>diagnostics_summary.json"]
            
            READ --> PHASE_ALL
            PHASE_ALL --> IMPL
            IMPL --> SPEC
            SPEC -->|"✅ PASS"| VERIF
            SPEC -->|"❌ FAIL"| IMPL
            VERIF --> OUTPUT
        end
        
        S5["Step5: 提取KPI<br/>吞吐/TTFT/TPOT/greedy"]
        S6["Step6: 对比裁决<br/>ADVANCE 或 ROLLBACK？"]
        S65["Step6.5: Issue Analyzer<br/>仅显著ROLLBACK触发<br/>KPI降>10%或greedy翻转<br/>→ 写 08_issues/"]
        S7["Step7: 展示对比<br/>表格 + Δ%"]
        S8["Step8: 持久化/丢弃<br/>ADVANCE→存代码<br/>ROLLBACK→清空"]
        S9["Step9: 更新状态<br/>state.json<br/>decision-log.jsonl"]
        S10["Step10: 循环<br/>回到 Step1"]
        
        S1 --> S2 --> S3 --> S4
        S4 --> SUBAGENT
        SUBAGENT --> S5 --> S6
        S6 -->|"ADVANCE"| S7
        S6 -->|"显著ROLLBACK"| S65
        S65 --> S7
        S7 --> S8 --> S9
        
        S9 -->|"only if ADVANCE"| S95
        S9 --> S10 --> S1
    end
    
    subgraph CIRCUIT_C["Step9.5: 回路C触发"]
        SIGNAL_CHECK{"检查知识信号<br/>━━━━━━━━<br/>① throughput_delta > 5%？<br/>② 新策略模式？<br/>③ 跨≥3轮连续ADVANCE？<br/>④ 刚修复ROLLBACK根因？"}
        SIGNAL_COUNT{"信号 ≥ 2？"}
        LAUNCH_C["启动 experiment-summarizer<br/>claude -p 进程隔离<br/>→ 写 knowledge_delta.json<br/>→ 追加 notebooks-cn/"]
        SKIP_C["NO_NEW_KNOWLEDGE<br/>跳过，继续循环"]
    end
    
    S95["Step9.5: 知识回流判断"] --> SIGNAL_CHECK
    SIGNAL_CHECK --> SIGNAL_COUNT
    SIGNAL_COUNT -->|"是"| LAUNCH_C
    SIGNAL_COUNT -->|"否"| SKIP_C
```

---

## 四、回路 B（知识进化路径）：新模型学习循环

```mermaid
stateDiagram-v2
    [*] --> EntryCheck: 触发条件<br/>① CLAUDE.md Step6 判定"未覆盖"→entry_reason=coverage_fail<br/>② 用户手动 /evolve model_id→entry_reason=manual

    state EntryCheck {
        [*] --> CheckReason: 读 evolution/state.json
        CheckReason --> CoverageFail: entry_reason==coverage_fail
        CheckReason --> Manual: entry_reason==manual
    }

    CoverageFail --> Round2: ⛔ KB明确不覆盖<br/>跳过Round1<br/>直接从SWITCH=ON开始

    state Manual {
        [*] --> R1_Attempt: phase=attempt_without_opensource<br/>SWITCH=OFF
        R1_Attempt --> R1_Explorer: Explorer: skip(纯知识库)
        R1_Explorer --> R1_Impl: Implementer: /phase-all
        R1_Impl --> R1_Verify: Verification: scripts/全部门禁
        R1_Verify --> R1_Judge: 全部PASS？
        R1_Judge --> DONE: 是 🎉 (KB已够)
        R1_Judge --> Round2: 否 (知识不够)
    }

    state Round2 {
        [*] --> R2_Setup: phase=attempt_with_opensource<br/>SWITCH=ON
        R2_Setup --> R2_Explorer: Explorer: full<br/>WebSearch论文/技术报告<br/>WebFetch HF config.json<br/>读 knowledge/vllm/ + knowledge/sglang/
        R2_Explorer --> R2_Output: 产出 exploration_report.md<br/>+ model_diff.json
        R2_Output --> R2_Impl: Implementer: 读探索报告+/phase-all
        R2_Impl --> R2_Verify: Verification: scripts/全部门禁
        R2_Verify --> R2_Judge: 全部PASS？
        R2_Judge --> Round3: 是 (进入固化)
        R2_Judge --> R2_Issue: 否 → 启动issue-analyzer<br/>写08_issues/
        R2_Issue --> R2_FailCheck: failure_count≥3？
        R2_FailCheck --> R2_Explorer: <3 (Explorer增量收集)
        R2_FailCheck --> HUMAN: ≥3 (暂停请求人类<br/>08_issues/含全部失败记录)
    }

    state Round3 {
        [*] --> R3_Consolidate: Knowledge Consolidator
        R3_Consolidate --> R3_Read: 读探索报告+成功代码+验证报告
        R3_Read --> R3_Write: 写入 notebooks-cn/<br/>02_model_specifics/<br/>06_experience/<br/>07_improvementPlan/<br/>00_contracts/ (需人类确认)
        R3_Write --> R3_Delta: 产出 knowledge_delta.json
        R3_Delta --> R3_Switch: SWITCH=OFF<br/>phase=verify_without_opensource
    }

    state Round4 {
        [*] --> R4_Attempt: phase=verify_without_opensource<br/>SWITCH=OFF(纯知识库)
        R4_Attempt --> R4_Explorer: Explorer: skip
        R4_Explorer --> R4_Impl: Implementer: 仅读更新后KB+/phase-all
        R4_Impl --> R4_Verify: Verification: scripts/全部门禁
        R4_Verify --> R4_Judge: 全部PASS？
        R4_Judge --> DONE: 是 🎉 (进化成功<br/>可切换 stage=tuning)
        R4_Judge --> R4_Issue: 否 → 启动issue-analyzer<br/>写08_issues/
        R4_Issue --> Round2: 知识还不够<br/>继续迭代
    }

    DONE: ["🎉 进化成功<br/>知识库已更新<br/>委派回 master/ 正常迭代"]

    Manual --> Round2
    Round2 --> Round3
    Round3 --> Round4
```

---

## 五、回路 C（知识回流）：实验经验 → 知识库

```mermaid
flowchart TD
    TRIGGER["触发点: master/MASTER.md Step9之后<br/>条件: verdict == ADVANCE"]
    
    TRIGGER --> DETECTOR{"信号检测器<br/>━━━━━━━━<br/>检查本轮是否有值得持久化的知识"}
    
    subgraph SIGNALS["四类知识信号"]
        S1["显著性能增益<br/>throughput > +5%<br/>TTFT/TPOT < -10%<br/>来源: benchmarks"]
        S2["新策略模式<br/>algorithmic/architectural<br/>且历史上首次<br/>来源: strategy.json"]
        S3["跨轮确认趋势<br/>≥3轮同方向连续ADVANCE<br/>来源: decision-log.jsonl"]
        S4["错误修复<br/>上一轮ROLLBACK<br/>本轮修复+ADVANCE<br/>来源: history[] + verdict"]
    end
    
    DETECTOR --> S1
    DETECTOR --> S2
    DETECTOR --> S3
    DETECTOR --> S4
    
    S1 --> COUNT{"信号 ≥ 2？"}
    S2 --> COUNT
    S3 --> COUNT
    S4 --> COUNT
    
    COUNT -->|"否"| SKIP["NO_NEW_KNOWLEDGE<br/>跳过，继续循环"]
    
    COUNT -->|"是"| SUMMARIZER["启动 Experiment Summarizer<br/>claude -p 进程隔离"]
    
    subgraph SUMMARIZER_INTERNAL["Experiment Summarizer 内部逻辑"]
        INPUT["输入:<br/>strategies/strategy-ID.json<br/>results/ID/benchmarks.jsonl<br/>results/ID/diagnostics_summary.json<br/>decision-log.jsonl<br/>notebooks-cn/现有知识"]
        
        CLASSIFIER{"知识归属分类器"}
        CAT1["调试技巧 → 06_experience/"]
        CAT2["参数规律 → 07_improvementPlan/"]
        CAT3["kernel优化 → 07_improvementPlan/"]
        CAT4["模型理解 → 02_model_specifics/"]
        CAT5["算子特征 → 03_operators/"]
        CAT6["并行策略 → 04_parallel_strategies/"]
        CAT7["框架约束 → 01_framework_design/"]
        CAT_BLOCK["⛔ 不写 00_contracts/"]
        
        OUTPUT_DELTA["输出 knowledge_delta.json<br/>iter_id/strategy/kpi_delta<br/>knowledge_category/target_file<br/>insight_summary/detail<br/>signal_strength:<br/>  confirmed(≥3轮)<br/>  probable(2轮)<br/>  tentative(仅本轮→只写json,不追加md)"]
        
        OUTPUT_MD["追加 notebooks-cn/target.md<br/>## Δ from iter-ID<br/>日期/策略/效果/根因/适用条件/信号强度"]
    end
    
    SUMMARIZER --> INPUT
    INPUT --> CLASSIFIER
    CLASSIFIER --> CAT1
    CLASSIFIER --> CAT2
    CLASSIFIER --> CAT3
    CLASSIFIER --> CAT4
    CLASSIFIER --> CAT5
    CLASSIFIER --> CAT6
    CLASSIFIER --> CAT7
    CLASSIFIER --> CAT_BLOCK
    CAT1 --> OUTPUT_DELTA
    CAT2 --> OUTPUT_DELTA
    CAT3 --> OUTPUT_DELTA
    CAT4 --> OUTPUT_DELTA
    CAT5 --> OUTPUT_DELTA
    CAT6 --> OUTPUT_DELTA
    CAT7 --> OUTPUT_DELTA
    OUTPUT_DELTA --> OUTPUT_MD
```

---

## 六、三角色对抗协作流（双轨制 + 进程隔离标注）

**标注约定**：`-->>` 虚线 = Agent 工具 spawn，`==>` 粗线 = Shell `claude -p` 独立进程。

```mermaid
flowchart TD
    MAIN["主Agent<br/>读契约 → 拆Task<br/>派子Agent → 收集结果<br/>⛔ 自己不写/不审/不测"]
    
    MAIN --> TRACK_SELECT{"实现任务类型？"}
    
    TRACK_SELECT -->|"首次大段构建<br/>新Phase"| FULL["完整串行路径<br/>impl → spec → verify"]
    TRACK_SELECT -->|"驳回后修复<br/>改动 ≤10行"| FAST["快速修复路径<br/>impl → verify 闭环"]
    TRACK_SELECT -->|"驳回后修复<br/>改动 >10行"| FULL
    
    subgraph FULL_TRACK["完整串行路径（首次大段构建，强制）"]
        F_IMPL["implementer<br/>[Agent工具 spawn]<br/>写代码 + 自读diff<br/>→ SUBMITTED<br/>⛔不跑测试，不宣判PASS"]
        F_SPEC["spec-reviewer<br/>[Shell claude -p 独立进程]<br/>对照契约逐条审查<br/>独立读代码<br/>⛔不跑测试<br/>→ PASS / FAIL"]
        F_VERIF["verification<br/>[Shell claude -p 独立进程]<br/>唯一测试执行者<br/>L0防假PASS + L0.5自检<br/>L0.6 agent自检<br/>L1 scripts/ + L2跨Phase<br/>L3 profiler<br/>→ PASS / FAIL"]
        
        F_IMPL -->|"[Agent] spawn"| F_SPEC
        F_SPEC -->|"✅ PASS"| F_VERIF
        F_SPEC -->|"❌ FAIL"| F_IMPL
        F_VERIF -->|"✅ PASS"| DELIVER["Phase 交付<br/>+ PID交叉验证<br/>PID(impl)≠PID(spec)≠PID(verif)≠PID(main)"]
        F_VERIF -->|"❌ FAIL"| F_IMPL
    end
    
    subgraph FAST_TRACK["快速修复路径（驳回后小修）"]
        FAST_IMPL["implementer<br/>[Agent工具 spawn]<br/>读FAIL报告 + 定位根因<br/>修改几行代码<br/>→ SUBMITTED<br/>⛔不跑测试"]
        FAST_VERIF["verification<br/>[Shell claude -p 独立进程]<br/>跑scripts/返回测试结果<br/>→ PASS / FAIL"]
        
        FAST_IMPL -->|"[Shell] claude -p"| FAST_VERIF
        FAST_VERIF -->|"✅ PASS"| DELIVER
        FAST_VERIF -->|"❌ FAIL"| FAST_RETRY{"连续2次FAIL？"}
        FAST_RETRY -->|"否"| FAST_IMPL
        FAST_RETRY -->|"是，升级"| F_IMPL
    end
    
    DELIVER --> MEMORY["写 MEMORY 文件<br/>phase_report/PHASEN_MEMORY.md"]
    MEMORY --> GIT["git commit 存档"]
```

---

## 七、完整示例：Qwen3.6 27B 时序图

```mermaid
sequenceDiagram
    actor User as 用户
    participant Main as 主Agent
    participant Env as .env_agent_infer
    participant Config as MODEL_DIR/config.json
    participant KB as notebooks-cn/知识库
    participant Evo as evolution/编排器
    participant Explorer as Explorer
    participant Impl as Implementer
    participant Verif as Verification
    participant Consolidator as Knowledge Consolidator
    participant IssueAnalyzer as Issue Analyzer
    participant Master as master/编排器
    participant Summarizer as Experiment Summarizer

    User->>Main: /phase-all (目标: Qwen3.6 27B)
    
    Note over Main,Env: Step 1: 环境配置
    Main->>Env: 检测 MODEL_DIR → Qwen3-8B
    Main->>User: AskUserQuestion → 用户提供 /data/model/Qwen3.6-27B
    Main->>Env: 更新 .env_agent_infer
    Main->>Config: 验证 config.json 可读 ✅
    
    Note over Main,Config: Step 2-3: 模型分析
    Main->>Config: 读 architectures/num_heads
    Main->>Main: 模型路由结论: Dense 模型
    
    Note over Main,KB: Step 4: 知识库覆盖检测
    Main->>KB: 检查 model_specs.md → 只有 Qwen3-8B ❌
    Main->>KB: 检查 02_model_specifics/ → 无 27B 文档 ❌
    Main->>Main: 判定: 未覆盖 → ⛔ 跳过无开源尝试<br/>entry_reason=coverage_fail → SWITCH=ON 直接进化
    
    Note over Evo: 第1轮: attempt_with_opensource (SWITCH=ON, explorer_mode=full)
    Main->>Evo: 委派进化编排器 (跳过 attempt_without_opensource)
    Evo->>Explorer: WebSearch "Qwen3.6 27B architecture" → 技术报告
    Evo->>Explorer: WebFetch HF config.json → 27B 实际维度
    Evo->>Explorer: 读 knowledge/vllm/qwen3_moe.py + sglang/qwen3.py
    Explorer-->>Evo: exploration_report.md + model_diff.json
    Evo->>Impl: 读探索报告 → 用正确维度 /phase-all
    Impl->>Verif: one_pass_rate=100% → 全部 PASS ✅
    Evo->>Consolidator: 启动 Knowledge Consolidator
    
    Note over Consolidator: 知识固化
    Consolidator->>KB: 写入 02_model_specifics/03_qwen3/02_27B_dense.md
    Consolidator->>KB: 追加 00_contracts/model_specs.md(需人类确认)
    Consolidator->>KB: 追加 06_experience/04_qwen3.6_tp_debug.md
    Consolidator-->>Evo: knowledge_delta.json
    Evo->>Evo: SWITCH=OFF, phase=verify_without_opensource
    
    Note over Evo: 第2轮: verify_without_opensource (SWITCH=OFF, 纯知识库重验)
    Evo->>Impl: 仅读更新后KB → /phase-all
    Impl->>Verif: one_pass_rate=100% → 全部 PASS ✅
    Evo-->>Main: 🎉 进化成功！知识库已更新
    
    Note over Main,Master: 委派回 master/ 回路A
    Main->>Master: 用更新后的KB跑迭代循环
    Master->>Master: 基线建立 → 诊断 → 策略 → 构建 → 验证
    Master->>Summarizer: ADVANCE后触发回路C
    Summarizer->>KB: 判定有无新知识 → 回流到 notebooks-cn/
    
    Note over User,KB: 最终: 知识库已包含 Qwen3.6 27B，推理框架已生成并可迭代优化 ✅
```

---

## 八、目录文件热力图（树形结构）

```mermaid
graph TD
    ROOT["metainferv3/"]
    
    ROOT --> CLAUDE["📄 CLAUDE.md<br/>🔴 P0 入口文件"]
    ROOT --> SKILL["📄 AGENT_SKILL.md<br/>🔴 P0 执行SOP+12条铁律"]
    ROOT --> FLOW["📄 PROJECT_FLOW.md<br/>🟢 P2 全局流程图"]
    ROOT --> ENV["📄 .env_agent_infer<br/>🟡 P1 环境变量(gitignore)"]
    ROOT --> GITIGNORE["📄 .gitignore<br/>🟢 P2"]
    
    ROOT --> NC["📁 notebooks-cn/<br/>🔴 P0 先验知识库"]
    NC --> NC_CONTRACTS["00_contracts/<br/>🔴 P0 11个API契约<br/>人类写, B/C慎改"]
    NC --> NC_OVERVIEW["00_overview/<br/>🟡 P1 项目总览"]
    NC --> NC_FW["01_framework_design/<br/>🟡 P1 框架设计"]
    NC --> NC_MODEL["02_model_specifics/<br/>🟡 P1 B/C可扩写"]
    NC --> NC_OPS["03_operators/<br/>🟡 P1 B/C可扩写"]
    NC --> NC_PARALLEL["04_parallel_strategies/<br/>🟡 P1 B/C可扩写"]
    NC --> NC_SERVICE["05_inference_service/<br/>🟢 P2"]
    NC --> NC_EXP["06_experience/<br/>🟡 P1 C自动追加"]
    NC --> NC_IMP["07_improvementPlan/<br/>🟡 P1 C自动追加"]
    NC --> NC_ISSUES["08_issues/<br/>🟡 P1 失败经验库<br/>进化+调优积累"]
    
    ROOT --> DOTCLAUDE["📁 .claude/"]
    DOTCLAUDE --> ROLES["roles/<br/>🔴 P0 7个Agent角色"]
    ROLES --> R_IMPL["implementer-inference.md"]
    ROLES --> R_SPEC["spec-reviewer-inference.md"]
    ROLES --> R_VERIF["verification-inference.md"]
    ROLES --> R_SUMM["experiment-summarizer.md<br/>🟢 P2 回路C"]
    ROLES --> R_EXPL["explorer.md<br/>🟢 P2 回路B"]
    ROLES --> R_CONS["knowledge-consolidator.md<br/>🟢 P2 回路B"]
    ROLES --> R_ISSUE["issue-analyzer.md<br/>🟢 P2 失败分析"]
    DOTCLAUDE --> SKILLS["skills/<br/>🟡 P1 Phase触发词"]
    
    ROOT --> MASTER_DIR["📁 master/<br/>🔴 P0 回路A编排器"]
    MASTER_DIR --> M_MASTER["MASTER.md"]
    MASTER_DIR --> M_STATE["state.json"]
    MASTER_DIR --> M_LOG["decision-log.jsonl"]
    MASTER_DIR --> M_STRATEGIES["strategies/"]
    MASTER_DIR --> M_RESULTS["results/"]
    MASTER_DIR --> M_SCRIPTS["scripts/call-sub-agent.sh"]
    
    ROOT --> EVO_DIR["📁 evolution/<br/>🟢 P2 回路B编排器"]
    EVO_DIR --> E_EVO["EVOLUTION.md"]
    EVO_DIR --> E_STATE["state.json"]
    EVO_DIR --> E_LOG["decision-log.jsonl"]
    EVO_DIR --> E_STRATEGIES["strategies/"]
    EVO_DIR --> E_RESULTS["results/"]
    EVO_DIR --> E_SCRIPTS["scripts/call-evo-agent.sh"]
    
    ROOT --> ENGINE["📁 engine/<br/>🔴 P0 生成产物:推理框架"]
    ENGINE --> E_KERNELS["kernels/"]
    ENGINE --> E_MODELS["models/"]
    ENGINE --> E_FW["framework/"]
    ENGINE --> E_TP["tp_layers/"]
    ENGINE --> E_SC["self_check.py"]
    
    ROOT --> LLM["📄 llm_engine.py<br/>🔴 P0 引擎主循环"]
    ROOT --> OAI["📄 openai_tp_server.py<br/>🔴 P0 OpenAI API服务"]
    ROOT --> SCRIPTS_DIR["📁 scripts/<br/>🔴 P0 28个测试合约(只读)"]
    ROOT --> REPORTS["📁 phase_report/<br/>🟡 P1 审查报告(gitignore)"]
    ROOT --> KNOWLEDGE["📁 knowledge/<br/>🟢 P2 开源代码缓存(gitignore)"]
    ROOT --> ITERATIONS["📁 iterations/<br/>🟢 P2 迭代工作区(gitignore)"]
```

---

## 九、关键决策节点速查

```mermaid
flowchart LR
    subgraph NODES["10个关键决策节点"]
        N1["① 环境配置<br/>CLAUDE.md Step0<br/>.env存在且匹配？<br/>是→Step1 / 否→问用户"]
        N2["② 模型分析<br/>CLAUDE.md Step6<br/>模型在KB覆盖范围？<br/>是→回路A / 否→回路B"]
        N3["③ 平台检测<br/>CLAUDE.md Step7<br/>GPU类型？<br/>NVIDIA→CustomAR<br/>AMD→RCCL fallback"]
        N4["④ 代码质量<br/>三角色流<br/>spec-reviewer？<br/>PASS→verification<br/>FAIL→打回impl"]
        N5["⑤ 测试验收<br/>三角色流<br/>verification？<br/>PASS→交付<br/>FAIL→打回impl"]
        N6["⑥ KPI裁决<br/>master Step6<br/>ADVANCE或ROLLBACK？<br/>显著ROLLBACK→issue-analyzer<br/>ADVANCE→Step7"]
        N7["⑦ 知识回流<br/>master Step9.5<br/>知识信号≥2？<br/>是→summarizer<br/>否→跳过"]
        N8["⑧ 进化阶段<br/>evolution 第1轮<br/>无开源通过？<br/>是→DONE<br/>否→开源辅助"]
        N9["⑨ 进化验证<br/>evolution 重验阶段<br/>无开源重验通过？<br/>是→DONE<br/>否→继续迭代"]
        N10["⑩ 失败分析<br/>显著失败时触发<br/>进化/调优任一失败→<br/>spawn issue-analyzer<br/>→ 写 08_issues/"]
    end
```

---

## 附录：三条回路总览

```mermaid
flowchart TD
    KB["notebooks-cn/<br/>先验知识库<br/>← 回路B/C可更新"]
    
    subgraph CIRCUIT_A["回路A: 成果路径"]
        A1["知识库 → 生成 → 验证"]
        A2["master/MASTER.md 编排<br/>KPI驱动，无限循环"]
    end
    
    subgraph CIRCUIT_B["回路B: 知识进化路径"]
        B1["Explorer → Implementer<br/>→ Consolidator → 重验"]
        B2["evolution/EVOLUTION.md 编排<br/>模型覆盖驱动，收敛即止"]
        B3["⛔ 无开源可独立生成"]
    end
    
    subgraph CIRCUIT_C["回路C: 知识回流"]
        C1["experiment-summarizer"]
        C2["ADVANCE时触发<br/>抽象实验收益<br/>更新 notebooks-cn/"]
    end
    
    KB --> CIRCUIT_A
    KB --> CIRCUIT_B
    CIRCUIT_A -->|"ADVANCE时触发"| CIRCUIT_C
    CIRCUIT_C -->|"写入"| KB
    CIRCUIT_B -->|"写入"| KB
```

---

> **LLM 解析提示**：每个 Mermaid 代码块都是独立、自包含的图。LLM 解析时关注 `subgraph`（子系统边界）、`-->|"标签"|`（带语义的边）、`{}`（决策菱形节点）即可重建完整的项目拓扑。
