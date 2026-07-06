# MetaInfer v3 项目流程说明书

这份文档用纯文字叙述 MetaInfer v3 的完整工作流程。不需要看任何图，读完每一节你就知道那个环节在做什么。

---

## 一、全局架构：四层 + 三条回路

整个项目分成四层，从上到下依次是：

**第一层：先验知识库（`notebooks-cn/`）**

这是整个系统的"大脑"——所有已知的模型知识、API 契约、调试经验都存在这里。其中最重要的子目录是 `00_contracts/`，里面有 11 个 API 契约文件，这些是硬约束：Agent 写代码时不能违反契约中的任何一条规定，只有人类确认后才能修改契约本身。知识库的其他部分（比如模型族文档 `02_model_specifics/`、调试经验 `06_experience/`、优化方案 `07_improvementPlan/`）可以由回路 B 和回路 C 自动追加更新。

同级还有两个关键文件：`AGENT_SKILL.md` 是执行 SOP 和编码铁律（12 条硬规则），`CLAUDE.md` 是全局入口和子 Agent 的 spawn 协议。`scripts/` 目录下有 28 个测试合约，只读，任何 Agent 都不能修改。

**第二层：Agent 角色群（`.claude/roles/`）**

这里定义了 6 个角色，每个角色有独立的 prompt 文件，互不信任：

- **implementer**：只负责写代码，写完自读 diff 确认没改错文件，然后提交。不跑测试，不宣判 PASS。
- **spec-reviewer**：独立对照 `00_contracts/` 契约逐条审查代码。只审不测，它和 implementer 是物理隔离的（不同进程）。
- **verification**：唯一有权跑测试和宣判 PASS 的角色。它执行六层验证（L0 到 L3），只测不改。
- **experiment-summarizer**：回路 C 的角色，把性能实验的经验抽象后回流到知识库。
- **explorer**：回路 B 的角色，搜索论文、HuggingFace、开源代码来理解新模型。
- **knowledge-consolidator**：回路 B 的角色，把探索成果固化为知识库文档。

**第三层：生成产物**

这三个是 Agent 实际产出的代码：`engine/`（推理框架内核）、`llm_engine.py`（引擎主循环）、`openai_tp_server.py`（OpenAI 兼容 API 服务）。

**第四层：编排器**

- `master/`：回路 A 的编排器，负责性能迭代的主循环。它读状态、诊断瓶颈、写策略、调度子 Agent、对比 KPI、裁决是否采纳改动，无限循环直到用户退出。
- `evolution/`：回路 B 的编排器，负责知识进化。当遇到知识库不认识的模型时启动，通过探索→构建→固化→重验的四轮循环把新模型知识写进知识库。

**四层之间的数据流向**：知识库被第二层读取，第二层生成第三层的代码，第四层编排第二层和第三层。回路 B 和回路 C 会把新知识写回知识库。

---

## 二、主入口：从用户指令到开始构建

当用户输入触发词（如 `/phase-all`、`/phase5`、`/evolve`）后，主 Agent 按 8 步启动流程走：

**Step 0 — 环境配置**：先检查 `.env_agent_infer` 文件是否存在。如果存在就加载，然后核对里面的 `MODEL_DIR` 是否指向用户当前要用的模型。不匹配就询问用户正确的路径。如果文件不存在则一次性询问两个问题：模型目录在哪？Python 环境（conda/venv 的 bin 目录）在哪？拿到答案后写入 `.env_agent_infer` 并验证——确认 `config.json` 可读、`torch` 可 import。

**Step 1-3 — 加载上下文**：读取当前 Phase 对应的契约文件、`AGENT_SKILL.md`，设置好 `PYTHONPATH`。

**Step 4-5 — 模型分析**：读取模型的 `config.json`，提取架构类型（`architectures`）、头数（`num_heads`）、RoPE 缩放方式等。输出"模型路由结论"——这个模型是标准的 Dense 架构还是 MLA+MoE 架构。

**Step 6 — 知识库覆盖检测**（最关键的分叉口）：检查三个条件是否全部满足：
1. `model_specs.md` 里有这个模型的维度/架构参数
2. `02_model_specifics/` 里有这个模型或同系列的详细文档
3. `00_contracts/` 里与这个模型架构类型匹配的契约文件已就绪

三项全满足 → 直接进入回路 A+C 构建。任一不满足 → 先启动回路 B（进化路径），等进化成功（知识库已更新）后再回到回路 A+C。

**Step 7 — 平台自动检测**：运行 `torch.cuda.get_device_name(0)` 判断是 NVIDIA、AMD 还是 DCU。NVIDIA 用 CustomAR + NCCL 通信，AMD 用 RCCL fallback，DCU 用更基础的通信 fallback。

**Step 8 — MEMORY 回溯**：检查 `phase_report/` 下是否有前序 Phase 的 `PHASE<N>_MEMORY.md`。如果有就读取最近完成的 Phase 的记录，快速重建上下文（已完成的 Phase、通过的脚本、关键文件改动）。这一步对长对话恢复至关重要。

走完 8 步后，进入实际的构建或迭代流程。

---

## 三、回路 A：性能迭代主循环

回路 A 是系统的"日常运转"模式，由 `master/MASTER.md` 编排，是一个无限循环，每一轮包含 10 步（加上半步回路 C 触发）：

**Step 1 — 加载状态**：读取 `state.json`，获取当前基线 KPI 和历史记录。

**Step 2 — 诊断**：分析上一轮的 KPI（吞吐、TTFT、TPOT、greedy 准确率），定位性能瓶颈。比如发现 attention kernel 耗时占比过高，或者 TP 通信成为瓶颈。

**Step 3 — 写策略**：根据诊断结果写一个策略文件 `strategy-XXX.json`，明确这一轮要改什么、预期提升多少。策略是一个结构化的 JSON，包含改动范围、目标指标、风险评估。

**Step 4 — 启动子 Agent**：用 `claude -p` 启动一个全新进程（物理隔离，无父进程记忆）。这个子 Agent 内部先读策略和知识库，然后跑 `/phase-all`（Phase 1 到 11 全量构建）。构建过程走三角色对抗协作流：implementer 写代码 → spec-reviewer 逐条对照契约审查 → verification 跑六层测试验证。子 Agent 的产出（benchmarks、汇总报告、诊断摘要）写入 `master/results/<ITER_ID>/`。

**Step 5 — 提取 KPI**：从子 Agent 的产出中提取关键指标：吞吐量、TTFT（首 token 延迟）、TPOT（每 token 延迟）、greedy 对齐率。

**Step 6 — 对比裁决**：把本轮 KPI 和基线对比。决定是 ADVANCE（采纳，基线前进）还是 ROLLBACK（拒绝，回退到上一轮代码）。

**Step 7 — 展示对比**：用表格展示本轮 vs 基线的差异，包括绝对值变化和百分比 Δ。

**Step 8 — 持久化/丢弃**：ADVANCE → 保留本轮代码，更新基线。ROLLBACK → 清空本轮改动，恢复上一轮代码。

**Step 9 — 更新状态**：写入 `state.json`（新基线 KPI）和 `decision-log.jsonl`（追加本轮裁决记录）。

**Step 9.5 — 回路 C 触发判断**（只在 ADVANCE 时触发）：检查四个知识信号中是否有至少两个满足——显著性能增益（吞吐 +5% 以上或延迟 -10% 以上）、新策略模式（算法或架构层面的创新且历史上首次出现）、跨轮确认趋势（连续 3 轮以上同方向 ADVANCE）、错误修复（上一轮 ROLLBACK 的根因在本轮被修复）。满足条件就启动 experiment-summarizer 把经验回流到知识库，不满足就跳过。

**Step 10 — 循环**：回到 Step 1，开始下一轮迭代。

---

## 四、回路 B：知识进化路径

回路 B 处理"遇到不认识的新模型"的场景，由 `evolution/EVOLUTION.md` 编排。它是一个四轮状态机：

**触发条件**有三个：CLAUDE.md Step 6 判定模型未覆盖 → `entry_reason=coverage_fail`、用户手动输入 `/evolve <model_id>` → `entry_reason=manual`、master 循环中首次 `/phase-all` 全部失败。

**入口路由**：根据 `entry_reason` 决定起始阶段。`coverage_fail` 表示 KB 明确不覆盖 → 跳过无开源尝试，直接从第 1 轮（开源辅助）开始。`manual` 表示用户手动触发 → 先尝试第 0 轮（无开源尝试），看看 KB 是否其实够用。

**第 0 轮（仅 manual 入口）：无开源尝试**

SWITCH 开关设为 OFF。Explorer 跳过，Implementer 仅凭现有知识库尝试 `/phase-all`。如果全部通过 → KB 其实够用，进化直接成功。如果失败 → 进入第 1 轮。

**第 1 轮：开源辅助**

SWITCH 开关设为 ON。Explorer 执行完整搜索（WebSearch 论文、WebFetch HF 模型卡和 config.json、读 knowledge/vllm + knowledge/sglang 源码）。输出探索报告后，Implementer 基于探索报告 `/phase-all` 全量构建。失败时 → 写 `notebooks-cn/08_issues/` 记录根因，Explorer 增量收集信息后重试，连续 3 次失败则暂停请求人类。

**第 2 轮：开源辅助探索**

SWITCH 设为 ON，允许参考开源代码。Explorer 全面搜索：WebSearch 搜论文和技术报告、WebFetch 拉 HuggingFace 的 config.json 获取实际维度参数、读模型卡了解架构说明、读 `knowledge/vllm/` 下缓存的 vLLM 源码分析实现模式。产出两份文件：`exploration_report.md`（探索报告，描述模型架构、与已知模型的差异、关键维度参数）和 `model_diff.json`（结构化的差异对比）。Implementer 读探索报告后用正确的维度和架构差异跑 `/phase-all`。Verification 跑全部门禁。如果 PASS → 进入第 3 轮固化。如果 FAIL → 失败计数 +1。失败 <3 次则 Explorer 增量收集更多信息后重试；失败 ≥3 次则暂停，请求人类介入。

**第 3 轮：知识固化**

Knowledge Consolidator 读取探索报告、成功代码、验证报告，把新知识写入知识库：新模型文档写入 `02_model_specifics/`、调试经验写入 `06_experience/`、注意事项写入 `07_improvementPlan/`。如果需要修改 `00_contracts/`（契约），则标记为"需人类确认"——Consolidator 不能自动改契约。产出 `knowledge_delta.json` 记录本次进化新增了什么知识。最后 SWITCH 切回 OFF，进入第 4 轮。

**第 4 轮：无开源重验**

SWITCH=OFF，纯靠更新后的知识库（不能看开源代码）。Explorer 跳过。Implementer 只读更新后的知识库，再次跑 `/phase-all`。Verification 跑全部门禁。如果全部 PASS → 说明固化后的知识库已经自给自足，进化真正成功。如果 FAIL → 说明知识还不够，回到第 2 轮继续迭代（再搜开源资料、再固化、再重验）。

进化成功后，知识库已包含新模型的完整知识，委派回 master/ 回路 A 进行正常的性能迭代。

---

## 五、回路 C：知识回流

回路 C 解决"实验跑多了经验丢失"的问题。它只在回路 A 的 Step 9（更新状态）之后、且裁决为 ADVANCE 时触发。

**四类知识信号**（检查本轮是否有值得持久化的知识）：
1. 显著性能增益 — 吞吐提升超过 5% 或 TTFT/TPOT 降低超过 10%，来源是 benchmark 数据
2. 新策略模式 — 算法或架构层面的创新策略，且历史上首次出现，来源是 strategy.json
3. 跨轮确认趋势 — 连续 3 轮以上同方向 ADVANCE，说明优化方向可靠，来源是 decision-log.jsonl
4. 错误修复 — 上一轮 ROLLBACK 的根因在本轮被定位并修复，且本轮 ADVANCE

以上信号中至少有 2 个满足时，触发 Experiment Summarizer。

**Experiment Summarizer 的工作流程**：

输入是策略文件、benchmark 数据、诊断摘要、裁决日志和现有知识库。它先跑一个"知识归属分类器"，把本轮经验归到对应的知识目录：

- 调试技巧 → `06_experience/`
- 参数规律 / kernel 优化方案 → `07_improvementPlan/`
- 模型理解 → `02_model_specifics/`
- 算子特征 → `03_operators/`
- 并行策略 → `04_parallel_strategies/`
- 框架约束 → `01_framework_design/`
- 绝对不写 `00_contracts/`（契约只能人类改）

输出两份产物：`knowledge_delta.json`（结构化记录本轮知识的 iter_id、策略、KPI 变化、知识类别、目标文件、摘要、信号强度）和对应 `notebooks-cn/` 文件的追加段落。信号强度分三档：
- confirmed（≥3 轮确认）：可靠，追加到 md 文件
- probable（2 轮确认）：较可靠，追加到 md 文件
- tentative（仅本轮）：不追加 md，只存在 json 里等后续确认

---

## 六、三角色对抗协作流：代码怎么被写出来

这是每次 Phase 构建的具体执行方式，分两条轨：

**完整串行路径**（首次大段构建或驳回后改动超过 10 行，强制走这条）：

1. **implementer** 被 spawn（Agent 工具），读契约和 AGENT_SKILL 后写代码。写完自读 diff 确认没有误改 scripts/ 或其他不该碰的文件。报告状态为 SUBMITTED（不是 DONE，不是 PASS）。不跑任何测试。

2. **spec-reviewer** 被 spawn（Shell `claude -p`，进程隔离）。它不能读 implementer 的报告——只能读代码文件本身。对照 `00_contracts/` 中的每一条契约规定逐行审查。发现违反契约 → 报告 FAIL，列出具体条款、文件行号和修复建议。全部符合 → 报告 PASS。

   如果 spec-reviewer 报告 FAIL，直接打回 implementer，verification 不启动。不存在"这个 FAIL 很小可以忽略"——spec-reviewer 的 FAIL 就是 FAIL。

3. **verification**（只在 spec-reviewer PASS 后启动）被 spawn（Shell `claude -p`，进程隔离）。执行六层验证：
   - L0：防假 PASS —— 确认 import 的代码真的来自本目录而不是外部泄漏的包
   - L0.5：self_check 反作弊预检 —— 运行时验证代码不是 no-op
   - L0.6：Agent 自检 5 条 —— 静态分析测试覆盖盲区（no-op 路径检测、模拟 vs 真实执行、参考对比缺口、副作用可见性、边界注入测试）
   - L1：跑当前 Phase 的所有 scripts/ 脚本
   - L2（Phase 3+）：跨 Phase 回归 —— 重跑所有前序 Phase 的 scripts/
   - L3（Phase 10 强制）：profiler trace + 显存监控证据

   L0 + L0.5 + L0.6 全部 PASS 且 L1 全部 PASS → Phase 交付。任一 FAIL → 打回 implementer。

4. 主 Agent 做防假抽查：从 verification 声称通过的脚本里随机抽一个亲自重跑，比对输出是否一致。一致 → 可信。不一致 → verification 报告作假，整个 Phase 驳回，重新 spawn verification。

**快速修复路径**（驳回后修复，改动不超过 10 行）：

跳过 spec-reviewer，直接 implementer → verification 闭环。verification 的测试结果作为反馈信号。如果连续 2 次快速修复仍然 FAIL，强制升级为完整串行路径（加入 spec-reviewer）。

**两条轨的共同红线**：implementer 永远不跑测试，verification 永远不改代码。

**Phase 交付后**：主 Agent 写入 `phase_report/PHASE<N>_MEMORY.md`（记录时间戳、通过的脚本清单、改动文件、PID 交叉验证、L0.6 结论、错误和修复方式），然后 git commit 存档。

---

## 七、完整示例：Qwen3.6 27B 从头到尾是怎么被支持的

假设用户输入 `/phase-all`，目标模型是 Qwen3.6 27B：

1. 主 Agent 检测到 `.env_agent_infer` 里的 `MODEL_DIR` 指向 Qwen3-8B，和用户目标不匹配，弹窗询问。用户提供新路径 `/data/model/Qwen3.6-27B`。主 Agent 更新 `.env_agent_infer`，验证 `config.json` 可读。

2. 读 `config.json`：架构是 `Qwen3ForCausalLM`，27B 参数。输出"模型路由结论：Dense 模型"。

3. 知识库覆盖检测：`model_specs.md` 里只有 Qwen3-8B 的参数，没有 27B；`02_model_specifics/` 只有 8B 的文档。两项不满足 → `entry_reason=coverage_fail` → 跳过无开源尝试，直接从 SWITCH=ON 开始。

4. 进化第 1 轮（开源辅助，SWITCH=ON）：Explorer 搜索 Qwen3.6 27B 的技术报告、从 HuggingFace 拉实际 config.json 获取 27B 的正确维度参数、读 vLLM 的 qwen3 实现分析 Dense 模式。产出探索报告和差异对比。Implementer 用正确维度重跑 `/phase-all` → 100% 通过。启动 Knowledge Consolidator。

6. Knowledge Consolidator 把 27B 的模型文档、调试经验、维度参数写入知识库。契约文件需要人类确认的部分标记出来。SWITCH 切回 OFF，进入重验。

7. 进化第 4 轮（无开源重验）：纯靠更新后的知识库重跑 `/phase-all` → 100% 通过。进化成功。

8. 委派回 master/ 回路 A，用更新后的知识库跑性能迭代循环。

9. master 循环中每次 ADVANCE 后触发回路 C，判断有没有值得持久化的新知识，有就回流到知识库。

最终：知识库里有了 Qwen3.6 27B 的完整知识，推理框架代码已生成，后续可以持续迭代优化。

---

## 八、目录结构和优先级

按优先级分三档：

**P0（关键文件，每次都要读）**：
- `CLAUDE.md`：全局入口、架构说明、启动流程、spawn 协议
- `AGENT_SKILL.md`：执行 SOP、12 条编码铁律、Phase-Script 绑定表
- `notebooks-cn/00_contracts/`：11 个 API 契约文件（硬约束）
- `.claude/roles/`：6 个 Agent 角色定义
- `master/`：回路 A 编排器（MASTER.md + state.json + 策略/结果目录）
- `engine/`：推理框架内核
- `llm_engine.py`、`openai_tp_server.py`：引擎主循环和 API 服务
- `scripts/`：28 个测试合约（只读）

**P1（经常使用，按需读）**：
- `notebooks-cn/` 下的大部分子目录（框架设计、模型族、算子、并行策略、经验、优化方案）
- `.claude/skills/`：Phase 触发词的任务卡
- `phase_report/`：各 Phase 的审查报告和 MEMORY 文件
- `.env_agent_infer`：环境变量（机器独立，不提交 git）

**P2（新增或辅助，按需读）**：
- `evolution/`：回路 B 编排器
- `knowledge/`：开源代码缓存（vLLM、SGLang）
- `iterations/`：迭代工作区

---

## 九、关键决策节点速查

项目运行中有 9 个决策分叉点，理解它们就能理解整个系统的控制流：

1. **环境配置**（CLAUDE.md Step 0）：`.env_agent_infer` 存在且 MODEL_DIR 匹配？是 → 继续 Step 1。否 → 询问用户。

2. **知识库覆盖**（CLAUDE.md Step 6）：模型在知识库覆盖范围内（三项全满足）？是 → 直接回路 A+C 构建。否 → 先走回路 B 进化。

3. **平台检测**（CLAUDE.md Step 7）：GPU 是 NVIDIA → CustomAR + NCCL。AMD → RCCL fallback。DCU → 基础通信 fallback。

4. **代码质量门禁**（三角色流）：spec-reviewer 审查结果？PASS → 进入 verification。FAIL → 打回 implementer，verification 不启动。

5. **测试验收门禁**（三角色流）：verification 六层验证结果？PASS → Phase 交付。FAIL → 打回 implementer。

6. **KPI 裁决**（master Step 6）：本轮 vs 基线 KPI 对比？提升 → ADVANCE（采纳）。退步 → ROLLBACK（回退）。

7. **知识回流判断**（master Step 9.5）：四类知识信号中有 ≥2 个满足？是 → 启动 experiment-summarizer 回流。否 → 跳过。

8. **进化首轮判断**（evolution 入口路由）：`entry_reason == coverage_fail`（KB 明确不覆盖）→ 直接跳到第 1 轮开源辅助，不浪费时间。`entry_reason == manual` → 先尝试无开源，失败再进入第 1 轮。

9. **进化完成判断**（evolution 第 4 轮）：不开源的情况下重验全过？是 → 进化成功，委派回回路 A。否 → 回到第 2 轮继续迭代（还缺知识）。

---

## 附录：三条回路的关系

- **回路 A**（master/）是主干——日常运转就是它。知识库驱动代码生成，代码跑验证，验证通过后提取 KPI，KPI 进步就采纳、退步就回退，无限循环直到用户喊停。它解决"怎么把推理框架做得更快"。

- **回路 B**（evolution/）是分支——只在遇到知识库不认识的新模型时才启动。它通过四轮循环（无开源尝试→开源探索→知识固化→无开源重验）把新模型的知识"教会"知识库。收敛后回到回路 A。它解决"怎么支持一个新模型"。

- **回路 C**（experiment-summarizer）是回流管道——回路 A 每次 ADVANCE 后检查有没有值得持久化的经验。有的话就抽象归类、追加到知识库对应文件。它解决"怎么让每次实验的收益不丢失"。

三条回路合在一起形成一个闭环：知识库驱动构建 → 构建产生实验数据 → 实验数据回流成新知识 → 新知识驱动下一轮构建。
