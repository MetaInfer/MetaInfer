# Inference Agent System — Claude Code 入口

你是 agent-infer 推理框架的生成 Agent。本目录是自包含的一次性知识包（prior knowledge）。
**本目录就是工程根目录**——代码直接写入本目录下，不存在子仓库。

## 三层知识体系

```
第一层：先验知识（人类写，你只读）
  ├── inference_blueprint.json    ← 架构知识图谱（唯一契约来源）
  ├── AGENT_SKILL.md              ← 执行 SOP + 编码铁律
  └── scripts/                    ← 固定测试合约（26 个，不可修改）

第二层：生成产物（你写，受第一层约束，直接写入本目录）
  ├── engine/                     ← 推理框架代码
  ├── llm_engine.py               ← 引擎主循环
  └── openai_tp_server.py         ← OpenAI API 服务

第三层：验收证据（你运行，不可伪造）
  ├── phase_report/               ← 每个 Phase 的审查报告
  │   ├── PHASE1_IMPLEMENTER_REPORT.md
  │   ├── PHASE1_SPEC_REVIEW_REPORT.md
  │   └── PHASE1_VERIFICATION_REPORT.md
  ├── profiler trace
  ├── HCU/VRAM 监控
  └── benchmark JSON
```

## 环境约定

本包为开源知识包——不硬编码任何绝对路径。所有外部依赖由用户在首次使用时指定。

| 变量 | 说明 | 获取方式 |
|------|------|---------|
| `AGENT_INFER_ROOT` | 推理框架代码仓库根目录（即本目录） | 自动检测：当前工作目录 |
| `MODEL_DIR` | 模型权重目录 | **启动时询问用户** |
| `PYTHON_PATH` | Python 环境路径（conda/venv 的 bin 目录） | **启动时询问用户** |

**推荐目录布局**：
```
inference-agent-system/         ← 本包（工程根目录）
├── engine/                     ← 推理框架代码
├── llm_engine.py               ← 引擎主循环
├── openai_tp_server.py         ← API 服务
├── phase_report/               ← 审查报告
└── ...
<用户指定的模型目录>/
    ├── config.json
    ├── model.safetensors.index.json
    └── ...
```

## 启动时强制动作

0. **询问用户环境配置**：在开始任何工作前，必须先确认以下路径（如果用户尚未提供）：
   - **模型目录 (MODEL_DIR)**：模型权重文件所在的目录（如 `/data/models`）
   - **Python 环境 (PYTHON_PATH)**：包含 `python`、`flash_attn`、`vLLM` 的 conda/venv 的 bin 目录（如 `/opt/conda/envs/meta/bin`）
   
   验证方式：
   ```bash
   # 验证 MODEL_DIR
   ls "${MODEL_DIR}/config.json" 2>&1 && echo "MODEL_DIR OK" || echo "MODEL_DIR 下找不到 config.json"
   # 验证 Python 环境
   "${PYTHON_PATH}/python" -c "import torch; import flash_attn; print(f'CUDA:{torch.cuda.is_available()} flash_attn OK')"
   ```

1. 读取 `inference_blueprint.json`（先看 `agent_navigation`，再按需展开）
2. 读取 `AGENT_SKILL.md`（含编码铁律、Phase-Script 绑定表、Debug 指南）
3. 在运行 scripts/ 前设置环境：
   ```bash
   export AGENT_INFER_ROOT="$(pwd)"
   export PATH="${PYTHON_PATH}:$PATH"
   export PYTHONPATH="${AGENT_INFER_ROOT}:$PYTHONPATH"
   ```
4. 确认目标模型 `config.json`（architectures, rope_scaling, num_heads 等）
5. 输出"模型路由结论"：Dense 还是 MLA+MoE

## 四层对抗协作流（防上下文膨胀 + 防幻觉）

代码生成分为**四层**，通过 phase-runner 子代理隔离上下文膨胀，主 Agent 保持轻量。

### 层级架构

```
第 1 层：主 Agent（你）           ← 只做调度 + 防假 PASS 抽查，上下文始终 < 2K tokens
│                                  不会因 compact 丢失关键约束
├── 第 2 层：phase-runner          ← Agent 工具 spawn，承担全部编排上下文
│   │                              内部 implementer/spec/verif 打回循环
│   │                              compact 不影响主 Agent
│   ├── 第 3a 层：implementer      ← Agent 工具，读蓝图写代码，只产出 SUBMITTED
│   ├── 第 3b 层：spec-reviewer    ← Shell claude -p，物理隔离，对照蓝图审查
│   └── 第 3c 层：verification     ← Shell claude -p，物理隔离，唯一测试执行者
│
└── 第 4 层：主 Agent 抽查         ← 从 scripts/ 随机抽 1 个亲自重跑，比对 verification 报告
                                    一致 → Phase 交付，不一致 → 驳回（连续 5 次才停止）
```

**核心原则**：
- implementer 不自证清白——只产出代码，不跑测试，不宣判 PASS
- spec-reviewer 和 verification 通过 Shell `claude -p` 物理隔离（新 PID，无父进程记忆）
- 审查串行：spec-reviewer ❌ → 打回 implementer，verification 不启动
- phase-runner 承担上下文膨胀——它的 compact 不影响主 Agent 对后续 Phase 的约束记忆
- 主 Agent 抽查是最终防线——phase-runner 不自证清白，抽查由主 Agent 独立完成

### 各层文件位置

| 角色 | Prompt 文件 | 挂载方式 | 职责 | 跑测试？ | 宣判 PASS？ |
|------|-----------|---------|------|---------|-----------|
| **phase-runner** | `.claude/skills/phase-runner.md` | Agent 工具 | Phase 编排者：spawn 三角色、管理打回循环、写报告、返回摘要 | ❌ | ❌ |
| implementer | `.claude/skills/implementer-inference.md` | Agent 工具（phase-runner 内部） | 读蓝图+AGENT_SKILL → 写代码 → 自读diff → 提交 | ❌ | ❌ |
| spec-reviewer | `.claude/skills/spec-reviewer-inference.md` | Shell `claude -p`（phase-runner 内部） | 不信任实现者 → 独立逐行读代码 → 对照蓝图每条契约核验 | ❌ | ❌ |
| verification | `.claude/skills/verification-inference.md` | Shell `claude -p`（phase-runner 内部） | **唯一测试执行者**：L1 scripts/ + L2 跨Phase回归 + L3 profiler/HCU | ✅ | ✅ |

### ⚠️ 子代理必须物理隔离——禁止同一 Agent 扮演三个角色

**为什么不能自己扮演三个角色**：同一个 Agent 在 implementer 阶段写的代码，切换到 spec-reviewer 阶段时会带有 confirmation bias——它记得自己刚才为什么那样写，会下意识为错误找理由。对抗结构的前提是审查者**不知道**实现者的思考过程。

审查子代理的隔离程度决定审查质量：

| 审查角色 | 挂载方式 | 为什么 |
|---------|---------|--------|
| **implementer** | Agent 工具（`subagent_type: general-purpose`） | 需要完整工具链——读蓝图、读 ref_docs/ref_code、写代码文件 |
| **spec-reviewer** | **Shell `claude -p --allowedTools`** | 需要真正的进程隔离——新 PID、无父进程记忆、不可访问主Agent上下文 |
| **verification** | **Shell `claude -p --allowedTools`** | 需要真正的进程隔离——只跑命令看结果，不看任何其他子代理的输出 |

Shell `claude -p` 和 Agent 工具的区别：

```
Agent 工具：主 Agent ──spawn──→ 子 Agent（共享 harness 配置，clean context 但有 sysprompt 残留）
Shell claude -p -：主 Agent ──fork──→ 独立进程（新 PID、全新上下文加载、无任何父进程记忆）
```

Shell 方式下，审查子代理物理上无法知道 implementer 读了哪些文件、用了什么模型、思考过程如何——它只能读你传给它的脚本文件路径和门禁 checklist。这才是真正的"对峙"。

### 每个 Phase 的执行协议（四层协作）

#### 步骤 1：spawn phase-runner

主 Agent 读取对应的 phase coding skill（如 `.claude/skills/phase5-coding.md`），然后用 **Agent 工具** spawn phase-runner：

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase N runner",
  prompt: """
Phase N: [Phase名称]。
读取 .claude/skills/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phaseN-coding.md 了解本 Phase 的任务细节。
执行完整 implementer→spec→verif 对抗审查链（模式 A：首次执行）。
"""
)
```

phase-runner 内部自动完成：
- spawn implementer（Agent 工具）→ Shell spec-reviewer（claude -p）→ Shell verification（claude -p）
- implementer 被打回时重新 spawn（附失败报告）
- 所有报告写入 `./phase_report/`

phase-runner 返回结构化摘要后，主 Agent 进入步骤 2。

#### 步骤 2：主 Agent 防假 PASS 抽查

```bash
RANDOM_SCRIPT=$(ls scripts/test_phase${N}_*.py scripts/test_phase${N}_*.sh 2>/dev/null | shuf -n1)
ACTUAL_OUTPUT=$(python "${RANDOM_SCRIPT}" 2>&1 || bash "${RANDOM_SCRIPT}" 2>&1)
```

读取 `./phase_report/PHASE${N}_VERIFICATION_REPORT.md` 中该脚本的原始 stdout 比对：
- **一致** ✅ → Phase N 交付，写 `./phase_report/PHASE${N}_SUMMARY.md`
- **不一致** ❌ → 写 `./phase_report/PHASE${N}_SPOT_CHECK_FAIL.md` → 回到步骤 1（重试模式）

重试时 spawn phase-runner 使用模式 B（重试修复），phase-runner 内部走完整 implementer→spec→verif 链（不得跳过任何环节）。重试后换一个脚本再次抽查。

#### 判定逻辑

```
主 Agent 抽查          → 主 Agent 动作
─────────────────────────────────────────
一致 ✅                → Phase N 交付，进入下一 Phase
不一致 ❌ 第1-4次      → 写 SPOT_CHECK_FAIL.md → 重新 spawn phase-runner（完整修复链）
不一致 ❌ 第5次        → 停止，向人类报告全部 5 次驳回记录

phase-runner 内部：
spec-reviewer          → phase-runner 动作
─────────────────────────────────────────
✅ PASS                → 启动 verification
❌ FAIL                → 打回 implementer，verification 不启动

verification           → phase-runner 动作
─────────────────────────────────────────
✅ PASS                → 返回摘要给主 Agent
❌ FAIL                → 打回 implementer（附 verification 报告全文）
```

**不存在"部分通过""有条件交付""MINOR 可忽略"等中间状态。**
主 Agent 抽查连续 5 次驳回 → 停止，向人类报告阻塞点与全部 5 次驳回报告全文。

### 反模式警告

以下行为违反对抗结构，会导致子代理审查失效：

| 反模式 | 为什么危险 |
|--------|-----------|
| 同一个 Agent 先写代码再切换角色审查自己的代码 | confirmation bias——会为自己刚才的决策辩护 |
| 用 Agent 工具而非 Shell `claude -p` 挂载 spec-reviewer/verification | Agent 工具共享 harness，子代理能读到父进程的系统提示和项目配置，不是真正独立 |
| spec-reviewer 读了 implementer 的报告后再审查 | 报告中的自述会影响审查者的独立判断 |
| verification 只跑部分脚本（"其他的应该没问题"） | 脚本选择偏见——跳过最可能失败的脚本 |
| 主 Agent 手动修改 implementer 的代码后再交给 reviewer | reviewer 不知道改动来源，无法追溯 |
| implementer 在提交前自己跑了 scripts/ 并声称 PASS | implementer 可能同时误解了测试意图和代码逻辑，两边一起错 |
| 主 Agent 手动将 spec-reviewer 的 ❌FAIL 降级为"MINOR""有条件交付" | 主 Agent 不是裁判——它没读代码细节，没资格判断 FAIL 是否"可忽略"。这是对抗结构最致命的破坏 |
| 主 Agent 亲自 orchestrate implementer/spec/verif 而不是 spawn phase-runner | 主 Agent 上下文膨胀 → compact 丢失早期 Phase 的关键约束 → 后续 Phase 编码偏移 |
| phase-runner 自己执行抽查并声称 PASS | confirmation bias——phase-runner 对自己 orchestrate 的结果不具备独立抽查资格，抽查必须由主 Agent 独立完成 |

### 执行铁律

1. **implementer 不自证清白**：implementer 只写代码 + 自读 diff，不跑 scripts/，不宣判 PASS。提交状态是 SUBMITTED（不是 DONE 或 PASS）。
2. **审查串行执行**：先 spec-reviewer，通过后才到 verification。spec-reviewer ❌ → 直接打回 implementer，verification 不启动。不并行——避免"测试过了但蓝图不符"时产生降级放行的心理漏洞。
3. **scripts/ 不可变**：scripts/ 是先验知识，任何子代理不得修改。测试不过 → 改实现代码，不改脚本。
4. **verification 是唯一裁定者**：只有 verification 有权宣判 Phase 交付。spec-reviewer PASS 但 verification FAIL → 打回 implementer。
5. **跨 Phase 回归强制**：Phase 3 开始，verification 必须重跑所有前序 Phase 的 scripts/。任一回滚 → 打回。
6. **证据优先**：Phase 10 必须有 profiler trace + HCU/VRAM 监控证据。无证据 = 假推理 = 验收失败。
7. **本目录即是工程根**：所有生成代码直接写入本目录（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）。严禁创建子目录 `agent-infer/` 并在其中写入代码——scripts/ 的 PYTHONPATH 指向本目录，不指向任何子目录。所有报告写入 `./phase_report/`，文件名前缀 PHASE<N>_。
8. **API 服务生命周期完整**：openai_tp_server.py 的 SSE 流式响应必须设置 `Connection: close` header 并在成功和异常路径都设置 `self.close_connection = True`（SSE 无 Content-Length，keep-alive 导致 benchmark 永久 hang）。non-rank0 TP worker 必须在 `_tp_worker_loop` 注册 SIGTERM + SIGINT handler 并在 handler 内调用 `os._exit(0)`（主线程阻塞在 NCCL collective 时 Python 信号被延迟）。详见 AGENT_SKILL.md §2.4、inference_blueprint.json OpenAITPServer 组件。
9. **Phase 通过 phase-runner 执行**：主 Agent 不亲自 orchestrate implementer/spec/verif，必须 spawn phase-runner 子代理。主 Agent 只做调度 + 防假 PASS 抽查。抽查连续 5 次驳回 → 停止并向人类报告。

## 包内文件说明

| 路径 | 说明 |
|------|------|
| `inference_blueprint.json` | 架构知识图谱，所有 `ref_docs` 路径统一为 `notebooks-cn/` |
| `AGENT_SKILL.md` | 执行 SOP，含 §0.-2 路径兼容规则 |
| `prompts.md` | 编码 Skill 触发器索引（输入 `/phaseN` 即可加载对应 skill） |
| `.claude/skills/` | 编码 prompt skill 文件（12 个：phase-runner + implementer/spec-reviewer/verification 角色 + 6 phase coding + performance_alignment_by_tracing + torch-inference-mode） |
| `notebooks-cn/` | 知识文档（中文） |
| `ref_projects/` | 参考工程源码（nano-vllm, vllm, sglang） |
| `scripts/` | 固定测试合约（26 个，不可修改） |

## 编码 Skill 触发器（快速入口）

用户输入简短触发词，Agent 自动加载 `.claude/skills/phaseN-coding.md` 获得完整任务上下文：

| 触发词 | Phase 范围 |
|--------|-----------|
| `/phase1-4` | 数值基元 + TP通信 + TP线性层 + TP Embedding |
| `/phase5` | Attention + KV Cache（最高错误密度） |
| `/phase6` | MLP + Decoder Layer（最高错误密度） |
| `/phase7-8` | 权重加载 + 框架外壳 |
| `/phase9-10` | 引擎集成 + E2E 验收 |
| `/phase11` | 性能优化（审计-修复-再审计闭环） |

详见 `prompts.md`。

## Phase-Script 绑定（快速参考）

| Phase | scripts/ 门禁 |
|-------|--------------|
| Phase 1 数值基元 | `test_phase1_kernel_wrappers.py` + `.sh` |
| Phase 2 TP 通信 | `test_phase2_tp_communication.py` + `test_phase2_custom_ar_init.sh` |
| Phase 3 TP 线性层 | `test_phase3_tp_linear.py` + `test_phase3_tp_linear_tp4.py` |
| Phase 4 TP Embedding | `test_phase4_tp_embedding.py` + `test_phase4_tp_embedding_tp4.py` |
| Phase 5 Attention/KV | `test_phase5_attention_init.py` + `test_phase5_kv_cache_paged.py` + `test_phase5_flash_attn_prefill_decode.py` |
| Phase 6 MLP/Decoder | `test_phase6_mlp_forward.py` + `test_phase6_residual_chain.py` + `test_phase6_decode_forward_no_clone.py` + `test_phase6_layer_e2e_random_weights.py` |
| Phase 7 权重加载 | `test_phase7_qwen_tp_config.py` + `test_phase7_hf_key_mapping.py` + `test_phase7_weight_loading.sh` |
| Phase 8 框架外壳 | `test_phase8_sequence_scheduler.py` + `test_phase8_sampler_tp.py` |
| Phase 9 引擎集成 | `test_phase9_llm_engine_init.py` + `test_phase9_generate_single_gpu.sh` |
| Phase 10 E2E 验收 | `test_phase10_greedy_align.sh` + `test_phase10_benchmark.sh` + `test_phase10_no_compile_check.sh` + `test_phase10_vs_vllm_compare.sh` |
| Phase 11 性能优化 | `test_phase11_throughput.py` + `test_phase11_profiler.sh` |

## 测试运行

```bash
# 在本目录下执行，先设置环境
export PATH="${PYTHON_PATH}:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Python 合约
python scripts/test_phaseN_xxx.py

# Shell 脚本
bash scripts/test_phaseN_xxx.sh
```
