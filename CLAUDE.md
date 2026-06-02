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

## 对抗子代理协作流（Superpowers 风格）

代码生成分为三个角色，独立子代理执行，互不信任。

**核心原则**：implementer 不自证清白——它只产出代码，不跑测试，不宣判 PASS。
审查串行执行：先 spec-reviewer（蓝图契约核验），通过后才到 verification（测试+证据）。
二者不并行——spec-reviewer ❌ 时，verification 根本不需要跑，节省资源且消除"测试都过了就放行"的降级冲动。

```
                    ┌─────────────────────┐
                    │  主 Agent（你）       │
                    │  读蓝图 → 拆 Task    │
                    │  派子代理 → 收集结果  │
                    └──────┬──────────────┘
                           │
                           ▼
                    ┌────────────┐
                    │ implementer│
                    │ 写代码      │
                    │ 自读diff    │
                    │ (不跑测试)   │
                    │ → SUBMITTED│
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐      ❌ FAIL
                    │spec-reviewer│ ──────────→ 打回 implementer
                    │ 对照蓝图审查 │
                    │ 独立读代码   │
                    │ 核对契约     │
                    └─────┬──────┘
                          │ ✅ PASS
                          ▼
                    ┌────────────┐      ❌ FAIL
                    │verification│ ──────────→ 打回 implementer
                    │ L1:scripts/ │
                    │ L2:跨Phase  │
                    │ L3:profiler │
                    │   +HCU证据  │
                    └─────┬──────┘
                          │ ✅ PASS
                          ▼
                    ┌────────────┐
                    │  Phase 交付 │
                    └────────────┘
```

### 子代理 Prompt 模板位置

| 角色 | Prompt 文件 | 职责 | 跑测试？ | 宣判 PASS？ |
|------|-----------|------|---------|-----------|
| implementer | `.claude/skills/implementer-inference.md` | 读蓝图+AGENT_SKILL → 写代码 → 自读diff → 提交 | ❌ | ❌ |
| spec-reviewer | `.claude/skills/spec-reviewer-inference.md` | 不信任实现者 → 独立逐行读代码 → 对照蓝图每条契约核验 | ❌ | ❌ |
| verification | `.claude/skills/verification-inference.md` | **唯一测试执行者**：L1 scripts/ + L2 跨Phase回归 + L3 profiler/HCU | ✅ | ✅ |

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

### 每个 Phase 的 spawn 协议

**步骤 1**：主 Agent 读取蓝图和 AGENT_SKILL.md，确定当前 Phase 的 Task 范围，然后用 **Agent 工具** spawn implementer：

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase N implementer",
  prompt: """
读取 .claude/skills/implementer-inference.md 了解你的角色边界。
你的 Task：实现 Phase N [具体组件名]。

启动前强制读取：
1. inference_blueprint.json 中与 Phase N 相关的契约节点（按 AGENT_SKILL.md §2.0.1 三步知识链路）
2. AGENT_SKILL.md §1 执行铁律
3. 涉及的 ref_docs 和 ref_code

要求：
- 只写代码，不跑 scripts/ 测试
- 自读 diff，确认没有修改 scripts/ 下的文件
- 报告状态为 SUBMITTED，不是 PASS
- 输出文件清单、改动的关键代码段、自检结果

代码直接写入本目录下（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）。
"""
)
```

**步骤 2**：implementer 返回后，主 Agent 先启动 spec-reviewer（Shell `claude -p`）：

```bash
claude -p "
读取 .claude/skills/spec-reviewer-inference.md 了解你的角色边界。

审查对象：./engine/ 下的代码文件。
（不要读 implementer 的报告或任何其他对话日志——只读代码文件本身）

审查标准：inference_blueprint.json 中与 Phase N 相关的全部契约节点。
逐条对照，给出 JSON Path + file:line + Expected/Actual/Fix。

将审查结果写入 ./phase_report/PHASE<N>_SPEC_REVIEW_REPORT.md。
文件头部必须包含 PID（os.getpid()）、Role=spec-reviewer、Timestamp、Phase=N。
"
```

spec-reviewer 返回后：
- ✅ PASS → 进入步骤 3（启动 verification）
- ❌ FAIL → **直接打回 implementer**，verification 不启动。主 Agent 不得以"测试还没跑"为由绕过此门禁

**步骤 3**：spec-reviewer ✅ 后，主 Agent 启动 verification（Shell `claude -p`）：

```bash
claude -p "
读取 .claude/skills/verification-inference.md 了解你的角色边界。

验收对象：./engine/ 下的代码文件。

验收内容（按 verification-inference.md 的 L0/L1/L2/L3 标准）：
- **L0（强制）**：防假 PASS 路径验证——确认 import 的代码来自本目录而非外部泄漏
- L1：运行 Phase N 的全部 scripts/ 脚本，记录每个的 PASS/FAIL
- L2（Phase 3+）：跨 Phase 回归——重跑所有前序 Phase 的 scripts/
- L3（Phase 10 强制）：profiler trace + HCU/VRAM 证据

不要读 implementer 或 spec-reviewer 的输出。只看测试结果。
全部 PASS 才算通过，任一 FAIL 则列出失败脚本 + 错误码。

将验收结果写入 ./phase_report/PHASE<N>_VERIFICATION_REPORT.md。
文件头部必须包含 PID（os.getpid()）、Role=verification、Timestamp、Phase=N。
"
```

verification 返回后，主 Agent 须完成**两步验证**才能进入步骤 4：

**步骤 3.5（防假 PASS 抽查）**：verification 报告声称全部 PASS 后，主 Agent **必须**从 Phase N 的 scripts/ 中随机抽取 1 个脚本，亲自重跑：

```bash
# 随机选 1 个脚本重跑，比对 verification 报告中的原始 stdout 是否一致
RANDOM_SCRIPT=$(ls scripts/test_phase${N}_*.py scripts/test_phase${N}_*.sh 2>/dev/null | shuf -n1)
ACTUAL_OUTPUT=$(python "${RANDOM_SCRIPT}" 2>&1 || bash "${RANDOM_SCRIPT}" 2>&1)
# 与 verification 报告中该脚本的原始 stdout 比对
```

- 输出一致 → verification 报告可信 → 进入步骤 4
- 输出不一致或脚本报错 → verification 报告作假 → **整个 Phase 驳回** → 重新 spawn verification（不是 implementer 的问题）
- 如果 Phase 只有 .sh 脚本（无 .py），用 bash 运行

**步骤 4**：抽查通过后，主 Agent 收集两个子代理的报告，作为**信使**（非裁判）汇总结果：

```
主 Agent 的步骤 4 职责边界：
  ✅ 执行步骤 3.5 的防假 PASS 抽查并记录结果
  ✅ 读取子代理报告，提取结论和 PID
  ✅ 将 spec-reviewer 和 verification 的原始结论原样写入汇总
  ✅ 交叉验证 PID 互不相同
  ❌ 不得修改、降级、或"解释"子代理的审查结论
  ❌ 不得新增"有条件交付""MINOR 放行""建议忽略"等中间状态
  ❌ 不得绕过子代理自行判断代码是否合格
```

判定逻辑（串行，硬编码，不可修改）：

```
spec-reviewer          → 主 Agent 动作
─────────────────────────────────────────
✅ PASS                → 进入步骤 3，启动 verification
❌ FAIL                → 直接打回 implementer，verification 不启动

（spec-reviewer ✅ 的前提下）
verification           → 主 Agent 动作
─────────────────────────────────────────
✅ PASS                → Phase N 交付，进入 Phase N+1
❌ FAIL                → 打回 implementer（附 verification 报告全文）
```

**不存在"部分通过""有条件交付""MINOR 可忽略"等中间状态。** spec-reviewer 或 verification 的 ❌ 就是 ❌，主 Agent 无权降级。
如有 implementer 连续 2 次被驳回 → 主 Agent 停下来，向人类报告阻塞点与驳回报告全文。

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

### 执行铁律

1. **implementer 不自证清白**：implementer 只写代码 + 自读 diff，不跑 scripts/，不宣判 PASS。提交状态是 SUBMITTED（不是 DONE 或 PASS）。
2. **审查串行执行**：先 spec-reviewer，通过后才到 verification。spec-reviewer ❌ → 直接打回 implementer，verification 不启动。不并行——避免"测试过了但蓝图不符"时产生降级放行的心理漏洞。
3. **scripts/ 不可变**：scripts/ 是先验知识，任何子代理不得修改。测试不过 → 改实现代码，不改脚本。
4. **verification 是唯一裁定者**：只有 verification 有权宣判 Phase 交付。spec-reviewer PASS 但 verification FAIL → 打回 implementer。
5. **跨 Phase 回归强制**：Phase 3 开始，verification 必须重跑所有前序 Phase 的 scripts/。任一回滚 → 打回。
6. **证据优先**：Phase 10 必须有 profiler trace + HCU/VRAM 监控证据。无证据 = 假推理 = 验收失败。
7. **本目录即是工程根**：所有生成代码直接写入本目录（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）。严禁创建子目录 `agent-infer/` 并在其中写入代码——scripts/ 的 PYTHONPATH 指向本目录，不指向任何子目录。所有报告写入 `./phase_report/`，文件名前缀 PHASE<N>_。

## 包内文件说明

| 路径 | 说明 |
|------|------|
| `inference_blueprint.json` | 架构知识图谱，所有 `ref_docs` 路径统一为 `notebooks-cn/` |
| `AGENT_SKILL.md` | 执行 SOP，含 §0.-2 路径兼容规则 |
| `notebooks-cn/` | 知识文档（中文） |
| `ref_projects/` | 参考工程源码（nano-vllm, vllm, sglang） |
| `scripts/` | 固定测试合约（26 个，不可修改） |

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
