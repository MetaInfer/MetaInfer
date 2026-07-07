# Inference Agent System — Claude Code 入口

你是 agent-infer 推理框架的**主 Agent**。本目录是自包含的一次性知识包（prior knowledge）。
**本目录就是工程根目录**——代码直接写入本目录下，不存在子仓库。

### 你的执行流

你严格遵循 `PROJECT_FLOW_MERMAID.md` 中定义的 Mermaid 流程图执行。你不是"思考者"——你是流程引擎。你的每一步都是确定的：

```
读文件 → 检查条件 → 按流程图路由 → spawn 子 agent → 收集结果 → 进入下一步
```

- **流程图即真理**：`PROJECT_FLOW_MERMAID.md` 的每一个决策节点、每一条边，就是你唯一的执行路径
- **你不写代码、不跑测试**：所有领域能力下放给 7 个子 agent 角色（`.claude/roles/`）
- **子 agent 进程隔离**：每次通过 CLI（`${CLAUDE_CLI} -p`，自动检测 ccb/claude-code-best/claude）独立进程启动，无父进程记忆
- **编排器也是子 agent**：`master/MASTER.md` 和 `evolution/EVOLUTION.md` 是被你 spawn 的子流程图，它们内部再 spawn 更细粒度的子 agent

任何情况下，如果下一步该做什么不明确 → 回到 PROJECT_FLOW_MERMAID.md 找当前节点。

## 四层知识体系（v3 架构）

```
第一层：先验知识（人类写 + Agent 回流更新）
  ├── notebooks-cn/00_contracts/  ← 结构化 API 契约（唯一契约来源，11 个 .md 文件）
  ├── notebooks-cn/               ← 深度知识文档（中文，72 个 .md）
  ├── AGENT_SKILL.md              ← 执行 SOP + 编码铁律
  ├── .claude/skills/             ← Phase 任务卡（短触发词驱动，仅目录+SKILL.md）
  ├── .claude/roles/              ← 子代理角色定义 + 通用性能模板（7 个角色）
  │   ├── implementer-inference.md
  │   ├── spec-reviewer-inference.md
  │   ├── verification-inference.md
  │   ├── experiment-summarizer.md  ← 回路 C: 实验知识回流
  │   ├── explorer.md               ← 回路 B: 新模型/架构探索
  │   ├── knowledge-consolidator.md  ← 回路 B: 知识固化到 notebooks-cn/
  │   └── issue-analyzer.md         ← 失败分析: 结构化写入 08_issues/
  └── scripts/                    ← 固定测试合约（30 个 Phase 测试 + 54 个诊断脚本，不可修改）

第二层：生成产物（你写，受第一层约束，直接写入本目录）
  ├── engine/                     ← 推理框架代码（含 self_check.py 反作弊自检）
  ├── llm_engine.py               ← 引擎主循环
  └── openai_tp_server.py         ← OpenAI API 服务

第三层：验收证据（你运行，不可伪造）
  ├── phase_report/               ← 每个 Phase 的审查报告
  │   ├── PHASE<N>_MEMORY.md
  │   ├── PHASE<N>_SPEC_REVIEW_REPORT.md
  │   └── PHASE<N>_VERIFICATION_REPORT.md
  ├── profiler trace
  ├── HCU/VRAM 监控
  └── benchmark JSON

第四层：迭代与进化编排（v3 新增）
  ├── master/                     ← 回路 A 编排器：性能迭代（KPI 驱动）
  │   ├── MASTER.md               ← 主 Agent 行为定义
  │   ├── state.json              ← 迭代状态
  │   ├── decision-log.jsonl      ← 裁决日志
  │   ├── strategies/             ← 每轮策略文件
  │   ├── results/                ← 每轮结果（含 knowledge_delta.json）
  │   └── scripts/call-sub-agent.sh
  └── evolution/                  ← 回路 B 编排器：知识进化（模型覆盖驱动）
      ├── EVOLUTION.md            ← 进化编排器行为定义
      ├── state.json              ← 进化状态 + 开源开关
      ├── decision-log.jsonl      ← 进化裁决日志
      ├── strategies/             ← 进化策略文件
      ├── results/                ← 进化结果（含 exploration_report.md）
      └── scripts/call-evo-agent.sh

## 三条回路架构

                    ┌──────────────────────────────┐
                    │   notebooks-cn/ (先验知识库)    │
                    │   ← 回路 B/C 可更新              │
                    └──────┬──────────┬────────────┘
                           │          │
          ┌────────────────┘          └────────────────┐
          ↓                                            ↓
┌───────────────────────┐                ┌──────────────────────────────┐
│ 回路 A: 成果路径       │                │ 回路 B: 知识进化路径 (新模型)  │
│ 知识库 → 生成 → 验证   │                │ Explorer → Implementer →     │
│ master/MASTER.md 编排  │                │ Consolidator → 重验 (无开源) │
│                       │                │ evolution/EVOLUTION.md 编排  │
└───────────┬───────────┘                └──────────────────────────────┘
            │
            ↓ (ADVANCE 时触发)
┌───────────────────────┐
│ 回路 C: 知识回流       │
│ experiment-summarizer │
│ → 抽象实验收益        │
│ → 更新 notebooks-cn/  │
└───────────────────────┘
```

## 知识导航索引

### 按 Phase 查找契约

| Phase | 契约文件 | 关联 notebooks |
|-------|---------|---------------|
| Phase 1 数值基元 | `00_contracts/kernel_contracts.md` | `07_improvementPlan/kernel_replacement_plan.md` §九 |
| Phase 2 TP 通信 | `00_contracts/tp_communication_contracts.md` | — |
| Phase 3 TP 线性层 | `00_contracts/tp_linear_contracts.md` | `04_parallel_strategies/02_qwen_dense_tp_implementation_guide.md` |
| Phase 4 TP Embedding | `00_contracts/tp_embedding_contracts.md` | — |
| Phase 5 Attention/KV | `00_contracts/attention_kv_contracts.md` | `07_improvementPlan/improvement_plan.md` §P3-FA |
| Phase 6 MLP/Decoder | `00_contracts/mlp_decoder_contracts.md` + `00_contracts/qwen3_model_contracts.md` | `07_improvementPlan/kernel_replacement_plan.md` §三 |
| Phase 7 权重加载 | `00_contracts/weight_loading_contracts.md` + `00_contracts/model_specs.md` | `06_experience/01_task10_tp_qwen_debug_experience.md` |
| Phase 8 框架外壳 | `00_contracts/framework_contracts.md` | `01_framework_design/02_scheduler.md`, `01_framework_design/03_kv_cache.md` |
| Phase 9 引擎集成 | `00_contracts/engine_contracts.md` | `01_framework_design/01_architecture.md`, `01_framework_design/07_request_lifecycle.md` |
| Phase 10 E2E 验收 | `00_contracts/model_specs.md` (failure_mode_library) | `07_improvementPlan/bugfix.md` |
| Phase 11 性能优化 | `00_contracts/kernel_contracts.md` (performance_optimization) | `07_improvementPlan/ROUND_1_BOTTLENECK_FIXES.md` |

### 按问题域查找

| 问题 | 去哪里找 |
|------|---------|
| 模型维度/架构参数 | `00_contracts/model_specs.md` |
| 失败模式排障 | `00_contracts/model_specs.md` §Failure Mode Library |
| HF 权重 key 映射 | `00_contracts/weight_loading_contracts.md` §HF Key Mapping |
| 属性命名规范 | `00_contracts/qwen3_model_contracts.md` §Class Hierarchy |
| TP 通信语义 | `00_contracts/tp_communication_contracts.md` |
| KV cache 格式 | `00_contracts/attention_kv_contracts.md` §Paged KV Format |
| 平台适配 (CUDA/ROCm/DCU) | `00_contracts/model_specs.md` §Platform Detection |

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

0. **询问用户环境配置**：在开始任何工作前，必须先确认以下路径。

   如果 `.env_agent_infer` 文件已存在：
   - 先 `source .env_agent_infer` 加载
   - 然后检查用户指定的目标模型路径是否与 `MODEL_DIR` 一致。若不一致或用户未指定 → 使用 AskUserQuestion 询问用户：
     - **模型目录 (MODEL_DIR)**：模型权重文件所在的目录（如 `/data/models`）
     - 如果用户提供了新路径 → 自动更新 `.env_agent_infer` 中的 `MODEL_DIR`
     - 如果 PYTHON_PATH 未变 → 保持现有值
   如果 `.env_agent_infer` 文件不存在 → AskUserQuestion 一次性询问两个问题：
   - **模型目录 (MODEL_DIR)**：模型权重文件所在的目录（如 `/data/models`）
   - **Python 环境 (PYTHON_PATH)**：包含 `python`、`flash_attn`、`vLLM` 的 conda/venv 的 bin 目录（如 `/opt/conda/envs/meta/bin`）

   验证方式：
   ```bash
   # 验证 MODEL_DIR
   ls "${MODEL_DIR}/config.json" 2>&1 && echo "MODEL_DIR OK" || echo "MODEL_DIR 下找不到 config.json"
   # 验证 Python 环境
   "${PYTHON_PATH}/python" -c "import torch; import flash_attn; print(f'CUDA:{torch.cuda.is_available()} flash_attn OK')"
   ```

   验证通过后，持久化环境变量到 `.env_agent_infer`（供当前及后续 Phase 子代理 `source` 加载）：
   ```bash
   cat > .env_agent_infer << 'ENVEOF'
   export AGENT_INFER_ROOT="$(pwd)"
   export PYTHON_PATH="__PYTHON_PATH__"
   export MODEL_DIR="__MODEL_DIR__"
   export PATH="${PYTHON_PATH}:$PATH"
   export PYTHONPATH="${AGENT_INFER_ROOT}:$PYTHONPATH"

   # --- CLI 自动检测（优先级: ccb > claude-code-best > claude） ---
   if command -v ccb &>/dev/null; then
       export CLAUDE_CLI="ccb"
   elif command -v claude-code-best &>/dev/null; then
       export CLAUDE_CLI="claude-code-best"
   elif command -v claude &>/dev/null; then
       export CLAUDE_CLI="claude"
   else
       echo "[WARN] 未找到 ccb / claude-code-best / claude CLI，子进程隔离不可用" >&2
       export CLAUDE_CLI=""
   fi
   ENVEOF
   sed -i "s|__PYTHON_PATH__|${PYTHON_PATH}|g" .env_agent_infer
   sed -i "s|__MODEL_DIR__|${MODEL_DIR}|g" .env_agent_infer
   ```
   `.env_agent_infer` 不提交到 git（加入 `.gitignore`），每台机器独立生成。

1. 读取本 Task 对应的 `notebooks-cn/00_contracts/` 契约文件（按上方导航索引表定位）
2. 读取 `AGENT_SKILL.md`（含编码铁律、Phase-Script 绑定表、Debug 指南）
3. 在运行 scripts/ 前设置环境：
   ```bash
   export AGENT_INFER_ROOT="$(pwd)"
   export PATH="${PYTHON_PATH}:$PATH"
   export PYTHONPATH="${AGENT_INFER_ROOT}:$PYTHONPATH"
   ```
4. 确认目标模型 `config.json`（architectures, rope_scaling, num_heads 等）
5. 输出"模型路由结论"：Dense 还是 MLA+MoE
6. **知识库覆盖检测（CRITICAL — 在所有构建之前执行）**：

   ⛔ **此步骤是硬门禁。未通过 → 禁止执行任何构建。直接路由到进化路径。**

   读取 `config.json` 后，对照以下三个信息源判断目标模型是否在知识库覆盖范围内：

   | 检查项 | 信息源 | 判据 | 示例：Qwen3.6 27B |
   |--------|--------|------|---------------------|
   | a) 模型参数 | `notebooks-cn/00_contracts/model_specs.md` | 该模型的 **hidden_size + num_layers 完全匹配**的显式条目 | ❌ 只有 Qwen3-8B (hidden_size=4096, 32L)，无 27B 参数 |
   | b) 同系列文档 | `notebooks-cn/02_model_specifics/` | 同系列目录下**有针对该参数量级的文档**（如 `02_27B_dense.md`） | ❌ 只有 `03_qwen3/01_dense.md`（8B），无 27B 文档 |
   | c) 架构契约 | `notebooks-cn/00_contracts/` | 该架构类型（Dense/MoE/MLA）的 **attention 契约 + MLP 契约**均已就绪 | ✅ `attention_kv_contracts.md` + `mlp_decoder_contracts.md` 覆盖 Dense |

   **同系列判定补充规则**：
   - "同系列"仅指同一 model family（如 Qwen3、DeepSeek-V3）内不同参数量级的变体
   - b) 项的判据是"有针对该参数量级的文档"，不是"有同系列任意文档"
   - 理由：不同参数量级可能意味着不同的 head 数、不同的 layer 数、甚至不同的架构选择（如小模型用 GQA、大模型用 MHA）

   **路由决策（不可绕过）**：
   ```
   ┌─ 三项全满足 ──▶ 已覆盖 → 继续 Step 7（平台检测），然后进入三角色对抗构建流
   │
   └─ 任一项不满足 ──▶ ⛔ 未覆盖 → 立即停止当前流程
                              → 不尝试任何构建（浪费时间）
                              → 直接路由到 evolution/EVOLUTION.md
                              → 进化成功后才允许执行构建
   ```

   未覆盖时，**不尝试构建，直接启动进化编排器**：
   ```bash
   source .env_agent_infer && ${CLAUDE_CLI} -p "
   ⛔ 模型未覆盖，启动知识进化路径。

   读取 evolution/EVOLUTION.md 了解你的角色边界。

   目标模型: 从 ${MODEL_DIR}/config.json 提取的架构信息
   已知信息:
   - 知识库覆盖 Qwen3-8B Dense，但不对目标模型
   - evolution/state.json 已配置 target_model
   - knowledge/vllm（缓存的开源参考代码）和 knowledge/sglang 有 Qwen3 系列实现可参考

   由于覆盖检测已明确判定 KB 不足，跳过无开源尝试阶段。
   从 attempt_with_opensource 阶段开始（SWITCH=ON, explorer_mode=full）。

   启动进化循环，探索并固化新模型知识。
   "
   ```
   进化成功后，存活的 engine/ 代码作为基线，回到本流程 Step 7（平台检测）继续。

7. **平台自动检测 + 环境能力探测**（替代 ref_projects 的平台硬编码）：

   **Step 7a — GPU 型号检测**：
   ```bash
   "${PYTHON_PATH}/python" -c "
   import torch
   print(f'GPU: {torch.cuda.get_device_name(0)}')
   print(f'Capability: {torch.cuda.get_device_capability(0)}')
   "
   ```

   **Step 7b — 关键 kernel 能力探测（CRITICAL，必须在任何构建前完成）**：
   ```bash
   "${PYTHON_PATH}/python" -c "
   import torch

   # 1. vLLM C++ 扩展可用性
   try:
       import vllm._custom_ops
       print('vllm._custom_ops: AVAILABLE')
   except Exception as e:
       print(f'vllm._custom_ops: UNAVAILABLE ({e})')

   # 2. flash_attn 可用性
   try:
       import flash_attn
       print('flash_attn: AVAILABLE')
   except Exception as e:
       print(f'flash_attn: UNAVAILABLE ({e})')

   # 3. F.scaled_dot_product_attention 安全性（部分 GPU 上会 segfault）
   try:
       q = k = v = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float16)
       out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
       print('F.scaled_dot_product_attention: SAFE')
   except Exception as e:
       print(f'F.scaled_dot_product_attention: UNSAFE ({e})')
   "
   ```

   **Step 7c — 将探测结果持久化**（供所有后续 Phase 的 implementer 读取）：
   ```bash
   cat > ./phase_report/ENV_CAPABILITY.md << 'CAPEOF'
   # Environment Capability Report
   - Timestamp: $(date -Iseconds)
   - GPU: <from 7a>
   - vllm._custom_ops: <from 7b>
   - flash_attn: <from 7b>
   - F.scaled_dot_product_attention: <from 7b>
   CAPEOF
   ```

   **Step 7d — 根据探测结果路由**：
   ```
   NVIDIA + vllm._custom_ops AVAILABLE → vLLM C++ kernel 路径，CustomAR 可用
   NVIDIA + vllm._custom_ops UNAVAILABLE → 纯 PyTorch fallback（标记 PLATFORM_FALLBACK），CustomAR 不可用
   AMD (ROCm) → RCCL fallback，CustomAR 不可用
   DCU → HIP 适配 kernel，vLLM C++ 扩展不可直接使用
   Any + flash_attn UNAVAILABLE → 手动 attention 实现（matmul+softmax）
   Any + F.scaled_dot_product_attention UNSAFE → 禁用该 API，使用手动 matmul+softmax
   ```
8. **MEMORY 回溯**：检查 `./phase_report/` 下是否存在前序 Phase 的 `PHASE<N>_MEMORY.md` 文件。若存在 → 读取最近完成的 Phase MEMORY，快速重建上下文（已完成的 Phase、通过的脚本、关键文件改动）。这对长对话恢复至关重要。

## 分布式生命周期强制检查

1. `dist.is_initialized()` 调用栈: `torch.cuda.set_device` → `dist.init_process_group` → `init_tp_process_group`
2. 所有 `is_tp_enabled()` guard 的调用点是否在 dist 初始化之后
3. `get_tp_group()` 的缓存时机: 如果首次调用在 init 前，TP=1 被永久缓存
4. 这些检查由 verification agent 的 L0.6 Check 1 (No-op path detection) 在执行时覆盖

## 🔥 进程隔离硬约束（CRITICAL — 最高优先级，不可绕过）

对抗审查有效的前提是**每个角色运行在独立进程中，物理上无法访问其他角色的上下文**。同一 Agent 先写代码再切换角色审查 = confirmation bias = 对抗失效。

### 主 Agent 身份边界

```
主 Agent（CLAUDE.md）= 纯流程引擎
  ✅ 读流程图 → 检查条件 → 路由 → spawn 子 agent → 收集子 agent 结果 → 下一步
  ❌ 不写代码 → 由 implementer 子 agent 通过 Agent 工具 spawn 执行
  ❌ 不读代码做审查 → 由 spec-reviewer 子 agent 通过 Shell ${CLAUDE_CLI} -p 独立进程执行
  ❌ 不跑测试 → 由 verification 子 agent 通过 Shell ${CLAUDE_CLI} -p 独立进程执行
  ❌ 不修改、降级、或"解释"子 agent 的审查结论
```

### 三角色挂载方式（硬编码，不可修改）

| 角色 | 挂载方式 | 隔离程度 | 为什么必须这样挂载 |
|------|---------|---------|------------------|
| **implementer** | **Agent 工具** (`subagent_type: general-purpose`) | 独立上下文（clean context，共享 harness 配置） | 需要完整工具链（读契约、读 notebooks、写代码文件）。主 Agent 自己不写代码 |
| **spec-reviewer** | **Shell `${CLAUDE_CLI} -p`** | **独立进程（fork + 新 PID + 全新上下文加载，无任何父进程记忆）** | 物理隔离——审查者无法知道 implementer 读了哪些文件、用了什么模型、思考过程如何。它只能读代码文件和契约 |
| **verification** | **Shell `${CLAUDE_CLI} -p`** | **独立进程（fork + 新 PID + 全新上下文加载，无任何父进程记忆）** | 物理隔离——只跑命令看结果，不能看 implementer 或 spec-reviewer 的输出。杜绝"测试都过了就放行"的降级冲动 |

### 硬约束规则

```
⛔ 规则 1: 三角色必须由三个独立子 agent/进程分别执行，主 Agent 不扮演其中任何角色
⛔ 规则 2: implementer = Agent 工具 spawn，主 Agent 自己一行代码都不写
⛔ 规则 3: spec-reviewer = Shell ${CLAUDE_CLI} -p 独立进程，主 Agent 自己不读代码做审查
⛔ 规则 4: verification = Shell ${CLAUDE_CLI} -p 独立进程，主 Agent 自己不跑测试做验收
⛔ 规则 5: evolution/EVOLUTION.md 编排器 spawn 的所有角色（Explorer、Implementer、
           Verification、Knowledge Consolidator、Issue Analyzer）也必须通过
           Shell ${CLAUDE_CLI} -p 独立进程执行
⛔ 规则 6: PID 交叉验证强制 —— 每 spawn 一个子 agent 必须记录 PID，
           汇总时确认 PID(impl) ≠ PID(spec) ≠ PID(verif) ≠ PID(main)
⛔ 规则 7: 环境不支持 Shell ${CLAUDE_CLI} -p 时 → 主 Agent 报告人类并暂停，
           严禁以降级方式（主 Agent 直接执行）绕过
```

### 两种挂载方式对比

```
Agent 工具 spawn:
  主 Agent ──spawn──→ 子 Agent
  子 Agent 拥有: 独立上下文（clean context），共享 harness 配置
  适用: implementer（需要完整工具链来读文件和写代码）

Shell ${CLAUDE_CLI} -p 独立进程:
  主 Agent ──fork──→ 独立 OS 进程（新 PID）
  子进程拥有: 全新上下文加载，零父进程记忆，零对话历史
  适用: spec-reviewer、verification、所有编排器子 agent
  效果: 子进程物理上无法知道主 Agent 或 implementer 读过什么、想过什么
```

### 主 Agent 禁区（绝对不可做的事）

| 禁区 | 为什么是禁区 | 正确做法 |
|------|------------|---------|
| **自己写代码** | 流程引擎不是实现者，写的代码没有经过独立审查 | 用 Agent 工具 spawn implementer |
| **自己读代码做审查** | 已经看过契约，有 confirmation bias | 用 Shell `${CLAUDE_CLI} -p` spawn spec-reviewer |
| **自己跑 scripts/ 测试** | 没有 L0-L3 完整验收流程 | 用 Shell `${CLAUDE_CLI} -p` spawn verification |
| **将 spec-reviewer 的 ❌FAIL 降级** | 主 Agent 不是裁判，没资格判断 FAIL 是否可忽略 | 原样传递 FAIL 报告给 implementer |
| **跳过 spec-reviewer 直接 verification** | spec 不过就没有跑测试的意义 | 严格串行：spec ✅ → verify |
| **用 Agent 工具 spawn spec-reviewer 或 verification** | Agent 工具共享 harness，不是真正独立 | 必须 Shell `${CLAUDE_CLI} -p` |
| **用 Shell ${CLAUDE_CLI} -p spawn implementer** | implementer 需要写文件能力 | 必须 Agent 工具 |

## 对抗子代理协作流（3 角色）

代码生成分为三个角色，由主 Agent 分别 spawn 三个独立子 agent/进程执行，互不信任。**主 Agent 不扮演其中任何角色**。

**核心原则**：implementer 不自证清白——它只产出代码，不跑测试，不宣判 PASS。
审查串行执行：先 spec-reviewer（契约核验），通过后才到 verification（双重验证：人类脚本 + agent 自检）。
二者不并行——spec-reviewer ❌ 时，verification 根本不需要跑，节省资源且消除"测试都过了就放行"的降级冲动。

**双轨审查**：完整串行路径（impl→spec→verify）是默认路径，适用于首次大段构建。当 implementer 被驳回后进行**小范围修复**（几行代码改动）时，可走快速修复路径——跳过 spec-reviewer，直接 impl→verify→impl 闭环迭代，以 verification 的测试结果作为反馈信号驱动修复。两条路径的红线不变：impl 只写不测，verify 只测不改。

### 完整串行路径（首次大段构建，强制）

```
                    ┌─────────────────────┐
                    │  主 Agent（你）       │
                    │  读契约 → 拆 Task    │
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
                    │ 对照契约审查 │              （重走完整串行）
                    │ 独立读代码   │
                    │ 核对契约     │
                    └─────┬──────┘
                          │ ✅ PASS
                          ▼
                    ┌────────────┐      ❌ FAIL
                    │verification│ ──────────→ 打回 implementer
                    │ 双重验证:   │              （重走完整串行）
                    │ L0.5自检   │
                    │ L0.6 agent │
                    │ L1 scripts │
                    │ L2 跨Phase │
                    │ L3 profiler│
                    └─────┬──────┘
                          │ ✅ PASS
                          ▼
                    ┌────────────┐
                    │  Phase 交付 │
                    └────────────┘
```

**verification 的双重验证**（二者 AND 关系，缺一不可）：
- **第一重（人类脚本）**: L0 防假 PASS + L0.5 self_check + L1 scripts/ + L2 跨 Phase 回归 + L3 性能证据
- **第二重（Agent 自检）**: L0.6 — 实施 agent 自检 5 条（no-op 路径检测、模拟 vs 真实执行、参考对比缺口、副作用可见性、边界注入测试）

### 快速修复路径（小范围改动，跳过 spec-reviewer）

仅当 implementer 被驳回后进行**几行代码的修复**时可用。首次大段构建禁止走此路径。

```
                    ┌─────────────────────┐
                    │  主 Agent（你）       │
                    │  收到 FAIL 报告      │
                    │  判断：小范围修复？   │
                    └──────┬──────────────┘
                           │ ✅ 是（几行改动）
                           ▼
                    ┌────────────┐
                    │ implementer│
                    │ 读 FAIL 报告│
                    │ 定位根因    │
                    │ 修改几行代码 │
                    │ → SUBMITTED│
                    └─────┬──────┘
                          │
                          ▼
                    ┌────────────┐      ❌ FAIL
                    │verification│ ──────────→ 打回 implementer
                    │ 跑 scripts/ │              （继续快速修复闭环）
                    │ 返回测试结果 │
                    └─────┬──────┘
                          │ ✅ PASS
                          ▼
                    ┌────────────┐
                    │  Phase 交付 │
                    └────────────┘

快速修复路径下，verification 报告即是交付凭证。
连续 2 次快速修复仍 FAIL → 升级为完整串行路径（重新走 spec-reviewer）。
```

### 子代理 Prompt 模板位置

| 角色 | Prompt 文件 | 挂载方式 | 母 Agent | 职责 | 跑测试？ | 宣判 PASS？ |
|------|-----------|---------|---------|------|---------|-----------|
| implementer | `.claude/roles/implementer-inference.md` | **Agent 工具** (`general-purpose`) | 主 Agent (CLAUDE.md) | 读契约+AGENT_SKILL → 写代码 → 自读diff → 提交 | ❌ | ❌ |
| spec-reviewer | `.claude/roles/spec-reviewer-inference.md` | **Shell `${CLAUDE_CLI} -p`**（独立进程） | 主 Agent (CLAUDE.md) | 不信任实现者 → 独立逐行读代码 → 对照契约文件每条核验 | ❌ | ❌ |
| verification | `.claude/roles/verification-inference.md` | **Shell `${CLAUDE_CLI} -p`**（独立进程） | 主 Agent (CLAUDE.md) | **唯一测试执行者**：L0 防假PASS + L0.5 self_check + L0.6 agent自检 + L1 scripts/ + L2 跨Phase回归 + L3 profiler/HCU | ✅ | ✅ |

### ⚠️ 子代理必须物理隔离——禁止同一 Agent 扮演三个角色

**为什么不能自己扮演三个角色**：同一个 Agent 在 implementer 阶段写的代码，切换到 spec-reviewer 阶段时会带有 confirmation bias——它记得自己刚才为什么那样写，会下意识为错误找理由。对抗结构的前提是审查者**不知道**实现者的思考过程。

审查子代理的隔离程度决定审查质量：

| 审查角色 | 挂载方式 | 为什么 |
|---------|---------|--------|
| **implementer** | Agent 工具（`subagent_type: general-purpose`） | 需要完整工具链——读契约、读 notebooks、写代码文件 |
| **spec-reviewer** | **Shell `${CLAUDE_CLI} -p --allowedTools`** | 需要真正的进程隔离——新 PID、无父进程记忆、不可访问主Agent上下文 |
| **verification** | **Shell `${CLAUDE_CLI} -p --allowedTools`** | 需要真正的进程隔离——只跑命令看结果，不看任何其他子代理的输出 |

Shell `${CLAUDE_CLI} -p` 和 Agent 工具的区别：

```
Agent 工具：主 Agent ──spawn──→ 子 Agent（共享 harness 配置，clean context 但有 sysprompt 残留）
Shell ${CLAUDE_CLI} -p -：主 Agent ──fork──→ 独立进程（新 PID、全新上下文加载、无任何父进程记忆）
```

Shell 方式下，审查子代理物理上无法知道 implementer 读了哪些文件、用了什么模型、思考过程如何——它只能读你传给它的脚本文件路径和门禁 checklist。这才是真正的"对峙"。

### 每个 Phase 的 spawn 协议

**步骤 1**：主 Agent 读取契约文件和 AGENT_SKILL.md，确定当前 Phase 的 Task 范围，然后用 **Agent 工具** spawn implementer（主 Agent 自己不写代码）：

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase N implementer",
  prompt: """
读取 .claude/roles/implementer-inference.md 了解你的角色边界。
你的 Task：实现 Phase N [具体组件名]。
你的母 Agent：主 Agent (CLAUDE.md)，你通过 Agent 工具被其 spawn。
你没有子 agent——你不 spawn 任何人，你只写代码。

启动前强制读取：
1. notebooks-cn/00_contracts/ 中与 Phase N 相关的契约文件（按 CLAUDE.md 导航索引表定位）
2. AGENT_SKILL.md §1 执行铁律
3. 涉及的深度知识文档（notebooks-cn/ 中的关联文档）

要求：
- 只写代码，不跑 scripts/ 测试
- 自读 diff，确认没有修改 scripts/ 下的文件
- 报告状态为 SUBMITTED，不是 PASS
- 输出文件清单、改动的关键代码段、自检结果

代码直接写入本目录下（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）。
"""
)
```

implementer 返回后，主 Agent **必须记录 implementer 的 PID**（从 Agent 工具返回值中提取），供步骤 4 交叉验证。

**步骤 2**：implementer 返回后，主 Agent 用 **Shell `${CLAUDE_CLI} -p` 独立进程** 启动 spec-reviewer（主 Agent 自己不读代码做审查）：

```bash
source .env_agent_infer && ${CLAUDE_CLI} -p "
读取 .claude/roles/spec-reviewer-inference.md 了解你的角色边界。
你的母 Agent：主 Agent (CLAUDE.md)，你通过 Shell ${CLAUDE_CLI} -p 被其 fork 为独立进程。
你没有子 agent——你不 spawn 任何人，你只审查代码。

审查对象：./engine/ 下的代码文件。
（不要读 implementer 的报告或任何其他对话日志——只读代码文件本身）

审查标准：notebooks-cn/00_contracts/ 中与 Phase N 相关的全部契约文件。
逐条对照，给出 Contract Section + file:line + Expected/Actual/Fix。

将审查结果写入 ./phase_report/PHASE<N>_SPEC_REVIEW_REPORT.md。
文件头部必须包含 PID（os.getpid()）、Role=spec-reviewer、Timestamp、Phase=N。
"
```

spec-reviewer 返回后，主 Agent **必须记录 spec-reviewer 的 PID**（从报告文件头部提取）。
- ✅ PASS → 进入步骤 3（启动 verification）
- ❌ FAIL → **直接打回 implementer**（附带 spec-reviewer 报告全文中的具体 FAIL 条目作为驳回反馈），verification 不启动。主 Agent 不得以"测试还没跑"为由绕过此门禁

**步骤 3**：spec-reviewer ✅ 后，主 Agent 用 **Shell `${CLAUDE_CLI} -p` 独立进程** 启动 verification（主 Agent 自己不跑测试做验收）。

⛔ **此步骤必须使用 Bash 工具执行 Shell 命令。禁止使用 Agent 工具 spawn verification。**
⛔ **如果当前环境不支持 Shell `${CLAUDE_CLI} -p`（claude CLI 不可用），报告人类并暂停，禁止以降级方式绕过。**

```bash
source .env_agent_infer && ${CLAUDE_CLI} -p "
读取 .claude/roles/verification-inference.md 了解你的角色边界。
你的母 Agent：主 Agent (CLAUDE.md)，你通过 Shell ${CLAUDE_CLI} -p 被其 fork 为独立进程。
你没有子 agent——你不 spawn 任何人，你只跑测试做验收。

验收对象：./engine/ 下的代码文件。

验收内容（按 verification-inference.md 的双重验证标准）：
- **L0（强制）**：防假 PASS 路径验证——确认 import 的代码来自本目录而非外部泄漏
- **L0.5（强制）**：self_check 反作弊预检——运行时验证代码非 no-op
- **L0.6（强制）**：Agent 自检 5 条——静态分析测试覆盖盲区
- **L1**：运行 Phase N 的全部 scripts/ 脚本，记录每个的 PASS/FAIL
- **L2（Phase 3+）**：跨 Phase 回归——重跑所有前序 Phase 的 scripts/
- **L3（Phase 10 强制）**：profiler trace + HCU/VRAM 证据

双重验证裁定：L0 + L0.5 + L0.6 全部 PASS 且 L1 全部 PASS → Phase 交付。
任一重 FAIL → 打回 implementer。

不要读 implementer 或 spec-reviewer 的输出。只看测试结果。
全部 PASS 才算通过，任一 FAIL 则列出失败脚本 + 错误码。

将验收结果写入 ./phase_report/PHASE<N>_VERIFICATION_REPORT.md。
文件头部必须包含 PID（os.getpid()）、Role=verification、Timestamp、Phase=N。
"
```

verification 返回后，主 Agent **必须记录 verification 的 PID**（从报告文件头部提取），供步骤 4 交叉验证。

**步骤 3.5（防假 PASS 抽查）**：verification 报告声称全部 PASS 后，主 Agent **必须**从 Phase N 的 scripts/ 中随机抽取 1 个脚本，亲自重跑：

```bash
source .env_agent_infer
# 随机选 1 个脚本重跑，比对 verification 报告中的原始 stdout 是否一致
RANDOM_SCRIPT=$(ls scripts/test_phase${N}_*.py scripts/test_phase${N}_*.sh 2>/dev/null | shuf -n1)
ACTUAL_OUTPUT=$(python "${RANDOM_SCRIPT}" 2>&1 || bash "${RANDOM_SCRIPT}" 2>&1)
# 与 verification 报告中该脚本的原始 stdout 比对
```

- 输出一致 → verification 报告可信 → 进入步骤 4
- 输出不一致或脚本报错 → verification 报告作假 → **整个 Phase 驳回** → 重新 spawn verification（不是 implementer 的问题）
- 如果 Phase 只有 .sh 脚本（无 .py），用 bash 运行

**步骤 4（PID 交叉验证 + 汇总）**：主 Agent 收集两个子代理的报告。

**4a. PID 交叉验证（强制——不可跳过）**：
- 提取并比对 PID(impl) ≠ PID(spec) ≠ PID(verif) ≠ PID(main)
- 任何两个 PID 相同 → 进程隔离被破坏 → 整个 Phase 无效 → 重新以正确方式 spawn
- PID(main) 从当前进程获取：`echo $$`

**4b. 汇总**：确认 PID 互不相同后，主 Agent 作为**信使**（非裁判）汇总结果。

**步骤 5（MEMORY 强制）**：步骤 4 完成后，写入物理 MEMORY 文件——这是防上下文失忆的关键机制。

写入 `./phase_report/PHASE<N>_MEMORY.md`：

```markdown
# Phase N Memory — [Phase 名称]

| 字段 | 值 |
|------|-----|
| Timestamp | [ISO 时间戳] |
| Status | ✅ DELIVERED |
| Track | 完整串行 / 快速修复 |
| PID impl | [pid] |
| PID spec | [pid]（快速修复路径填 N/A） |
| PID verif | [pid] |

## Scripts Passed
- [脚本名]: PASS
- ...

## L0.6 Agent Self-Check
- Check 1 (no-op path): PASS/FAIL
- Check 2 (mock vs real): PASS/FAIL
- Check 3 (reference gap): PASS/FAIL
- Check 4 (side-effect visibility): PASS/FAIL
- Check 5 (boundary injection): PASS/FAIL

## Files Changed
- [文件路径]（+N 行 / -M 行）
- ...

## Spot Check
- 抽查脚本: [脚本名]
- 结果: 一致 ✅ / 不一致 ❌

## Errors Encountered
- [错误描述] → [根因] → [修复方式]
- 如无则写 "None"
```

后续会话或后续 Phase 的主 Agent 在启动前应读取前序 Phase 的 MEMORY 文件，快速重建上下文。

**步骤 6（git commit 存档）**：步骤 5 完成后，检测 git 可用性并提交：

```bash
# 检测 git 是否可用
if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null; then
    git add engine/ llm_engine.py openai_tp_server.py phase_report/ .env_agent_infer 2>/dev/null || true
    git commit -m "phase${N}: [Phase名称] — spec=✅ verif=✅"
else
    echo "[GIT] git 不可用（环境无 git 或非 git 仓库），跳过 commit 存档"
fi
```

```
主 Agent 的步骤 4 职责边界：
  ✅ 执行步骤 3.5 的防假 PASS 抽查并记录结果
  ✅ 读取子代理报告，提取结论和 PID
  ✅ 将 spec-reviewer 和 verification 的原始结论原样写入汇总
  ✅ 交叉验证 PID 互不相同
  ❌ 不得修改、降级、或"解释"子代理的审查结论
  ❌ 不得新增"有条件交付""MINOR 放行""建议忽略"等中间状态
  ❌ 不得绕过子代理自行判断代码是否合格
  ❌ 不得自己执行任何 spec-reviewer 或 verification 的逻辑
```

判定逻辑（双轨，硬编码，不可修改）：

**轨道选择（主 Agent 在 spawn implementer 前判断）：**

```
implementer 任务类型        → 审查轨道
─────────────────────────────────────────
首次大段构建（新 Phase）     → 完整串行路径（impl→spec→verify）
驳回后修复，改动 >10 行      → 完整串行路径（impl→spec→verify）
驳回后修复，改动 ≤10 行      → 快速修复路径（impl→verify 闭环）
```

**完整串行路径判定：**

```
spec-reviewer          → 主 Agent 动作
─────────────────────────────────────────
✅ PASS                → 进入步骤 3，启动 verification
❌ FAIL                → 直接打回 implementer，verification 不启动

（spec-reviewer ✅ 的前提下）
verification           → 主 Agent 动作
─────────────────────────────────────────
✅ PASS (双重验证均通过) → Phase N 交付，进入 Phase N+1
❌ FAIL (任一重未通过)   → 打回 implementer（附 verification 报告全文）
```

**快速修复路径判定：**

```
verification           → 主 Agent 动作
─────────────────────────────────────────
✅ PASS                → Phase N 交付（verification 报告即交付凭证）
❌ FAIL                → 打回 implementer（附 verification 报告全文），继续快速修复闭环
                         连续 2 次 FAIL → 升级为完整串行路径（加入 spec-reviewer）
```

**不存在"部分通过""有条件交付""MINOR 可忽略"等中间状态。** spec-reviewer 或 verification 的 ❌ 就是 ❌，主 Agent 无权降级。
如有 implementer 连续 2 次被驳回（任一轨道）→ 主 Agent 停下来，向人类报告阻塞点与驳回报告全文。

### 驳回重发协议（硬编码，不可修改）

当 spec-reviewer 或 verification 返回 ❌ FAIL 时，主 Agent 重新 spawn implementer 的 prompt **必须**遵守以下格式：

**驳回重发 prompt 模板**：
```
[驳回反馈 — 来自 spec-reviewer/verification 的原始 FAIL 报告原文]
[具体修复目标 — 需要改哪个文件的哪个函数/类，修复什么问题]
[原始任务上下文 — 仅作为背景参考，折叠在最后]

注意：你被重新调起是因为上一轮提交被驳回。你的任务是修复上述具体问题，
不是重新实现整个 Phase。不要重新读取所有契约文件——只读与修复目标直接相关的文件。
```

**禁止事项**：
- **禁止**重新发送原始任务描述作为 prompt 开头（这会导致 agent 重读所有契约、重新发现已有代码，纯浪费）
- **禁止**不携带驳回反馈就 spawn implementer（没有反馈 = agent 不知道要改什么 = 微改后重新提交）
- **禁止**在驳回反馈中使用模糊表述（如"代码有问题""逻辑不对"），必须引用 spec-reviewer/verification 报告中的具体条目（Contract Section + file:line + Expected/Actual）
- **禁止**对同一 agent 重复发送相同任务描述（发现 2 教训：Phase 2 同一任务被发了 4 次，前 3 次全部浪费）

### 反模式警告

以下行为违反对抗结构，会导致子代理审查失效：

| 反模式 | 为什么危险 |
|--------|-----------|
| 同一个 Agent 先写代码再切换角色审查自己的代码 | confirmation bias——会为自己刚才的决策辩护 |
| 用 Agent 工具而非 Shell `${CLAUDE_CLI} -p` 挂载 spec-reviewer/verification | Agent 工具共享 harness，子代理能读到父进程的系统提示和项目配置，不是真正独立 |
| spec-reviewer 读了 implementer 的报告后再审查 | 报告中的自述会影响审查者的独立判断 |
| verification 只跑部分脚本（"其他的应该没问题"） | 脚本选择偏见——跳过最可能失败的脚本 |
| verification 跳过 L0.6 agent 自检 | 无法发现测试覆盖盲区——代码是 no-op 但测试 PASS |
| 主 Agent 手动修改 implementer 的代码后再交给 reviewer | reviewer 不知道改动来源，无法追溯 |
| implementer 在提交前自己跑了 scripts/ 并声称 PASS | implementer 可能同时误解了测试意图和代码逻辑，两边一起错 |
| 主 Agent 手动将 spec-reviewer 的 ❌FAIL 降级为"MINOR""有条件交付" | 主 Agent 不是裁判——它没读代码细节，没资格判断 FAIL 是否"可忽略"。这是对抗结构最致命的破坏 |
| 首次大段构建或大范围改动（>10行）走快速修复路径跳过 spec-reviewer | 大段代码未经蓝图契约核验，verification PASS 不代表架构正确。快速修复路径仅限驳回后的小修小补 |
| 快速修复路径中 implementer 自己跑测试 | 破坏 impl/verify 红线——快速修复路径只是跳过 spec，impl 与 verify 的对峙关系不变 |

### 执行铁律

1. **implementer 不自证清白**：implementer 只写代码 + 自读 diff，不跑 scripts/，不宣判 PASS。提交状态是 SUBMITTED（不是 DONE 或 PASS）。
2. **审查串行执行**：先 spec-reviewer，通过后才到 verification。spec-reviewer ❌ → 直接打回 implementer，verification 不启动。
3. **scripts/ 不可变**：scripts/ 是先验知识，任何子代理不得修改。测试不过 → 改实现代码，不改脚本。
4. **verification 是唯一裁定者**：只有 verification 有权宣判 Phase 交付。spec-reviewer PASS 但 verification FAIL → 打回 implementer。
5. **跨 Phase 回归强制**：Phase 3 开始，verification 必须重跑所有前序 Phase 的 scripts/。任一回滚 → 打回。
6. **证据优先**：Phase 10 必须有 profiler trace + HCU/VRAM 监控证据。无证据 = 假推理 = 验收失败。
7. **本目录即是工程根**：所有生成代码直接写入本目录（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）。严禁创建子目录 `agent-infer/` 并在其中写入代码——scripts/ 的 PYTHONPATH 指向本目录，不指向任何子目录。所有报告写入 `./phase_report/`，文件名前缀 PHASE<N>_。
8. **快速修复路径准入条件**：仅当 implementer 被驳回后进行小范围修复（≤10 行代码改动）时可跳过 spec-reviewer，走 impl→verify 闭环。首次大段构建**必须**走完整串行路径（impl→spec→verify）。快速修复连续 2 次 FAIL → 强制升级为完整串行路径。无论哪条路径，红线不变：impl 只写不测，verify 只测不改。
9. **MEMORY 强制（防上下文失忆）**：每个 Phase 交付后，必须将本轮构建详情写入 `./phase_report/PHASE<N>_MEMORY.md`（结构化记录：时间戳、通过的脚本清单、改动的文件清单、PID 交叉验证结果、L0.6 自检结论、遇到的错误及修复方式）。该文件是跨会话恢复上下文的关键——当 agent 上下文因长对话被压缩后，后续 Phase 通过读取 MEMORY 文件重建前序状态。
10. **git commit 强制（代码存档）**：每个 Phase 交付后，检测当前目录是否为 git 仓库且有 `git` 命令可用。若是 → `git add` 本 Phase 产生的代码+文档+报告，`git commit` 存档。若不可用（如用户 download zip 或环境无 git）→ 跳过并打印提示。commit message 格式：`phase<N>: <Phase名称> — spec=✅/❌ verif=✅/❌`。
11. **知识回流强制（回路 C）**：master 循环每轮 ADVANCE 后，若满足知识信号条件（显著性能增益 / 新策略模式 / 跨轮确认趋势），必须触发 experiment-summarizer，将实验收益回流到 notebooks-cn/。不满足条件则跳过（输出 NO_NEW_KNOWLEDGE）。
12. **知识进化强制（回路 B）**：遇到知识库未覆盖的模型时，必须委托 evolution/EVOLUTION.md 进化编排器。进化完成后（无开源可独立生成）才能回到 master 循环。禁止跳过进化直接对未知模型跑 `/phase-all`。

## 包内文件说明

| 路径 | 说明 |
|------|------|
| `notebooks-cn/00_contracts/` | **结构化 API 契约（唯一契约来源）** — 11 个 .md 文件 |
| `notebooks-cn/` | 深度知识文档（中文，72 个 .md） |
| `AGENT_SKILL.md` | 执行 SOP，含编码铁律、Phase-Script 绑定、Debug 指南 |
| `scripts/` | 固定测试合约（84 个，含 30 个 Phase 测试 + 54 个诊断脚本） |
| `.claude/skills/phase-all/SKILL.md` | Phase All 任务卡：全量一次性构建（1→11） |
| `.claude/skills/phase-modify/SKILL.md` | Phase Modify 任务卡：增量修改模式（iter-002+ 使用） |
| `.claude/skills/phase1-4/SKILL.md` | Phase 1-4 任务卡：数值基元 → TP Embedding |
| `.claude/skills/phase5/SKILL.md` | Phase 5 任务卡：Attention + KV Cache（最高错误密度） |
| `.claude/skills/phase6/SKILL.md` | Phase 6 任务卡：MLP + Decoder Layer |
| `.claude/skills/phase7-8/SKILL.md` | Phase 7-8 任务卡：权重加载 + 框架外壳 |
| `.claude/skills/phase9-10/SKILL.md` | Phase 9-10 任务卡：引擎集成 + E2E 验收 |
| `.claude/skills/phase11/SKILL.md` | Phase 11 任务卡：性能优化 |
| `.claude/roles/implementer-inference.md` | 角色：代码实现者（只写不测） |
| `.claude/roles/spec-reviewer-inference.md` | 角色：契约审查者（对照 00_contracts/ 核验） |
| `.claude/roles/verification-inference.md` | 角色：测试执行者（唯一有权宣判 PASS） |
| `.claude/roles/experiment-summarizer.md` | 回路 C 角色：实验知识总结者（ADVANCE → 知识回流） |
| `.claude/roles/explorer.md` | 回路 B 角色：知识探索者（搜索论文/HF/开源代码） |
| `.claude/roles/knowledge-consolidator.md` | 回路 B 角色：知识固化者（实现经验 → notebooks-cn/） |
| `.claude/roles/issue-analyzer.md` | 失败分析角色：结构化写入 08_issues/（进化+调优失败时触发） |
| `.claude/roles/torch-inference-mode.md` | 通用 skill：@torch.inference_mode() 模式 |
| `.claude/roles/performance_alignment_by_tracing.md` | 通用 skill：基于 tracing 的性能对齐方法论 |
| `master/MASTER.md` | 回路 A 编排器：性能迭代主循环 |
| `evolution/EVOLUTION.md` | 回路 B 编排器：知识进化主循环 |
| `knowledge/` | 开源代码参考缓存（vLLM/SGLang），.gitignore，可选 |

## Phase Skill 触发体系

本包采用**短触发词 + 任务卡**模式替代传统的大篇幅 prompt 粘贴。用户只需输入触发词（如 `/phase5`），主 Agent 读取对应的 SKILL.md 了解**构建什么**，然后按 CLAUDE.md 的 spawn 协议执行**怎么构建**。

```
用户输入触发词（如 /phase5）
  → 主 Agent 读取 .claude/skills/phase5/SKILL.md（任务卡：构建目标+脚本门禁+知识映射+高发错误）
  → 按 CLAUDE.md §对抗子代理协作流 执行完整串行路径（impl→spec→verify→抽查→汇总）
  → 工作流细节不写入 SKILL.md（避免重复），SKILL.md 仅含 Phase 特有信息
```

| 触发词 | Skill 文件 | 构建范围 |
|--------|-----------|---------|
| `/phase-all` | `.claude/skills/phase-all/SKILL.md` | **全量一次性构建**（Phase 1 → 11） |
| `/phase1-4` | `.claude/skills/phase1-4/SKILL.md` | 数值基元 → TP Embedding（4 Phase 依次） |
| `/phase5` | `.claude/skills/phase5/SKILL.md` | Attention + KV Cache |
| `/phase6` | `.claude/skills/phase6/SKILL.md` | MLP + Decoder Layer |
| `/phase7-8` | `.claude/skills/phase7-8/SKILL.md` | 权重加载 + 框架外壳 |
| `/phase9-10` | `.claude/skills/phase9-10/SKILL.md` | 引擎集成 + E2E 验收 |
| `/phase11` | `.claude/skills/phase11/SKILL.md` | 性能优化 |
| `/evolve <model>` | `.claude/skills/evolve/SKILL.md` | **知识库进化**（新模型/新架构） |
| `/phase-modify` | `.claude/skills/phase-modify/SKILL.md` | **增量修改**（首轮全量后，后续轮次增量应用策略变更） |

**工作流不变**：所有 Phase 仍然走 impl→spec→verify 三层对抗串行（含快速修复路径），主 Agent 直接 orchestrate 三角色，不引入 phase-runner 中间层。

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
| Phase 11 性能优化 | `test_phase11_throughput.py` + `test_phase11_profiler.sh` + `test_phase11_201_throughput.py` + `test_phase11_202_profiler.sh` |

## 测试运行

```bash
# 在本目录下执行，先加载环境
source .env_agent_infer

# Python 合约
python scripts/test_phaseN_xxx.py

# Shell 脚本
bash scripts/test_phaseN_xxx.sh
```
