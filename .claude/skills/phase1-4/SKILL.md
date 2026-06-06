# Skill: Phase 1-4 编码（数值基元 → TP Embedding）

## 触发词

`/inference:phase1-4` 或 `/phase1-4`

## 概述

依次完成 Phase 1 → Phase 2 → Phase 3 → Phase 4 的编码任务。每个 Phase 通过 spawn phase-runner 子代理完成内部对抗审查，主 Agent 只做调度和防假 PASS 抽查。

---

## 你的角色

你是**主 Agent**——只做高层调度和抽查，不亲自 orchestrate 三角色。每个 Phase 通过 spawn phase-runner 子代理执行，你只看到结构化摘要，保持上下文轻量。

## 本次任务

依次完成 Phase 1 → Phase 2 → Phase 3 → Phase 4，每一 Phase 严格按照 CLAUDE.md 的 spawn 协议执行：

  Phase 1: 数值基元（7 个 vLLM kernel wrapper）
  Phase 2: TP 通信（all_reduce_sum + all_gather_last_dim + CustomAR init）
  Phase 3: TP 线性层（Column/Row/Merged/QKV Parallel Linear）
  Phase 4: TP Embedding（VocabParallelEmbedding + ParallelLMHead）

## 执行方式

本 Phase 使用 **phase-runner 子代理** 完成编码和对抗审查。你（主 Agent）只做高层调度和防假 PASS 抽查——不再亲自 orchestrate implementer/spec/verif，避免上下文膨胀导致 compact 后约束丢失。

### 步骤 0：环境配置（仅 Phase 1 启动前执行一次）

在开始任何编码工作前，必须先向用户确认两个关键路径。如果用户已经提供过且 `.env_agent_infer` 文件存在，则跳过此步骤，直接 `source .env_agent_infer`。

**0.1 询问用户：**

```
使用 AskUserQuestion 工具一次性询问用户两个问题：

问题1: Python 环境
说明：请提供 conda 环境名（如 "meta"）或 python bin 目录的完整路径（如 "/opt/conda/envs/meta/bin"）。
      如果是 conda 环境名，Agent 会自动解析为 bin 目录路径。

问题2: 模型目录
说明：模型权重文件所在的目录（如 "/data/models"）。该目录下应包含 config.json。
```

**0.2 解析 PYTHON_PATH：**

如果用户提供的是 conda 环境名（不含 `/` 的字符串），自动解析为 bin 目录：

```bash
# 方式1：通过 conda info --envs 解析
CONDA_ENV_NAME="<用户提供的环境名>"
CONDA_ENV_PATH=$(conda info --envs 2>/dev/null | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')
if [ -n "${CONDA_ENV_PATH}" ]; then
  PYTHON_PATH="${CONDA_ENV_PATH}/bin"
else
  # 方式2：通过 which conda 推断 base 路径再拼接
  CONDA_BASE=$(dirname $(dirname $(which conda 2>/dev/null)))
  PYTHON_PATH="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin"
fi
```

如果用户直接提供了完整路径（含 `/`），则直接使用。

**0.3 验证路径：**

```bash
# 验证 MODEL_DIR
ls "${MODEL_DIR}/config.json" 2>&1 && echo "MODEL_DIR OK" || echo "MODEL_DIR 下找不到 config.json"

# 验证 Python 环境
"${PYTHON_PATH}/python" -c "import torch; import flash_attn; print(f'CUDA:{torch.cuda.is_available()} flash_attn OK')"
```

两项验证都通过后才继续，否则向用户报告失败原因并重新询问。

**0.4 持久化环境变量：**

验证通过后，写入 `.env_agent_infer` 文件供所有后续 Phase 使用：

```bash
cat > .env_agent_infer << ENVEOF
export AGENT_INFER_ROOT="\$(pwd)"
export PYTHON_PATH="${PYTHON_PATH}"
export MODEL_DIR="${MODEL_DIR}"
export PATH="\${PYTHON_PATH}:\$PATH"
export PYTHONPATH="\${AGENT_INFER_ROOT}:\$PYTHONPATH"
ENVEOF
```

**0.5 设置当前 shell 环境 + 模型路由：**

```bash
export AGENT_INFER_ROOT="$(pwd)"
export PATH="${PYTHON_PATH}:$PATH"
export PYTHONPATH="${AGENT_INFER_ROOT}:$PYTHONPATH"

# 读取模型 config.json 输出路由结论
python -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
print('architectures:', cfg.get('architectures', 'UNKNOWN'))
print('num_hidden_layers:', cfg.get('num_hidden_layers', '?'))
print('num_attention_heads:', cfg.get('num_attention_heads', '?'))
print('num_key_value_heads:', cfg.get('num_key_value_heads', '?'))
"
```

根据 architectures 输出"模型路由结论"：Dense 还是 MLA+MoE。

然后才开始执行 Phase 1。

### 步骤 1：spawn phase-runner（首次）

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase N runner",
  prompt: """
Phase N: [Phase名称]。
读取 .claude/roles/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phase1-4/SKILL.md 了解本 Phase 的任务细节。
执行完整 implementer→spec→verif 对抗审查链（模式 A：首次执行）。
"""
)
```

phase-runner 返回结构化摘要后，进入步骤 2。

### 步骤 2：主 Agent 防假 PASS 抽查

从当前 Phase 的 scripts/ 中随机抽取 1 个脚本，亲自重跑：

```bash
# 随机选 1 个脚本
RANDOM_SCRIPT=$(ls scripts/test_phase${N}_*.py scripts/test_phase${N}_*.sh 2>/dev/null | shuf -n1)
# 运行
ACTUAL_OUTPUT=$(python "${RANDOM_SCRIPT}" 2>&1 || bash "${RANDOM_SCRIPT}" 2>&1)
```

读取 `./phase_report/PHASE${N}_VERIFICATION_REPORT.md`，找到该脚本的原始 stdout，与 ACTUAL_OUTPUT 比对：
- **一致** ✅ → 该 Phase 交付，进入步骤 3
- **不一致** ❌ → 写 `./phase_report/PHASE${N}_SPOT_CHECK_FAIL.md`（失败脚本、期望 vs 实际），回到步骤 1 但用重试模式：

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase N runner (RETRY)",
  prompt: """
Phase N RETRY。
读取 ./phase_report/PHASE${N}_SPOT_CHECK_FAIL.md 了解失败原因。
读取 .claude/roles/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phase1-4/SKILL.md 了解任务细节。
执行完整 implementer→spec→verif 修复链（模式 B：重试修复，不得跳过任何环节）。
"""
)
```

重试后再次抽查（换一个脚本）。连续 5 次驳回 → 停止，向人类报告全部 5 次驳回记录。

### 步骤 3：写 Phase 汇总

抽查通过后，写 `./phase_report/PHASE${N}_SUMMARY.md`，含：
- PID 交叉验证（implementer/spec-reviewer/verification 的 PID 互不相同）
- 抽查脚本和结果
- 原样转述子代理结论（禁止降级/修改）

然后 `[PROGRESS] Phase N 完成`，进入下一 Phase。

## Phase-Script 绑定

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 1 | test_phase1_kernel_wrappers.py + test_phase1_kernel_wrappers.sh |
| Phase 2 | test_phase2_tp_communication.py + test_phase2_custom_ar_init.sh |
| Phase 3 | test_phase3_tp_linear.py + test_phase3_tp_linear_tp4.py |
| Phase 4 | test_phase4_tp_embedding.py + test_phase4_tp_embedding_tp4.py |

## Phase 知识映射（AGENT_SKILL.md §2.0.1）

| Phase | 必读 JSON 路径 | 必读 ref_docs | 必查 ref_code |
|-------|---------------|-------------|-------------|
| **Phase 1** | qwen3_kernel_contracts（7 kernel 签名）→ global_primitives_constraints.rmsnorm_precision_law | kernel_replacement_plan.md §九（完整 kernel 调用契约表 + Snippet A-F） | vllm/_custom_ops.py:420-423, vllm/_custom_ops.py:400-410, vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda |
| **Phase 2** | tp_distributed_runtime（init 顺序）→ collectives.all_reduce_sum（custom_op 注册+fake）→ collectives.all_gather_last_dim → collectives.custom_ar_all_reduce（两套 IPC buffer+init_state_machine） | — | vllm/_custom_ops.py:640-680 |
| **Phase 3** | tp_linear_layers（4 种 Linear 伪代码）→ qwen3_8b_model_dims（**_verified_config: gate_up=[6144,4096] NOT [6400,4096]_**） | qwen_dense_tp_implementation_guide.md, task10_tp_qwen_debug_experience.md | — |
| **Phase 4** | tp_embedding_and_lm_head（VocabParallel mask + ParallelLMHead gather） | — | — |

## 关键约束

- 主 Agent 只做调度 + 抽查，不亲自 orchestrate 三角色
- phase-runner 内部 implementer/spec-reviewer/verification 三角色物理隔离（Shell claude -p）
- 审查串行：spec ✅ 才到 verif。spec ❌ 时 verif 不启动
- 主 Agent 抽查是最终裁定——不一致就驳回，连续 5 次才停止
- 主 Agent 禁止降级/修改子代理结论。禁止"有条件交付"
- scripts/ 不可修改。测试不过 → 改实现代码，不改脚本
- Phase 3 开始，verification 必须做跨 Phase 回归（重跑前序 Phase 的全部 scripts/）
- 代码直接写入本目录下（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）
