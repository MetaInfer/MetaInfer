# Skill: Phase 1-4 编码（数值基元 → TP Embedding）

## 触发词

`/inference:phase1-4` 或 `/phase1-4`

## 概述

依次完成 Phase 1 → Phase 2 → Phase 3 → Phase 4 的编码任务，严格按照 implementer → spec-reviewer → verification 对抗子代理协作流执行。

---

## 环境

- 模型权重: `${MODEL_DIR}`
- Conda: `${PYTHON_PATH}/python`
- GPU: 4×A800（CUDA_VISIBLE_DEVICES=0,1,2,3）

## 你的角色

读取本目录的 CLAUDE.md，理解你的角色、三层知识体系和对抗子代理协作流。

## 本次任务

依次完成 Phase 1 → Phase 2 → Phase 3 → Phase 4，每一 Phase 严格按照 CLAUDE.md 的 spawn 协议执行：

  Phase 1: 数值基元（7 个 vLLM kernel wrapper）
  Phase 2: TP 通信（all_reduce_sum + all_gather_last_dim + CustomAR init）
  Phase 3: TP 线性层（Column/Row/Merged/QKV Parallel Linear）
  Phase 4: TP Embedding（VocabParallelEmbedding + ParallelLMHead）

## 每 Phase 的执行步骤

### 步骤 1：implementer 子代理

使用 Agent 工具 spawn implementer（subagent_type: general-purpose），读取 .claude/skills/implementer-inference.md。
implementer 只写代码，不跑测试。完成后写出 ./phase_report/PHASE<N>_IMPLEMENTER_REPORT.md（含 PID、Role、Timestamp、Phase=N，status=SUBMITTED）。

### 步骤 2：spec-reviewer（Shell claude -p 独立审查）

```bash
claude -p --allowedTools "Read(*),Write(*),Bash(*)" "读取 ${AGENT_INFER_ROOT}/.claude/skills/spec-reviewer-inference.md。审查 ./engine/ 下的代码。对照 inference_blueprint.json 中 Phase N 的契约逐条核验。将 SPEC_REVIEW_REPORT.md 写入 ./phase_report/（文件名前缀 PHASE<N>_）。文件头含 PID（os.getpid()）、Role=spec-reviewer、Timestamp。"
```

如果 spec ❌ → 打回 implementer 重写，verification 不启动。如果 ✅ → 进入步骤 3。

### 步骤 3：verification（Shell claude -p 独立验收，仅 spec ✅ 后执行）

```bash
claude -p --allowedTools "Read(*),Write(*),Bash(*)" "读取 ${AGENT_INFER_ROOT}/.claude/skills/verification-inference.md。验收 Phase N：运行 scripts/ 下 Phase N 对应的全部测试脚本。Phase 3+ 必须额外做跨 Phase 回归（重跑前序所有 Phase 的 scripts/）。将 VERIFICATION_REPORT.md 写入 ./phase_report/（文件名前缀 PHASE<N>_）。文件头含 PID（os.getpid()）、Role=verification、Timestamp。"
```

### 步骤 4：汇总

步骤 3.5: 主 Agent 抽查（verification 返回后）：从 Phase N 的 scripts/ 中随机抽 1 个重跑，比对 verification 报告的原始 stdout。一致 → 进入步骤 4。不一致 → 整个验收作废，重新 spawn verification。
步骤 4: 主 Agent 汇总——读取三个报告和抽查结果，验证 PID 互不相同，原样汇总入 ./phase_report/PHASE{N}_SUMMARY.md。禁止降级/修改子代理结论。

代码直接写入本目录下（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）。

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

- implementer 不跑测试、不判 PASS（状态为 SUBMITTED）
- spec-reviewer 和 verification 通过独立的 Shell claude -p 进程执行（新的 PID）
- 审查串行：先 spec-reviewer，通过后才到 verification。spec ❌ 时 verification 不启动
- 主 Agent 是信使非裁判——禁止降级/修改子代理结论。禁止"有条件交付"
- 三个子代理的 PID 必须互不相同
- scripts/ 不可修改。测试不过 → 改实现代码，不改脚本
- Phase 3 开始，verification 必须做跨 Phase 回归（重跑前序 Phase 的全部 scripts/）

## 防长上下文遗忘机制

每完成一个 Phase 后，重新打开 AGENT_SKILL.md §2.0.1 确认下一 Phase 的知识链路。
每完成一个 Phase 后，输出一行进度：`[PROGRESS] Phase N 完成，spec=✅/❌，verif=✅/❌`。
