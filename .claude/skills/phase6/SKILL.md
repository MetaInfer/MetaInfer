# Skill: Phase 6 编码（MLP + Decoder Layer）

## 触发词

`/inference:phase6` 或 `/phase6`

## 概述

构建 Phase 6: MLP + Decoder Layer —— 整个流水线中**错误密度与 Phase 5 并列最高的阶段**。通过 spawn phase-runner 子代理完成内部对抗审查，主 Agent 只做调度和抽查。

---

## 环境

- **必须先执行** `source .env_agent_infer` 加载 MODEL_DIR 和 PYTHON_PATH
- 模型权重: `${MODEL_DIR}`
- Conda: `${PYTHON_PATH}/python`
- GPU: 4×A800
- Phase 1-5 的代码已存在于 ./engine/ 下，不要重复生成。
- **关键**：`engine/models/qwen.py` 已存在——Phase 5 创建了完整 QwenAttentionTP + QwenDecoderLayerTP + RMSNorm，以及 QwenMLPTP stub（仅 __init__，forward 是空壳）。Phase 6 的职责是修改 qwen.py，**不是新建文件**：
  - **补全 QwenMLPTP.forward**：stub → 完整 gate_up→silu_and_mul→down 链
  - **审查 QwenDecoderLayerTP 的 prefill/decode 路径**：Phase 5 已实现，Phase 6 需确保 residual chain 与蓝图完全一致
  - **禁止修改 QwenAttentionTP 和 RMSNorm**：Phase 5 已完成并通过验收

## 你的角色

你是**主 Agent**——只做高层调度和抽查，不亲自 orchestrate 三角色。通过 spawn phase-runner 子代理执行 Phase 6，你只看到结构化摘要，保持上下文轻量。

## 本次任务

构建 Phase 6: MLP + Decoder Layer

**在现有 `engine/models/qwen.py` 上修改，不是新建文件。**

必须补全/实现:
- QwenMLPTP.forward: gate_up_proj(MergedColumnParallelLinear) → silu_and_mul → down_proj(RowParallelLinear)（Phase 5 只有 stub，**替换为完整实现**）
- QwenDecoderLayerTP.forward(prefill): 确认 input_layernorm → attention.forward → post_attention_layernorm → mlp 链正确
- QwenDecoderLayerTP.forward_decode(decode): 确认 fused_add_rms_norm(input,residual,self.input_layernorm.weight) → attention.forward_decode → fused_add_rms_norm(attn_out,residual,self.post_attention_layernorm.weight) → mlp 链正确
- Residual chain: 首层 res=None → clone+rms_norm；后续层 fused_add_rms_norm（res+=hs; hs=rms_norm(res)）

## ⚠️ 最关键约束：FM-003

**所有 4 处 fused_add_rms_norm 的 weight 参数必须是本层 self.weight。**
过去 V5/V15/V17 三轮审计 Agent 反复犯的错误：将 post_mlp 的 weight 错误引用为下一层的 input_layernorm.weight。
用 id() 做 identity check（非 value check），确保每层只用自己的 weight。

## 执行方式

### 步骤 1：spawn phase-runner

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase 6 runner",
  prompt: """
Phase 6: MLP + Decoder Layer。
读取 .claude/roles/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phase6/SKILL.md 了解本 Phase 的任务细节。
执行完整 implementer→spec→verif 对抗审查链（模式 A：首次执行）。
"""
)
```

phase-runner 返回结构化摘要后，进入步骤 2。

### 步骤 2：主 Agent 防假 PASS 抽查

```bash
RANDOM_SCRIPT=$(ls scripts/test_phase6_*.py scripts/test_phase6_*.sh 2>/dev/null | shuf -n1)
ACTUAL_OUTPUT=$(python "${RANDOM_SCRIPT}" 2>&1 || bash "${RANDOM_SCRIPT}" 2>&1)
```

读取 `./phase_report/PHASE6_VERIFICATION_REPORT.md` 中该脚本的原始 stdout 比对：
- **一致** ✅ → Phase 6 交付，写 `./phase_report/PHASE6_SUMMARY.md`
- **不一致** ❌ → 写 `./phase_report/PHASE6_SPOT_CHECK_FAIL.md` → 回到步骤 1（重试模式）：

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase 6 runner (RETRY)",
  prompt: """
Phase 6 RETRY。
读取 ./phase_report/PHASE6_SPOT_CHECK_FAIL.md 了解失败原因。
读取 .claude/roles/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phase6/SKILL.md 了解任务细节。
执行完整 implementer→spec→verif 修复链（模式 B：重试修复，不得跳过任何环节）。
"""
)
```

重试后换一个脚本再次抽查。连续 5 次驳回 → 停止，向人类报告。

## Phase Script 绑定

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 6 | test_phase6_mlp_forward.py + test_phase6_residual_chain.py + test_phase6_decode_forward_no_clone.py + test_phase6_layer_e2e_random_weights.py |

verification L2: 重跑 Phase 1-5 的全部 scripts/（共 11 个脚本）。

## Phase 6 知识映射

### 必读 JSON 路径

1. qwen3_tp_model_interfaces.mlp — gate_up→silu_and_mul→down 链
2. qwen3_tp_model_interfaces.decode_forward_pattern — **完整 forward_decode 方法体 pseudocode（可直接抄入）**
3. qwen3_tp_model_interfaces.prefill_forward_pattern — prefill 完整数据流 8 步
4. qwen3_tp_model_interfaces.class_hierarchy.QwenMLPTP + QwenDecoderLayerTP — __init__ attr 名
5. qwen3_kernel_contracts.fused_add_rms_norm — **4 处调用均为本层 self.weight**

### 必读 ref_docs

- kernel_replacement_plan.md §三（Snippet B: fused_add_rms_norm, Snippet C: silu_and_mul）

## ⚠️ Phase 6 高发错误

1. **FM-003 跨层 weight**: fused_add_rms_norm 用了下一层的 weight → 输出无 shape 错误但数值全错
2. **gate_up=6400**: 旧 intermediate_size=12800 → gate_up=6400，正确是 6144（12288/4×2）
3. **Eager 路径残留 clone()**: forward_decode 含 .clone() → ~15% 吞吐回退
4. **residual 链断裂**: 首层 res=None 时错误调用了 fused_add_rms_norm 而非 rms_norm

## 关键约束

- 主 Agent 只做调度 + 抽查，不亲自 orchestrate 三角色
- phase-runner 内部 implementer/spec-reviewer/verification 三角色物理隔离（Shell claude -p）
- 审查串行：spec ✅ 才到 verif。spec ❌ 时 verif 不启动
- verif 做 L1（Phase 6 脚本）+ L2（Phase 1-5 回归）
- 主 Agent 抽查是最终裁定——不一致就驳回，连续 5 次才停止
- 主 Agent 禁止降级/修改子代理结论
- PID 互不相同
- scripts/ 不可修改。测试不过 → 改实现代码，不改脚本
