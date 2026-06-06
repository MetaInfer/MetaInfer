# Skill: Phase 6 编码（MLP + Decoder Layer）

## 触发词

`/inference:phase6` 或 `/phase6`

## 概述

构建 Phase 6: MLP + Decoder Layer —— 整个流水线中**错误密度与 Phase 5 并列最高的阶段**。

---

## 环境

- 模型权重: `${MODEL_DIR}`
- Conda: `${PYTHON_PATH}/python`
- GPU: 4×A800
- Phase 1-5 的代码已存在于 ./engine/ 下，不要重复生成。
- **关键**：`engine/models/qwen.py` 已存在——Phase 5 创建了完整 QwenAttentionTP + QwenDecoderLayerTP + RMSNorm，以及 QwenMLPTP stub（仅 __init__，forward 是空壳）。Phase 6 的职责是修改 qwen.py，**不是新建文件**：
  - **补全 QwenMLPTP.forward**：stub → 完整 gate_up→silu_and_mul→down 链
  - **审查 QwenDecoderLayerTP 的 prefill/decode 路径**：Phase 5 已实现，Phase 6 需确保 residual chain 与蓝图完全一致
  - **禁止修改 QwenAttentionTP 和 RMSNorm**：Phase 5 已完成并通过验收

## 你的角色

读取本目录的 CLAUDE.md。Phase 6 是整个流水线中**错误密度与 Phase 5 并列最高的阶段**。

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

## 执行步骤

步骤 1: implementer → ./phase_report/PHASE6_IMPLEMENTER_REPORT.md（SUBMITTED）
步骤 2: spec-reviewer → ./phase_report/PHASE6_SPEC_REVIEW_REPORT.md（Shell claude -p，独立 PID）
步骤 3: verification → ./phase_report/PHASE6_VERIFICATION_REPORT.md（L1+L2，独立 PID）
步骤 4: 主 Agent 汇总（含步骤 3.5 防假 PASS 抽查）→ ./phase_report/PHASE6_SUMMARY.md（PID 交叉验证）

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

- implementer 不跑测试、不判 PASS
- spec 先审 → ✅ 才到 verif
- verif 做 L1+L2
- 主 Agent 禁止降级子代理结论
- PID 互不相同
