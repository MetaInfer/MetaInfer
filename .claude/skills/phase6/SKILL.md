# Phase 6：MLP + Decoder Layer

## 触发词

`/phase6`

## 角色

你是主 Agent。按 CLAUDE.md 的 spawn 协议执行完整串行路径。Phase 6 错误密度与 Phase 5 并列最高。

**关键前提**：`engine/models/qwen.py` 已存在（Phase 5 产物）。Phase 6 是**修改**现有文件，不是新建：
- **补全 QwenMLPTP.forward**：stub → 完整 gate_up→silu_and_mul→down 链
- **审查 QwenDecoderLayerTP 的 prefill/decode 路径**：确保 residual chain 正确
- **禁止修改 QwenAttentionTP 和 RMSNorm**：Phase 5 已完成

## 任务

- QwenMLPTP.forward：gate_up_proj(MergedColumnParallelLinear) → silu_and_mul → down_proj(RowParallelLinear)
- QwenDecoderLayerTP.forward(prefill)：input_layernorm → attention.forward → post_attention_layernorm → mlp
- QwenDecoderLayerTP.forward_decode：fused_add_rms_norm(input,residual,self.input_layernorm.weight) → attn → fused_add_rms_norm(attn_out,residual,self.post_attention_layernorm.weight) → mlp
- Residual chain：首层 res=None → rms_norm；后续层 fused_add_rms_norm

## Phase-Script 绑定

| 脚本 | 门禁 |
|------|------|
| `test_phase6_mlp_forward.py` | MLP gate_up→silu_and_mul→down 链正确性 |
| `test_phase6_residual_chain.py` | residual chain 不断裂 |
| `test_phase6_decode_forward_no_clone.py` | forward_decode 零 clone |
| `test_phase6_layer_e2e_random_weights.py` | 随机权重下层端到端 |

verif L2：重跑 Phase 1-5 全部 11 个脚本。

## 知识映射

- Blueprint：`qwen3_tp_model_interfaces.mlp` → `decode_forward_pattern`（完整方法体 pseudocode）→ `prefill_forward_pattern` → `class_hierarchy.QwenMLPTP` + `QwenDecoderLayerTP`
- ref_docs：`kernel_replacement_plan.md` §三（Snippet B: fused_add_rms_norm, Snippet C: silu_and_mul）

## ⚠️ 四大高发错误

1. **FM-003 跨层 weight**：fused_add_rms_norm 用了下一层的 weight → 数值全错无 shape 报错。所有 4 处调用必须用**本层 self.weight**，用 id() 做 identity check
2. **gate_up=6400**：旧 intermediate_size=12800 → 正确是 6144（12288/4×2）
3. **Eager 路径残留 clone()**：forward_decode 含 .clone() → ~15% 吞吐回退
4. **residual 链断裂**：首层 res=None 时错误调用了 fused_add_rms_norm 而非 rms_norm

## 关键约束

- 4 处 fused_add_rms_norm 的 weight 全部是本层 self.input_layernorm.weight 或 self.post_attention_layernorm.weight
- 禁止修改 QwenAttentionTP 和 RMSNorm
