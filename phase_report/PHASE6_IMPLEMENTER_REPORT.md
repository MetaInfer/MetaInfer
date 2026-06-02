PID=948013 | Role=implementer | Timestamp=2026-05-30T00:00:00Z | Phase=6 | Status=SUBMITTED

# Phase 6 Implementer Report: MLP + Decoder Layer

## Implemented

- **engine/models/qwen.py** — QwenMLPTP.forward() and QwenDecoderLayerTP.forward() / forward_decode() reviewed and patched.

### Changes Made

1. **QwenDecoderLayerTP.forward() signature fix** (line 340):
   - Old: `def forward(self, hidden_states, positions, max_seq_len, residual=None)`
   - New: `def forward(self, hidden_states, positions, layer_cache=None, max_seq_len=None, residual=None)`
   - Reason: Blueprint `prefill_forward_pattern.layer_forward_pseudocode` and `model_forward_pseudocode` both require `layer_cache` as a parameter. The caller (Phase 7 QwenForCausalLMTP.forward) passes `layer_cache=None` as a keyword argument. Without this parameter, Phase 7 integration would raise TypeError.

### Reviewed — No Changes Needed

1. **QwenMLPTP.forward()** (lines 300-305): gate_up_proj -> silu_and_mul -> down_proj chain is correct.
   - `gate_up` shape: [B, T, 2 * intermediate/tp] = [B, T, 6144] for TP=4
   - `out` shape: gate_up.shape[-1] // 2 = 3072 = intermediate/tp
   - `silu_and_mul(out, gate_up)` correctly uses pre-allocated output tensor
   - `return self.down_proj(out)` yields [B, T, hidden_size]

2. **QwenDecoderLayerTP.forward_decode()** (lines 361-381): Matches blueprint `decode_forward_pattern.full_method_body` exactly. No clone() in eager path except for the first-layer residual establishment (res=None case).

3. **QwenDecoderLayerTP.forward() residual chain** (lines 340-359): Matches blueprint `prefill_forward_pattern.layer_forward_pseudocode`:
   - First layer (res=None): clone() + rms_norm(hs, res, self.input_layernorm.weight)
   - Subsequent layers: fused_add_rms_norm(hs, res, self.input_layernorm.weight)
   - Post-attention: fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight)
   - MLP forward, return (mlp_out, res)

4. **FM-003 verification** — All 6 RMSNorm call sites checked:
   - forward() line 349: `rms_norm(hs, res, self.input_layernorm.weight, ...)` — res=None path
   - forward() line 351: `fused_add_rms_norm(hs, res, self.input_layernorm.weight, ...)` — res!=None path
   - forward() line 355: `fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight, ...)`
   - forward_decode() line 371: `rms_norm(hs, res, self.input_layernorm.weight, ...)` — res=None path
   - forward_decode() line 373: `fused_add_rms_norm(hs, res, self.input_layernorm.weight, ...)` — res!=None path
   - forward_decode() line 377: `fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight, ...)`
   All weight parameters reference `self.input_layernorm.weight` or `self.post_attention_layernorm.weight`. No cross-layer references.

## Blueprint Nodes Read

- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.mlp`
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern.full_method_body`
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.prefill_forward_pattern.layer_forward_pseudocode`
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenMLPTP`
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenDecoderLayerTP`
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.silu_and_mul`
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint`
- `notebooks-cn/07_improvementPlan/kernel_replacement_plan.md` §三 (Snippet B: fused_add_rms_norm, Snippet C: silu_and_mul)
- `AGENT_SKILL.md` §1 + §2.0.1

## Self-Diff Review

- Only 1 line changed: `forward()` signature added `layer_cache=None` and gave `max_seq_len` a default of `None`
- No files outside `engine/models/qwen.py` were modified
- No `scripts/` files were touched
- QwenAttentionTP and RMSNorm were not modified (off-limits per task)

## Known Issues

- None
