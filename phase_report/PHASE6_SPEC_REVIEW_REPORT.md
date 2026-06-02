# Phase 6 Spec Review Report

- **PID**: 949691
- **Role**: spec-reviewer
- **Timestamp**: 2026-05-30T00:00:00Z
- **Phase**: 6
- **Reviewed Files**: `engine/models/qwen.py` (lines 276-381), `engine/tp_layers/linear.py` (lines 246-316 for cross-reference)

---

## Spec Compliance: ✅ PASS

Spec 审查通过，代码与蓝图契约一致，可移交 verification。

---

## Evidence Chain

### Contract 1: QwenMLPTP.forward — gate_up→silu_and_mul→down 链

- **[qwen3_tp_model_interfaces.mlp.gate_up_merged]**: ✅ @ `qwen.py:300-305`
  - `gate_up_proj` 输出 `[B,T,2*intermediate/tp]`（`MergedColumnParallelLinear` 默认 `gather_output=False`，line 262 确认）
  - `torch.empty(x.shape[0], x.shape[1], gate_up.shape[-1] // 2, ...)` 预分配 `[B,T,intermediate/tp]`
  - `silu_and_mul(out, gate_up)` in-place 激活
  - `self.down_proj(out)` RowParallelLinear + all_reduce_sum
  - 对 Qwen3-8B TP=4: gate_up=[B,T,6144] → silu_and_mul → [B,T,3072] ✓

- **[qwen3_tp_model_interfaces.mlp.gate_up_dimension_6144_not_6400]**: ✅ @ `qwen.py:293-295`
  - `cfg.intermediate_size` 动态读取，无硬编码
  - `MergedColumnParallelLinear` 内部 `local_out = 2 * out_features // tp_size` (`linear.py:272`) → 对 Qwen3-8B (intermediate=12288, tp=4) = 6144
  - 无误判为 6400（SwiGLU 3/2 膨胀的错误假设）的风险

- **[qwen3_tp_model_interfaces.mlp.class_hierarchy.QwenMLPTP.attrs]**: ✅ @ `qwen.py:293-298`
  - `self.gate_up_proj = MergedColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, tp_size=tp_size, bias=False)` — 属性名精确匹配
  - `self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, tp_size=tp_size, bias=False)` — 属性名精确匹配
  - `gather_output` 未显式传递但 `MergedColumnParallelLinear.__init__` 默认值 `False`（`linear.py:262`），功能等价

### Contract 2: QwenDecoderLayerTP.forward_decode — full_method_body 逐行对照

- **[decode_forward_pattern.full_method_body.residual_chain]**: ✅ @ `qwen.py:368-373`
  - `hs, res = hidden_states, residual` — 别名赋值（eager path 无 clone）✓
  - `if res is None: res = hs.clone(); rms_norm(hs, res, self.input_layernorm.weight, ...)` — 首层初始化 ✓
  - `else: fused_add_rms_norm(hs, res, self.input_layernorm.weight, ...)` — 后续层 ✓

- **[decode_forward_pattern.full_method_body.attention]**: ✅ @ `qwen.py:375`
  - `hs = self.self_attn.forward_decode(hs, positions, kv_len, max_seq_len)` — 调用 decode 注意力 ✓

- **[decode_forward_pattern.full_method_body.post_attn_norm]**: ✅ @ `qwen.py:377-378`
  - `fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)` ✓

- **[decode_forward_pattern.full_method_body.mlp]**: ✅ @ `qwen.py:380-381`
  - `mlp_out = self.mlp(hs); return mlp_out, res` ✓

### Contract 3: QwenDecoderLayerTP.forward (prefill) — layer_forward_pseudocode 逐行对照

- **[prefill_forward_pattern.layer_forward_pseudocode.residual_chain]**: ✅ @ `qwen.py:347-351`
  - `if res is None: res = hs.clone(); rms_norm(hs, res, self.input_layernorm.weight, ...)` — 首层 ✓
  - `else: fused_add_rms_norm(hs, res, self.input_layernorm.weight, ...)` — 后续层 ✓

- **[prefill_forward_pattern.layer_forward_pseudocode.attention]**: ✅ @ `qwen.py:353`
  - `hs = self.self_attn.forward(hs, positions, max_seq_len)` — 调用 prefill 注意力 ✓

- **[prefill_forward_pattern.layer_forward_pseudocode.post_attn_norm]**: ✅ @ `qwen.py:355-356`
  - `fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)` ✓

- **[prefill_forward_pattern.layer_forward_pseudocode.mlp_return]**: ✅ @ `qwen.py:358-359`
  - `mlp_out = self.mlp(hs); return mlp_out, res` ✓

### Contract 4: fused_add_rms_norm.constraint — weight 全部来自本层

- **[qwen3_kernel_contracts.fused_add_rms_norm.constraint]**: ✅ — 全部 6 处核验通过

| # | 位置 | 调用 | weight 参数 | 角色 |
|---|------|------|------------|------|
| 1 | `qwen.py:349` | `rms_norm(hs, res, ...)` | `self.input_layernorm.weight` | prefill pre-attn (首层) |
| 2 | `qwen.py:351` | `fused_add_rms_norm(hs, res, ...)` | `self.input_layernorm.weight` | prefill pre-attn (后续层) |
| 3 | `qwen.py:355-356` | `fused_add_rms_norm(hs, res, ...)` | `self.post_attention_layernorm.weight` | prefill post-attn |
| 4 | `qwen.py:371` | `rms_norm(hs, res, ...)` | `self.input_layernorm.weight` | decode pre-attn (首层) |
| 5 | `qwen.py:373` | `fused_add_rms_norm(hs, res, ...)` | `self.input_layernorm.weight` | decode post-attn (后续层) |
| 6 | `qwen.py:377-378` | `fused_add_rms_norm(hs, res, ...)` | `self.post_attention_layernorm.weight` | decode post-attn |

- **无跨层 weight 引用**。所有 weight 参数均为 `self.input_layernorm.weight` 或 `self.post_attention_layernorm.weight`。

### Contract 5: 4 个高发错误逐条核验

- **[FM-003 跨层 weight]**: ✅ — 全部 6 处 `self.xxx.weight`，无跨层引用

- **[gate_up 维度 6144 非 6400]**: ✅ @ `qwen.py:293-295` + `linear.py:272`
  - 从 `cfg.intermediate_size` 动态计算，无硬编码
  - `MergedColumnParallelLinear` 内部 `local_out = 2 * intermediate // tp`
  - Qwen3-8B intermediate_size=12288, tp=4 → 6144 ✓

- **[decode 无 clone()]**: ✅ @ `qwen.py:361-381`
  - `forward_decode` 方法体无无条件 `clone()`（`hs, res = hidden_states, residual` 直接别名）
  - 首层 `res = hs.clone()` 是 residual 初始化，蓝图标定为 `# first layer only`，属正确行为
  - Eager 路径无性能回退的 `clone()` 残留 ✓

- **[residual 链首层 clone+rms_norm，后续 fused_add_rms_norm]**: ✅ @ `qwen.py:347-351` (prefill) + `qwen.py:369-373` (decode)
  - `res is None` → `clone()` + `rms_norm()` ✓
  - `res is not None` → `fused_add_rms_norm()` ✓

### Phase 5 组件未篡改检查

- **[RMSNorm 类定义]**: ✅ @ `qwen.py:46` — `class RMSNorm(nn.Module)` 行号/签名不变
  - `__init__(self, hidden_size, eps=1e-6)` @ line 61 ✓
  - `forward(self, x, residual=None)` @ line 66 ✓

- **[QwenAttentionTP 类定义]**: ✅ @ `qwen.py:80` — `class QwenAttentionTP(nn.Module)` 行号/签名不变
  - `__init__(self, cfg, tp_size=None)` @ line 97 ✓
  - `forward(self, hidden_states, positions, max_seq_len)` @ line 160 ✓
  - `forward_decode(self, hidden_states, positions, kv_len, max_seq_len)` @ line 223 ✓

---

## Issues Found: 无

无 FAIL 项。全部 5 类契约逐条核验通过。

---

## Blueprint Information Gaps: 无

无信息断裂或矛盾。

---

## 总结

`engine/models/qwen.py` 中 Phase 6 相关代码（QwenMLPTP.forward、QwenDecoderLayerTP.forward、QwenDecoderLayerTP.forward_decode）精确匹配 `inference_blueprint.json` 的全部契约：

- MLP gate_up→silu_and_mul→down 链：✅
- Decode forward_decode 方法体：✅
- Prefill forward 方法体：✅
- 全部 6 处 fused_add_rms_norm/rms_norm 的 weight 均为本层 weight：✅
- 4 个高发错误全部排除：✅
- Phase 5 组件（RMSNorm、QwenAttentionTP）未被篡改：✅

**Spec 审查通过，代码与蓝图契约一致，可移交 verification。**
