# PHASE9 SPEC REVIEW REPORT

- **PID**: 995070
- **Role**: spec-reviewer
- **Timestamp**: 2026-05-30T08:14:37Z
- **Phase**: 9
- **Spec Compliance**: ✅ PASS

---

## 审查任务（聚焦）

重新审查 `engine/models/qwen.py` 中以下两项：
1. `RMSNorm.forward` — 是否始终返回 2-tuple `(out, residual_or_None)`
2. `QwenAttentionTP` 的 `q_norm` / `k_norm` 调用 — 是否已改为 tuple unpacking

上次报告（PID 178234）的 **Issue 1 (CRITICAL)**：`RMSNorm.forward` 返回单 Tensor 但调用方 `hidden_states, _ = self.norm(...)` 做元组解包。

---

## 1. RMSNorm.forward — 返回值审查

**位置**: `engine/models/qwen.py` 第 72–79 行

```python
def forward(self, x, residual=None):
    if residual is None:
        out = torch.empty_like(x)
        rms_norm(out, x.contiguous(), self.weight, self.eps)
        return out, None                     # ← 2-tuple (Tensor, None)
    else:
        fused_add_rms_norm(x.contiguous(), residual.contiguous(), self.weight, self.eps)
        return x, residual                    # ← 2-tuple (Tensor, Tensor)
```

| 分支 | 返回值 | 类型 |
|------|--------|------|
| `residual is None` | `(out, None)` | `tuple[Tensor, NoneType]` |
| `residual is not None` | `(x, residual)` | `tuple[Tensor, Tensor]` |

**判定**: ✅ **两个分支均返回 2-tuple**。旧 bug（含 residual 分支返回单 `x`）已修复。

---

## 2. RMSNorm.forward 的全部 7 个调用点

### 2.1 QwenAttentionTP — q_norm/k_norm（4 处）

**Prefill** `forward()` 第 187–188 行：
```python
q, _ = self.q_norm(q)       # ✅ tuple unpack, residual=None → 返回 (out, None)
k, _ = self.k_norm(k)       # ✅ tuple unpack, residual=None → 返回 (out, None)
```

**Decode** `forward_decode()` 第 249–250 行：
```python
q, _ = self.q_norm(q)       # ✅ tuple unpack, residual=None → 返回 (out, None)
k, _ = self.k_norm(k)       # ✅ tuple unpack, residual=None → 返回 (out, None)
```

**判定**: ✅ 全部使用 `q, _ = self.q_norm(q)` 形式的 tuple unpacking。

### 2.2 QwenForCausalLMTP — self.norm（3 处）

**Prefill** `forward()` 第 502 行：
```python
hidden_states, _ = self.norm(hidden_states, residual)
# residual is Tensor → fused_add_rms_norm 分支 → 返回 (x, residual)
```

**Decode** `forward()` 第 513 行：
```python
hidden_states, _ = self.norm(hidden_states, residual)
# residual is Tensor → fused_add_rms_norm 分支 → 返回 (x, residual)
```

**forward_decode()** 第 543 行：
```python
hidden_states, _ = self.norm(hidden_states, residual)
# residual is Tensor → fused_add_rms_norm 分支 → 返回 (x, residual)
```

**判定**: ✅ 全部使用 `hidden_states, _ = self.norm(...)` 形式的 tuple unpacking。

---

## 3. QwenDecoderLayerTP 中的低层 kernel 调用（不涉及 RMSNorm.forward）

QwenDecoderLayerTP 的 `forward()` 和 `forward_decode()` 直接调用从 `engine.kernels.vllm_wrappers` 导入的裸 `rms_norm()` / `fused_add_rms_norm()` 函数，**不走 RMSNorm.forward**。

| 行号 | 调用 | 函数来源 | 返回值处理 |
|------|------|---------|-----------|
| 355 | `rms_norm(hs, res, ...)` | vLLM wrapper | bare call（in-place） |
| 357 | `fused_add_rms_norm(hs, res, ...)` | vLLM wrapper | bare call（in-place） |
| 361 | `fused_add_rms_norm(hs, res, ...)` | vLLM wrapper | bare call（in-place） |
| 377 | `rms_norm(hs, res, ...)` | vLLM wrapper | bare call（in-place） |
| 379 | `fused_add_rms_norm(hs, res, ...)` | vLLM wrapper | bare call（in-place） |
| 383 | `fused_add_rms_norm(hs, res, ...)` | vLLM wrapper | bare call（in-place） |

**判定**: ✅ 低层 kernel 均为原地修改函数，不返回 tuple，不需要解包。与 RMSNorm.forward 的 2-tuple 契约无关。

---

## 4. 蓝图契约对照

**JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces`

| 蓝图契约 | 代码位置 | 状态 |
|---------|---------|------|
| `decode_forward_pattern.full_method_body` 第 970 行: `q = self.q_norm(q); k = self.k_norm(k)` | `qwen.py:187-188`, `249-250` | ✅ 实现为 `q, _ = self.q_norm(q)`（蓝图伪代码简化，实际 tuple 解包正确） |
| `prefill_forward_pattern.layer_forward_pseudocode` 第 1046 行: `q = self.q_norm(q); k = self.k_norm(k)` | `qwen.py:187-188` | ✅ 同上 |
| `model_forward_pseudocode` 第 1104 行: `hidden_states, _ = self.norm(hidden_states, residual)` | `qwen.py:502,513,543` | ✅ 完全匹配 |
| `class_hierarchy.QwenAttentionTP.attrs` 第 1156-1157 行: `self.q_norm = RMSNorm(...)`, `self.k_norm = RMSNorm(...)` | `qwen.py:141-142` | ✅ 完全匹配 |

---

## 最终判定

| 审查项 | 文件:行号 | 状态 |
|--------|-----------|------|
| RMSNorm.forward 两个分支均返回 2-tuple | `qwen.py:72-79` | ✅ PASS |
| QwenAttentionTP.forward q_norm 解包 | `qwen.py:187` | ✅ PASS |
| QwenAttentionTP.forward k_norm 解包 | `qwen.py:188` | ✅ PASS |
| QwenAttentionTP.forward_decode q_norm 解包 | `qwen.py:249` | ✅ PASS |
| QwenAttentionTP.forward_decode k_norm 解包 | `qwen.py:250` | ✅ PASS |
| QwenForCausalLMTP.forward (prefill) self.norm 解包 | `qwen.py:502` | ✅ PASS |
| QwenForCausalLMTP.forward (decode) self.norm 解包 | `qwen.py:513` | ✅ PASS |
| QwenForCausalLMTP.forward_decode self.norm 解包 | `qwen.py:543` | ✅ PASS |
| QwenDecoderLayerTP 低层 kernel 调用 | `qwen.py:355,357,361,377,379,383` | ✅ N/A（非 RMSNorm.forward） |

**总判定: ✅ PASS** — `RMSNorm.forward` 的两个分支均返回 2-tuple `(out, residual_or_None)`；全部 7 处 RMSNorm.forward 调用点均使用正确的 tuple unpacking；4 处 q_norm/k_norm 调用均使用 `q, _ = self.q_norm(q)` 形式解包。上次 CRITICAL bug 已完全修复。
