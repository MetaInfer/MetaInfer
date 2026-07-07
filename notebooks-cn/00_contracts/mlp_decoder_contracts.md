# MLP & Decoder Layer — API 契约

> 关联 notebooks: `07_improvementPlan/kernel_replacement_plan.md` §三

## 概述

QwenMLPTP (gate_up → silu_and_mul → down) + QwenDecoderLayerTP (residual chain)。
源实现文件: `engine/models/qwen.py`

---

## 接口签名

### QwenMLPTP

```python
class QwenMLPTP(nn.Module):
    def __init__(self, cfg):
        self.gate_up_proj = MergedColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False)
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, bias=False)
    
    def forward(self, x):  # x: [B, T, hidden_size]
        gate_up = self.gate_up_proj(x)  # [B, T, 2*intermediate/tp]
        out = torch.empty(B, T, intermediate//tp, dtype=x.dtype, device=x.device)
        silu_and_mul(out, gate_up)  # out = SiLU(gate) * up
        return self.down_proj(out)  # [B, T, hidden_size]
```

### QwenDecoderLayerTP

**属性命名 (权重加载关键)**:
- `.self_attn` 非 `.attention`
- `.mlp`
- `.input_layernorm` 非 `.ln_1`
- `.post_attention_layernorm` 非 `.ln_2`

### Prefill Forward (Decoder Layer)

```python
def forward(self, hidden_states, positions, layer_cache, max_seq_len, residual=None):
    if residual is None:
        residual = hidden_states.clone()
        rms_norm(hidden_states, residual, self.input_layernorm.weight, self.input_layernorm.eps)
    else:
        fused_add_rms_norm(hidden_states, residual, self.input_layernorm.weight, self.input_layernorm.eps)
    attn_out = self.self_attn.forward(hidden_states, positions, max_seq_len)
    fused_add_rms_norm(attn_out, residual, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
    mlp_out = self.mlp(attn_out)
    return mlp_out, residual
```

### Decode Forward (Decoder Layer)

```python
def forward_decode(self, hidden_states, positions, kv_len, max_seq_len, residual=None):
    hs, res = hidden_states, residual
    if res is None:
        res = hs.clone()
        rms_norm(hs, res, self.input_layernorm.weight, self.input_layernorm.eps)  # first layer only
    else:
        fused_add_rms_norm(hs, res, self.input_layernorm.weight, self.input_layernorm.eps)
    hs = self.self_attn.forward_decode(hs, positions, kv_len, max_seq_len)
    fused_add_rms_norm(hs, res, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
    mlp_out = self.mlp(hs)
    return mlp_out, res
```

---

## 数据流约束

### Prefill 完整 8 步

```
1. input_ids → embed_tokens → hidden_states [B, L_prompt, hidden_size]
2. 逐层 forward(hidden_states, positions, layer_cache=None, max_seq_len, residual=None)
3.   层内: input_layernorm → attention.forward (prefill) → post_attn_layernorm → mlp
4.     attention: qkv_proj → Q/K norm → rotary_embedding → flash_attn_varlen_func(causal=True)
5.     K,V 写入 paged cache (attention 之后，非之前)
6.     o_proj → all_reduce_sum
7. 所有层完成后: model.norm(hidden_states, residual) → lm_head → logits
8. 每序列 _kv_len_gpu[0] 初始化为各自 seq.seq_len()
```

### Decode 关键差异

| 维度 | Prefill | Decode |
|------|---------|--------|
| attention kernel | flash_attn_varlen_func | flash_attn_with_kvcache |
| causal | True | False |
| KV cache | allocate + write | read + append 1 token |
| layer method | forward() | forward_decode() |
| hidden_states shape | [1, S, H] | [1, 1, H] |

### Residual Chain 语义

- vLLM-style DecoderLayer 返回 `(mlp_out, residual)` 分离二元组
- 最终 norm 前必须手动合并: `hidden_states = hidden_states + residual`
- 缺失此合并 → top-1 token 为 `!` 且 logit=0.0 (FM-017)

---

## 陷阱与反模式

- **FM-016**: RMSNorm 计算顺序 — HF 权重已针对 `self.weight * x_f.to(bf16)` 的精度路径训练，不能用 `(self.weight.float() * x_f).to(bf16)`
- **FM-017**: forward 缺失 `hs = hs + res` — 36 层循环后必须手动合并 residual
- **FM-003**: fused_add_rms_norm 全部使用本层 self.weight — 无跨层 weight 依赖
- **MLP-001**: silu_and_mul input 前 gate 后 up (非 up-gate)
- **MLP-002**: 属性名 `.gate_up_proj` 非 `.gate_proj`，`.self_attn` 非 `.attention`
- **MLP-003**: `silu_and_mul` 的 out tensor 需预分配，不能用 `torch.empty_like(gate_up[..., :half])` (含隐式 CUDA 查询) — 用 `torch.empty(...)` + 显式参数

---

## QwenHybridDecoderLayerTP (Qwen3.5 Hybrid)

<!-- source: evolution/evo-001, timestamp=2026-07-03 -->

### 概述

Hybrid decoder layer 根据 `layer_type` 分发到不同的 attention 子模块:
- `"linear_attention"` → QwenGatedDeltaNetTP
- `"full_attention"` → QwenFullAttentionTP

所有层共享同一个 MLP (QwenMLPTP) 和 Qwen3_5RMSNorm (zeros-init)。

### 属性命名

- `.layer_type`: `"linear_attention"` 或 `"full_attention"`
- `.linear_attn`: QwenGatedDeltaNetTP (仅 linear_attention 层, 非 None)
- `.self_attn`: QwenFullAttentionTP (仅 full_attention 层, 非 None)
- `.mlp`: QwenMLPTP
- `.input_layernorm`: Qwen3_5RMSNorm (**zeros-init**, 非标准 RMSNorm)
- `.post_attention_layernorm`: Qwen3_5RMSNorm (**zeros-init**)

### Prefill Forward

```python
def forward(self, hidden_states, positions, layer_cache, max_seq_len, residual=None):
    # Norm (使用 _effective_weight() = 1.0 + self.weight 兼容 vLLM kernel)
    if residual is None:
        residual = hidden_states.clone()
        rms_norm(hidden_states, residual,
                 self.input_layernorm._effective_weight(),
                 self.input_layernorm.variance_epsilon)
    else:
        fused_add_rms_norm(hidden_states, residual,
                           self.input_layernorm._effective_weight(),
                           self.input_layernorm.variance_epsilon)

    # Attention dispatch
    if self.layer_type == "linear_attention":
        attn_out = self.linear_attn.forward(hidden_states)
    else:
        attn_out = self.self_attn.forward(hidden_states, positions, max_seq_len)

    # Post-attention norm (same _effective_weight pattern)
    fused_add_rms_norm(attn_out, residual,
                       self.post_attention_layernorm._effective_weight(),
                       self.post_attention_layernorm.variance_epsilon)

    # MLP
    mlp_out = self.mlp(attn_out)
    return mlp_out, residual
```

### Decode Forward

```python
def forward_decode(self, hidden_states, positions, kv_len, max_seq_len, residual=None):
    # Same norm pattern as prefill...

    if self.layer_type == "linear_attention":
        hs = self.linear_attn.forward_decode(hs)  # uses internal _conv_state + _recurrent_state
    else:
        hs = self.self_attn.forward_decode(hs, positions, kv_len, max_seq_len)  # uses KV cache

    # Same post-norm + MLP pattern...
```

### 与 Qwen3-8B DecoderLayer 的关键差异

| 维度 | Qwen3-8B | Qwen3.5/3.6 |
|------|---------|-------------|
| Norm 类型 | RMSNorm (standard, ones init) | Qwen3_5RMSNorm (zeros init, 1+w formula) |
| Norm kernel 调用 | `rms_norm(out, x, weight, eps)` | `rms_norm(out, x, _effective_weight(), eps)` |
| Attention 分派 | 单一 self_attn | per-layer type dispatch |
| linear_attn 子模块 | 不存在 | QwenGatedDeltaNetTP (48 layers) |
| self_attn 子模块 | QwenAttentionTP (merged QKV) | QwenFullAttentionTP (separate Q/K/V + gate) |
| Decode 路径缓存 | 统一 KV cache | linear_attn: state cache; full_attn: KV cache |

### 陷阱

- **HYBRID-001**: Qwen3_5RMSNorm 使用 `_effective_weight()` (返回 1+w) 而非 `.weight` 传递给 vLLM kernel
- **HYBRID-002**: linear_attn.forward 不需要 positions/max_seq_len；self_attn.forward 需要。接口不对称
- **HYBRID-003**: linear_attn.forward_decode 不需要 kv_len（使用内部 state）；self_attn.forward_decode 需要。接口不对称
- **HYBRID-004**: MTP head 的 decoder layer 必须覆盖 layer_type 为 "full_attention"（MTP 总是 full attention）
