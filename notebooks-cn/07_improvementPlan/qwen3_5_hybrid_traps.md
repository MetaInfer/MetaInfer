# Qwen3.5/3.6 Hybrid — 实现陷阱与注意事项

<!-- source: evolution/evo-001, open_source_enabled=true, timestamp=2026-07-03 -->

## 核心不可推断差异

以下知识无法仅从 `config.json` 推断，必须在实现前了解：

### TRAP-1: Qwen3_5RMSNorm 的 zeros-init 语义

**陷阱**: 标准 RMSNorm weight init = ones，Qwen3_5RMSNorm weight init = zeros。如果用标准实现加载 Qwen3.5 权重 → 输出全零。

**为什么不可推断**: config.json 中无 "norm_variant" 字段。只能从 HF code 或实际权重值推断（检查 weight.min() == weight.max() == 0）。

**实现要点**:
```python
class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))  # NOT ones
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)
```

### TRAP-2: GatedDeltaNet 的 state 维度序

**陷阱**: State 的期望维度是 `[Dk, Dv]` 而非 `[Dv, Dk]`。如果维度序错误，outer product `k^T @ delta` 会产生 shape mismatch 或静默错误。

**为什么不可推断**: config.json 只给出 `linear_key_head_dim` 和 `linear_value_head_dim`，不指定存储顺序。HF 实现使用 `[Dk, Dv]`。

**实现要点**: 所有 state 创建、更新、输出计算必须使用 `[B, H, Dk, Dv]`。输出计算 `o = sum(S * q, dim=-2)` 依赖 Dk 在倒数第二维。

### TRAP-3: conv_state 缓存协议

**陷阱**: causal conv1d (kernel=4, depthwise) 在 decode 时需要前 3 个 token 的原始 QKV 值。若不缓存 → zero-padding → 3000× 误差。

**为什么不可推断**: config.json 的 `linear_conv_kernel_dim=4` 提示了 kernel size，但不说明 decode 路径如何获取历史上下文。HF 实现会缓存。

**实现要点**:
- Prefill 结束时：`_conv_state = mixed_qkv[:, :, -4:].clone()` — 保存 raw QKV（conv 之前的值）
- Decode 时：`mixed_qkv_with_context = torch.cat([_conv_state, current_qkv], dim=-1)` — 拼接历史
- Decode 后：`_conv_state = combined[:, :, -4:].clone()` — 滚动更新

### TRAP-4: FullAttention q_proj 双倍输出

**陷阱**: q_proj 的输出维度是 `2 * num_heads * head_dim`，后半部分用于 output gate。忘记分离 → Q 维度错误 + gate 未生效。

**为什么不可推断**: config.json 有 `attn_output_gate: true` 提示 gate 存在，但不说明 gate 值与 Q 共享同一个 projection。HF 实现中 q_proj 输出 double size。

**实现要点**:
```python
q_double = self.q_proj(hidden_states)  # [B, S, 2 * q_size]
q, gate = torch.chunk(q_double.view(B, S, num_heads, head_dim * 2), 2, dim=-1)
# ... attention ...
out = out * torch.sigmoid(gate.reshape(B, S, q_size))
```

### TRAP-5: MRoPE mrope_section 不可推断

**陷阱**: `mrope_section` 决定三个位置维度的维度分配。不能从 `head_dim=256` 和 `partial_rotary_factor=0.25` 推断。

**为什么不可推断**: rotary_dim=64 可推断，但 `[11,11,10]` 的分配方式是先验知识。不能用 `[64/3, 64/3, 64/3]` 近似。

**实现要点**: 必须从 `config.rope_parameters.mrope_section` 或 `config.mrope_section` 精确读取。

### TRAP-6: weight key prefix model.language_model.*

**陷阱**: 所有 transformer 层的 HF key 前缀为 `model.language_model.*`，而非 Qwen3-8B 的 `model.*`。

**为什么不可推断**: config.json 中无 prefix 信息。Qwen3_5ForConditionalGeneration 包装了 Qwen3_5Model 的 `.language_model` 子模块，这只是 HF 模型类的 Python 属性命名。

**实现要点**: 权重加载的正则 pattern 必须匹配 `model\.language_model\.layers\.(\d+)\.(.+)`。

### TRAP-7: 混合缓存的 per-layer 类型

**陷阱**: 不同层类型需要不同的缓存基础设施。linear_attention 层需要 conv_state + recurrent_state，full_attention 层需要 paged KV cache。引擎的 cache 管理必须支持 per-layer 类型分派。

**为什么不可推断**: config.json 的 `layer_types` 列表给出了每层类型，但不说明不同层类型有完全不同的缓存需求。

**实现要点**: Decode 路径中，linar_attention 层调用 `forward_decode(hidden_states)` 仅依赖内部 state，full_attention 层调用 `forward_decode(hidden_states, positions, kv_len, max_seq_len)` 需要 KV cache 参数。

### TRAP-8: GatedDeltaNet 输出计算顺序

**陷阱**: Recurrent rule 的输出 `o_t = sum(S_t * q_t, dim=-2)` 必须使用 **POST-update** 的 state。如果在 state update 之前计算输出，结果完全错误。

**为什么不可推断**: 公式 `o_t = S_t @ q_t` 中 `S_t` 是 update 前还是后的 state 是论文级别的先验知识。HF 实现使用 post-update。

**实现要点**: loop body 的顺序必须是: state decay → kv_mem → delta → **state update** → output compute。

---

## 数值稳定性注意事项

### GatedDeltaNet fp32 累积

Recurrence 使用 fp32 进行 state 累积（`initial_dtype` 为 bf16 但内部计算为 fp32）。在 bf16 下做 recurrent state 更新会导致 ~1e-2 级别的累积误差，足够产生错误 token。

### gate exponentiation

`g_t = g[:, :, i].exp()` 在每个 timestep exponentiate。如果 A_log 较大负值，exp(-large) → 0 是正确行为。但如果 A_log 为正值，exp → large 会导致 state 膨胀。

### Q/K L2 Normalize

`use_qk_l2norm_in_kernel=True` 在 recurrence 前对 Q 和 K 做 L2 normalize。这是 GatedDeltaNet 的必要步骤——不做 normalize 会导致 state 不稳定爆炸。

---

## 可复用组件（无需重新实现）

以下组件与 Qwen3-8B 完全相同，可直接复用：

| 组件 | 复用来源 | 注意事项 |
|------|---------|---------|
| QwenMLPTP (SwiGLU) | engine/models/qwen.py | gate-up 拼接顺序相同 |
| VocabParallelEmbedding | engine/tp_layers/embedding.py | dim=0 切片 |
| ParallelLMHead | engine/tp_layers/embedding.py | tied embedding 检查 |
| Paged KV cache 机制 | engine/models/qwen.py | 仅 full_attention 层，block_size=256 |
| silu_and_mul kernel | engine/kernels/vllm_wrappers.py | — |
| rms_norm / fused_add_rms_norm | engine/kernels/vllm_wrappers.py | — |
| ColumnParallelLinear / RowParallelLinear | engine/tp_layers/linear.py | — |
| MergedColumnParallelLinear | engine/tp_layers/linear.py | — |
| CustomAR / all_reduce_sum | engine/tp_layers/distributed.py | 平台检测路由 |

## 新增组件

| 组件 | 实现位置 | 复杂度 |
|------|---------|--------|
| Qwen3_5RMSNorm | engine/models/qwen3_5.py | 低 (~15 行) |
| Qwen3_5RMSNormGated | engine/models/qwen3_5.py | 低 (~15 行) |
| _DeltaNetQKVColumnParallelWeight | engine/models/qwen3_5.py | 中 (~50 行, head-aware sharding) |
| QwenGatedDeltaNetTP | engine/models/qwen3_5.py | 高 (~200 行, prefill + decode) |
| QwenFullAttentionTP | engine/models/qwen3_5.py | 高 (~250 行, MRoPE + gate) |
| QwenHybridDecoderLayerTP | engine/models/qwen3_5.py | 中 (~80 行, dispatch) |
| Qwen3_5TPConfig | engine/models/qwen3_5.py | 低 (~50 行, config 解析) |
| MRoPE cos/sin cache builder | engine/models/qwen3_5.py | 中 (~40 行) |
| _torch_causal_conv1d_silu | engine/models/qwen3_5.py | 低 (~15 行) |
| _torch_recurrent_gated_delta_rule | engine/models/qwen3_5.py | 高 (~80 行, 数值敏感) |

## 验证策略

1. **逐组件验证**: 每个 Norm/Attention/MLP 组件独立与 HF 参考对齐
2. **Prefill/Decode 分离**: GatedDeltaNet 的 prefill 和 decode 路径分别验证
3. **conv_state 连续性**: 验证 prefill→decode 的 state 传递正确（单 token prefill + 单 token decode vs 双token prefill）
4. **TP=4 一致性**: 验证 TP=4 forward 与 TP=1 参考在 all-gather 后一致
5. **Full model E2E**: 完整 prompt 的 logits 与 HF 参考对齐
