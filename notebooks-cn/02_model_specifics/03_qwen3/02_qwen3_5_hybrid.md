# Qwen3.5/3.6 Hybrid — 混合架构概述

<!-- source: evolution/evo-001, open_source_enabled=true, timestamp=2026-07-03 -->

## 模型标识

| 字段 | 值 |
|------|-----|
| model_id | Qwen3.6-27B (Qwen3.5 architecture family) |
| architectures | Qwen3_5ForConditionalGeneration |
| model_type | dense_hybrid (linear_attention + full_attention) |
| closest_known | Qwen3-8B (dense, GQA) |

## 架构参数

| 参数 | 值 | 来源 |
|------|-----|------|
| num_hidden_layers | 64 | config.json |
| hidden_size | 5120 | config.json |
| intermediate_size | 17408 | config.json |
| num_attention_heads | 24 | config.json |
| num_key_value_heads | 4 | config.json |
| head_dim | 256 | config.json |
| vocab_size | 248320 | config.json |
| max_position_embeddings | 262144 | config.json |
| rms_norm_eps | 1e-06 | config.json |
| rope_theta | 10000000 | config.json↔rope_parameters |
| partial_rotary_factor | 0.25 | config.json↔rope_parameters |
| mrope_section | [11, 11, 10] | config.json↔rope_parameters |
| full_attention_interval | 4 | config.json |
| linear_key_head_dim | 128 | config.json |
| linear_num_key_heads | 16 | config.json |
| linear_value_head_dim | 128 | config.json |
| linear_num_value_heads | 48 | config.json |
| linear_conv_kernel_dim | 4 | config.json |
| attn_output_gate | true | config.json |
| mtp_num_hidden_layers | 1 | config.json |
| weight_key_prefix | model.language_model.* | 不可从 config.json 推断 |
| config_nesting | text_config {...} | 不可从 config.json 推断 |

## 与 Qwen3-8B 的关键差异

### 1. 混合层调度 (Hybrid Layer Dispatch)

Qwen3-8B 所有 36 层使用标准 GQA attention。Qwen3.6-27B 的 64 层中:
- **48 层 linear_attention** (GatedDeltaNet)
- **16 层 full_attention** (FullAttention + MRoPE)

模式: 每 4 层一组，3 linear + 1 full，重复 16 次 (共 64 层)。
`config.layer_types[layer_idx]` 为 `"linear_attention"` 或 `"full_attention"`。

### 2. RMSNorm 三变体

| 变体 | Weight init | 公式 | 使用位置 |
|------|------------|------|---------|
| RMSNorm (标准) | ones | `x * rsqrt(variance+eps) * w` | FullAttention Q/K norms (per-head) |
| Qwen3_5RMSNorm | **zeros** | `x * rsqrt(variance+eps) * (1+w)` | decoder input_layernorm, post_attention_layernorm, final norm |
| Qwen3_5RMSNormGated | ones | `x * rsqrt(variance+eps) * w * silu(gate)` | GatedDeltaNet 内部 output norm |

**关键陷阱**: Qwen3_5RMSNorm 的 weight 初始化为 **zeros** 而非 ones。用标准 RMSNorm 加载此模型权重会产生全局缩放错误——zeros 权重下标准公式 `x*rsqrt*w` 输出全零，而正确公式 `x*rsqrt*(1+w)` 在 w=0 时输出 `x*rsqrt`（即普通 normalize，无额外缩放）。

Qwen3_5RMSNorm 的 `_effective_weight()` 方法返回 `1.0 + self.weight`，供 vLLM kernel (`rms_norm` / `fused_add_rms_norm`) 兼容调用。

### 3. GatedDeltaNet 线性注意力 (48 层)

```
输入 hidden_states [B, T, 5120]
  → in_proj_qkv: ColumnParallel → [B, T, 2560] (Q:512 + K:512 + V:1536 per-rank, TP=4)
  → causal conv1d (depthwise, kernel=4, groups=C, SiLU)
  → split Q [B,T,4,128], K [B,T,4,128], V [B,T,12,128]
  → in_proj_a (Replicate): 48 per-head dt projections
  → in_proj_b (Replicate): 48 per-head beta projections
  → gate g = -exp(A_log) * softplus(a + dt_bias)
  → beta = sigmoid(b)
  → Q/K L2 normalize → repeat_interleave(×3) to match V heads
  → Recurrent Gated Delta Rule (逐 token 循环)
  → in_proj_z (ColumnParallel): output gate projection
  → Qwen3_5RMSNormGated(core_out, z): x * rsqrt * weight * silu(gate)
  → out_proj (RowParallel)
```

**Recurrence formula 精确定义**:
```
对每个 token t:
  S_t = S_{t-1} * exp(g_t)        # state decay (Dk × Dv)
  kv_mem = sum(S_t * k_t, dim=-2)  # [H, Dv]
  delta = (v_t - kv_mem) * beta_t  # [H, Dv]
  S_t = S_t + k_t^T @ delta        # state update (outer product)
  o_t = S_t * q_t, sum over dim=-2 # [H, Dv] ← 使用 POST-update 的 state
```

**State 维度**: `[B, H, Dk, Dv] = [B, 48/tp, 128, 128]`。Dk 在前，Dv 在后（与某些实现相反）。

**conv_state 缓存**: 必须缓存最近 4 个 token 的**原始 QKV 投影值**（conv1d 之前的值），用于 decode 阶段提供因果卷积的左侧上下文。Zero-padding 会导致 ~3000× 的数值误差。

### 4. FullAttention (16 层)

与 Qwen3-8B 的 FullAttention 的主要差异:

| 维度 | Qwen3-8B | Qwen3.6-27B |
|------|---------|-------------|
| QKV 投影 | 合并 qkv_proj (1个GEMM) | 分离 q_proj, k_proj, v_proj (3个GEMM) |
| Q 投影输出 | `q_size` | **`2 * q_size`** (Q + output gate) |
| 位置编码 | Standard RoPE (NeoX), rotary_dim=head_dim=128 | **MRoPE** (interleaved, 3D), partial_rotary_factor=0.25 |
| Q/K Norm | RMSNorm (standard) | RMSNorm (standard, 但注意与 Qwen3_5RMSNorm 区分) |
| Output gate | 无 | `attn_out * sigmoid(gate)`, gate 来自 q_proj 后半部分 |
| KV cache | Paged (block_size=256) | Paged (block_size=256) — 仅 full_attention 层有 |

**Output gate 实现要点**: `q_proj` 输出维度为 `2 * q_size`，用 `torch.chunk(2, dim=-1)` 分离 Q 和 gate。Attention output 乘以 `sigmoid(gate)` 后送入 `o_proj`。

### 5. MRoPE 配置

- `partial_rotary_factor=0.25` → rotary_dim = 64 (head_dim=256 × 0.25)
- `mrope_section=[11, 11, 10]` → 三个位置维度的 cos 维度，全维度为 [22, 22, 20]，总和 64
- `mrope_interleaved=true`
- 对纯文本推理 (1D position_ids): 三个 MRoPE 段合并为单一 cos/sin 缓存，退化为标准 RoPE
- NeoX-style rotation (前后半分 rotate_half)，与 Qwen3 系列一致
- **陷阱**: mrope_section 必须从 config.rope_parameters 读取，不能用 `head_dim/3` 推断

### 6. 权重前缀

所有 HF key 前缀为 `model.language_model.*`（而非 Qwen3-8B 的 `model.*`）。这是因为 `Qwen3_5ForConditionalGeneration` 包装了 `Qwen3_5Model` 的 `.language_model` 子模块。

**完整 key pattern**:
```
model.language_model.embed_tokens.weight
model.language_model.layers.{i}.input_layernorm.weight
model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.language_model.layers.{i}.self_attn.{q,k}_norm.weight
model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight
model.language_model.layers.{i}.linear_attn.in_proj_z.weight
model.language_model.layers.{i}.linear_attn.in_proj_a.weight
model.language_model.layers.{i}.linear_attn.in_proj_b.weight
model.language_model.layers.{i}.linear_attn.A_log
model.language_model.layers.{i}.linear_attn.dt_bias
model.language_model.layers.{i}.linear_attn.conv1d.weight
model.language_model.layers.{i}.linear_attn.norm.weight
model.language_model.layers.{i}.linear_attn.out_proj.weight
model.language_model.layers.{i}.post_attention_layernorm.weight
model.language_model.layers.{i}.mlp.gate_proj.weight
model.language_model.layers.{i}.mlp.up_proj.weight
model.language_model.layers.{i}.mlp.down_proj.weight
model.language_model.norm.weight
lm_head.weight
mtp.fc.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
mtp.layers.0.*  (1 MTP decoder layer, always full_attention)
```

### 7. 已知陷阱: decode 路径中 conv_state 不缓存

GatedDeltaNet 的 `forward_decode` 依赖 `self._conv_state` 提供因果卷积的历史上下文。如果 prefill 阶段未正确设置 `_conv_state`（例如直接从 decode 开始而无 prefill），卷积的左侧填充会用 zero-padding，导致输出严重错误（误差 ~3000×）。

**正确流程**: prefill → 自动缓存 `_conv_state` → decode 从缓存读取。跳过 prefill 直接 decode 是已知的失败路径。

## 实现架构 (已工作代码)

代码位置: `engine/models/qwen3_5.py`

```
Qwen3_5ForCausalLMTP
├── embed_tokens: VocabParallelEmbedding
├── layers: ModuleList[QwenHybridDecoderLayerTP × 64]
│   └── QwenHybridDecoderLayerTP
│       ├── layer_type ∈ {"linear_attention", "full_attention"}
│       ├── [linear_attn]: QwenGatedDeltaNetTP
│       │   ├── in_proj_qkv: _DeltaNetQKVColumnParallelWeight (head-aware sharding)
│       │   ├── conv1d_weight: [C, 1, 4] (channel-matched sharding)
│       │   ├── in_proj_z: ColumnParallelLinear
│       │   ├── in_proj_a, in_proj_b: nn.Linear (Replicate)
│       │   ├── A_log, dt_bias: per v_head parameters
│       │   ├── norm: Qwen3_5RMSNormGated
│       │   └── out_proj: RowParallelLinear
│       ├── [self_attn]: QwenFullAttentionTP
│       │   ├── q_proj: ColumnParallelLinear (2× q_size)
│       │   ├── k_proj, v_proj: ColumnParallelLinear
│       │   ├── o_proj: RowParallelLinear
│       │   ├── q_norm, k_norm: RMSNorm (standard, per-head)
│       │   └── paged KV cache (block_size=256)
│       ├── mlp: QwenMLPTP (gate_up → silu_and_mul → down)
│       ├── input_layernorm: Qwen3_5RMSNorm
│       └── post_attention_layernorm: Qwen3_5RMSNorm
├── norm: Qwen3_5RMSNorm
├── lm_head: ParallelLMHead
└── mtp: QwenMTPHead (optional)
```

## 验证方法

- 使用 `test_phase7_qwen_tp_config.py` 验证 TP config 读取
- 使用 `test_phase7_hf_key_mapping.py` 验证权重 key 映射
- 使用 `test_phase5_attention_init.py` 验证 attention 层初始化
- 使用 GatedDeltaNet prefill/decode 单 token 与 HF 参考对齐检查
- 不开源验证通过标准: GatedDeltaNet forward 数值误差 < 1e-3, FullAttention forward 误差 < 1e-5
