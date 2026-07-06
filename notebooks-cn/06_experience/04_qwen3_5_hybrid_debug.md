# Qwen3.5/3.6 Hybrid — 调试经验

<!-- source: evolution/evo-001, open_source_enabled=true, timestamp=2026-07-03 -->

## BUG-001: Qwen3_5RMSNorm 用标准 RMSNorm 加载 → 输出全零

**Symptom**: 模型加载后 forward 输出全零或 NaN，单层 norm 输出 value 全为 0。

**Root Cause**: Qwen3_5RMSNorm 的 HF 权重初始化为 **zeros**（不是 ones）。标准 RMSNorm 公式 `x * rsqrt * weight` 在 weight=0 时输出全零。正确公式为 `x * rsqrt * (1 + weight)`，invert 了 "weight 表示额外缩放" 的语义。

**Detection**: 检查 `input_layernorm.weight` 的统计量。若 min ≈ max ≈ 0（而非 ~1.0），则为 Qwen3_5RMSNorm 变体。

**Fix**: 实现独立的 `Qwen3_5RMSNorm` 类，forward 使用 `(1.0 + self.weight)` 而非裸 `self.weight`。

**Verification**: 对同一输入，HF Qwen3_5RMSNorm 与自定义实现输出误差 < 1e-7。

---

## BUG-002: GatedDeltaNet state 维度顺序错误 → 输出 NaN

**Symptom**: GatedDeltaNet 首次 recurrent step 后输出 NaN。

**Root Cause**: State 维度为 `[Dk, Dv] = [128, 128]`（Dk 在前），但实现时错误地使用了 `[Dv, Dk]`。outer product `k^T @ delta` 中 k 为 [Dk]，delta 为 [Dv]，结果应为 [Dk, Dv]。若 state 初始化为 [Dv, Dk]，outer product 维度不匹配，产生静默错误或 NaN。

**Detection**: 在 recurrence loop 的第一次迭代后检查 state 的 shape 和是否有 NaN。

**Fix**: 确保 state 初始化为 `torch.zeros(B, H, Dk, Dv)`，即 key_dim 在前。所有 einsum/outer product 操作与此顺序一致。

**Verification**: state shape assert `[B, H, Dk, Dv]` = `[B, H/tp, 128, 128]`。

---

## BUG-003: conv_state 缓存未初始化 → decode 输出崩溃

**Symptom**: Prefill 输出正确，但第一个 decode token 输出严重偏离（logit diff > 100）。错误逐 token 累积，导致完全乱码。

**Root Cause**: GatedDeltaNet 的 causal conv1d (kernel=4) 在 decode 阶段依赖 `_conv_state` 提供前 3 个 token 的原始 QKV 值作为左侧填充。若 `_conv_state is None`（prefill 未执行或缓存未保存），decode 使用 zero-padding，因果卷积输出完全错误。

**Detection**: 检查 decode 路径中 `self._conv_state is not None` 的分支。若进入 else (zero-pad) 分支且之前确实有 prefill，则为缓存丢失。

**Fix**: 
1. Prefill 的 forward() 结束时强制保存 `_conv_state = mixed_qkv_c[:, :, -k:].clone()`
2. Decode 的 forward_decode() 使用 `torch.cat([self._conv_state, current_qkv], dim=-1)` 拼接历史上下文
3. Decode 后更新 `_conv_state = combined[:, :, -k:].clone()`

**Verification**: 单 token prefill + 单 token decode 与两 token prefill (no decode) 的输出误差 < 1e-5。

---

## BUG-004: MRoPE cache 构建中使用错误的 mrope_section → RoPE 错位

**Symptom**: FullAttention 层输出与 HF 参考偏差较大 (~0.1)，但 Q/K norm 已对齐。

**Root Cause**: MRoPE 的 cos/sin cache 构建依赖 `mrope_section` 决定三个位置维度的频率分配。若使用 `head_dim/3` 推断（如 `[21, 21, 22]`）而非 config 中的实际值 `[11, 11, 10]`，每个维度分配的 rotary 维度不对，导致旋转角度完全错误。

**Detection**: 打印 `cfg.mrope_section` 并与 config.json↔rope_parameters 对比。不能用除法推断。

**Fix**: `mrope_section = rope_params.get("mrope_section", [])` 直接从 config 读取。cache 构建时每个 section 的 cos 维度 = mrope_section[i]，全维度 = 2 * cos 维度。

**Verification**: 比较自定义 MRoPE cache[0, 0, :] 与 HF 参考的 cos 值，误差 < 1e-7。

---

## BUG-005: FullAttention q_proj 忘记分离 gate → attention 输出错误

**Symptom**: FullAttention 层输出 token 分布异常，top-1 概率偏低。

**Root Cause**: `q_proj` 输出维度为 `2 * q_size`（Q + gate 各一半）。忘记用 `torch.chunk(2, dim=-1)` 分离，将整个输出当作 Q 使用，导致: (a) Q 维度错误，(b) output gate 未应用。

**Detection**: 检查 `q_proj` 的 `output_size_per_partition` 是否为 `2 * num_heads_per_rank * head_dim`。

**Fix**: 
```python
q, gate = torch.chunk(q_double.view(B, S, num_heads, head_dim * 2), 2, dim=-1)
# attention computation...
out = out * torch.sigmoid(gate.reshape(B, S, q_size))
```

**Verification**: gate 的 sigmoid 值应在 [0, 1] 内，输出不应全等于 attention output。

---

## BUG-006: 权重 key 前缀 model.language_model.* 匹配失败 → 静默跳过

**Symptom**: `load_weights()` 无报错但模型输出随机/全零，logits 异常。

**Root Cause**: Qwen3.5 的 HF key 前缀为 `model.language_model.layers.{i}.*`，而非 Qwen3-8B 的 `model.layers.{i}.*`。若正则 pattern 为 `model\.layers\.(\d+)\.(.+)`，所有层权重被静默跳过——keys() 循环中的 `continue` 使错误不可见。

**Detection**: 加载后检查 `self.layers[0].input_layernorm.weight` 是否仍为初始值 (zeros)。

**Fix**: 正则改为 `model\.language_model\.layers\.(\d+)\.(.+)`。对不匹配的 key 打印警告而非静默跳过。

**Verification**: 加载后 `input_layernorm.weight[0]` ≠ 0（Qwen3_5RMSNorm 的 HF 权重虽然有负值但非全零）。

---

## BUG-007: TP=4 下 GatedDeltaNet A_log/dt_bias 未切分 → size mismatch

**Symptom**: `A_log.data.copy_(tensor)` 报错 `size mismatch`。

**Root Cause**: `A_log` 的 TP 实现形状为 `[v_heads_per_rank] = [12]`，但 HF 权重为 `[num_v_heads] = [48]`。直接 copy 会 shape mismatch。

**Fix**: 手动切分: `a_local = tensor[tp.rank * 12 : (tp.rank + 1) * 12]`，然后 copy。

**Verification**: 每个 rank 的 A_log 值与 HF 全量的对应 slice 一致。

---

## 数值误差参考范围

基于 TP=4 bf16 推理的经验误差预期:

| 组件 | 期望误差 (vs HF) |
|------|-----------------|
| Qwen3_5RMSNorm | < 1e-7 |
| GatedDeltaNet (prefill, 单token) | < 1e-3 (recurrent path, fp32 accumulation) |
| GatedDeltaNet (decode, 单token) | < 1e-3 |
| FullAttention (prefill) | < 1e-5 |
| FullAttention (decode) | < 1e-5 |
| Full model forward (prompt) | < 1e-3 |
| Decoder layer (任意类型) | < 1e-4 |
