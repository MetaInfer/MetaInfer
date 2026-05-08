# Apple Silicon (MPS) 推理引擎优化实录

本文档记录了在 Apple Silicon MPS 后端上优化 LLM 推理引擎的全过程，包含成功的优化策略、失败尝试、以及平台特定限制。目标模型为 QwenPaw-Flash-2B（2.7B Dense，24 层混合注意力），decode 速度从 23 tok/s 优化到 47 tok/s。

---

## 一、性能优化时间线

| 阶段 | 优化项 | 速度提升 | 累计速度 |
|------|--------|---------|---------|
| 基线 | 正确实现，无优化 | — | 23.2 tok/s |
| 1 | torch.bmm 替代 broadcast+sum（recurrent kernel） | +20% | 27.8 tok/s |
| 2 | MLP gate_up_proj 合并（2 matmul → 1） | ~5% | ~29 tok/s |
| 3 | Linear attention fused in_proj（4 matmul → 1） | ~8% | ~32 tok/s |
| 4 | Full attention fused QKV（3 matmul → 1） | ~5% | ~34 tok/s |
| 5 | RoPE inv_freq 预计算 | ~3% | ~35 tok/s |
| 6 | A_log exp 缓存 | ~2% | ~36 tok/s |
| 7 | Decode 循环状态管理优化 | ~5% | ~38 tok/s |
| 8 | Single-token decode fast path | ~8% | ~41 tok/s |
| 9 | **Top-p 采样 top-k 预筛选** | **+64%** | **~47 tok/s** |
| 最终 | 全部优化叠加 | — | **45-51 tok/s（均值 47）** |

**最大单一优化**：采样 top-k 预筛选。采样时间从 12.68ms 降到 0.88ms（-93%）。根因是 248K 大词表上 `torch.sort` 的开销极大，先取 top-1024 候选再做 top-p 排序，避免对全词表排序。

---

## 二、成功的优化策略

### 2.1 融合投影（Fused Projection）

将多个独立线性层合并为单个大矩阵乘法，减少 kernel launch 开销和内存带宽消耗：

```python
# MLP: gate_proj + up_proj → gate_up_proj
gate_up = self.gate_up_proj(x)  # [B, H] → [B, 2*intermediate]
gate, up = gate_up.chunk(2, dim=-1)
output = self.down_proj(F.silu(gate) * up)

# Full attention: q_proj + k_proj + v_proj → qkv_proj
qkv = self.qkv_proj(x)  # [B, H] → [B, q_dim + 2*kv_dim]
q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

# Linear attention: in_proj_qkv + in_proj_z + in_proj_b + in_proj_a → in_proj
# [B, H] → [B, qkv_dim + z_dim + b_dim + a_dim]
```

**权重加载时合并**：`torch.cat([gate_w, up_w], dim=0)`，避免推理时拼接。

**MPS 上的效果**：MPS 对大矩阵乘法的利用效率高于多个小矩阵乘法。每次合并大约减少 1-2ms 的 kernel launch 开销。

### 2.2 torch.bmm 替代手动 broadcast

Gated DeltaNet 的 recurrent kernel 中，原来用手动 broadcast + sum 实现 batched 矩阵乘法：

```python
# 慢：手动 broadcast + element-wise multiply + sum
kv = (k.unsqueeze(-1) * S_state).sum(dim=-2)  # [B, H, D] × [B, H, D, V] → sum

# 快：bmm 一次调用
q_flat = q.reshape(B * H, 1, D)
S_flat = S_state.reshape(B * H, D, V_dim)
out_flat = torch.bmm(q_flat, S_flat)  # 单次 bmm
```

**效果**：decode 速度 +20%。bmm 是 BLAS 级别优化，比手动 broadcast + sum 高效得多。

### 2.3 Single-Token Decode Fast Path

Decode 阶段每次只处理 1 个 token（S=1）。通用的 recurrent kernel 包含循环、多维 indexing 和 reshape，对 S=1 来说全是浪费：

```python
def torch_recurrent_gated_delta_rule_single(q, k, v, ...):
    # S=1 特化：去掉所有循环和多维操作
    # 直接对 (B*H, D) 和 (B*H, D, V) 做 bmm
```

**效果**：~8% 提升。简化了 PyTorch 的 dispatch 和 shape 推导开销。

### 2.4 采样 Top-K 预筛选

**问题**：248K 词表上 top-p 采样需要对全词表排序（`torch.sort`），耗时 12.68ms，占总推理时间的 96%。

**解决方案**：先取 top-1024 候选，再在候选集内做 top-p：

```python
k = min(1024, vocab_size)
top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
sorted_logits, sorted_idx = torch.sort(top_k_logits, descending=True, dim=-1)
# ... top-p mask, multinomial on reduced set ...
# 反向映射：multinomial_idx → sorted_idx → top_k_idx → original vocab idx
```

**效果**：采样时间 12.68ms → 0.88ms（-93%），decode 速度 +64%。

**索引映射要点**：top-k 预筛选引入三层间接索引（原始 vocab → top-k 子集 → sorted 子集 → multinomial 采样），每层都必须正确反向映射。这是高频出错区域（参见 Bug #10）。

### 2.5 Decode 循环状态管理优化

原始实现每步调用 `states_to_kwargs()` 重建 kwargs 字典。优化后只构建一次，每步 in-place 更新变化的字段：

```python
kw_list = states_to_kwargs(states)  # 只构建一次
for step in range(max_tokens):
    dec_buf[0, 0] = seq.output_ids[-1]
    pos_buf[0, 0] = seq.total_tokens - 1
    for idx, s in enumerate(states):
        kw = kw_list[idx]
        kw["position_ids"] = pos_buf  # in-place 更新
        if s.layer_type == "full_attention":
            kw["cache_len"] = s.cache_len
        else:
            kw["recurrent_state"] = s.recurrent
            kw["conv_state"] = s.conv
```

### 2.6 其他微优化

- **RoPE inv_freq 预计算**：`1.0 / (theta ** (torch.arange(0, dim, 2) / dim))` 只计算一次，缓存为模型属性
- **A_log exp 缓存**：`exp(self.A_log.float())` 只计算一次，避免每步重复 exp
- **Prefill 512 tokens ~0.33s (1548 tok/s)**，prefill 路径不是瓶颈

---

## 三、失败尝试

### 3.1 RoPE 预计算查找表（MPS 负优化）

**尝试**：预计算完整 RoPE cos/sin 表，decode 时通过 tensor indexing 查找，替代每步重新计算。

**CPU 基准测试**：查找比计算快 107 倍。看起来应该很棒。

**MPS 实测**：decode 速度从 43.4 → 38.5 tok/s（-11%）。

**根因**：MPS 后端对 fancy indexing（`tensor[:, pos_tensor]`）的实现效率极低。每次索引操作都需要启动新的 Metal compute kernel，启动开销远大于直接计算（element-wise 乘法 + cos/sin）节省的时间。

**结论**：**在 MPS 上避免使用 tensor indexing，优先使用 element-wise 操作**。直接计算（compute-bound 的数学运算）可以被 MPS 的 compute graph 优化合并，而 indexing 操作会打断优化链。

### 3.2 torch.compile（MPS 不可用）

**尝试**：编译 decode 路径消除 Python 开销。尝试了 `aot_eager` 和 `inductor` 两个 backend。

**aot_eager 失败**：
- full attention 层的 KV cache in-place 赋值 `key_cache[:, cache_len:cache_len+S] = k` 不兼容图捕获
- linear attention 的 chunk kernel 中 `transpose/contiguous` 在 MPS 上触发 `weakref` 错误

**inductor 失败**：MPS 后端 `PythonDispatcher dispatch` 错误。

**结论**：PyTorch 2.11 的 `torch.compile` 对 MPS 后端支持不够成熟。CUDA 上的成熟优化（compile、GQA SDPA）在 MPS 上不可用。

### 3.3 Thinking Token 截断（模型行为 hack）

**尝试 1**：在 chat template 末尾注入 `<think\>\n\n</think\>\n\n` 让模型跳过思考。
**结果**：模型在 prompt 中看到 `<think\></think\>` 后确实跳过思考直接回答，但如果去掉这些 token，模型会自行生成 `<think\>` 进入 thinking mode。

**尝试 2**：在 decode 循环中过滤 `</think\>` 之前的所有 token。
**结果**：当模型直接跳过思考时（因为 prompt 中有 `<think\></think\>`），第一个 token 就不是 `</think\>`，过滤逻辑永远等不到标记。当去掉 prompt 中的标记时，thinking 内容消耗了 max_tokens 预算，最终输出为空。

**结论**：**不要试图 hack 模型的思考过程**。chat template 是模型训练时固定的输入格式，任何修改都会破坏模型的预期行为。

---

## 四、MPS 平台限制汇总

| 限制 | 影响 | 解决方案 |
|------|------|---------|
| 不支持 GQA in SDPA | `F.scaled_dot_product_attention` 要求 num_kv_heads == num_heads | 用 `repeat_interleave` 扩展 KV heads |
| `torch.empty()` 产生全零 | 初始化为 0 而非随机值 | 显式调用 `_init_weights()` |
| Fancy indexing 慢 | tensor indexing kernel launch 开销高 | 用 element-wise 操作替代 |
| `torch.compile` 不成熟 | 图捕获失败 | 保持 eager 模式 |
| float16 精度不足 | recurrent state 累积误差 | 线性注意力 recurrent state 使用 float32 |
| multinomial 采样需在 CPU | MPS multinomial 结果不正确 | logits 先 `.cpu()` 再采样 |

---

## 五、性能分解（最终状态）

**Decode 性能基准**（MPS, float16）：均值 ~47 tok/s

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Model forward | ~18ms | 84% |
| Sampling (top-k + top-p) | ~0.88ms | 4% |
| Token decode + overhead | ~2.5ms | 12% |

**Prefill 性能**：512 tokens ~0.33s (1548 tok/s)

### 模型前向内部时间分布（decode，单 token）

| 组件 | 估算占比 | 说明 |
|------|---------|------|
| 18 层 linear attention | ~45% | recurrent kernel bmm + fused in_proj |
| 6 层 full attention | ~20% | fused QKV + SDPA + KV cache 更新 |
| 24 层 MLP | ~25% | fused gate_up + down |
| Embed + Norm + lm_head | ~10% | 词嵌入、归一化、输出投影 |

---

## 六、通用优化方法论

从本次优化中提炼的方法论，适用于任何 LLM 推理引擎的性能调优：

### 6.1 先量化，再优化

1. **建立基线**：在未优化状态下测量 decode 速度
2. **profiling**：分别测量 forward、sampling、overhead 的耗时
3. **定位瓶颈**：哪个阶段占最多时间？是 compute-bound 还是 memory-bound？

### 6.2 按收益排序优化

1. **最大瓶颈优先**：采样（96% 时间）→ top-k 预筛选（-93%）
2. **低风险高收益**：融合投影（减少 matmul 次数）
3. **中等风险中等收益**：kernel 级优化（bmm、fast path）
4. **高风险低收益**：platform-specific 优化（可能负优化）

### 6.3 每步验证

1. **正确性**：每个优化后都跑冒烟测试，检查输出质量
2. **性能**：每个优化后都跑 benchmark，确认提升
3. **回退准备**：负优化立即回退，记录失败原因

### 6.4 平台特性验证

CPU 上的基准测试结果不能直接推广到 MPS/CUDA：
- **CPU 上的热点 ≠ MPS 上的热点**：fancy indexing 在 CPU 上很快（缓存友好），在 MPS 上很慢（kernel launch 开销）
- **CPU 上的优化 ≠ MPS 上的优化**：预计算查找表在 CPU 上 107x 加速，在 MPS 上 -11%
- **必须在目标平台上实测**
