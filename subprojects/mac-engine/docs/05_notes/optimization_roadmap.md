# 优化路线图

> 基于 2026-06-02 profiling 数据生成。

## 当前状态

| 指标 | 自研引擎 (Phase 1+) | MLX 基线 (vllm-metal / mlx-lm) | 差距 |
|------|-------------------|-------------------------------|------|
| 吞吐 | 9.0 tok/s | 17.8 / 17.6 tok/s | 1.96x |
| TPOT | 110.1 ms/tok | 55.6 ms/tok | 1.98x |
| TTFT | 254.7 ms | 237.8 ms | 1.07x |
| 内存 | 2.7 GB | 7.3 / 15.8 GB | 0.37x |

> 内存差距因 dtype 不同：自研 float32 (2.7GB RSS)，mlx_lm bfloat16 (15.8GB)。

### 已尝试优化的结果

| 优化 | 预期收益 | 实际 | 结论 |
|------|---------|------|------|
| mx.compile sample (E04) | +10-15% | 中性 (9.3→9.0) | 采样开销微不足道 |
| mx.async_eval stream pipeline (E05) | +55-80% | 中性 (9.0→9.0) | MLX 0.31.2 + 自定义模型上无加速 |
| bfloat16 dtype | +50-100% | **退化** (9.0→3.4) | 自定义模型架构 bf16 有未知 bug |

## 差距拆解

### 算子级 profiling 结果

| 算子 | 批量 (100x) | 单次 (Python loop) | 说明 |
|------|------------|-------------------|------|
| 模型 prefill (L=11) | 43.7 ms | - | 含图构建 |
| 模型 decode (1 token, cached) | **0.8 ms** | **~266 ms** | **333x gap!** |
| attention_decode | 0.017 ms | - | 单层计时不准确 |

### 耗时拆解 (256 token generate, 32.7s 总耗时)

```
┌─────────────────────────────────────────────┐
│ 模型计算 (prefill + decode x255)   ~1%  │
│ MLX 图构建开销                     ~50%  │
│ Python 循环 (argmax/item/array)     ~30%  │
│ KVCache 分配 + 掩码生成            ~10%  │
│ tokenizer + yield                  ~5%   │
│ 采样 + 其他                         ~4%   │
└─────────────────────────────────────────────┘
```

**核心瓶颈**: MLX lazy evaluation 导致每步 `model()` 调用都重建计算图。批量 100x 测试显示纯计算仅 0.8ms/tok，但单步 Python 循环需 266ms/tok——333x 差距全部来自图构建和 Python 开销。

---

## Phase 1: 低垂果实 (预期 → ~60% baseline, 10.7 tok/s)

### O1: `mx.compile` 包裹 decode 步骤 ⭐⭐⭐

**预期收益**: +80-150%

**状态**: ⚠️ 已尝试 (E04) — `mx.compile` 因 mutable KV Cache 无法直接包裹 `model()`，仅编译采样函数（中性结果，9.3→9.0 tok/s）。完整 decode step 编译需重新设计 cache 接口。

**风险**: 高。`mx.compile` 不支持 mutable 状态（KV Cache 就地更新），需改为 functional cache 或支持 compile 的 cache 设计。

---

### O2: 减少 Python 循环开销

**预期收益**: +10-20%

**手段**:
1. 消除 `.item()` 调用：用 `mx.argmax` 结果直接作为下一个 forward 的输入（避免 Python int 往返）
2. 批量 decode：一次 forward 预测多个 token（speculative decoding 的简化版）
3. 预分配 next_input 数组：复用内存

```python
# Before: per-token Python overhead
next_id = int(mx.argmax(logits).item())  # sync + int conversion
next_input = mx.array([[next_id]])        # new allocation

# After: stay in MX graph
next_logits = logits[0, -1, :]
next_id = mx.argmax(next_logits, axis=-1)
# Use expand_dims instead of new allocation
```

**风险**: 低。仅改变 Python 调用模式。

---

## Phase 2: 结构优化 (预期 → ~75% baseline, 13.3 tok/s)

### O3: 固定形状 KVCache (预分配)

**预期收益**: +5-10%

**原理**: 当前 KVCache 按需增长（step=256），首次增长触发内存分配 + 拷贝。固定形状可消除此开销。

```python
# Phase 1: 动态增长
cache = KVCache()  # 按需分配

# Phase 2: 固定形状
cache = KVCache(max_len=2048, pre_allocated=True)  # 一次分配
```

**风险**: 低。内存占用增加有限（2048 × 0.04GB = 缓存自身不大）。

---

### O4: 消除逐层掩码重建

**预期收益**: +5-10%

**原理**: 当前 `Qwen3Model.__call__` 每层使用同一个 `mask` tensor。但 MX lazy eval 可能导致每层重建掩码。改为预计算并传递同一引用。

**实施**: 确保 mask 在进入层循环前完全 evaluated（`mx.eval(mask)`），消除 lazy rebuild。

**风险**: 极低。1 行改动。

---

### O5: dtype 优化 (float16 / bfloat16)

**预期收益**: +10-20%

**状态**: ⚠️ 已尝试 — bfloat16 导致吞吐从 9.0→3.4 tok/s（退化 2.6x）。mlx_lm 使用 bfloat16 获 17.6 tok/s，说明退化来自自定义模型架构的某个操作（可能是 nn.RMSNorm / causal mask dtype 不匹配 / attention 隐式 upcast）。需进一步 profiling 定位。

**风险**: 中。bfloat16 可能影响精度，需与 golden 逐 token 对比。

---

## Phase 3: 深度优化 (预期 → ~85%+ baseline, 15+ tok/s)

### O6: Custom Metal kernel (热点)

**预期收益**: +20-40%

**原理**: 用 Metal Shading Language 重写注意力等热点算子。MLX 的 `mx.fast.scaled_dot_product_attention` 已是优化过的 kernel，进一步优化空间主要在：
- 自定义 SDPA（融合 softmax + mask）
- Grouped-query attention 专用 kernel

**参考**: 上游 CUDA 分支用 Triton MLA Decode Kernel 获得 50%+ 提升。

**风险**: 高。Metal Shading Language 开发 + 调试成本大。

---

### O7: 投机解码 (Speculative Decoding)

**预期收益**: +20-50%（多并发场景）

**原理**: 用小 draft model 预测 N 个 token，主模型一次 forward 验证 N 个。在单请求场景收益有限。

**参考**: SGLang 的 Eagle 投机解码。

**风险**: 高。需要 draft model 加载 + 验证逻辑，复杂度大。

---

## 汇总

| ID | 优化项 | 预期收益 | 难度 | 风险 | 优先级 | 状态 |
|----|--------|---------|------|------|--------|------|
| O1 | mx.compile decode | +80-150% | 高 | 高 | ⭐⭐⭐ | ⚠️ 已尝试 (中性) |
| O2 | 减少 Python 开销 | +10-20% | 低 | 低 | ⭐⭐⭐ | ✅ 已完成 (E04) |
| O3 | 固定形状 KVCache | +5-10% | 低 | 低 | ⭐⭐ | ✅ 已完成 (E04) |
| O4 | 消除掩码重建 | +5-10% | 低 | 极低 | ⭐⭐ | ⬜ 待开始 |
| O5 | bfloat16 dtype | +10-20% | 中 | 中 | ⭐⭐ | ⚠️ 已尝试 (退化) |
| O6 | Custom Metal kernel | +20-40% | 高 | 高 | ⭐ | ⬜ 待开始 |
| O7 | 投机解码 | +20-50% | 高 | 高 | ⭐ | ⬜ 待开始 |
| — | mx.async_eval stream pipeline | +55-80% | 中 | 中 | ⭐⭐⭐ | ⚠️ 已尝试 (E05, 中性) |

### 推荐实施顺序

```
当前: 9.3 tok/s (52.2%)
  ↓ O1: mx.compile (+100%)  
O1: ~18.6 tok/s (104%)  ← 可能直接超越基线
  ↓ O2: Python 开销 (-15%)
  ~21.4 tok/s (120%)
```

**O1 是关键杠杆**。如果 mx.compile 成功，后续优化（O2-O5）的边际收益降低。建议优先验证 O1，根据结果决定是否继续 O2-O5。

---

日期: 2026-06-02
