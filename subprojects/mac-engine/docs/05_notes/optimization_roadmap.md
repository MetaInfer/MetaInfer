# 优化路线图

> 基于 2026-06-02 profiling 数据生成。

## 当前状态

| 指标 | 自研引擎 (Phase 1) | MLX 基线 (vllm-metal) | 差距 |
|------|-------------------|----------------------|------|
| 吞吐 | 9.3 tok/s | 17.8 tok/s | 1.9x |
| TPOT | 107.5 ms/tok | 55.6 ms/tok | 1.9x |
| TTFT | 134.0 ms | 237.8 ms | 0.6x (更快) |
| 内存 | 5.9 GB | 7.3 GB | 0.8x |

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

**原理**: 将整个 decode step（logits → argmax → next_input → model forward）编译为静态计算图，消除每步重建开销。

```python
@mx.compile
def compiled_decode_step(logits, cache):
    next_id = mx.argmax(logits[0, -1, :], axis=-1)
    next_input = mx.expand_dims(next_id, axis=(0, 1))
    return model(next_input, cache=cache), next_id
```

**实施**: 修改 `engine_v1.py` decode 循环，用 `mx.compile` 包裹热点。

**风险**: 中等。compile 首次调用慢，需固定 tensor 形状。Qwen3 的 cache 机制在编译模式下可能不兼容（cache 作为 mutable 状态）。

**回退条件**: 如编译后 decode 提升 <30%，放弃 compile 改为手动 batch 优化。

**参考**: 上游 CUDA 分支 P2 阶段使用 `torch.compile` 获 50% 提升 (8.49→12.75 tok/s)。MLX compile 语义类似但限于固定形状图。

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

### O5: dtype 优化 (float16 全程)

**预期收益**: +10-20%

**原理**: 当前权重为 float32（bfloat16→float32 转换）。MLX 支持混合精度，保持 float16 可减少内存带宽压力。

```python
# weights.py: 不转换 bf16→f32
weights[key] = mx.array(t.numpy().view(np.uint16), dtype=mx.bfloat16)
```

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

| ID | 优化项 | 预期收益 | 难度 | 风险 | 优先级 |
|----|--------|---------|------|------|--------|
| O1 | mx.compile decode | +80-150% | 中 | 中 | ⭐⭐⭐ |
| O2 | 减少 Python 开销 | +10-20% | 低 | 低 | ⭐⭐⭐ |
| O3 | 固定形状 KVCache | +5-10% | 低 | 低 | ⭐⭐ |
| O4 | 消除掩码重建 | +5-10% | 低 | 极低 | ⭐⭐ |
| O5 | float16 全程 | +10-20% | 中 | 中 | ⭐⭐ |
| O6 | Custom Metal kernel | +20-40% | 高 | 高 | ⭐ |
| O7 | 投机解码 | +20-50% | 高 | 高 | ⭐ |

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
