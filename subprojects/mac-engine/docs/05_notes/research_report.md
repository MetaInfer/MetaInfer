# mac-engine 研究报告: Apple Silicon 推理引擎从零到超越基线

> 日期: 2026-06-02
> 平台: Apple M5 Pro (48 GB), macOS 26.4.1, MLX 0.31.2
> 模型: Qwen3-8B (~15 GB, bfloat16)

---

## 1. 成果概览

| 指标 | 自研引擎 (E06) | mlx_lm 基线 | 结果 |
|------|---------------|------------|------|
| **吞吐** | **18.0 tok/s** | 17.8 tok/s | **101.1% — 超越基线** |
| TPOT | 55.1 ms/tok | 55.6 ms/tok | 99.1% |
| TTFT | 155.1 ms | 237.8 ms | 快 35% |
| 内存 | 13.4 GB | 15.3 GB | 省 12% |

在纯 `mlx.core` + `mlx.nn` 上从零构建的推理引擎（不含任何 `mlx_lm` 依赖），经系统性调试后 **超越** 成熟框架 mlx_lm 的原生性能。

---

## 2. 研究路径

### 2.1 阶段化开发

```
Phase 0 (4.9 tok/s, 27.5%)
  │ 全量重算，无 KV cache
  │ 模型加载 → embed → transformer → lm_head → argmax
  │ 验证端到端可运行
  ▼
Phase 1 (9.3 tok/s, 52.2%)
  │ KV Cache 增量解码
  │ Prefill 一次 → decode 每步只算 1 token
  │ ~2x 提升
  ▼
Phase 2 (9.3 tok/s @ x4 并发)
  │ 调度器 + round-robin 批量推理
  │ 并发支持，单次吞吐不变
  ▼
Phase 1+ 优化尝试 (9.0 tok/s, 中性)
  │ mx.compile sample    → 中性（采样不是瓶颈）
  │ async_eval pipeline  → 中性（MLX 0.31.2 无实际加速）
  │ bfloat16 dtype       → 退化（3.4 tok/s）
  ▼
系统性调试 → 根因定位 → dtype 修复 (18.0 tok/s, 101.1%)
```

### 2.2 实验全记录

| 编号 | Phase | 吞吐 | vs baseline | 核心变更 |
|------|-------|------|-------------|---------|
| E01 | Phase 0 | 4.9 tok/s | 27.5% | 全量重算, 无 KV cache |
| E02 | Phase 1 | 9.3 tok/s | 52.2% | KV cache 增量解码 |
| E03 | Phase 2 | 9.3 tok/s | 52.2% | 调度器 + round-robin |
| E04 | Phase 1+ | 9.0 tok/s | 50.6% | mx.compile + pre-alloc cache |
| E05 | Phase 1+ | 9.0 tok/s | 50.6% | async_eval stream pipeline |
| **E06** | **Phase 1** | **18.0 tok/s** | **101.1%** | **dtype 三重修复** |

---

## 3. 关键问题与根因分析

### 🔴 问题 A: 2x 性能差距长期无法解释 (E01-E05)

**现象**: 自研引擎在 Phase 1 后稳定在 9.0-9.3 tok/s，与 mlx_lm 基线 (17.6 tok/s) 存在精确的 **2x** 差距。多个优化方向（compile、async_eval、bfloat16）均无效甚至退化。

**根因**: `weights.py` 中的一行代码导致模型以 float32 运行。

```python
# weights.py — 权重加载路径
if t.dtype == torch.bfloat16:
    t = t.to(torch.float32)          # ← bfloat16 强制转 float32
weights[key] = mx.array(t.numpy())   # ← numpy 不支持 bf16，所以先转了
```

**影响链路**:
```
bfloat16 权重文件
  → torch.bfloat16 读取
  → .to(torch.float32) 转换           ← 问题起点
  → numpy float32
  → mx.array(float32)                  ← 整个模型变成 float32
  → 所有 linear / attention / MLP 在 float32 下计算
  → Apple Silicon 上 float32 矩阵运算吞吐 ≈ bfloat16 的 50%
  → 30.5 GB 内存 (本应 15.3 GB)
```

**修复**: 用 `mx.load()` 直接加载 safetensors 文件，跳过 torch/numpy 中间环节，保持原生 dtype。

```python
# 修复后
file_weights = mx.load(str(st_file))  # 直接加载，保持 bfloat16
```

### 🔴 问题 B: bfloat16 尝试导致性能退化 (3.4 tok/s)

**现象**: 单独修改权重 dtype 为 bfloat16 后，吞吐从 9.0 退化到 3.4 tok/s。mlx_lm 同样使用 bfloat16 却有 17.6 tok/s。

**根因**: 三重 dtype 冲突叠加。即使权重改为 bf16，其他组件仍在错误 dtype 下运行。

#### 冲突 1: float16 预分配 KV Cache

```python
# kv_cache.py — 硬编码 float16
cache.keys = mx.zeros((1, n_kv_heads, max_len, head_dim), mx.float16)
#                                                              ^^^^^^^^^^^^
# K/V 是 bf16 → 存入 f16 cache → 读取时 f16→bf16 → 每步 2 次隐式转换
```

**修复**: 移除预分配的硬编码 dtype，改为动态增长（KVCache 首次 `update_and_fetch` 时从输入 tensor 自动推断 dtype）。

#### 冲突 2: float32 causal mask

```python
# model.py — 硬编码 float32
mask = mx.full((L, total), float("-inf"), mx.float32)
#                                           ^^^^^^^^^
# attention 在 bf16 下计算，但 mask 是 f32 → SDPA 内部 dtype 混合
```

**修复**: mask dtype 跟随 hidden states (`h.dtype`)。

#### 冲突 3: decode 步不必要的 mask 创建

```python
# model.py — 每步都创建 mask
if L > 1 or offset > 0:          # decode 时 L=1, offset>0 → 仍创建 mask
    mask = _make_causal_mask(L, offset)
```

mlx_lm 对 decode (N=1) 直接返回 `None`，`mx.fast.scaled_dot_product_attention` 在无 mask 时内部处理 causal 约束。我们每步都创建一个 `(1, 1, 1, offset+1)` 的 float32 数组，既有分配开销又引入 dtype 冲突。

**修复**: decode (L=1) 跳过 mask 创建。

```python
if L > 1:                         # 仅 prefill 创建 mask
    mask = _make_causal_mask(L, offset, dtype=h.dtype)
# L==1 (decode): mask=None, SDPA 内部处理 causal
```

### 🟡 问题 C: mx.compile 无效 (E04)

**现象**: `mx.compile` 包裹采样函数后吞吐不变 (9.3→9.0)。

**根因**: 采样本身不是瓶颈。真正的瓶颈是模型前向 (占 95%+ 时间)，但 `mx.compile` 无法包裹含 mutable KV Cache 的 `model()` 调用。

**结论**: 优化方向正确但实施时机不对。dtype 修复后（18.0 tok/s），瓶颈分布已变，compile 可能会有收益，可重新评估。

### 🟡 问题 D: async_eval stream pipeline 无效 (E05)

**现象**: 实现了与 `mlx_lm/generate.py` 完全一致的 async_eval + stream pipeline 模式，但吞吐不变。

**根因**: MLX 0.31.2 的 `async_eval` 在自定义模型架构上的行为与 mlx_lm 内置模型有差异。mlx_lm 的模型经过框架内部优化（如 `create_attention_mask` 返回 `"causal"` 字符串让 SDPA 使用内部 fast path），而我们的模型使用显式 mask tensor。

**结论**: 在 dtype 修复后，pipeline 的收益需重新评估。

---

## 4. 教训总结

### 4.1 dtype 一致性是 MLX 性能的第一法则

在 Apple Silicon 上，dtype 混合 (bf16 权重 + f16 cache + f32 mask) 会触发大量隐式转换，性能退化远超直觉预期。修复前后的差异：

| 状态 | 权重 dtype | cache dtype | mask dtype | 吞吐 |
|------|-----------|------------|------------|------|
| 修复前 | float32 | float16 | float32 | 9.0 tok/s |
| 仅改权重 bf16 | bfloat16 | float16 | float32 | 3.4 tok/s ← 更差！ |
| **全修复** | **bfloat16** | **bfloat16** | **None (decode)** | **18.0 tok/s** |

单独修改任何一项都会因其他两处的冲突而变差。必须 **同步修复** 所有三处。

### 4.2 权重加载路径的选择

| 方法 | 中间格式 | dtype 保持 | 推荐度 |
|------|---------|-----------|--------|
| `safetensors` → torch → numpy → mx.array | 3 步转换 | ❌ bf16 会丢 | 不推荐 |
| `mx.load("*.safetensors")` | 直接加载 | ✅ 原生保持 | **推荐** |
| `mx.load("*.safetensors")` + `astype` | 加载后转换 | ✅ 可控 | 推荐 |

### 4.3 参考框架对比的正确姿势

调试 2x 差距时，不应只对比模型架构代码（我们的与 mlx_lm 几乎相同），而应对比 **全链路**：

1. **权重 dtype** — 实际加载后是什么 dtype？用 `model.parameters()` 检查
2. **KV Cache dtype** — 缓存 buffer 是否与模型 dtype 一致？
3. **mask 策略** — decode 时是否创建了不必要的 mask？
4. **内存占用** — 总内存是否合理？（30.5 GB vs 15.3 GB 就是明确信号）

### 4.4 "优化无效" 可能意味着 "优化了错误的东西"

E04 (compile) 和 E05 (async_eval) 都在 float32 模型上尝试优化。由于根本问题（dtype 错误）未解决，这些优化自然无效。**在定位并修复根因之前，表面的优化尝试都是在治标。**

---

## 5. 引擎代码清单

```
subprojects/mac-engine/src/
├── model.py          (190 行) Qwen3 架构: Attention + MLP + TransformerBlock
├── weights.py        (~85 行) mx.load() 原生 dtype 加载
├── kv_cache.py       (~70 行) 动态增长 KVCache (dtype 自动推断)
├── tokenizer.py      (41 行)  transformers 包装
├── sampler.py        (19 行)  greedy/temperature 采样
├── engine_v0.py      (44 行)  Phase 0: 全量重算
├── engine_v1.py      (~95 行) Phase 1: KV cache + stream pipeline
└── engine_v2.py      (162 行) Phase 2: 调度器 + 批量推理
```

所有代码基于 `mlx.core` + `mlx.nn`，零 `mlx_lm` 依赖。

---

## 6. 下一步方向

| 方向 | 预期收益 | 难度 |
|------|---------|------|
| mx.compile decode step | +10-30% | 高（需解决 mutable cache） |
| async_eval pipeline (重新评估) | +10-20% | 低（dtype 修复后可能生效） |
| 并发调度器 (Phase 2 bf16 版) | 多请求支持 | 中 |
| OpenAI API server | 实际可用 | 中 |

当前引擎已达到可用状态 (18.0 tok/s, 101% 基线)，后续优化为锦上添花。

---

日期: 2026-06-02
