# mac-engine 优化空间调研报告

> 日期: 2026-06-02
> 当前状态: 18.0 tok/s (Qwen3-8B, bfloat16, Apple M5 Pro 48GB)
> 调研方法: 3 个并行 agent 分别调研 MLX 框架层、算法层、竞品对比

---

## 核心结论

**Decode 是 memory-bandwidth-bound**: 每个 token 需读取全部 ~15.3 GB 权重，实测带宽 ~293 GB/s，理论下限 ≈ 52 ms/tok (19.2 tok/s)。当前 55.6 ms/tok 已经非常接近理论上限。

**这意味着**:
- 减少"计算量"的优化（算子融合、attention 优化）收益极小——因为瓶颈不在计算
- 真正的突破点在 **减少权重读取次数**（投机解码、量化）和 **框架层开销消除**（wired_limit、async_eval）

### 瓶颈分布

```
┌──────────────────────────────────────────────────────────┐
│ 权重读取 (15.3 GB @ 293 GB/s)                   ~92%  │ ← 瓶颈
│ lm_head (4096×151936 matmul)                     ~5%   │
│ Attention (SDPA, GQA)                            ~0.5% │
│ RMSNorm × 108                                    ~0.1% │
│ RoPE × 72                                        ~0.5% │
│ 采样 (argmax 151936)                             ~0.5% │
│ Python 开销 (graph build, sync)                  ~1.5% │
└──────────────────────────────────────────────────────────┘
```

---

## 优先级排序

### P0: wired_limit — 一行代码的潜在 10-50% 提升

**原理**: `mx.set_wired_limit()` 告诉 macOS 将指定大小内存保持常驻（不可 page out），避免 GPU 访问时的 page fault 延迟。

**现状**: mlx_lm 在 `stream_generate` 和 `BatchGenerator` 中都设置了 wired_limit，**我们的引擎完全没用**。

```python
# mlx_lm/generate.py:229-266
model_bytes = tree_reduce(lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0)
max_rec_size = mx.device_info()["max_recommended_working_set_size"]
old_limit = mx.set_wired_limit(max_rec_size)
```

**实测数据**: M5 Pro 的 `max_recommended_working_set_size` = 40.2 GB。模型 15 GB + KV cache ~0.3 GB，远小于 40 GB，理论上不会 page out。但如果系统内存压力大（其他进程占用），page out 会造成偶发延迟抖动。

**实施**: 1 行代码
```python
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
```

**预期收益**: +0-50%（如果当前无 page out 则 0%，如果有则立竿见影）
**风险**: 低。wired_limit 只是建议值，macOS 可能忽略
**参考**: `mlx_lm/generate.py:229-266,714`

---

### P1: 4-bit 量化推理 — 2-4x 吞吐提升

**原理**: 将 bf16 权重量化为 4-bit，权重体积缩小 4x（15 GB → ~4 GB），读取时间等比缩短。Apple Silicon GPU 有原生 int4 矩阵乘支持。

**竞品数据**:
- llama.cpp + GGUF Q4_K_M: M4 Max 上 Qwen3-8B 约 76 tok/s
- mlx-lm + 4-bit 量化: 类似性能，MLX 原生支持 `mx.quantize` / `mx.quantized_matmul`
- mlx-community 已有量化版 Qwen3 模型可直接下载

**技术路径**:
1. **加载预量化模型**: 使用 mlx-community 的 4-bit 量化版，或用 `mlx_lm.convert` 量化
2. **自研路径**: 用 `mx.quantize(weights, group_size=64, bits=4)` 量化，推理时用 `mx.quantized_matmul`
3. **精度**: 4-bit 量化典型 perplexity 增加 <0.5%，对实际输出质量影响很小

**实施难度**: 中。需要改造 `weights.py` 支持量化权重加载，改造 `model.py` 的 `nn.Linear` 为 `mx.quantized_matmul`

**预期收益**: 2-4x（18 tok/s → ~50-70 tok/s）
**风险**: 精度损失可控，但需要验证 Qwen3 特有的 QK-Norm + RoPE 在 4-bit 下是否稳定
**参考**: MLX 量化 API, llama.cpp GGUF 性能数据

---

### P2: async_eval pipeline 重评估 — 在 bf16 下可能生效

**原理**: `mx.async_eval` 启动 GPU 计算后立即返回，Python 线程继续构建下一步计算图，实现 GPU-CPU 流水线。

**现状**: E05 实验在 float32 下测试，结果中性 (9.0→9.0)。但 dtype 修复后 (18.0 tok/s)，瓶颈分布已变——GPU 计算占比更高，async pipeline 的 overlap 收益可能显现。

**mlx_lm 的完整模式** (`generate.py:396-470`):
```python
generation_stream = mx.new_thread_local_stream(mx.default_device())

y, logprobs = _step(prompt)
mx.async_eval(y, logprobs)

while True:
    next_y, next_lp = _step(y)        # 构建下一步图
    mx.async_eval(next_y, next_lp)     # 异步启动 GPU
    yield y.item(), logprobs           # Python 处理期间 GPU 已在计算
    y, logprobs = next_y, next_lp
```

**我们的代码**: `engine_v1.py:47-96` 已实现 `generate_stream()`，完全复制了 mlx_lm 的模式。

**实施**: 零成本，已有代码，只需重新 bench
**预期收益**: +5-15%
**风险**: 中。MLX 0.31.2 的 async_eval 标记为 "experimental"
**参考**: `engine_v1.py:47-96`, `mlx_lm/generate.py:396-470`

---

### P3: "causal" 字符串 mask — 免费的 SDPA fast path

**原理**: `mx.fast.scaled_dot_product_attention` 原生支持 `mask="causal"` 字符串参数，这是一个 Metal kernel fast path，不需要创建 mask tensor。

**现状**: 我们的 prefill 创建显式 mask 数组（`_make_causal_mask`），mlx_lm 对 prefill 返回 `"causal"` 字符串。

```python
# mlx_lm/models/base.py:45-55
def create_attention_mask(h, cache=None, ...):
    if N == 1: return None           # decode: 无 mask
    return "causal"                   # prefill: 字符串 fast path!

# 我们的 model.py — 创建显式数组
mask = _make_causal_mask(L, 0, dtype=h.dtype)  # prefill 仍用显式 mask
```

**实施**: 修改 `Qwen3Model.__call__`，prefill 时传 `"causal"` 字符串替代显式 mask
```python
elif L > 1:
    mask = "causal"  # 让 SDPA 用内部 fast path
```

**预期收益**: +2-5%（仅影响 prefill，decode 已优化）
**风险**: 极低。SDPA 官方支持此参数
**参考**: `mlx_lm/models/base.py:45-55`, MLX SDPA API 文档

---

### P4: 投机解码 (Speculative Decoding) — 突破带宽瓶颈的算法

**原理**: 用小模型 (draft) 快速生成 N 个候选 token，大模型一次 forward 验证 N 个。如果 accept rate = 70%，spec 5 tokens → 平均每步产出 4.5 tokens，等效吞吐 4.5x。

**关键数据**:
- Qwen3-0.6B (~1.2 GB) 是理想的 draft model，共享 tokenizer
- 48 GB 内存完全装得下 8B + 0.6B = ~16.5 GB
- Draft forward: ~4 ms (小模型)
- Verify forward: ~56 ms (大模型 prefill-style)
- 假设 accept rate 70%, spec 5: 等效 4.5/60 = **75 tok/s**

**实施难度**: 高。需要:
1. 加载并管理两个模型
2. 实现 draft → verify 循环
3. Verify forward 需支持多 token prefill (当前只支持单 token decode)
4. Accept/reject 逻辑 + KV cache 回滚

**预期收益**: +30-80%（理论，需实测验证）
**风险**: 高。实现复杂，accept rate 不确定
**参考**: `notebooks/05_non_core_features/07_speculative_decoding.md`

**入门替代 — n-gram 投机**:
- 无需额外模型，用 n-gram 匹配生成候选
- Accept rate 20-40%，但 draft 成本几乎为零
- 预期收益: +5-15%，实施难度: 低

---

### P5: Batched Decode (Continuous Batching) — server 场景核心

**原理**: 多个请求共享一次权重读取。Decode 瓶颈是 15 GB 权重读取，batch=4 仍然只读一次 → 理论上 4x 吞吐。

**竞品数据**:
- vllm-mlx: 16 并发下聚合吞吐 4.3x
- mlx_lm BatchGenerator: 原生支持 continuous batching

**关键限制**:
- 单请求延迟不变（甚至略微增加因 batch padding）
- 需要 MLX batched decode 的实测验证（graph compilation 开销可能随 batch 增加）
- Phase 2 引擎 (`engine_v2.py`) 已有 round-robin 调度器，但未测 batch 性能

**实施难度**: 中。需要改造 KVCache 支持 batch 维度、实现 batched forward
**预期收益**: +50-200% 吞吐（server 场景）
**风险**: 中。MLX batched decode 性能未验证
**参考**: `engine_v2.py`, mlx_lm `BatchGenerator`

---

## 不建议投入的方向

| 方向 | 不推荐原因 |
|------|-----------|
| **算子融合** (RMSNorm+残差, SwiGLU, QK-Norm+RoPE) | 实测各算子总耗时 <1% decode 时间，融合收益 <0.5% |
| **KV Cache 量化** (int8/int4) | 2048 seq KV 仅 288 MB，量化后带宽节省 <0.5 ms/tok (<1%) |
| **Paged KV Cache** | 单请求连续 KV cache 已最优，paging 增加开销 |
| **Sliding Window Attention** | Attention 仅占 0.5% decode 时间，窗口限制收益微乎其微 |
| **自定义 Metal SDPA kernel** | MLX 内置 SDPA 已是优化过的 Metal kernel，进一步提升空间极小 |
| **mx.compile 完整 decode step** | 实测 compile 模型 forward 无显著收益 (55.2 vs 55.9 ms)。MLX lazy eval 已经很高效，且 compile 与 mutable KVCache 不兼容 |

---

## 汇总路线图

```
当前: 18.0 tok/s (101% baseline)
  │
  ├─ P0: wired_limit (+0-50%)      ← 1 行代码，立即测试
  │   → 18-27 tok/s
  │
  ├─ P2: async_eval 重评估 (+5-15%)  ← 已有代码，重新 bench
  │   → 19-31 tok/s
  │
  ├─ P3: "causal" 字符串 mask (+2-5%)  ← prefill 优化
  │   → 20-33 tok/s
  │
  ├─ P1: 4-bit 量化 (+200-400%)     ← 最大单步收益
  │   → 50-70 tok/s
  │
  └─ P4: 投机解码 (+30-80%)         ← 突破带宽瓶颈
      → 24-58 tok/s (bf16 路径)
```

**推荐实施顺序**: P0 (wired_limit) → P3 (causal mask) → P2 (async_eval) → P1 (量化) → P4 (投机解码)

P0-P3 是零成本或极低成本的"低垂果实"，可立即测试。P1 和 P4 需要较多工程投入。

---

日期: 2026-06-02
