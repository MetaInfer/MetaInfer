---
name: infer-optimize-plan
description: >
  分析自研引擎与成熟框架之间的性能差距，定位瓶颈并规划优化方向。
  产出优先级排序的优化路线图，标注每项优化的预期收益、实施难度和风险。
  当用户说"优化方向"、"差距分析"、"下一步优化"、"optimization plan"、
  "怎么提升"、"瓶颈在哪"、"还能怎么优化"时立即触发。
  适用于引擎开发完成目标效率后的持续改进阶段。
---

# Infer-Optimize-Plan: 优化方向发掘与规划

自研引擎达到 ≥70% baseline 后，分析剩余的 30% 差距从哪来，制定可执行的优化路线图。

---

## 前置条件

- 自研引擎可运行，有基准数据
- 基准数据完整
- 基线表 §3 有 ≥3 条实验记录（能看到性能演进趋势）

---

## 第一步：差距拆解

将总差距拆解到具体维度，逐项分析。

### 1.1 算子级对比

对引擎的每个关键算子和 MLX 对等实现进行微量计时：

```python
import time, mlx.core as mx

def time_op(op_fn, *args, warmup=5, repeats=50):
    for _ in range(warmup):
        op_fn(*args)
    mx.metal.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        op_fn(*args)
    mx.metal.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1000  # ms

# Example: compare attention implementations
```

输出：
```
| 算子 | 自研 (ms) | MLX (ms) | 差距 | 说明 |
|------|----------|----------|------|------|
| attention_prefill | 2.3 | 1.8 | +28% | SDPA 实现差异 |
| attention_decode | 0.15 | 0.12 | +25% | cache 访问模式 |
| ffn | 0.8 | 0.75 | +7% | 差距小 |
| token_embed | 0.05 | 0.04 | +25% | 可能 dtype 不一致 |
```

### 1.2 框架开销拆解

区分"模型计算"和"框架开销"：

```python
import time

# 纯模型前向耗时
t0 = time.perf_counter()
logits = model(input_ids)  # 无 KV cache, batch=1
t1 = time.perf_counter()
model_time = t1 - t0

# 引擎完整 step 耗时
t0 = time.perf_counter()
next_token = engine.step()  # 含调度、采样、tokenize
t1 = time.perf_counter()
engine_time = t1 - t0

framework_overhead = engine_time - model_time
```

输出：
```
| 类别 | 耗时 (ms) | 占比 |
|------|----------|------|
| 模型前向 | 18.5 | 82% |
| 调度开销 | 1.2 | 5% |
| 采样 | 0.8 | 3% |
| tokenizer | 1.5 | 7% |
| 其他 | 0.5 | 2% |
```

### 1.3 内存分析

```bash
# 运行时内存占用
python3 -c "
import psutil, mlx.core as mx
# ... load model, run inference ...
print(f'RSS={psutil.Process().memory_info().rss/1024**3:.1f}GB')
print(f'Metal active={mx.metal.get_active_memory()/1024**3:.1f}GB')
print(f'Metal peak={mx.metal.get_peak_memory()/1024**3:.1f}GB')
print(f'Cache memory={mx.metal.get_cache_memory()/1024**3:.1f}GB')
"
```

---

## 第二步：优化方向分类

将所有可能的优化手段归为四类，按投入产出比排序：

### Tier 1: 低投入高回报（P0，应优先做）

| 方向 | 典型手段 | 预期收益 | 难度 | 风险 |
|------|---------|---------|------|------|
| dtype 对齐 | float16 全程，消除隐式 float32 | +10-20% | 低 | 精度损失 |
| compile | `mx.compile()` 包裹 decode step | +15-30% | 低 | 首次调用慢 |
| 减少转换 | 消除 np↔mx 不必要拷贝 | +5-10% | 低 | 无 |

### Tier 2: 中投入中回报（P1，Tier 1 不够时做）

| 方向 | 典型手段 | 预期收益 | 难度 | 风险 |
|------|---------|---------|------|------|
| 算子融合 | RMSNorm + residual、SwiGLU 融合 | +10-15% | 中 | 精度验证 |
| Custom Metal kernel | 替代 Python 实现的热点算子 | +20-40% | 高 | 开发成本 |
| 内存池 | Paged KV cache、prefix caching | +10-20% | 中 | 复杂度 |
| 异步调度 | GPU 计算与 CPU 调度流水线化 | +5-15% | 中 | 时序 bug |

### Tier 3: 高投入低回报（P2，有余力才做）

| 方向 | 典型手段 | 预期收益 | 难度 | 风险 |
|------|---------|---------|------|------|
| 量化 | 4/8-bit 权重量化 | 内存减半，速度+10-20% | 中 | 精度损失 |
| 投机解码 | Draft + verify 流水线 | +20-50%（高并发） | 高 | 实现复杂 |
| PD 分离 | Prefill/Decode 分离部署 | +30-50%（高并发） | 高 | 架构变更 |

### Tier 4: 平台特化（视情况）

| 方向 | 典型手段 |
|------|---------|
| ANE 卸载 | 将部分计算卸载到 Apple Neural Engine |
| GPU 亲和性 | 绑定特定 GPU core 组 |
| 动态电压频率 | 利用 M 系列芯片的功耗管理策略 |

---

## 第三步：生成优化路线图

基于差距拆解结果，从 Tier 列表中选取匹配的优化手段，生成路线图。

**路线图格式**：

```markdown
## 优化路线图

### 当前状态
- 自研引擎: 32.1 tok/s (71.0% baseline)
- MLX 基线: 45.2 tok/s
- 目标: 逐步逼近 90%+ baseline

### Phase 1: 低垂果实（预期 → 80% baseline）

| ID | 优化项 | 预期提升 | 实施 |
|----|--------|---------|------|
| O1 | mx.compile 包裹 decode | +20% | 1 行代码 |
| O2 | 消除 float64 隐式转换 | +10% | 检查 dtype |

### Phase 2: 结构优化（预期 → 90% baseline）

| ID | 优化项 | 预期提升 | 实施 |
|----|--------|---------|------|
| O3 | 算子融合 (RMSNorm+残差) | +12% | 自定义 MLX 层 |
| O4 | 内存池 + Paged KV Cache | +10% | 参考 notebooks |

### Phase 3: 深度优化（预期 → 95%+ baseline）

| ID | 优化项 | 预期提升 | 实施 |
|----|--------|---------|------|
| O5 | Custom Metal kernel (热点) | +25% | Metal Shading Language |
| O6 | 投机解码 | +30% | 参考 SGLang 实现 |

### 不建议的方向

- XXX: 原因...
```

---

## 第四步：输出与存档

将优化路线图写入 `docs/05_notes/optimization_roadmap.md`，同步更新基线表 §4。

**终端摘要**：

```
✅ 优化路线图已生成

当前: 32.1 tok/s (71.0% baseline)
差距拆解: 模型=82%, 框架开销=11%, tokenizer=7%

优先实施 (Phase 1):
- O1: mx.compile → 预期 +20% → ~38.5 tok/s (85%)
- O2: dtype 对齐 → 预期 +10% → ~42.4 tok/s (94%)

路线图 → docs/05_notes/optimization_roadmap.md
```

---

## 关键约束

1. **量化差距，不猜测**：每个结论必须有 benchmark 数据支撑。
2. **优先级 = 收益/投入**：先做改动小、提升大的，不是先做"看起来高级"的。
3. **单变量实验**：一次只改一个东西，测了再改下一个。
4. **回退策略**：每个优化标注回退条件（"如果提升<5%则放弃"）。
5. **不要为了优化牺牲正确性**：所有优化后必须通过 greedy decode 逐 token 对比验证。
