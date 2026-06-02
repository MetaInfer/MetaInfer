---
name: infer-engine-build
description: >
  基于 MLX Core API 编写精简推理引擎，目标效率 ≥70% mlx-lm 基线。
  从知识库 (notebooks/) 提取框架设计模式，严格遵循阶段化开发流程：
  最简可用 → KV Cache → 批量推理 → 性能微调。
  当用户说"写推理引擎"、"构建引擎"、"engine build"、"达到 70%"、
  "开始写引擎"、"engine from scratch"、"从零写推理"时立即触发。
  适用于 Apple Silicon 平台（MLX），也可适配其他框架后端。
---

# Infer-Engine-Build: 自研推理引擎构建

从 MLX Core API 出发，构建精简推理引擎。目标：**≥70% mlx-lm 基准吞吐**。

---

## 前置条件

1. 实验基线表已存在（`infer-baseline` → `docs/01_planning/experiment_baseline.md`）
2. MLX 基准数据已填入（`infer-mlx-ref` → 基线表 §2）
3. 目标模型权重可正常加载

---

## 核心原则

**不造轮子，不复制框架。** 从 MLX 最简 API 起步，需要什么加什么。每步验证，不做无证据的"优化"。

**参考知识库**：遇到设计决策时，查阅 `notebooks/` 中的对应文档：
- 框架设计: `01_framework_design/` （架构、调度器、KV Cache、内存池、Sampler）
- 实现模式: `06_implementation_patterns/` （推荐模式 + 反模式）
- 模型专项: `02_model_specifics/` （具体模型的 attention、MoE 等）

---

## 阶段化开发流程

### Phase 0: 最简可用（≥30% baseline）

**目标**：模型加载 + 单 token 生成，跑通端到端。

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load  # 仅用加载，不用 generate

# 最简循环
model, tokenizer = load(model_path)
prompt = "Hello"
ids = mx.array(tokenizer.encode(prompt))[None, :]  # [1, seq_len]

# 一次性 prefill（无 KV cache）
logits = model(ids)  # [1, seq_len, vocab_size]
next_token = mx.argmax(logits[0, -1, :])

print(tokenizer.decode([next_token.item()]))
```

**验证**：输出合理 token，无崩溃。记录到基线表 §3。

**不要做的事**：
- ❌ 写 KV cache（Phase 1 才加）
- ❌ 写调度器（Phase 2 才加）
- ❌ 写采样器（先用 argmax）
- ❌ 写自定义 attention kernel（先用 MLX 内置 SDPA）

---

### Phase 1: KV Cache 增量解码（≥50% baseline）

**目标**：prefill 一次存 KV，decode 每次只算一个 token。

参考：`notebooks/01_framework_design/03_kv_cache.md`

**关键实现**：
- 模型前向时传入 `cache` 参数，MLX 模型自带 `past_key_values` 支持
- Prefill: 完整序列前向 → 缓存 KV → 取最后 token logits
- Decode: 单 token 前向 → 读取缓存 → 追加新 KV → 取 logits

```python
# Prefill
logits, cache = model(input_ids)  # full sequence
next_token = mx.argmax(logits[:, -1, :])

# Decode step
logits, cache = model(mx.array([[next_token]]), cache=cache)
next_token = mx.argmax(logits[:, -1, :])
```

**验证**：与 Phase 0 同 prompt 输出完全一致（greedy decode），吞吐显著提升。

---

### Phase 2: 批量推理 + 简单调度（≥65% baseline）

**目标**：支持多个请求并发，prefill 优先调度 + 连续批处理。

参考：`notebooks/01_framework_design/02_scheduler.md`

**关键实现**：
- 请求队列：waiting → running → finished
- Prefill batch: 多个新请求一次前向
- Decode batch: 多个进行中请求一次前向（各自传入自己的 cache）

```python
# Batch decode
token_batch = mx.array([[s.last_token] for s in running_seqs])
cache_batch = [s.cache for s in running_seqs]  # or use MLX's cache format
logits, new_caches = model(token_batch, cache=cache_batch)
```

**验证**：2-4 并发，每个请求输出与串行完全一致。

---

### Phase 3: 性能微调（≥70% baseline）

**目标**：缩小与 mlx-lm 的差距到 30% 以内。

**优先检查项**（按投入产出比排序）：

1. **compile**：`mx.compile()` 包裹 decode step 函数
2. **dtype**：确认用 float16（不是 float32）
3. **unnecessary copies**：避免 numpy ↔ mlx array 转换
4. **batch padding**：不等长序列 padding 策略
5. **memory pool**：参考 `01_framework_design/06_memory_pool.md`

每改一个参数 → 跑一次 benchmark → 记录到基线表。只保留有提升的改动。

---

## 代码组织规范

引擎源码放在子项目的 `src/` 目录：

```
src/
├── __init__.py
├── engine.py        # 主入口：模型加载 + generate()
├── model_runner.py  # 前向封装：prefill / decode
├── scheduler.py     # 请求调度：waiting → batch → finished
├── sampler.py       # 采样：greedy / top-p / temperature
├── kv_cache.py      # KV cache 管理（复用 MLX 内置或自定义）
└── config.py        # 模型/引擎配置常量
```

每个文件不超过 200 行，保持精简。超过即拆分。

---

## 记录规范

每次 benchmark 之后，更新基线表 §3：

```markdown
| 实验编号 | 日期 | 吞吐 (tok/s) | vs baseline | 内存 (GB) | 备注 |
|---------|------|-------------|-------------|-----------|------|
| E01 | 0602 | 15.3 | 33.8% | 1.8 | Phase 0: 最简循环, 无 KV cache |
| E02 | 0602 | 24.1 | 53.3% | 2.0 | Phase 1: KV cache 增量解码 |
```

**重要**：每次实验用 git commit 保存代码，commit message 格式：
```
engine(phase<N>): <what changed> (<throughput> tok/s, <X>% baseline)
```

---

## 退出条件

- [ ] Phase 3 完成，吞吐 ≥70% baseline
- [ ] 所有 Phase 的输出正确性验证通过（与 Phase 0 逐 token 对比）
- [ ] 基线表 §3 有 ≥3 条实验记录

达成后，输出摘要：

```
✅ 自研引擎达到目标

Qwen2.5-0.5B:
- 自研: 32.1 tok/s (71.0% baseline)
- MLX:  45.2 tok/s (100%)
- 差距: 13.1 tok/s (29.0%)

Phase 总结: E01(15.3)→E02(24.1)→E03(32.1)
代码: src/ (engine.py, model_runner.py, scheduler.py, sampler.py, kv_cache.py)

下一步: /infer-optimize-plan 分析剩余 29% 差距，规划优化方向
```
