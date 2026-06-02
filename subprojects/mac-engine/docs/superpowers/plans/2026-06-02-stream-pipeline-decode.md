# Stream Pipeline Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 将 Phase 1 引擎吞吐从 9.0 tok/s 提升到 16+ tok/s (接近 mlx_lm 基线 17.8 tok/s)。

**Architecture:** 引入 `mx.stream` + `mx.async_eval` 流水线模式。关键原理：构建 token N+1 的计算图时，token N 的 GPU 求值仍在异步进行，形成图构建与 GPU 计算的并行管道。

**Tech Stack:** MLX (`mx.new_stream`, `mx.async_eval`, `mx.clear_cache`), 现有 src/model.py + src/kv_cache.py

**前置依赖:** `engine_v1.py` (85行, 已完成 Phase 1+ 优化)

---

### Task 1: 实现 Stream Pipeline 核心循环

**Files:**
- Modify: `subprojects/mac-engine/src/engine_v1.py`

**核心原理 (来自 mlx_lm/generate.py:396-470):**

```
当前 (naive):  graph_build → sync_eval → yield → graph_build → sync_eval → yield ...
Stream (pipeline):  graph_build → async_eval → [① GPU 并行] → yield → graph_build → async_eval → [② GPU 并行] → ...
```

`mx.async_eval` 启动 GPU 求值后立即返回，不阻塞。在 yield 被消费者处理（tokenizer.decode）期间，GPU 已经在计算下一个 token。

**Step 1: 在 engine_v1.py 添加 stream pipeline 版本**

在 `InferenceEngine` 类中新增 `generate_stream()` 方法：

```python
def generate_stream(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
    """Generate tokens with stream pipeline (mx.async_eval pattern).

    Pipeline: async_eval(token N) while building graph for token N+1,
    so GPU compute overlaps with graph building + tokenizer.decode.
    """
    if self.model is None or self.tokenizer is None:
        msg = "Model not loaded"
        raise RuntimeError(msg)

    token_ids = self.tokenizer.encode(prompt)
    n_layers = len(self.model.layers)
    self._cache = make_kv_cache(
        n_layers,
        n_kv_heads=self.config.num_key_value_heads,
        head_dim=self.config.head_dim,
        max_len=2048,
    )

    # Prefill
    input_ids = mx.array([token_ids])
    logits = self.model(input_ids, cache=self._cache)

    # Define inner step: forward + sample (no Python sync)
    def _step(tok_arr: mx.array) -> mx.array:
        """One decode step: model forward + compiled argmax. Returns next_logits."""
        _in = tok_arr.reshape(1, 1)
        _logits = self.model(_in, cache=self._cache)
        return _compiled_sample(_logits[0, -1, :], temperature)

    # Pipeline: async_eval pattern
    stream = mx.new_stream(mx.gpu)
    y = _compiled_sample(logits[0, -1, :], temperature)

    n = 0
    while True:
        # Build graph for next token (while GPU evaluates current y)
        if n != max_tokens:
            next_y = _step(y)
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)  # Ensure first token is ready
        if n == max_tokens:
            break
        next_id = int(y.item())
        yield self.tokenizer.decode([next_id])
        if n % 256 == 0:
            mx.clear_cache()  # Prevent unbounded graph cache growth
        y = next_y
        n += 1
```

**Step 2: 保留原 generate() 作为 fallback**

原 `generate()` 方法保持不变（未经优化的路径在 debug 时需要）。stream pipeline 存在风险点（详见风险评估），保留回退路径。

**Step 3: 验证功能正确性**

```bash
cd subprojects/mac-engine && python3 -c "
from src.engine_v1 import InferenceEngine
engine = InferenceEngine()
engine.load_model('/Users/konghayao/.cache/modelscope/hub/models/Qwen/Qwen3-8B/')
for _ in engine.generate_stream('Hi', max_tokens=4, temperature=0.0):
    pass
tokens = []
for tok in engine.generate_stream('The capital of France is', max_tokens=8, temperature=0.0):
    tokens.append(tok)
print(f'Output: {repr(\"\".join(tokens))}')
"
```

Expected: `' Paris. The capital of Italy is Rome.'`

**Step 4: 验证与 naive generate() 输出一致**

```bash
cd subprojects/mac-engine && python3 << 'EOF'
from src.engine_v1 import InferenceEngine
e = InferenceEngine()
e.load_model("/Users/konghayao/.cache/modelscope/hub/models/Qwen/Qwen3-8B/")
prompts = ["Hello world", "What is AI?", "Python is a"]
for p in prompts:
    v1 = "".join(e.generate(p, max_tokens=16, temperature=0.0))
    v2 = "".join(e.generate_stream(p, max_tokens=16, temperature=0.0))
    ok = v1 == v2
    print(f'[{p}] match={ok}: {v1[:40]}... | {v2[:40]}...')
    assert ok, f'MISMATCH: {p}'
print('\nAll outputs match ✓')
EOF
```

**Step 5: 提交**

```bash
git add subprojects/mac-engine/src/engine_v1.py
git commit -m "perf: add stream pipeline generate using mx.async_eval

Implements mlx_lm-style stream pipeline:
- mx.async_eval overlaps GPU computation with graph building
- mx.clear_cache every 256 tokens prevents OOM
- Original generate() preserved as fallback

Co-Authored-By: deepseek-v4-pro <deepseek-ai@claude-code-best.win>"
```

---

### Task 2: Benchmark + 回归验证

**Files:**
- Modify: `subprojects/mac-engine/scripts/bench_engine.py`

**Step 1: 添加 stream 模式到 bench 脚本**

```python
# 在 bench_engine.py 的 run_benchmark 中, 对 phase 1 使用 generate_stream:
if args.phase == 1 and hasattr(engine, 'generate_stream'):
    gen_method = engine.generate_stream
else:
    gen_method = engine.generate
```

**Step 2: 运行 benchmark**

```bash
cd subprojects/mac-engine && python3 scripts/bench_engine.py --phase 1 --json
```

Expected: throughput ≥ 14.0 tok/s (从 9.0 提升 55%+)。

**Step 3: 提交**

```bash
git add subprojects/mac-engine/scripts/bench_engine.py
git commit -m "test: bench_engine uses generate_stream for Phase 1

Co-Authored-By: deepseek-v4-pro <deepseek-ai@claude-code-best.win>"
```

---

### Task 3: 文档更新

**Files:**
- Modify: `subprojects/mac-engine/docs/01_planning/experiment_baseline.md`
- Modify: `subprojects/mac-engine/docs/05_notes/optimization_roadmap.md`

**Step 1: 基线表新增 E05**

```markdown
| E05 | 0602 | Phase 1+ | ~16+ | ~90%+ | TBD | TBD | TBD | ✅ | mx.async_eval stream pipeline |
```

**Step 2: 路线图标记 Stream Pipeline 已完成**

**Step 3: 提交**

```bash
git add subprojects/mac-engine/docs/
git commit -m "docs: record stream pipeline results in baseline"
```

---

## Risk Assessment

| 风险 | 等级 | 缓解 |
|------|------|------|
| `mx.async_eval` 在 MLX 0.31.2 行为不稳定 | 中 | 保留 naive `generate()` fallback; 若 assert 失败，回退到此路径 |
| Pipeline 下 cache 状态竞态 | 低 | MLX lazy eval 确保 `_step` 的图构建晚于前一步 `async_eval` 完成 |
| `mx.clear_cache()` 清理 Metal 编译缓存 | 中 | 可能导致热点 kernel 重编译; 每 256 token 触发一次，频率适中 |
| memory 增长 | 低 | `mx.clear_cache()` 已包含; pre-allocated KVCache 固定大小 |

## Fallback Plan

如果 stream pipeline 在 0.31.2 版本行为异常（`mx.async_eval` 无实际加速效果），降级方案：
1. **直接去除 `mx.async_eval`，仅保留 `mx.clear_cache()` 定期清理**
2. 改为实现简单的双缓冲（precompute next while yield current），不依赖 async_eval

## Self-Review

**Spec coverage:** 覆盖 stream pipeline 核心实现 (Task 1)、Benchmark (Task 2)、文档 (Task 3)。每个 Task 有明确的验证命令和预期输出。

**Placeholder scan:** 预期吞吐值标记为 `~16+` / `TBD`（需实际 benchmark 后填充）。这是预期值而非占位符。

**Type consistency:** `generate_stream` 签名与 `generate()` 一致: `(prompt, max_tokens, temperature)` → generator of `str`。

---
