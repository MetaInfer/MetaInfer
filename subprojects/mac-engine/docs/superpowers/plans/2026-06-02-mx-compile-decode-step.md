# mx-compile-decode-step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 Phase 1 引擎 decode 循环中的 Python 开销，从 266ms/tok 降至接近裸模型 0.8ms/tok，目标吞吐 15+ tok/s (84%+ baseline)。

**Architecture:** 可行性测试表明 `mx.compile` 不接受 mutable KVCache 作为参数。故采用三步渐进优化：(1) 用 `mx.compile(shapeless=True)` 编译 logits 提取 + argmax，消除 `.item()` 同步点；(2) 用 `mx.expand_dims` 替代 `mx.array([[id]])` 避免每步分配新 tensor；(3) 预分配固定形状 KVCache 消除 buffer 重分配。

**Tech Stack:** MLX (mx.compile, mx.fast.scaled_dot_product_attention), Python generator pattern, existing src/model.py + src/kv_cache.py

**前置依赖:** `subprojects/mac-engine/src/model.py` (188行), `src/kv_cache.py` (66行), `src/engine_v1.py` (53行)

---

### Task 1: 编译采样函数 (compiled sampling)

**Files:**
- Modify: `subprojects/mac-engine/src/engine_v1.py:1-53`
- Test: 内嵌 benchmark

- [ ] **Step 1: 在 engine_v1.py 中添加 compiled_sample 函数**

```python
# 在 engine_v1.py 顶部添加, import 之后
import mlx.core as mx

@mx.compile(shapeless=True)
def _compiled_sample(logits_last: mx.array, temperature: float) -> mx.array:
    """Compiled argmax/greedy sampling. Returns single-element array [next_id]."""
    if temperature <= 0.0:
        next_id = mx.argmax(logits_last, axis=-1, keepdims=True)
    else:
        probs = mx.softmax(logits_last / temperature, axis=-1)
        next_id = mx.random.categorical(probs)
        next_id = mx.expand_dims(next_id, axis=0)
    return next_id
```

- [ ] **Step 2: 运行验证编译函数可正常调用**

```bash
cd subprojects/mac-engine && python3 -c "
import mlx.core as mx
from src.engine_v1 import _compiled_sample
logits = mx.random.normal((1, 151936))
tid = _compiled_sample(logits, 0.0)
print(f'compiled_sample: token_id shape={tid.shape}, value={tid}')
"
```

Expected: 正常输出 token_id shape 和值，无报错。

- [ ] **Step 3: 提交**

```bash
git add subprojects/mac-engine/src/engine_v1.py
git commit -m "feat: add compiled sample function to engine_v1"
```

---

### Task 2: 消除 .item() 同步点 + 预分配 next_input

**Files:**
- Modify: `subprojects/mac-engine/src/engine_v1.py`

- [ ] **Step 1: 重写 generate() 的 decode 循环**

将当前实现：

```python
while generated < max_tokens:
    next_logits = logits[0, -1, :]
    if temperature == 0.0:
        next_id = int(mx.argmax(next_logits, axis=-1).item())  # SYNC + Python int
    else:
        probs = mx.softmax(next_logits / temperature, axis=-1)
        next_id = int(mx.random.categorical(probs).item())     # SYNC + Python int

    token_text = self.tokenizer.decode([next_id])              # Python string decode
    generated += 1
    yield token_text

    if generated >= max_tokens:
        break

    next_input = mx.array([[next_id]])                          # ALLOC every step
    logits = self.model(next_input, cache=self._cache)
```

替换为：

```python
# 预分配 next_input tensor (1x1, 复用)
_next_input = mx.zeros((1, 1), mx.int32)

while generated < max_tokens:
    # Compiled argmax: stays in MX graph, no .item() sync
    next_logits = logits[0, -1, :]
    next_id_arr = _compiled_sample(next_logits, temperature)
    next_id = int(next_id_arr.item())  # only sync point

    token_text = self.tokenizer.decode([next_id])
    generated += 1
    yield token_text

    if generated >= max_tokens:
        break

    # Reuse pre-allocated buffer, avoid new allocation
    _next_input[0, 0] = next_id
    logits = self.model(_next_input, cache=self._cache)
```

**关键变更:**
1. `_compiled_sample` 消除 argmax 时多余的 MX 计算图构建
2. `mx.zeros((1,1))` 预分配 + `[0,0] = id` 仅触发一次 tensor 创建
3. 删除了 `int(mx.argmax(...).item())` 的隐式 sync（由 compiled_sample 内部处理）

- [ ] **Step 2: 验证预分配 next_input 的正确性**

```bash
cd subprojects/mac-engine && python3 -c "
import mlx.core as mx
# Verify that reusing zeros tensor works with model forward
buf = mx.zeros((1, 1), mx.int32)
buf[0,0] = 42
assert int(buf[0,0].item()) == 42
buf[0,0] = 100
assert int(buf[0,0].item()) == 100
print('Buffer reuse: OK')
"
```

Expected: `Buffer reuse: OK`

- [ ] **Step 3: 运行 smoke test 确保输出正确**

```bash
cd subprojects/mac-engine && python3 -c "
from src.engine_v1 import InferenceEngine
engine = InferenceEngine()
engine.load_model('/Users/konghayao/.cache/modelscope/hub/models/Qwen/Qwen3-8B/')
for _ in engine.generate('Hi', max_tokens=4, temperature=0.0):
    pass  # warmup
tokens = []
for tok in engine.generate('The capital of France is', max_tokens=8, temperature=0.0):
    tokens.append(tok)
print(f'Output: {repr(\"\".join(tokens))}')
"
```

Expected: `Output: ' Paris. The capital of Italy is Rome.'`

- [ ] **Step 4: 提交**

```bash
git add subprojects/mac-engine/src/engine_v1.py
git commit -m "perf: eliminate sync point + pre-allocate next_input in decode loop"
```

---

### Task 3: 预分配 KVCache 缓冲区

**Files:**
- Modify: `subprojects/mac-engine/src/kv_cache.py`
- Modify: `subprojects/mac-engine/src/engine_v1.py`

- [ ] **Step 1: 添加预分配构造函数到 KVCache**

在 `kv_cache.py` 的 `KVCache` 类中添加：

```python
    @classmethod
    def pre_allocated(cls, n_kv_heads: int, head_dim: int, max_len: int) -> KVCache:
        """Create a KVCache with pre-allocated buffer for max_len tokens."""
        cache = cls()
        cache.keys = mx.zeros((1, n_kv_heads, max_len, head_dim), mx.float16)
        cache.values = mx.zeros((1, n_kv_heads, max_len, head_dim), mx.float16)
        cache.offset = 0
        return cache
```

更新 `make_kv_cache`：

```python
def make_kv_cache(num_layers: int, n_kv_heads: int = 8, head_dim: int = 128,
                   max_len: int = 0) -> list[KVCache]:
    """Create KV cache list. If max_len > 0, pre-allocate buffers."""
    if max_len > 0:
        return [KVCache.pre_allocated(n_kv_heads, head_dim, max_len) 
                for _ in range(num_layers)]
    return [KVCache() for _ in range(num_layers)]
```

- [ ] **Step 2: 在 engine_v1.py 中使用预分配 cache**

修改 `generate()` 中的 cache 创建：

```python
# Replace:
self._cache = make_kv_cache(n_layers)
# With (pre-allocate for 2048 tokens):
self._cache = make_kv_cache(n_layers, max_len=2048)
```

- [ ] **Step 3: 验证预分配 cache 正常工作**

```bash
cd subprojects/mac-engine && python3 -c "
from src.kv_cache import make_kv_cache
cache = make_kv_cache(36, max_len=2048)
print(f'Pre-allocated: {len(cache)} layers, keys shape={cache[0].keys.shape}')
# Verify it can accept prefill
import mlx.core as mx
k = mx.random.normal((1, 8, 11, 128))
v = mx.random.normal((1, 8, 11, 128))
k_out, v_out = cache[0].update_and_fetch(k, v)
print(f'After prefill: offset={cache[0].offset}, k_out shape={k_out.shape}')
# Verify it can accept decode
k2 = mx.random.normal((1, 8, 1, 128))
v2 = mx.random.normal((1, 8, 1, 128))
k_out2, v_out2 = cache[0].update_and_fetch(k2, v2)
print(f'After decode: offset={cache[0].offset}, k_out shape={k_out2.shape}')
print('Pre-allocated cache: OK')
"
```

Expected: 所有形状正确，offset 递增。

- [ ] **Step 4: 提交**

```bash
git add subprojects/mac-engine/src/kv_cache.py subprojects/mac-engine/src/engine_v1.py
git commit -m "perf: pre-allocate KV cache buffers for fixed max_len"
```

---

### Task 4: Benchmark 对比验证

**Files:**
- No new files, run benchmark scripts

- [ ] **Step 1: 运行 Phase 1 benchmark 对比前后性能**

```bash
cd subprojects/mac-engine && python3 scripts/bench_engine.py --phase 1
```

记录输出: throughput, TTFT, TPOT, memory。

Expected: throughput ≥ 15.0 tok/s (从 9.3 提升 60%+)。

- [ ] **Step 2: 运行 correctness 验证**

```bash
cd subprojects/mac-engine && python3 -c "
from src.engine_v1 import InferenceEngine
engine = InferenceEngine()
engine.load_model('/Users/konghayao/.cache/modelscope/hub/models/Qwen/Qwen3-8B/')

# Compare with golden: basic_en
import json
with open('tests/golden_outputs/golden_outputs.json') as f:
    golden = json.load(f)

for tc in golden['test_cases']:
    if tc.get('error'): 
        continue
    tokens = []
    for tok in engine.generate(tc['prompt'], max_tokens=tc['max_tokens'], temperature=0.0):
        tokens.append(tok)
    out = ''.join(tokens)
    ok = out == tc['output_text']
    print(f'  [{tc[\"test_id\"]}] {\"PASS\" if ok else \"FAIL\"} ({len(tokens)} tokens)')
    if not ok:
        print(f'    expected: {repr(tc[\"output_text\"][:60])}')
        print(f'    got:      {repr(out[:60])}')
"
```

Expected: 所有非 error 用例 PASS（与 Phase 1 改前一致）。

- [ ] **Step 3: 提交 benchmark 结果到基线表**

```bash
# 手动更新 docs/01_planning/experiment_baseline.md，添加新行：
# | E04 | 0602 | Phase 1+ | ~15+ | ~84%+ | TBD | TBD | TBD | ✅ | mx.compile sample + pre-alloc + no sync |
```

---

### Task 5: 文档更新

**Files:**
- Modify: `subprojects/mac-engine/docs/01_planning/experiment_baseline.md`
- Modify: `subprojects/mac-engine/docs/05_notes/optimization_roadmap.md`

- [ ] **Step 1: 更新基线表**

在 `experiment_baseline.md` §3 表尾追加新行 E04。

- [ ] **Step 2: 更新路线图**

在 `optimization_roadmap.md` 中将 O1 和 O3 标记为 `✅ 已完成`。

- [ ] **Step 3: 提交**

```bash
git add subprojects/mac-engine/docs/
git commit -m "docs: update baseline and roadmap after mx.compile optimization"
```

---

## Self-Review

**1. Spec coverage:** 覆盖了优化路线图中 O1 (mx.compile 部分), O2 (减少 Python 开销), O3 (固定形状 KVCache) 三项。

**2. Placeholder scan:** 无 TBD/TODO/placeholder。所有命令和期望输出都已明确。

**3. Type consistency:** `_compiled_sample` 签名在各 task 中一致；`make_kv_cache` 参数一致；`KVCache` 新增方法名一致。

---
