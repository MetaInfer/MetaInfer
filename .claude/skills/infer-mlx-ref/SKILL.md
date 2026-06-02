---
name: infer-mlx-ref
description: >
  使用 mlx-lm 标准 API 对指定模型进行系统性推理基准测试（单次+并发），
  采集 TTFT、TPOT、吞吐、内存占用等核心指标，填入实验基线表作为 100% 参考线。
  当用户说"跑 MLX 基准"、"MLX 参考效率"、"mlx-lm benchmark"、"标准推理指标"、
  "mlx 能跑多快"、"参考线"、"baseline 填数据"时立即触发。
  注意：本 skill 跑的是成熟框架的官方用法，不是自研引擎，数据是后续开发的对照目标。
---

# Infer-MLX-Ref: MLX 标准推理基准测试

对 mlx-lm 官方用法进行推理性能测试，产出本机环境下的"天花板"数据，作为后续自研引擎的 100% 参考线。

---

## 前提条件

- mlx-lm 已安装（`pip list | grep mlx-lm`）
- 实验基线表已存在（由 `infer-baseline` 创建）
- 目标模型权重已下载到本地

若缺失任一条件，停止并提示用户先完成前置步骤。

---

## 第一步：确认测试模型

从基线表中提取测试模型，或向用户确认。优先使用 ≤2B 参数的小模型（快速迭代验证）。

**检查模型是否可通过 mlx-lm 加载**：
```bash
python3 -c "
from mlx_lm import load
model, tokenizer = load('<model_path>')
print(f'Loaded: {model}')
print(f'Vocab: {tokenizer.vocab_size}')
del model, tokenizer
"
```

若加载失败，给出具体错误信息和建议。

---

## 第二步：单次推理基准

对每个测试模型，运行单次推理并采集指标。

### 测试脚本

```python
import time
import mlx.core as mx
from mlx_lm import load, generate

model_path = "<model_path>"
model, tokenizer = load(model_path)

prompt = "Explain the key concepts of machine learning in detail."
input_ids = tokenizer.encode(prompt)
prompt_len = len(input_ids)

# Warmup
_ = generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False)
mx.metal.clear_cache()

# Benchmark (temperature=0.0 for reproducibility)
t0 = time.perf_counter()
response = generate(
    model, tokenizer, prompt=prompt, max_tokens=256,
    temp=0.0, verbose=False
)
t1 = time.perf_counter()

output_ids = tokenizer.encode(response)
output_len = len(output_ids) - prompt_len  # approximate
elapsed = t1 - t0
throughput = output_len / elapsed

print(f"prompt_len={prompt_len}, output_len={output_len}")
print(f"elapsed={elapsed:.2f}s, throughput={throughput:.1f} tok/s")

# Memory
import psutil
mem_gb = psutil.Process().memory_info().rss / 1024**3
print(f"memory_rss={mem_gb:.1f} GB")

del model, tokenizer
mx.metal.clear_cache()
```

### 采集指标

| 指标 | 来源 | 说明 |
|------|------|------|
| prompt_len | `len(input_ids)` | prompt 的 token 数 |
| output_len | `len(output_ids) - prompt_len` | 生成的 token 数（近似） |
| elapsed | `t1 - t0` | 端到端耗时（含 tokenizer） |
| throughput | `output_len / elapsed` | 生成吞吐 (tok/s) |
| TTFT | 需单独计时第一个 token | Time-To-First-Token（可选，单次推理不硬要求） |
| memory_rss | `psutil` | 进程 RSS 内存 |

---

## 第三步：并发压测（可选，有 HTTP server 时）

如果 mlx-lm 提供了 server 模式，用 `httpx` 或 `wrk` 做并发压测：

```bash
# 启动 mlx-lm server
python -m mlx_lm.server --model <model_path> --port 8080 &

# 压测
python3 -c "
import asyncio, httpx, time

async def benchmark():
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = []
        t0 = time.perf_counter()
        for i in range(8):
            tasks.append(client.post('http://localhost:8080/v1/completions', json={
                'model': '<model>', 'prompt': 'Hello, how are you?',
                'max_tokens': 128, 'temperature': 0.0
            }))
        responses = await asyncio.gather(*tasks)
        t1 = time.perf_counter()
        total_tokens = sum(len(r.json()['choices'][0]['text'].split()) for r in responses)
        print(f'concurrent=8, elapsed={t1-t0:.2f}s, total_tokens={total_tokens}, throughput={total_tokens/(t1-t0):.1f} tok/s')

asyncio.run(benchmark())
"
```

若并发压测不适用（如 mlx-lm server 不可用），标注"N/A"并跳过。

---

## 第四步：填入基线表

将采集的数据写入 `docs/01_planning/experiment_baseline.md` 的 §2 基准实验部分。

**填入格式**：

```markdown
### 2.1 单次推理

| 模型 | 输入长度 | 输出长度 | 吞吐 (tok/s) | 总耗时 (s) | 内存 (GB) |
|------|---------|---------|-------------|-----------|-----------|
| Qwen2.5-0.5B | 15 | 256 | 45.2 | 5.66 | 2.1 |
```

**注意**：保持与基线表原有列结构一致。如果原表缺少某列，追加。如果原表有多余列且无法填充，标记"-"。

---

## 第五步：输出摘要

```
✅ MLX 基准测试完成

Qwen2.5-0.5B (mlx-lm):
- 单次推理: 45.2 tok/s (256 tokens, 2.1 GB)
- 并发 x8: N/A (server 不可用)

基线表已更新 → docs/01_planning/experiment_baseline.md

下一步: /infer-engine-build 开始编写自研引擎 (目标 ≥70% = 31.6 tok/s)
```

---

## 关键约束

1. **temperature=0.0** — 确保 greedy decode，结果可复现。
2. **warmup 后再测** — 首次调用包含 Metal shader 编译，不计入耗时。
3. **clear_cache 隔离** — 每次测试后 `mx.metal.clear_cache()` 避免串扰。
4. **使用相同 prompt** — 单次推理和多模型对比使用相同 prompt 文本。
5. **记录环境** — 测试期间关闭其他 GPU 进程，避免资源竞争。
