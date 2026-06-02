---
name: infer-ref-bench
description: >
  对指定模型进行系统性推理基准测试（单次+并发），自动根据当前平台选择最优成熟
  推理框架，采集 TTFT、TPOT、吞吐、内存占用等核心指标，填入实验基线表作为 100%
  参考线。当用户说"跑基准测试"、"跑 ref bench"、"标准推理指标"、
  "能跑多快"、"参考线"、"baseline 填数据"时立即触发。
  平台适配：
    - macOS Apple Silicon → MLX / llama.cpp
    - Linux + NVIDIA GPU  → vLLM / SGLang
    - 其他               → llama.cpp
  注意：本 skill 跑的是成熟框架的官方用法，不是自研引擎，数据是后续开发的对照目标。
---

# Infer Ref-Bench: 平台自适应推理基准测试

根据当前平台自动选择最优成熟推理框架，评测模型性能作为 100% 参考线。

---

## 第零步：平台检测与框架选择

### 检测脚本

```bash
# OS
uname -s

# Apple Silicon
sysctl -n machdep.cpu.brand_string 2>/dev/null

# NVIDIA GPU
nvidia-smi -L 2>/dev/null
```

### 框架选择矩阵

| 平台 | 主选框架 | 备选框架 | 说明 |
|------|---------|---------|------|
| macOS + Apple Silicon | MLX (mlx-lm) | llama.cpp | MLX 为 Apple 官方，最优性能 |
| Linux + NVIDIA GPU | vLLM | SGLang | vLLM 生态最成熟，CUDA 性能天花板 |
| Linux + no GPU | llama.cpp | - | CPU-only 推理 |
| 其他 | llama.cpp | - | 跨平台通用方案 |

### 框架可用性检测

```bash
# MLX (Apple Silicon only)
python3 -c "import mlx_lm; print('mlx-lm=' + mlx_lm.__version__)" 2>/dev/null || echo "MLX_NOT_AVAILABLE"

# vLLM (Linux+NVIDIA only)
python3 -c "import vllm; print('vllm=' + vllm.__version__)" 2>/dev/null || echo "VLLM_NOT_AVAILABLE"

# SGLang (Linux+NVIDIA only)
python3 -c "import sglang; print('sglang=' + sglang.__version__)" 2>/dev/null || echo "SGLANG_NOT_AVAILABLE"

# llama.cpp (universal fallback)
which llama-cli 2>/dev/null || python3 -c "import llama_cpp; print('llama-cpp-python=' + llama_cpp.__version__)" 2>/dev/null || echo "LLAMACPP_NOT_AVAILABLE"
```

选择优先级：主选框架 → 备选框架 → 提示用户安装。

---

## 第一步：确认测试模型

从基线表中提取测试模型，或向用户确认。

**检查模型权重是否存在**：
```bash
find ~/.cache/huggingface/hub/ ~/.cache/modelscope/ -maxdepth 6 -name "*.safetensors" 2>/dev/null | head -5
find ~/.cache/huggingface/hub/ ~/.cache/modelscope/ -maxdepth 6 -name "*.gguf" 2>/dev/null | head -5
```

若未找到目标模型权重，提示用户下载：
- MLX: 使用 `mlx-lm` convert 或下载 mlx-community 版本
- vLLM/SGLang: 使用 HuggingFace/ModelScope 下载 safetensors
- llama.cpp: 使用 `llama.cpp` convert 或下载 GGUF

---

## 第二步：框架特定基准测试

根据第零步选择的框架，执行对应的测试脚本。

### 分支 A：MLX (macOS Apple Silicon)

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

import psutil
mem_gb = psutil.Process().memory_info().rss / 1024**3
print(f"memory_rss={mem_gb:.1f} GB")

del model, tokenizer
mx.metal.clear_cache()
```

### 分支 B：vLLM (Linux + NVIDIA GPU)

```python
import time
from vllm import LLM, SamplingParams

model_path = "<model_path>"
llm = LLM(model=model_path, trust_remote_code=True)

prompt = "Explain the key concepts of machine learning in detail."
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

# Warmup
_ = llm.generate([prompt], sampling_params)

# Benchmark
t0 = time.perf_counter()
outputs = llm.generate([prompt], sampling_params)
t1 = time.perf_counter()

prompt_len = len(outputs[0].prompt_token_ids)
output_len = len(outputs[0].outputs[0].token_ids)
elapsed = t1 - t0
throughput = output_len / elapsed

print(f"prompt_len={prompt_len}, output_len={output_len}")
print(f"elapsed={elapsed:.2f}s, throughput={throughput:.1f} tok/s")

import psutil
mem_gb = psutil.Process().memory_info().rss / 1024**3
print(f"memory_rss={mem_gb:.1f} GB")
```

### 分支 C：SGLang (Linux + NVIDIA GPU)

```python
import time
import sglang as sgl
from sglang import Runtime

model_path = "<model_path>"
runtime = Runtime(model_path=model_path)
runtime.start()

prompt = "Explain the key concepts of machine learning in detail."

# Warmup
_ = runtime.generate(prompt, {"max_new_tokens": 16, "temperature": 0.0})

# Benchmark
t0 = time.perf_counter()
output = runtime.generate(prompt, {"max_new_tokens": 256, "temperature": 0.0})
t1 = time.perf_counter()

elapsed = t1 - t0
# token count depends on SGLang's tokenizer integration
print(f"elapsed={elapsed:.2f}s")

runtime.shutdown()
```

### 分支 D：llama.cpp (通用回退)

```bash
# 使用 llama-cli 命令行
llama-cli -m <model.gguf> -p "Explain the key concepts of machine learning in detail." -n 256 --temp 0.0 -t 0
```

或用 llama-cpp-python：

```python
import time
from llama_cpp import Llama

model_path = "<model.gguf>"
llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8)

prompt = "Explain the key concepts of machine learning in detail."

# Warmup
_ = llm(prompt, max_tokens=16, temperature=0.0, echo=True)

# Benchmark
t0 = time.perf_counter()
output = llm(prompt, max_tokens=256, temperature=0.0, echo=True)
t1 = time.perf_counter()

prompt_len = len(llm.tokenize(prompt.encode()))
output_len = len(llm.tokenize(output['choices'][0]['text'].encode())) - prompt_len
elapsed = t1 - t0
throughput = output_len / elapsed if elapsed > 0 else 0

print(f"prompt_len={prompt_len}, output_len={output_len}")
print(f"elapsed={elapsed:.2f}s, throughput={throughput:.1f} tok/s")
```

---

## 第三步：采集 Golden Outputs（正确性基准）

> **这是最关键的一步。** 后续自研引擎的正确性验证依赖此步骤产出的参考输出。所有对比通过 OpenAI 兼容 API（`/v1/completions`）进行，确保接口一致性。

### 3.1 定义测试用例集

基于性能测试的 prompt 进行扩展，覆盖不同输入模式：

```python
# tests/test_cases.py
TEST_CASES = [
    {"id": "basic_en", "prompt": "Explain the key concepts of machine learning in detail.", "max_tokens": 256},
    {"id": "basic_zh", "prompt": "请用中文解释量子计算的基本原理。", "max_tokens": 256},
    {"id": "short_prompt", "prompt": "Hello", "max_tokens": 128},
    {"id": "code_gen", "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"", "max_tokens": 200},
    {"id": "long_context", "prompt": "The history of artificial intelligence begins with " * 50, "max_tokens": 128},
    {"id": "edge_empty", "prompt": "", "max_tokens": 64},  # 边界: 空 prompt
    {"id": "edge_special", "prompt": "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", "max_tokens": 128},
]
```

### 3.2 通过 OpenAI API 采集参考输出

启动参考框架的 server（OpenAI 兼容接口），然后采集输出：

```bash
# 启动参考 server
python -m mlx_lm.server --model <model_path> --port 8080 &
# 或 vLLM
vllm serve <model_path> --port 8080 &
```

```python
# scripts/collect_golden.py
"""采集参考框架的 golden outputs，保存到 tests/golden_outputs/"""
import json, time
from pathlib import Path
from openai import OpenAI

BASE_URL = "http://localhost:8080/v1"
OUTPUT_DIR = Path("tests/golden_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(base_url=BASE_URL, api_key="not-needed")

# 需要从 test_cases.py 导入
from test_cases import TEST_CASES

results = []
for tc in TEST_CASES:
    t0 = time.perf_counter()
    resp = client.completions.create(
        model="<model>",
        prompt=tc["prompt"],
        max_tokens=tc["max_tokens"],
        temperature=0.0,
        echo=False,
    )
    elapsed = time.perf_counter() - t0

    record = {
        "test_id": tc["id"],
        "prompt": tc["prompt"],
        "max_tokens": tc["max_tokens"],
        "output_text": resp.choices[0].text,
        "output_tokens": resp.usage.completion_tokens,
        "prompt_tokens": resp.usage.prompt_tokens,
        "finish_reason": resp.choices[0].finish_reason,
        "elapsed_s": round(elapsed, 3),
        "framework": "<mlx-lm>",
        "framework_version": "<0.31.3>",
        "model": "<Qwen3-8B>",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    results.append(record)
    print(f"[{tc['id']}] {record['output_tokens']} tokens, {elapsed:.2f}s")

# 写入 golden 文件
golden_file = OUTPUT_DIR / "golden_outputs.json"
with open(golden_file, "w", encoding="utf-8") as f:
    json.dump({
        "meta": {
            "framework": "<mlx-lm>",
            "framework_version": "<0.31.3>",
            "model": "<Qwen3-8B>",
            "temperature": 0.0,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": {  # 从基线表复制
                "chip": "Apple M5 Pro",
                "gpu_cores": 20,
                "memory_gb": 48,
            },
        },
        "test_cases": results,
    }, f, ensure_ascii=False, indent=2)

print(f"\nGolden outputs → {golden_file} ({len(results)} test cases)")
```

### 3.3 正确性验证脚本（供 infer-bench / infer-engine-build 复用）

```python
# scripts/verify_correctness.py
"""验证目标引擎输出与 golden outputs 是否一致。
通过 OpenAI-compatible API 调用两个 endpoint，对比 greedy decode 输出。
"""
import json, sys
from openai import OpenAI

REF_URL = "http://localhost:8080/v1"   # 参考引擎
TARGET_URL = "http://localhost:8081/v1"  # 自研引擎

ref_client = OpenAI(base_url=REF_URL, api_key="not-needed")
target_client = OpenAI(base_url=TARGET_URL, api_key="not-needed")

# 加载 golden outputs
golden = json.load(open("tests/golden_outputs/golden_outputs.json"))

passed, failed = 0, 0
for tc in golden["test_cases"]:
    target_resp = target_client.completions.create(
        model="<model>",
        prompt=tc["prompt"],
        max_tokens=tc["max_tokens"],
        temperature=0.0,
        echo=False,
    )
    target_text = target_resp.choices[0].text
    ref_text = tc["output_text"]

    if target_text == ref_text:
        passed += 1
        print(f"[PASS] {tc['test_id']}")
    else:
        failed += 1
        # 定位第一个差异位置
        for i, (a, b) in enumerate(zip(target_text, ref_text)):
            if a != b:
                ctx = max(0, i - 20)
                print(f"[FAIL] {tc['test_id']} @ pos {i}:")
                print(f"  ref:    ...{repr(ref_text[ctx:i+20])}...")
                print(f"  target: ...{repr(target_text[ctx:i+20])}...")
                break
        else:
            min_len = min(len(ref_text), len(target_text))
            print(f"[FAIL] {tc['test_id']}: length mismatch (ref={len(ref_text)}, target={len(target_text)})")
            if min_len > 0:
                print(f"  first {min_len} chars: {'MATCH' if ref_text[:min_len] == target_text[:min_len] else 'DIVERGE'}")

print(f"\nResult: {passed}/{passed+failed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
```

**验证标准**: temperature=0.0，所有测试用例 greedy decode 输出**逐字符完全一致**。任一用例不一致 → 整体 FAIL。

---

## 第四步：并发压测（可选）

### vLLM / SGLang (推荐，原生 HTTP API)

```bash
# vLLM server
vllm serve <model_path> --port 8080 &

# 或 SGLang
python -m sglang.launch_server --model-path <model_path> --port 8080 &

# 压测
python3 -c "
import asyncio, httpx, time

async def benchmark():
    async with httpx.AsyncClient(timeout=120) as client:
        t0 = time.perf_counter()
        tasks = [client.post('http://localhost:8080/v1/completions', json={
            'model': '<model>',
            'prompt': 'What is the capital of France?',
            'max_tokens': 128, 'temperature': 0.0
        }) for _ in range(8)]
        responses = await asyncio.gather(*tasks)
        t1 = time.perf_counter()
        print(f'concurrent=8, elapsed={t1-t0:.2f}s')
asyncio.run(benchmark())
"
```

### MLX server (macOS)

```bash
python -m mlx_lm.server --model <model_path> --port 8080 &
# 同上压测脚本
```

---

## 第五步：填入基线表

将采集的数据写入 `subprojects/<project>/docs/01_planning/experiment_baseline.md` 的基准实验部分。

**填入原则**：
- 保持与基线表原有列结构一致
- 标注使用的框架和版本
- 若某列无法填充，标记"-"
- 追加新行而非覆盖已有数据

---

## 第六步：输出摘要

```
✅ <framework> 基准测试完成

<model> (<framework> v<version>):
- 单次推理: 45.2 tok/s (256 tokens, 2.1 GB)
- 并发 x8: 120.5 tok/s (if applicable)
- Golden outputs: tests/golden_outputs/golden_outputs.json (7 test cases)

基线表已更新 → subprojects/<project>/docs/01_planning/experiment_baseline.md

下一步: /infer-engine-build 开始编写自研引擎 (目标 ≥70% = 31.6 tok/s)
```

---

## 关键约束

1. **temperature=0.0** — 确保 greedy decode，结果可复现。
2. **warmup 后再测** — 首次调用包含编译/初始化开销，不计入耗时。
3. **隔离测试** — 每次测试后清理缓存，避免串扰。
4. **使用相同 prompt** — 跨框架对比使用相同 prompt 文本。
5. **记录环境** — 测试期间关闭其他 GPU 进程，避免资源竞争。
6. **平台适应性** — 优先使用平台原生最优框架，不跨平台对比。
7. **Golden outputs 必须通过 OpenAI API 采集** — 保证自研引擎可以用同一协议验证，避免 tokenizer 实现差异干扰对比。
8. **Golden 文件一旦写入不可覆盖** — 如需更新 golden，先 rename 旧文件加上版本后缀，便于追溯。
