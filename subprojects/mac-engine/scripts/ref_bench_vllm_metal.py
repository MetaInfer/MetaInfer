#!/usr/bin/env python3
"""vLLM-Metal 参考基准测试脚本。

按 infer-ref-bench skill 流程，对 Qwen3-8B 进行系统性基准测试：
1. 单次推理 (TTFT, TPOT, 吞吐, 内存)
2. 并发压测
3. Golden Outputs 采集

平台: Apple M5 Pro / 48GB / macOS 26.4.1
框架: vllm-metal v0.1.0 (MLX backend)
模型: Qwen/Qwen3-8B (safetensors)

用法:
    python ref_bench_vllm_metal.py

输出:
    - 终端打印摘要
    - tests/golden_outputs/golden_outputs.json
    - 返回值 (JSON) 供上层脚本填入基线表
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import mlx.core as mx
import psutil
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

# --- 配置 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
GOLDEN_DIR = TESTS_DIR / "golden_outputs"

MODEL_PATH = os.environ.get(
    "REF_BENCH_MODEL_PATH",
    str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B/"),
)
MODEL_NAME = "Qwen/Qwen3-8B"
FRAMEWORK_NAME = "vllm-metal"
FRAMEWORK_VERSION = "0.1.0"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8099
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# --- 测试用例 ---
TEST_CASES = [
    {"id": "basic_en", "prompt": "Explain the key concepts of machine learning in detail.", "max_tokens": 256},
    {"id": "basic_zh", "prompt": "请用中文解释量子计算的基本原理。", "max_tokens": 256},
    {"id": "short_prompt", "prompt": "Hello", "max_tokens": 128},
    {"id": "code_gen", "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"", "max_tokens": 200},
    {"id": "long_context", "prompt": "The history of artificial intelligence begins with " * 50, "max_tokens": 128},
    {"id": "edge_empty", "prompt": "", "max_tokens": 64},
    {
        "id": "edge_special",
        "prompt": "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        "max_tokens": 128,
    },
]

# 主要 benchmark prompt (用于性能测试)
BENCHMARK_PROMPT = "Explain the key concepts of machine learning in detail."
BENCHMARK_MAX_TOKENS = 256


# --- 工具函数 ---
def wait_for_server(url: str, timeout: float = 120.0) -> float:
    """等待 server 就绪，返回等待耗时秒数。"""
    import urllib.request

    t0 = time.perf_counter()
    deadline = t0 + timeout
    while time.perf_counter() < deadline:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=2)
            if resp.status == 200:
                return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server did not become ready within {timeout}s")


def start_server(model_path: str) -> subprocess.Popen:
    """启动 vllm-metal server 并返回进程句柄。"""
    env = os.environ.copy()
    env["VLLM_METAL_DEBUG"] = "0"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm_metal.server",
            "--model", model_path,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--log-level", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """停止 vllm-metal server。"""
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def rss_memory_gb() -> float:
    """当前进程 RSS 内存 (GB)。"""
    return psutil.Process().memory_info().rss / (1024**3)


# --- 第 1 步：单次推理 (直接调用 MLX，获取 TTFT/TPOT) ---
def run_single_direct_benchmark(model_path: str) -> dict[str, Any]:
    """使用 mlx_lm 直接进行单次推理，采集 TTFT 和 TPOT。"""
    print("\n[1/3] 单次推理基准 (mlx_lm direct, TTFT/TPOT)...")

    model, tokenizer = mlx_load(model_path, tokenizer_config={"trust_remote_code": True})

    prompt = BENCHMARK_PROMPT
    input_ids = tokenizer.encode(prompt)
    prompt_len = len(input_ids)

    # Warmup
    sampler = make_sampler(temp=0.0)
    for _ in stream_generate(model, tokenizer, prompt=prompt, max_tokens=16, sampler=sampler):
        pass
    mx.metal.clear_cache()

    # Benchmark: stream_generate 可以获取逐 token
    sampler = make_sampler(temp=0.0)
    ttft = None
    total_time = 0.0
    output_len = 0

    t_start = time.perf_counter()
    for i, response in enumerate(stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=BENCHMARK_MAX_TOKENS, sampler=sampler,
    )):
        if i == 0:
            ttft = time.perf_counter() - t_start
        output_len += 1
    t_end = time.perf_counter()

    total_time = t_end - t_start
    tpot = (total_time - ttft) / (output_len - 1) if output_len > 1 and ttft else 0.0
    throughput = output_len / total_time if total_time > 0 else 0.0
    mem_gb = rss_memory_gb()

    result = {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "tpot_ms_per_tok": round(tpot * 1000, 1),
        "total_time_s": round(total_time, 2),
        "throughput_tok_per_s": round(throughput, 1),
        "memory_gb": round(mem_gb, 1),
    }

    print(f"  prompt_len={prompt_len}, output_len={output_len}")
    print(f"  TTFT={result['ttft_ms']}ms, TPOT={result['tpot_ms_per_tok']}ms/tok")
    print(f"  total={total_time:.2f}s, throughput={throughput:.1f} tok/s, memory={mem_gb:.1f} GB")

    del model, tokenizer
    mx.metal.clear_cache()

    return result


# --- 第 2 步：并发压测 (通过 vllm-metal server API) ---
async def run_concurrent_benchmark(server_url: str, concurrency: int = 8) -> dict[str, Any]:
    """通过 server API 进行并发压测。"""
    print(f"\n[2/3] 并发压测 (concurrency={concurrency})...")

    # Warmup
    async with httpx.AsyncClient(timeout=120) as client:
        await client.post(
            f"{server_url}/v1/completions",
            json={"model": MODEL_NAME, "prompt": "Hi", "max_tokens": 16, "temperature": 0.0},
        )

    async def make_request(client: httpx.AsyncClient) -> tuple[float, int]:
        t0 = time.perf_counter()
        resp = await client.post(
            f"{server_url}/v1/completions",
            json={
                "model": MODEL_NAME,
                "prompt": BENCHMARK_PROMPT,
                "max_tokens": 128,
                "temperature": 0.0,
            },
        )
        elapsed = time.perf_counter() - t0
        data = resp.json()
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return elapsed, completion_tokens

    async with httpx.AsyncClient(timeout=120) as client:
        t_start = time.perf_counter()
        tasks = [make_request(client) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    total_wall = t_end - t_start
    individual_times = [r[0] for r in results]
    total_output_tokens = sum(r[1] for r in results)
    total_throughput = total_output_tokens / total_wall if total_wall > 0 else 0

    result = {
        "concurrency": concurrency,
        "total_wall_time_s": round(total_wall, 2),
        "total_output_tokens": total_output_tokens,
        "total_throughput_tok_per_s": round(total_throughput, 1),
        "mean_request_time_s": round(sum(individual_times) / len(individual_times), 2),
        "max_request_time_s": round(max(individual_times), 2),
        "min_request_time_s": round(min(individual_times), 2),
    }

    print(f"  wall_time={total_wall:.2f}s, total_tokens={total_output_tokens}")
    print(f"  total_throughput={total_throughput:.1f} tok/s")
    print(f"  mean_req={result['mean_request_time_s']}s, min={result['min_request_time_s']}s, max={result['max_request_time_s']}s")

    return result


# --- 第 3 步：Golden Outputs 采集 (通过 server API) ---
async def collect_golden_outputs(server_url: str) -> dict[str, Any]:
    """通过 vllm-metal server /v1/completions 采集 golden outputs。"""
    print("\n[3/3] Golden Outputs 采集...")

    results = []
    async with httpx.AsyncClient(timeout=120) as client:
        for tc in TEST_CASES:
            t0 = time.perf_counter()
            resp = await client.post(
                f"{server_url}/v1/completions",
                json={
                    "model": MODEL_NAME,
                    "prompt": tc["prompt"],
                    "max_tokens": tc["max_tokens"],
                    "temperature": 0.0,
                },
            )
            elapsed = time.perf_counter() - t0
            data = resp.json()

            if "choices" not in data:
                print(f"  [{tc['id']}] ERROR: {data}")
                record = {
                    "test_id": tc["id"],
                    "prompt": tc["prompt"],
                    "max_tokens": tc["max_tokens"],
                    "output_text": "",
                    "output_tokens": 0,
                    "prompt_tokens": 0,
                    "finish_reason": "error",
                    "elapsed_s": round(elapsed, 3),
                    "error": str(data),
                    "framework": FRAMEWORK_NAME,
                    "framework_version": FRAMEWORK_VERSION,
                    "model": MODEL_NAME,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                results.append(record)
                continue

            record = {
                "test_id": tc["id"],
                "prompt": tc["prompt"],
                "max_tokens": tc["max_tokens"],
                "output_text": data["choices"][0]["text"],
                "output_tokens": data["usage"]["completion_tokens"],
                "prompt_tokens": data["usage"]["prompt_tokens"],
                "finish_reason": data["choices"][0]["finish_reason"],
                "elapsed_s": round(elapsed, 3),
                "framework": FRAMEWORK_NAME,
                "framework_version": FRAMEWORK_VERSION,
                "model": MODEL_NAME,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            results.append(record)
            print(f"  [{tc['id']}] {record['output_tokens']} tokens, {elapsed:.2f}s")

    # 写入 golden 文件
    golden_file = GOLDEN_DIR / "golden_outputs.json"
    golden_data = {
        "meta": {
            "framework": FRAMEWORK_NAME,
            "framework_version": FRAMEWORK_VERSION,
            "model": MODEL_NAME,
            "temperature": 0.0,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": {
                "chip": "Apple M5 Pro",
                "gpu_cores": 20,
                "memory_gb": 48,
            },
        },
        "test_cases": results,
    }

    golden_file.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_file, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, ensure_ascii=False, indent=2)

    print(f"\n  Golden outputs → {golden_file} ({len(results)} test cases)")
    return {"golden_file": str(golden_file), "test_case_count": len(results)}


# --- 主流程 ---
async def main_async(server_url: str) -> dict[str, Any]:
    """运行所有异步测试 (需 server 已启动)。"""
    concurrent_result = await run_concurrent_benchmark(server_url)
    golden_result = await collect_golden_outputs(server_url)
    return {"concurrent": concurrent_result, "golden": golden_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM-Metal Reference Benchmark")
    parser.add_argument("--model", default=MODEL_PATH, help="Model path")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--no-server", action="store_true", help="Don't start server (assume it's running)")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    model_path = args.model
    server_url = f"http://{SERVER_HOST}:{args.port}"

    results: dict[str, Any] = {
        "framework": FRAMEWORK_NAME,
        "framework_version": FRAMEWORK_VERSION,
        "model": MODEL_NAME,
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {"chip": "Apple M5 Pro", "gpu_cores": 20, "memory_gb": 48},
    }

    server_proc = None

    try:
        # Step 1: Single-inference benchmark (direct MLX, no server needed)
        results["single"] = run_single_direct_benchmark(model_path)

        # Start server for later steps
        if not args.no_server:
            print("\nStarting vllm-metal server...")
            server_proc = start_server(model_path)
            try:
                wait_time = wait_for_server(server_url)
                print(f"  Server ready in {wait_time:.1f}s")
            except TimeoutError:
                stop_server(server_proc)
                print("ERROR: Server failed to start", file=sys.stderr)
                sys.exit(1)

        # Step 2 & 3: Concurrent + Golden (async, through server)
        async_results = asyncio.run(main_async(server_url))
        results.update(async_results)

        # Print summary
        s = results["single"]
        c = results.get("concurrent", {})
        print("\n" + "=" * 60)
        print(f"{FRAMEWORK_NAME} v{FRAMEWORK_VERSION} 基准测试完成")
        print("=" * 60)
        print(f"\n{MODEL_NAME}:")
        print(f"  单次推理: {s['throughput_tok_per_s']} tok/s")
        print(f"    TTFT={s['ttft_ms']}ms, TPOT={s['tpot_ms_per_tok']}ms/tok")
        print(f"    prompt_len={s['prompt_len']}, output_len={s['output_len']}, memory={s['memory_gb']} GB")
        if c:
            print(f"  并发 x{c.get('concurrency', '?')}: {c.get('total_throughput_tok_per_s', '?')} tok/s")
        print(f"\n  Golden outputs → {results.get('golden', {}).get('golden_file', '?')}")

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))

    finally:
        if server_proc:
            print("\nStopping server...")
            stop_server(server_proc)


if __name__ == "__main__":
    main()
