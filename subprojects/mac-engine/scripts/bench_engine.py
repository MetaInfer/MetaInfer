#!/usr/bin/env python3
"""自研引擎 benchmark 脚本。

对指定版本的引擎进行单次推理性能测试 (TTFT, TPOT, 吞吐, 内存)。

用法:
    python bench_engine.py --phase 0  # 跑 Phase 0 版本的 bench
"""

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import mlx.core as mx
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = str(
    Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B/"
)
BENCHMARK_PROMPT = "Explain the key concepts of machine learning in detail."
BENCHMARK_MAX_TOKENS = 256


def rss_memory_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def run_benchmark(engine, model_path: str, prompt: str, max_tokens: int) -> dict:
    """Run single-inference benchmark with token-level timing."""
    print(f"  Loading model: {model_path}")
    engine.load_model(model_path)

    # Warmup
    print("  Warmup...")
    gen_method = getattr(engine, "generate_stream", engine.generate)
    for _ in gen_method(prompt, max_tokens=16, temperature=0.0):
        pass

    # Benchmark
    print(f"  Benchmark: max_tokens={max_tokens}")
    ttft = None
    total_time = 0.0
    output_len = 0
    first_token_text = ""

    t_start = time.perf_counter()
    for i, token_text in enumerate(gen_method(prompt, max_tokens=max_tokens, temperature=0.0)):
        if i == 0:
            ttft = time.perf_counter() - t_start
            first_token_text = token_text
        output_len += 1
    t_end = time.perf_counter()

    total_time = t_end - t_start
    tpot = (total_time - ttft) / (output_len - 1) if output_len > 1 and ttft else 0.0
    throughput = output_len / total_time if total_time > 0 else 0.0
    mem_gb = rss_memory_gb()

    result = {
        "output_len": output_len,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "tpot_ms_per_tok": round(tpot * 1000, 1),
        "total_time_s": round(total_time, 2),
        "throughput_tok_per_s": round(throughput, 1),
        "memory_gb": round(mem_gb, 1),
        "first_token": repr(first_token_text[:50]) if first_token_text else "",
    }

    print(f"  output_len={output_len}, TTFT={result['ttft_ms']}ms, "
          f"TPOT={result['tpot_ms_per_tok']}ms/tok")
    print(f"  total={total_time:.2f}s, throughput={throughput:.1f} tok/s, "
          f"memory={mem_gb:.1f} GB")

    del engine
    mx.clear_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark engine")
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3],
                        help="Phase number (0/1/2/3)")
    parser.add_argument("--model", default=MODEL_PATH, help="Model path")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    # Import corresponding phase engine
    if args.phase == 0:
        from src.engine_v0 import InferenceEngine as Engine
    elif args.phase == 1:
        from src.engine_v1 import InferenceEngine as Engine
    elif args.phase == 2:
        from src.engine_v2 import InferenceEngine as Engine
    else:
        from src.engine_v3 import InferenceEngine as Engine

    engine = Engine()
    result = run_benchmark(engine, args.model, BENCHMARK_PROMPT, BENCHMARK_MAX_TOKENS)

    # Phase 2+: also run concurrent benchmark (reuse engine, don't reload)
    if args.phase >= 2:
        # Use the already-loaded engine for concurrent test
        print(f"\n  Concurrent x4 benchmark...")
        prompts = [
            BENCHMARK_PROMPT,
            "What is deep learning and how does it work?",
            "Explain neural networks step by step.",
            "Describe the transformer architecture.",
        ]
        t0 = time.perf_counter()
        results = engine.generate_batch(prompts, max_tokens=64)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(r) for r in results)
        conc_throughput = total_tokens / elapsed
        conc_mem = rss_memory_gb()
        print(f"  total_tokens={total_tokens}, elapsed={elapsed:.2f}s, "
              f"throughput={conc_throughput:.1f} tok/s, memory={conc_mem:.1f} GB")
        result["concurrent"] = {
            "num_requests": len(prompts),
            "total_tokens": total_tokens,
            "elapsed_s": round(elapsed, 2),
            "throughput_tok_per_s": round(conc_throughput, 1),
            "memory_gb": round(conc_mem, 1),
        }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
