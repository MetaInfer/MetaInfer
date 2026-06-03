#!/usr/bin/env python3
"""mlx_lm vs mac-engine 统一基准测试 (逐场景串行)。

每个框架只加载一次模型，场景间仅清 KV cache。
按 prompt_len 和 gen_len 两个维度展开测试矩阵。

用法:
    cd subprojects/mac-engine
    python scripts/bench_compare.py --rounds 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B/")

# (prompt_approx_tokens, max_tokens, label)
TEST_MATRIX = [
    (10,   256,  "sp_256"),
    (10,   1024, "sp_1024"),
    (10,   2048, "sp_2048"),
    (500,  256,  "mp_256"),
    (500,  1024, "mp_1024"),
    (2000, 256,  "lp_256"),
    (2000, 1024, "lp_1024"),
    (4000, 256,  "xp_256"),
]


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def build_prompt(approx_tokens: int) -> str:
    if approx_tokens <= 20:
        return "Explain the key concepts of machine learning in detail."
    base = "The history of artificial intelligence spans several decades, encompassing many breakthroughs and innovations. "
    chars_needed = int(approx_tokens / 0.3)
    return base * max(1, chars_needed // len(base))


def run_one_mlx_lm(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """单次 mlx_lm 生成测量。"""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    prompt_len = len(tokenizer.encode(prompt))

    ttft = None
    output_len = 0
    t_start = time.perf_counter()
    for i, resp in enumerate(stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler,
    )):
        if i == 0:
            ttft = time.perf_counter() - t_start
        output_len += 1
    t_end = time.perf_counter()

    total = t_end - t_start
    tpot = (total - ttft) / (output_len - 1) if output_len > 1 and ttft else 0.0
    return {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "tpot_ms": round(tpot * 1000, 2),
        "total_s": round(total, 2),
        "throughput": round(output_len / total, 1) if total > 0 else 0,
        "memory_gb": round(_rss_gb(), 1),
    }


def run_one_mac_engine(engine, prompt: str, max_tokens: int) -> dict:
    """单次 mac-engine 生成测量。"""
    prompt_len = len(engine.tokenizer.encode(prompt))

    ttft = None
    output_len = 0
    t_start = time.perf_counter()
    for i, _ in enumerate(engine.generate(prompt, max_tokens=max_tokens, temperature=0.0)):
        if i == 0:
            ttft = time.perf_counter() - t_start
        output_len += 1
    t_end = time.perf_counter()

    total = t_end - t_start
    tpot = (total - ttft) / (output_len - 1) if output_len > 1 and ttft else 0.0
    return {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "tpot_ms": round(tpot * 1000, 2),
        "total_s": round(total, 2),
        "throughput": round(output_len / total, 1) if total > 0 else 0,
        "memory_gb": round(_rss_gb(), 1),
    }


def agg(key: str, results: list[dict]):
    vals = [r[key] for r in results if r[key] is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("mlx_lm vs mac-engine 统一基准测试")
    print(f"Rounds={args.rounds}, Scenarios={len(TEST_MATRIX)}")
    print("=" * 70)

    all_results = []

    # ── Phase 1: mlx_lm (加载一次) ──
    print("\n[Phase 1] 加载 mlx_lm...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})

    # Warmup
    print("  Warmup...")
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    for _ in stream_generate(model, tokenizer, prompt="Hello", max_tokens=16,
                             sampler=make_sampler(temp=0.0)):
        pass
    mx.clear_cache()

    for prompt_tok, max_tok, label in TEST_MATRIX:
        prompt = build_prompt(prompt_tok)
        rounds_data = []
        for r in range(args.rounds):
            print(f"  [{label}] round {r+1}/{args.rounds}...", end=" ", flush=True)
            d = run_one_mlx_lm(model, tokenizer, prompt, max_tok)
            rounds_data.append(d)
            print(f"{d['throughput']} tok/s, TPOT={d['tpot_ms']}ms, TTFT={d['ttft_ms']}ms")
            mx.clear_cache()

        # Store for later comparison
        for r in rounds_data:
            r["_framework"] = "mlx_lm"
            r["_scenario"] = label
        all_results.append({
            "scenario": label, "prompt_approx": prompt_tok, "max_tokens": max_tok,
            "framework": "mlx_lm",
            "prompt_len": rounds_data[0]["prompt_len"],
            "output_len": agg("output_len", rounds_data),
            "ttft_ms": agg("ttft_ms", rounds_data),
            "tpot_ms": agg("tpot_ms", rounds_data),
            "throughput": agg("throughput", rounds_data),
            "memory_gb": agg("memory_gb", rounds_data),
        })

    del model, tokenizer
    mx.clear_cache()
    print("  mlx_lm 完成，已释放")

    # ── Phase 2: mac-engine (加载一次) ──
    print("\n[Phase 2] 加载 mac-engine...")
    from src.engine_v1 import InferenceEngine
    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    # Warmup
    print("  Warmup...")
    for _ in engine.generate("Hello", max_tokens=16, temperature=0.0):
        pass
    mx.clear_cache()

    for prompt_tok, max_tok, label in TEST_MATRIX:
        prompt = build_prompt(prompt_tok)
        rounds_data = []
        for r in range(args.rounds):
            print(f"  [{label}] round {r+1}/{args.rounds}...", end=" ", flush=True)
            d = run_one_mac_engine(engine, prompt, max_tok)
            rounds_data.append(d)
            print(f"{d['throughput']} tok/s, TPOT={d['tpot_ms']}ms, TTFT={d['ttft_ms']}ms")
            mx.clear_cache()

        all_results.append({
            "scenario": label, "prompt_approx": prompt_tok, "max_tokens": max_tok,
            "framework": "mac_engine",
            "prompt_len": rounds_data[0]["prompt_len"],
            "output_len": agg("output_len", rounds_data),
            "ttft_ms": agg("ttft_ms", rounds_data),
            "tpot_ms": agg("tpot_ms", rounds_data),
            "throughput": agg("throughput", rounds_data),
            "memory_gb": agg("memory_gb", rounds_data),
        })

    del engine
    mx.clear_cache()
    print("  mac-engine 完成，已释放")

    # ── Phase 3: 汇总对比 ──
    print("\n" + "=" * 70)
    print("汇总对比表")
    print("=" * 70)
    print(f"{'Scenario':<12} {'p_len':>5} {'gen':>5} │ "
          f"{'mlx_tp':>7} {'mlx_tpot':>8} {'mlx_ttft':>8} │ "
          f"{'eng_tp':>7} {'eng_tpot':>8} {'eng_ttft':>8} │ "
          f"{'Ratio':>6} {'TPOTΔ':>7} {'TTFT_r':>6}")
    print("─" * 110)

    for prompt_tok, max_tok, label in TEST_MATRIX:
        mlx = next(r for r in all_results if r["framework"] == "mlx_lm" and r["scenario"] == label)
        eng = next(r for r in all_results if r["framework"] == "mac_engine" and r["scenario"] == label)

        ratio = round(eng["throughput"] / mlx["throughput"], 3) if mlx["throughput"] else None
        tpot_diff = round(eng["tpot_ms"] - mlx["tpot_ms"], 2) if eng["tpot_ms"] and mlx["tpot_ms"] else None
        ttft_r = round(eng["ttft_ms"] / mlx["ttft_ms"], 3) if eng["ttft_ms"] and mlx["ttft_ms"] else None

        print(f"{label:<12} {mlx['prompt_len']:>5} {max_tok:>5} │ "
              f"{mlx['throughput']:>7.1f} {mlx['tpot_ms']:>8.2f} {mlx['ttft_ms']:>8.1f} │ "
              f"{eng['throughput']:>7.1f} {eng['tpot_ms']:>8.2f} {eng['ttft_ms']:>8.1f} │ "
              f"{ratio:>6.3f} {tpot_diff:>+7.2f} {ttft_r:>6.3f}")

    # Save
    out_path = PROJECT_ROOT / "scripts" / "bench_compare_results.json"
    # Build paired results
    paired = []
    for prompt_tok, max_tok, label in TEST_MATRIX:
        mlx = next(r for r in all_results if r["framework"] == "mlx_lm" and r["scenario"] == label)
        eng = next(r for r in all_results if r["framework"] == "mac_engine" and r["scenario"] == label)
        ratio = round(eng["throughput"] / mlx["throughput"], 3) if mlx["throughput"] else None
        tpot_diff = round(eng["tpot_ms"] - mlx["tpot_ms"], 2) if eng["tpot_ms"] and mlx["tpot_ms"] else None
        ttft_r = round(eng["ttft_ms"] / mlx["ttft_ms"], 3) if eng["ttft_ms"] and mlx["ttft_ms"] else None
        paired.append({
            "scenario": label, "prompt_tokens_approx": prompt_tok, "max_tokens": max_tok,
            "mlx_lm": mlx, "mac_engine": eng,
            "comparison": {"throughput_ratio": ratio, "tpot_diff_ms": tpot_diff, "ttft_ratio": ttft_r},
        })

    with open(out_path, "w") as f:
        json.dump({"meta": {"rounds": args.rounds, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                   "results": paired}, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")

    if args.json:
        print(json.dumps(paired, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
