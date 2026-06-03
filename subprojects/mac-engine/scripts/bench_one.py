#!/usr/bin/env python3
"""mlx_lm 或 mac-engine 逐场景基准测试 (模型加载一次)。

用法:
    cd subprojects/mac-engine
    python scripts/bench_one.py mlx_lm              # 跑 mlx_lm 全场景
    python scripts/bench_one.py mac_engine           # 跑 mac-engine 全场景
    python scripts/bench_one.py mlx_lp sp_256        # 仅跑一个场景
    python scripts/bench_one.py summary              # 打印已保存结果
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B/")

SCENARIOS = [
    ("sp_256",  10,   256),
    ("sp_1024", 10,   1024),
    ("sp_2048", 10,   2048),
    ("mp_256",  500,  256),
    ("mp_1024", 500,  1024),
    ("lp_256",  2000, 256),
    ("lp_1024", 2000, 1024),
    ("xp_256",  4000, 256),
]

RESULTS_FILE = PROJECT_ROOT / "scripts" / "bench_compare_results.json"


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def build_prompt(approx_tokens: int) -> str:
    if approx_tokens <= 20:
        return "Explain the key concepts of machine learning in detail."
    base = ("The history of artificial intelligence spans several decades, "
            "encompassing many breakthroughs and innovations. ")
    return base * max(1, int(approx_tokens / 0.3) // len(base))


def load_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"meta": {"timestamp": ""}, "results": {}}


def save_results(data: dict):
    data["meta"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_mlx_lm(scenario_filter: str | None = None):
    """加载 mlx_lm 一次，跑所有场景。"""
    from mlx_lm import load as mlx_load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    data = load_results()
    print("[mlx_lm] 加载模型...")
    model, tokenizer = mlx_load(MODEL_PATH, tokenizer_config={"trust_remote_code": True})

    # Warmup
    print("[mlx_lm] Warmup...")
    for _ in stream_generate(model, tokenizer, prompt="Hi", max_tokens=16,
                             sampler=make_sampler(temp=0.0)):
        pass
    mx.clear_cache()

    for label, prompt_tok, max_tok in SCENARIOS:
        if scenario_filter and label != scenario_filter:
            continue
        prompt = build_prompt(prompt_tok)
        prompt_len = len(tokenizer.encode(prompt))

        print(f"[mlx_lm] {label} (p={prompt_len}, gen={max_tok})...", end=" ", flush=True)
        sampler = make_sampler(temp=0.0)
        ttft = None
        count = 0
        t0 = time.perf_counter()
        for i, _ in enumerate(stream_generate(model, tokenizer, prompt=prompt,
                                               max_tokens=max_tok, sampler=sampler)):
            if i == 0:
                ttft = time.perf_counter() - t0
            count += 1
        elapsed = time.perf_counter() - t0
        tpot = (elapsed - ttft) / (count - 1) if count > 1 and ttft else 0
        tp = round(count / elapsed, 1)
        print(f"{tp} tok/s, TPOT={tpot*1000:.2f}ms, TTFT={ttft*1000:.1f}ms")

        data["results"][label] = data["results"].get(label, {})
        data["results"][label]["mlx_lm"] = {
            "prompt_len": prompt_len, "output_len": count,
            "ttft_ms": round(ttft*1000, 1) if ttft else None,
            "tpot_ms": round(tpot*1000, 2),
            "throughput": tp,
            "memory_gb": round(_rss_gb(), 1),
        }
        save_results(data)
        mx.clear_cache()

    del model, tokenizer
    mx.clear_cache()
    print("[mlx_lm] 完成")


def run_mac_engine(scenario_filter: str | None = None):
    """加载 mac-engine 一次，跑所有场景。"""
    from src.engine_v1 import InferenceEngine

    data = load_results()
    print("[mac-engine] 加载模型...")
    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    # Warmup
    print("[mac-engine] Warmup...")
    for _ in engine.generate("Hi", max_tokens=16, temperature=0.0):
        pass
    mx.clear_cache()

    for label, prompt_tok, max_tok in SCENARIOS:
        if scenario_filter and label != scenario_filter:
            continue
        prompt = build_prompt(prompt_tok)
        prompt_len = len(engine.tokenizer.encode(prompt))

        print(f"[mac-engine] {label} (p={prompt_len}, gen={max_tok})...", end=" ", flush=True)
        ttft = None
        count = 0
        t0 = time.perf_counter()
        for i, _ in enumerate(engine.generate(prompt, max_tokens=max_tok, temperature=0.0)):
            if i == 0:
                ttft = time.perf_counter() - t0
            count += 1
        elapsed = time.perf_counter() - t0
        tpot = (elapsed - ttft) / (count - 1) if count > 1 and ttft else 0
        tp = round(count / elapsed, 1)
        print(f"{tp} tok/s, TPOT={tpot*1000:.2f}ms, TTFT={ttft*1000:.1f}ms")

        data["results"][label] = data["results"].get(label, {})
        data["results"][label]["mac_engine"] = {
            "prompt_len": prompt_len, "output_len": count,
            "ttft_ms": round(ttft*1000, 1) if ttft else None,
            "tpot_ms": round(tpot*1000, 2),
            "throughput": tp,
            "memory_gb": round(_rss_gb(), 1),
        }
        save_results(data)
        mx.clear_cache()

    del engine
    mx.clear_cache()
    print("[mac-engine] 完成")


def print_summary():
    data = load_results()
    results = data.get("results", {})

    print(f"\n{'='*115}")
    print("汇总对比表")
    print(f"{'='*115}")
    print(f"{'Scenario':<12} {'p_len':>5} {'gen':>5} │ "
          f"{'mlx_tp':>7} {'mlx_tpot':>8} {'mlx_ttft':>9} │ "
          f"{'eng_tp':>7} {'eng_tpot':>8} {'eng_ttft':>9} │ "
          f"{'Ratio':>6} {'TPOTΔ':>7} {'TTFT_r':>7}")
    print("─" * 115)

    for label, prompt_tok, max_tok in SCENARIOS:
        if label not in results:
            continue
        r = results[label]
        m = r.get("mlx_lm", {})
        e = r.get("mac_engine", {})

        if not m or not e:
            missing = "mlx" if not m else "eng"
            print(f"{label:<12} {r.get('mlx_lm', {}).get('prompt_len', '?'):>5} {max_tok:>5} │ "
                  f"  ({missing} missing)")
            continue

        ratio = round(e["throughput"] / m["throughput"], 3)
        tpot_d = round(e["tpot_ms"] - m["tpot_ms"], 2)
        ttft_r = round(e["ttft_ms"] / m["ttft_ms"], 3) if m["ttft_ms"] else None

        print(f"{label:<12} {m['prompt_len']:>5} {max_tok:>5} │ "
              f"{m['throughput']:>7.1f} {m['tpot_ms']:>8.2f} {m['ttft_ms']:>9.1f} │ "
              f"{e['throughput']:>7.1f} {e['tpot_ms']:>8.2f} {e['ttft_ms']:>9.1f} │ "
              f"{ratio:>6.3f} {tpot_d:>+7.2f} {ttft_r:>7.3f}")

    print(f"\nJSON: {RESULTS_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    scenario = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "summary":
        print_summary()
    elif cmd == "mlx_lm":
        run_mlx_lm(scenario)
    elif cmd == "mac_engine":
        run_mac_engine(scenario)
    else:
        print("用法: python bench_one.py [mlx_lm|mac_engine|summary] [scenario]")
        print(f"场景: {', '.join(s[0] for s in SCENARIOS)}")
