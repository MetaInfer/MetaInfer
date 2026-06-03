#!/usr/bin/env python3
"""mac-engine 对抗性测试套件 — P0 必须通过的测试。

测试修复后的引擎在极端条件下的正确性和性能。
串行执行（MLX GPU 资源竞争约束）。

用法:
    cd subprojects/mac-engine
    python scripts/adversarial_test.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B/")
BASELINE_PROMPT = "Explain the key concepts of machine learning in detail."

results: list[dict] = []
failures: list[str] = []


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


def _record(test_id: str, name: str, passed: bool, data: dict):
    entry = {"test_id": test_id, "name": name, "passed": passed, **data}
    results.append(entry)
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} [{test_id}] {name}")
    if not passed:
        failures.append(f"[{test_id}] {name}")
    for k, v in data.items():
        print(f"         {k}: {v}")


# ─────────────────────────────────────────────
# A08: Greedy 确定性 (10 轮 byte 级对比)
# ─────────────────────────────────────────────
def test_a08_determinism():
    print("\n" + "=" * 60)
    print("A08: Greedy 确定性测试 (10 轮)")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    outputs = []
    for i in range(10):
        tokens = []
        for text in engine.generate(BASELINE_PROMPT, max_tokens=256, temperature=0.0):
            tokens.append(text)
        full = "".join(tokens)
        outputs.append(full)
        if i == 0:
            mx.clear_cache()

    # Byte-level comparison
    first_hash = hashlib.md5(outputs[0].encode()).hexdigest()
    all_same = all(hashlib.md5(o.encode()).hexdigest() == first_hash for o in outputs)

    _record("A08", "Greedy 确定性 (10轮)", all_same, {
        "first_hash": first_hash,
        "output_len": len(outputs[0]),
        "first_50_chars": repr(outputs[0][:50]),
    })

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# A01: 长序列 decode 性能衰减
# ─────────────────────────────────────────────
def test_a01_long_decode():
    print("\n" + "=" * 60)
    print("A01: 长序列 decode 性能衰减")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    gen_lengths = [128, 256, 512, 1024, 2048]
    tpot_data = {}

    for max_tok in gen_lengths:
        t0 = time.perf_counter()
        count = 0
        for _ in engine.generate("Hello", max_tokens=max_tok, temperature=0.0):
            count += 1
        elapsed = time.perf_counter() - t0
        tpot = (elapsed * 1000) / count if count > 1 else 0
        tpot_data[max_tok] = round(tpot, 2)
        print(f"  max_tokens={max_tok}: {count} tokens, {tpot:.2f} ms/tok, "
              f"{count/elapsed:.1f} tok/s")
        mx.clear_cache()

    # Check degradation: TPOT at 2048 should be within 20% of TPOT at 256
    degradation = (tpot_data[2048] - tpot_data[256]) / tpot_data[256]
    passed = degradation < 0.20  # <20% degradation is acceptable

    _record("A01", "长序列 decode 衰减", passed, {
        "tpot_128": f"{tpot_data[128]} ms",
        "tpot_256": f"{tpot_data[256]} ms",
        "tpot_512": f"{tpot_data[512]} ms",
        "tpot_1024": f"{tpot_data[1024]} ms",
        "tpot_2048": f"{tpot_data[2048]} ms",
        "degradation_256_to_2048": f"{degradation:.1%}",
    })

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# A02: 长 prompt prefill 压力测试
# ─────────────────────────────────────────────
def test_a02_long_prefill():
    print("\n" + "=" * 60)
    print("A02: 长 prompt prefill 压力测试")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    # Construct long prompts by repeating a sentence
    base = "The quick brown fox jumps over the lazy dog. "
    token_per_char = 0.3  # rough estimate
    target_lengths = [64, 256, 1024, 4096]

    ttft_data = {}
    for target in target_lengths:
        # Build prompt with approximately target tokens
        prompt = base * max(1, int(target / (len(base) * token_per_char)))
        actual_tokens = len(engine.tokenizer.encode(prompt))

        t0 = time.perf_counter()
        count = 0
        for _ in engine.generate(prompt, max_tokens=16, temperature=0.0):
            count += 1
        ttft_total = time.perf_counter() - t0

        # Estimate TTFT: total - decode time
        decode_time = count * 0.055  # ~55ms per decode token
        ttft_est = (ttft_total - decode_time) * 1000
        ttft_data[target] = round(ttft_est, 1)
        print(f"  prompt_tokens≈{actual_tokens}: TTFT≈{ttft_est:.0f}ms, "
              f"total={ttft_total:.2f}s")
        mx.clear_cache()

    # Check TTFT scales roughly linearly (not O(n²))
    # Linear: ttft_4096/ttft_64 should be < 100 (not quadratic)
    ratio = ttft_data.get(4096, 0) / max(ttft_data.get(64, 1), 1)
    passed = ratio < 100 and ttft_data.get(4096, 999999) < 60000  # < 60s for 4K

    _record("A02", "长 prompt prefill", passed, {
        "ttft_64tok": f"{ttft_data.get(64, 'N/A')} ms",
        "ttft_256tok": f"{ttft_data.get(256, 'N/A')} ms",
        "ttft_1024tok": f"{ttft_data.get(1024, 'N/A')} ms",
        "ttft_4096tok": f"{ttft_data.get(4096, 'N/A')} ms",
        "scaling_ratio_4096_vs_64": f"{ratio:.1f}x",
    })

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# A10: KV cache offset 边界条件
# ─────────────────────────────────────────────
def test_a10_kv_boundary():
    print("\n" + "=" * 60)
    print("A10: KV cache offset 边界条件")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    all_passed = True
    checks = {}

    # Test 1: Normal prefill + decode
    output1 = "".join(engine.generate("Hello world", max_tokens=32, temperature=0.0))
    checks["normal_output"] = repr(output1[:30])
    checks["normal_ok"] = len(output1) > 0
    all_passed = all_passed and checks["normal_ok"]

    # Test 2: Single token prompt
    output2 = "".join(engine.generate("a", max_tokens=32, temperature=0.0))
    checks["single_tok_prompt"] = repr(output2[:30])
    checks["single_tok_ok"] = len(output2) > 0
    all_passed = all_passed and checks["single_tok_ok"]

    # Test 3: Long prefill (1000+ tokens)
    long_prompt = "Machine learning is " * 100
    output3 = "".join(engine.generate(long_prompt, max_tokens=32, temperature=0.0))
    checks["long_prefill_ok"] = len(output3) > 0
    all_passed = all_passed and checks["long_prefill_ok"]

    # Test 4: All layers have same offset after generate
    engine.generate("test", max_tokens=64, temperature=0.0)
    if engine._cache is not None:
        offsets = [c.offset for c in engine._cache]
        all_same_offset = len(set(offsets)) == 1
        checks["offset_consistency"] = all_same_offset
        checks["offsets_value"] = offsets[0]
        all_passed = all_passed and all_same_offset
    else:
        all_passed = False
        checks["offset_consistency"] = False

    _record("A10", "KV cache 边界条件", all_passed, checks)

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# A15: KV cache 满载 (>2048 tokens)
# ─────────────────────────────────────────────
def test_a15_kv_full():
    print("\n" + "=" * 60)
    print("A15: KV cache 满载 (>2048 tokens)")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    # Use a medium prompt + long generation to exceed 2048
    prompt = "Write a story. "
    prompt_tokens = len(engine.tokenizer.encode(prompt))

    t0 = time.perf_counter()
    count = 0
    for _ in engine.generate(prompt, max_tokens=2100, temperature=0.0):
        count += 1
    elapsed = time.perf_counter() - t0

    total_seq = prompt_tokens + count
    mem_gb = _rss_gb()

    # Check: should not crash, output should be reasonable
    passed = count >= 2000 and total_seq > 2048

    _record("A15", "KV cache 满载 (>2048)", passed, {
        "total_seq_len": total_seq,
        "tokens_generated": count,
        "elapsed_s": round(elapsed, 2),
        "memory_gb": round(mem_gb, 1),
    })

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# A06: 多轮延迟抖动 (20 轮)
# ─────────────────────────────────────────────
def test_a06_latency_jitter():
    print("\n" + "=" * 60)
    print("A06: 多轮延迟抖动 (20 轮)")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    tpots = []
    for i in range(20):
        t0 = time.perf_counter()
        count = 0
        for _ in engine.generate(BASELINE_PROMPT, max_tokens=64, temperature=0.0):
            count += 1
        elapsed = time.perf_counter() - t0
        tpot = (elapsed * 1000) / count
        tpots.append(tpot)
        if i == 0:
            print(f"  Round 0 (cold): {tpot:.2f} ms/tok")

    tpots_arr = tpots[1:]  # skip first (cold start)
    p50 = sorted(tpots_arr)[len(tpots_arr) // 2]
    p99 = sorted(tpots_arr)[int(len(tpots_arr) * 0.99)]
    mean = sum(tpots_arr) / len(tpots_arr)
    stdev = (sum((x - mean)**2 for x in tpots_arr) / len(tpots_arr))**0.5

    # P99 should be within 15% of P50
    jitter_ratio = (p99 - p50) / p50
    passed = jitter_ratio < 0.15

    _record("A06", "多轮延迟抖动", passed, {
        "cold_start_tpot": f"{tpots[0]:.2f} ms",
        "p50": f"{p50:.2f} ms",
        "p99": f"{p99:.2f} ms",
        "mean": f"{mean:.2f} ms",
        "stdev": f"{stdev:.2f} ms",
        "jitter_ratio": f"{jitter_ratio:.1%}",
    })

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# Edge: 空 prompt + EOS token
# ─────────────────────────────────────────────
def test_edge_cases():
    print("\n" + "=" * 60)
    print("Edge: 空 prompt / max_tokens=1 / special tokens")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    all_passed = True
    checks = {}

    # Empty prompt
    try:
        out = "".join(engine.generate("", max_tokens=16, temperature=0.0))
        checks["empty_prompt_ok"] = len(out) > 0
        checks["empty_prompt_output"] = repr(out[:30])
    except Exception as e:
        checks["empty_prompt_ok"] = False
        checks["empty_prompt_error"] = str(e)
    all_passed = all_passed and checks.get("empty_prompt_ok", False)

    # max_tokens=1
    try:
        out = "".join(engine.generate("Hello", max_tokens=1, temperature=0.0))
        checks["max_tokens_1_ok"] = True
        checks["max_tokens_1_output"] = repr(out[:30])
    except Exception as e:
        checks["max_tokens_1_ok"] = False
        checks["max_tokens_1_error"] = str(e)
    all_passed = all_passed and checks.get("max_tokens_1_ok", False)

    # Special tokens prompt
    try:
        out = "".join(engine.generate("<|im_start|>user\nHi<|im_end|>", max_tokens=16, temperature=0.0))
        checks["special_tokens_ok"] = True
        checks["special_tokens_output"] = repr(out[:30])
    except Exception as e:
        checks["special_tokens_ok"] = False
        checks["special_tokens_error"] = str(e)
    all_passed = all_passed and checks.get("special_tokens_ok", False)

    # generate_stream works
    try:
        out = "".join(engine.generate_stream("Test", max_tokens=16, temperature=0.0))
        checks["stream_ok"] = True
        checks["stream_output"] = repr(out[:30])
    except Exception as e:
        checks["stream_ok"] = False
        checks["stream_error"] = str(e)
    all_passed = all_passed and checks.get("stream_ok", False)

    _record("EDGE", "边界条件 (空 prompt / max_tokens=1 / stream)", all_passed, checks)

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# A20: Multi-turn 状态隔离
# ─────────────────────────────────────────────
def test_a20_multi_turn():
    print("\n" + "=" * 60)
    print("A20: Multi-turn 状态隔离")
    print("=" * 60)

    from src.engine_v1 import InferenceEngine

    engine = InferenceEngine()
    engine.load_model(MODEL_PATH)

    # Generate same prompt twice on same engine instance
    out1 = "".join(engine.generate(BASELINE_PROMPT, max_tokens=64, temperature=0.0))
    out2 = "".join(engine.generate(BASELINE_PROMPT, max_tokens=64, temperature=0.0))

    hash1 = hashlib.md5(out1.encode()).hexdigest()
    hash2 = hashlib.md5(out2.encode()).hexdigest()

    # Both should produce identical output (greedy determinism + state isolation)
    passed = hash1 == hash2

    _record("A20", "Multi-turn 状态隔离", passed, {
        "hash_round1": hash1,
        "hash_round2": hash2,
        "output_match": hash1 == hash2,
    })

    del engine
    mx.clear_cache()


# ─────────────────────────────────────────────
# Main: run all tests serially
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("mac-engine 对抗性测试套件")
    print(f"Model: {MODEL_PATH}")
    print(f"Memory: {_rss_gb():.1f} GB (initial)")
    print("=" * 60)

    # Run P0 tests in order
    test_a08_determinism()
    test_a01_long_decode()
    test_a02_long_prefill()
    test_a10_kv_boundary()
    test_a15_kv_full()
    test_a06_latency_jitter()
    test_edge_cases()
    test_a20_multi_turn()

    # Summary
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n  {passed}/{total} 通过")

    if failures:
        print("\n  失败项:")
        for f in failures:
            print(f"    ❌ {f}")
    else:
        print("\n  🎉 全部通过!")

    # Save JSON results
    out_path = PROJECT_ROOT / "scripts" / "adversarial_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果: {out_path}")


if __name__ == "__main__":
    main()
