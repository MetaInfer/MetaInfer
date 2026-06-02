#!/usr/bin/env python3
"""优化差距分析脚本: 算子级计时 + 框架开销拆解 + 内存分析。

用法:
    python profile_gaps.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.kv_cache import make_kv_cache
from src.model import Qwen3Config, Qwen3Attention, Qwen3MLP, Qwen3Block, Qwen3Model
from src.weights import load_qwen3_model
from src.tokenizer import Tokenizer

MODEL_PATH = str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B/")
H = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 12288
N_LAYERS = 36

def time_op(fn, warmup=3, repeats=20):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats * 1000  # ms


def main():
    print("=" * 60)
    print("🔥 优化差距分析: 算子级计时")
    print("=" * 60)

    # ---- Load model for accurate op-level timing ----
    model, config = load_qwen3_model(MODEL_PATH)
    tok = Tokenizer(MODEL_PATH)
    print()

    # ---- Prefill attention ----
    L_prefill = 128
    x_pf = mx.random.normal((1, L_prefill, H))
    attn = model.model.layers[0].self_attn
    mask_pf = mx.triu(mx.full((L_prefill, L_prefill), float("-inf"), mx.float32), k=1)
    mask_pf = mask_pf.reshape(1, 1, L_prefill, L_prefill)

    t_pf = time_op(lambda: attn(x_pf, mask_pf))
    print(f"  attention_prefill (L={L_prefill}): {t_pf:.1f} ms")

    # ---- Decode attention ----
    L_decode = 1
    x_dec = mx.random.normal((1, L_decode, H))
    kvc = make_kv_cache(1)
    attn(x_pf, mask_pf, kvc[0])  # prefill cache
    mx.synchronize()

    t_dec = time_op(lambda: attn(x_dec, None, kvc[0]))
    print(f"  attention_decode (L=1, cached): {t_dec:.4f} ms")

    # ---- Full forward (prefill) ----
    prompt = "Explain the key concepts of machine learning in detail."
    ids = tok.encode(prompt)
    input_ids = mx.array([ids])

    # Warmup
    model(input_ids)
    mx.synchronize()

    t_model_prefill = time_op(lambda: model(input_ids))
    print(f"\n  model.prefill (L={len(ids)}): {t_model_prefill:.1f} ms")

    # ---- Full forward (decode) ----
    cache = make_kv_cache(len(model.layers))
    model(input_ids, cache=cache)
    mx.synchronize()
    next_input = mx.array([[1]])
    t_model_decode = time_op(lambda: model(next_input, cache=cache))
    print(f"  model.decode (L=1, cached): {t_model_decode:.1f} ms")

    # ---- FFN only ----
    mlp = model.model.layers[0].mlp
    x_ffn = mx.random.normal((1, 1, H))
    t_ffn = time_op(lambda: mlp(x_ffn))
    print(f"  ffn (1 token): {t_ffn:.4f} ms")

    # ---- Embedding + Norm ----
    emb = model.model.embed_tokens
    t_emb = time_op(lambda: emb(input_ids))
    print(f"  embed (L={len(ids)}): {t_emb:.4f} ms")

    # ---- Entire layer ----
    block = model.model.layers[0]
    x_blk = mx.random.normal((1, 1, H))
    t_block = time_op(lambda: block(x_blk))
    print(f"  transformer_block (1 token): {t_block:.1f} ms")

    # ---- ======================================== ----
    print("\n" + "=" * 60)
    print("📊 差距拆解 (256 token generate)")
    print("=" * 60)

    # Measure full generate without and with KV cache
    from src.engine_v0 import InferenceEngine as E0
    from src.engine_v1 import InferenceEngine as E1

    # Phase 1 (KV cache)
    e1 = E1()
    e1.load_model(MODEL_PATH)
    for _ in e1.generate("Hi", max_tokens=4, temperature=0.0):
        pass
    mx.synchronize()

    # Time prefill alone
    ids_bm = tok.encode(prompt)
    cache_bm = make_kv_cache(len(model.layers))
    input_bm = mx.array([ids_bm])
    t_prefill = time_op(lambda: model(input_bm, cache=cache_bm), warmup=1, repeats=10)
    print(f"  模型 prefill (L={len(ids_bm)}): {t_prefill:.1f} ms")

    # Time one decode alone
    next_input = mx.array([[1]])
    t_decode_one = time_op(lambda: model(next_input, cache=cache_bm), warmup=1, repeats=50)
    print(f"  模型 decode (1 token, cached): {t_decode_one:.1f} ms")

    # Time full generate
    t0 = time.perf_counter()
    count = 0
    for _ in e1.generate(prompt, max_tokens=256, temperature=0.0):
        count += 1
    t_gen = time.perf_counter() - t0
    print(f"\n  Phase 1 完整生成: {count} tokens in {t_gen:.2f}s")
    print(f"    throughput = {count/t_gen:.1f} tok/s")

    # Estimate breakdown
    t_decode_total = (count - 1) * t_decode_one / 1000
    t_total_est = t_prefill / 1000 + t_decode_total
    overhead = t_gen - t_total_est
    print(f"\n  耗时拆解:")
    print(f"    prefill:        {t_prefill:.0f} ms ({t_prefill/t_gen/10:.1f}%)")
    print(f"    decode (x{count-1}): {t_decode_total:.2f}s ({t_decode_total/t_gen*100:.1f}%)")
    print(f"    overhead:       {overhead:.2f}s ({overhead/t_gen*100:.1f}%)")
    print(f"    ─ sampler + yield + 调度 + compiling")

    # ---- ======================================== ----
    print("\n" + "=" * 60)
    print("💾 内存分析")
    print("=" * 60)

    import psutil
    rss = psutil.Process().memory_info().rss / 1024**3
    try:
        active = mx.metal.get_active_memory() / 1024**3
        peak = mx.metal.get_peak_memory() / 1024**3
        cache_mem = mx.metal.get_cache_memory() / 1024**3
        print(f"  RSS:          {rss:.1f} GB")
        print(f"  Metal active: {active:.1f} GB")
        print(f"  Metal peak:   {peak:.1f} GB")
        print(f"  Metal cache:  {cache_mem:.1f} GB")
    except Exception as e:
        print(f"  RSS: {rss:.1f} GB (Metal mem API unavailable: {e})")

    # KV cache size
    n_kv = config.num_key_value_heads
    head_dim = config.head_dim
    kv_per_layer = 2 * n_kv * head_dim * 2  # 2x (K+V), float16
    kv_total = kv_per_layer * config.num_hidden_layers
    seq_len = len(ids_bm) + 256
    kv_memory = kv_total * seq_len / 1024**3
    print(f"\n  理论 KV cache (seq={seq_len}, {n_kv} kv_heads, {head_dim}d):")
    print(f"    每层: {kv_per_layer/1024:.1f} KB")
    print(f"    {config.num_hidden_layers} 层 × 2 (K+V): {kv_total/1024:.1f} KB")
    print(f"    {seq_len} 个位置: {kv_memory:.2f} GB")

    # ---- ======================================== ----
    print("\n" + "=" * 60)
    print("🎯 优化优先级评估")
    print("=" * 60)

    decode_ms = t_decode_one
    prefill_ms = t_prefill
    
    # Compare with MLX baseline
    mlx_baseline_decode = 55.6  # ms (from vllm-metal ref bench)
    mlx_bl_prefill = 238  # ms

    print(f"\n  算子级对比:")
    print(f"    prefill: ours={prefill_ms:.0f}ms, mlx_lm={mlx_bl_prefill}ms, gap={prefill_ms/mlx_bl_prefill-1:.0%}")
    print(f"    decode:  ours={decode_ms:.1f}ms, mlx_lm={mlx_baseline_decode}ms, gap={decode_ms/mlx_baseline_decode-1:.0%}")
    
    gap_decode = decode_ms - mlx_baseline_decode
    print(f"\n  decode 差距来源推测:")
    print(f"    - mx.compile 未包裹: 每步重建计算图 (+30-50%)")
    print(f"    - 掩码生成 (mx.triu): 每层每步重建 (+5-10%)")
    print(f"    - 无 float16 优化: 隐式 float32 提升 (+10-20%)")
    print(f"    - KVCache 访问模式: 未连续内存 (+5-10%)")

    print(f"\n  🔑 优先建议:")
    if gap_decode > 20:
        print(f"    1. mx.compile(decode_step) → 预期 +30-50% decode")
        print(f"    2. 固定掩码 (预计算) → 预期 +5-10%")
        print(f"    3. float16 全程 → 预期 +10-20%")
    
    print(f"\n=== 完成 ===")


if __name__ == "__main__":
    main()
