#!/usr/bin/env python3
"""极限性能剖析 — decode 单步 55ms 逐微秒拆解。

策略:
  - 完整调用用 time_single (逐次 eval+sync) 测量
  - 子算子用 time_batch (排队 N 次, 一次 sync) 测量以摊薄 eval 开销
  - 最终以完整调用为准, 子算子仅做比例参考

用法:
    cd subprojects/mac-engine
    python scripts/deep_profile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.kv_cache import KVCache, make_kv_cache
from src.model import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
from src.weights import load_qwen3_model
from src.tokenizer import Tokenizer
from src.sampler import greedy_sample

MODEL_PATH = str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B/")

H = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 12288
N_LAYERS = 36
VOCAB_SIZE = 151936


def time_single(fn, warmup=5, repeats=50):
    """Each iteration: fn() → mx.synchronize(). Returns avg ms."""
    for _ in range(warmup):
        fn()
    mx.synchronize()
    times_ns = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)
    return sum(times_ns) / len(times_ns) / 1e6


def time_batch(fn, warmup=3, repeats=50):
    """Queue all repeats, sync once. Returns avg ms per iteration."""
    for _ in range(warmup):
        fn()
    mx.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    mx.synchronize()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / repeats / 1e6


def main():
    print("=" * 72)
    print("  极限性能剖析: Decode Step 逐微秒拆解")
    print("=" * 72)

    # ── 1. Load ────────────────────────────────────────────────────
    print("\n[1/9] 加载模型...")
    model, config = load_qwen3_model(MODEL_PATH)
    tok = Tokenizer(MODEL_PATH)
    prompt = "Explain the key concepts of machine learning in detail."
    ids = tok.encode(prompt)
    input_ids = mx.array([ids])
    decode_token = mx.array([[ids[-1]]])
    print(f"  Prompt: {len(ids)} tokens")

    # ── 2. Real TPOT baseline ──────────────────────────────────────
    print("\n[2/9] 实测 TPOT 基线...")

    # Manual decode loop
    cache_gen = make_kv_cache(N_LAYERS)
    logits = model(input_ids, cache=cache_gen)
    mx.eval(logits)
    mx.synchronize()

    first_logits = logits[0, -1, :]
    next_id = int(mx.argmax(first_logits, axis=-1).item())

    decode_times = []
    for step in range(64):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        next_input = mx.array([[next_id]])
        logits = model(next_input, cache=cache_gen)
        mx.eval(logits)
        next_logits = logits[0, -1, :]
        next_id_arr = mx.argmax(next_logits, axis=-1)
        mx.eval(next_id_arr)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        next_id = int(next_id_arr.item())
        decode_times.append((t1 - t0) / 1e6)

    stable_tpot = sum(decode_times[5:]) / len(decode_times[5:])
    print(f"  稳态 TPOT (skip 5): {stable_tpot:.2f} ms → {1000/stable_tpot:.1f} tok/s")
    print(f"  全部 {len(decode_times)} 步: 平均 {sum(decode_times)/len(decode_times):.2f} ms")

    del cache_gen
    mx.clear_cache()

    # ── 3. Isolated model forward ──────────────────────────────────
    print("\n[3/9] 隔离 model forward (不含采样)...")

    cache_fwd = make_kv_cache(N_LAYERS)
    model(input_ids, cache=cache_fwd)
    mx.synchronize()

    fwd_times = []
    for i in range(30):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = model(decode_token, cache=cache_fwd)
        mx.eval(out)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        fwd_times.append((t1 - t0) / 1e6)

    fwd_stable = sum(fwd_times[3:]) / len(fwd_times[3:])
    print(f"  model forward: {fwd_stable:.2f} ms")
    print(f"  采样开销: {stable_tpot - fwd_stable:.2f} ms")

    del cache_fwd
    mx.clear_cache()

    # ── 4. Embedding ───────────────────────────────────────────────
    print("\n[4/9] Embedding lookup...")
    emb = model.model.embed_tokens
    t_emb = time_single(
        lambda: mx.eval(emb(decode_token)), warmup=5, repeats=100)
    print(f"  embed_tokens: {t_emb:.4f} ms")

    # ── 5. Per-layer timing ────────────────────────────────────────
    print(f"\n[5/9] 逐层计时 (×30 each)...")

    cache_layer = make_kv_cache(N_LAYERS)
    h = model.model.embed_tokens(input_ids)
    mx.eval(h)
    for layer, c in zip(model.model.layers, cache_layer):
        h = layer(h, mask=None, cache=c)
    mx.eval(h)
    mx.synchronize()

    h_dec = model.model.embed_tokens(decode_token)
    mx.eval(h_dec)
    mx.synchronize()

    layer_times = []
    for i, layer in enumerate(model.model.layers):
        def _fn(l=layer, hd=h_dec, c=cache_layer[i]):
            r = l(hd, mask=None, cache=c)
            mx.eval(r)
        t = time_single(_fn, warmup=3, repeats=30)
        layer_times.append(t)
        if i % 6 == 0 or i == N_LAYERS - 1:
            print(f"  Layer {i:2d}: {t:.3f} ms")

    total_layers = sum(layer_times)
    avg_layer = total_layers / N_LAYERS
    print(f"  总计: {total_layers:.2f} ms, 平均: {avg_layer:.3f} ms/层")

    del cache_layer
    mx.clear_cache()

    # ── 6. Layer 0 breakdown ───────────────────────────────────────
    print(f"\n[6/9] Layer 0 内部拆解...")

    # Setup layer 0 cache with prefill
    cache0 = make_kv_cache(1)
    _h = model.model.embed_tokens(input_ids)
    mx.eval(_h)
    _r = model.model.layers[0](_h, mask=None, cache=cache0[0])
    mx.eval(_r)
    mx.synchronize()

    block0 = model.model.layers[0]
    attn0 = block0.self_attn
    mlp0 = block0.mlp
    h_test = model.model.embed_tokens(decode_token)
    mx.eval(h_test)
    mx.synchronize()

    # Measure eval overhead baseline
    t_sync_only = time_single(lambda: mx.synchronize(), warmup=5, repeats=100)
    dummy = mx.array([1.0])
    t_eval_only = time_single(lambda: mx.eval(dummy), warmup=5, repeats=100)
    print(f"  mx.synchronize() 开销: {t_sync_only:.4f} ms")
    print(f"  mx.eval(dummy) 开销:   {t_eval_only:.4f} ms")
    eval_overhead = t_sync_only + t_eval_only
    print(f"  eval+sync 基线开销:    {eval_overhead:.4f} ms (需从子算子测量中扣除)")

    # ─── Full module calls (single eval each) ───
    # Rebuild cache0 for each measurement
    def _fresh_cache0():
        c = make_kv_cache(1)
        _x = model.model.embed_tokens(input_ids)
        _y = model.model.layers[0](_x, mask=None, cache=c[0])
        mx.eval(_y)
        mx.synchronize()
        return c

    # input_layernorm
    t_in_norm = time_single(
        lambda: mx.eval(block0.input_layernorm(h_test)),
        warmup=5, repeats=100)
    x_normed = block0.input_layernorm(h_test)
    mx.eval(x_normed)
    mx.synchronize()

    # Attention (full)
    c_attn = _fresh_cache0()
    t_attn = time_single(
        lambda: mx.eval(attn0(x_normed, mask=None, cache=c_attn[0])),
        warmup=3, repeats=30)
    print(f"\n  attention (full): {t_attn:.4f} ms (含 eval 开销)")
    print(f"  attention (估 GPU): {t_attn - eval_overhead:.4f} ms")

    # Residual
    c_res = _fresh_cache0()
    r_attn = attn0(x_normed, mask=None, cache=c_res[0])
    mx.eval(r_attn)
    mx.synchronize()
    t_res1 = time_single(
        lambda: mx.eval(h_test + r_attn),
        warmup=5, repeats=100)

    h_after = h_test + r_attn
    mx.eval(h_after)
    mx.synchronize()

    # post_attn_layernorm
    t_post_norm = time_single(
        lambda: mx.eval(block0.post_attention_layernorm(h_after)),
        warmup=5, repeats=100)
    x_mlp = block0.post_attention_layernorm(h_after)
    mx.eval(x_mlp)
    mx.synchronize()

    # MLP (full)
    t_mlp = time_single(
        lambda: mx.eval(mlp0(x_mlp)),
        warmup=3, repeats=30)
    print(f"  MLP (full): {t_mlp:.4f} ms (含 eval 开销)")
    print(f"  MLP (估 GPU): {t_mlp - eval_overhead:.4f} ms")

    # ─── Sub-operation batch timing (amortize eval) ───
    print("\n  子算子 batch 测量 (摊薄 eval 开销):")

    # Attention sub-ops
    t_q_proj_b = time_batch(
        lambda: mx.eval(attn0.q_proj(x_normed)), warmup=3, repeats=100)
    t_k_proj_b = time_batch(
        lambda: mx.eval(attn0.k_proj(x_normed)), warmup=3, repeats=100)
    t_v_proj_b = time_batch(
        lambda: mx.eval(attn0.v_proj(x_normed)), warmup=3, repeats=100)
    print(f"    q_proj: {t_q_proj_b:.4f} ms")
    print(f"    k_proj: {t_k_proj_b:.4f} ms")
    print(f"    v_proj: {t_v_proj_b:.4f} ms")

    q_raw = attn0.q_proj(x_normed)
    k_raw = attn0.k_proj(x_normed)
    v_raw = attn0.v_proj(x_normed)
    mx.eval(q_raw, k_raw, v_raw)
    mx.synchronize()
    B, L, D = x_normed.shape
    q = q_raw.reshape(B, L, attn0.n_heads, attn0.head_dim)
    k = k_raw.reshape(B, L, attn0.n_kv_heads, attn0.head_dim)
    v = v_raw.reshape(B, L, attn0.n_kv_heads, attn0.head_dim)
    mx.eval(q, k, v)
    mx.synchronize()

    t_q_norm_b = time_batch(
        lambda: mx.eval(attn0.q_norm(q)), warmup=3, repeats=100)
    t_k_norm_b = time_batch(
        lambda: mx.eval(attn0.k_norm(k)), warmup=3, repeats=100)
    print(f"    q_norm: {t_q_norm_b:.4f} ms")
    print(f"    k_norm: {t_k_norm_b:.4f} ms")

    q_n = attn0.q_norm(q).transpose(0, 2, 1, 3)
    k_n = attn0.k_norm(k).transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)
    mx.eval(q_n, k_n, v_t)
    mx.synchronize()

    offset = cache0[0].offset
    t_rope_q_b = time_batch(
        lambda: mx.eval(attn0.rope(q_n, offset=offset)), warmup=3, repeats=100)
    t_rope_k_b = time_batch(
        lambda: mx.eval(attn0.rope(k_n, offset=offset)), warmup=3, repeats=100)
    print(f"    rope_q: {t_rope_q_b:.4f} ms")
    print(f"    rope_k: {t_rope_k_b:.4f} ms")

    q_r = attn0.rope(q_n, offset=offset)
    k_r = attn0.rope(k_n, offset=offset)
    mx.eval(q_r, k_r)
    mx.synchronize()

    # KV cache update
    t_kv_b = time_batch(
        lambda: mx.eval(*cache0[0].update_and_fetch(k_r, v_t)),
        warmup=3, repeats=50)
    print(f"    cache.update: {t_kv_b:.4f} ms")

    k_full, v_full = cache0[0].update_and_fetch(k_r, v_t)
    mx.eval(k_full, v_full)
    mx.synchronize()

    # SDPA
    t_sdpa_b = time_batch(
        lambda: mx.eval(mx.fast.scaled_dot_product_attention(
            q_r, k_full, v_full, scale=attn0.scale)),
        warmup=3, repeats=100)
    print(f"    SDPA: {t_sdpa_b:.4f} ms")

    attn_out = mx.fast.scaled_dot_product_attention(
        q_r, k_full, v_full, scale=attn0.scale)
    attn_out_r = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    mx.eval(attn_out_r)
    mx.synchronize()

    t_o_proj_b = time_batch(
        lambda: mx.eval(attn0.o_proj(attn_out_r)), warmup=3, repeats=100)
    print(f"    o_proj: {t_o_proj_b:.4f} ms")

    attn_sub_batch = (t_q_proj_b + t_k_proj_b + t_v_proj_b + t_q_norm_b +
                      t_k_norm_b + t_rope_q_b + t_rope_k_b + t_kv_b +
                      t_sdpa_b + t_o_proj_b)
    print(f"  Attention 子算子 batch 合计: {attn_sub_batch:.4f} ms")
    print(f"  Attention 完整调用:          {t_attn:.4f} ms")

    # MLP sub-ops
    print("\n  MLP 子算子 batch 测量:")
    t_gate_b = time_batch(
        lambda: mx.eval(mlp0.gate_proj(x_mlp)), warmup=3, repeats=100)
    t_up_b = time_batch(
        lambda: mx.eval(mlp0.up_proj(x_mlp)), warmup=3, repeats=100)
    print(f"    gate_proj: {t_gate_b:.4f} ms")
    print(f"    up_proj:   {t_up_b:.4f} ms")

    gate_out = mlp0.gate_proj(x_mlp)
    mx.eval(gate_out)
    mx.synchronize()
    t_silu_b = time_batch(
        lambda: mx.eval(nn.silu(gate_out)), warmup=3, repeats=100)
    print(f"    silu:      {t_silu_b:.4f} ms")

    up_out = mlp0.up_proj(x_mlp)
    silu_out = nn.silu(gate_out)
    mx.eval(up_out, silu_out)
    mx.synchronize()
    t_mul_b = time_batch(
        lambda: mx.eval(silu_out * up_out), warmup=3, repeats=100)
    print(f"    mul:       {t_mul_b:.4f} ms")

    activated = silu_out * up_out
    mx.eval(activated)
    mx.synchronize()
    t_down_b = time_batch(
        lambda: mx.eval(mlp0.down_proj(activated)), warmup=3, repeats=100)
    print(f"    down_proj: {t_down_b:.4f} ms")

    mlp_sub_batch = t_gate_b + t_up_b + t_silu_b + t_mul_b + t_down_b
    print(f"  MLP 子算子 batch 合计: {mlp_sub_batch:.4f} ms")
    print(f"  MLP 完整调用:          {t_mlp:.4f} ms")

    # ── 7. Final Norm + LM Head + Sampling ────────────────────────
    print(f"\n[7/9] Final Norm + LM Head + Sampling...")

    cache_final = make_kv_cache(N_LAYERS)
    _h = model.model.embed_tokens(input_ids)
    for layer, c in zip(model.model.layers, cache_final):
        _h = layer(_h, mask=None, cache=c)
    mx.eval(_h)
    mx.synchronize()
    h_dec2 = model.model.embed_tokens(decode_token)
    for layer, c in zip(model.model.layers, cache_final):
        h_dec2 = layer(h_dec2, mask=None, cache=c)
    mx.eval(h_dec2)
    mx.synchronize()

    t_final_norm = time_single(
        lambda: mx.eval(model.model.norm(h_dec2)),
        warmup=5, repeats=100)
    h_normed = model.model.norm(h_dec2)
    mx.eval(h_normed)
    mx.synchronize()

    t_lm_head = time_single(
        lambda: mx.eval(model.lm_head(h_normed)),
        warmup=5, repeats=30)
    logits = model.lm_head(h_normed)
    mx.eval(logits)
    mx.synchronize()
    logits_last = logits[0, -1, :]

    t_argmax = time_single(
        lambda: mx.eval(mx.argmax(logits_last, axis=-1)),
        warmup=5, repeats=100)

    # int(arr.item())
    argmax_r = mx.argmax(logits_last, axis=-1)
    mx.eval(argmax_r)
    mx.synchronize()
    t_item = 0.0
    for _ in range(50):
        t0 = time.perf_counter_ns()
        _ = int(argmax_r.item())
        t_item += (time.perf_counter_ns() - t0) / 1e6
    t_item /= 50

    print(f"  final_norm: {t_final_norm:.4f} ms")
    print(f"  lm_head:    {t_lm_head:.4f} ms")
    print(f"  argmax:     {t_argmax:.4f} ms")
    print(f"  item():     {t_item:.4f} ms")

    del cache_final
    mx.clear_cache()

    # ── 8. Python overhead ────────────────────────────────────────
    print(f"\n[8/9] Python 开销...")
    t_tok_decode = 0.0
    for _ in range(50):
        t0 = time.perf_counter_ns()
        tok.decode([1, 2, 3, 4, 5])
        t_tok_decode += (time.perf_counter_ns() - t0) / 1e6
    t_tok_decode /= 50
    print(f"  tokenizer.decode(5): {t_tok_decode:.4f} ms")

    # ── 9. Memory bandwidth ───────────────────────────────────────
    print(f"\n[9/9] 内存带宽...")
    dev_info = mx.device_info()
    print(f"  Device: {dev_info.get('name', 'unknown')}")
    print(f"  Memory: {dev_info.get('memory_size', 0) / 1e9:.1f} GB")

    # Bandwidth: linear projection as a proxy
    # q_proj: [1,1,4096] × [4096, 4096] → read 4096*4096*2 = 32MB
    # Time it takes to do this gives us: bytes_read / time = effective bandwidth
    q_proj_bytes = 4096 * 4096 * 2  # bf16
    k_proj_bytes = 4096 * 1024 * 2
    v_proj_bytes = 4096 * 1024 * 2
    o_proj_bytes = 4096 * 4096 * 2
    gate_proj_bytes = 4096 * 12288 * 2
    up_proj_bytes = 4096 * 12288 * 2
    down_proj_bytes = 12288 * 4096 * 2
    lm_head_bytes = 4096 * 151936 * 2

    # Use the batch measurements for bandwidth estimation
    # batch measurement approximates GPU-only time (amortized eval)
    all_projs = {
        "q_proj (4096→4096)": (t_q_proj_b, q_proj_bytes),
        "k_proj (4096→1024)": (t_k_proj_b, k_proj_bytes),
        "v_proj (4096→1024)": (t_v_proj_b, v_proj_bytes),
        "o_proj (4096→4096)": (t_o_proj_b, o_proj_bytes),
        "gate_proj (4096→12288)": (t_gate_b, gate_proj_bytes),
        "up_proj (4096→12288)": (t_up_b, up_proj_bytes),
        "down_proj (12288→4096)": (t_down_b, down_proj_bytes),
    }

    print("\n  线性层带宽效率:")
    for name, (t_ms, nbytes) in all_projs.items():
        bw = nbytes / 1e6 / t_ms  # MB/ms = GB/s
        print(f"    {name}: {t_ms:.4f} ms → {bw:.1f} GB/s")

    # LM head bandwidth
    t_lm_b = time_batch(
        lambda: mx.eval(model.lm_head(h_normed)), warmup=3, repeats=30)
    lm_bw = lm_head_bytes / 1e6 / t_lm_b
    print(f"    lm_head (4096→151936): {t_lm_b:.4f} ms → {lm_bw:.1f} GB/s")

    # Average effective bandwidth across large projections
    large_projs = [(t_gate_b, gate_proj_bytes), (t_up_b, up_proj_bytes),
                   (t_down_b, down_proj_bytes)]
    total_bytes = sum(b for _, b in large_projs)
    total_proj_time = sum(t for t, _ in large_projs)
    avg_bw = total_bytes / 1e6 / total_proj_time
    print(f"\n  MLP 层平均有效带宽: {avg_bw:.1f} GB/s")

    # Metal memory
    try:
        active = mx.get_active_memory() / 1024**3
        peak = mx.get_peak_memory() / 1024**3
        print(f"  Metal active: {active:.2f} GB, peak: {peak:.2f} GB")
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    tpot = stable_tpot

    # Estimate real GPU time per component
    # Full-call measurements minus eval overhead give GPU-only time
    # But we don't know exact eval overhead for each call
    # Instead, use layer_times (full layer evals) as the ground truth
    # and proportionally split using sub-op ratios

    # Proportions from batch sub-op measurements
    attn_gpu_est = t_attn - eval_overhead  # single eval overhead
    mlp_gpu_est = t_mlp - eval_overhead
    norm_gpu_est = t_in_norm - eval_overhead
    post_norm_gpu_est = t_post_norm - eval_overhead
    res_gpu_est = t_res1 - eval_overhead

    # Ensure non-negative
    attn_gpu_est = max(attn_gpu_est, 0)
    mlp_gpu_est = max(mlp_gpu_est, 0)
    norm_gpu_est = max(norm_gpu_est, 0)
    post_norm_gpu_est = max(post_norm_gpu_est, 0)
    res_gpu_est = max(res_gpu_est, 0)

    single_layer_gpu = attn_gpu_est + mlp_gpu_est + norm_gpu_est + post_norm_gpu_est + res_gpu_est * 2

    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║  Decode Step 时间分配表 (Layer 0 拆解 × 36 层)                    ║")
    print("╚" + "═" * 70 + "╝")
    print()

    def row(label, ms, total):
        pct = ms / total * 100 if total > 0 else 0
        print(f"  {label:<40s} {ms:>7.2f} ms ({pct:5.1f}%)")

    print(f"  实测 TPOT: {tpot:.1f} ms  |  model forward: {fwd_stable:.1f} ms  |  采样+开销: {tpot - fwd_stable:.1f} ms")
    print("  " + "━" * 66)

    row("Embedding lookup", t_emb, tpot)
    print("  " + "─" * 66)

    for i, t in enumerate(layer_times):
        row(f"  Layer {i:2d}", t, tpot)

    print("  " + "─" * 66)
    print(f"  {'  36 层小计':<40s} {total_layers:>7.2f} ms ({total_layers/tpot*100:5.1f}%)")
    print("  " + "─" * 66)

    # Attention breakdown (estimated from layer proportions)
    row("  其中 Attention (估)", attn_gpu_est * N_LAYERS, tpot)
    row("  其中 MLP (估)", mlp_gpu_est * N_LAYERS, tpot)
    row("  其中 RMSNorm×2 (估)", (norm_gpu_est + post_norm_gpu_est) * N_LAYERS, tpot)
    row("  其中 residual×2 (估)", res_gpu_est * 2 * N_LAYERS, tpot)
    print("  " + "─" * 66)

    # Attention sub-op breakdown (from batch measurements)
    attn_total_batch = attn_sub_batch
    qkv_pct = (t_q_proj_b + t_k_proj_b + t_v_proj_b) / attn_total_batch
    norm_pct = (t_q_norm_b + t_k_norm_b) / attn_total_batch
    rope_pct = (t_rope_q_b + t_rope_k_b) / attn_total_batch
    kv_pct = t_kv_b / attn_total_batch
    sdpa_pct = t_sdpa_b / attn_total_batch
    o_proj_pct = t_o_proj_b / attn_total_batch

    attn_total_gpu = attn_gpu_est * N_LAYERS
    row(f"    qkv_proj ({qkv_pct*100:.0f}%)", attn_total_gpu * qkv_pct, tpot)
    row(f"    qk_norm ({norm_pct*100:.0f}%)", attn_total_gpu * norm_pct, tpot)
    row(f"    RoPE ({rope_pct*100:.0f}%)", attn_total_gpu * rope_pct, tpot)
    row(f"    KV cache ({kv_pct*100:.0f}%)", attn_total_gpu * kv_pct, tpot)
    row(f"    SDPA ({sdpa_pct*100:.0f}%)", attn_total_gpu * sdpa_pct, tpot)
    row(f"    o_proj ({o_proj_pct*100:.0f}%)", attn_total_gpu * o_proj_pct, tpot)

    print("  " + "─" * 66)
    # MLP sub-op breakdown
    mlp_total_batch = mlp_sub_batch
    gate_pct = t_gate_b / mlp_total_batch
    up_pct = t_up_b / mlp_total_batch
    silu_mul_pct = (t_silu_b + t_mul_b) / mlp_total_batch
    down_pct = t_down_b / mlp_total_batch

    mlp_total_gpu = mlp_gpu_est * N_LAYERS
    row(f"    gate_proj ({gate_pct*100:.0f}%)", mlp_total_gpu * gate_pct, tpot)
    row(f"    up_proj ({up_pct*100:.0f}%)", mlp_total_gpu * up_pct, tpot)
    row(f"    silu+mul ({silu_mul_pct*100:.0f}%)", mlp_total_gpu * silu_mul_pct, tpot)
    row(f"    down_proj ({down_pct*100:.0f}%)", mlp_total_gpu * down_pct, tpot)

    print("  " + "─" * 66)
    row("Final RMSNorm", t_final_norm, tpot)
    row("LM Head (4096→151936)", t_lm_head, tpot)
    row("Argmax + item()", t_argmax + t_item, tpot)
    row("Tokenizer decode", t_tok_decode, tpot)
    print("  " + "━" * 66)

    measured_sum = t_emb + total_layers + t_final_norm + t_lm_head + t_argmax + t_item
    unaccounted = tpot - measured_sum
    row("Measured total", measured_sum, tpot)
    row("Unaccounted", unaccounted, tpot)

    # ── Category summary ──────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  算子类型占比                                        │")
    print("  └─────────────────────────────────────────────────────┘")

    # Use layer_times as ground truth for per-layer cost
    attn_total_ms = attn_gpu_est * N_LAYERS
    mlp_total_ms = mlp_gpu_est * N_LAYERS
    norm_total_ms = (norm_gpu_est + post_norm_gpu_est) * N_LAYERS + t_final_norm
    res_total_ms = res_gpu_est * 2 * N_LAYERS

    print(f"  Attention (36层):     {attn_total_ms:>7.2f} ms ({attn_total_ms/tpot*100:5.1f}%)")
    print(f"  MLP (36层):           {mlp_total_ms:>7.2f} ms ({mlp_total_ms/tpot*100:5.1f}%)")
    print(f"  RMSNorm (36×2 + 1):   {norm_total_ms:>7.2f} ms ({norm_total_ms/tpot*100:5.1f}%)")
    print(f"  Residual (36×2):      {res_total_ms:>7.2f} ms ({res_total_ms/tpot*100:5.1f}%)")
    print(f"  LM Head:              {t_lm_head:>7.2f} ms ({t_lm_head/tpot*100:5.1f}%)")
    print(f"  Sampling:             {t_argmax+t_item:>7.2f} ms ({(t_argmax+t_item)/tpot*100:5.1f}%)")
    print(f"  Embed + tokenizer:    {t_emb+t_tok_decode:>7.2f} ms ({(t_emb+t_tok_decode)/tpot*100:5.1f}%)")

    other = tpot - attn_total_ms - mlp_total_ms - norm_total_ms - res_total_ms - t_lm_head - t_argmax - t_item - t_emb
    print(f"  未计入:               {other:>7.2f} ms ({other/tpot*100:5.1f}%)")

    # ── Bandwidth analysis ─────────────────────────────────────────
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  带宽瓶颈分析                                        │")
    print("  └─────────────────────────────────────────────────────┘")

    attn_weight_mb = (q_proj_bytes + k_proj_bytes + v_proj_bytes + o_proj_bytes) / 1e6
    mlp_weight_mb = (gate_proj_bytes + up_proj_bytes + down_proj_bytes) / 1e6
    lm_head_mb = lm_head_bytes / 1e6
    total_weight_mb = (attn_weight_mb + mlp_weight_mb) * N_LAYERS + lm_head_mb

    print(f"  每层权重: Attention {attn_weight_mb:.1f} MB + MLP {mlp_weight_mb:.1f} MB = {attn_weight_mb+mlp_weight_mb:.1f} MB")
    print(f"  36层 + LM Head: {total_weight_mb:.0f} MB = {total_weight_mb/1024:.2f} GB")

    if avg_bw > 0:
        theory_min = total_weight_mb / avg_bw  # MB / (MB/ms) = ms
        print(f"  理论最低延迟 (@{avg_bw:.0f} GB/s): {theory_min:.1f} ms")
        print(f"  实测 TPOT: {tpot:.1f} ms")
        print(f"  带宽利用率 (线性层部分): {theory_min / tpot * 100:.1f}%")

    print()
    print(f"{'=' * 72}")
    print("  剖析完成")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
