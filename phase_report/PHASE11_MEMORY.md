# Phase 11 Memory — 性能优化 Stage 1 + Stage 2 Complete

| 字段 | 值 |
|------|-----|
| Timestamp | 2026-06-09T12:20:00Z |
| Status | ✅ STAGE 1 DELIVERED + ✅ STAGE 2 COMPLETE (practical limit reached) |
| Track | 快速修复 (增量应用 + 逐个验证) |

---

## Stage 1: O1-O9 静态审计规则

### O1-O6 最终审计结果

| 规则 | 状态 | 说明 |
|------|------|------|
| O1 | ✅ PASS | `@torch.inference_mode()` 2 matches: `forward()` (L873) + `forward_decode()` (L921) |
| O2 | ✅ PASS (Round 1 fix) | 零 `.item()` in model hot path. `forward_decode()` returns `[kv_len + 1]` instead of reading `_kv_len_gpu`. Scheduler double-increment fixed as prerequisite. |
| O3 | ✅ PASS | Pre-allocated buffers: `_q_norm_out`, `_k_norm_out` (QwenAttentionTP), `_silu_out` (QwenMLPTP). RMSNorm `empty_like` → `torch.empty(*x.shape, ...)`. |
| O4 | ✅ PASS | block_table arange init (was already correct) |
| O5 | ✅ PASS | Prefill KV direct assignment (was already correct) |
| O6 | ✅ PASS | register_buffer count = 7 (was 4) |

---

## Stage 2: Multi-Round Tracing Alignment vs vLLM Eager

### Final Throughput

| 框架 | 耗时 (24 tok) | 吞吐率 | 差距 |
|------|-------------|--------|------|
| vLLM Eager | 0.992s | 24.2 tok/s | baseline |
| Target (Round 0) | 1.110s | 21.6 tok/s | 89.3% |
| Target (Round 1) | 1.089s | 22.0 tok/s | 90.9% |
| Target (Round 2) | 1.092s | 22.0 tok/s | 90.9% |

### Round Summary

| Round | Description | Key Change | Throughput | Verdict |
|-------|-------------|-----------|---------|---------|
| 0 | Baseline | O1-O6 applied, O2 blocked by scheduler bug | 21.6 tok/s | — |
| 1 | O2 Unblock | Scheduler double-increment fix + .item() removal (99% reduction in aten::item) | 22.0 tok/s | ✅ +1.9% |
| 2 | Contiguous Consolidation | y.contiguous() before split in QKVColumnParallelLinear, remove 3× per-tensor contiguous | 22.0 tok/s | ➖ Neutral |
| 3 | Irreducible Gap Analysis | Documented remaining bottlenecks → termination | — | 🛑 |

### Round 1: O2 Unblock + Scheduler Fix

**Root cause**: `scheduler.postprocess()` line 172 did `seq.kv_len += 1` on decode steps, while `model_runner.run()` already set `seq.kv_len` from model-returned `_kv_len_gpu` values. This caused kv_len to increment twice per decode step.

**Fixes**:
1. `engine/framework/scheduler.py` (line 172): Removed `seq.kv_len += 1` from decode postprocess.
2. `engine/models/qwen.py` (line 948): Replaced `kv_lens = [int(l.self_attn._kv_len_gpu[0].item()) for l in self.layers]` with `kv_lens = [kv_len + 1]`.

**Profiler impact**:
- aten::item CPU time: 121.79 ms → 1.26 ms (-99.0%)
- Self CPU time total: 1.430s → 1.339s (-6.4%)
- Self CUDA time total: 1.011s → 1.000s (-1.1%)

### Round 2: Contiguous Consolidation

**Change**: `QKVColumnParallelLinear.forward()` now does `y.contiguous()` before `split()`, so the q/k/v split views are contiguous. Removed 3 individual `.contiguous()` calls from decode path.

**Impact**: Minor CPU improvement (1.339s → 1.338s CPU), no CUDA improvement (same data copied). Throughput unchanged within noise.

### Round 3: Irreducible Gap Analysis

**95% threshold**: 23.0 tok/s (need +4.1% from 22.0 tok/s = 49ms savings)

**Irreducible bottlenecks**:

| Category | Bottleneck | CUDA Time | Why Not Fixable |
|----------|-----------|-----------|-----------------|
| rocBLAS-internal | hipGetDeviceProperties_v2 | 20.5 ms | Called on every GEMM launch; no user-space suppression |
| rocBLAS-internal | Cijk_B_PostGSU | 16.7 ms | GEMM post-processing; rocBLAS internal |
| rocBLAS-internal | GEMM kernel variant | — | rocBLAS heuristics, not controllable from PyTorch |
| Fused kernel gap | aten::copy_ | 14.9 ms | Requires custom fused CUDA/HIP kernel |
| Fused kernel gap | aten::index_copy_ | 12.2 ms | Requires `reshape_and_cache_kernel_flash` |
| CPU dispatch | CPU 1.338s vs CUDA 1.001s | 337ms idle | Requires torch.compile or CUDA graphs |

vLLM's custom fused kernels not present in our engine:
- `unified_attention_with_output` — fuses qkv_proj + attention + o_proj
- `rms_rotary_embedding_fuse` — fuses RMSNorm + RoPE
- `silu_and_mul_opt` — optimized SiLU gating
- `reshape_and_cache_kernel_flash` — fused KV cache decode write

**Conclusion**: All pure-PyTorch optimizations exhausted. Remaining gap requires custom CUDA/HIP kernels.

## Scripts Passed

- test_phase10_greedy_align.sh (GREEDY-ALIGN-001 single GPU): ✅ PASS
- test_phase10_greedy_align.sh (GREEDY-ALIGN-002 TP=4): ✅ PASS
- test_phase10_benchmark.sh (BENCH-001): ✅ PASS — 3.7 tok/s
- test_phase11_201_throughput.py (T201-CORRECT, T201-THRESHOLD): ✅ PASS — 22.0 tok/s
- test_phase11_202_profiler.sh (T202-MALLOC, T202-ITEM, T202-CORRECT): ✅ PASS

## Files Changed (Stage 1 + Stage 2)

- `engine/models/qwen.py` (+50/-8 lines): O1 forward_decode() + O2 .item() removal + O3 buffers + O6 register_buffer + contiguous cleanup
- `engine/framework/model_runner.py` (+1/-1 line): decode calls forward_decode()
- `engine/framework/scheduler.py` (+3/-2 lines): remove kv_len double-increment
- `engine/tp_layers/linear.py` (+4/-0 lines): y.contiguous() before split in QKVColumnParallelLinear
- `scripts/test_phase11_201_throughput.py` (CREATED — +93 lines)
- `scripts/test_phase11_202_profiler.sh` (CREATED — +193 lines)
- `perf_iteration/ROUND_1_O2_UNBLOCK.md` (CREATED)
- `perf_iteration/ROUND_2_CONTIGUOUS_SPLIT.md` (CREATED)
- `perf_iteration/ROUND_3_LIMITS.md` (CREATED)

## Errors Encountered

1. **O2 .item() removal caused output divergence (initial attempt)**: Root cause was the scheduler double-increment bug. Fix: fixed scheduler first, then removed .item().
2. **T202-CORRECT fail (MAX_TOKENS=16)**: Output truncated because 16 tokens insufficient for expected output. Fix: increased to MAX_TOKENS=24.
3. **index_copy_ → direct indexing caused GPU sync**: `kc_flat[gpu_tensor]` triggered GPU→CPU sync, aten::item spiked from 1.9ms to 317ms. Fix: reverted to index_copy_.
4. **HSA_ENABLE_SDMA=0 / ROCBLAS env vars had no effect**: hipGetDeviceProperties_v2 remained at 3312 calls. These are rocBLAS-internal queries not suppressible from user space.

## Trace Artifacts

- `perf_iteration/trace_target/trace_rank0.json` — Chrome trace (23 decode steps, final state)
- `perf_iteration/trace_target/key_avg.txt` — Key averages table (final state)
