# Round 1 — O2 Unblock: Scheduler Fix + .item() Elimination

## Throughput Comparison

| 框架 | 耗时 | 吞吐率 | 差距 |
|------|------|--------|------|
| Target (Round 0) | 1.110s | 21.6 tok/s | baseline |
| Target (Round 1) | 1.089s | 22.0 tok/s | +1.9% |

(24 tokens, single GPU, warmup included, test_phase11_201_throughput.py)

## Optimization Process

| Step | Change | Elapsed | Throughput |
|------|--------|---------|------------|
| 0 | Round 0 (O1+O3+O6 applied, O2 blocked) | 1.110s | 21.6 tok/s |
| 1 | Fix scheduler double-increment (remove `seq.kv_len += 1`) | — | — |
| 2 | Replace `.item()` with `kv_len + 1` in forward_decode() | 1.089s | 22.0 tok/s |

## Applied Fixes

### Fix 1: Scheduler double-increment bug
**File**: `engine/framework/scheduler.py:172`
**Before**: `seq.kv_len += 1` (decode postprocess)
**After**: Removed — ModelRunner is now single source of truth for kv_len
**Why**: The model already returns updated `_kv_len_gpu` values from `forward_decode()`. Runner sets `s.kv_len` from these values. Scheduler incrementing again caused kv_len to advance 2 per decode step (4→5 from model, 5→6 from scheduler). The system worked because `flash_attn_with_kvcache` uses `cache_seqlens=self._kv_len_gpu` (correct GPU value), not the passed parameter.

### Fix 2: Eliminate .item() GPU sync from forward_decode()
**File**: `engine/models/qwen.py:948`
**Before**: `kv_lens = [int(l.self_attn._kv_len_gpu[0].item()) for l in self.layers]`
**After**: `kv_lens = [kv_len + 1]  # one decode token added, no GPU sync`
**Why**: Each decode step processes exactly one token, so KV cache length always increases by exactly 1. All 36 layers have identical kv_len because they all process the same sequence length. Reading from GPU was redundant GPU→CPU synchronization taking 5.3ms per step.

## Profiler Comparison (23 decode steps)

| Metric | Round 0 | Round 1 | Change |
|--------|---------|---------|--------|
| aten::item CPU time | 121.79 ms | 1.26 ms | **-99.0%** |
| aten::item per step | 5.30 ms | 0.05 ms | **-99.0%** |
| Self CPU time total | 1.430s | 1.339s | -6.4% |
| Self CUDA time total | 1.011s | 1.000s | -1.1% |
| Typical decode step CUDA | ~60-61 ms | ~54-58 ms | -5% |
| cudaMalloc in decode | 0 | 0 | ✅ |
| T202-CORRECT | FAIL | PASS | ✅ |

## Remaining Bottlenecks

| Priority | Category | Detail | Actionable? |
|----------|----------|--------|-------------|
| 1 | CPU overhead | 1.339s CPU > 1.000s CUDA | Partially — Python dispatch overhead |
| 2 | aten::copy_ | 3473 calls, 32.4ms CPU | Investigate unnecessary contiguous() |
| 3 | aten::index_copy_ | 1656 calls, 32.8ms CPU | Could replace with direct assignment |
| 4 | hipGetDeviceProperties | 3312 calls, 19.6ms CUDA | Likely rocBLAS-internal |

## Errors Encountered

1. **O2 .item() removal broke output (initial attempt, before scheduler fix)**: Output diverged from baseline because kv_len was being double-incremented. Removing .item() exposed the inconsistency. Fix: fix scheduler first, then apply O2.
