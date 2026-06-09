# Round 2 — Consolidate qkv contiguous into single call

## Throughput Comparison

| 框架 | 耗时 | 吞吐率 | 差距 |
|------|------|--------|------|
| Target (Round 1) | 1.089s | 22.0 tok/s | baseline |
| Target (Round 2) | 1.092s | 22.0 tok/s | +0.0% |

(24 tokens, single GPU, warmup included, test_phase11_201_throughput.py)

## Optimization Process

| Step | Change | Elapsed | Throughput |
|------|--------|---------|------------|
| 0 | Round 1 (O2 unblocked) | 1.089s | 22.0 tok/s |
| 1 | y.contiguous() before split, remove 3× per-tensor contiguous | 1.092s | 22.0 tok/s |

## Applied Fix

### Consolidate qkv contiguous calls
**File**: `engine/tp_layers/linear.py:329`, `engine/models/qwen.py:358-362`
**Before**: `F.linear` → `split()` → 3× `.contiguous()` on q, k, v individually
**After**: `F.linear` → `.contiguous()` on y → `split()` → views already contiguous
**Why**: rocBLAS may return non-contiguous `F.linear` output. Previously, `split()` inherited non-contiguous layout and each of q/k/v needed its own `.contiguous()` call. Making the parent tensor contiguous first produces contiguous split views, reducing 3 contiguous calls to 1 per layer per decode step.

## Profiler Comparison (23 decode steps)

| Metric | Round 1 | Round 2 | Change |
|--------|---------|---------|--------|
| Self CPU time total | 1.354s | 1.338s | -1.2% |
| Self CUDA time total | 1.003s | 1.001s | -0.2% |
| aten::copy_ CPU time | 36.2 ms | 32.3 ms | -10.8% |
| aten::copy_ CUDA time | 15.1 ms | 14.9 ms | -1.3% |
| aten::copy_ calls | 3473 | 3473 | unchanged* |
| aten::index_copy_ CUDA | 12.2 ms | 12.2 ms | unchanged |
| T202-CORRECT | PASS | PASS | ✅ |

*Call count unchanged because per-tensor `.contiguous()` on already-contiguous tensors is a no-op (not profiled as copy_). The remaining 3473 copy_ calls come from y.contiguous() (828) + RMSNorm x.contiguous() (~1656) + prefill KV cache write (~20) + other sources.

## Analysis

This optimization reduces CPU dispatch overhead (3 calls → 1 call per qkv_proj) but does NOT reduce GPU copy volume — the same data is copied either way. The 1.2% CPU improvement is within measurement noise for end-to-end throughput.

Key insight: `.contiguous()` on an already-contiguous tensor is NOT profiled as `aten::copy_` (it's a no-op that returns self). This means the per-tensor q/k/v `.contiguous()` calls WERE triggering actual copies in the previous code, all 3 of them. Consolidating to 1 y.contiguous() copies the same total bytes (q_size + kv_size + kv_size) in a single memcpy, which has the same GPU throughput.

## Errors Encountered

None.
