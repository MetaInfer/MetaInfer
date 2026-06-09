# Phase 10 Memory — E2E Acceptance

| 字段 | 值 |
|------|-----|
| Timestamp | 2026-06-09T11:30:00Z |
| Status | ✅ DELIVERED |
| Track | 完整串行 + 快速修复 (CustomAR gate + log suppression) |

## Scripts Passed
- test_phase10_no_compile_check.sh: ✅ PASS
- test_phase10_vs_vllm_compare.sh: ✅ PASS
- test_phase10_benchmark.sh: ✅ PASS — 3.9 tok/s single GPU
- test_phase10_greedy_align.sh (GREEDY-ALIGN-001 single GPU): ✅ PASS — output matches baseline exactly
- test_phase10_greedy_align.sh (GREEDY-ALIGN-002 TP=4): ✅ PASS — output matches baseline exactly

## Files Changed

### Core fix — CustomAR safety gate
- `engine/tp_layers/distributed.py` (+28 lines): Added `_USE_CUSTOMAR` flag defaulting to False. CustomAR kernel (`ops.all_reduce`) triggers `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on ROCm 6.3.3 + RCCL 2.22.3 when called after rocBLAS GEMM operations (F.linear). Standalone CustomAR on fresh tensors works; any CustomAR after GEMM crashes with SIGABRT. NCCL fallback (`dist.all_reduce`) is reliable — verified with 73+ all_reduce calls across full 36-layer TP=4 prefill + decode. CustomAR re-enable: `export META_INFER_CUSTOMAR_ENABLE=1`.
- Evidence: 5 diagnostic scripts confirmed the GEMM→CustomAR interaction is the exact crash point.

### Log suppression (stdout pollution fixes)
- `llm_engine.py` (+4 lines): Set `VLLM_LOGGING_LEVEL=ERROR` and `NCCL_DEBUG=WARN` at module level (before engine imports). vLLM DEBUG logs and RCCL NCCL INFO messages were polluting stdout, breaking test output parsing.
- `scripts/test_phase10_greedy_align.sh` (+4/-3 lines): Added `NCCL_DEBUG=WARN` to both Python templates. Fixed TP=4 stdout extraction to filter RCCL/NCCL diagnostic lines (`worker`, `NCCL`, `RCCL`, `HIP version`, `ROCm version`, `Hostname`, `Librccl`) using extended `grep -vE`.
- `scripts/test_phase10_benchmark.sh` (+1 line): Added `NCCL_DEBUG=WARN` to inline Python.

### Previous changes (carried forward)
- `openai_tp_server.py` (CREATED — +413 lines)
- `llm_engine.py` (+8 lines: empty prompt validation in generate() + _enqueue(), init_tp_distributed() idempotent guard)
- `engine/tp_layers/distributed.py` (+3 lines: init_tp_distributed() idempotent guard)
- `scripts/test_phase10_benchmark.sh` (bc→awk float comparison)
- `scripts/test_phase10_greedy_align.sh` (platform-aware GPU detection: nvidia-smi → rocm-smi → torch.cuda)

## TP=4 Crash Root Cause (FOUND)

**Location**: `engine/tp_layers/distributed.py:222` — `ops.all_reduce(_custom_ar_handle, x, out, _buf_ptrs[rank], _max_size)`

**Symptom**: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` → SIGABRT

**Root cause**: vLLM `ops.all_reduce` CustomAR kernel crashes on ROCm 6.3.3 + RCCL 2.22.3 + Hygon K500SM_AI when called with tensor data produced by rocBLAS GEMM operations (F.linear). rocBLAS workspace memory appears to overlap with CustomAR IPC buffer region, causing GPU memory access violation.

**Evidence chain**:
1. `_debug_tp4_allreduce_repeat.py`: Simple all_reduce × 2 on randn → PASS. F.linear + all_reduce → CRASH
2. `_debug_tp4_nccl_vs_customar.py`: PATH A (5 CustomAR calls on randn) → PASS. PATH B (10 NCCL calls) → PASS. PATH C (re-init CustomAR + QKV + o_proj) → CRASH
3. `_debug_tp4_nccl_only.py`: Full model TP=4 with NCCL → prefill ✅, decode ✅, 73 all_reduce ✅
4. `_debug_tp4_fix_verify.py`: Full LLMEngine TP=4 with NCCL fallback → output matches baseline exactly ✅

**Fix**: `_USE_CUSTOMAR` safety gate disabled by default. NCCL fallback used for all all_reduce operations. Verified output is bit-identical to baseline.

## Spec-Reviewer Issues (all resolved)
| ISSUE | Severity | Resolution |
|-------|----------|-----------|
| ISSUE-1: Missing SIGTERM/SIGINT handler | CRITICAL | ✅ FIXED |
| ISSUE-2: Missing init_dist_if_needed() | HIGH | ✅ NOT A BUG |
| ISSUE-3: Missing argparse CLI | MEDIUM | ✅ FIXED |
| ISSUE-4: Wrong function name | MEDIUM | ✅ FIXED |
| ISSUE-5: Missing finish_reason before [DONE] | LOW | ✅ FIXED |

## Debug Scripts (to clean up)
11 scripts in `scripts/_debug_tp4_*.py` — diagnostic artifacts from root cause investigation. Can be removed.
