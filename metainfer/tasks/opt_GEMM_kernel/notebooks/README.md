# opt_GEMM_kernel knowledge base

This directory is the public, read-only knowledge base supplied to planning,
implementation and review agents. It is not an evaluator and contains no
held-out shapes.

Read in this order:

1. `00_task_contract.md`
2. `01_submission_contract.md`
3. `02_evaluation_protocol.md`
4. The optimization or profiling note needed by the current iteration

For K100/gfx928 small-M/large-K kernels, read
`09_small_M_splitK_sdot4.md`. It records the 128-bit load + split-K + SDOT4
implementation, TP=1 real-weight measurements, initial dispatch guidance,
correctness requirements and known risks.

For the gfx928 INT8 MMAC/TensorCore general kernel and the measured small-M
choice between MMAC and split-K, read
`10_gfx928_MMAC_tensorcore_general_GEMM.md`. It records the four-Wave tile,
128-bit global loads, proven fragment layout, TP=1 comparisons and dispatch
guidance.

Live task-owner steering semantics are documented in
`06_human_guidance.md`.

For the current `gemm_champon.cpp` engineering record, including the K=256
Wave64 DPP+SDOT4 path, cross-M weight reuse, safe aligned/unaligned global
loads, arbitrary-shape fallbacks, misaligned-pointer tests, measured TOPS and
remaining multi-stream workspace limitations, read
`11_champion_engineering_DPP_alignment_generality.md`.

For the measured follow-up work on large-M MMAC L2-aware CTA swizzling,
last-arriving-CTA fused split-K, the exact `M=1,N=8192,K=1024` BM=1
specialization, physical-versus-logical bandwidth interpretation, and the
shape guards and concurrency cautions required to use those techniques, read
`12_MMAC_CTA_swizzle_fused_splitK_BM1.md`.
