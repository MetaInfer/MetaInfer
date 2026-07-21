# Scaled W8A8 GEMM harness

This directory is a complete, task-author-owned evaluator for the supplied
DeepSeek-style W8A8 weights. Select this directory as **Harness path**, and
select the separate `model_weights/` directory as **Weight directory**.
MetaInfer snapshots both under task state before the baseline runs. Optimization
agents receive the public ABI and shapes, but cannot edit either snapshot.

The matrix-multiply baseline is kept in the separate initial submission at
`../../initial_submissions/myGEMM_kernel/`. Select that directory as **Kernel
path**. Combined demo code containing `main()`, allocation, testing, and timing
does not live in the frozen Harness or candidate submission.

## Required weight directory

`model_weights/` must contain `info.json` plus these separate raw files:

```text
q_proj_a.bin             q_proj_a_scale.bin
q_proj_b.bin             q_proj_b_scale.bin
kv_proj.bin              kv_proj_scale.bin
o_proj.bin               o_proj_scale.bin
moe_w1.bin               moe_w1_scale.bin
moe_w2.bin               moe_w2_scale.bin
moe_w3.bin               moe_w3_scale.bin
```

`evaluate_native.cpp` checks every shape and exact file length against the
metadata supplied for this task. It does not assume a concatenated binary or
byte offsets.

## Weight derivation for TP rank 0

- `wqkv_a`: concatenate `q_proj_a` and `kv_proj` on N; unchanged for TP4/TP8.
- `wq_b`: take the first `32768 / TP` columns of `q_proj_b` and its scale.
- `wo_b`: take the first `8192 / TP` rows of `o_proj`; output scale is unchanged.
- `shared_gate_up_proj`: take the first `2048 / TP` columns from each of
  `moe_w1` and `moe_w3`, then concatenate them and their scales on N.
- `shared_down_proj`: take the first `2048 / TP` rows of `moe_w2`; output scale
  is unchanged.

All loading, slicing, concatenation and host-to-device copies occur outside the
timed interval. `indexer.wq_b` is intentionally excluded until its independent
weight tensor and scale are supplied.

## Activation and timed scope

For each case the harness deterministically generates BF16 `A[M,K]`, then does
per-row symmetric quantization:

```text
A_scale[m] = max(abs(A[m,:])) / 127
A_int8     = clamp(round(A / A_scale), -127, 127)
```

The candidate receives `A_int8`, `W_int8`, `A_scale`, and `W_scale`. GPU events
measure only `launch_w8a8_gemm(...)`; activation quantization, allocation,
weight preprocessing and copies are excluded.

Correctness checks the complete result against a frozen, independent GPU INT32
reference kernel and also recomputes deterministic sentinel points with CPU
INT64 accumulation.
Benchmarking covers TP4/TP8 and `M = 1,2,4,8,16,4096` with 10 warmups and 100
GPU-event samples per case.

## Candidate ABI

```cpp
extern "C" int launch_w8a8_gemm(
    const int8_t* a,
    const int8_t* w,
    const float* a_scale,
    const float* w_scale,
    void* y_bf16,
    int M,
    int N,
    int K,
    void* stream);
```

Return zero after enqueueing work on the supplied stream. The shared library
and frozen native harness executable are built together by MetaInfer's fixed
CMake/hipcc or CMake/nvcc route. The harness then loads the candidate library
from `METAINFER_BUILD_ARTIFACT_DIR`.
