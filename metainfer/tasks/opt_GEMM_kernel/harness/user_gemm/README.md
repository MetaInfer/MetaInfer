# Scaled W8A8 GEMM harness

This directory is the task-author-owned evaluator for the supplied
DeepSeek-style W8A8 weights. Select it as **Harness path** and select the
separate `model_weights/` directory as **Weight directory**. MetaInfer freezes
both before baseline certification. Agents receive the public ABI, shapes, and
sanitized evidence but cannot edit either snapshot.

The editable HIP seed is under `../../initial_submissions/myGEMM_kernel/` and is
selected as **Kernel path**. Allocation, references, testing, and profiling
control do not belong in a candidate submission.

## Required weights and TP rank 0 derivation

`model_weights/` contains `info.json` and one raw file for each tensor/scale:

```text
q_proj_a.bin             q_proj_a_scale.bin
q_proj_b.bin             q_proj_b_scale.bin
kv_proj.bin              kv_proj_scale.bin
o_proj.bin               o_proj_scale.bin
moe_w1.bin               moe_w1_scale.bin
moe_w2.bin               moe_w2_scale.bin
moe_w3.bin               moe_w3_scale.bin
```

`evaluate_native.cpp` validates every filename, dtype, shape, and exact byte
length. It does not assume concatenated binaries or hidden offsets.

- `wqkv_a`: concatenate `q_proj_a` and `kv_proj` on N; unchanged for TP4/TP8.
- `wq_b`: first `32768 / TP` columns of `q_proj_b` and its scale.
- `wo_b`: first `8192 / TP` rows of `o_proj`; output scale is unchanged.
- `shared_gate_up_proj`: first `2048 / TP` columns from each of `moe_w1` and
  `moe_w3`, then concatenate on N.
- `shared_down_proj`: first `2048 / TP` rows of `moe_w2`; output scale is
  unchanged.

Loading, slicing, concatenation, packing, and host-to-device copies occur before
the marked interval. `indexer.wq_b` remains excluded until its independent
weight and scale are supplied.

## Activation, correctness, and ABI

The harness deterministically generates BF16 `A[M,K]`, then performs per-row
symmetric quantization:

```text
A_scale[m] = max(abs(A[m,:])) / 127
A_int8     = clamp(round(A / A_scale), -127, 127)
```

The candidate receives prepared `A_int8`, `W_int8`, `A_scale`, and `W_scale` and
produces only the BF16 result. Correctness compares the complete output with a
frozen independent GPU INT32 reference and recomputes deterministic sentinel
points with CPU INT64 accumulation.

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

Return zero after enqueueing work on the supplied stream. MetaInfer owns the
fixed CMake/hipcc build and loads the resulting library from
`METAINFER_BUILD_ARTIFACT_DIR`.

## Task-local hipprof performance protocol

K100 latency is collected only by the frozen scripts in this directory:

```bash
python3 run_hipprof_suite.py --output-dir "$METAINFER_REPORT_DIR/hipprof-suite"
python3 analyze_hipprof_suite.py "$METAINFER_REPORT_DIR/hipprof-suite"
```

The system runner launches the suite with its actual Python interpreter, the
frozen candidate artifact, frozen weights, and frozen benchmark protocol. The
matrix contains TP4/TP8 workloads at `M = 1,2,4,8,16,4096`, for 60 shapes total.
Candidate and Triton setup, JIT, allocation, copies, packing, workspace setup,
and synchronization finish before each marked interval.

Each trace interval contains 110 steady-state logical calls. The first 10 are
warmup and the final 100 are measured. For every call, the analyzer sums
`DurationNs` for all GEMM GPU dispatches, including split-K and reduction, then
takes the arithmetic mean of the final 100 sums. It verifies the exact call
count and stable final dispatch pattern. Host launch API and synchronization
time are outside this GPU operator latency.

Separate `--pmc`, `--pmc-read`, and `--pmc-write` passes provide HBM read/write
bytes and bandwidth, L2 hit behavior, VGPR/AGPR/SGPR, LDS, scratch, dispatch,
workgroup, and wave metadata. Occupancy remains unavailable unless hipprof
reports a reliable value. PMC replay duration is never used as latency.

Every one of the 60 shapes is a hard performance gate. A candidate must be
strictly faster than frozen Triton on each shape and cross the current Champion
noise threshold on each shape; no weight or aggregate average can hide a loss.
