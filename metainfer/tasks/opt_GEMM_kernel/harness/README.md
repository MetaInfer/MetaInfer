# GEMM harness authoring area

This directory is the task-local place for evaluator harnesses. A harness is
provided by the task author; it is not generated or modified by the kernel
optimization agent.

Select `user_gemm/` in the Web UI's **Harness path** field and the separate
`model_weights/` directory in **Weight directory**. At task start MetaInfer
copies the selected directories to:

```text
<task-state>/system_evaluator/
<task-state>/system_weights/
```

Both copies are SHA-256 fingerprinted. The evaluator is checked before and
after every command, and the weight directory is outside every agent iteration
workspace. Optimization agents receive only the public contract and sanitized
results, not either private directory.

## Phase ownership

```text
S_baseline     MetaInfer build -> harness correctness -> harness benchmark
A_plan         agent; no harness execution
B_implement    agent edits submission/ only
C_test         MetaInfer SystemBuilder -> frozen harness correctness command
D_review       agent reviews compile/correctness evidence
E_perf_test    frozen harness benchmark command -> champion decision
F_perf_plan    agent analyzes performance and plans the next iteration
```

`S_baseline` is preflight; the six-phase outer loop is A through F.

Thus `harness` and `evaluator_bundle` refer to the same artifact. The latter is
kept as the requirements/API key for compatibility.

## Required files

Every selectable harness directory must contain `task.yaml`. Its commands must
write a JSON object to `METAINFER_REPORT_PATH` and return zero only when the
phase completed normally and its report is valid.

MetaInfer supplies these environment variables:

- `METAINFER_EVALUATOR_BUNDLE`: frozen harness directory.
- `METAINFER_SUBMISSION_DIR`: source submission being evaluated.
- `METAINFER_BUILD_ARTIFACT_DIR`: system-built candidate artifact directory.
- `METAINFER_REPORT_PATH`: required JSON output path.
- `METAINFER_EVALUATION_PHASE`: `correctness` or `benchmark`.
- `METAINFER_EVALUATION_ROLE`: `baseline` or `candidate`.
- `METAINFER_BUILD_FINGERPRINT`: frozen compiler/build identity.
- `METAINFER_BENCHMARK_PROTOCOL`: frozen JSON timing protocol.
- `METAINFER_WEIGHT_BUNDLE`: frozen directory containing `info.json` and the
  separate tensor `.bin` files.
- `METAINFER_WEIGHT_SHA256`: fingerprint of that frozen weight directory.

The harness should locate and load the candidate shared library from
`METAINFER_BUILD_ARTIFACT_DIR`. Do not compile the candidate itself: CMake,
hipcc/nvcc, target architecture and candidate flags are the first internal
gate of `C_test` and remain owned by MetaInfer.

## Trust rules

- Put CPU/PyTorch references, input generation, tolerances and case definitions
  in the harness.
- Include all correctness cases in the JSON report, including private cases.
  MetaInfer removes private details before feedback reaches an agent.
- Benchmark only the operation covered by the public ABI. Exclude allocation,
  host/device copies and process startup from `latency_ms`.
- Use deterministic inputs, GPU-event timing, warmup and repeated samples.
- Never report success before the reference comparison actually passes.
- Keep harness build products outside this source directory so the frozen
  bundle digest remains stable.

`user_gemm/evaluate_native.cpp` is the concrete W8A8 runner for the supplied tensor
metadata. Its README documents the TP4/TP8 slicing and concatenation rules.
