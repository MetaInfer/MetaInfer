# opt_GEMM_kernel

An independent MetaInfer task for arena-style W8A8 GEMM kernel optimization. It
does not import or modify `opt_kernel`, `gen_cpp_infer_framework`, or
`gen_infer_framework`.

## Runtime inputs

- `initial_submission`: initial HIP challenger and optimization-seed directory.
- `evaluator_bundle`: task-author-provided harness containing `task.yaml`, the
  correctness runner, and the task-local hipprof suite. The UI calls this
  **Harness path**. MetaInfer freezes it before execution.
- `weight_bundle`: task-author-provided `model_weights/` containing `info.json`
  and one raw `.bin` per tensor. MetaInfer freezes it under `system_weights/`,
  outside every optimizer-agent workspace.
- hardware profile selection. The registered production profile is
  **Hygon K100 / gfx928**.

`task.yaml::public_contract` is the source of truth for dtype, layout, numerics,
and ABI. Its benchmark matrix owns the exact shapes. The UI renders these
values read-only rather than asking the task owner to duplicate them.

## System-owned execution

The submission may list source/include paths and allowlisted build options in
`submission.yaml`; it does not own CMake, compiler, architecture, evaluator, or
profiler commands. The K100 hardware profile freezes DTK/HIP, gfx928, CMake +
Ninja, `-O3`, hipprof, all profiler arguments, and a fingerprint of the resolved
tools and protocol.

The evaluator and weight snapshots are SHA-256 verified at every gate. Agents
receive only the public contract, notebooks, current submission, and sanitized
system evidence. They cannot replace correctness, timing, scoring, or promotion
logic.

## Performance protocol

K100 performance latency comes only from the required task-local hipprof trace
suite. For each of the 60 frozen benchmark shapes, candidate and Triton setup,
JIT, allocation, copies, weight preprocessing, packing, workspace initialization,
and synchronization complete before the marked interval. The interval contains
110 steady-state calls: 10 warmup calls followed by 100 measured calls.

For each logical GEMM call, MetaInfer sums `DurationNs` for every related GPU
dispatch. It then takes the arithmetic mean of the final 100 operator sums.
This is GPU operator time only: host launch API time and synchronization overhead
are excluded. A split-K main kernel plus reduction is therefore one operator
sample containing both GPU dispatch durations.

Every iteration remeasures the current Champion and candidate in the same
round. Reports retain all raw operator samples and expose mean, median,
standard deviation, CV, and observed range. Results near the noise boundary
trigger a second equal-size hipprof trace for both sides; the decision uses the
arithmetic mean of all raw `DurationNs` operator samples. No shape weighting or
synthetic aggregate latency is used.

hipprof `--pmc`, `--pmc-read`, and `--pmc-write` run separately. Routine
iterations collect them only for failed diagnostic shapes; a promotable
candidate receives a full-shape PMC archive. They provide
HBM traffic/bandwidth, L2 behavior, VGPR/AGPR/SGPR, LDS, scratch, dispatch and
wave metadata. Occupancy or wave residency is shown only when the profiler
reports a reliable value. PMC replay duration is never latency. Each profiler
pass records its real wall time and has an independent timeout. Missing hipprof,
incomplete cases, unstable dispatch patterns, mismatched protocol fingerprints,
or collection/analyzer failures are infrastructure failures; there is no event
or rocprof timing fallback for K100.

## Loop and promotion

```text
Certified Triton baseline -> Certified Initial HIP challenger
-> A plan -> B implement -> C test -> D review -> E perf test -> F perf plan
```

`S_baseline` is one-time preflight, not a seventh iteration phase. It certifies
Triton correctness/performance, then independently builds and certifies Initial
HIP. `C_test` runs the system build and frozen correctness command. `E_perf_test`
runs the all-shape hipprof suite and the immutable performance-report gate.

A candidate must satisfy all of these conditions:

1. compile and pass every declared correctness case;
2. return one finite positive hipprof operator latency for every benchmark shape;
3. preserve the certified lineage that originally beat Triton on every shape;
4. be below the same-round `champion_ms * (1 - noise_threshold)` on every shape.

There are no shape weights, critical-shape exceptions, or aggregate score that
can compensate for a losing shape. When Triton remains Champion, the next HIP
iteration still starts from certified Initial HIP because Triton has no editable
HIP submission tree.

The authoritative performance data is an immutable JSON report referenced by
relative task-state path plus SHA-256. Triton, Initial HIP, every iteration, and
Champion records point to these reports. Cold restart verifies and reloads the
referenced report; iteration scores, timeline fields, and UI summaries are
historical or derived views and never drive promotion.

The detail page exposes raw per-shape baseline/candidate/Champion latency,
speedup, regression, kernel dispatch breakdown, modeled rates from frozen
metadata, HBM read/write/total bandwidth, L2, registers, LDS/scratch, and
available wave/occupancy evidence. It does not produce a weighted overall score.

Live task-owner guidance is durable input to the next planner or implementer,
but remains a hypothesis. It cannot alter compilation, correctness, profiler,
all-shape, or Champion gates.

See `harness/README.md` for harness ownership,
`notebooks/02_evaluation_protocol.md` for report and gate semantics, and
`notebooks/04_profiling.md` for the K100 hipprof route.
