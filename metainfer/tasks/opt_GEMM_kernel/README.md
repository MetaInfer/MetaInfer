# opt_GEMM_kernel

An independent MetaInfer task for arena-style GEMM kernel optimization. It
does not import or modify `opt_kernel`, `gen_cpp_infer_framework`, or
`gen_infer_framework`.

## Runtime inputs

- `initial_submission`: initial HIP challenger and optimization-seed directory.
- `evaluator_bundle`: task-author-provided harness directory containing
  `task.yaml` and its correctness and benchmark runners. In the UI this is
  called **Harness path**. MetaInfer snapshots it as the system-owned frozen
  evaluator before execution.
- `weight_bundle`: task-author-provided `model_weights/` directory containing
  `info.json` and one raw `.bin` per tensor. The UI calls it **Weight
  directory**. MetaInfer freezes it separately under `system_weights/`, outside
  all optimizer-agent workspaces.
- hardware profile selection. The first registered profile is **Hygon K100 / gfx928**.

The evaluator's `task.yaml::public_contract` owns dtype, layout and ABI, while
its benchmark cases own shapes. These values are parsed once, frozen, supplied
to agents, and displayed read-only; they are not duplicated as manual UI
fields.

The initial submission includes a constrained `submission.yaml`; it does not
own CMake, compiler or profiler commands. The task-local
`orchestrator/hardware_profiles.yaml` binds the K100 selection to DTK/HIP,
gfx928, CMake + Ninja, `-O3`, and a preferred `hipprof --pmc` route with
rocprofv3/rocprof fallbacks. MetaInfer
resolves the installed executables,
materializes `system_build/{build_profile.json,CMakeLists.txt,build.sh}`, and
freezes the device compiler, host C++ compiler, CMake, Ninja/Make generator,
GPU architecture, fixed flags, and their fingerprint.

The evaluator bundle is copied into task state before agents run and checked
against a SHA-256 manifest before and after each gate. The optimizer only
receives public notebooks and sanitized feedback.

The six iteration phases match the C++/Python framework loop exactly:
`A_plan -> B_implement -> C_test -> D_review -> E_perf_test -> F_perf_plan`.
`S_baseline` is a one-time preflight and is not a seventh loop phase. It first
certifies the frozen Triton implementation (correctness, event benchmark, and
PMC) as the iteration-0 Champion, then independently compiles and certifies the
Initial HIP submission with its own correctness, benchmark, PMC, and artifact
directories. Initial HIP replaces Triton only when the existing evaluator and
noise/critical-regression gates accept it. Inside
`C_test`, MetaInfer runs its fixed SystemBuilder and then the harness
correctness command. `E_perf_test` first runs the full frozen event-timed
benchmark and then profiles only three representative public shapes with the
fixed K100 counter groups. The Harness `profile CASE_ID` entrypoint performs
activation generation/quantization, weight loading and copies before its one
candidate GEMM launch, so those preparation costs are not attributed to GEMM.
`D_review` reviews C evidence; `F_perf_plan` analyzes E evidence and prepares
the next optimization.
See `harness/README.md` for the authoring workspace and runtime protocol.

## Loop

```text
Certified Triton Champion -> Certified Initial HIP challenger
-> A plan -> B implement -> C test -> D review -> E perf test -> F perf plan
```

Each iteration starts from the persisted HIP Champion source. While Triton is
still Champion, it starts from the independently certified Initial HIP source
because Triton has no editable HIP submission tree. A candidate must pass every
declared correctness and performance case, satisfy the weighted and critical
shape gates, and beat the champion by more than the noise threshold before it
is promoted.

The task registers its own New Task card and creation form. Its detail page is
kernel-specific: certified hardware/build identity, weighted latency, speedup,
TFLOPS, modelled memory bandwidth, measured memory bandwidth, L2 hit rate,
compute busy, VGPR/LDS pressure, critical-shape regression, per-case profile,
and champion history. Modelled TFLOPS/bandwidth come from frozen evaluator
metadata; hardware counters come only from the frozen system profiler.

The detail page also provides a live optimization-guidance queue. A task owner can
submit an optimization hypothesis at any time; it is durably delivered to the
next planner or implementer launch and shown as pending/applied in the UI.
Guidance can affect generated candidates but never changes evaluator or
champion gates.

See `notebooks/02_evaluation_protocol.md` for the evaluator bundle schema and
structured report examples.
