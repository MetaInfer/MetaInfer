# K100 / gfx928 fixed profiling route

The Web UI's `Hygon K100` selection resolves one system-owned build and profile.
Agents do not select tools, construct commands, or provide timing results.

## Frozen build

```text
cmake -S system_build -B ITER_BUILD -G Ninja \
  -DMETAINFER_SUBMISSION_FILE=ITER_BUILD/submission.cmake
cmake --build ITER_BUILD --target \
  metainfer_gemm_candidate metainfer_gemm_harness
```

The generated build freezes the resolved DTK `hipcc`, Release `-O3`, C++/HIP 17,
and `HIP_ARCHITECTURES=gfx928`. Resolved paths, versions, flags, architecture,
and the BuildProfile fingerprint are written to the compile report.

## Required hipprof suite

K100 performance uses only the frozen task-local hipprof suite:

```text
<resolved-python> <FROZEN_HARNESS>/run_hipprof_suite.py \
  --hipprof /opt/dtk/bin/hipprof --output-dir <SYSTEM_OUTPUT>
<resolved-python> <FROZEN_HARNESS>/analyze_hipprof_suite.py <SYSTEM_OUTPUT>
```

The active K100 profile accepts hipprof only. A missing executable, suite,
analyzer, shape, pass, or matching profiler/protocol fingerprint is an
infrastructure failure. There is no GPU Event, rocprofv3, legacy rocprof, or
PMC-duration latency fallback.

For both Triton and candidate, the suite performs one trace collection and
separate `--pmc`, `--pmc-read`, and `--pmc-write` collections. Tensor generation,
quantization, weight loading/packing, workspace allocation, candidate setup,
Triton JIT, and synchronization complete before each marked host interval. Only
repeated steady-state GEMM calls occur inside the interval.

## Trace operator latency

Every shape has 110 trace calls. The first 10 are warmup and the final 100 are
measured. hipprof trace rows are selected using the manifest's realtime host
boundaries. Legacy manifests without realtime boundaries may translate their
monotonic interval with a boot-stable realtime-minus-monotonic offset; a warmup
kernel timestamp is not used to infer that offset.

The analyzer verifies:

1. trace, PMC, read, and write manifests contain the same case IDs and M/N/K;
2. manifest call counts match the frozen collection protocol;
3. selected dispatch counts divide exactly into logical calls;
4. final measured calls have a stable kernel dispatch pattern.

For one logical call:

```text
operator_us = sum(DurationNs of every related GPU dispatch) / 1000
```

The reported latency is the arithmetic mean of the final 100 `operator_us`
values. Repeated same-name dispatches are first summed within a call, then their
per-call contributions are averaged for the kernel breakdown. The longest
kernel name is only a resource-label hint; it never replaces operator latency.
Host launch API and synchronization time are outside this metric.

## PMC diagnostics

DTK hipprof `--pmc-type 3` internally replays hardware counter groups and emits
merged indexed columns per original dispatch. The analyzer sums indexed values
such as `TCC_HIT[0..N]` and `TCC_MISS[0..N]`, and aggregates every operator
dispatch before normalizing by logical call count.

Separate read/write passes derive physical HBM request bytes and bandwidth.
The compact report retains, when actually reported:

- HBM read/write bytes and read/write/total GB/s;
- L2 hit percentage;
- VGPR, AGPR, SGPR, LDS, and scratch for the selected resource-label kernel;
- grid size, workgroup size, wave size, waves per workgroup, and dispatch count;
- occupancy or wave-residency only when hipprof exposes a reliable field.

Replay `DurationNs` or `DispatchNs` is instrumentation time and is never copied
into benchmark latency. `occupancy_pct` remains unavailable rather than being
estimated from incomplete metadata.

## Interpretation checklist

Start from every shape's summed operator latency and dispatch breakdown. Use PMC
to test a bounded hypothesis, for example excessive physical HBM traffic, weak
L2 reuse, high register/LDS pressure, partial/reduction overhead, or insufficient
parallelism. Do not optimize one counter in isolation: lower occupancy can win
through data reuse or instruction-level parallelism, while higher bandwidth can
still lose if it increases dispatch or reduction work.

A performance improvement is accepted only if every frozen shape beats Triton
and every shape crosses the current Champion noise threshold. Representative
cases may guide diagnosis, but the profiler report and promotion gate retain all
60 shapes. Notebook timings and older profiler records are historical evidence,
not current service-level targets.
