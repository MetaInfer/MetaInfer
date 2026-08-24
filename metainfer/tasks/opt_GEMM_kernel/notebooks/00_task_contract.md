# GEMM optimization contract

The candidate implements the GEMM family defined by the frozen evaluator:

```text
C = epilogue(alpha * op(A) @ op(B) + beta * C_or_bias)
```

Exact dtype, transpose flags, layouts, strides, batching, alignment, numerics,
epilogue, legal approximation, and workspace limits come from
`task.yaml::public_contract`. It is the source of truth supplied to agents and
shown read-only in the UI. Unspecified behavior must not be guessed silently.

Acceptance requires all of the following:

- the system-owned build succeeds;
- every public and held-out correctness case is returned and passes;
- hipprof returns one finite positive operator latency for every performance
  shape under the exact frozen methodology;
- each shape is strictly faster than the frozen Triton baseline;
- each shape crosses the current Champion by the configured noise threshold.

Operator latency is the arithmetic mean of the final trace samples after summing
all GPU dispatch `DurationNs` belonging to each logical GEMM call. Host launch,
JIT, allocation, copies, preprocessing, and synchronization are outside timing.
PMC replay supplies diagnostics only and never supplies latency.

There are no performance weights, critical-shape exceptions, or aggregate score
that can compensate for a losing shape. Before optimization, Triton and Initial
HIP are independently built, correctness-checked, and profiled under the frozen
BuildProfile. Their immutable report references are the only performance facts
used later.

Only files under `submission/` are candidate deliverables. Agent-written tests,
benchmarks, profiler commands, or pass/fail logic never become system gates.
