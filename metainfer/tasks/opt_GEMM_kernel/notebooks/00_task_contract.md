# GEMM optimization contract

The candidate implements the GEMM family described in the task requirements:

```text
C = epilogue(alpha * op(A) @ op(B) + beta * C_or_bias)
```

The exact dtype, transpose flags, layouts, strides, batching, alignment,
epilogue, legal approximation and workspace limits come from the user task and
the evaluator bundle. `task.yaml::public_contract` is the frozen source of
truth supplied to agents and shown read-only in the UI. Unspecified behavior
must not be guessed silently.

Acceptance requires all of the following:

- the system compiler command succeeds;
- every declared public and held-out correctness case is returned and passes;
- every performance case is returned under one fixed timing methodology;
- trace-weighted speedup clears the configured minimum;
- no critical shape exceeds its regression limit;
- the candidate beats the current champion by more than the configured noise
  threshold.

Before the optimization loop starts, the original submission must compile,
pass every correctness case, and produce a complete benchmark under the frozen
BuildProfile. This certified measurement is the only baseline used later.

Only files under `submission/` are candidate deliverables. Agent-written test
or benchmark scripts are useful local diagnostics but never become gates.
