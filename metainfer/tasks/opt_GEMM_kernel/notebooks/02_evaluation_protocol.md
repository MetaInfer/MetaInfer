# Fixed evaluation protocol

The task snapshots the evaluator and weight bundles before any agent runs. Their
SHA-256 manifests are verified at system gates. MetaInfer separately owns the
BuildProfile, generated CMake, compiler, GPU architecture, hipprof command, and
counter groups.

The evaluator `task.yaml` contains the public contract, correctness command,
profile entry point, correctness cases, exact benchmark shapes, and frozen
hipprof protocol. It does not assign performance weights or criticality:

```yaml
schema_version: 2
name: example-gemm
public_contract:
  operation: "Y = scaled_int8_gemm(A, W, A_scale, W_scale)"
  dtype: {a: int8, b: int8, accumulation: int32, c: bfloat16}
  layout: {a: row_major, b: row_major, c: row_major}
  abi:
    entrypoint: launch_w8a8_gemm
    signature: "launch_w8a8_gemm(A, W, A_scale, W_scale, Y, M, N, K, stream)"
commands:
  correctness:
    argv: [python3, evaluate.py, correctness]
    timeout_s: 7200
  profile:
    argv: [python3, evaluate.py]
    timeout_s: 1800
cases:
  correctness: [public-1, heldout-1]
  private: [heldout-1]
  benchmark:
    - id: decode-gemm
      shape: {m: 1, n: 4096, k: 4096}
      bytes: 16797696
    - id: prefill-gemm
      shape: {m: 4096, n: 4096, k: 4096}
      bytes: 50331648
benchmark_protocol:
  warmup: 10
  samples: 100
  trace_calls: 110
  timer: hipprof_gpu_kernel_duration_ns
  statistic: arithmetic_mean
  operator_aggregation: sum_gpu_kernel_duration_per_call
  synchronization: hipprof_trace
  timed_scope: operator_gpu_dispatches_only
  host_launch_time_included: false
  pmc_timing_used: false
acceptance:
  noise_threshold: 0.01
```

`public_contract` is the only source of truth for dtype, layout, numerics, and
candidate ABI. Benchmark `shape` is mandatory. Frozen optional `flops` and
`bytes` metadata is used only to derive diagnostic TFLOPS or modeled bandwidth;
it does not affect pass/fail.

## Correctness and performance reports

The correctness command writes JSON to `METAINFER_REPORT_PATH`:

```json
{
  "passed": true,
  "cases": [
    {"id": "public-1", "passed": true, "max_abs_error": 0.0}
  ]
}
```

Performance is not supplied by an agent-authored benchmark command. The
system-owned profiler runs the frozen task-local hipprof suite and constructs a
canonical benchmark report from trace operator times:

```json
{
  "schema_version": 2,
  "passed": true,
  "methodology": {
    "warmup": 10,
    "samples": 100,
    "trace_calls": 110,
    "timer": "hipprof_gpu_kernel_duration_ns",
    "statistic": "arithmetic_mean",
    "operator_aggregation": "sum_gpu_kernel_duration_per_call",
    "synchronization": "hipprof_trace",
    "timed_scope": "operator_gpu_dispatches_only",
    "host_launch_time_included": false,
    "pmc_timing_used": false
  },
  "timing_source": "hipprof GPU kernel DurationNs",
  "timed_scope": "operator_gpu_dispatches_only",
  "profile_report": {
    "path": "logs/001/candidate-hardware-profile.json",
    "sha256": "..."
  },
  "cases": [
    {
      "id": "decode-gemm",
      "latency_ms": 0.011,
      "shape": {"m": 1, "n": 4096, "k": 4096},
      "dispatch_count": 2,
      "kernel_breakdown_us": {"split": 8.0, "reduce": 3.0}
    }
  ]
}
```

The methodology must exactly match the frozen protocol. Expected case IDs must
have a one-to-one mapping to finite positive latency values; missing, duplicate,
or unexpected cases fail validation. Each logical call's latency is the sum of
all related GPU dispatch `DurationNs`. PMC replay duration cannot populate
`latency_ms`.

## Every-shape gates

For every frozen benchmark case:

```text
candidate_ms < triton_baseline_ms
candidate_ms < champion_ms * (1 - noise_threshold)
```

Champion evaluation uses a strict boundary where required by promotion so an
equality at the threshold cannot become a hidden improvement. A failure on any
shape rejects the candidate. `worst_case_speedup` and failed IDs are diagnostics,
not aggregate substitutes for the gate.

## Performance report as source of truth

Canonical reports are written atomically and referenced by task-state-relative
path plus SHA-256:

```text
baseline/baseline-benchmark-report.json
certified/initial-hip/candidate-benchmark-report.json
logs/<NNN>/candidate-benchmark-report.json
```

The Champion v2 record stores its submission digest and measurement-report
reference, not copied per-shape latency or an aggregate score. Promotion and
cold restart verify the digest and reload the referenced report. Iteration score,
timeline, and API summaries are derived historical views and cannot drive a
future promotion.

The UI derives optional rates only from frozen work metadata and authoritative
latency:

```text
TFLOPS = flops / latency_ms / 1e9
GB/s   = bytes / latency_ms / 1e6
```

If optional work metadata is absent, latency and the all-shape gate remain valid
while the corresponding rate is unavailable.
