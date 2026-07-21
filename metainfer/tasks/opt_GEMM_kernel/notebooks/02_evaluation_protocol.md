# Fixed evaluation protocol

The task snapshots an external evaluator bundle into task state before the
first agent runs. A SHA-256 manifest is checked before and after every system
evaluation command. Compilation is not an evaluator command: MetaInfer owns
the frozen BuildProfile, CMakeLists.txt and build.sh. The evaluator bundle
contains only correctness and benchmark commands:

```yaml
schema_version: 2
name: example-gemm
public_contract:
  operation: "C = alpha * A @ B + beta * C"
  dtype: {a: fp16, b: fp16, accumulation: fp32, c: fp16}
  layout:
    {a: row_major, b: row_major, c: row_major, trans_a: false, trans_b: false}
  abi:
    entrypoint: launch_gemm
    signature: "launch_gemm(A, B, C, M, N, K, stream)"
commands:
  correctness:
    argv: [python3, evaluate.py, correctness]
    timeout_s: 1200
  benchmark:
    argv: [python3, evaluate.py, benchmark]
    timeout_s: 1800
cases:
  correctness: [public-1, public-2, heldout-1]
  private: [heldout-1]
  benchmark:
    - id: decode-gemm
      weight: 2000
      critical: true
      shape: {m: 1, n: 4096, k: 4096, batch: 1}
      bytes: 33570816
    - id: prefill-gemm
      weight: 100
      critical: false
      shape: {m: 2048, n: 4096, k: 4096, batch: 1}
      bytes: 67108864
benchmark_protocol:
  warmup: 10
  samples: 100
  timer: gpu_event
acceptance:
  min_weighted_speedup: 1.01
  noise_threshold: 0.01
  max_critical_regression: 0.03
  require_all_cases: true
```

Each command writes JSON to `METAINFER_REPORT_PATH`.

`public_contract` is mandatory and is the only source of truth for dtype,
layout and candidate ABI. Benchmark case `shape` is mandatory. The creation UI
does not ask the task owner to duplicate these fields: after the evaluator is
frozen, the task detail page renders the extracted contract read-only and the
orchestrator injects exactly the same contract into planner/implementer
prompts.

Correctness report:

```json
{
  "passed": true,
  "cases": [
    {"id": "public-1", "passed": true, "max_abs_error": 0.001}
  ]
}
```

Benchmark report:

```json
{
  "passed": true,
  "methodology": {"warmup": 10, "samples": 100, "timer": "gpu_event"},
  "cases": [
    {
      "id": "decode-gemm",
      "latency_ms": 0.11
    }
  ]
}
```

Before any optimizer runs, the system compiles the original submission and
runs correctness and benchmark with `METAINFER_EVALUATION_ROLE=baseline`.
That report and its BuildProfile fingerprint are frozen. Candidate runs use
`role=candidate`; they only report their own latency. Weight and criticality
come from task.yaml, not from measurement reports.

For GEMM profiler display, each benchmark case may declare `shape` and
`bytes`. When `shape` is present, the frozen spec derives FLOPs as
`2 * M * N * K * batch`; an explicit positive `flops` value overrides that
derivation for fused or non-standard work. `bytes` is the task author's
declared total device-memory traffic for the case and should include every
tensor read/write required by the ABI. Candidate reports never provide these
values.

The methodology object must exactly match `benchmark_protocol` for both
baseline and candidate. The orchestrator computes weighted speedup as:

```text
sum(weight_i * baseline_ms_i) / sum(weight_i * candidate_ms_i)
```

The UI derives profiler rates from the frozen work metadata and measured
latency:

```text
TFLOPS = flops / latency_ms / 1e9
GB/s   = bytes / latency_ms / 1e6
```

If `shape`/`flops` or `bytes` is omitted, latency and speedup remain valid and
the corresponding TFLOPS or bandwidth tile is shown as unavailable.
