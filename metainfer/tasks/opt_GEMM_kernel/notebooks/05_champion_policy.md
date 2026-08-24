# Champion/challenger policy

Every HIP iteration starts from the persisted HIP Champion. While Triton remains
Champion, iterations start from certified Initial HIP because Triton has no
editable HIP submission tree.

A challenger is eligible only after the system build and every correctness case
pass. Its immutable hipprof performance report must contain exactly one finite
positive operator latency for every frozen benchmark shape. Promotion then
requires both per-shape gates:

```text
candidate_ms < triton_baseline_ms
candidate_ms < champion_ms * (1 - noise_threshold)
```

The strict promotion comparison rejects equality where it would not represent a
real improvement. One failed shape rejects the challenger; no weighted mean,
critical-shape exception, or favorable aggregate can compensate.

`champion.json` v2 stores Champion kind, source iteration, submission SHA-256,
promotion metadata, and a task-state-relative measurement-report path plus
SHA-256. It does not copy per-shape latency or aggregate score. Promotion and
cold restart verify and reload that report. Iteration score and timeline values
are derived historical snapshots only.

Failed and non-promoted candidates remain in iteration history for diagnosis but
never become the next starting implementation. The selected HIP artifact is
stored under `champion/submission/`; a Triton Champion has no copied HIP source.
