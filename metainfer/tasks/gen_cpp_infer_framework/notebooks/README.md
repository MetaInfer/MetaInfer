# C++ Inference Framework Knowledge Base

This task-owned knowledge base covers the Qwen3-8B GGUF C++ inference path
from hardware constraints through OpenAI-compatible HTTP serving. The
single-Z200 baseline supports F16 or Q8_0 weights; the Tensor Parallel baseline
currently requires non-quantized F16 GGUF matrices and reserves quantized TP for
a later large-model extension. `routing.yaml` is the machine-readable source of truth
for which documents each agent role must read.

Do not infer precedence from directory or alphabetical order. The orchestrator
injects a deterministic required set and a bounded optional set into every
agent prompt. Paged KV, Continuous Batching, and Tensor Parallelism are
independent capabilities: selecting one never silently enables another.
Combination contracts activate only when all capabilities in that combination
are selected. `distributed/tensor_parallel.md` overrides only the selected TP
task's single-card weight-storage/dequantization clauses; model math and
validation rules remain in force.

## Contents

| Area | Document | Purpose |
| --- | --- | --- |
| Blueprint | `reference/implementation_blueprint.md` | Integrated module ownership, interface boundaries, data flow, capability branches, and completion definition. |
| Implementation | `reference/implementation_sequence.md` | Layer-by-layer B-phase implementation order, shape/state ledger, Tick transaction, cleanup, and bounded checks derived from verified candidate 008. |
| Platform | `platform/z200_hardware.md` | Z200 hardware, HIP environment, memory, Wave64, profiling, and launch constraints. |
| Model | `model/qwen3/forward_graph.md` | Qwen3 forward graph and model-specific computation order. |
| Model | `model/qwen3/qwen3_8b_contract.md` | Fixed config, independent weights, KV cache, prefill/decode, and validation. |
| Backend | `backend/z200/qwen3_operator_contract.md` | HIP calls, Q8_0 dequantization, hipBLAS GEMM, layouts, and workspaces. |
| Format | `formats/gguf/qwen3_loader.md` | GGUF parsing, fixed-config validation, device ownership, and tokenizer metadata. |
| Runtime | `runtime/single_sequence_runtime.md` | Buffers, RoPE/KV state, prefill/decode, sampler, and generation state machine. |
| Runtime | `runtime/paged_kv_cache.md` | Physical KV blocks, sequence block tables, transactional capacity, paged attention, and lifecycle tests. |
| Runtime | `runtime/continuous_batching.md` | Dynamic scheduling, packed decode, ragged/chunked prefill, sequence isolation, and concurrency tests; Paged-KV integration applies only when both are selected. |
| Runtime | `runtime/paged_continuous_state_machine.md` | Transaction ordering for Paged KV plus Continuous Batching. |
| Distributed | `distributed/tensor_parallel.md` | Non-quantized F16 TP2 weight sharding, collectives, coordination, and multi-GPU tests; Paged KV applies only when independently selected. |
| Distributed | `distributed/tp_paged_kv_contract.md` | Logical shared state, rank-local block tables, group prepare/commit, and TP/Paged-KV failure boundaries. |
| Distributed | `distributed/tp_continuous_batching_contract.md` | Shared packed steps, rank-local dense slots, group failure, and optional Paged override for TP plus Continuous Batching. |
| Serving | `serving/openai_http_server.md` | C++ HTTP server, OpenAI schema, `serve.sh`, process lifecycle, and pipeline-C contract. |
| Validation | `validation/z200_numeric_tests.md` | C0.1 operator tests, CPU references, reports, and pipeline integration. |
| Validation | `validation/capability_checklists.md` | Exact minimum implementation, Numeric case IDs, metadata, behavior probes, and forbidden substitutes. |
| Performance | `performance/optimization_playbook.md` | Measurement decomposition, format/capability-specific optimization routes, experiments, and promotion gates. |
| Case Study | `case_studies/008_tp2_paged_continuous.md` | Verified TP2 + Paged KV + Continuous Batching integration evidence, reusable patterns, and explicit non-generalizable choices. |

## Reference Assets

| File | Purpose |
| --- | --- |
| `reference/qwen3_z200_kernels.hip.cpp` | Correctness-first HIP kernels plus F16 and Q8_0 hipBLAS wrappers. |
| `reference/tokenizer.hpp` / `reference/tokenizer.cpp` | Minimal Qwen3 byte-level BPE tokenizer and single-turn chat formatting. |
| `reference/gguf_loader_template.hpp` | Compiled checked alignment and tensor file-range template. |
| `reference/scheduler_block_manager_template.hpp` | Compiled transactional block manager and scheduler admission template. |
| `reference/tp_sharding_template.hpp` | Compiled TP head and column/row shard range template. |
| `reference/framework_wiring_template.hpp` | Compiled frozen config, logical StepPlan, rank-local batch snapshot, initialization journal, and transactional Tick template. |
| `reference/numeric_harness_template.hpp` | Compiled capability-aware Numeric case registry and JSON report template; missing required cases fail. |

## Routing Policy

`routing.yaml` combines four inputs:

1. The agent role, such as planner, implementer, debugger, reviewer, or perf planner.
2. Frozen task requirements, including hardware and optional runtime features.
3. Failure and review context, such as GGUF, HTTP, numeric, memory, or concurrency errors.
4. A per-role cap on optional reading. Required documents never count against this cap.

For a classified C failure, task-root `failure_playbook.yaml` adds the exact
must-read document IDs, compiled reference templates, root-cause checks, and
required repair evidence. `failure_routes.yaml` remains responsible only for
symptom matching, source scope, timeout, and targeted commands.

When adding a document, register it in `routing.yaml`, add it to at least one
role or conditional route, update cross-references, and extend routing tests.
