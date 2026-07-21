# C++ Inference Framework Knowledge Base

This task-owned knowledge base covers the fixed Qwen3-8B, single-Z200 C++
inference path from hardware constraints through the HTTP serving interface.

Read only the notes needed for the current component, in the order shown
below when implementing the complete framework.

| Order | Note | Purpose |
| ---: | --- | --- |
| 1 | `01_hardware.md` | Z200 hardware, HIP environment, memory, wavefront, and launch constraints. |
| 2 | `02_qwen3_forwrad compute.md` | Qwen3 forward graph and model-specific computation order. |
| 3 | `03_qwen3_8b_contract.md` | Fixed Qwen3-8B config and model contract: independent weights, KV cache, prefill/decode, and validation. |
| 4 | `04_qwen3_z200_operator_contract.md` | Z200 HIP calls, independent Q/K/V and gate/up linears, Q8_0 dequantization, hipBLAS GEMM, and workspaces. |
| 5 | `05_qwen3_gguf_loader_notes.md` | GGUF parsing, fixed-config validation, move-only device-weight ownership, and tokenizer metadata handoff. |
| 6 | `06_qwen3_runtime_notes.md` | Runtime buffers, RoPE/KV state, prefill/decode, logits/sampler boundary, and generation state machine. |
| 7 | `07_qwen3_http_server_contract.md` | C++ HTTP server, OpenAI-compatible schema, `serve.sh`, process lifecycle, and pipeline-C acceptance contract. |
| 8 | `08_qwen3_z200_numeric_test_contract.md` | C0.1 fast Z200 operator numeric tests, CPU references, CMake target sharing, reports, and build-to-HTTP pipeline integration. |
| 9 | `09_continuous_batching_contract.md` | Multi-request continuous batching: scheduler, sequence slots, KV-cache ownership, runtime/kernel interfaces, concurrency safety, and acceptance tests. Read this before changing any B=1 runtime/HTTP concurrency rule. |

Reference implementation assets in this directory:

| File | Purpose |
| --- | --- |
| `qwen3_z200_kernels.hip.cpp` | Current correctness-first HIP kernels and Q8_0 hipBLAS wrapper. |
| `tokenizer.hpp` / `tokenizer.cpp` | Minimal Qwen3 byte-level BPE tokenizer and single-turn chat prompt formatting. |
