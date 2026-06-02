# Phase 10 Implementer Report

- **PID**: 2641677
- **Role**: implementer
- **Timestamp**: 2026-05-30T09:30:00
- **Phase**: 10
- **Status**: SUBMITTED

## Implemented

Created `openai_tp_server.py` — OpenAI-compatible HTTP API for TP inference.
Single file, 490 lines.

### Files Created
- `./openai_tp_server.py` (new)

### Key Components

1. **Engine Singleton** (`_get_or_create_engine`):
   - Thread-safe lazy init with `threading.Lock`
   - Calls `init_tp_distributed()` if not already initialized
   - Creates `LLMEngine` with `inference_backend='qwen_tp'`

2. **TP Worker Loop** (`_tp_worker_loop`):
   - Non-rank0 processes: loads model, waits for `broadcast_object_list` from rank 0
   - Receives `{'action': 'generate', 'prompt': ..., 'max_tokens': ..., ...}` payload
   - Executes identical `engine.generate()` (non-streaming) or `begin_generation()` + `step()` loop (streaming)
   - Exits on `{'action': 'shutdown'}`
   - Blueprint: `OpenAITPServer.architecture.tp_sync_mechanism.non_rank0_flow`

3. **HTTP Server** (`ThreadingHTTPServer` + `CompletionHandler`):
   - `GET /health`: returns `{"status": "ok", "model": "qwen-tp", ...}`
   - `POST /v1/completions`: parses JSON body (`prompt`, `max_tokens`, `temperature`, `stream`, `top_p`)
   - Non-streaming (`_handle_sync`): broadcast -> `engine.generate()` -> JSON response with `choices/text/usage`
   - Streaming (`_handle_stream`): SSE with `data: {json}\n\n` per-token chunks, final chunk with `finish_reason='stop'`, `data: [DONE]\n\n`
   - All requests serialized via `engine_lock` (per blueprint: parallel requests cause NCCL order mismatch -> deadlock)

4. **TP Broadcast** (`_tp_broadcast`):
   - `dist.broadcast_object_list([payload], src=0)` — rank 0 sends, all ranks receive
   - Only called when `dist.world_size > 1`
   - Blueprint: `OpenAITPServer.architecture.tp_sync_mechanism.broadcast_obj`

5. **CLI Entry Point** (`main`):
   - Args: `--model-dir`, `--backend`, `--host`, `--port`, `--max-num-seqs`, `--max-num-batched-tokens`
   - Environment variable fallbacks: `MODEL_DIR`, `PORT`, `INFERENCE_BACKEND`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`
   - Rank routing: `world_size > 1 and rank > 0` -> `_tp_worker_loop`, else -> `run_server`
   - `torch.cuda.set_device(local_rank)` before anything else
   - Handles hidden `--local-rank` arg injected by torchrun

6. **Graceful Shutdown**:
   - SIGINT/SIGTERM -> `server.shutdown()` (triggers `serve_forever()` return)
   - After `serve_forever()` returns: broadcasts `{'action': 'shutdown'}` to non-rank0 workers
   - NCCL ops kept outside signal handler context (signal-safety)

## Blueprint Nodes Read

- `framework_layer.components[13] OpenAITPServer` — full component spec (architecture, tp_sync_mechanism, streaming, non_streaming, startup_sequence, benchmark_usage)
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime` — init_sequence, collectives
- `AGENT_SKILL.md §2.4 OpenAI TP Server Architecture` — TP sync mechanism, endpoints, startup
- `AGENT_SKILL.md §1 执行铁律` — execution directives
- `llm_engine.py` — LLMEngine API surface (generate, begin_generation, step, has_unfinished_requests, get_generation_outputs, _enqueue)
- `engine/models/qwen.py` — QwenForCausalLMTP.forward_decode() signature
- `engine/sampler.py` — tp_sample() for TP-safe sampling
- `engine/structs.py` — Sequence class with dual-track block_table

## Self-Diff Review

- **Modified files**: Only `./openai_tp_server.py` (new file)
- **Untouched**: `scripts/` (all 26 files), `engine/` (14 files), `llm_engine.py`, all phase reports
- **No YAGNI**: No features beyond the spec — only /health and /v1/completions, no /v1/chat/completions, no /v1/models, no metrics endpoint
- **No hardcoded fake inference**: All generation goes through `LLMEngine.generate()` or `LLMEngine.step()`
- **No scripts/ modification**: Verified, all 26 scripts unchanged

## Known Issues

- **None** — The implementation follows the blueprint contracts precisely. The server depends on Phase 1-9 code being correct (imports from `llm_engine`, `engine/`). If prior phases have bugs, they will surface during Phase 10 verification.

## Design Notes

1. **Streaming uses step-based API**: `begin_generation()` + `step()` loop instead of a single `generate()` call. This gives per-token granularity for SSE chunks. Non-rank0 workers run the identical step sequence driven by the initial broadcast.

2. **Shutdown broadcast is post-serve_forever**: NCCL `broadcast_object_list` is not signal-safe. The signal handler only calls `server.shutdown()` (thread-safe), and the shutdown broadcast to TP workers happens in the normal execution flow after `serve_forever()` returns.

3. **Engine is eagerly initialized**: `run_server()` creates the engine before starting the HTTP server, so startup errors (missing model, OOM) surface immediately rather than on the first request.
