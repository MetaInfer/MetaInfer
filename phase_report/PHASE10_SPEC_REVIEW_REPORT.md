# Phase 10 Spec Review Report

| Field | Value |
|-------|-------|
| PID | 4088031 |
| Role | spec-reviewer |
| Timestamp | 2026-06-09T06:12:00Z |
| Phase | 10 |
| Verdict | ❌ FAIL |

---

## Evidence Chain (Contracts Verified)

- `components.OpenAITPServer.architecture.server_type`: ✅ @ openai_tp_server.py:19,48,283 -- Uses `ThreadingHTTPServer` + `BaseHTTPRequestHandler` (not Flask/FastAPI)
- `components.OpenAITPServer.architecture.endpoints[0]` (GET /health): ✅ @ openai_tp_server.py:74-82 -- `do_GET` with `/health` path returns `{"status":"ok","rank":<n>}`
- `components.OpenAITPServer.architecture.endpoints[1]` (POST /v1/completions): ✅ @ openai_tp_server.py:93-137 -- `do_POST` with `/v1/completions` path
- `components.OpenAITPServer.architecture.tp_sync_mechanism.broadcast_obj`: ✅ @ openai_tp_server.py:33-41 -- Uses `dist.broadcast_object_list([payload if rank==0 else None], src=0)`
- `components.OpenAITPServer.architecture.tp_sync_mechanism.rank0_flow`: ✅ @ openai_tp_server.py:130 -- Rank0 broadcasts command before entering lock-protected generate block
- `components.OpenAITPServer.architecture.tp_sync_mechanism.non_rank0_flow`: ✅ @ openai_tp_server.py:299-331 -- `while True` loop, broadcast_obj receive, shutdown check, same generate/step under lock
- `components.OpenAITPServer.architecture.tp_sync_mechanism.request_lock`: ✅ @ openai_tp_server.py:133,275,281 -- `threading.Lock()` serializes all requests; shared across rank0 handler and non-rank0 worker
- `components.OpenAITPServer.architecture.streaming.critical_constraints.connection_header`: ✅ @ openai_tp_server.py:198 -- `Connection: close` header on SSE response
- `components.OpenAITPServer.architecture.streaming.critical_constraints.close_connection_flag`: ✅ @ openai_tp_server.py:62-64,195,233,158,176,76,84,95,107 -- `self.close_connection = True` set on ALL code paths (try success, except error, do_GET, do_POST error)
- `components.OpenAITPServer.architecture.streaming.Content-Type`: ✅ @ openai_tp_server.py:197 -- `text/event-stream` for SSE responses
- `components.OpenAITPServer.architecture.streaming.final_chunk`: ✅ @ openai_tp_server.py:228,238 -- `data: [DONE]\n\n` sent in both success and error path
- `components.OpenAITPServer.architecture.non_streaming`: ✅ @ openai_tp_server.py:166-181 -- JSON response with `model`/`object`/`choices[].text`/`usage` format
- `components.OpenAITPServer.architecture.tp_sync_mechanism.shutdown`: ✅ @ openai_tp_server.py:291-293,306-307 -- Rank0 broadcasts `{"action":"shutdown"}` on KeyboardInterrupt/finally; non-rank0 breaks out of loop on receipt
- `AGENT_SKILL.md §2.4 SSE close_connection`: ✅ @ openai_tp_server.py (all paths) -- `self.close_connection = True` set unconditionally before any response

---

## Issues Found

### ISSUE-1 (CRITICAL): Missing Worker Signal Handler -- SIGTERM/SIGINT + os._exit(0)

- **JSON Path**: `components.OpenAITPServer.architecture.worker_signal_handling`
- **Source**: blueprint lines 384-388; AGENT_SKILL.md lines 326-328
- **@ openai_tp_server.py**: nowhere (absent from entire file)
- **Expected**:
  - Register `signal.signal(signal.SIGTERM, handler)` and `signal.signal(signal.SIGINT, handler)` for non-rank0 workers
  - Handler must accept `(signum, frame)` two params and call `os._exit(0)` -- bypasses Python interpreter, kills process directly
  - Blueprint past_bug_20260606: "缺少 signal handler → torchrun kill 后 4 个 python 子进程残留，各占 ~6.5GB GPU 显存"
- **Actual**: No `import signal` anywhere in the file. No signal handler registered. No `os._exit()` call anywhere. Non-rank0 worker thread blocks on `dist.broadcast_object_list` (NCCL C-call), so SIGTERM/SIGINT are queued by Python and delivered only after the C call returns -- which may never happen.
- **Fix**: Add before the `while True` loop in `_tp_worker_loop()` (line 301):
  ```python
  import signal
  import os as _os
  def _shutdown_handler(signum, frame):
      _os._exit(0)
  signal.signal(signal.SIGTERM, _shutdown_handler)
  signal.signal(signal.SIGINT, _shutdown_handler)
  ```

---

### ISSUE-2: Missing `init_dist_if_needed()` -- No Dist Process Group Initialization

- **JSON Path**: `components.OpenAITPServer.startup_sequence[1]`
- **Source**: blueprint line 392
- **@ openai_tp_server.py**: nowhere (missing from entire file)
- **Expected**: `2. init_dist_if_needed(): WORLD_SIZE>1 时 dist.init_process_group('nccl')` -- a safety guard ensuring dist is initialized before any collective operations.
- **Actual**: Code assumes `dist` is already initialized. The `__main__` block (lines 338-347) reads `TP_SIZE` but never calls `dist.init_process_group`. If the server is launched without `torchrun`, dist calls will fail with "Default process group has not been initialized". `_get_rank()` only checks `dist.is_initialized()`, does not initialize.
- **Fix**: Add function and call before `run_tp_server()`:
  ```python
  def init_dist_if_needed():
      if int(os.environ.get("WORLD_SIZE", "1")) > 1:
          if not (dist.is_available() and dist.is_initialized()):
              dist.init_process_group("nccl")
  ```

---

### ISSUE-3: Missing CLI Arguments -- Parsing Swapped for Hardcoded env vars

- **JSON Path**: `components.OpenAITPServer.startup_sequence[0]`
- **Source**: blueprint lines 390-391
- **@ openai_tp_server.py**: lines 338-347
- **Expected**: argparse with `--model-dir`, `--backend` (tp/qwen_tp/deepseek_tp/hf), `--host`, `--port`, `--max-num-seqs`, `--max-num-batched-tokens`, `--max-new-tokens-cap`
- **Actual**: `__main__` only reads `MODEL_DIR`, `PORT`, `TP_SIZE` from `os.environ`. No `argparse`. Hardcoded: `inference_backend="qwen_tp"` (line 272), `host="0.0.0.0"` (line 283), `max_num_seqs=4` (line 273). Missing entirely: `--max-num-batched-tokens`, `--max-new-tokens-cap`.
- **Fix**: Add argparse with all startup_sequence arguments, pass them through to engine constructor and server configuration.

---

### ISSUE-4: Wrong Function Name and Signature -- `run_tp_server` vs `run_tp_generation_loop`

- **JSON Path**: `components.OpenAITPServer.startup_sequence[3]`
- **Source**: blueprint lines 394, 2261
- **@ openai_tp_server.py**: lines 255-296
- **Expected**: `4. run_tp_generation_loop(engine, host, port, max_new_tokens_cap)` -- engine created externally (step 3) and passed in as parameter.
- **Actual**: `run_tp_server(model_dir, port, tp_size)` -- different name, different signature. Creates `LLMEngine` internally at line 269 instead of accepting it as parameter. Does not accept `host` or `max_new_tokens_cap`.
- **Fix**: Rename to `run_tp_generation_loop`, change signature to accept `(engine: LLMEngine, host: str, port: int, max_new_tokens_cap: int)`, move `LLMEngine(...)` construction to caller (step 3).

---

### ISSUE-5: Missing `finish_reason='stop'` Final Chunk Before `[DONE]` in SSE Streaming

- **JSON Path**: `components.OpenAITPServer.architecture.streaming.flow`
- **Source**: blueprint line 373
- **@ openai_tp_server.py**: lines 227-229
- **Expected**: `SSE data chunks → final chunk finish_reason='stop' → data: [DONE]` -- a final SSE data chunk with `finish_reason: "stop"` sent before the `data: [DONE]\n\n` termination marker.
- **Actual**: Only `data: [DONE]\n\n` is sent (line 228). No data chunk with `finish_reason: "stop"` precedes it. Token-by-token chunks use `{"choices":[{"text":...,"index":0}]}` with no finish_reason anywhere.
- **Fix**: Before line 228 (`self.wfile.write(b"data: [DONE]\n\n")`), insert:
  ```python
  stop_chunk = json.dumps({"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]}, ensure_ascii=False)
  self.wfile.write(f"data: {stop_chunk}\n\n".encode())
  self.wfile.flush()
  ```

---

## Blueprint Information Gaps

None identified. The blueprint contracts for `OpenAITPServer` and `phase_10_e2e_acceptance` are clearly specified with concrete expected values, function signatures, and critical constraints.

---

## Summary

| # | Severity | Category | Description |
|---|----------|----------|-------------|
| ISSUE-1 | CRITICAL | Process lifecycle | Missing SIGTERM/SIGINT + os._exit(0) -- causes GPU VRAM leak on shutdown |
| ISSUE-2 | HIGH | Startup sequence | Missing init_dist_if_needed() -- dist uninitialized without torchrun |
| ISSUE-3 | MEDIUM | Startup sequence | Missing CLI args; --backend/--host/--max-num-seqs hardcoded |
| ISSUE-4 | MEDIUM | Contract violation | Wrong function name (run_tp_server vs run_tp_generation_loop) and wrong signature |
| ISSUE-5 | LOW | Streaming compliance | Missing finish_reason='stop' chunk before [DONE] in SSE stream |

**Verdict: ❌ FAIL** -- 5 violations found. Code does not fully match the blueprint contract at `components.OpenAITPServer`. Implementer must fix all 5 issues before re-submission for spec review.

All structural contracts (server type, endpoints, TP sync via broadcast_object_list, streaming SSE headers, close_connection on all paths, response format) are correctly implemented. The violations are in lifecycle handling (ISSUE-1/2), CLI surface (ISSUE-3/4), and streaming protocol detail (ISSUE-5).
