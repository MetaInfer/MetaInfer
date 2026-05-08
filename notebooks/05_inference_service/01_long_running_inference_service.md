# Inference Framework Servitization: Long-Running Processes and Multiple Prompts (Reference ref_projects)

This document explains how to change the usage of "loading the model every time `python` cold starts" to a **long-running service**: **the framework (including model weights, KV pool, scheduler) is initialized only once**, and new prompts **only enter as new requests into the scheduling**, without needing to re-`from_pretrained`.

All paths are relative to **`meta-infer/ref_projects/`** under the repository root.

---

## 1. Core Differences

| Mode | Behavior | Typical Cost |
|------|----------|-------------|
| **Script/Test Single Process** | In-process `LLMEngine()` → `generate()` → exit | Each process startup loads weights, builds pool |
| **Servitization** | Long-running process; HTTP/ZMQ/IPC receives requests; internal loop `step()` or `run_forever()` | **Only first** load; subsequent requests only add `Sequence`/messages |

Implementation key points:

1. **Single-process minimal service**: `engine = LLMEngine(...)` placed at module level or in `main`, executed only once; use **FastAPI/`while True`** to call `generate(prompt)` multiple times (need to consider **concurrency and thread safety**; simple version can be serial).
2. **Multi-process service**: **API process** only handles send/receive; **Worker process** holds `ModelRunner`/`Engine`, delivers tokenized requests or `GenerateReqInput` via **queue / ZMQ / Pipe**.
3. **Communication primitives**: Synchronous RPC, ZeroMQ (nano-sglang / mini-sglang), SharedMemory + Event (nano-vllm multi-card `ModelRunner` worker).

---

## 2. nano-vllm: Engine Long-Running, `generate` Can Be Called Multiple Times

### 2.1 Design Key Points

- **`LLMEngine`** creates **`ModelRunner`** (and multiple sub-processes for TP), **`Scheduler`**, **`tokenizer`** in `__init__` (see `nanovllm/engine/llm_engine.py`).
- **`generate()`** (approx. 60-90 lines) calls **`add_request`** on the scheduler, then **`while not self.is_finished(): self.step()`** until all prompts in this batch are complete; does **not** reconstruct `ModelRunner` in each `generate`.
- Therefore: if in the **same process**, calling **`llm.generate(batch_a)`** then **`llm.generate(batch_b)`** sequentially, as long as the scheduler semantics support "multi-round batches" or clearing state (nano-vllm's `generate` runs all `add_request`ed seqs in one `is_finished` loop, **need to confirm whether scheduler clears after batch** — from code, it runs all added seqs within one `generate`). **Servitization** common practice: **persistent `LLM` instance**, call **`generate`** for each batch of user requests, or expose **`add_request` + outer `while` `step()`**.

### 2.2 Reference Source Code Paths and Line Numbers

| Content | File | Line (approx.) |
|---------|------|----------------|
| `LLMEngine.__init__`: creates `ModelRunner`, `Scheduler`, tokenizer | `nano-vllm/nanovllm/engine/llm_engine.py` | 17–35 |
| `add_request`, `step`, `is_finished`, `generate` main loop | Same file | 43–90 |
| External alias `LLM` | `nano-vllm/nanovllm/llm.py` | 1–5 |
| Single script: one `LLM(...)` one `generate` | `nano-vllm/example.py` | 6–24 |
| Multi-card: rank0 and worker via `SharedMemory` + `Event` calling `ModelRunner.call` | `nano-vllm/nanovllm/engine/model_runner.py` | 61–89 (`loop`/`read_shm`/`write_shm`/`call`) |

**Servitization idea (minimal change)**: Global **`llm = LLM(model_path, ...)`** initialized once in process; each HTTP request parses body to get `prompt`, calls **`llm.generate([prompt], sampling_params)`** (if scheduler supports continuous multi-round; otherwise need to check `Scheduler.is_finished` and batch boundaries). More fine-grained can follow **`add_request` + `step`** to drive the loop yourself (same file lines 43–55).

---

## 3. nano-sglang: FastAPI + Multi-Process + Router Long-Running Loop

### 3.1 Design Key Points

1. **`launch_server`** (`server.py`): Allocate port → **`TokenizerManager`** → start **`start_router_process`** (Router + Model RPC) and **`start_detokenizer_process`** → wait for pipe to receive **`"init ok"`** → thread **`uvicorn.run(app)`** (approx. 73–137 lines).
2. **HTTP layer**: **`POST /generate`** calls **`tokenizer_manager.generate_request(obj)`** (approx. 39–53 lines); model **already loaded in sub-process**, request only serializes **`GenerateReqInput`**.
3. **Router**: **`RouterManager.loop_for_recv_requests`** continuously **`recv_pyobj`** receives requests; **`loop_for_forward`** calls **`model_client.step`** and sends results to detokenizer (`managers/router/manager.py` approx. 33–78 lines).

### 3.2 Reference Source Code Paths and Line Numbers

| Content | File | Line (approx.) |
|---------|------|----------------|
| FastAPI routes `/generate`, `/v1/completions`, `launch_server` | `nano-sglang/python/sglang/srt/server.py` | 27–71, 73–137 |
| `TokenizerManager`: ZMQ connections to router/detokenizer, `generate_request` | `nano-sglang/python/sglang/srt/managers/tokenizer_manager.py` | 64–119 (`__init__` and `generate_request` start) |
| Router dual loop: receive requests + `model_client.step` | `nano-sglang/python/sglang/srt/managers/router/manager.py` | 14–78, `start_router_process` 56–78 |

---

## 4. mini-sglang: ZMQ Frontend + Sub-Process `Scheduler.run_forever`

### 4.1 Design Key Points

1. **`launch_server`** (`server/launch.py`): **`run_api_server`** first builds **`FrontendManager` (global state)**, then **`start_backend()`** launches multi-process: **one `_run_scheduler` per TP rank**, inline **`Scheduler(...).run_forever()`** (approx. 16–37, 40–113 lines); also starts tokenizer/detokenizer processes.
2. **`Scheduler.run_forever`** (`scheduler/scheduler.py`): **Infinite loop** `overlap_loop` / `normal_loop`, internally **`receive_msg`** gets **`UserMsg`** etc. from ZMQ, then **`_forward`**; **`Engine` created once in `Scheduler.__init__`** (approx. 45–49 lines).
3. **`run_api_server`** (`server/api_server.py`): Sets **`_GLOBAL_STATE`** → **`start_backend()`** → **`uvicorn.run(app)`** (approx. 403–442 lines).
4. **HTTP**: **`POST /generate`** does **`state.new_user()`** + **`send_one(TokenizeMsg(...))`**, does not reload model (approx. 229–247 lines).

### 4.2 Reference Source Code Paths and Line Numbers

| Content | File | Line (approx.) |
|---------|------|----------------|
| Sub-process entry: `_run_scheduler` → `scheduler.run_forever()` | `mini-sglang/python/minisgl/server/launch.py` | 16–37, 59–113 |
| `Scheduler.__init__` creates `Engine`; `run_forever` infinite loop | `mini-sglang/python/minisgl/scheduler/scheduler.py` | 45–76, 120–131 |
| `FrontendManager`, `/generate`, `run_api_server` | `mini-sglang/python/minisgl/server/api_server.py` | 100–159, 229–247, 403–442 |
| **Offline single process**: `LLM` inherits `Scheduler`, `generate` internally calls `run_forever()` until end (not HTTP, but demonstrates "one Engine") | `mini-sglang/python/minisgl/llm/llm.py` | 28–98 |

---

## 5. Mapping to Current `meta-infer/llm_engine.py` Implementation Suggestions

Current **`LLMEngine`** already loads HF model and **`KVMemoryPool`/`Scheduler`** in **`__init__**. To achieve "servitization":

1. **Minimal**: Write a separate **`server_minimal.py`** (or FastAPI), start with **`engine = LLMEngine(...)`** once; each request calls **`engine.generate(prompt, ...)`**. Note: when **multiple requests are concurrent**, whether `Scheduler`/`Sequence` is thread-safe — if not designed with locks, should **single worker serial** or **request queue + single-thread execution**.
2. **Align with ref**: In multi-process, let **only worker** hold GPU-side `model`, frontend only does tokenize/HTTP, passes `prompt` or `input_ids` via **ZMQ/queue** (following nano-sglang / mini-sglang).

---

## 6. Summary Table

| Project | Persistent Core Object | New Prompt Entry | Typical IPC/Protocol |
|---------|----------------------|------------------|---------------------|
| nano-vllm | `LLMEngine` / `ModelRunner` | `add_request` or `generate` | Single machine multi-process: SharedMemory (model_runner) |
| nano-sglang | Router + ModelRpcClient + TokenizerManager | HTTP → `generate_request` | ZMQ, Pipe |
| mini-sglang | `Scheduler` + `Engine` | HTTP → `TokenizeMsg` → ZMQ | ZMQ, multi-process Queue ack |

---

*Document for cross-referencing `ref_projects` implementation of long-running inference services; line numbers may drift with upstream changes, search by class/function name.*
