# Architecture - LLM Inference Framework Overall Design

## Core Insight

All LLM inference frameworks share a common architectural pattern: a **layered pipeline** that transforms HTTP requests into GPU computations and back. The differences between frameworks lie in how they implement each layer, not in the overall structure.

## Common Architecture Pattern

Every inference framework consists of these layers (from outer to inner):

```
[API Layer] → [Request Management] → [Scheduler] → [Model Runner] → [GPU Execution]
     ↑              ↓                     ↓              ↓                ↓
 Response      Tokenization          KV Cache Mgmt   Weight Loading   Attention/MLP
                                     Memory Pool      CUDA Graphs      Sampling
```

## Process Model

### Single-Process (nano-vllm style)

The simplest model: everything runs in one process with one main loop.

```
Main Process:
  while requests_pending:
    batch = scheduler.schedule()
    output = model_runner.run(batch)
    scheduler.postprocess(output)
```

**Pros**: Simple, no IPC overhead.  
**Cons**: Tokenization/detokenization blocks GPU execution.  
**When to use**: Offline batch inference, simple benchmarks.

### Multi-Process (nano-sglang / mini-sglang style)

Separates CPU-bound work from GPU-bound work into different processes.

```
Process 1 (Tokenizer):     HTTP → tokenize → ZMQ → Router
Process 2 (Router/Engine):  ZMQ → schedule → forward → ZMQ
Process 3 (Detokenizer):    ZMQ → detokenize → ZMQ → Tokenizer
```

**IPC Mechanisms observed**:

- **ZeroMQ (ZMQ)**: Used for request/response passing between processes (nano-sglang, mini-sglang)
- **multiprocessing.Pipe**: Used for initial state synchronization
- **SharedMemory + Events**: Used for TP rank coordination (nano-vllm)
- **NCCL**: Used for tensor-level communication between GPU ranks

**Pros**: CPU work doesn't block GPU; enables streaming responses.  
**Cons**: IPC complexity, debugging difficulty.  
**When to use**: Production serving with high throughput requirements.

### Overlap Scheduling (mini-sglang advanced)

The most optimized model: overlaps CPU scheduling with GPU execution using separate CUDA streams.

```
Step N:   [GPU: execute batch N] [CPU: prepare batch N+1]
Step N+1: [GPU: execute batch N+1] [CPU: process results N, prepare N+2]
```

**Key technique**: Use a dedicated CUDA stream for metadata preparation so it runs concurrently with the model forward pass on the main stream.

## Core Components (Every Framework Needs These)

### 1. Request State Machine

Every request goes through a lifecycle:

```
WAITING → RUNNING (prefill) → RUNNING (decode) → FINISHED
                ↑                                    |
                └── PREEMPTED (optional) ←───────────┘
```

Data tracked per request:

- `input_ids` / `token_ids`: The full token sequence
- `status`: Current lifecycle stage
- `block_table` or `table_idx`: Mapping to physical KV cache locations
- `num_cached_tokens` / `cached_len`: How many tokens have cached KV data
- `sampling_params`: Temperature, top-k, top-p, etc.

### 2. Scheduler

Decides what to compute next. Core decisions:

- **Prefill vs Decode**: Which phase gets priority?
- **Batch composition**: Which requests go into the next batch?
- **Memory budget**: Can we fit more requests without OOM?

### 3. KV Cache Manager

Manages GPU memory for key-value tensors:

- **Block/Page allocation**: Mapping virtual token positions to physical memory
- **Prefix caching** (optional): Reusing KV data across requests with shared prefixes
- **Eviction**: Freeing memory when capacity is reached

### 4. Model Runner

Executes the actual neural network forward pass:

- **Input preparation**: Building tensors from the scheduled batch
- **Forward pass**: Running the model
- **CUDA graph replay** (optional): Pre-captured execution for decode batches
- **Sampling**: Converting logits to next tokens

## Design Decision Matrix


| Decision         | Simple Choice          | Performance Choice        |
| ---------------- | ---------------------- | ------------------------- |
| Process model    | Single process         | Multi-process with ZMQ    |
| KV cache         | Contiguous per-request | Paged blocks              |
| Prefix caching   | None                   | Radix tree                |
| Decode execution | Eager PyTorch          | CUDA graph replay         |
| Scheduling       | FCFS                   | Prefix-aware (LPM/Weight) |
| Prefill strategy | Full prompt at once    | Chunked prefill           |
| Preemption       | None (reject on OOM)   | Swap to waiting queue     |


## Source Code Reference


| Component     | nano-vllm                 | nano-sglang                       | mini-sglang              |
| ------------- | ------------------------- | --------------------------------- | ------------------------ |
| Entry point   | `llm.py`                  | `server.py`                       | `core.py`                |
| Engine loop   | `engine/llm_engine.py`    | `managers/router/manager.py`      | `engine/engine.py`       |
| Scheduler     | `engine/scheduler.py`     | `managers/router/scheduler.py`    | `scheduler/scheduler.py` |
| KV cache      | `engine/block_manager.py` | `managers/router/radix_cache.py`  | `kvcache/radix_cache.py` |
| Model runner  | `engine/model_runner.py`  | `managers/router/model_runner.py` | `engine/engine.py`       |
| Memory pool   | (in model_runner)         | `memory_pool.py`                  | `kvcache/mha_pool.py`    |
| Request state | `engine/sequence.py`      | `managers/router/infer_batch.py`  | `core.py`                |


