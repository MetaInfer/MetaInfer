# Scheduler - Continuous Batching and Request Scheduling

## Core Concept

The scheduler is the "brain" of an LLM inference framework. It decides **which requests** to process in each step and **in which phase** (prefill or decode). The key innovation over naive batching is **continuous batching**: requests can enter and exit the batch at any time, rather than waiting for an entire batch to finish.

## Two-Phase Execution Model

LLM inference has two fundamentally different phases:

### Prefill (Extend) Phase
- **Input**: Multiple prompt tokens for a new request
- **Compute**: All prompt tokens processed in parallel (like training)
- **Output**: KV cache populated for all prompt positions, plus the first generated token
- **Characteristic**: Compute-bound, high arithmetic intensity

### Decode Phase
- **Input**: A single new token per request
- **Compute**: Attention over all previous KV cache entries + MLP for one position
- **Output**: One new token per request
- **Characteristic**: Memory-bandwidth-bound, low arithmetic intensity

### Why This Matters for Scheduling
Prefill and decode have different resource profiles. A good scheduler must balance:
- **Prefill latency**: How quickly a new request gets its first token (TTFT)
- **Decode throughput**: How many tokens/second are generated across all running requests
- **Memory budget**: KV cache is finite; each running request consumes memory

## Scheduling Algorithms

### 1. Prefill-Priority (nano-vllm style)
```
def schedule():
    if waiting_queue is not empty:
        # Try to schedule as many prefills as possible
        batch = select_from_waiting(budget=max_num_batched_tokens)
        return batch, is_prefill=True
    else:
        # All waiting requests handled, do decode
        batch = running_queue
        return batch, is_prefill=False
```

**Logic**: Always try to start new requests first. Only decode when no new requests are waiting.

**Trade-off**: Good TTFT, but can starve decode requests if new requests keep arriving.

### 2. Interleaved Prefill-Decode (nano-sglang style)
```
def forward_step():
    if can_schedule_new_requests() and num_decode_steps >= threshold:
        batch = get_new_fill_batch()  # prefill
        num_decode_steps = 0
    else:
        batch = running_batch  # decode
        num_decode_steps += 1
```

**Logic**: Run a fixed number of decode steps (e.g., 10) between prefill batches. This reduces scheduling overhead and amortizes the cost of switching contexts.

### 3. Overlap Scheduling (mini-sglang style)
```
def overlap_loop():
    next_batch = schedule_next_batch()
    while next_batch:
        current_batch = next_batch
        # GPU executes current_batch
        future = engine.forward_batch(current_batch)
        # CPU prepares next batch while GPU is busy
        next_batch = schedule_next_batch()
        # Wait for GPU
        future.wait()
        postprocess(current_batch)
```

**Logic**: Overlap CPU scheduling with GPU computation to hide scheduling latency. Uses separate CUDA streams for metadata preparation.

## Key Scheduling Decisions

### Batch Size Budget
Every scheduler enforces a token budget:
- `max_num_batched_tokens`: Maximum tokens in a single prefill batch (e.g., 16384)
- `max_num_seqs`: Maximum concurrent sequences (e.g., 512)

### Chunked Prefill
When a prompt is too long to fit in one batch:
```
if prompt_len > remaining_budget:
    chunk_size = remaining_budget
    schedule only chunk_size tokens
    mark request as partially prefilled
    # Will continue in next prefill step
```

**Key rule in nano-vllm**: Only one sequence per batch can be chunked (the last one added).

**Key rule in mini-sglang**: A `PrefillAdder` manages the chunking logic, creating `ChunkedReq` objects that track how much of the prompt has been processed.

### Preemption (nano-vllm)
When KV cache is full during decode:
```
def handle_oom_during_decode():
    # Move lowest-priority sequences back to waiting
    victim = running_queue.pop()
    deallocate_blocks(victim)
    victim.status = WAITING
    waiting_queue.appendleft(victim)
```

### Prefix-Aware Scheduling (nano-sglang)
Three heuristics for reordering the queue to maximize cache hits:
1. **FCFS**: Simple first-come-first-served (default)
2. **LPM (Longest Prefix Match)**: Prioritize requests whose prompts share the most tokens with the radix cache
3. **Weight**: Tree-based scoring that considers how many pending requests share each cache node — favors branches that benefit the most requests

## Memory Budget Calculation

Before scheduling a request, the scheduler must check if there's enough KV cache:

```python
# nano-vllm style (block-based)
needed_blocks = ceil(prompt_len / block_size) - cached_blocks
can_schedule = block_manager.num_free_blocks >= needed_blocks

# nano-sglang style (token-based)
needed_tokens = prompt_len - prefix_cached_len
can_schedule = token_pool.available() >= needed_tokens
```

### Decode Memory Reservation (mini-sglang)
During decode, each request needs space for one more token per step. The scheduler must **reserve** space:
```python
# Reserve one full page per running request to prevent mid-decode OOM
inflight_tokens = num_running_reqs * page_size
available = total_pages - used_pages - inflight_reserved
```

## Postprocessing

After a forward step, the scheduler updates state:
```python
def postprocess(batch, sampled_tokens):
    for req, token in zip(batch.reqs, sampled_tokens):
        req.append_token(token)
        if token == eos_token or req.output_len >= max_tokens:
            req.status = FINISHED
            free_kv_cache(req)
            # In radix cache systems: insert tokens into cache for future reuse
            cache.insert(req.token_ids, req.cache_indices)
```

## Design Template

A minimal scheduler needs:
1. **Two queues**: `waiting` (new requests) and `running` (generating)
2. **schedule()**: Select the next batch respecting memory and token budgets
3. **postprocess()**: Update state, detect finished requests, free resources
4. **Memory check**: Query the KV cache manager for available capacity

Optional enhancements (add based on requirements):
- Chunked prefill (for very long prompts)
- Preemption (for oversubscribed systems)
- Prefix-aware scheduling (when radix cache is used)
- Overlap scheduling (for maximum throughput)
