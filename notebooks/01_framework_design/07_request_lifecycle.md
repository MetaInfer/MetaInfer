# Request Lifecycle - End-to-End Flow

## Overview

This document traces a single request from HTTP input to generated output, showing how all framework components interact.

## Lifecycle Stages

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. API Receive   │ HTTP POST /generate {prompt, params}             │
├──────────────────┼──────────────────────────────────────────────────┤
│ 2. Tokenize      │ text → token_ids using HF tokenizer              │
├──────────────────┼──────────────────────────────────────────────────┤
│ 3. Queue          │ Create Req/Sequence, add to waiting queue        │
├──────────────────┼──────────────────────────────────────────────────┤
│ 4. Schedule       │ Scheduler picks request, checks memory budget    │
├──────────────────┼──────────────────────────────────────────────────┤
│ 5. Cache Match    │ (Optional) Radix cache finds shared prefix       │
├──────────────────┼──────────────────────────────────────────────────┤
│ 6. Memory Alloc   │ Allocate KV cache blocks/tokens for new content  │
├──────────────────┼──────────────────────────────────────────────────┤
│ 7. Prefill        │ Process all (uncached) prompt tokens at once     │
├──────────────────┼──────────────────────────────────────────────────┤
│ 8. Decode Loop    │ Generate one token per step, repeat until done   │
├──────────────────┼──────────────────────────────────────────────────┤
│ 9. Finish Check   │ EOS token? Max length? Stop string?              │
├──────────────────┼──────────────────────────────────────────────────┤
│ 10. Cache Insert   │ (Optional) Insert token sequence into radix cache │
├──────────────────┼──────────────────────────────────────────────────┤
│ 11. Memory Free    │ Deallocate KV cache (or leave in cache)          │
├──────────────────┼──────────────────────────────────────────────────┤
│ 12. Detokenize     │ token_ids → text                                │
├──────────────────┼──────────────────────────────────────────────────┤
│ 13. API Response   │ Return generated text to client                  │
└──────────────────┴──────────────────────────────────────────────────┘
```

## Detailed Flow: Single-Process (nano-vllm)

```python
# === Stage 1-3: Client side ===
engine = LLMEngine(config)
seq = Sequence(prompt_tokens, sampling_params)
engine.scheduler.add(seq)  # → waiting queue

# === Stage 4-9: Engine loop ===
while not seq.is_finished():
    # Stage 4: Schedule
    batch, is_prefill = engine.scheduler.schedule()
    #   - Checks waiting queue first (prefill priority)
    #   - Verifies block_manager has enough free blocks
    #   - If not: preempts running sequences

    # Stage 5-6: Memory allocation (inside scheduler.schedule)
    #   - block_manager.allocate(seq) assigns physical blocks
    #   - For prefix caching: checks hash_to_block_id for hits

    # Stage 7 or 8: Forward pass
    next_tokens = engine.model_runner.run(batch, is_prefill)
    #   - prepare_prefill/decode: builds input tensors
    #   - model forward: runs all transformer layers
    #   - Attention layers use context.slot_mapping to write KV cache
    #   - sampler: converts logits to tokens

    # Stage 9: Postprocess
    engine.scheduler.postprocess(batch, next_tokens)
    #   - Appends token to sequence
    #   - Checks EOS / max_tokens
    #   - If finished: block_manager.deallocate(seq)

# === Stage 12-13: Client side ===
output_text = tokenizer.decode(seq.output_tokens)
```

## Detailed Flow: Multi-Process (nano-sglang)

```
Process 1: TokenizerManager (main)
├── HTTP POST /generate arrives
├── Tokenize: text → token_ids
├── Create request with unique RID
├── Send TokenizedGenerateReqInput via ZMQ PUSH → Router
├── Wait on asyncio.Event for this RID
│
Process 2: Router (ModelRpcServer)
├── Receive request via ZMQ PULL
├── Create Req object, add to forward_queue
├── match_prefix(input_ids) on RadixCache
│   → Returns (prefix_indices, last_node)
│   → Only uncached tokens need computation
├── Schedule: fit into memory budget
│   → Allocate from ReqToTokenPool + TokenToKVPool
├── Execute forward pass (extend mode)
│   → model_runner.forward(batch, EXTEND)
│   → Attention writes KV to TokenToKVPool
│   → Sample next token
├── Transition to running_batch (decode mode)
├── Decode loop:
│   ├── forward(batch, DECODE) → one token per request
│   ├── Check finished (EOS, max_tokens, stop_string)
│   ├── If finished:
│   │   ├── Insert into RadixCache
│   │   ├── Free ReqToTokenPool slot
│   │   └── Decrease refs in TokenToKVPool
│   └── Send partial/final results via ZMQ PUSH → Detokenizer
│
Process 3: Detokenizer
├── Receive token IDs via ZMQ PULL
├── Batch decode: token_ids → text
├── Trim stop strings if needed
├── Send BatchStrOut via ZMQ PUSH → TokenizerManager
│
Process 1: TokenizerManager (continues)
├── Receive decoded text
├── Set asyncio.Event for matching RID
└── Return HTTP response to client
```

## Detailed Flow: Overlap Scheduling (mini-sglang)

```python
# Scheduler.overlap_loop():
next_batch = schedule_next_batch()

while next_batch is not None:
    current_batch = next_batch

    # === GPU: Execute current batch (async) ===
    with engine.context.forward_batch(current_batch):
        if current_batch.phase == "prefill":
            logits = engine.model.forward(current_batch)
        else:
            logits = engine.graph_runner.replay(current_batch)
        tokens = sampler(logits)

    # === CPU: While GPU runs, prepare next batch ===
    #   This runs on a separate CUDA stream
    next_batch = schedule_next_batch()
    #   1. Check finished requests from previous step
    #   2. Free completed requests' resources
    #   3. Accept new requests from tokenizer
    #   4. Match prefixes in radix cache
    #   5. Allocate memory for new batch
    #   6. Build batch tensors

    # === Synchronize ===
    # GPU finishes, process results
    postprocess(current_batch, tokens)
```

## Key Interaction Points

### Scheduler ↔ KV Cache Manager
- **At schedule time**: "Can I fit this request?" → Check free blocks/tokens
- **At prefill**: "Allocate N blocks for this request" → Reserve physical memory
- **At each decode step**: "Allocate 1 more slot" → Extend allocation
- **At finish**: "Free this request's memory" → Return blocks to pool

### Model Runner ↔ KV Cache
- **At forward time**: Model Runner provides `slot_mapping` to attention layers
- Attention layers use `slot_mapping` to **write** new KV data
- Attention layers use `block_tables` to **read** existing KV data

### Scheduler ↔ Radix Cache
- **At schedule time**: "What prefix does this request share?" → `match_prefix()`
- **At schedule time**: Increment `ref_count` on matched nodes → Prevent eviction
- **At finish**: "Insert this request's tokens into cache" → `insert()`
- **At finish**: Decrement `ref_count` → Allow eviction if no other users
- **At OOM**: "Evict LRU entries" → `evict(num_tokens)` → Free physical memory

## Termination Conditions

A request finishes when any of these are true:
1. **EOS token**: Model generates the end-of-sequence token
2. **Max tokens**: Generated `output_len >= max_tokens` from sampling params
3. **Stop string**: Generated text contains a user-specified stop string
4. **Max context length**: Total sequence length reaches model's context limit

## Streaming Output

For streaming responses, tokens are sent incrementally:
```python
# nano-sglang: After each decode step
if req.stream:
    partial_output = BatchTokenIDOut(req.rid, [new_token_id])
    send_to_detokenizer(partial_output)
    # Detokenizer sends partial text back to TokenizerManager
    # TokenizerManager sends SSE event to client
```

## Design Template

To implement the full lifecycle:
1. **Tokenization**: Use HuggingFace tokenizer (or custom)
2. **Request creation**: Create typed object with state tracking
3. **Queue management**: Waiting + running queues
4. **Memory check**: Query KV cache before scheduling
5. **Forward pass**: Prefill then decode loop
6. **Finish detection**: Check EOS, max_tokens, stop strings
7. **Cleanup**: Free memory, optionally cache for reuse
8. **Detokenization**: Convert tokens back to text
