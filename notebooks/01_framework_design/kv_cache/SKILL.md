---
name: kv-cache
description: >-
  Designs prefix-aware KV cache and radix-tree caching for LLM inference engines.
  Use when implementing or refactoring block tables, paged KV, radix prefix cache,
  cache hit/miss, eviction, or when the user mentions nano-vllm BlockManager,
  nano-sglang/mini-sglang RadixCache, or Agent-generated inference frameworks.
---

# KV Cache (Logical Layer)

## Division of Responsibilities with Memory Pool

- **KV Cache component**: Manages **which tokens' KV already exist, whether they can be reused, and how to index them** — block tables, prefix matching, radix trees, reference counting, eviction policies, and interaction with the scheduler.
- **Memory Pool component**: Manages **where GPU KV tensor buffers come from and how slots/pages are allocated** — see the sibling skill `memory-pool`.

The two are connected via **physical slot indices** (block id, token slot, `indices` tensor): KV Cache hit returns existing indices; miss requests new slots from Memory Pool and writes back to the tree/block table.

## Source Code Comparison (ref_projects)

| Project | Main File | Responsibility Summary |
|---------|-----------|----------------------|
| nano-vllm | `nanovllm/engine/block_manager.py` | Block-level allocation + **xxhash block content hashing** prefix reuse; `hash_to_block_id`; `ref_count` |
| nano-sglang | `python/sglang/srt/managers/router/radix_cache.py` | Radix tree: `match_prefix` / `insert` / `evict`; `inc_ref_counter` / `dec_ref_counter`; `evictable_size_` |
| mini-sglang | `python/minisgl/kvcache/radix_cache.py` | `RadixPrefixCache`: `match_prefix`, `insert_prefix` (aligned by `page_size`), `lock_handle`, LRU leaf node eviction |
| mini-sglang (no prefix) | `python/minisgl/kvcache/naive_cache.py` | `NaivePrefixCache`: never hits, no eviction; placeholder implementation |

## nano-vllm: BlockManager Key Points

- **`Block`**: `block_id`, `ref_count`, `hash`, `token_ids`; `reset()` sets `ref_count` to 1.
- **`allocate(seq)`**: Iterates by sequence blocks; full blocks use `compute_hash(token_ids, prefix_hash)` to look up `hash_to_block_id`; on hit, `ref_count += 1` and can update `num_cached_tokens`; on miss, takes a block from `free_block_ids` and `update`s the hash table.
- **`deallocate(seq)`**: Reverse-iterates `block_table`, `ref_count--`, returns free blocks when reaching 0.
- **`may_append(seq)`**: When sequence length crosses block boundary, attaches a new block; when block fills, computes current block hash using prefix block hash and registers it.

Coordination with **ModelRunner**: `prepare_prefill` / `prepare_decode` use `block_table` to generate `slot_mapping` and `block_tables` (see `model_runner.py`), for attention to write/read paged KV.

## nano-sglang: RadixCache Key Points

- **`TreeNode`**: `children` (edge key is token sequence), `value` (corresponding KV slot index list for that edge, often a tensor fragment), `ref_counter`, `last_access_time`.
- **`match_prefix(key)`**: Matches along tree; on partial match, `_split_node` splits the edge.
- **`insert(key, value)`**: Inserts remaining suffix as new leaf or extends path.
- **`evict(num_tokens, evict_callback)`**: Leaf node min-heap (by `last_access_time`); skips if `ref_counter > 0`; `evict_callback` is responsible for returning `node.value` corresponding slots to **TokenToKVPool**.
- **`inc_ref_counter` / `dec_ref_counter`**: Updates along parent chain; maintains `evictable_size_` when `ref_counter` transitions 0↔1.

## mini-sglang: RadixPrefixCache Key Points

- Stores **`torch.Tensor` key/value** on edges (value is physical KV pool indices); `get_match_len` can use `fast_compare_key`.
- **`insert_prefix`**: `insert_len = align_down(len, page_size)`, aligned with **MHAKVCache pages**.
- **`lock_handle`**: When request holds prefix, does `ref_count++/--` along path, maintaining `evictable_size` / `protected_size`.
- **`evict`**: LRU leaf nodes (`timestamp`), only evictable when `ref_count==0`; after deleting leaf, if parent becomes leaf and is evictable, re-inserts into heap.

## Checklist for Agent-Generated Inference Frameworks

1. **Is it paged**: Block table (nano-vllm) or token slots + radix (sglang family)?
2. **Prefix semantics**: Only full block hash (nano-vllm) or arbitrary prefix (Radix)?
3. **Reference lifecycle**: When to `inc`/`dec` at request start/end/preemption? Is `ref_count==0` guaranteed before eviction?
4. **Interface with runner**: Output is `block_tables` + `slot_mapping` or `req_to_token` rows?
5. **MLA / GQA**: Logical layer unchanged, but **per-token KV byte count** affects pool capacity (estimated by memory-pool skill).

## Further Reading

- [03_kv_cache.md](../03_kv_cache.md)
- [01_architecture.md](../01_architecture.md) (source code comparison table)

**Cursor skill copy**: `meta-infer/.cursor/skills/kv-cache/SKILL.md` (for Agent discovery)
