---
name: memory-pool
description: >-
  Pre-allocates GPU KV buffers and implements slot/page allocation with optional
  reference counting. Use when implementing ReqToTokenPool, TokenToKVPool,
  MHAKVCache, nano-vllm allocate_kv_cache, profiling GPU memory for KV capacity,
  or Agent-generated inference frameworks.
---

# Memory Pool (Physical Layer)

## Division of Responsibilities with KV Cache

- **Memory Pool**: **Pre-allocates** large KV tensors; **O(1) or batch** allocation/reclamation of **slots or pages**; `mem_state` / free stack; optional **reference counting** for prefix sharing; exposes `kv_data` / `get_key_buffer` views to kernels.
- **KV Cache**: Who occupies which slots, Radix/block table logic; see sibling skill `kv-cache`.

## Source Code Comparison (ref_projects)

| Project | Main File | Responsibility Summary |
|---------|-----------|----------------------|
| nano-vllm | `nanovllm/engine/model_runner.py` → `allocate_kv_cache` | Single tensor `[2, layers, num_blocks, block_size, kv_heads, head_dim]`; binds `k_cache`/`v_cache` to modules per layer |
| nano-sglang | `python/sglang/srt/memory_pool.py` | `ReqToTokenPool` + `TokenToKVPool`; request slots and KV slots separated |
| mini-sglang | `python/minisgl/kvcache/mha_pool.py` | `MHAKVCache`: page-level buffer + `store_kv` → `store_cache` kernel |

## nano-vllm: allocate_kv_cache

1. **Memory budget**: `mem_get_info` + `memory_stats` peak/current;
   `block_bytes = 2 * num_layers * block_size * num_kv_heads_tp * head_dim * dtype.itemsize` (under TP: `num_kv_heads // world_size`).
2. **`num_kvcache_blocks`**: `(total * gpu_memory_utilization - used - peak + current) // block_bytes`.
3. **`kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`**.
4. **Layer binding**: Iterates `model.modules()`, for layers containing `k_cache`/`v_cache`: `module.k_cache = kv_cache[0, layer_id]` (view).

Block ID allocation/release is maintained by **`BlockManager`** (kv-cache skill), with one-to-one correspondence to physical tensor rows.

## nano-sglang: Two-Level Pool

### ReqToTokenPool

- `req_to_token`: `[max_reqs, max_context_len]` int32, **one row per request**, stores the **global KV slot index** for each position.
- `mem_state`: `[max_reqs]` bool, **1=that request slot is available (unoccupied)**, set to 0 on `alloc`; opposite to the common "1=occupied" intuition, follow the source code.
- `alloc(need_size)`: `nonzero(mem_state)` takes the first `need_size` request slot indices; restores to 1 on `free`.

### TokenToKVPool

- `kv_data[layer]`: `[size, 2, head_num, head_dim]` (one tensor per layer, adapted for Triton/FlashInfer).
- `mem_state`: **int16 reference count** (0 means allocatable); `alloc` → `add_refs`; `free` → `decrease_refs`.
- `alloc_contiguous(need_size)`: Finds **contiguous** ranges in free indices, satisfying some kernel preferences.
- `get_kv_data_flashinfer`: Reshapes to FlashInfer NHD page format.

**Data flow**: Radix `match_prefix` gets existing value tensor → new tokens `alloc` new slots → `req_to_token[req, pos] = slot_idx`; attention scatter-writes to `kv_data` by slot.

## mini-sglang: MHAKVCache

- **`_kv_buffer`**: `[2, num_layers, num_pages, page_size, local_kv_heads, head_dim]`; under TP: `local_kv_heads = div_even(...)`.
- **`k_cache(i)` / `v_cache(i)`**: K/V sub-tensor view of layer `i`.
- **`store_kv(k, v, out_loc, layer_id)`**: Calls `store_cache` Triton kernel, writes into flattened `view(num_pages*page_size, heads, dim)` by `out_loc`.

Aligned with **RadixPrefixCache**: `insert_prefix` length aligned by `page_size`, ensuring index consistency with page boundaries.

## Capacity Estimation Template (Agent-Usable)

```text
bytes_per_token = 2 * num_layers * num_kv_heads_after_tp * head_dim * dtype_bytes
# When paged:
bytes_per_block = bytes_per_token * block_size   # nano-vllm
# Or pages:
bytes_per_page = bytes_per_token * page_size     # mini-sglang MHA
available = total_gpu_bytes * utilization - model_and_activation_peak
num_slots = available // bytes_per_block_or_page
```

GQA: Use **KV head count** not Q head count.

## Checklist for Agent-Generated Inference Frameworks

1. Is the pool calculated by peak memory after **model loading + warmup**?
2. Is dtype/device consistent with the model? Under TP, is head dimension divided by `world_size`?
3. For prefix sharing, is **reference counting** used instead of simple bool?
4. Are the **buffer views** needed by runner exposed (by layer / by page)?
5. **Free** order of `ReqToTokenPool` and `TokenToKVPool`: release request rows first, then decrement KV references, to avoid dangling indices.

## Further Reading

- [06_memory_pool.md](../06_memory_pool.md)
- [01_architecture.md](../01_architecture.md) (source code comparison table)

**Cursor skill copy**: `meta-infer/.cursor/skills/memory-pool/SKILL.md` (for Agent discovery)
