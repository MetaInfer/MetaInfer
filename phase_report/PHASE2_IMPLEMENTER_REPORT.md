# Phase 2 Implementer Report

**PID**: 925497
**Role**: implementer
**Timestamp**: 2026-05-30T05:57:45Z
**Phase**: 2
**Status**: SUBMITTED

---

## Implemented

### Files Changed
- `engine/tp_layers/distributed.py` — fully rebuilt (overwrite old file)
- `engine/tp_layers/__init__.py` — no changes needed (no broadcast-related deprecated symbols)

### Key Changes

#### Three bugs fixed (P0-3):

**Bug 1 — buf_ptrs IPC exchange: `broadcast_object_list` -> `all_gather_object`**

Old code:
```python
buf_handles = [None] * world_size
for src in range(world_size):
    obj = [buf_ptrs_alloc[src][1]]
    dist.broadcast_object_list(obj, src=src, group=gloo_group)
    buf_handles[src] = obj[0]
```

New code: Both meta_ptrs and buf_ptrs exchanges use `dist.all_gather_object` via the shared `_allocate_and_exchange_handles` helper. No `broadcast_object_list` anywhere in init_custom_ar.

**Bug 2 — buf_ptrs allocated world_size buffers instead of 1**

Old code:
```python
buf_ptrs_alloc = []
for _ in range(world_size):
    bp, bh = ops.allocate_shared_buffer_and_handle(max_size)
    buf_ptrs_alloc.append((bp, bh))
```

New code: `_allocate_and_exchange_handles` calls `ops.allocate_shared_buffer_and_handle(size)` exactly once per invocation. Each rank allocates 1 buffer, not world_size buffers.

**Bug 3 — No shared helper for the two exchange phases**

New helper extracted:
```python
def _allocate_and_exchange_handles(size, gloo_group, rank, world_size):
    from vllm import _custom_ops as ops
    raw_ptr, ipc_handle = ops.allocate_shared_buffer_and_handle(size)
    handles = [None] * world_size
    dist.all_gather_object(handles, ipc_handle, group=gloo_group)
    pointers = [ops.open_mem_handle(h) if i != rank else raw_ptr for i, h in enumerate(handles)]
    return pointers
```

Called twice in init_custom_ar:
- Phase A: `meta_ptrs = _allocate_and_exchange_handles(meta_size + max_size, gloo_group, rank, world_size)`
- Phase B: `buf_ptrs = _allocate_and_exchange_handles(max_size, gloo_group, rank, world_size)`

### Preserved unchanged (verified correct):
- `all_reduce_sum(x)`: @torch.library.custom_op -> CustomAR P2P -> NCCL fallback -> TP=1 x.clone()
- `all_reduce_sum.register_fake`: torch.empty_like(x)
- `all_gather_last_dim(x)`: dist.all_gather + torch.cat
- `init_tp_distributed()`: torch.cuda.set_device + dist.init_process_group('nccl', 'env://')
- `is_tp_enabled()`, `get_tp_size()`, `get_tp_rank()`, `get_custom_ar_handle()`
- `CustomAllReduceHandle` class with all_reduce method
- `_check_p2p()`: torch.cuda.can_device_access_peer
- Module-level `_custom_ar_handle: Any = None`, `_tp_initialized: bool = False`
- Entire `try/except` wrapping of init_custom_ar (failure -> handle=None -> NCCL fallback)

### init_custom_ar corrected flow (matches test_phase2_custom_ar_init.sh Phase A/B/C):

```
Phase A: _allocate_and_exchange_handles(meta_size + max_size, gloo_group, rank, world_size)
         -> meta_ptrs (each rank has raw pointer to every other rank's metadata+staging buffer)

Phase B: _allocate_and_exchange_handles(max_size, gloo_group, rank, world_size)
         -> buf_ptrs (each rank has raw pointer to every other rank's staging buffer)

Phase C: rank_data = torch.empty(max_size, dtype=uint8, device=cuda:{rank})
         _ptr = ops.init_custom_ar(meta_ptrs, rank_data, rank, fully_connected)
         ops.register_buffer(_ptr, buf_ptrs)
```

---

## Blueprint Nodes Read

### JSON Paths (from inference_blueprint.json):
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime`
  - `rank_size_contract`
  - `collectives.all_reduce_sum` (custom_op registration + pseudocode + register_fake)
  - `collectives.all_gather_last_dim` (pseudocode)
  - `init_sequence` (5-step bootstrap)
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce`
  - `constraint.init_state_machine` (full pseudocode with try/except)
  - `constraint.init_state_machine._failure_fallback_contract`
  - `constraint.register_buffer_detail` (_allocate_and_exchange_handles pseudocode)
  - `constraint.two_buffer_sets`
  - `constraint.broadcast_object_list_usage_note`

### Ref Documents:
- AGENT_SKILL.md section 1 (execution iron rules)
- AGENT_SKILL.md section 2.0 (Phase-Script binding table)
- AGENT_SKILL.md section 2.1 (code structure + engine files)
- AGENT_SKILL.md section 2.2 (CustomAR initialization notes)
- CLAUDE.md (Phase-Script binding quick reference)

---

## Self-Diff Review

- **Only `distributed.py` was modified** — `__init__.py` was read, confirmed no broadcast-related deprecated symbols to clean up, no changes needed.
- No files outside `engine/tp_layers/` were touched.
- No scripts/ files were modified.
- The `_allocate_and_exchange_handles` helper is correctly file-local (prefixed with `_`), not exported — matching the blueprint's scope.
- `from vllm import _custom_ops as ops` import is inside both the helper and `CustomAllReduceHandle.all_reduce` — consistent with old behavior (lazy import to avoid dependency failures at module load).

---

## Known Issues

- None. The three identified bugs (broadcast_object_list, world_size buffer loop, missing shared helper) have all been addressed.
- The `__init__.py` was verified to have no broadcast-related deprecated exports — no cleanup needed.
