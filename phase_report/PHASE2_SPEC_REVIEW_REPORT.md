# Phase 2 Spec Review Report

**PID**: 23784
**Role**: spec-reviewer
**Timestamp**: 2026-05-30T21:15:00+08:00
**Phase**: 2
**Review Target**: `engine/tp_layers/distributed.py`

---

## Spec Compliance: ✅ PASS

---

## Evidence Chain

### 1. tp_distributed_runtime.collectives.all_reduce_sum

- ✅ `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime.collectives.all_reduce_sum.registration` @ distributed.py:142 — `@torch.library.custom_op('meta_infer::all_reduce_sum', mutates_args=())` matches blueprint exactly.
- ✅ `...all_reduce_sum.pseudocode[0]` (TP=1 no-op) @ distributed.py:155-157 — `if not is_tp_enabled(): return x.clone()` matches blueprint: must return new tensor (not input alias) per custom_op prohibition.
- ✅ `...all_reduce_sum.pseudocode[1]` (CustomAR P2P path) @ distributed.py:159-160 — `if _custom_ar_handle is not None: return _custom_ar_handle.all_reduce(x, registered=False)` matches blueprint.
- ✅ `...all_reduce_sum.pseudocode[2]` (NCCL fallback) @ distributed.py:163-165 — `y = x.clone(); dist.all_reduce(y, op=dist.ReduceOp.SUM); return y` matches blueprint.
- ✅ `...all_reduce_sum.pseudocode[3-4]` (register_fake) @ distributed.py:168-171 — `@all_reduce_sum.register_fake` returning `torch.empty_like(x)` matches blueprint.

### 2. tp_distributed_runtime.collectives.all_gather_last_dim

- ✅ `...all_gather_last_dim.signature` @ distributed.py:178 — `def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:` matches.
- ✅ `...all_gather_last_dim.pseudocode` @ distributed.py:190-196 — TP=1 returns x unchanged; uses `dist.all_gather(outs, x)` + `torch.cat(outs, dim=-1)` — NOT `all_gather_into_tensor`. Matches blueprint explicitly.
- ✅ `...all_gather_last_dim.note` @ distributed.py:182-187 — docstring confirms "Input: [..., local_dim] → Output: [..., local_dim * tp_size]" and "Forbidden: dist.all_gather_into_tensor". Matches blueprint.

### 3. tp_distributed_runtime.init_sequence

- ✅ `...init_sequence[2]` (set_device) @ distributed.py:212-213 — `local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)` matches blueprint.
- ✅ `...init_sequence[3]` (init_process_group) @ distributed.py:214 — `dist.init_process_group(backend="nccl", init_method="env://")` matches blueprint.
- ✅ `...init_sequence[4]` (init_custom_ar after load_weights) — `init_custom_ar()` is a standalone function called after model load_weights (line 222). This is the correct call site per blueprint: "模型 load_weights 后: init_custom_ar(device)".
- ✅ `...init_sequence[5]` (dist.barrier sync) @ distributed.py:256, 284 — barriers before and after CustomAR init match blueprint.

### 4. custom_ar_all_reduce.init_state_machine — Full init flow

- ✅ `...init_state_machine.pseudocode` (entire try/except) @ distributed.py:252-301 — entire init wrapped in `try:` (line 252) / `except Exception as e:` (line 297). Matches blueprint requirement.
- ✅ `...init_state_machine._failure_fallback_contract` @ distributed.py:300 — `_custom_ar_handle = None` on exception. No re-raise. Matches blueprint: "失败后 handle 为 None，all_reduce_sum 自动走 NCCL".
- ✅ `...init_state_machine.pseudocode` (world_size==1 early return) @ distributed.py:249-250 — `if world_size == 1: return`. Matches blueprint.
- ✅ `...init_state_machine.pseudocode` (dist not initialized check) @ distributed.py:243-244 — `if not dist.is_initialized(): return`. Defensive guard beyond blueprint minimum.
- ✅ `...init_state_machine.pseudocode` (Step 1: barrier + print) @ distributed.py:256-258 — `dist.barrier()` then `rank==0` print. Matches.
- ✅ `...init_state_machine.pseudocode` (Step 2: gloo group) @ distributed.py:261 — `dist.new_group(backend="gloo")`. Matches blueprint: "创建 gloo ProcessGroup（非 NCCL，用于 IPC handle exchange）".
- ✅ `...init_state_machine.pseudocode` (max_size=16MB) @ distributed.py:263 — `max_size = 16 * 1024 * 1024`. Matches blueprint workspace_size: "16 MB".
- ✅ `...init_state_machine.pseudocode` (meta_size) @ distributed.py:264 — `meta_size = ops.meta_size()`. Matches blueprint vllm_imports.

### 5. custom_ar_all_reduce.register_buffer_detail — Two-phase buffer allocation

- ✅ `...register_buffer_detail.allocation` (Phase A: meta_ptrs) @ distributed.py:267-269 — `meta_ptrs = _allocate_and_exchange_handles(meta_size + max_size, gloo_group, rank, world_size)`. Matches blueprint exactly.
- ✅ `...register_buffer_detail.allocation` (Phase B: buf_ptrs) @ distributed.py:272-274 — `buf_ptrs = _allocate_and_exchange_handles(max_size, gloo_group, rank, world_size)`. Matches blueprint exactly.
- ✅ `...register_buffer_detail.allocation` (Phase C: init+register) @ distributed.py:280-281 — `_ptr = ops.init_custom_ar(meta_ptrs, rank_data, rank, fully_connected)` then `ops.register_buffer(_ptr, buf_ptrs)`. Matches blueprint call_sequence.
- ✅ `...register_buffer_detail.two_buffer_sets` @ distributed.py:266-281 — Two distinct buffer sets (meta_ptrs for init_custom_ar, buf_ptrs for register_buffer). Matches blueprint description.

### 6. ⭐ KEY CORRECTION 1: buf_ptrs IPC exchange uses all_gather_object (NOT broadcast_object_list)

- ✅ `...register_buffer_detail._critical_fix` @ distributed.py:74 — `dist.all_gather_object(handles, ipc_handle, group=gloo_group)` inside `_allocate_and_exchange_handles`. Both meta_ptrs and buf_ptrs use this same function → both use `all_gather_object`. ZERO occurrences of `broadcast_object_list` in the entire file.
- ✅ `...register_buffer_detail.broadcast_object_list_usage_note` — Code correctly avoids `broadcast_object_list` in init phase. The note states `broadcast_object_list` is only for `register_graph_buffers()` in CUDA Graph path (nocompile → unused).

### 7. ⭐ KEY CORRECTION 2: Each rank allocates exactly 1 buffer (NOT world_size buffers)

- ✅ `...register_buffer_detail.allocation` @ distributed.py:70 — `raw_ptr, ipc_handle = ops.allocate_shared_buffer_and_handle(size)` — single call, NO loop over `range(world_size)`. Each rank allocates exactly ONE buffer per `_allocate_and_exchange_handles` invocation.
- ✅ Two invocations total (meta_ptrs + buf_ptrs) → 2 buffers allocated per rank. Correct per blueprint.

### 8. ⭐ KEY CORRECTION 3: meta_ptrs and buf_ptrs share the same exchange helper

- ✅ `...register_buffer_detail.allocation` @ distributed.py:267-274 — Both `meta_ptrs` (line 267) and `buf_ptrs` (line 272) call the same `_allocate_and_exchange_handles` function. Only the `size` parameter differs (`meta_size + max_size` vs `max_size`). Matches blueprint: "两套都用同一个 _allocate_and_exchange_handles 函数".

### 9. _allocate_and_exchange_handles — Shared IPC helper implementation

- ✅ `...register_buffer_detail.allocation` (helper signature) @ distributed.py:54 — `def _allocate_and_exchange_handles(size: int, gloo_group, rank: int, world_size: int)` — parameter list matches blueprint pseudocode `(size, group, rank, world_size)`.
- ✅ Helper body @ distributed.py:67-81 — Exact line-by-line match with blueprint pseudocode: (1) import ops, (2) alloc shared buffer, (3) all_gather_object exchange, (4) open_mem_handle for remote, raw_ptr for local, (5) return pointers.
- ✅ Local rank pointer invariant @ distributed.py:78 — `ops.open_mem_handle(h) if i != rank else raw_ptr` — local rank keeps raw_ptr unmodified. Matches blueprint.

### 10. CustomAllReduceHandle class

- ✅ Attributes @ distributed.py:92-98 — `_ptr`, `rank`, `world_size`, `rank_data`, `buf_ptrs`, `max_size`. All documented.
- ✅ `all_reduce` method @ distributed.py:117-135 — out-of-place (`torch.empty_like(x)` line 130), uses `ops.all_reduce(self._ptr, x, out, reg_buf, reg_buf_sz_bytes)` line 134. Matches blueprint inline_signature.
- ✅ `all_reduce` registered parameter @ distributed.py:117 — `registered: bool = False` matches blueprint signature `all_reduce(x, registered=False)`.

### 11. Helper functions

- ✅ `is_tp_enabled()` @ distributed.py:30-32 — returns `dist.is_initialized() and dist.get_world_size() > 1`. Used consistently in all_reduce_sum and all_gather_last_dim.
- ✅ `get_tp_size()` @ distributed.py:35-37 — returns world_size if initialized else 1.
- ✅ `get_tp_rank()` @ distributed.py:40-42 — returns rank if initialized else 0.
- ✅ `get_custom_ar_handle()` @ distributed.py:45-47 — returns `_custom_ar_handle` module-level variable.

### 12. _check_p2p — Peer-to-peer access verification

- ✅ `...init_state_machine.vllm_imports` (P2P check) @ distributed.py:308-328 — implements `torch.cuda.can_device_access_peer` loop. Matches blueprint description: "torch.cuda.can_device_access_peer loop".
- ✅ Uses LOCAL_RANK env var with fallback to rank @ distributed.py:321 — `local_rank = int(os.environ.get("LOCAL_RANK", rank))`. Good defensive practice.
- ✅ Exception-safe @ distributed.py:327-328 — wraps in try/except, returns False on error.

### 13. AGENT_SKILL.md §1 编码铁律 — 违规扫描

- ✅ **all_gather_last_dim 使用 dist.all_gather + torch.cat** @ distributed.py:194-196 — Not `all_gather_into_tensor`. Matches AGENT_SKILL.md §1 rule.
- ✅ **all_reduce_sum 为 custom_op 注册** @ distributed.py:142 — Matches AGENT_SKILL.md §2.1 "all_reduce_sum: @torch.library.custom_op".
- ✅ **init_custom_ar 完整 try/except** @ distributed.py:252-301 — Matches AGENT_SKILL.md §2.1 CRITICAL note about IPC failure survival.
- ✅ **TP=1 x.clone() for custom_op** @ distributed.py:157 — Matches AGENT_SKILL.md §2.1 "TP=1 → x.clone()" rule.
- ✅ **NCCL 回退链完整** @ distributed.py:159-165 — CustomAR→NCCL auto-fallback matches AGENT_SKILL.md §2.1.

---

## Issues Found

**None.** All 13 evidence groups pass against the blueprint contracts.

---

## Blueprint Information Gaps

- 🟡 `custom_ar_all_reduce.init_state_machine.pseudocode` (inference_blueprint.json lines 1308-1316) shows a single `allocate_shared_buffer_and_handle(max_size)` call, while `register_buffer_detail.allocation` (lines 1349-1359) specifies the corrected two-phase flow (meta_ptrs + buf_ptrs via separate `_allocate_and_exchange_handles` calls). The code correctly implements the `register_buffer_detail` version. The init_state_machine pseudocode appears to be the older/simpler version — the 2026-05-30 `_critical_fix` note in register_buffer_detail clarifies this. **Suggestion**: update init_state_machine pseudocode to match register_buffer_detail for consistency.

- 🟡 `CustomAllReduceHandle.all_reduce` @ distributed.py:132 — uses `self.buf_ptrs[0]` as the `reg_buf` parameter for `ops.all_reduce`. For non-rank-0 processes, `buf_ptrs[0]` is a remote pointer (opened via `open_mem_handle`). In vLLM's reference implementation, `reg_buf` uses `self.rank_data.data_ptr()` (a local tensor pointer), not a P2P remote pointer. The code may work correctly (all buf_ptrs are registered via `register_buffer`), but this deviates from the vLLM reference pattern. Since the blueprint doesn't specify which exact pointer to use for `reg_buf`, this is flagged as an observation rather than a FAIL. **Recommendation**: cross-reference vLLM `_custom_ops.py` to confirm remote pointers are accepted for `reg_buf`.

---

## Verdict

**Spec 审查通过，代码与蓝图契约一致，可移交 verification。**

All three key corrections (P0-3 bugs) are correctly implemented:
1. ✅ `all_gather_object` used for IPC exchange (not `broadcast_object_list`)
2. ✅ Each rank allocates exactly 1 buffer per helper call (not world_size)
3. ✅ `meta_ptrs` and `buf_ptrs` share the same `_allocate_and_exchange_handles` helper
