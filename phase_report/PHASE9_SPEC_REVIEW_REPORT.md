# Phase 9 Spec Review Report

| Field | Value |
|-------|-------|
| PID | 4035417 |
| Role | spec-reviewer |
| Timestamp | 2026-06-09T00:00:00Z |
| Phase | 9 |

---

## Spec Compliance: ❌ FAIL

6 contract violations found across Scheduler, ModelRunner, and Model forward() bridges.

---

## Evidence Chain (Verified Contracts)

### LLMEngine.__init__ — 7 steps (components[6].full_api_surface.__init__)

- **Step 1 (device + TP init)**: ✅ @ `llm_engine.py:67-71` — `torch.cuda.set_device(local_rank)`, `self.device = torch.device(f"cuda:{local_rank}")`, `self.dtype = torch.bfloat16`, `init_tp_distributed()`
- **Step 2 (_select_tp_backend)**: ✅ @ `llm_engine.py:76-78,130-150` — Reads `config.json` `architectures[0]`, routes `"Qwen"→"qwen_tp"`, `"Deepseek"/"DeepSeek"→"deepseek_tp"`, else `ValueError`
- **Step 3 (create Runner)**: ✅ @ `llm_engine.py:83-90` — `TPModelRunner(model_dir, tp_size=tp_size)` created for `qwen_tp`/`deepseek_tp`. `self.block_size = 256` set
- **Step 4 (eos_token_id)**: ✅ @ `llm_engine.py:95` — `self.eos_token_id = self.runner.tokenizer.eos_token_id`
- **Step 5 (_estimate_kv_blocks)**: ✅ @ `llm_engine.py:156-163` — `max_pos // self.block_size` (dynamic from `cfg.max_position_embeddings`)
- **Step 6 (BlockManager tp_mode)**: ✅ @ `llm_engine.py:107-111` — `BlockManager(num_blocks=max_blocks, tp_mode=(inference_backend in ('qwen_tp','deepseek_tp')), block_size=self.block_size)`
- **Step 7 (Scheduler block_size injection)**: ✅ @ `llm_engine.py:116-118` — `Scheduler(block_size=self.block_size, max_blocks=max_blocks)` with `block_size=256`

### LLMEngine Public API (components[6].full_api_surface)

- **generate() signature**: ✅ @ `llm_engine.py:169-175` — `generate(self, prompts, max_new_tokens=256, temperature=0.0, top_p=None) → Union[str, List[str]]`
- **step() signature**: ✅ @ `llm_engine.py:259-263` — `step(self, temperature=0.0, top_p=None) → List[Sequence]`
- **has_unfinished_requests()**: ✅ @ `llm_engine.py:241-253` — Returns `bool(self._waiting) or bool(self._running)`
- **get_generation_outputs()**: ✅ @ `llm_engine.py:338-350` — `tokenizer.decode(seq.output_ids)`

### CRITICAL-01 Bridge (scheduler_tp_runner_bridge)

- **block_size=256 for TP**: ✅ @ `llm_engine.py:85,117` — Injected to Scheduler
- **BlockManager(tp_mode=True)**: ✅ @ `llm_engine.py:107-111, block_manager.py:71-73,119-120` — allocate/free are no-ops
- **num_free from runner**: ✅ @ `llm_engine.py:281-282` — TP path routes to `runner.get_num_free_blocks()`
- **KV cache QwenAttentionTP self-managed**: ✅ @ `qwen.py:183-202,293-295` — lazy alloc, torch.arange, paged KV

### ModelRunner / TPModelRunner (components[3].tp_runner_actual_flow)

- **Prefill dispatch**: ✅ @ `engine/framework/model_runner.py:130-135` — `model(input_ids, past_key_values=None, ...)` → `s.kv_len = s.seq_len()`
- **Decode dispatch**: ✅ @ `engine/framework/model_runner.py:151-156` — `model(input_ids, past_key_values=kv_len, ...)` → `s.kv_len += 1` (CPU arithmetic)
- **Sampling**: ✅ @ `engine/framework/model_runner.py:163-165` — `sampler.sample(logits[:, -1, :], temperature, top_p)`
- **get_num_free_blocks()**: ✅ @ `engine/framework/model_runner.py:173-186` — follows `scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner.impl` formula

### QwenForCausalLMTP (qwen3_tp_model_interfaces)

- **class_hierarchy**: ✅ @ `qwen.py:659-687` — `embed_tokens`, `layers` (ModuleList), `norm`, `lm_head`; `self.self_attn` not `self.attention`
- **QwenAttentionTP attrs**: ✅ @ `qwen.py:110-177` — All attrs present: `_kv_block_size=256`, `_key_cache/_value_cache=None`, `_kv_len_gpu` (int32), `_slot_mapping_decode` (int64)
- **@torch.inference_mode()**: ✅ @ `qwen.py:852` — on `QwenForCausalLMTP.forward()`, covers both prefill and decode paths
- **Prefill path**: ✅ @ `qwen.py:877-894` — embed → layers.forward() loop → norm → lm_head → logits
- **Decode path**: ✅ @ `qwen.py:896-916` — embed → layers.forward_decode() loop → norm → lm_head → logits

### Iron Laws (AGENT_SKILL.md encoding rules)

- **block_size=256 for TP**: ✅ @ `llm_engine.py:85`, `qwen.py:157`
- **block_table dtype=int32**: ✅ @ `qwen.py:201`, `qwen.py:294`, `sequence.py:117`
- **Q-K-V cat order**: ✅ @ `qwen.py:745,835` — `torch.cat([q, k, v], dim=0)`
- **Gate-Up cat order**: ✅ @ `qwen.py:753,844` — `torch.cat([gate, up], dim=0)`
- **fused_add_rms_norm uses self.weight**: ✅ @ `qwen.py:502,515,546,557` — `self.input_layernorm.weight` / `self.post_attention_layernorm.weight`
- **KV head replication**: ✅ @ `qwen.py:120-124` — `tp > num_kv_heads → num_kv_heads=1`
- **Dimensions from config.json dynamic read**: ✅ @ `qwen.py:605-649` — `QwenTPConfig.from_config()`, no hardcoded dims
- **_slot_mapping_decode dtype=int64**: ✅ @ `qwen.py:168`
- **No .item() in llm_engine.py**: ✅ @ `llm_engine.py` — zero matches
- **No .item() in qwen.py forward paths**: ✅ @ `qwen.py` — only in comments

### Load weights (qwen_hf_key_mapping + load_weights_pseudocode)

- **14 HF key mappings**: ✅ @ `qwen.py:763-828` — All mapped: embed_tokens, lm_head, norm, 3x QKV, o_proj, q_norm, k_norm, 3x MLP (gate/up/down), 2x layernorm
- **barrier + CustomAR after load**: ✅ @ `qwen.py:757-761` — `dist.barrier()` then `init_custom_ar()`
- **Q-K-V merge order**: ✅ @ `qwen.py:745,835` — Q-K-V, not K-Q-V
- **Gate-Up merge order**: ✅ @ `qwen.py:753,844` — gate-up, not up-gate
- **Double shard guard**: ✅ @ `qwen.py:747,755,792` — `load_weight_shard()` internally handles

### dtype / device / contiguous

- **bfloat16 throughout**: ✅ @ `llm_engine.py:70`, `model_runner.py:68`, `qwen.py:196`
- **Device routing correct**: ✅ @ `llm_engine.py:67-69`, `model_runner.py:63-67`
- **block_table int32**: ✅ @ multiple locations
- **Positions int64**: ✅ @ `qwen.py:882,904`
- **input_ids_tensor explicit device**: ✅ @ `model_runner.py:126,145`

### Phase 9 eager gates

- **no .item() in forward_decode**: ✅ @ `qwen.py` — batch kv_len read external to model
- **no dead loop**: ✅ @ `llm_engine.py:213-214,241-253` — proper while-loop with all_finished detection
- **overlength rejection**: ✅ @ `scheduler.py:95-100` — REJECTED status for `required_blocks > max_blocks`

---

## Issues Found (6 violations)

---

### Issue 1: QwenForCausalLMTP.forward() return signature — missing kv_lens 2-tuple

- **JSON Path**: `qwen3_tp_model_interfaces.model_forward_pseudocode`
- **File:Line**: `engine/models/qwen.py:894` (prefill return), `engine/models/qwen.py:916` (decode return)
- **Expected**: `return logits, kv_lens` — model.forward() must return a 2-tuple per blueprint lines 1126 and 1235. Prefill should return `kv_lens = None`. Decode must return `kv_lens = [int(l.self_attn._kv_len_gpu[0].item()) for l in self.layers]` (extracted after all layers complete, per line 1123 / 1232).
- **Actual**: Both prefill (line 894) and decode (line 916) return only `logits`. No `kv_lens` is returned. The runner compensates by tracking `s.kv_len` via CPU arithmetic (`s.kv_len = s.seq_len()` / `s.kv_len += 1`), but the model's forward() contract is violated.
- **Fix**: In decode branch (before line 916), add kv_lens extraction: `kv_lens = [int(l.self_attn._kv_len_gpu[0].item()) for l in self.layers]`. Change return to `return logits, kv_lens`. Update runner to unpack 2-tuple: `logits, new_kv_lens = self.model(...)`.

### Issue 2: begin_generation() API signature mismatch

- **JSON Path**: `components[6].full_api_surface.begin_generation`
- **File:Line**: `llm_engine.py:224-235`
- **Expected**: `begin_generation(self, prompts, max_new_tokens, temperature, top_p) -> None` — the method should accept raw prompt strings plus tokenization params, perform `tokenizer.encode` internally, create Sequence objects, and add them to the scheduler's waiting queue. This is the blueprint's documented API for "批量加入 prompt 到调度器".
- **Actual**: `begin_generation(self, seqs: List[Sequence]) -> None` — accepts pre-created Sequence objects. Tokenization is done externally in `_enqueue()`. The method only sets status=WAITING and appends to `_waiting`.
- **Fix**: Implement `begin_generation()` matching the blueprint signature — it should internally call `_enqueue()` and handle the full prompt-to-sequence pipeline. Current method can be kept as a private helper `_add_seqs_to_waiting()`.

### Issue 3: Scheduler queues owned by LLMEngine, not Scheduler itself

- **JSON Path**: `data_flow_contracts.scheduler_to_runner.schedule_complete_method`
- **File:Line**: `engine/framework/scheduler.py:67-69` (__init__), `engine/framework/scheduler.py:75-79` (schedule signature)
- **Expected**: `Scheduler.__init__(self, memory_pool, max_num_seqs, max_num_batched_tokens)` with `self.waiting = []`, `self.running = []`. `schedule(self, num_free)` — single parameter, internally iterates `self.waiting` and `self.running`. Queues are internal state of the Scheduler, not external lists owned by LLMEngine.
- **Actual**: `Scheduler.__init__(self, block_size, max_blocks)` — no `memory_pool`, `max_num_seqs`, `max_num_batched_tokens`. No `self.waiting` or `self.running`. `schedule(self, waiting_seqs, running_seqs, num_free_blocks)` — three parameters, queues passed from LLMEngine. The Scheduler is stateless between calls.
- **Fix**: Move `waiting` and `running` lists into Scheduler as `self.waiting` and `self.running`. Add `add_request(seq)` method that LLMEngine calls to enqueue. Restore `schedule(self, num_free)` signature that operates on internal queues. This enables `_reserved_blocks` tracking (Issue 4).

### Issue 4: Scheduler missing _reserved_blocks counter (prefill_timing_gap unfixed)

- **JSON Path**: `scheduler_tp_runner_bridge.prefill_timing_gap`
- **File:Line**: `engine/framework/scheduler.py:75-136` (entire schedule method)
- **Expected**: The scheduler must maintain `_reserved_blocks` counter per blueprint pseudocode (lines 1697-1704). During prefill scheduling, `reserved` tracks blocks promised to pending prefill sequences. Decode check uses `num_free - reserved` to account for prefill reservations not yet reflected in `_kv_len_gpu`. `postprocess()` resets `_reserved_blocks = 0`.
- **Actual**: No `_reserved_blocks` field exists. Schedule uses a local variable `free = num_free_blocks` and decrements it directly as it adds sequences. After prefill returns, decode path reuses the already-decremented local `free` variable, which happens to work by accident for B=1 single-seq but violates the prefill_timing_gap fix contract.
- **Fix**: Add `self._reserved_blocks = 0` to `__init__`. In `schedule()`: use `num_free = num_free_blocks - self._reserved_blocks` as effective budget. Track reservations: `self._reserved_blocks += reserved` after prefill, `self._reserved_blocks += decode_count` after decode. In `postprocess()`: `self._reserved_blocks = 0`.

### Issue 5: Runner class name and file location

- **JSON Path**: `components[3].ModelRunner.impl_code`
- **File:Line**: `engine/framework/model_runner.py:34` (class definition)
- **Expected**: `engine/models/qwen.py::QwenTPModelRunner` — the blueprint lists impl_code as `["llm_engine.py::RealModelRunner", "engine/models/qwen.py::QwenTPModelRunner", "engine/models/deepseek_v2.py::DeepseekTPModelRunner"]`. The Qwen TP runner should be named `QwenTPModelRunner` and located in `engine/models/qwen.py`, co-located with the Qwen model class it wraps.
- **Actual**: `engine/framework/model_runner.py::TPModelRunner` — a single generic runner class for all backends in `engine/framework/`. This breaks the blueprint's separation of Qwen-specific and DeepSeek-specific runners.
- **Fix**: Rename class to `QwenTPModelRunner` and either move to `engine/models/qwen.py` or create a type alias `QwenTPModelRunner = TPModelRunner` in `engine/models/qwen.py`. Alternatively, if a single generic runner is the intended architecture, the blueprint should be updated to reflect this.

### Issue 6: .clone() present in QwenDecoderLayerTP.forward_decode()

- **JSON Path**: `phase_9_engine_integration.eager_gate.no_clone_in_forward_decode`
- **File:Line**: `engine/models/qwen.py:545`
- **Expected**: Eager gate: `"no_clone_in_forward_decode": "eager 路径 forward_decode 不含 clone()"`. Decode path must not contain any `.clone()` call. The blueprint specifies residual is never None in decode — it is carried forward from prefill.
- **Actual**: `forward_decode()` contains `residual = hidden_states.clone()` on line 545, guarded by `if residual is None`. While unreachable in normal execution (residual always passed from prefill's first layer), the `.clone()` call syntactically exists in `forward_decode()`.
- **Fix**: Remove the `if residual is None` guard and its `.clone()` body from `forward_decode()` (lines 544-551). Replace with an assertion or simply expect residual to always be non-None. If defensive coding is desired, use `torch.empty_like()` for a new residual buffer rather than `.clone()`.

---

## Blueprint Information Gaps

- **`.item() in get_num_free_blocks()` vs O2 zero_item_cpu_sync**: The `scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner.impl` pseudocode explicitly uses `.item()` to read `_kv_len_gpu`, but O2 says `get_num_free_blocks()` should return a constant with no `.item()`. These are contradictory. The current code follows the scheduler_tp_runner_bridge (using `.item()` at `model_runner.py:183`), which is a valid interpretation. Resolution needed in blueprint.

- **`max_seq_len` default value**: Blueprint pseudocode shows `max_seq_len=528` as default, but the actual model config yields `max_position_embeddings=40960` (for Qwen3-8B). The blueprint value is a placeholder; no functional impact since the real value is read from config.

- **@torch.inference_mode() count**: O1 audit_check says `grep '@torch.inference_mode' engine/models/qwen.py` should have 2 matches (for `forward()` and `forward_decode()` on `QwenForCausalLMTP`). The actual implementation has a single unified `forward()` method (line 852) that covers both prefill and decode via `past_key_values` branching. Only 1 match. This is a design trade-off (unified vs split forward) rather than a missing decorator — both prefill and decode paths are covered.

---

## Summary

- **Verified contracts**: 45+ checked and passed
- **Violations**: 6 (all structural/API contract deviations)
- **Uncontested iron laws**: All encoding rules verified clean
- **CRITICAL-01 bridge**: Fully satisfied
- **dtype/device/contiguous**: All consistent and correct
