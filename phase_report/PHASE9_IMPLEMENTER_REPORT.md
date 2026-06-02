# Phase 9 Implementer Report

**PID**: 293083 (Python process that ran syntax checks)
**Role**: implementer
**Timestamp**: 2026-05-30
**Phase**: 9
**Status**: SUBMITTED

## Implemented

### New Files

1. **engine/memory_pool.py** — KVMemoryPool class
   - `__init__(self, num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype_size=2)`
   - `estimate_num_blocks_dense()` static method: Dense KV budget formula (K+V per token = layers * kv_heads * head_dim * 2 * elem_bytes)
   - GPU placeholders NOT created in TP path (KV managed by QwenAttentionTP internally)
   - Blueprint: components[1] KVMemoryPool + _estimate_kv_blocks.dense_pseudocode

2. **llm_engine.py** — LLMEngine + QwenTPModelRunner + RealModelRunner (stub)
   - **QwenTPModelRunner**:
     - `__init__`: loads QwenForCausalLMTP + tokenizer, sets eos_token_id with fallback chain
     - `get_num_free_blocks()`: kv_len from layer[0]._kv_len_gpu → max_blocks - allocated
     - `run(seqs, is_prefill, temperature, top_p)`: prefill (ragged cat + model.forward) / decode (model.forward_decode) dispatch + tp_sample
   - **RealModelRunner**: stub that raises NotImplementedError (HF path not in scope)
   - **LLMEngine**:
     - `__init__` 7-step flow: set_device → route backend → create runner → eos_token_id → block_size (256 TP / 16 HF) → estimate KV blocks → KVMemoryPool + Scheduler with injected _block_size + _max_blocks
     - `_select_tp_backend()`: reads config.json architectures[0], routes Qwen→qwen_tp, DeepSeek→deepseek_tp
     - `_estimate_kv_blocks()`: torch.cuda.mem_get_info → Dense formula via KVMemoryPool.estimate_num_blocks_dense
     - `generate(prompt, max_new_tokens, temperature, top_p) → str`: enqueue → while-loop(schedule→run→postprocess) → decode
     - `begin_generation(prompts, ...)`: batch enqueue for step-based API
     - `has_unfinished_requests() → bool`: checks scheduler.is_finished()
     - `step(temperature, top_p) → list[Sequence]`: single scheduling step + finish detection
     - `get_generation_outputs() → list[str]`: decoded texts for active sequences
     - `_get_num_free_blocks()`: routes TP→runner.get_num_free_blocks(), HF→_max_blocks fallback
     - `_enqueue()`: tokenizer.encode → Sequence(sampling params) → scheduler.add
     - `_check_finish(seq)` + `_finish_cleanup(seq)`: EOS/max_tokens detection + FINISHED transition + resource release

### Modified Files

3. **engine/models/qwen.py** — added `forward_decode()` method to QwenForCausalLMTP (line 518-545)
   - Signature: `forward_decode(self, input_ids, positions, kv_len, max_seq_len) → logits`
   - Handles decode path: embed_tokens → layer.forward_decode loop → norm → lm_head
   - Does NOT modify existing `forward()` method — purely additive

## Blueprint Nodes Read

- `framework_layer.components[6] LLMEngine.full_api_surface` — __init__ 7-step flow + generate 5-step while-loop + step/begin_generation/has_unfinished/get_outputs API
- `framework_layer.components[3] ModelRunner.tp_runner_actual_flow` — run_method_impl, prefill/decode dispatch, get_num_free_blocks
- `framework_layer.components[1] KVMemoryPool` — responsibility boundary, TP path note
- `framework_layer.data_flow_contracts.scheduler_tp_runner_bridge` — block_size injection, num_free routing, BlockManager degradation, prefill_timing_gap
- `framework_layer.data_flow_contracts.scheduler_to_runner.schedule_complete_method` — schedule(num_free) signature
- `framework_layer.data_flow_contracts.scheduler_to_runner.postprocess_complete_method` — postprocess(batch, is_prefill, tokens)
- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.decode_forward_pattern.unified_signature` — forward_decode signature
- AGENT_SKILL.md §2.3 LLMEngine API 表面 (Phase 9 构建目标)
- AGENT_SKILL.md §2.0.1 Phase 9 完整知识链路

## Self-Diff Review

- Verified `engine/memory_pool.py` syntax: PASS
- Verified `llm_engine.py` syntax: PASS
- Verified `engine/models/qwen.py` syntax after edit: PASS (forward_decode added without modifying existing code)
- Verified no changes to `scripts/` directory: CONFIRMED
- Verified no changes to other engine files: CONFIRMED (only qwen.py modified, and only additive)
- Tested `KVMemoryPool.estimate_num_blocks_dense()` with sample values: produces 193 blocks (reasonable for 10GB free)

## Known Issues

- None known. All blueprint contracts for Phase 9 are implemented.
- RealModelRunner (HF path) is a stub — by design, only TP path is in scope.
- DeepSeek TP runner is a stub — by design, only Qwen TP is in scope.
- The prefill path calls the existing `model.forward(input_ids, past_key_values=None, max_seq_len=...)` which returns `(logits, kv_lens)` tuple; the runner unpacks this with `logits, _ = ...`. This is compatible with the existing forward signature without modification.

## File Manifest

| File | Action | Lines |
|------|--------|-------|
| `engine/memory_pool.py` | NEW | ~50 |
| `llm_engine.py` | NEW | ~300 |
| `engine/models/qwen.py` | ADDED method `forward_decode` | +28 |
