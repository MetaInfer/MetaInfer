# PHASE1_IMPLEMENTER_REPORT.md

- **PID**: 888513
- **Role**: implementer
- **Timestamp**: 2026-05-30 04:30 UTC
- **Phase**: 1
- **Status**: SUBMITTED

## Implemented

Created 3 files under `engine/`:

1. **`engine/__init__.py`** — Package init for the engine module.
2. **`engine/kernels/__init__.py`** — Package init for kernels subpackage; re-exports all 7 kernel symbols.
3. **`engine/kernels/vllm_wrappers.py`** — 7 vLLM / flash_attn kernel wrappers:

| # | Kernel | Wrapper Signature | Ref Source |
|---|--------|-----------------|------------|
| 1 | `rms_norm` | `rms_norm(out, input, weight, epsilon) -> None` | vllm/_custom_ops.py:420-423 |
| 2 | `fused_add_rms_norm` | `fused_add_rms_norm(input, residual, weight, epsilon) -> None` | vllm/_custom_ops.py:420-423 |
| 3 | `silu_and_mul` | `silu_and_mul(out, input) -> None` | vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda |
| 4 | `rotary_embedding` | `rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox) -> None` | vllm/_custom_ops.py:400-410 |
| 5 | `cos_sin_cache` | `_get_cos_sin_cache(max_pos, head_dim, rope_theta) -> Tensor` | vllm/model_executor/layers/rotary_embedding/base.py:76-84 |
| 5b | `make_cos_sin_cache` | `make_cos_sin_cache(max_position, head_size, rope_theta, dtype, device) -> Tensor` | same |
| 6 | `flash_attn_varlen_func` | Re-exported from `flash_attn` (no wrapper needed) | flash_attn package |
| 7 | `flash_attn_with_kvcache` | Re-exported from `flash_attn.flash_attn_interface` (no wrapper needed) | flash_attn package |

### Key Design Decisions

- **Thin wrappers only**: All wrappers delegate directly to vLLM kernel calls with no extra logic.
- **`import vllm._C` at module top**: Triggers `torch.ops._C.silu_and_mul` registration at import time, before any `silu_and_mul()` call.
- **flash_attn direct re-exports**: nocompile scenario requires no `custom_op` registration per `inference_blueprint.json > _scope_note` (line 1226). Direct imports suffice.
- **Module-level cos_sin_cache registry**: `_cos_sin_cache_registry` dict shared across all layers, keyed by `(max_pos, head_dim, rope_theta)`. Returns CPU tensor for lazy GPU transfer.
- **`make_cos_sin_cache` produces `[max_pos, head_dim]`** shape (NOT `[max_pos, 2*head_dim]`). Uses fp32 computation internally, casts result to bf16.

## Blueprint Nodes Read

### inference_blueprint.json (JSON paths)

- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts` — all 7 kernel signatures, constraints, ref_source
  - `rms_norm` (line 1239): out 预分配, input contiguous, all bf16
  - `fused_add_rms_norm` (line 1246): 双 in-place, 本层 weight
  - `silu_and_mul` (line 1263): out 预分配, import vllm._C
  - `rotary_embedding` (line 1270): q/k in-place, 3D [tokens,heads,dim], is_neox=True
  - `custom_ar_all_reduce` (line 1282): Phase 2 material, not implemented here
  - `cos_sin_cache_strategy` (line 1274): registry + lazy GPU transfer
- `model_layer.architecture_knowledge_base.global_primitives_constraints.rmsnorm_precision_law` (line 2161): vLLM CUDA kernel mandatory, no PyTorch manual RMSNorm
- `agent_navigation` (line 9): how to find TP knowledge
- `todo_generation_playbook.phase_1_numeric_primitives` (line 1624): implementation todos + test commands
- `_scope_note` (line 1226): nocompile → flash_attn direct import, no custom_op

### AGENT_SKILL.md

- Section 1: 执行铁律 (execution prime directives)
- Section 2.0: Phase-Script binding table
- Section 2.0.1: Three-step knowledge chain for Phase 1

### ref_docs

- `notebooks-cn/07_improvementPlan/kernel_replacement_plan.md` — full document
  - Section 九: Complete kernel call contract table (9.1-9.6)
  - Section 三: Snippets A-F code templates
  - Section 0: Verification reports for all Snippets

### ref_code

- `vllm/_custom_ops.py:420-423` — rms_norm + fused_add_rms_norm signatures confirmed
- `vllm/_custom_ops.py:400-410` — rotary_embedding signature confirmed
- `vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda` — silu_and_mul confirmed uses `torch.ops._C.silu_and_mul`

### scripts/ test contracts (read-only)

- `scripts/test_phase1_kernel_wrappers.py` — 8 contracts (KERNEL-001 through KERNEL-008)
- `scripts/test_phase1_kernel_wrappers.sh` — environment dependency check

## Self-Diff Review

- [x] Complete implementation of all 7 kernel wrappers per Task spec — **YES**
- [x] No extra functionality beyond Phase 1 scope (YAGNI) — **YES** (only kernel wrapper functions, no model classes, no TP layers)
- [x] No modifications to `scripts/` directory — **CONFIRMED** (all scripts/ timestamps unchanged from May 28; only 3 new files under engine/)
- [x] RMSNorm uses vLLM kernel (not PyTorch manual) — **YES** (delegates to `vllm._custom_ops.rms_norm` / `fused_add_rms_norm`)
- [x] No `load_weights()` concerns in Phase 1 — **N/A** (Phase 1 has no weight loading)
- [x] No class_hierarchy naming concerns — **N/A** (Phase 1 is free functions only)
- [x] No accidental file modifications — **CONFIRMED** (only engine/__init__.py, engine/kernels/__init__.py, engine/kernels/vllm_wrappers.py created)

## Known Issues

- None. All 7 wrappers are thin passthrough calls to verified vLLM kernel signatures.
- The vLLM and flash_attn packages are not importable in this specific Python environment — this is expected as the test runner will use a conda environment with these packages installed. The `scripts/test_phase1_kernel_wrappers.sh` script explicitly verifies this.
