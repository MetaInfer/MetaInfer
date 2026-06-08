# Phase 1 Spec-Review Report

| 字段 | 值 |
|------|-----|
| Role | spec-reviewer |
| Phase | 1 |
| Timestamp | 2026-06-09T00:00:00Z |
| PID | os.getpid() |

---

## Evidence Chain — Contract Verification

### A. `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts`

#### A1. rms_norm

- JSON Path: `qwen3_kernel_contracts.rms_norm`
- Contract `inline_signature`: `def rms_norm(out: Tensor[*,H], input: Tensor[*,H], weight: Tensor[H], epsilon: float) -> None`
- Actual @ `engine/kernels/vllm_wrappers.py:19-24`: `def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None`
- **PASS** — parameter names, order, and types match the blueprint inline_signature exactly. No default values added. Return type `None` matches.

- JSON Path: `qwen3_kernel_contracts.rms_norm` (import path)
- Contract `ref_source`: `vllm/_custom_ops.py:420-423`
- Actual @ `engine/kernels/vllm_wrappers.py:16`: `from vllm._custom_ops import rms_norm as _vllm_rms_norm`
- **PASS** — import source matches. Uses vLLM CUDA kernel (complies with rmsnorm_precision_law, no hand-written PyTorch).

- JSON Path: `qwen3_kernel_contracts.rms_norm.constraint`
- Contract: `out 预分配 (empty_like)；input 必须 contiguous；out/input/weight 同 dtype`
- Actual: docstring @ `engine/kernels/vllm_wrappers.py:28-30` documents out as pre-allocated, input MUST be contiguous. Actual enforce at call site (not in wrapper — correct, wrapper is black-box).
- **PASS** — constraints correctly documented. Wrapper does not inject `.to()` or unnecessary Python loops (eager_gate check).

#### A2. fused_add_rms_norm

- JSON Path: `qwen3_kernel_contracts.fused_add_rms_norm`
- Contract `inline_signature`: `def fused_add_rms_norm(input!: Tensor[*,H], residual!: Tensor[*,H], weight: Tensor[H], epsilon: float) -> None`
- Actual @ `engine/kernels/vllm_wrappers.py:47-52`: `def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None`
- **PASS** — parameter names, order, and types match exactly.

- JSON Path: `qwen3_kernel_contracts.fused_add_rms_norm` (import path)
- Contract `ref_source`: `vllm/_custom_ops.py:420-423`
- Actual @ `engine/kernels/vllm_wrappers.py:44`: `from vllm._custom_ops import fused_add_rms_norm as _vllm_fused_add_rms_norm`
- **PASS** — import source matches.

- JSON Path: `qwen3_kernel_contracts.fused_add_rms_norm.constraint`
- Contract: `双 in-place: residual+=input; input=rms_norm(residual). 所有 fused_add_rms_norm 调用均使用本层的 self.input_layernorm.weight 或 self.post_attention_layernorm.weight`
- Actual: docstring @ `engine/kernels/vllm_wrappers.py:61-66` documents the dual in-place behavior correctly. Weight usage rule documented at line 65: "Post-mlp call uses THIS layer's post_attention_layernorm.weight. All 4 calls in decode use self-layer weight (physically traced)."
- **PASS** — constraints correctly documented.

#### A3. silu_and_mul

- JSON Path: `qwen3_kernel_contracts.silu_and_mul`
- Contract `inline_signature`: `torch.ops._C.silu_and_mul(out!: Tensor[*,d], input: Tensor[*,2*d]) -> None`
- Actual @ `engine/kernels/vllm_wrappers.py:81-84`: `def silu_and_mul(out: torch.Tensor, input: torch.Tensor) -> None`
- **PASS** — parameter order `(out, input)` matches the blueprint. Underlying call @ line 97: `torch.ops._C.silu_and_mul(out, input)` matches.

- JSON Path: `qwen3_kernel_contracts.silu_and_mul.wrapper` (import vllm._C trigger)
- Contract: `torch.ops._C.silu_and_mul(out!, input) (需 import vllm._C 触发注册)`
- Actual @ `engine/kernels/vllm_wrappers.py:78`: `import vllm._C  # noqa: F401 — triggers torch.ops._C.silu_and_mul registration`
- **PASS** — `import vllm._C` is present and executed before the function definition (line 78 before line 81). This triggers the `torch.ops._C.silu_and_mul` registration as required.

#### A4. rotary_embedding

- JSON Path: `qwen3_kernel_contracts.rotary_embedding`
- Contract `inline_signature`: `def rotary_embedding(positions: Tensor[N] int64, query!: Tensor[N,H,D], key!: Tensor[N,Kv,D]|None, head_size: int, cos_sin_cache: Tensor[M,D], is_neox: bool) -> None`
- Actual @ `engine/kernels/vllm_wrappers.py:109-116`: `def rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None, head_size: int, cos_sin_cache: torch.Tensor, is_neox: bool) -> None`
- **PASS** — parameter names, order, and types match the blueprint inline_signature exactly. `is_neox` has no default value (matches blueprint — blueprint specifies `is_neox: bool` without default).

- JSON Path: `qwen3_kernel_contracts.rotary_embedding.cos_sin_cache_strategy`
- Contract `registry`: `_cos_sin_cache_registry: dict[tuple, Tensor] = {}  # 模块级`
- Actual @ `engine/kernels/vllm_wrappers.py:144`: `_COS_SIN_CACHE_REGISTRY: dict[tuple, torch.Tensor] = {}`
- **PASS** — module-level registry dict exists. Name differs (`_COS_SIN_CACHE_REGISTRY` vs `_cos_sin_cache_registry`) but functional role identical; blueprint does not have a formal `inline_signature` requiring exact naming.

- Contract `factory`: `_get_cos_sin_cache(max_pos, head_dim, rope_theta): key=(max_pos,head_dim,rope_theta); if key not in registry: registry[key]=make_cos_sin_cache(...); return registry[key]`
- Actual @ `engine/kernels/vllm_wrappers.py:179-210`: function `_get_cos_sin_cache(max_pos, head_dim, rope_theta, dtype=bf16, device=None)` uses `key = (max_pos, head_dim, rope_theta)` and registry lookup pattern. Extra parameters `dtype` and `device` have defaults and do not break the 3-arg calling contract.
- **PASS** — core contract (3 required params, registry key, factory pattern) matches.

- Contract `lazy_gpu`: CPU creation in `__init__`, GPU transfer on first forward
- Actual: docstring @ `engine/kernels/vllm_wrappers.py:191-199` documents the lazy GPU transfer convention correctly.
- **PASS** — lazy GPU transfer strategy documented.

#### A5. flash_attn_varlen_func (Kernel 6)

- JSON Path: `todo_generation_playbook.phase_1_numeric_primitives.implementation_todos[5]`
- Contract: `flash_attn_varlen_func 直接 from flash_attn.flash_attn_interface import（nocompile 无需 custom_op）`
- Actual @ `engine/kernels/vllm_wrappers.py:219`: `from flash_attn import flash_attn_varlen_func`
- **FAIL** — import path does not match the blueprint contract.
  - Expected: `from flash_attn.flash_attn_interface import flash_attn_varlen_func`
  - Actual: `from flash_attn import flash_attn_varlen_func`
  - Fix: Change line 219 from `from flash_attn import flash_attn_varlen_func` to `from flash_attn.flash_attn_interface import flash_attn_varlen_func`
  - Note: Both paths resolve to the same function object (flash_attn top-level re-exports it), but the blueprint explicitly specifies `flash_attn.flash_attn_interface` as the import path. The nocompile direct-import constraint (no custom_op registration) is still satisfied.

#### A6. flash_attn_with_kvcache (Kernel 7)

- JSON Path: `todo_generation_playbook.phase_1_numeric_primitives.implementation_todos[6]`
- Contract: `flash_attn_with_kvcache 直接 from flash_attn.flash_attn_interface import（nocompile 无需 custom_op）`
- Actual @ `engine/kernels/vllm_wrappers.py:232`: `from flash_attn.flash_attn_interface import flash_attn_with_kvcache`
- **PASS** — import path matches blueprint exactly. Direct import, no custom_op registration (nocompile mode compliant).

### B. `model_layer.architecture_knowledge_base.global_primitives_constraints.rmsnorm_precision_law`

- JSON Path: `global_primitives_constraints.rmsnorm_precision_law`
- Contract: `RMSNorm 使用 vLLM 标品 CUDA kernel` — must use vLLM CUDA kernel, NOT hand-written PyTorch. Pattern: `from engine.kernels.vllm_wrappers import rms_norm; out = torch.empty_like(x); rms_norm(out, x.contiguous(), self.weight, self.eps)`
- Actual @ `engine/kernels/vllm_wrappers.py:16,36`: imports `vllm._custom_ops.rms_norm` and delegates to it. No hand-written PyTorch RMSNorm implementation present anywhere in the codebase.
- **PASS** — vLLM CUDA kernel used exclusively for both `rms_norm` and `fused_add_rms_norm`. No deprecated manual pattern (`x.float(); var = ...; out = ...`) detected.

### C. AGENT_SKILL.md Section 1 Encoding Rules (Phase 1 relevant)

- Rule: "契约优先" — all implementation constrained by `inference_blueprint.json`
- Actual: All signatures align with blueprint contracts (except A5 noted above).
- **PASS** (with one exception noted in FAIL section).

- Rule: "单路径优先" — no monolithic multi-branch code.
- Actual: `engine/kernels/vllm_wrappers.py` contains only 7 focused kernel wrappers, each with clear single-path delegation to vLLM/flash_attn backends.
- **PASS**.

### D. scripts/ Directory Integrity Check

- JSON Path: `CLAUDE.md` execution iron-law #3: `scripts/ 不可变`
- Actual: All 28 files in `scripts/` have timestamps `Jun 9 01:13` — no modifications.
- **PASS** — scripts/ directory untouched. No file created, modified, or deleted.

---

## Issues Found — FAIL

### FAIL-001: flash_attn_varlen_func import path mismatch

- **JSON Path**: `inference_blueprint.json > todo_generation_playbook.phase_1_numeric_primitives.implementation_todos[5]`
- **File:Line**: `engine/kernels/vllm_wrappers.py:219`
- **Expected**: `from flash_attn.flash_attn_interface import flash_attn_varlen_func`
- **Actual**: `from flash_attn import flash_attn_varlen_func`
- **Fix**: Change line 219 of `engine/kernels/vllm_wrappers.py`:
  ```python
  # Before:
  from flash_attn import flash_attn_varlen_func  # noqa: F401 — re-exported below

  # After:
  from flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: F401 — re-exported below
  ```
- **Rationale**: The blueprint contract at `implementation_todos[5]` explicitly specifies the import path `flash_attn.flash_attn_interface`. The code uses `flash_attn` (top-level re-export). While both resolve to the same function object, the contract is explicit about the source sub-module. This is the single deviation across all 7 kernel contracts.

---

## Blueprint Information Gaps

- `qwen3_kernel_contracts` does not have explicit `inline_signature` entries for `flash_attn_varlen_func` and `flash_attn_with_kvcache` as standalone kernel contract nodes. Their import paths are specified only in `phase_1_numeric_primitives.implementation_todos`. Recommend adding formal inline_signature entries under `qwen3_kernel_contracts` for completeness.

---

## Conclusion

**Spec Compliance: FAIL**

| # | Check Item | Result |
|---|-----------|--------|
| 1 | rms_norm signature (out, input, weight, epsilon) | PASS |
| 2 | rms_norm import (vllm._custom_ops) | PASS |
| 3 | fused_add_rms_norm signature (input, residual, weight, epsilon) | PASS |
| 4 | fused_add_rms_norm import (vllm._custom_ops) | PASS |
| 5 | silu_and_mul signature (out, input) | PASS |
| 6 | silu_and_mul uses torch.ops._C.silu_and_mul | PASS |
| 7 | silu_and_mul: import vllm._C triggers registration | PASS |
| 8 | rotary_embedding signature (positions, query, key, head_size, cos_sin_cache, is_neox) | PASS |
| 9 | rotary_embedding import (vllm._custom_ops) | PASS |
| 10 | rotary_embedding: is_neox default matches blueprint (no default, matches contract) | PASS |
| 11 | _get_cos_sin_cache: module-level registry + key tuple + factory | PASS |
| 12 | _get_cos_sin_cache: lazy GPU transfer strategy | PASS |
| 13 | flash_attn_with_kvcache: import path matches blueprint | PASS |
| 14 | **flash_attn_varlen_func: import path mismatch** | **FAIL** |
| 15 | rmsnorm_precision_law: uses vLLM CUDA kernel (no hand-written PyTorch) | PASS |
| 16 | scripts/ directory integrity: no modifications | PASS |
| 17 | No extra .to() / unnecessary Python loops in wrappers | PASS |
| 18 | engine/kernels/__init__.py exists with proper docstring | PASS |

**Final Verdict: FAIL**

Reason: 1 of 18 checks failed. flash_attn_varlen_func import path does not match the blueprint contract specification. All other 17 checks pass.

Implementer must fix FAIL-001 (line 219 of `engine/kernels/vllm_wrappers.py`) before this Phase can advance to verification.
