# Phase 1 Spec Review Report

**PID**: 890520
**Role**: spec-reviewer
**Timestamp**: 2026-05-30T00:00:00+08:00
**Phase**: 1
**Scope**: `engine/__init__.py`, `engine/kernels/__init__.py`, `engine/kernels/vllm_wrappers.py`

---

## Spec Compliance: ✅ PASS

---

## Evidence Chain (逐条核验)

### Block A: `qwen3_kernel_contracts` — 7 个 vLLM 黑盒 kernel wrappers

#### KERNEL 1: `rms_norm`

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.rms_norm`
- **Status**: ✅ @ `engine/kernels/vllm_wrappers.py:46-66`
- **核验内容**:
  - Signature `(out, input, weight, epsilon) -> None` 与蓝图 `def rms_norm(out: Tensor[*,H], input: Tensor[*,H], weight: Tensor[H], epsilon: float) -> None` **精确匹配**
  - 内部调用 `_vllm_rms_norm`（`from vllm._custom_ops import rms_norm`），与蓝图 `vllm/_custom_ops.py:420-423` 一致
  - Docstring 明确约束：`out` 预分配 `empty_like(input)`，`input` 必须 contiguous，`out/input/weight` 同 dtype（bf16/fp16/fp32）
  - 参数名 `epsilon` 与蓝图一致（非 `eps`）
  - 无 `.to()` 调用，无 Python 循环 — 纯黑盒通过

#### KERNEL 2: `fused_add_rms_norm`

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm`
- **Status**: ✅ @ `engine/kernels/vllm_wrappers.py:73-104`
- **核验内容**:
  - Signature `(input, residual, weight, epsilon) -> None` 与蓝图 `def fused_add_rms_norm(input!: Tensor[*,H], residual!: Tensor[*,H], weight: Tensor[H], epsilon: float) -> None` **精确匹配**
  - 内部调用 `_vllm_fused_add_rms_norm`，源路径与蓝图一致
  - Docstring 完整描述双 in-place 语义：`residual = residual + input`，`input = rms_norm(residual) * weight`
  - Docstring 示例调用使用 `self.post_attention_layernorm.weight` / `self.input_layernorm.weight` — 与本层 self.weight 铁律一致
  - 编码铁律 §7.3 "fused_add_rms_norm 全部使用本层 self.weight" — **已满足**（wrapper 本身不耦合任何特定 weight，由调用方传入；docstring 示例指导正确用法）

#### KERNEL 3: `silu_and_mul`

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.silu_and_mul`
- **Status**: ✅ @ `engine/kernels/vllm_wrappers.py:111-130`
- **核验内容**:
  - Signature `(out, input) -> None` 与蓝图 `torch.ops._C.silu_and_mul(out!, input)` **匹配**
  - 文件头部已执行 `import vllm._C`（触发 `torch.ops._C` 注册）— 蓝图前置要求已满足
  - 调用 `torch.ops._C.silu_and_mul(out, input)`，与蓝图一致
  - Docstring 正确描述 `input: [*, 2*d]`（前半 gate 后半 up），`out: [*, d]`（预分配）

#### KERNEL 4: `rotary_embedding`

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.rotary_embedding`
- **Status**: ✅ @ `engine/kernels/vllm_wrappers.py:137-165`
- **核验内容**:
  - Signature `(positions, query, key, head_size, cos_sin_cache, is_neox) -> None` 与蓝图 `def rotary_embedding(positions: Tensor[N] int64, query!: Tensor[N,H,D], key!: Tensor[N,Kv,D]|None, head_size: int, cos_sin_cache: Tensor[M,D], is_neox: bool) -> None` **精确匹配**
  - 调用 `_vllm_rotary_embedding`，与蓝图 `vllm/_custom_ops.py:400-410` 一致
  - Docstring 正确约束：`positions` 为 `int64 (torch.long)` 1D，`q/k` 为 3D `[tokens, heads, dim]` 非 4D
  - `key` 可为 `None`（类型注解 `torch.Tensor | None`）
  - `is_neox` 无默认值，Docstring 说明 Qwen3 严格 True
  - `cos_sin_cache` 格式约束：`[max_pos, head_size]` 非 `[2*head_size]`

#### KERNEL 5: `cos_sin_cache` 工厂 + 模块级 registry

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.rotary_embedding.cos_sin_cache_strategy`
- **Status**: ✅ @ `engine/kernels/vllm_wrappers.py:172-233`
- **核验内容**:
  - **Registry**: `_cos_sin_cache_registry: dict[tuple, torch.Tensor] = {}` — 模块级，与蓝图 `dict[tuple, Tensor]` **一致**
  - **Key 结构**: `(max_pos, head_dim, rope_theta)` — **精确匹配**蓝图 `key=(max_pos, head_dim, rope_theta)`
  - **工厂函数**: `_get_cos_sin_cache(max_pos, head_dim, rope_theta)` — 签名与蓝图一致
  - **Lazy GPU 策略**: `make_cos_sin_cache` 默认 `device=None`（CPU 创建），调用方负责 `.to(device)` — 与蓝图 "CPU 创建在 `__init__`，首次 forward 时 lazy GPU transfer" **一致**
  - **Cache shape**: `torch.cat((cos, sin), dim=-1)` → `[max_position, head_size]`（**非** `[2*head_size]`）— 与蓝图 `eager_gate.cos_sin_shape` "cache shape [max_pos, head_size] 非 [2*head_size]" **一致**
  - **内部格式**: 前 `head_size//2` 为 cos，后 `head_size//2` 为 sin — 与蓝图 "vLLM kernel 内部自行处理 NeoX 风格的 cos/sin 重复" **一致**
  - **数值公式**: `inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_size, 2) / head_size))` — 标准 RoPE，与 vLLM `RotaryEmbeddingBase._compute_cos_sin_cache` 一致

#### KERNEL 6 & 7: `flash_attn_varlen_func` / `flash_attn_with_kvcache`

- **JSON Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts` (source_impl 含 engine/kernels/)
- **Status**: ✅ @ `engine/kernels/vllm_wrappers.py:27-28`
- **核验内容**:
  - `flash_attn_varlen_func` 直接从 `flash_attn` 包 re-export
  - `flash_attn_with_kvcache` 直接从 `flash_attn.flash_attn_interface` re-export
  - `__init__.py:17-18` 明确注释：nocompile 场景直接 import，无需 custom_op 注册（custom_op+register_fake 仅在 torch.compile tracing 时需要）— 与蓝图 `flash_attention_integration_contract.decode_path.custom_op_registration` 的 Phase 分工一致（custom_op 注册属于 Phase 5 及 torch.compile 路径，Phase 1 仅提供原始 import）

### Block B: `rmsnorm_precision_law` — 全局精度约束

- **JSON Path**: `model_layer.architecture_knowledge_base.global_primitives_constraints.rmsnorm_precision_law`
- **Status**: ✅ @ `engine/kernels/vllm_wrappers.py:19-20`
- **核验内容**:
  - RMSNorm **使用 vLLM 标品 CUDA kernel**（`from vllm._custom_ops import rms_norm` / `fused_add_rms_norm`），**非**手写 PyTorch RMSNorm
  - 蓝图 directive: "kernel 内部 fp32 计算，调用方只需确保 out 预分配、input contiguous" — wrapper docstring 已完整记录这些约束
  - 蓝图 `_nano_vllm_override`: "nano-vllm 原始 RMSNorm 实现需整体替换为 vLLM kernel wrapper" — 已满足，本包无 nano-vllm 手写 PyTorch RMSNorm 残留

### Block C: `eager_gate` — Phase 1 门禁

- **JSON Path**: `framework_layer.todo_generation_playbook.phase_1_numeric_primitives.eager_gate`
- **Status**: ✅
- **核验内容**:
  - **cpu_dispatch**: 所有 kernel wrapper 不含 `.to()` 调用、不含 Python `for` 循环 — ✅（`make_cos_sin_cache` 的 `torch.arange` + `torch.einsum` 是向量化操作，非 Python 循环；`_get_cos_sin_cache` 的 `if key not in` 是 registry 查找，不违反此约束）
  - **cos_sin_shape**: `[max_pos, head_size]` 非 `[2*head_size]` — ✅（已在上文 KERNEL 5 中核验）
  - **rope_style**: Neox 正确 — ✅（`is_neox` 参数透传，wrapper 不加默认值，由调用方传入模型实际风格）

### Block D: 导出接口完整性

- **JSON Path**: （隐式契约 — `engine/kernels/__init__.py` 应导出全部 7 个 kernel wrapper）
- **Status**: ✅ @ `engine/kernels/__init__.py:6-14`
- **核验内容**:
  - `__init__.py` 显式导入 8 个符号：`rms_norm`, `fused_add_rms_norm`, `silu_and_mul`, `rotary_embedding`, `_get_cos_sin_cache`, `make_cos_sin_cache`, `flash_attn_varlen_func`, `flash_attn_with_kvcache`
  - 与 `vllm_wrappers.py` `__all__` 完全对齐
  - 无遗漏，无多余

---

## Blueprint Information Gaps (🟡 标记)

- **🟡 `qwen3_kernel_contracts.rotary_embedding.constraint`**: 描述 "输入为 **2D** [num_tokens, heads, head_dim] (非4D)"，但同节点的 `inline_signature` 使用 `Tensor[N,H,D]`（3 维：tokens × heads × dim）。代码 docstring 采用 "3D [tokens, heads, dim]"（与 inline_signature 一致）。这是蓝图自身的维度描述不一致，不影响实现正确性。建议将 constraint 文本修正为 "3D [num_tokens, heads, head_dim] (非4D)" 以消除歧义。

---

## Issues Found

**无。**

所有 Phase 1 契约节点均已逐条核验，代码与蓝图一致。未发现违规、遗漏或多余实现。

---

## Verdict

**Spec 审查通过。代码与蓝图契约一致，可移交 verification。**

审查范围：
- `engine/__init__.py` — 包标识
- `engine/kernels/__init__.py` — 8 符号导出
- `engine/kernels/vllm_wrappers.py` — 7 个 kernel wrapper 完整实现

未审查范围（不在此 Phase 1 scope 内）：
- `qwen3_kernel_contracts.custom_ar_all_reduce` — 属于 Phase 2（TP 通信）
- `qwen3_kernel_contracts.sdpa_enable_gqa` — 属于 Phase 5（Attention）
- `qwen3_kernel_contracts.qkv_merged_projection` — 属于 Phase 3（TP 线性层）
- `rmsnorm_precision_law` 的调用方代码（RMSNorm 类的 forward()）— 属于后续 Phase
