# Kernel Wrappers — API 契约

> 蓝图来源: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts`
> 关联 notebooks: `07_improvementPlan/kernel_replacement_plan.md` §九, `07_improvementPlan/qwen3_effective_changes.md`

## 概述

7 个 vLLM 黑盒 kernel wrapper — 全部从 vLLM installed package 源码提取纯净包装，禁止修改内部逻辑。
对应 kernel_replacement_plan.md Stage 1-7。

源实现文件: `engine/kernels/vllm_wrappers.py`, `engine/models/qwen.py`, `engine/tp_layers/linear.py`

---

## 接口签名

### 1. rms_norm

- **签名**: `def rms_norm(out: Tensor[*, H], input: Tensor[*, H], weight: Tensor[H], epsilon: float) -> None`
- **来源**: `vllm/_custom_ops.py:420-423`
- **约束**: out 预分配 (`empty_like`); input 必须 contiguous; out/input/weight 同 dtype (bf16)
- **用法**: `RMSNorm.forward(x) → rms_norm(out, x.contiguous(), weight, eps)`
- **精度**: kernel 内部 fp32 计算，调用方只需确保 out 预分配、input contiguous

### 2. fused_add_rms_norm

- **签名**: `def fused_add_rms_norm(input!: Tensor[*, H], residual!: Tensor[*, H], weight: Tensor[H], epsilon: float) -> None`
- **约束**: 双 in-place: `residual += input`; `input = rms_norm(residual)`
- **权重规则**: 所有 4 次调用均使用本层的 `self.input_layernorm.weight` 或 `self.post_attention_layernorm.weight`。不存在跨层 weight 依赖
- **residual chain pseudocode**:
```python
hs, res = hidden_states, residual
if res is None: res = hs.clone(); rms_norm(hs, res, self.input_layernorm.weight, eps)  # layer 0 only
else: fused_add_rms_norm(hs, res, self.input_layernorm.weight, eps)  # res+=hs; hs=rms_norm(res)
# ... attention (prefill or decode) ...
fused_add_rms_norm(attn_out, res, self.post_attention_layernorm.weight, eps)
# ... mlp ... ; return mlp_out, res
```

### 3. silu_and_mul

- **签名**: `torch.ops._C.silu_and_mul(out!: Tensor[*, d], input: Tensor[*, 2*d]) -> None`
- **来源**: `vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda`
- **约束**: out 预分配 `[B, S, intermediate/tp]`; input 为 MergedColumnParallelLinear 输出 `[B, S, 2*intermediate/tp]` (前 gate 后 up)
- **用法**:
```python
gate_up = self.gate_up_proj(x)
out = torch.empty(B, S, intermediate//tp, dtype=x.dtype, device=x.device)
silu_and_mul(out, gate_up)
return down_proj(out)
```
- **注意**: 需 `import vllm._C` 触发注册

### 4. rotary_embedding

- **签名**: `def rotary_embedding(positions: Tensor[N] int64, query!: Tensor[N, H, D], key!: Tensor[N, Kv, D] | None, head_size: int, cos_sin_cache: Tensor[M, D], is_neox: bool) -> None`
- **来源**: `vllm/_custom_ops.py:400-410`
- **约束**:
  - q/k in-place 修改
  - 输入为 2D `[num_tokens, heads, head_dim]` (非 4D)
  - cos_sin_cache 格式 `[max_pos, head_size]` (前 head_size//2 cos，后 head_size//2 sin)
  - Qwen3: `is_neox=True` (前后分半 rotate_half)
- **CosSinCache 策略**:
  - 模块级 registry: `_cos_sin_cache_registry: dict[tuple, Tensor] = {}`
  - factory: `_get_cos_sin_cache(max_pos, head_dim, rope_theta)` — key=`(max_pos, head_dim, rope_theta)`
  - lazy GPU transfer: `self._cos_sin_cache_cpu` 在 `__init__` 创建; `self._cos_sin_cache_gpu = None`; 首次 forward 时 `.to(device)`

### 5. flash_attn_varlen_func (prefill)

- **签名**: `flash_attn_varlen_func(q: Tensor[total_tokens, heads, dim], k: Tensor[total_tokens, kv_heads, dim], v: Tensor[total_tokens, kv_heads, dim], cu_seqlens_q: Tensor[int32], cu_seqlens_k: Tensor[int32], max_seqlen_q: int, max_seqlen_k: int, causal: bool = True) -> Tensor[total_tokens, heads, dim]`
- **约束**: B=1 时 cu_seqlens 由 QwenAttentionTP 内部自动构造 `[0, num_tokens]`; K/V 来自当前投影产出 (非从 cache 读取)
- **用法 (B=1)**:
```python
cu = torch.tensor([0, num_tokens], dtype=torch.int32, device=q.device)
out = flash_attn_varlen_func(q_flat, k_flat, v_flat, cu, cu, num_tokens, num_tokens, causal=True)
```

### 6. flash_attn_with_kvcache (decode)

- **签名**: `flash_attn_with_kvcache(q: Tensor[1, 1, heads, dim], k_cache: Tensor[num_blocks, block_size, kv_heads, dim], v_cache: Tensor[num_blocks, block_size, kv_heads, dim], kv_len: Tensor[int32], block_table: Tensor[int32], softmax_scale: float, causal: bool = False) -> Tensor[1, 1, heads, dim]`
- **约束**: nocompile 模式直接 `from flash_attn.flash_attn_interface import flash_attn_with_kvcache` (无需 custom_op 注册)
- **kv_len**: GPU tensor `[1]` int32，不可传 Python int
- **block_table**: `[1, max_blocks]` int32

### 7. index_copy_ (KV cache write)

- **签名**: `kc_flat.index_copy_(dim=0, index=slot_mapping: Tensor[int64], source=k_flat: Tensor[heads, dim])`
- **用法**: decode 路径写入 1 token
```python
self._slot_mapping_decode[0] = self._kv_len_gpu[0]
k_write = k.reshape(1, self.num_kv_heads, self.head_dim)
kc_flat = self._key_cache.view(-1, self.num_kv_heads, self.head_dim)
kc_flat.index_copy_(0, self._slot_mapping_decode, k_write)
```

---

## 精度约束

| Kernel | 精度要求 |
|--------|---------|
| rms_norm | kernel 内部 fp32 计算 |
| fused_add_rms_norm | kernel 内部 fp32 计算 |
| silu_and_mul | all bf16 |
| rotary_embedding | cos/sin cache fp32 创建 → `.to(input_dtype)` |
| flash_attn_varlen_func | all bf16 |
| flash_attn_with_kvcache | all bf16 |

---

## 陷阱与反模式

- **FM-002**: RMSNorm contiguous 约束 — 传入 4D QK norm tensor (view 结果) 时需 `.contiguous()`
- **FM-003**: fused_add_rms_norm 跨层 weight — 所有调用用本层 weight，勿跨层
- **FM-004**: CosSinCache 格式 — shape `[max_pos, head_size]` 非 `[2*head_size]`
- **FM-007**: RoPE Neox vs GPT-J — Qwen3 必须 `is_neox=True`
- **FM-008**: paged KV block_size — 必须 ≥256 (flash_attn 硬性要求)
- **KERNEL-001**: 禁止手写 PyTorch RMSNorm — 必须使用 vLLM CUDA kernel
