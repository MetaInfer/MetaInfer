# Qwen3-8B 对标 vllm 检查清单

> 对比基准: vllm 0.15.1 Qwen3-0.6B torch.profiler trace  
> 目标: Qwen3-8B 吞吐率接近 vllm

## 0. 已实现 ✅

| 组件 | 状态 | 位置 |
|------|------|------|
| vllm rotary_embedding kernel | ✅ | `qwen.py:233-240` |
| vllm silu_and_mul (act_and_mul) | ✅ | `qwen.py:316-317` |
| Merged gate_up_proj Linear | ✅ | `qwen.py:309-313` |
| fused_qkv_proj (单次 GEMM) | ✅ | `qwen.py:222` |
| paged KV cache | ✅ | `qwen.py:192-287` |
| block_table + slot_mapping | ✅ | `qwen.py:254,257` |
| custom all-reduce | ✅ | `qwen.py:569` |
| CUDA Graph (manual capture) | ✅ | `qwen.py:365-405, 598-609` |
| torch.compile (optional) | ✅ | `qwen.py:570-592` (META_INFER_COMPILE=1) |
| rms_norm (vllm fused) | ✅ | `qwen.py:97-110` |

## 1. 需确认/排查项 ⚠️

### 1.1 CUDA Graph 是否在 benchmark 中生效

当前 graph capture 发生在 init 时（`_capture_decode_graph`），需确认：
- benchmark 脚本是否设置了 `META_INFER_CUDA_GRAPH=1`（默认是 1）
- capture 的 graph 参数是否与实际 decode 匹配（kv_len=4 适合短 prompt，长 prompt 可能需要 re-capture）

### 1.2 torch.compile 是否启用

当前需要 `META_INFER_COMPILE=1` 才启用。需确认 benchmark 是否设置。

### 1.3 RMS Norm 是否使用 vllm 融合 kernel

检查 `qwen.py:97` 的 `RMSNorm` 实现：
```python
# 如果用的是 PyTorch native, 替换为 vllm 的 rms_norm_kernel
```

### 1.4 KV cache write 是否使用融合 kernel

当前 decode KV write (`qwen.py:281-287`)：
```python
self._slot_mapping_decode[0] = self._kv_len_gpu[0]
# 用的是 Python view indexing 写入, 不是 vllm 的 reshape_and_cache_flash
reshape_and_cache(key_flat, value_flat, self._slot_mapping_decode,
                 self._key_cache, self._value_cache)
```
如果 `reshape_and_cache` 是 vllm 的 custom op → ✅，否则 → ❌

### 1.5 DecoderLayer residual add + RMS norm  融合

当前 `QwenDecoderLayerTP.forward()`:
```python
hidden_states = self.input_layernorm(x)       # RMS norm
attn_out, kv = self.self_attn(hidden_states, ...)
x = x + attn_out                              # residual add (独立 kernel)
hidden_states = self.post_attention_layernorm(x)  # RMS norm (独立 kernel)
```

vllm 用 `fused_add_rms_norm_kernel` 把 add + rms_norm 合并为一个 kernel。当前缺失。

## 2. 需添加的改动 📋

### 2.1 Fused Residual Add + RMS Norm

**改法 1**: 从 vllm 包装 fused_add_rms_norm
```python
# vllm_wrappers.py 添加:
from vllm import _custom_ops as ops
def fused_add_rms_norm(x, residual, weight, eps):
    """x = residual + x; x = rms_norm(x, weight, eps) — single kernel"""
    ops.fused_add_rms_norm(x, residual, weight, eps)
    return x
```

**改法 2**: 替换 DecoderLayer 的 forward
```python
# 改前:
hidden_states = self.input_layernorm(x)
...
x = x + attn_out
hidden_states = self.post_attention_layernorm(x)

# 改后:
attn_hidden = self.input_layernorm(x)
...
hidden_states = fused_add_rms_norm(attn_out, x, self.post_attention_layernorm.weight, eps)
```

### 2.2 确认 rms_norm 是 vllm fused kernel

检查 `qwen.py:97-110`:
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.rms_norm_bf16(x, self.weight, self.variance_epsilon)
        # 如果已是这样 → ✅ 否则需要改
```

### 2.3 确认 benchmark 环境变量

查看 `run_compare_metainfer_vllm.sh` 是否设置了：
```bash
META_INFER_CUDA_GRAPH=1  # 默认应已生效
META_INFER_COMPILE=1     # 可能需要显式设置
```

## 3. 预期收益

| 改动 | 预期提升 |
|------|---------|
| fused_add_rms_norm | +5-8% |
| 确认/修复 CUDA Graph | 如果未生效则 +30-40% |
| 确认/修复 torch.compile | 如果未生效则 +20-30% |
| reshape_and_cache_flash | +3-5% |

**如果 CUDA Graph 和 torch.compile 都未生效**，修复后预期从 ~30 tok/s → 50+ tok/s。
