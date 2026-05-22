# P3-FA Flash Attention Profiling 分析报告

## 1. 测试环境

- **模型**: DeepSeek-V2-Lite-Chat (MLA, QK headdim=192, V headdim=128)
- **GPU**: 4x A800 80GB, TP=4
- **Conda 环境**: meta (flash_attn 2.8.3, PyTorch 2.9.1+cu128)
- **Profiling**: PyTorch Profiler (CPU + CUDA activities, record_shapes=True)

## 2. P2 基线使用的 Kernel

| Kernel 名称 | 所属函数 | 调用次数 | CPU 耗时 | 说明 |
|---|---|---|---|---|
| `aten::_scaled_dot_product_efficient_attention` | PyTorch SDPA 入口 | 648 | 49.87ms | CPU 端调度 |
| `aten::_efficient_attention_forward` | SDPA 内核调度 | 648 | 31.65ms | 调用 CUTLASS |
| `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80` | CUTLASS FlashAttention | 648 | 6.52ms | **实际 GPU 计算 kernel** |
| `triton_poi_fused__scaled_dot_product_efficient_attention_constant_pad_nd_...` | attn_mask 构造 | 1242 | 18.75ms | Triton 融合 kernel |

## 3. P3-FA 使用的 Kernel

| Kernel 名称 | 所属函数 | 调用次数 | CPU 耗时 | 说明 |
|---|---|---|---|---|
| `flash_attn::_flash_attn_varlen_forward` | `flash_attn_interface._flash_attn_varlen_forward` | 648 | 55.65ms | CPU 端入口总耗时 |
| `flash_attn/flash_attn_interface.py(142): _flash_attn_varlen_forward` | 同上 Python 层 | 648 | 39.60ms | **Python wrapper 开销** |
| `flash_attn/flash_attn_interface.py(164): <listcomp>` | 参数列表构造 | 648 | 2.48ms | 开销 |
| `flash_attn/flash_attn_interface.py(19): maybe_contiguous` | 输入 contiguous 检查 | 1944 | 1.71ms | 每次 forward 3 次调用 |
| `void flash::flash_fwd_kernel<192, 128, 64, 8, ...true>` | FA2 forward kernel (prefill) | 27 | 0.22ms | headdim=192 |
| `void flash::flash_fwd_kernel<192, 128, 64, 8, ...false>` | **FA2 forward kernel (decode)** | 621 | 5.15ms | **实际 GPU 计算 kernel** |
| `triton_poi_fused_..._constant_pad_nd_..._7` | V-padding (prefill) | 27 | 0.44ms | `F.pad(v, [0, 64])` |
| `triton_poi_fused_..._constant_pad_nd_..._8` | V-padding (decode) | 621 | 0.79ms | 同上 |
| `triton_poi_fused_..._constant_pad_nd_..._view_9` | 输出 unpad | 1242 | 13.14ms | `out[:, :, :128]` |
| `triton_poi_fused_..._constant_pad_nd_..._view_10` | 输出 unpad | 1242 | 9.98ms | 同上 |

**在 Perfetto 中定位**：打开 trace_p3_fa.json，搜索 `flash_fwd_kernel` 或 `cutlassF` 可直接定位到 GPU kernel。

## 4. 性能对比

### 4.1 Attention 层总耗时

| 指标 | P2 | P3-FA | 差异 |
|---|---|---|---|
| Attention 总耗时 | 791.13ms | 906.59ms | **+14.6%** |

### 4.2 GPU Kernel 真实对比

| | GPU kernel 耗时 | CPU 端总耗时 |
|---|---|---|
| P2 CUTLASS (`fmha_cutlassF_bf16_aligned_32x128_gmem_sm80`) | 6.52ms | 49.87ms |
| P3-FA FA2 (`flash::flash_fwd_kernel<192, 128, 64, 8...>`) | **5.37ms** | 55.65ms |

**FA2 GPU kernel 比 CUTLASS 快 18%**（5.37ms vs 6.52ms）。

### 4.3 P3-FA 额外开销拆解

| 额外开销来源 | 耗时 | 占 P3-FA attention 总耗时 |
|---|---|---|
| Python wrapper (`flash_attn_interface.py`) | 39.60ms | 4.4% |
| V-padding/unpadding Triton kernel | 24.30ms | 2.7% |
| `maybe_contiguous` 检查 | 1.71ms | 0.2% |
| **合计额外开销** | **65.61ms** | **7.2%** |

## 5. 根因分析

### 5.1 P3-FA 为什么比 P2 慢 28%？

基准测试结果：P2 = 11.04 tok/s, P3-FA = 7.90 tok/s (**-28%**)

attention 层差异只有 +14.6%（+115ms），但总吞吐下降 28%，说明 **attention 以外也有开销增加**（如 torch.compile 重编译、tensor shape 变化等）。

### 5.2 Attention 层根因

1. **flash_attn 的 Python wrapper 开销 39.6ms**
   - `flash_attn` 包通过 Python 调用 CUDA kernel，每层每步都有 Python 函数调用、参数检查、tensor contiguous 检查
   - P2 的 SDPA 走 PyTorch C++ 内核路径（`aten::_efficient_attention_forward`），无 Python 层开销

2. **V-padding/unpadding 额外 24.3ms**
   - DeepSeek-V2 MLA 的 K headdim=192, V headdim=128，FA2 要求 K/V headdim 相同
   - 每步每层都要 `F.pad(v, [0, 64])`（pad 128→192）和 `out[:, :, :128]`（unpad）
   - P2 的 SDPA 原生支持不同 K/V headdim，无需 padding

3. **FA2 GPU kernel 本身比 CUTLASS 快 18%**
   - FA2: 5.37ms vs CUTLASS: 6.52ms
   - 但被 Python wrapper 和 V-padding 开销抵消

### 5.3 vLLM 的做法

vLLM 在 SM80 (A800) 上对 DeepSeek-V2 使用 **Triton MLA kernel**（`decode_attention_fwd`），而非 `flash_attn_varlen_func`：

- Triton MLA 在**压缩潜空间**（kv_lora_rank=512）中操作，避免 K/V headdim 不匹配
- KV cache 存储 `[kv_lora_rank + rope_dim]`（512+64=576），而非展开的 Q/K/V
- 输出 `[B, num_heads, kv_lora_rank]`，再通过 `_v_up_proj` 投影回 `v_head_dim`
- 无 V-padding 开销，无 Python wrapper 开销

vLLM 吞吐 26.70 tok/s，是 P2 的 2.4x，P3-FA 的 3.4x。

## 6. 修改计划

| 方案 | 策略 | 预期效果 | 复杂度 |
|---|---|---|---|
| **A（推荐）** | DeepSeek decode 回退 SDPA + 预分配 attn_mask | 消除 flash_attn Python wrapper + V-padding 开销 | 低 |
| B | 使用 PyTorch 内置 `F.scaled_dot_product_attention` 替代 `flash_attn` 包 | 消除 Python wrapper，保留 V-padding | 中 |
| C | 实现 Triton MLA kernel（参考 vLLM） | 从根本上解决 K/V headdim 不匹配 | 高 |

**方案 A 详细步骤**：
1. DeepSeek decode 路径：恢复 SDPA + attn_mask（P2 方式）
2. 预分配 attn_mask 缓冲区 `[1, 1, 1, max_seq_len]`，避免每步 `torch.zeros` + 填充
3. 保持 `torch.compile`（固定 shape，无重编译）
4. Qwen 保持 `flash_attn_varlen_func`（K/V headdim 相同，无 V-padding 开销）
