---
name: Qwen3 TP debug - attention precision and RoPE bf16 alignment
description: Root cause analysis and fixes for Qwen3 TP=4 garbled output. Key fixes: separate projections, CPU-first creation, bf16 RoPE to match HF precision
type: feedback
---

## Summary

Qwen3 TP=4 produced garbled output ("ござい。  2020年，  2020年，") due to a cascade of numerical precision issues. The root cause is NOT a single bug but a chain of bf16 numerical imprecision that compounds through 36 layers.

## Root Causes (fixed)

### 1. RoPE cos/sin precision mismatch (most impactful)
- **What**: Our cos/sin cache stores values in fp32, HF uses bf16
- **Effect**: At small angles, cos(θ) ≈ 0.998 in fp32 but rounds to 1.0 in bf16. This tiny difference at position 1 (cos_diff=0.001825 at dim 13/64) cascades through 36 layers
- **Fix**: Cast cos/sin to input dtype (`x.dtype`, which is bf16) before rotation in `apply_rotary_emb_half_half`. Previously line was `cos = cos.to(x_fp.dtype)` (fp32→fp32), now `cos = cos.to(x.dtype)` (fp32→bf16→fp32 for computation)
- **File**: `engine/tp_layers/rotary.py` line 19-21

### 2. Fused QKV projection bf16 matmul non-determinism
- **What**: `QKVParallelLinear` uses fused matmul `x @ [W_q, W_k, W_v].T`. CUDA dispatches a different kernel for the larger fused matmul vs three separate matmuls, giving slightly different bf16 results
- **Effect**: Q/K values differ by ~0.003906 even with identical weights and inputs, amplified by RMSNorm
- **Fix**: Replace fused `QKVParallelLinear` with three separate `ColumnParallelLinear` (q_proj, k_proj, v_proj)
- **Blueprint confirmed**: `qwen_dense_loader.split_dim_0` lists q_proj/k_proj/v_proj separately

### 3. Fused MLP gate/up projection bf16 non-determinism
- **Same issue** as QKV: fused `MergedColumnParallelLinear` for gate+up gives different bf16 results than separate gate_proj/up_proj
- **Fix**: Separate gate_proj and up_proj as individual `ColumnParallelLinear`

### 4. Model creation on CUDA causes non-deterministic weight loading
- **What**: Creating model with `torch.set_default_device("cuda:N")` means weights are created on CUDA with uninitialized memory, then loaded from CPU via `.copy_()`. The CUDA→CPU transfer can introduce garbage from uninitialized CUDA memory
- **Fix**: Remove `torch.set_default_device()` during model creation. Create on CPU, load weights on CPU, then `.cuda()` after loading

### 5. Attention softmax must use bf16 matmul to match HF
- **What**: Our manual attention computed `q @ k.T` in fp32, but HF's `eager_attention_forward` uses bf16
- **Fix**: Compute `torch.matmul(q_t, k_t.transpose(-2,-1)) * self.scale` in bf16, only cast to fp32 for softmax

## Remaining issues
After all fixes, Layer 0 still shows 4.4 diff from HF. Individual components (Q, K, V, Q_norm, K_norm, RoPE) match with <0.001 diff. The accumulation of these tiny bf16 rounding errors through attention + o_proj + MLP causes the divergence. The output changed from pure garbage "ござい" to partially meaningful "ござい？の答えは" but still not correct Chinese.

**Likely cause of remaining error**: Each bf16 operation rounds with ~0.001-0.01 error. Through 6 operations per layer x 36 layers, the error compounds. This is a fundamental limitation of bf16 not being bit-deterministic across different implementations.

## Verification Guide
```bash
# Compare individual components with HF
python -c "
from engine.models.qwen import QwenTPModelRunner
r = QwenTPModelRunner(model_dir='/path/to/model', tp_size=1)
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained('/path/to/model', ...)
# Compare weights, Q/K/V outputs, norms, RoPE, attention, MLP
"
```

## Related Files
- `engine/tp_layers/rotary.py` - bf16 RoPE fix
- `engine/tp_layers/attention.py` - bf16 matmul + fp32 softmax
- `engine/models/qwen.py` - separate QKV/MLP, CPU-first creation
- `inference_blueprint.json` → `model_layer.qwen_series_dense` - confirms separate projections
