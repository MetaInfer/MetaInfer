# Task 13: TP=2 Inference Correctness Fix — RMSNorm fp32 Weight Multiplication

## Date
2026-05-07

## Problem Phenomenon
DeepSeek-V2-Lite TP=2 inference output garbled, first token is 12350 (HF ground truth is 185).

## Root Cause Analysis

### Three-Level Cascade Failure Chain

1. **RowParallelLinear all_reduce** (Layer 0): Due to TP sharding前后 matmul using different CUDA BLAS kernels (split vs full matrix),
   bf16 results have ~0.008 element-level difference.

2. **RMSNorm amplifies error**: Original implementation `x_fp32.to(x.dtype).mul_(self.weight)` first converts normalized result to bf16,
   then multiplies by weight. bf16 multiplication's small perturbation changes RMS calculation's denominator, causing error amplification ~3x (0.008 → 0.023).

3. **MoE Gate routing forks** (Layer 2): The amplified perturbation (~0.023) is sufficient to change top-6 expert selection,
   error jumps to 0.25. Then cascades through remaining 25 MoE layers, final output completely deviates.

### Propagation Path
```
Layer 0: RowParallelLinear all_reduce → bf16 error ~0.008
Layer 1: post_norm RMSNorm amplifies → ~0.023 (3x)
Layer 1: MoE output still correct (routing identical)
Layer 2: post_norm RMSNorm amplifies → ~0.023
Layer 2: MoE gate routing CHANGES → ~0.25 (***)
Layer 3: post_norm amplifies → ~0.03
Layer 3: MoE completely wrong experts → ~4.0 (***)
Layers 4-26: error saturated at ~4.0, mean slowly grows
```

## Fix

### Core Fix: RMSNorm Uses fp32 Weight Multiplication

**Wrong implementation** (`agent-infer3` original code):
```python
# engine/tp_layers/linear.py (old)
out = x_fp32.to(x.dtype).mul_(self.weight)  # bf16 multiplication amplifies error!
```

**Correct implementation** (reference `meta-infer`):
```python
# engine/tp_layers/linear.py (new)
return (x * self.weight.float()).to(dtype)  # fp32 multiplication, then cast to bf16
```

Key difference: weight multiplication in fp32 precision, result then cast to bf16. bf16 multiplication (`a_bf16 * w_bf16`) rounding behavior is sensitive to small input perturbations,
while fp32 multiplication (`a_fp32 * w_fp32`) has sufficient precision margin to absorb perturbations.

### Fix 2: PagedAttention GQA Head Repeat Layout Error

**Wrong implementation** (`engine/tp_layers/attention.py`):
```python
# expand+reshape produces [kv0, kv1, kv0, kv1] layout, each Q head attends to wrong KV head
k = k.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(-1, self.num_heads, self.head_dim)
v = v.unsqueeze(1).expand(-1, n_rep, -1, -1).reshape(-1, self.num_heads, self.head_dim)
```

**Correct implementation**:
```python
# repeat_interleave produces [kv0, kv0, kv1, kv1] layout, each Q head attends to correct KV head
k = k.repeat_interleave(n_rep, dim=1)
v = v.repeat_interleave(n_rep, dim=1)
```

`expand` copies entire tensor along new dimension, `reshape` merges dimensions with interleaved排列.
In Qwen3-8B (num_heads=32, num_kv_heads=8), Q heads 0-3 should attend to KV head 0,
but actually attend to mix of KV heads 0,1,2,3. DeepSeek-V2 unaffected (num_heads = num_kv_heads).

### Companion Fixes

1. **Distributed initialization**: Replaced `init_tp_process_group` + `_TP_GROUP` approach with `init_tp_distributed()`,
   directly using WORLD process group, avoiding `_TP_GROUP` being None edge case.

2. **Weight loading**: Switched from `weight_loader` on `nn.Parameter` approach to `load_weight_shard` on module approach,
   using `safe_open` + `get_slice` lazy loading, avoiding loading full weights to CPU memory.

3. **MoE 2D/3D compatibility**: `ExpertParallelMoE` expects 3D input `[B, T, H]`, but decoder layer passes 2D `[N, H]`.
   Added dimension handling in `DeepseekV2MoE.forward()`.

4. **Model building on CPU**: Avoid creating model parameters directly on GPU (8B+ params cause OOM),
   changed to CPU create model → load weights → move to GPU.

## Files Involved

| File | Operation | Description |
|------|-----------|-------------|
| `engine/tp_layers/distributed.py` | Replace | Use meta-infer version (simplified process group management) |
| `engine/tp_layers/linear.py` | Replace | Use meta-infer version (RMSNorm fp32 multiplication + load_weight_shard) |
| `engine/tp_layers/embedding.py` | Replace | Use meta-infer version (load_weight_shard) |
| `engine/tp_layers/moe.py` | New | ExpertParallelMoE + _ExpertMLP |
| `engine/tp_layers/attention.py` | Fix | GQA repeat_interleave replaces expand+reshape |
| `engine/tp_layers/__init__.py` | Replace | Correct export list |
| `engine/models/deepseek_v2.py` | Rewrite | Use new tp_layers + safetensors direct loading + paged KV cache |
| `engine/models/qwen.py` | Rewrite | Use new tp_layers + safetensors direct loading + paged KV cache |
| `engine/models/__init__.py` | New | Export QwenTPModelRunner, DeepseekTPModelRunner |

## Test Verification

### DeepSeek-V2-Lite TP=2
```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29506 \
  -m pytest tests/test_deepseek_tp_real.py -v -s
```

**Result**: 2 passed
- Weight loading: ✓ (embed: [51200, 2048], q_proj: [1536, 2048])
- Prefill + Decode: first token = **185** (matches HF ground truth!)
- Generated output: `人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它致力于创建能够`
- Logits stats: mean=-9.65 std=5.88

### Qwen3-8B TP=2
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29510 \
  -m pytest tests/test_qwen_tp_real.py -v -s
```

**Result**: 2 passed
- Weight loading: ✓ (embed: [75968, 4096], q_proj: [2048, 4096])
- Prefill first token = **220** (matches meta-infer ground truth!)
- Generated output: `人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在开发能够执行通常`
- Logits stats: mean=-3.98 std=3.70

### GPU Monitoring Evidence
- Hardware: 8x NVIDIA A800 80GB
- DeepSeek-V2-Lite TP=2 per-card memory: ~16 GiB (params) + ~4 GiB (KV cache)
- Qwen3-8B TP=2 per-card memory: ~8 GiB (params) + ~4 GiB (KV cache)

## Key Lessons

1. **RMSNorm numerical sensitivity**: Weight multiplication must be in fp32. bf16 multiplication is sensitive to input perturbations,
   normalization denominator (RMS) small changes amplified through weight multiplication, causing downstream MoE gate routing errors.

2. **bf16 matmul non-determinism**: Different CUDA BLAS kernels produce different bf16 results for same mathematical operation.
   TP sharding changes matrix shape, triggering different kernel selection. This is底层 hardware behavior, unavoidable,
   can only be absorbed through upstream numerical precision (fp32).

3. **MoE Gate fragility**: softmax + topk is extremely sensitive to input perturbations. 0.023 difference is sufficient to change
   top-6 expert selection. After cascading through 26 MoE layers, output completely deviates.

4. **GQA head repeat layout**: `expand` + `reshape` vs `repeat_interleave` produce different KV head排列.
   Former interleaves `[kv0, kv1, kv0, kv1]`, causing Q head to attend to wrong KV head, resulting in GQA models (Qwen3) output garbled.
   Latter correctly排列 `[kv0, kv0, kv1, kv1]`. DeepSeek-V2 unaffected (num_heads == num_kv_heads).

## References
- meta-infer correct implementation: `/home/honglin/meta-infer/engine/tp_layers/`
- HF DeepSeek-V2 implementation: `transformers.models.deepseek_v2`
- Previous conversation transcript: `/home/honglin/.claude/projects/-home-honglin/672f90dc-28fd-43ae-859d-46cf6f5dff2b.jsonl`
