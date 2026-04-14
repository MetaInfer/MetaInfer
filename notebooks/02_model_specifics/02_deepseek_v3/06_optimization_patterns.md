# DeepSeek V3 - Optimization Patterns

## FP8 Quantization

DeepSeek V3 is designed with FP8 training in mind, and inference frameworks can exploit this for significant speedups.

### FP8 in Attention (MLA-specific)
```python
# The absorbed W_UK can be quantized to FP8 for faster GEMM
# Q_absorbed = Q @ W_UK^T  →  FP8 GEMM
q_absorbed = fp8_gemm(q_nope, w_uk_fp8, scale_q, scale_w)
```

### FP8 in MoE
Expert weights can be stored and computed in FP8:
```python
# Expert weights: [num_experts, out_size, in_size] in FP8
# Token activations: dynamically quantized to FP8 per-expert-block
fused_moe_fp8(hidden_fp16, expert_w1_fp8, expert_w2_fp8,
              scale_w1, scale_w2, topk_weights, topk_ids)
```

## DeepEP (Elastic Expert Parallelism)

For efficient all-to-all communication in EP mode:
```
Step 1: Local routing  → determine which tokens go to which GPU
Step 2: All-to-all     → send tokens to the GPU hosting the target expert
Step 3: Expert compute → each GPU processes its local experts
Step 4: All-to-all     → send results back to the originating GPU
Step 5: Combine        → weighted sum of expert outputs
```

DeepEP optimizes steps 2 and 4 with:
- Overlapped communication and computation
- Elastic batching to handle load imbalance
- RDMA support for cross-node communication

## Summary of Optimizations Relevant to Code Generation

When generating inference code for DeepSeek V3, the following optimizations should be considered:

1. **MLA absorbed mode**: Pre-compute absorbed projections to reduce per-token attention cost
2. **FP8 GEMM**: Use FP8 kernels for linear layers and expert computations
3. **Fused MoE kernel**: Essential for performance with 256 experts
4. **Grouped Top-K**: Ensure expert diversity across groups
5. **KV cache compression**: Store only compressed latent + RoPE key
6. **MTP speculation**: Enable built-in speculative decoding when latency matters
