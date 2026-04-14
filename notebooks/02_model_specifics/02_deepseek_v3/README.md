# DeepSeek V3 - Architecture Overview

## What Makes DeepSeek V3 Different

DeepSeek V3 is a production-grade LLM that introduces several architectural innovations over standard transformers. These innovations significantly impact the inference framework design and must be understood to generate correct inference code.

## Key Innovations

| Innovation | Standard Approach | DeepSeek V3 Approach | Impact on Inference |
|-----------|-------------------|---------------------|-------------------|
| Attention | Multi-Head Attention (MHA) or GQA | Multi-Latent Attention (MLA) | Smaller KV cache, different attention kernel |
| MLP | Dense FFN or simple MoE | Fine-grained MoE with shared experts | Expert routing, all-to-all communication |
| Prediction | Single next-token | Multi-Token Prediction (MTP) | Built-in speculative decoding |
| Long context | Full attention | Native Sparse Attention (NSA) | Sparse token selection, sub-linear attention |

## Architecture Structure

```
Input Tokens
    ↓
[Embedding]
    ↓
[DecoderLayer × N]
    │
    ├── [RMSNorm]
    ├── [MLA Attention]  ← Instead of standard MHA/GQA
    ├── [Residual]
    ├── [RMSNorm]
    ├── [MoE MLP]        ← Instead of dense MLP
    │   ├── [Shared Expert(s)]  ← Always active
    │   └── [Routed Experts × K of N]  ← Top-K selection
    └── [Residual]
    ↓
[RMSNorm]
    ↓
[LM Head]
    ↓
[MTP Heads × M]  ← Predict future tokens (optional, for speculative decoding)
```

## Sub-Documents

- [02_mla_attention.md](02_mla_attention.md) - Multi-Latent Attention mechanism
- [03_moe.md](03_moe.md) - Fine-grained Mixture of Experts with shared experts
- [04_mtp.md](04_mtp.md) - Multi-Token Prediction for speculative decoding
- [05_nsa.md](05_nsa.md) - Native Sparse Attention for long contexts
- [06_optimization_patterns.md](06_optimization_patterns.md) - FP8 optimizations and kernel patterns
