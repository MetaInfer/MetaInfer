# Knowledge Base Index

## 00 Overview

- [README](00_overview/README.md) — Project overview, knowledge structure, source projects

## 01 Framework Design (Core Methodology)

- [Architecture](01_framework_design/01_architecture.md) — System topology, process models, component relationships
- [Scheduler](01_framework_design/02_scheduler.md) — Continuous batching, prefill/decode scheduling, preemption
- [KV Cache](01_framework_design/03_kv_cache.md) — Paged attention, radix/prefix caching, memory blocks
- [Model Runner](01_framework_design/04_model_runner.md) — Forward pass execution, CUDA graphs, weight loading
- [Sampler](01_framework_design/05_sampler.md) — Token sampling strategies (greedy, top-k, top-p)
- [Memory Pool](01_framework_design/06_memory_pool.md) — GPU memory allocation, pool management
- [Request Lifecycle](01_framework_design/07_request_lifecycle.md) — End-to-end flow from HTTP to response

## 02 Model Specifics

- [Transformer Models](02_model_specifics/01_transformer_models.md) — Dense transformer patterns (Llama, Qwen, Mistral)
- [DeepSeek V3 Overview](02_model_specifics/02_deepseek_v3/README.md) — Architecture innovations summary
- [MLA Attention](02_model_specifics/02_deepseek_v3/02_mla_attention.md) — Multi-Latent Attention mechanism
- [MoE](02_model_specifics/02_deepseek_v3/03_moe.md) — Fine-grained MoE with shared experts
- [MTP](02_model_specifics/02_deepseek_v3/04_mtp.md) — Multi-Token Prediction / speculative decoding
- [NSA](02_model_specifics/02_deepseek_v3/05_nsa.md) — Native Sparse Attention for long contexts
- [Optimizations](02_model_specifics/02_deepseek_v3/06_optimization_patterns.md) — FP8, DeepEP patterns
- [Qwen3 Overview](02_model_specifics/03_qwen3/README.md) — Qwen3 Dense/MoE architecture overview
- [Qwen3 Dense](02_model_specifics/03_qwen3/01_dense.md) — Dense variant: QK Norm, weights, TP, code template
- [Qwen3 MoE](02_model_specifics/03_qwen3/02_moe.md) — MoE variant: router, experts, fused kernel, weight packing

## 03 Operators

- [Attention Ops](03_operators/01_attention_ops.md) — Flash Attention, FlashInfer, Triton attention kernels

## 04 Parallel Strategies

- [Tensor Parallelism](04_parallel_strategies/01_tensor_parallel.md) — Column/row parallel, all-reduce, NCCL

## 05 Non-Core Features

### Extractable Complexity (can be omitted entirely)

- [Multi-Model Support](05_non_core_features/01_multi_model_support.md) — Model registry (extractable complexity)
- [Multi-Quantization](05_non_core_features/02_multi_quantization.md) — Quantization plugin system (extractable)
- [Platform Abstraction](05_non_core_features/03_platform_abstraction.md) — Multi-hardware support (extractable)

### Production Optional Features (important, with integration guides)

- [PD Disaggregation](05_non_core_features/04_pd_disaggregation.md) — Prefill-Decode separation with KV transfer
- [KVCache Connector](05_non_core_features/05_kvcache_connector.md) — Cross-node KV cache transfer interface
- [Post-Processing](05_non_core_features/06_post_processing.md) — Detokenization, stop strings, streaming, output formatting
- [Speculative Decoding](05_non_core_features/07_speculative_decoding.md) — EAGLE, draft model, MTP with integration patterns
- [Guided Decoding](05_non_core_features/08_guided_decoding.md) — JSON schema, regex, grammar-constrained generation

## 06 Implementation Patterns

- [Code Patterns](06_implementation_patterns/01_code_patterns.md) — Proven patterns: global context, merged proj, fused ops
- [Debugging War Stories](06_implementation_patterns/02_debugging_war_stories.md) — Qwen3.5 混合注意力引擎 15 个 bug 的完整调试记录
- [Anti-Patterns](06_implementation_patterns/03_anti_patterns.md) — Complexity traps to avoid in generated code
- [MPS Optimization Guide](06_implementation_patterns/04_mps_optimization_guide.md) — Apple Silicon MPS 推理引擎优化实录（23→47 tok/s）