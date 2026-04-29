# LLM Inference Framework Knowledge Base

## Purpose

This knowledge base distills the common patterns and design principles found across multiple LLM inference frameworks (vLLM, SGLang, nano-vllm, nano-sglang, mini-sglang). It serves as a methodology reference for an AI Agent system that generates customized, minimal inference frameworks tailored to specific deployment requirements.

## Knowledge Structure

### Core Framework Design (`01_framework_design/`)

Universal methodology for building any LLM inference framework. These are the essential building blocks that every inference system needs, regardless of the specific model or deployment scenario.

- **Architecture** - Overall system topology, process model, and component relationships
- **Scheduler** - Continuous batching, prefill/decode scheduling, preemption
- **KV Cache** - Paged attention, radix/prefix caching, memory block management
- **Model Runner** - Forward pass execution, CUDA graph capture/replay
- **Sampler** - Token sampling strategies (greedy, top-k, top-p, temperature)
- **Memory Pool** - GPU memory allocation, pool management, utilization estimation
- **Request Lifecycle** - End-to-end flow from HTTP request to generated response

### Model-Specific Knowledge (`02_model_specifics/`)

Architecture-specific patterns and implementation details for different model families. When generating code for a specific model, reference the relevant sub-section.

- **Transformer Models** - General dense transformer patterns (Llama, Qwen, Mistral)
- **DeepSeek V3** - MLA attention, MoE routing, MTP (Multi-Token Prediction), NSA

### Operators (`03_operators/`)

Kernel-level knowledge for performance-critical operations.

- **Attention Ops** - Flash Attention, FlashInfer, Triton attention, paged KV access
- (Extensible: RoPE, MoE kernels, normalization, etc.)

### Parallel Strategies (`04_parallel_strategies/`)

How to distribute computation across multiple GPUs.

- **Tensor Parallelism** - Column/row splitting, all-reduce patterns, weight sharding

### Non-Core Features (`05_non_core_features/`)

Features that are common in production frameworks but can be independently added or removed based on deployment requirements. Divided into two categories:

**Extractable Complexity** (can be omitted entirely for minimal frameworks):

- **Multi-Model Support** - Model registry, dynamic dispatch
- **Multiple Quantization Methods** - AWQ, GPTQ, FP8, etc.
- **Platform Abstraction** - Multi-hardware support (CUDA, ROCm, XPU, TPU)

**Production Optional Features** (important in real deployments, each document includes integration guidance):

- **PD Disaggregation** - Prefill-Decode separation for independent scaling
- **KVCache Connector** - Cross-node KV cache transfer interface (NCCL, RDMA, Mooncake, NIXL)
- **Post-Processing** - Incremental detokenization, stop string detection, streaming, output formatting
- **Speculative Decoding** - EAGLE, draft model, MTP/NextN for latency reduction
- **Guided Decoding** - JSON schema, regex, grammar-constrained generation (XGrammar, Outlines)

### Implementation Patterns (`06_implementation_patterns/`)

Code-level patterns and anti-patterns observed across projects.

- **Code Patterns** - Common implementation patterns that work well
- **Anti-Patterns** - Complexity traps to avoid in generated code

## How to Use This Knowledge Base

1. **For framework generation**: Start with `01_framework_design/` to understand the core architecture. Every generated framework needs these components.
2. **For model-specific code**: Consult `02_model_specifics/` for the target model architecture.
3. **For performance**: Reference `03_operators/` and `04_parallel_strategies/` for GPU-optimized implementations.
4. **For feature additions**: Check `05_non_core_features/`. Each production optional feature document includes specific integration guidance — where to add hooks, what components to modify, and configuration parameters. These features can be seamlessly added to the generated framework based on user requirements.
5. **For code quality**: Follow `06_implementation_patterns/` to generate clean, maintainable code.

## Source Projects


| Project     | Role                | Description                                                                                                                         |
| ----------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| nano-vllm   | Positive example    | ~20 files, implements PagedAttention + continuous batching for Qwen3                                                                |
| nano-sglang | Positive example    | ~35 files, implements RadixCache + multi-process serving for Llama/Mixtral                                                          |
| mini-sglang | Positive example    | ~60 files, official SGLang team simplified version with CUDA kernels and TP                                                         |
| vllm        | Reference (complex) | 270+ models, 20+ quantization methods, 15+ attention backends. Source for production features like PD disaggregation, KV connectors |
| sglang      | Reference (complex) | 100+ models, speculative decoding, guided decoding, multimodal. Source for production features like EAGLE, XGrammar integration     |
