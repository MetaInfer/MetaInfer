# Non-Core Feature: Multi-Model Support

## What It Is

Production frameworks like vLLM (270+ models) and SGLang (100+ models) support a vast array of model architectures. This requires:
- Model registry / dispatch tables
- Dynamic architecture loading
- Generic weight mapping
- Adapter patterns for different attention/MLP variants

## Why It's Non-Core

A generated inference framework targets a **specific model**. You don't need a registry when you know exactly which model to run.

## What Production Frameworks Do

### Model Registry (sglang/vllm pattern)
```python
# Global registry mapping HF model type → inference implementation
MODEL_REGISTRY = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "MistralForCausalLM": MistralForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
    # ... 100+ entries
}

def load_model(config):
    model_class = MODEL_REGISTRY[config.architectures[0]]
    return model_class(config)
```

### Dynamic Architecture Detection
```python
# Must handle:
# - Different attention types (MHA, GQA, MQA, MLA)
# - Different MLP types (dense, gated, MoE)
# - Different normalization (LayerNorm, RMSNorm)
# - Different positional encoding (RoPE, ALiBi, absolute)
# - Different embedding/head tying strategies
```

## What a Generated Framework Should Do Instead

Hard-code the specific model architecture:
```python
# No registry, no dispatch — just the model you need
class MyModel(nn.Module):
    def __init__(self, config):
        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MyDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)
```

## Complexity Saved

- No registry lookup or dispatch tables
- No generic weight mapping logic
- No conditional branches for different architectures
- No abstract base classes for layers
- Simpler weight loading (direct mapping)
