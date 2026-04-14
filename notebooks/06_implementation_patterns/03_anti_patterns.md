# Anti-Patterns in LLM Inference Frameworks

## Complexity Traps Found in Production Frameworks

These patterns are present in vLLM and SGLang (the negative examples) and should be avoided in generated inference code.

### 1. Over-Abstraction: Plugin Systems for Single-Use Components

**Anti-pattern**:
```python
# vLLM: QuantizationConfig → QuantizationMethod → QuantizedLinear
# When you only need AWQ, this creates 3 layers of indirection

class QuantizationConfig(ABC):
    @abstractmethod
    def get_quant_method(self, layer, prefix): ...

class AWQConfig(QuantizationConfig):
    def get_quant_method(self, layer, prefix):
        return AWQLinearMethod()

class AWQLinearMethod(LinearMethodBase):
    def create_weights(self, ...): ...
    def apply(self, ...): ...
```

**Better approach for generated code**:
```python
# Direct implementation, no indirection
class AWQLinear(nn.Module):
    def forward(self, x):
        return awq_gemm(x, self.qweight, self.scales, self.qzeros)
```

### 2. Environment Variable Explosion

**Anti-pattern**:
```python
# vLLM/SGLang: hundreds of env vars controlling behavior
VLLM_USE_FLASHINFER = os.getenv("VLLM_USE_FLASHINFER", "0")
VLLM_ATTENTION_BACKEND = os.getenv("VLLM_ATTENTION_BACKEND", "")
VLLM_USE_V1 = os.getenv("VLLM_USE_V1", "1")
SGLANG_USE_RADIX_CACHE = os.getenv("SGLANG_USE_RADIX_CACHE", "1")
# ... dozens more, creating invisible configuration surface
```

**Better approach**: Hard-code decisions at generation time. The AI agent chooses the configuration; the generated code has no runtime branches.

### 3. Dynamic Model Dispatch

**Anti-pattern**:
```python
# Runtime model selection with string matching
MODEL_REGISTRY = {...}  # 270+ entries
model_cls = MODEL_REGISTRY.get(config.architectures[0])
if model_cls is None:
    raise ValueError(f"Unsupported model: {config.architectures[0]}")
model = model_cls(config)
```

**Better approach**: The model is known at generation time. Import and instantiate directly.

### 4. Backward Compatibility Layers

**Anti-pattern**:
```python
# Supporting both v0 and v1 engine simultaneously
class LLMEngine:
    def __init__(self):
        if USE_V1:
            self._engine = V1Engine()
        else:
            self._engine = V0Engine()

    def step(self):
        if USE_V1:
            return self._engine.step_v1()
        else:
            return self._engine.step_v0()
```

**Better approach**: Pick one implementation. Generated code doesn't need migration paths.

### 5. Universal Configuration Objects

**Anti-pattern**:
```python
# A config class that tries to represent every possible model
@dataclass
class ModelConfig:
    # Standard transformer
    hidden_size: int
    num_heads: int
    # MoE specific
    num_experts: Optional[int] = None
    num_shared_experts: Optional[int] = None
    # MLA specific
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    # Mamba specific
    state_size: Optional[int] = None
    # Vision specific
    image_size: Optional[int] = None
    patch_size: Optional[int] = None
    # ... 50+ optional fields
```

**Better approach**: Config contains only what the specific model needs.

### 6. Kernel Fallback Chains

**Anti-pattern**:
```python
def get_attention_kernel():
    if has_flash_attn_v3() and is_hopper():
        return FlashAttnV3Backend()
    elif has_flash_attn_v2():
        return FlashAttnV2Backend()
    elif has_flashinfer():
        return FlashInferBackend()
    elif has_triton():
        return TritonBackend()
    elif has_xformers():
        return XformersBackend()
    else:
        return TorchSDPABackend()
```

**Better approach**: The target hardware is known. Use the best kernel directly.

### 7. Excessive Logging and Metrics

**Anti-pattern**:
```python
# Every component has its own logger, metrics collector, profiler hooks
class Scheduler:
    def schedule(self):
        logger.debug(f"Scheduling {len(self.waiting)} requests")
        metrics.gauge("scheduler.waiting_count", len(self.waiting))
        with profiler.trace("scheduler.schedule"):
            # ... actual logic (3 lines)
        metrics.histogram("scheduler.schedule_time", elapsed)
        logger.debug(f"Scheduled {len(batch)} sequences")
```

**Better approach**: Generated code is simple enough to understand without pervasive instrumentation. Add targeted logging only for debugging.

### 8. Generic Data Structures with Type Tags

**Anti-pattern**:
```python
# Using type tags to represent different message types
class Message:
    type: str  # "generate_req", "abort_req", "batch_result", ...
    data: Any

def handle_message(msg):
    if msg.type == "generate_req":
        handle_generate(msg.data)
    elif msg.type == "abort_req":
        handle_abort(msg.data)
    # ... many more branches
```

**Better approach**: Use typed function calls or typed dataclasses with discriminated unions.

## Summary: What to Avoid in Generated Code

| Trap | Root Cause | Generated Code Approach |
|------|-----------|------------------------|
| Plugin systems | Supporting future unknowns | Direct implementation |
| Env var explosion | Runtime configurability | Compile-time decisions |
| Model registry | Supporting many models | Single model, direct import |
| Compatibility layers | Supporting old versions | Clean single implementation |
| Universal configs | One-size-fits-all | Model-specific config |
| Kernel fallbacks | Multi-platform | Single optimal kernel |
| Excessive instrumentation | Debugging at scale | Minimal, targeted logging |
| Type-tagged messages | Generic messaging | Typed interfaces |
