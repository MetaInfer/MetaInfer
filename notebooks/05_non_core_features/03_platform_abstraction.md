# Non-Core Feature: Platform Abstraction

## What It Is

Production frameworks support multiple hardware platforms beyond NVIDIA CUDA:
- AMD ROCm
- Intel XPU
- Google TPU
- CPU (x86, ARM)
- Specialized accelerators

## Why It's Non-Core

A generated framework targets a **specific hardware platform**. Cross-platform abstractions add indirection that hurts both readability and performance.

## What Production Frameworks Do

### Platform Detection (vLLM)
```python
# vllm/platforms/
class CudaPlatform(Platform):
    def get_attn_backend(self):
        return FlashAttentionBackend

class RocmPlatform(Platform):
    def get_attn_backend(self):
        return RocmAttentionBackend

class TpuPlatform(Platform):
    def get_attn_backend(self):
        return PallasAttentionBackend

# Runtime dispatch
platform = detect_platform()
attn_backend = platform.get_attn_backend()
```

### Kernel Selection
```python
# Production: check platform, fall back to alternatives
if platform.is_cuda():
    if has_flash_attn:
        use_flash_attn()
    elif has_triton:
        use_triton_attn()
    else:
        use_pytorch_sdpa()
elif platform.is_rocm():
    use_rocm_attn()
```

## What a Generated Framework Should Do Instead

```python
# Directly use the target platform's optimal kernel
from flash_attn import flash_attn_varlen_func

def attention(q, k, v, ...):
    return flash_attn_varlen_func(q, k, v, ...)
```

## Complexity Saved

- No platform detection logic
- No kernel fallback chains
- No abstract platform interfaces
- No conditional compilation
- Direct imports of platform-specific libraries
