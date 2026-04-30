# Non-Core Feature: Multiple Quantization Methods

## What It Is

Production frameworks support 20+ quantization methods for loading and running models with reduced precision. Each method has its own:
- Weight format and layout
- Dequantization kernel
- Linear layer implementation
- Calibration data requirements

## Supported Methods in Production (vLLM example)

| Category | Methods |
|----------|---------|
| Weight-only | AWQ, GPTQ, Marlin, GGUF, BitsAndBytes |
| Activation-Weight | W8A8 (INT8), FP8 (W8A8), MXFP4/8 |
| Hardware-specific | FBGEMM_FP8, Marlin (GPU-optimized GPTQ) |
| Framework | TorchAO, ModelOpt, AutoRound |

## Why It's Non-Core

A generated framework targets a **specific quantization** (or no quantization at all). You only need one dequantization kernel, not a plugin system.

## How Production Frameworks Handle It

### Quantization Config Dispatch
```python
# Each quantization method implements a QuantizationConfig
class AWQConfig(QuantizationConfig):
    def get_linear_method(self):
        return AWQLinearMethod()

class GPTQConfig(QuantizationConfig):
    def get_linear_method(self):
        return GPTQLinearMethod()

# Linear layers use the method to create/apply weights
class QuantizedLinear(nn.Module):
    def __init__(self, method, ...):
        self.weight = method.create_weights(...)

    def forward(self, x):
        return method.apply(self.weight, x)
```

### AWQ Example (nano-sglang)
```python
class AWQLinearMethod:
    def create_weights(self, input_size, output_size):
        # 4-bit packed weights
        qweight = Parameter(torch.empty(input_size, output_size // 8, dtype=torch.int32))
        qzeros = Parameter(torch.empty(input_size // group_size, output_size // 8, dtype=torch.int32))
        scales = Parameter(torch.empty(input_size // group_size, output_size, dtype=torch.float16))
        return {"qweight": qweight, "qzeros": qzeros, "scales": scales}

    def apply(self, weights, x):
        return awq_gemm_triton(x, weights["qweight"], weights["scales"], weights["qzeros"])
```

## What a Generated Framework Should Do Instead

If the model uses a specific quantization:
```python
# Hard-code the specific quantization kernel
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        self.qweight = Parameter(...)  # Specific format
        self.scales = Parameter(...)

    def forward(self, x):
        return specific_dequant_gemm(x, self.qweight, self.scales)
```

If the model uses FP16/BF16 (no quantization):
```python
# Just use standard linear layers
class Linear(nn.Module):
    def forward(self, x):
        return F.linear(x, self.weight)
```

## Complexity Saved

- No quantization config parsing
- No linear method dispatch
- No plugin architecture for new quantization types
- No generic weight format handling
- Single kernel path, can be fully optimized
