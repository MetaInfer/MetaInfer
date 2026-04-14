# 多量化支持（可抽离功能）

## 1. 量化概述

量化通过降低权重和激活的精度来减少内存占用和加速推理。

### 1.1 常见量化方法

| 方法 | 权重精度 | 激活精度 | 特点 |
|------|----------|----------|------|
| FP16 | 16-bit | 16-bit | 标准精度 |
| BF16 | 16-bit | 16-bit | 更大动态范围 |
| FP8 | 8-bit | 8-bit | 需要特殊硬件支持 |
| INT8 | 8-bit | 8-bit | 需要校准 |
| INT4 | 4-bit | 16-bit | 权重量化 |
| AWQ | 4-bit | 16-bit | 激活感知 |
| GPTQ | 4-bit | 16-bit | 训练后量化 |
| GGUF | 多种 | 多种 | CPU友好 |

### 1.2 复杂性来源

```python
# vLLM 支持的量化方法
QUANTIZATION_METHODS = [
    "awq", "fp8", "fbgemm_fp8", "fp_quant",
    "modelopt", "modelopt_fp4", "modelopt_mxfp8",
    "gguf", "gptq_marlin", "awq_marlin", "gptq",
    "compressed-tensors", "bitsandbytes", "experts_int8",
    "quark", "moe_wna16", "torchao", "inc",
    "mxfp4", "mxfp8", "cpu_awq", "online",
    # ... 30+ 方法
]
```

## 2. 量化层设计

### 2.1 量化配置基类

```python
class QuantizationConfig(ABC):
    """量化配置基类"""
    
    @abstractmethod
    def get_linear_method(self) -> "LinearMethod":
        """返回线性层量化方法"""
        pass
    
    @abstractmethod
    def get_scaled_act_names(self) -> list[str]:
        """返回需要缩放的激活层名"""
        pass

class AWQConfig(QuantizationConfig):
    """AWQ量化配置"""
    def __init__(self, weight_bits=4, group_size=128):
        self.weight_bits = weight_bits
        self.group_size = group_size
    
    def get_linear_method(self):
        return AWQLinearMethod(self)
```

### 2.2 量化线性层

```python
class AWQLinearMethod:
    """AWQ量化线性层方法"""
    
    def apply(self, layer, x):
        """应用AWQ量化"""
        # 量化权重
        qweight = layer.qweight
        scales = layer.scales
        
        # 反量化
        weight = self.dequantize(qweight, scales)
        
        # 线性计算
        return F.linear(x, weight)

class QuantizedLinear(nn.Module):
    """量化线性层"""
    def __init__(self, in_features, out_features, quant_config):
        super().__init__()
        self.quant_method = quant_config.get_linear_method()
        
        # 根据量化方法存储不同格式的权重
        if isinstance(quant_config, AWQConfig):
            self.qweight = nn.Parameter(...)  # 量化权重
            self.scales = nn.Parameter(...)   # 缩放因子
        elif isinstance(quant_config, FP8Config):
            self.weight = nn.Parameter(...)   # FP8权重
    
    def forward(self, x):
        return self.quant_method.apply(self, x)
```

## 3. 量化权重加载

### 3.1 权重格式差异

```python
# 不同量化方法有不同的权重格式

# AWQ格式
{
    "qweight": packed_int4,    # 打包的4-bit权重
    "qzeros": packed_zeros,    # 打包的零点
    "scales": fp16_scales,     # FP16缩放因子
}

# GPTQ格式
{
    "qweight": packed_int4,    # 打包的4-bit权重
    "qzeros": packed_zeros,
    "scales": fp16_scales,
    "g_idx": group_indices,    # 组索引
}

# FP8格式
{
    "weight": fp8_weight,      # FP8权重
    "input_scale": scale,      # 输入缩放
    "weight_scale": scale,     # 权重缩放
}
```

### 3.2 权重加载逻辑

```python
def load_quantized_weights(model, state_dict, quant_config):
    """加载量化权重"""
    for name, param in model.named_parameters():
        if name in state_dict:
            # 根据量化方法处理
            if isinstance(quant_config, AWQConfig):
                load_awq_weight(param, state_dict[name])
            elif isinstance(quant_config, GPTQConfig):
                load_gptq_weight(param, state_dict[name])
            elif isinstance(quant_config, FP8Config):
                load_fp8_weight(param, state_dict[name])
            # ... 每种量化方法都需要处理
```

## 4. 量化算子

### 4.1 反量化算子

```python
# AWQ反量化（Triton实现）
@triton.jit
def awq_dequant_kernel(
    qweight_ptr,
    scales_ptr,
    output_ptr,
    GROUP_SIZE: tl.constexpr,
):
    """
    AWQ反量化内核
    INT4 → FP16
    """
    idx = tl.program_id(0)
    
    # 加载量化权重
    qweight = tl.load(qweight_ptr + idx)
    
    # 计算group索引
    group_idx = idx // GROUP_SIZE
    scale = tl.load(scales_ptr + group_idx)
    
    # 反量化
    weight = qweight * scale
    
    tl.store(output_ptr + idx, weight)
```

### 4.2 量化计算

```python
# 量化矩阵乘法
def quantized_matmul(x, qweight, scales):
    """
    量化矩阵乘法
    x: [M, K] FP16
    qweight: [K, N] INT4 packed
    scales: [K/group_size, N] FP16
    """
    # 方式1: 反量化后计算
    weight = dequantize(qweight, scales)
    output = x @ weight.T
    
    # 方式2: 混合精度计算（更快）
    output = quantized_gemm(x, qweight, scales)
    
    return output
```

## 5. 复杂性分析

### 5.1 代码量

```
vLLM quantization目录:
├── awq.py              ~500行
├── gptq.py             ~600行
├── fp8.py              ~400行
├── marlin_utils.py     ~800行
├── bitsandbytes.py     ~300行
├── ... (30+文件)
总计: ~20K行
```

### 5.2 维护成本

- 每种量化方法需要独立的测试
- 不同硬件平台支持不同
- 与新模型架构的兼容性

## 6. 抽离建议

### 6.1 精简框架策略

**策略：只支持FP16/BF16**

```python
class Linear(nn.Module):
    """标准FP16线性层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float16)
        )
    
    def forward(self, x):
        return F.linear(x, self.weight)
```

### 6.2 如果需要量化

如果确实需要量化以节省内存：

```python
# 只支持一种量化方法
class AWQLinear(nn.Module):
    """固定AWQ量化"""
    def __init__(self, in_features, out_features, group_size=128):
        super().__init__()
        self.qweight = nn.Parameter(...)
        self.scales = nn.Parameter(...)
```

## 7. 量化选择建议

### 7.1 精简框架推荐

| 场景 | 推荐精度 | 原因 |
|------|----------|------|
| 学习/研究 | FP16 | 最简单 |
| 生产部署 | BF16/FP16 | 兼容性好 |
| 内存受限 | AWQ 4-bit | 平衡质量 |

### 7.2 不推荐的量化

| 量化方法 | 不推荐原因 |
|----------|------------|
| 多量化支持 | 增加大量代码 |
| 动态量化选择 | 增加运行时复杂度 |
| 稀疏量化 | 实现复杂 |

## 8. 总结

### 8.1 核心观点

量化是重要的优化技术，但支持多种量化方法会显著增加框架复杂度：

1. 每种量化方法需要独立的权重加载逻辑
2. 每种量化方法需要独立的算子实现
3. 不同量化方法的配置参数不同

### 8.2 精简建议

对于精简框架：
1. 只支持FP16/BF16
2. 如果必须量化，只支持一种方法（如AWQ）
3. 不提供量化选择参数
