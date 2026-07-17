# RMSNorm与RoPE原生算子

## 1. RMSNorm

接口按最后一维归一化，输入/输出Shape一致，Weight为`[hidden]`。FP16/BF16输入
至少使用FP32归约。Epsilon来自模型配置；Residual Fusion必须明确输入是否被修改。

```cpp
Status RmsNorm(TensorView output, TensorView input,
               TensorView weight, float epsilon,
               const OperatorContext& context);
```

Kernel可采用Block Reduction或向量化Load，但必须处理Hidden尾部、Alignment和
非连续View。任何In-place行为写入接口和Implementation ID，调用方不得猜测。

## 2. RoPE

RoPE输入Q/K、逻辑Positions和预计算Cache。实现必须区分Interleaved/NeoX、
Scaling和MRoPE；未知Layout返回Unsupported。

```cpp
Status ApplyRope(TensorView q, TensorView k, TensorView positions,
                 const RopeConfig& config, const RopeCache& cache,
                 const OperatorContext& context);
```

Position范围在Launch前验证。物理KV Slot和Block ID不得进入Position。Cache DType、
最大长度和Section Layout写入Build/Runtime Manifest。

## 3. 测试

- RMSNorm全零、大值、不同Epsilon、Hidden尾部和DType；
- Fused与Unfused Residual链逐层一致；
- RoPE Position 0/边界、不同Layout、Scaling和MRoPE；
- 相同Token放入不同KV物理位置时RoPE结果相同；
- Kernel Launch失败、越界Position和错误Stride非零失败。

