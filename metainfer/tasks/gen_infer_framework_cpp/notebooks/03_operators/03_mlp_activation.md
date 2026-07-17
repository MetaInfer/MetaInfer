# MLP、Activation与Projection算子

## 1. 语义

Qwen Dense MLP通常执行：

```text
gate = X * W_gate^T
up   = X * W_up^T
hidden = SiLU(gate) * up
out = hidden * W_down^T
```

Gate/Up可打包为一次GEMM，但Weight Layout必须与Loader Pack Order一致。Down为
Row Parallel时，Local Partial Output需要AllReduce后才能进入Residual。

## 2. BLAS边界

Model层传递语义Shape和Layout给统一GEMM Adapter，不直接调用厂商API。Adapter
验证M/N/K、Leading Dimension、Transpose、Compute DType、Alignment和Stream。
算法选择只使用Probe通过的能力，并缓存到固定Shape/DType Key。

## 3. Activation

SiLU Multiply使用稳定实现并支持尾部元素。融合Kernel必须声明输入是否为
`[tokens, 2*intermediate]`打包布局或两个独立View。MoE按Expert执行相同数学，
但Token Assignment和Scatter属于模型/并行层。

## 4. Fusion规则

先建立GEMM + 独立Activation的Reference，再启用Bias/Activation/Residual Fusion。
Fusion不得改变Residual所有权、TP通信位置或错误观察边界。每个Fusion记录支持
Shape、DType、Architecture、Workspace和Fallback。

## 5. 测试

非方阵和尾部GEMM、Gate/Up Pack顺序、SiLU极值、FP16/BF16与FP32 Reference、
TP Shard重组、Fusion/Unfused逐元素比较、错误Leading Dimension和OOM。

