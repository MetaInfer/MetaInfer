# 原生Transformer模型通用模式

本文定义Dense Decoder-only Transformer的可复用语义。具体Checkpoint仍以目标
`config.json`和模型家族文档为准。

## 1. 配置驱动

```cpp
struct TransformerConfig {
  std::int64_t hidden_size;
  std::int64_t intermediate_size;
  std::int64_t num_layers;
  std::int64_t num_attention_heads;
  std::int64_t num_kv_heads;
  std::int64_t head_dim;
  std::int64_t vocab_size;
  double norm_epsilon;
  RopeConfig rope;
  ActivationKind activation;
};
```

所有Shape、Layer数、位置编码和Special Token来自Checkpoint；示例模型值不能作为
未知模型默认值。Loader在设备分配前验证整除关系、最大长度和后端DType支持。

## 2. 标准Decoder Layer

通用Pre-Norm层执行：

```text
x -> Norm -> QKV -> Position Encoding -> Causal Attention -> O Projection
  -> Residual -> Norm -> Gate/Up -> Activation*Up -> Down -> Residual
```

模型家族可以改变Norm位置、Bias、Q/K Norm、RoPE Layout、Attention类型或MLP，
因此通用类只复用接口和Storage，不把某一模型的顺序写死在Backend中。

## 3. Attention与GQA

Query Head和KV Head数量分别来自配置。GQA中多个Query Head共享KV Head；映射为
`query_head / (num_query_heads / num_kv_heads)`，但TP后必须使用Local/Replicated
策略重新验证。Prefill和Decode共享模型权重与数学语义，仅输入Metadata和KV访问
方式不同。

## 4. MLP与Norm

SwiGLU通常为`down(silu(gate(x)) * up(x))`。Gate/Up可以打包，但Checkpoint
映射必须明确顺序。RMSNorm统计采用稳定累加类型；Residual Fusion只能在与未融合
Reference逐层对比后启用。

## 5. Position Encoding

RoPE使用请求逻辑Position，不使用物理KV Offset。模型可能采用NeoX/Interleaved、
Scaling或MRoPE，必须由`RopeConfig`分派独立Implementation ID。Cache长度和最大
位置在Request Admission前验证。

## 6. 模型接口

```cpp
class CausalLanguageModel {
 public:
  virtual Result<TensorStorage> Prefill(const PrefillBatch&, RuntimeContext&) = 0;
  virtual Result<TensorStorage> Decode(const DecodeBatch&, RuntimeContext&) = 0;
  virtual const ModelConfig& config() const noexcept = 0;
  virtual ~CausalLanguageModel() = default;
};
```

模型不拥有Scheduler和HTTP；RuntimeContext提供有生命周期约束的KV、Workspace、
Operator和Stream。

## 7. 验证阶梯

依次验证Config、Tokenizer、Weight Mapping、Embedding、单算子、单Layer、Prefill
Logits、带KV Decode Logits和多Token生成。End-to-end文本不能替代中间结果；
中间结果也不能替代真实服务Oracle。

