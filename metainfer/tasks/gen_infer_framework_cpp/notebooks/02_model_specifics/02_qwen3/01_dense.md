# Qwen3 原生前向与 Decode Runtime

先读：`00_contracts/qwen3_model_contracts.md`和`04_model_loader.md`。

先实现 Unfused Reference Path，再做 Fused/Architecture-specific Path。两条路径使用同一份权重和 Tensor Metadata；Debug 模式可导出指定中间 Tensor 进行对比。

## 1. 模型类结构

```cpp
class Qwen3Attention {
 public:
  Result<void> Prefill(const AttentionInput& input,
                       KvLayerView kv,
                       TensorView output,
                       BackendStream stream) const;
  Result<void> Decode(const AttentionInput& input,
                      const BlockTableBatchView& blocks,
                      KvLayerView kv,
                      TensorView output,
                      BackendStream stream) const;
};

class Qwen3Mlp {
 public:
  Result<void> Forward(const TensorView& input,
                       TensorView output,
                       WorkspaceLease& workspace,
                       BackendStream stream) const;
};

class Qwen3DecoderLayer {
 public:
  Result<void> Prefill(const LayerInput&, KvLayerView,
                       TensorView output, BackendStream) const;
  Result<void> Decode(const LayerInput&, BlockTableBatchView,
                      KvLayerView, TensorView output, BackendStream) const;
};

class Qwen3ForCausalLM {
 public:
  Result<ModelStepOutput> Forward(const ModelStepInput& input,
                                  KvBatchView kv,
                                  BackendStream stream) const;
};
```

Layer 持有/引用只读 Weight Storage，Forward 输入输出是 View。Layer 不拥有 Scheduler Request。

## 2. Shape 约定

统一使用 Flattened Token Layout 可以减少 Padding：

```text
hidden:    [total_tokens, hidden_size]
q:         [total_tokens, local_q_heads, head_dim]
k/v:       [total_tokens, local_or_replicated_kv_heads, head_dim]
positions: [total_tokens]
cu_seqlens:[batch + 1]     Prefill 变长序列
logits:    [selected_tokens, vocab_or_vocab_shard]
```

构建 `ModelStepInput` 时必须验证所有数组长度与 `StepPlan` 一致：

```cpp
struct ModelStepInput {
  StepKind kind;
  TensorView token_ids;       // int32 [total_tokens]
  TensorView positions;       // int32/int64 [total_tokens]
  TensorView sequence_ids;    // int32 [total_tokens]
  TensorView slot_mapping;    // int32/int64 [total_tokens]
  TensorView cu_seqlens;      // int32 [batch + 1], prefill
};
```

## 3. Q/K Norm 与 RoPE

Qwen3 Per-head Q/K Norm 作用在每个 Head 的 `head_dim` 上，不是 Decoder Input RMSNorm。

```text
q = reshape(q_proj(x), [tokens, q_heads, head_dim])
k = reshape(k_proj(x), [tokens, kv_heads, head_dim])
q = rms_norm_per_head(q, q_norm_weight)
k = rms_norm_per_head(k, k_norm_weight)
q = rope(q, positions)
k = rope(k, positions)
```

RoPE Pair 公式（常见 Interleaved Pair 语义，具体 Layout 必须与模型一致）：

```cpp
const float x0 = x[..., 2 * pair];
const float x1 = x[..., 2 * pair + 1];
out[..., 2 * pair]     = x0 * cos_value - x1 * sin_value;
out[..., 2 * pair + 1] = x0 * sin_value + x1 * cos_value;
```

`position` 来自 Request 逻辑 Token 位置，不能用 Flattened Token Index。

## 4. Prefill 数据流

```cpp
Result<ModelStepOutput> Qwen3ForCausalLM::Prefill(
    const ModelStepInput& input,
    KvBatchView kv,
    BackendStream stream) const {
  RETURN_IF_ERROR(ValidatePrefillInput(input, config_));

  ASSIGN_OR_RETURN(TensorStorage hidden,
                   workspaces_.AcquireHidden(input.token_ids.shape[0], stream));
  RETURN_IF_ERROR(embedding_.Lookup(input.token_ids, hidden.view(), stream));

  TensorView current = hidden.view();
  for (std::int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    ASSIGN_OR_RETURN(TensorView next, workspaces_.NextHidden(layer));
    RETURN_IF_ERROR(layers_[layer].Prefill(
        LayerInput{current, input.positions, input.cu_seqlens,
                   input.slot_mapping},
        kv.Layer(layer), next, stream));
    current = next;
  }

  ASSIGN_OR_RETURN(TensorView normalized, workspaces_.FinalHidden());
  RETURN_IF_ERROR(final_norm_.Forward(current, normalized, stream));

  ASSIGN_OR_RETURN(TensorView selected,
                   SelectLastValidTokenPerSequence(normalized,
                                                   input.cu_seqlens,
                                                   stream));
  ASSIGN_OR_RETURN(TensorStorage logits,
                   lm_head_.Forward(selected, stream));
  return ModelStepOutput{std::move(logits)};
}
```

此代码强调调用顺序和错误传播；Workspace 的真实 Ownership 需遵守内存 Contract。

## 5. Attention Prefill Reference Path

最初可以为 Tiny/Correctness Path 物化 Score：

```text
scores = Q * K^T * scale
scores += causal_and_length_mask
probabilities = stable_softmax(scores)
context = probabilities * V
```

但生产长上下文必须使用 Tiled/Streaming Attention，避免 `[heads, seq, seq]` 显存。Reference Path 保留给小 Shape 测试。

GQA Head 映射通常满足：

```cpp
const int queries_per_kv = num_query_heads / num_kv_heads;
const int kv_head = query_head / queries_per_kv;
```

必须先验证整除与 TP 策略，不能在 Kernel 内静默截断。

## 6. Decode 数据流

Decode 输入每个 Active Sequence 一枚最新 Token：

```cpp
Result<ModelStepOutput> Qwen3ForCausalLM::Decode(
    const ModelStepInput& input,
    const BlockTableBatchView& blocks,
    KvBatchView kv,
    BackendStream stream) const {
  RETURN_IF_ERROR(ValidateDecodeInput(input, blocks, config_));
  // embedding -> every layer decode -> final norm -> lm head
  // 每层在 slot_mapping 指定位置写入新 K/V，随后直接遍历 Paged KV。
  // 不得每步把全部历史 KV concatenate 到临时 Tensor。
  return DecodeImpl(input, blocks, kv, stream);
}
```

Engine 只有在所有 Layer 和 Sampling 成功后才提交 `kv_length` 和 Token。Kernel Failure 时 Request 状态不能前进一半。

## 7. MLP Reference Path

```cpp
Result<void> Qwen3Mlp::Forward(const TensorView& input,
                               TensorView output,
                               WorkspaceLease& workspace,
                               BackendStream stream) const {
  TensorView gate = workspace.Subview("gate");
  TensorView up = workspace.Subview("up");
  TensorView activated = workspace.Subview("activated");

  RETURN_IF_ERROR(gemm_->Run(input, gate_weight_, gate, stream));
  RETURN_IF_ERROR(gemm_->Run(input, up_weight_, up, stream));
  RETURN_IF_ERROR(ops_->SiluMultiply(gate, up, activated, stream));
  RETURN_IF_ERROR(gemm_->Run(activated, down_weight_, output, stream));
  return {};
}
```

后续可以 Pack Gate/Up 或 Fuse SiLU，但必须与此基线比较。

## 8. DType 边界

分别声明：

```cpp
struct NumericPolicy {
  DType weight_dtype;
  DType activation_dtype;
  DType reduction_dtype;
  DType logits_dtype;
  DType kv_cache_dtype;
};
```

Conversion 必须位于命名边界。BLAS Compute Type、RMSNorm/Softmax Accumulation、KV Store/Load Conversion 都要有测试。禁止由 BLAS 默认值隐式决定数值语义。

## 9. Sampling 边界

ModelRunner 输出 Logits，Sampler 负责 Token：

```cpp
class Sampler {
 public:
  Result<std::vector<std::int32_t>> Sample(
      const TensorView& logits,
      Span<const SamplingParams> params,
      BackendStream stream);
};
```

TP 模式下 Rank 0 采样后 Broadcast Token。Greedy `temperature=0` 必须确定性。

## 10. 正确性阶梯

```text
Embedding Lookup
RMSNorm
Q/K Per-head Norm
RoPE
单个 Projection/GEMM
Tiny 单头 Attention
单个 Decoder Layer
完整 Prefill Logits
一次带 Cache Decode
连续若干 Greedy Token
不同长度 Batch
TP=1 与 TP=N
```

每一级记录 Shape、有限值、Tolerance 和 Reference Hash。只有最后文本看起来合理不能证明模型正确。
