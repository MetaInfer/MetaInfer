# Qwen3 原生运行时契约

> 权威级别：C++ 卡片中 Qwen3 模型正确性的强制契约。

所有维度、Special Token ID、RoPE 参数和 MoE 参数必须来自目标模型的 `config.json`。Qwen3-8B 的常见数值只能作为测试 Fixture，不能成为未知 Checkpoint 的默认值。

卡片字段 `model_family`、`weight_dtype`、`kv_cache_dtype` 是用户策略输入。`Auto` 的含义是从 Checkpoint 与已探测 Backend 能力中推导并验证，不是允许 Agent 任意选择转换类型。

## 1. 配置结构与校验

```cpp
struct Qwen3Config {
  std::int64_t hidden_size = 0;
  std::int64_t intermediate_size = 0;
  std::int64_t num_hidden_layers = 0;
  std::int64_t num_attention_heads = 0;
  std::int64_t num_key_value_heads = 0;
  std::int64_t head_dim = 0;
  std::int64_t vocab_size = 0;
  std::int64_t max_position_embeddings = 0;
  double rms_norm_eps = 0.0;
  double rope_theta = 0.0;
  bool tie_word_embeddings = false;

  // MoE-only。Dense 模型必须保持为 0/false。
  std::int64_t num_experts = 0;
  std::int64_t num_experts_per_tok = 0;
  bool has_shared_expert = false;
};

Status ValidateQwen3Config(const Qwen3Config& cfg, int tp_size);
```

至少验证：

```cpp
CHECK_GT(cfg.hidden_size, 0);
CHECK_GT(cfg.num_attention_heads, 0);
CHECK_EQ(cfg.hidden_size % cfg.num_attention_heads, 0);
CHECK_EQ(cfg.num_attention_heads % tp_size, 0);
CHECK_GT(cfg.num_key_value_heads, 0);
CHECK_GT(cfg.num_hidden_layers, 0);
CHECK_GT(cfg.vocab_size, 0);
```

当 `head_dim` 明确存在时以配置为准；仅在字段缺失时才允许显式计算 `hidden_size / num_attention_heads`。

## 2. 支持范围声明

Build Manifest 必须声明支持：

```text
Qwen3 Dense
Qwen3 MoE
或两者
```

Dense-only Runtime 读取到 MoE 配置时必须在分配显存之前拒绝。禁止忽略 Expert 字段后按 Dense MLP 继续运行。

## 3. Dense Decoder 的强制顺序

每层必须按语义顺序执行：

1. 保存 Residual，执行 Input RMSNorm；
2. Q/K/V Projection；
3. 若 Checkpoint 定义，执行 Qwen3 Per-head Q/K RMSNorm；
4. 根据逻辑 Position 对 Q/K 执行 RoPE；
5. Causal GQA Attention，并写入/读取 KV；
6. Output Projection 和 Residual Addition；
7. Post-attention RMSNorm；
8. Gate/Up Projection；
9. `SiLU(gate) * up`；
10. Down Projection 和 Residual Addition。

最后执行 Final RMSNorm 和 LM Head。只有 `tie_word_embeddings=true` 时才允许共享 Embedding/LM Head Storage。

推荐把 Layer 接口写成明确的 Prefill/Decode 输入：

```cpp
class Qwen3DecoderLayer {
 public:
  Result<void> Prefill(const TensorView& hidden,
                       const PositionView& positions,
                       KvLayerView kv,
                       TensorView output,
                       BackendStream stream) const;

  Result<void> Decode(const TensorView& hidden_one_token,
                      const PositionView& positions,
                      const BlockTableView& blocks,
                      KvLayerView kv,
                      TensorView output,
                      BackendStream stream) const;
};
```

## 4. Shape 契约

- `hidden_size % num_attention_heads == 0`；
- Query Head 必须能按声明的策略切分到 TP Rank；
- GQA 的 KV Head 必须明确“切分”或“复制/分组”策略；
- Packed QKV 的 Pack Order 必须写入 Weight Mapping 并有测试；
- Vocab Padding/Shard 不得改变对外 Token ID；
- Tensor Layout/Stride 必须显式，禁止仅凭总字节数解释 Shape。

典型 Q/K/V 语义 Shape：

```text
Q: [tokens, local_q_heads, head_dim]
K: [tokens, local_or_replicated_kv_heads, head_dim]
V: [tokens, local_or_replicated_kv_heads, head_dim]
```

## 5. 数值规则

- RMS 统计和 Softmax Reduction 使用经过验证的 Accumulation DType；
- Attention Scale 只能应用一次；
- Stable Softmax 必须先减 Row Max；
- Mask 使用逻辑 Position，并覆盖不等长 Prefill；
- Greedy Argmax 在相同 Logits 下必须确定性处理 Tie；
- 不支持的 Weight/KV DType 在模型加载前拒绝，禁止静默转换。

## 6. 权重加载契约

必须读取 Safetensors Index（如果存在），并在 Upload 前验证：

- 必需 Key 全部存在且没有重复来源；
- Source DType、Shape 和 Byte Range 正确；
- TP Slice 与目标 Tensor Shape 一致；
- Q/K Norm、Gate/Up/Down、Final Norm、LM Head 的命名映射完整；
- Layout Transform 有明确理由与独立测试。

禁止通过任意 Reshape/Transpose 让错误字节“看起来能装入目标 Tensor”。

## 7. 正确性门槛

固定小 Prompt 至少比较：

```text
Embedding
一个 Decoder Layer 输入/输出
Prefill Logits
一次带 KV 的 Decode Logits
连续若干 Greedy Token
```

比较时使用与 DType 匹配的绝对/相对误差。只有 End-to-end 文本正确而没有中间结果测试，不足以定位模型数学错误。
