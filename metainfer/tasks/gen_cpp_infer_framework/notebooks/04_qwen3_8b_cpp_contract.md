# Qwen3-8B C++ 推理实现契约

> 用途：它定义“必须实现什么、张量应是什么形状、权重怎样装载、prefill/decode 怎样共用 KV cache”，而不绑定 CUDA、CPU 或某个张量库。
> 配套阅读：`02_qwen3_forwrad compute.md` 解释 llama.cpp 的计算图；本文优先给出自己的 C++ 类边界和实现验收条件。

## 1. 首个可用版本的范围

目标是 **Qwen3-8B Dense、单请求（B=1）、单卡** 的自回归生成：

```text
HF config.json + safetensors
        ↓
ModelConfig / WeightLoader
        ↓
Qwen3Model::forward_prefill() 或 forward_decode()
        ↓
KVCache + BackendOps (GEMM / RMSNorm / RoPE / GQA)
        ↓
最后一个位置的 logits → Sampler → 下一个 token
```

第一版不需要实现 TP、连续批处理、paged allocator、CUDA Graph、LoRA、MoE、speculative decoding 或服务调度。它们是吞吐优化/框架能力；先让单序列的 logits 与参考实现一致。

虽然来源契约的运行范围是 `B=1, TP=4`，这里的主设计故意采用单卡全量权重。这样没有 all-reduce，也更容易验证数学与权重布局；TP=4 的扩展规则见第 9 节。

## 2. 配置：运行时读取，8B 数字只作校验

不要把下表硬编码为唯一真相。加载 `config.json`，并校验模型与这些 Qwen3-8B 参考值一致；若没有 `head_dim`，使用 `hidden_size / num_attention_heads`。

| 字段 | Qwen3-8B 参考值 | C++ 用途 |
| --- | ---: | --- |
| `hidden_size` (`H`) | 4096 | token hidden state 的宽度 |
| `intermediate_size` (`I`) | 12288 | SwiGLU 的 gate/up 宽度 |
| `num_hidden_layers` (`L`) | 36 | Decoder block 数 |
| `num_attention_heads` (`Nq`) | 32 | Query head 数 |
| `num_key_value_heads` (`Nkv`) | 8 | Key/Value head 数，说明使用 GQA |
| `head_dim` (`D`) | 128 | 每个 attention head 的维度 |
| `vocab_size` (`V`) | 151936 | embedding 与 lm head 行数 |
| `rms_norm_eps` | `1e-6` | 所有 RMSNorm 的 epsilon |
| `rope_theta` | `1_000_000` | RoPE 基频 |
| `max_position_embeddings` | 40960 | cache/position 的参考最大长度 |
| `tie_word_embeddings` | `false` | 8B 参考模型的 lm head 不与 embedding 复用 |

必须满足：

```text
H == Nq * D == 32 * 128 == 4096
GQA group size = Nq / Nkv == 4
q_size = Nq * D == 4096
kv_size = Nkv * D == 1024
fused_qkv_size = q_size + 2 * kv_size == 6144
```

`Nq % Nkv == 0` 是 GQA 内核可直接按组映射的前提；不要只因 `H` 对得上就跳过它。

## 3. 推荐的 C++ 数据模型

以下接口把模型常量、权重、运行期 KV 状态分开。`Tensor` 的行主序/列主序可由后端自行决定，但 `linear(x, w)` 的数学含义必须统一为 `x @ w^T`，即 HF 的 `[out_features, in_features]` 权重不必转置后再保存。

```cpp
struct ModelConfig {
    int hidden_size;          // H = 4096
    int intermediate_size;    // I = 12288
    int num_layers;           // L = 36
    int num_q_heads;          // Nq = 32
    int num_kv_heads;         // Nkv = 8
    int head_dim;             // D = 128
    int vocab_size;           // V = 151936
    int max_seq_len;
    float rms_norm_eps;
    float rope_theta;
    bool tie_word_embeddings;
};

struct LayerWeights {
    Tensor attn_norm;         // [H]
    Tensor qkv;               // [Q + K + V, H] = [6144, 4096]
    Tensor q_norm;            // [D] = [128]
    Tensor k_norm;            // [D] = [128]
    Tensor o_proj;            // [H, H]
    Tensor ffn_norm;          // [H]
    Tensor gate_up;           // [2I, H] = [24576, 4096]
    Tensor down_proj;         // [H, I] = [4096, 12288]
};

struct KVCacheLayer {
    Tensor key;               // [capacity, Nkv, D]
    Tensor value;             // [capacity, Nkv, D]
};

struct KVCache {
    std::vector<KVCacheLayer> layers;  // L 个层独立的 K/V
    int capacity;
    int length;                        // 已写入 token 数
};
```

`KVCache::length` 是一个请求的逻辑长度，而不是每层各自维护的计数器。每一层必须在同一个 position 写入 K/V；否则层间会读取不同 token 历史，结果会静默错误。

## 4. 权重加载与融合布局

HF checkpoint 是分离投影，而运行时可将它们融合为较少的 GEMM。这个融合只改变存储/调度，不能改变行段顺序。

| HF key | HF shape（8B） | C++ 目标 | 操作 |
| --- | --- | --- | --- |
| `model.embed_tokens.weight` | `[V, H]` | `embed_tokens` | 直接复制 |
| `layers.i.self_attn.q_proj.weight` | `[4096, 4096]` | `qkv[0:4096, :]` | 放在 Q 段 |
| `layers.i.self_attn.k_proj.weight` | `[1024, 4096]` | `qkv[4096:5120, :]` | 紧跟 Q 的 K 段 |
| `layers.i.self_attn.v_proj.weight` | `[1024, 4096]` | `qkv[5120:6144, :]` | 最后的 V 段 |
| `layers.i.self_attn.o_proj.weight` | `[4096, 4096]` | `o_proj` | 直接复制 |
| `layers.i.self_attn.q_norm.weight` | `[128]` | `q_norm` | 直接复制 |
| `layers.i.self_attn.k_norm.weight` | `[128]` | `k_norm` | 直接复制 |
| `layers.i.input_layernorm.weight` | `[4096]` | `attn_norm` | 直接复制 |
| `layers.i.post_attention_layernorm.weight` | `[4096]` | `ffn_norm` | 直接复制 |
| `layers.i.mlp.gate_proj.weight` | `[12288, 4096]` | `gate_up[0:12288, :]` | gate 在前 |
| `layers.i.mlp.up_proj.weight` | `[12288, 4096]` | `gate_up[12288:24576, :]` | up 在后 |
| `layers.i.mlp.down_proj.weight` | `[4096, 12288]` | `down_proj` | 直接复制 |
| `model.norm.weight` | `[4096]` | `final_norm` | 直接复制 |
| `lm_head.weight` | `[V, H]` | `lm_head` | 直接复制；仅配置要求时复用 embedding |

两条不可交换的顺序：

```text
qkv     = concat(Q, K, V, dim=0)       // 不是 Q-V-K
gate_up = concat(gate, up, dim=0)      // 不是 up-gate
```

应在加载完成后检查每个张量的名称、dtype、shape、字节数和是否已填充。加载失败时立即报错；不要为缺失的 `q_norm` 或 `k_norm` 填 1 并继续运行。

## 5. 单层前向：唯一的数学顺序

设输入 `x` 为 `[T, H]`，其中 prefill 时 `T` 是本次 prompt token 数，decode 时 `T=1`。每层使用下面顺序：

```text
residual = x
h        = RMSNorm(x, attn_norm)

[q, k, v] = split(linear(h, qkv), [4096, 1024, 1024])
q = reshape(q, [T, 32, 128])
k = reshape(k, [T,  8, 128])
v = reshape(v, [T,  8, 128])

q = RMSNorm_per_head(q, q_norm)
k = RMSNorm_per_head(k, k_norm)
q = RoPE(q, positions)
k = RoPE(k, positions)

write_kv(layer_cache, positions, k, v)
a = GQA(q, layer_cache.K[0:kv_end], layer_cache.V[0:kv_end], causal_mask)
x = residual + linear(reshape(a, [T, H]), o_proj)

residual = x
h        = RMSNorm(x, ffn_norm)
[gate, up] = split(linear(h, gate_up), [I, I])
x = residual + linear(silu(gate) * up, down_proj)
```

层循环结束后：

```text
hidden = RMSNorm(x, final_norm)
logits = linear(hidden, lm_head)       // [T, V]
```

RMSNorm 的标量定义为：

```text
rms_norm(x, w) = x * rsqrt(mean(x * x, dim=-1) + eps) * w
```

`q_norm` / `k_norm` 的 weight 是 `[D]`，所以它们独立作用于每个 head 的最后一维；不能把它们误当为长度 `H` 的 layer norm。来源实现还要求匹配其 BF16 计算路径，不能为了“稳定”无条件换成全程 FP32；归约精度和输出 dtype 应作为后端契约显式测试。

这里的残差写法把加法明确物化。来源的 vLLM 风格实现可返回 `(output, residual)` 并让调用者在下一次 RMSNorm 合并；两种调度都可以，但在你的 C++ 实现中每一个 `residual + ...` 必须恰好发生一次，不能省略也不能重复。

## 6. Attention、RoPE 与 GQA

对 query head `hq`，它对应的 KV head 是：

```text
hkv = hq / (Nq / Nkv) = hq / 4
score[t] = dot(q[hq], K[t, hkv]) / sqrt(D)
```

然后仅对可见位置做 softmax，并加权同一 `hkv` 的 V。缩放系数固定为：

```text
1 / sqrt(128) ≈ 0.08838835
```

Qwen3 的顺序是 **Q/K per-head RMSNorm → RoPE → 写 K cache / attention**。尤其不能把未 RoPE 的 K 写进 cache，也不能遗漏 Q/K norm。V 不做这两个变换，直接写入 V cache。

RoPE 的位置应是绝对 position：本次输入的第 `j` 个 token 使用 `position = start_pos + j`。缓存复用后不能把 decode token 又从 position 0 开始旋转。

## 7. Prefill 与 Decode 的运行时契约

| 阶段 | 输入 | position / KV 写入 | attention 可读范围 | 需要的 logits |
| --- | --- | --- | --- | --- |
| Prefill | prompt `[S]` | 写 `[start_pos, start_pos + S)` | 每个 token 只读到自身，保持 causal | 通常只保留最后一行 |
| Decode | 新 token `[1]` | 写 `cache.length` | `0 .. cache.length`（含刚写入位置） | 这一行用于采样下一个 token |

推荐的框架入口：

```cpp
Tensor Qwen3Model::forward_prefill(std::span<const int32_t> token_ids,
                                   KVCache& cache, int start_pos);
Tensor Qwen3Model::forward_decode(int32_t token_id, KVCache& cache);
```

这两个入口最终调用同一个 `forward_tokens(token_ids, positions, cache, causal)`，差别只是 token 数、mask 与 cache 的读写范围。完成 prefill 后设置 `cache.length = start_pos + S`；decode 成功后递增一次。超出 `cache.capacity` 必须返回错误，不能写越界。

以参考长度 `40960`、FP16/BF16 cache 为例，仅一个序列的 K/V cache 约为：

```text
2 (K,V) * 36 * 40960 * 8 * 128 * 2 bytes
= 6,039,797,760 bytes ≈ 5.625 GiB
```

因此 `max_seq_len` 是实打实的显存接口，而不仅是 RoPE 参数。第一版可用连续 dense cache；随后需要多请求时再换 block/paged cache。

## 8. 生成循环与验证顺序

最小生成循环应当是：

```text
prefill(prompt) → last_logits → sample(token_0)
decode(token_0) → logits      → sample(token_1)
decode(token_1) → logits      → sample(token_2)
...
```

先实现 greedy：`argmax(last_logits)`。temperature、top-k、top-p 只影响如何从 logits 选择 token，不应放入模型 forward，也不能改变 KV cache 的位置推进。

验收按下面顺序做，能最快定位错误：

1. 加载检查：36 层的每个 key 都存在，融合后的 QKV/gate-up shape 正确。
2. 固定小 prompt，比较 embedding、第一层 Q/K/V、第一层 attention 输出、第一层 MLP 输出、最终最后一行 logits；逐级比较比只比较最终 token 更容易发现布局错误。
3. 对同一 token 序列，比较「一次性 full prefill 的最后 logits」与「前 `S-1` token prefill + 第 `S` token decode」的最后 logits。两者应在所选 dtype 的允许误差内一致。
4. 固定 greedy sampling，比较连续生成的 token id；不一致时优先检查 QKV 拼接、gate/up 顺序、Q/K norm、RoPE position、KV 写入索引和残差加法。
5. 只在正确性通过后，才融合 RMSNorm+residual、QKV GEMM、RoPE、paged attention 或加入 CUDA Graph。

## 9. 后续 TP=4 扩展（不是第一版前置条件）

来源契约的 TP=4 每 rank 数值为：`Nq=8`、`Nkv=2`、`q_size=1024`、`kv_size=256`。可沿输出维切分并在每 rank 内按同样的 `Q-K-V`、`gate-up` 顺序融合：

```text
local_qkv     = [1024 + 256 + 256, 4096] = [1536, 4096]
local_gate_up = [2 * (12288 / 4), 4096]  = [6144, 4096]
```

`qkv` 与 `gate_up` 是 column-parallel：各 rank 计算自己的输出分片。`o_proj` 与 `down_proj` 是 row-parallel：各 rank 计算部分和，然后对 hidden-size 输出执行 all-reduce。embedding/lm head 沿 vocab 维切分，lm head 若不 gather 则 sampler 需要做跨 rank 的 global argmax 或 global sampling。

TP 下要保证每个 rank 的 K/V cache 只存自己的两个 KV heads；不要在每 rank 复制全量 KV heads，否则既浪费显存也容易和本地 Q head 的映射混淆。

## 10. 实现前检查表

- [ ] 从 `config.json` 读取所有字段，并验证 `H == Nq * D`、`Nq % Nkv == 0`。
- [ ] HF 权重名使用 Qwen3-8B 的 `model.layers.*` 前缀，而非 Qwen3.5/3.6 的 `model.language_model.*`。
- [ ] QKV 融合顺序是 Q → K → V，gate-up 融合顺序是 gate → up。
- [ ] 每层拥有独立 K/V cache；K 在 q/k norm 和 RoPE 后写入。
- [ ] 注意力缩放为 `1/sqrt(head_dim)`，并按 4 个 Q head 共享一个 KV head 实现 GQA。
- [ ] 两次残差加法均存在：attention 后一次，MLP 后一次。
- [ ] 最终 RMSNorm 后才做 lm head，并在第一版仅提取最后位置的 logits。
- [ ] 先完成 prefill/decode 等价性测试，再做融合或并行优化。
