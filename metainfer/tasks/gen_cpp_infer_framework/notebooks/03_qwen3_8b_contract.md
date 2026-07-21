# Qwen3-8B C++ 推理实现契约

> 用途：它定义“必须实现什么、张量应是什么形状、权重怎样装载、prefill/decode 怎样共用 KV cache”，而不绑定 CUDA、CPU 或某个张量库。
> 配套阅读：`02_qwen3_forwrad compute.md` 解释 llama.cpp 的计算图；本文优先给出自己的 C++ 类边界和实现验收条件。
> Z200 算子与调用知识库：`04_qwen3_z200_operator_contract.md`。算子事实来源是 `qwen3_z200_kernels.hip.cpp`；Agent 生成单卡版本时必须先读算子契约，再按参考源码接线。所有带权重矩阵乘统一由 `qwen3_z200_q8_linear_fp32()` 调用 `hipblasGemmEx`。

## 1. 首个可用版本的范围

目标是 **Qwen3-8B Dense、单请求（B=1）、单卡** 的自回归生成：

```text
Qwen3-8B Q8_0 GGUF
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

> **多并发实现覆盖说明：** 本文后续关于单请求 `B=1`、单一 `KVCache::length` 和单行 logits 的描述仅适用于首版 bring-up。要实现多个 HTTP 请求的 continuous batching，必须先阅读并遵守 `09_continuous_batching_contract.md`；其中的 `slot_id`、`RuntimeBatch` 和 `[layer][slot][position][head][dim]` KV 所有权契约覆盖本文的单序列生命周期描述。

虽然来源契约的运行范围是 `B=1, TP=4`，这里的主设计故意采用单卡全量权重。这样没有 all-reduce，也更容易验证数学与权重布局；TP=4 的扩展规则见第 9 节。

## 2. 配置：固定 Qwen3-8B，GGUF metadata 用于校验

当前小框架不做通用模型注册，只支持固定 Qwen3-8B。下表是 runtime 真值；Loader 读取 GGUF metadata 后逐项校验，任何关键字段不一致都应报错，而不是动态切换到其他尺寸。

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
    Tensor q_proj;            // [Q, H] = [4096, 4096]
    Tensor k_proj;            // [K, H] = [1024, 4096]
    Tensor v_proj;            // [Vkv, H] = [1024, 4096]
    Tensor q_norm;            // [D] = [128]
    Tensor k_norm;            // [D] = [128]
    Tensor o_proj;            // [H, H]
    Tensor ffn_norm;          // [H]
    Tensor gate_proj;         // [I, H] = [12288, 4096]
    Tensor up_proj;           // [I, H] = [12288, 4096]
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

## 4. 权重加载布局与可选融合

GGUF 为当前目标提供独立 Q/K/V 和 gate/up tensor。首版保持独立，直接匹配现有 compact kernel 接口；后续可以融合为较少的 GEMM，但融合只改变存储/调度，不能改变行段顺序，而且必须补齐 `04_qwen3_z200_operator_contract.md` 第 6 节的布局 helper。

| HF key | HF shape（8B） | C++ 目标 | 操作 |
| --- | --- | --- | --- |
| `model.embed_tokens.weight` | `[V, H]` | `embed_tokens` | 直接复制 |
| `layers.i.self_attn.q_proj.weight` | `[4096, 4096]` | `q_proj` | 首版独立保存 |
| `layers.i.self_attn.k_proj.weight` | `[1024, 4096]` | `k_proj` | 首版独立保存 |
| `layers.i.self_attn.v_proj.weight` | `[1024, 4096]` | `v_proj` | 首版独立保存 |
| `layers.i.self_attn.o_proj.weight` | `[4096, 4096]` | `o_proj` | 直接复制 |
| `layers.i.self_attn.q_norm.weight` | `[128]` | `q_norm` | 直接复制 |
| `layers.i.self_attn.k_norm.weight` | `[128]` | `k_norm` | 直接复制 |
| `layers.i.input_layernorm.weight` | `[4096]` | `attn_norm` | 直接复制 |
| `layers.i.post_attention_layernorm.weight` | `[4096]` | `ffn_norm` | 直接复制 |
| `layers.i.mlp.gate_proj.weight` | `[12288, 4096]` | `gate_proj` | 首版独立保存 |
| `layers.i.mlp.up_proj.weight` | `[12288, 4096]` | `up_proj` | 首版独立保存 |
| `layers.i.mlp.down_proj.weight` | `[4096, 12288]` | `down_proj` | 直接复制 |
| `model.norm.weight` | `[4096]` | `final_norm` | 直接复制 |
| `lm_head.weight` | `[V, H]` | `lm_head` | 直接复制；仅配置要求时复用 embedding |

若后续启用融合，两条不可交换的顺序是：

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

q = linear(h, q_proj)
k = linear(h, k_proj)
v = linear(h, v_proj)
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
gate = linear(h, gate_proj)
up   = linear(h, up_proj)
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

## 8. Z200 单卡 Q8_0 落地流程（首版 Agent 参考）

本节把前面的通用数学契约收敛为当前目标机器上的一条可执行路径：**一张 Hygon Z200（约 15.98 GiB 显存）、Qwen3-8B Q8_0、B=1、HIP + hipBLAS**。首版目标是正确运行，不以 decode 性能为验收条件。

完整的 public wrapper、反量化公式、hipBLAS 参数表、buffer 布局和逐算子调用顺序见 `04_qwen3_z200_operator_contract.md`。若本文摘要与算子契约对接口的描述不同，以算子契约和 `qwen3_z200_kernels.hip.cpp` 的实际签名为准。

### 8.1 首版数据类型边界

当前 `qwen3_z200_kernels.hip.cpp` 的非线性算子接口使用 FP32，线性层则通过 FP16 临时矩阵调用 hipBLAS：

| 对象 | 首版存储/计算类型 | 说明 |
| --- | --- | --- |
| 模型线性权重 | Q8_0，常驻显存 | 不在启动时把全模型永久展开为 FP16 |
| Embedding 权重 | Q8_0，常驻显存 | 只查表并解量化本次 token 对应的行 |
| RMSNorm、RoPE、attention、SwiGLU、残差的输入输出 | FP32 | 与当前自定义 kernel 接口保持一致 |
| 线性层输入临时区 | FP16 | 每次线性调用前由 `x_fp32` cast 得到 |
| 线性层权重临时区 | FP16 | 每次线性调用前由当前 Q8_0 矩阵解量化得到 |
| hipBLAS 累加与线性输出 | FP32 | `FP16 x FP16 -> FP32`，输出直接交给现有 FP32 kernel |
| 当前 K/V cache | FP32 | 因当前 cache/attention kernel 使用 `float*`；后续可统一改为 FP16 |

Q8_0 的块布局必须与加载文件一致：每 32 个权重共享一个 FP16 scale，随后是 32 个有符号 INT8 值。

```cpp
struct BlockQ8_0 {
    half   d;       // 2 bytes
    int8_t qs[32];  // 32 bytes
};                  // 共 34 bytes

w[i] = fp16_to_fp32(block.d) * fp32(block.qs[i % 32]);
```

加载器必须校验 `sizeof(BlockQ8_0) == 34`、被量化矩阵的 `K % 32 == 0`、shape 与 GGUF 元数据一致。首版保持独立 Q/K/V 和 gate/up。若后续融合，只能按输出行拼接已经量化的完整行，不能切开或重排一行内部的 Q8_0 block。

标准 hipBLAS 的 INT8 GEMM 不能直接表达“FP32/FP16 激活 × GGUF Q8_0 权重”：Q8_0 的 scale 沿 K 维每 32 个元素变化。因此首版不要把 `BlockQ8_0*` 直接传给普通 INT8 GEMM。当前正确路径是：

```text
x_fp32 ──cast──> x_fp16 ───────────────┐
                                       ├─ hipblasGemmEx ──> y_fp32
W_q8_0 ──dequant──> W_fp16 ────────────┘
```

`qwen3_z200_q8_linear_fp32()` 已封装这三个步骤，并约定行主序数学语义：

```text
Y[M, N] = X[M, K] @ W[N, K]^T
```

它每调用一次都会重新 cast 当前激活、重新解量化当前矩阵，然后 enqueue 一次 hipBLAS GEMM。这条路径**能作为正确性基线**，但 decode 时会反复读完整权重并产生较大的解量化开销；不能把它描述成最终性能方案。

### 8.2 启动时只分配一次的资源

Agent 生成 runtime 时应在模型初始化阶段完成以下工作，禁止在逐层 forward 中反复 `hipMalloc/hipFree`：

1. 构造固定 Qwen3-8B config，并用 GGUF metadata 执行第 2 节的 shape 校验。
2. 加载 GGUF Q8_0 tensor；按第 4 节首版独立映射 Q/K/V 和 gate/up，Q8_0 权重常驻 device memory。
3. 创建一个 hipBLAS handle，并让所有 kernel、cast、dequant、GEMM 使用同一条 HIP stream，依靠流内顺序保证依赖。
4. 分配可复用的 FP32 激活区、attention/MLP 中间区、logits、KV cache。
5. 分配一个 FP16 激活 workspace，容量至少为本次最大 GEMM 的 `M * K` 个元素。prefill 太长时应先按 token chunk 限制 `M`，而不是无上限扩大 workspace。
6. 分配一个 FP16 权重 workspace，多个线性层串行复用。若 LM Head 首版不分块，它必须容纳最大矩阵 `[V, H]`。

Qwen3-8B 最大的 FP16 临时权重是 LM Head：

```text
151936 * 4096 * 2 bytes
= 1,244,659,712 bytes
= 1187 MiB ≈ 1.159 GiB
```

作为对比，首版单个 gate 或 up 的临时权重为：

```text
12288 * 4096 * 2 bytes = 96 MiB
```

即使后续 fused gate-up 也只有 192 MiB，仍会在 LM Head 处失败。首版最简单的策略是分配约 1.159 GiB 的单个权重 workspace，让所有线性层复用；若 `hipMemGetInfo` 显示余量不足，再实现第 8.5 节的 LM Head 分块。

当前 FP32 KV cache 在 `max_seq_len=4096` 时为：

```text
2 * 36 * 4096 * 8 * 128 * 4 bytes = 1.125 GiB
```

若直接按 `40960` 分配则是 `11.25 GiB`，加上 Q8_0 权重、1.159 GiB 权重 workspace 和其他激活后无法在约 15.98 GiB 的卡上安全运行。因此：

- 第一版把运行时 `max_seq_len` 默认限制在 4096 或更小，并在启动时根据实际空闲显存校验；
- 不要因为 `config.max_position_embeddings == 40960` 就无条件分配 40960 长度的 cache；
- 后续把 K/V cache 改为 FP16 后，4096 长度会降为约 0.5625 GiB，但长上下文仍要做实际显存预算；
- Q8_0 模型文件/权重实际字节数必须从 GGUF tensor 信息统计，不能只用“8B 参数”估算后承诺一定可跑。

### 8.3 单卡 prefill 参考调用流程

下面是 Agent 应生成的调用骨架。`T` 可以是整个 prompt，也可以是受 workspace 限制的 prefill chunk；首版 B 固定为 1。

```text
token_ids[T]
  └─ qwen3_z200_launch_embedding_lookup_q8_0
       └─ x_fp32[T, 4096]                 // 只解 token 对应行，不展开整个 embedding

for layer = 0 .. 35:
  residual = x_fp32
  h = RMSNorm(x_fp32, attn_norm)          // FP32

  q = q8_linear(h, q_weight,
                M=T, N=4096, K=4096)      // compact FP32 [T,4096]
  k = q8_linear(h, k_weight,
                M=T, N=1024, K=4096)      // compact FP32 [T,1024]
  v = q8_linear(h, v_weight,
                M=T, N=1024, K=4096)      // compact FP32 [T,1024]
  per_head_rms(q, q_norm)
  per_head_rms(k, k_norm)
  RoPE(q, absolute_positions)
  RoPE(k, absolute_positions)
  write_kv(layer, positions, k, v)
  attn = prefill_gqa(q, K_cache, V_cache, causal=true)

  attn_out = q8_linear(attn, o_proj,
                       M=T, N=4096, K=4096)
  x_fp32 = residual + attn_out

  residual = x_fp32
  h = RMSNorm(x_fp32, ffn_norm)
  gate = q8_linear(h, gate_weight,
                   M=T, N=12288, K=4096)
  up = q8_linear(h, up_weight,
                 M=T, N=12288, K=4096)
  ffn = SwiGLU(gate, up)
  down = q8_linear(ffn, down_proj,
                   M=T, N=4096, K=12288)
  x_fp32 = residual + down

last_hidden = RMSNorm(x_fp32[T-1], final_norm)
logits[1,V] = q8_linear(last_hidden, lm_head,
                        M=1, N=151936, K=4096)
```

关键点是 LM Head 的 `M=1`：prefill 生成只需要最后一个 prompt token 的 logits。绝不能为了取最后一行而计算 `[T, 151936]` 的完整 logits。若将 prompt 分块，还必须确保 attention 保持全局 causal 语义，且最终仅在最后一个 chunk 的最后一个位置执行 LM Head。

首版独立投影直接得到 compact tensor，不需要 QKV split/pack 或 strided SwiGLU。若后续启用 fused QKV/gate-up 优化，这两个 helper 就会成为 `T>1` prefill 的正确性前置条件，详见 `04_qwen3_z200_operator_contract.md` 第 6 节。

### 8.4 单卡 decode 参考调用流程

Decode 复用同一套 buffer 和线性接口，但所有线性层的 `M=1`：

```text
token_id -> Q8_0 embedding row -> x_fp32[1,H]

for each layer:
  attn norm
  q8 q/k/v independent linears (M=1)
  q/k per-head norm -> RoPE(position=cache.length)
  write K/V at cache.length
  decode GQA over [0, cache.length]
  q8 o linear (M=1) -> residual add
  ffn norm
  q8 gate/up independent linears (M=1) -> SwiGLU
  q8 down linear (M=1) -> residual add

final norm
q8 lm_head linear (M=1) -> logits[V]
forward success -> cache.length += 1
```

在此正确性版本中，每层的 Q/K/V、O、gate、up、down，以及最终 LM Head 都会在各自调用前解量化一次。不要把“每个算子都解量化”理解成要单独保存 36 层的 FP16 副本；它们按流串行地覆盖同一个权重 workspace。

为了先跑通，允许这种重复解量化；为了提高 decode 速度，后续应优先实现直接读取 Q8_0 的 fused GEMV/GEMM（decode 的 `M=1` 特别适合 GEMV），而不是长期依赖“整矩阵解到 FP16 + hipBLAS”。

### 8.5 Embedding 与 LM Head 的特殊处理

**Embedding 不是 GEMM。** 调用 `qwen3_z200_launch_embedding_lookup_q8_0()`，根据 token id 找到 `[H]` 行并在读取时解成 FP32。一次 prompt 只解 `T * H` 个元素；decode 只解一行。禁止先把 `[V,H]` embedding 全部展开为 FP16/FP32。

**LM Head 首版可以不分块。** 使用通用 `qwen3_z200_q8_linear_fp32()`，传入 `M=1, N=V, K=H`，把整个独立 `lm_head.weight` 解到约 1.159 GiB FP16 workspace 后做一次 GEMM。Qwen3-8B 的 `tie_word_embeddings=false`，不能默认拿 embedding 指针代替 LM Head。

只有在显存检查失败时，才将 vocab 维分块。例如 `vocab_chunk=16384`：

```text
for v0 in range(0, V, vocab_chunk):
    rows = min(vocab_chunk, V - v0)
    dequant lm_head[v0:v0+rows, :] -> smaller FP16 workspace
    GEMM [1,H] x [rows,H]^T -> logits[v0:v0+rows]
```

分块不改变数学结果，只减少峰值 workspace；会增加 kernel/GEMM 调度次数。第一版显存足够时不应把它设为前置条件。

### 8.6 Agent 实现边界与验收

Agent 按本节实现时，必须满足：

- 所有 workspace 在初始化时分配并在模型销毁时释放，逐 token 路径无 `hipMalloc/hipFree`；
- 每个 launch 和 hipBLAS 调用都检查返回值，错误信息至少包含 layer、算子名和 `(M,N,K)`；
- 同一线性调用的 cast、dequant、GEMM 位于同一 stream，权重 workspace 在 GEMM 消费完成前不得被下一条并发 stream 覆盖；
- 正常逐层执行不调用 `hipDeviceSynchronize()`；调试时可在层边界同步定位错误；
- 启动日志打印 Q8_0 权重总字节数、KV cache、两个 FP16 workspace、FP32 激活/中间区和剩余显存；
- 先通过第 9 节的数值验证，再评估或实现 Q8_0 fused GEMV、FP16 KV cache、prefill 优化和 LM Head 分块。

这份首版流程保证的是接口和数学路径闭合。Z200 的 `gfx906` 编译、hipBLAS 版本兼容性以及实卡数值结果仍必须在目标环境验证，不能只以静态源码测试代替。

## 9. 生成循环与验证顺序

最小生成循环应当是：

```text
prefill(prompt) → last_logits → sample(token_0)
decode(token_0) → logits      → sample(token_1)
decode(token_1) → logits      → sample(token_2)
...
```

先实现 greedy：`argmax(last_logits)`。temperature、top-k、top-p 只影响如何从 logits 选择 token，不应放入模型 forward，也不能改变 KV cache 的位置推进。

验收按下面顺序做，能最快定位错误：

1. 加载检查：36 层的每个独立 Q/K/V、gate/up key 都存在且 shape 正确；`output.weight` 必须独立存在。
2. 固定小 prompt，比较 embedding、第一层 Q/K/V、第一层 attention 输出、第一层 MLP 输出、最终最后一行 logits；逐级比较比只比较最终 token 更容易发现布局错误。
3. 对同一 token 序列，比较「一次性 full prefill 的最后 logits」与「前 `S-1` token prefill + 第 `S` token decode」的最后 logits。两者应在所选 dtype 的允许误差内一致。
4. 固定 greedy sampling，比较连续生成的 token id；不一致时优先检查 Q/K/V layout、gate/up 配对、Q/K norm、RoPE position、KV 写入索引和残差加法。
5. 只在正确性通过后，才融合 RMSNorm+residual、QKV GEMM、RoPE、paged attention 或加入 CUDA Graph。

## 10. 后续 TP=4 扩展（不是第一版前置条件）

来源契约的 TP=4 每 rank 数值为：`Nq=8`、`Nkv=2`、`q_size=1024`、`kv_size=256`。可沿输出维切分并在每 rank 内按同样的 `Q-K-V`、`gate-up` 顺序融合：

```text
local_qkv     = [1024 + 256 + 256, 4096] = [1536, 4096]
local_gate_up = [2 * (12288 / 4), 4096]  = [6144, 4096]
```

`qkv` 与 `gate_up` 是 column-parallel：各 rank 计算自己的输出分片。`o_proj` 与 `down_proj` 是 row-parallel：各 rank 计算部分和，然后对 hidden-size 输出执行 all-reduce。embedding/lm head 沿 vocab 维切分，lm head 若不 gather 则 sampler 需要做跨 rank 的 global argmax 或 global sampling。

TP 下要保证每个 rank 的 K/V cache 只存自己的两个 KV heads；不要在每 rank 复制全量 KV heads，否则既浪费显存也容易和本地 Q head 的映射混淆。

## 11. 实现前检查表

- [ ] 使用固定 Qwen3-8B config，并读取 GGUF metadata 验证 `H == Nq * D`、`Nq % Nkv == 0` 和全部目标尺寸。
- [ ] HF 权重名使用 Qwen3-8B 的 `model.layers.*` 前缀，而非 Qwen3.5/3.6 的 `model.language_model.*`。
- [ ] 首版 Q/K/V 和 gate/up 权重独立；若后续融合，顺序必须是 Q → K → V、gate → up，并补齐多 token 布局 helper。
- [ ] 每层拥有独立 K/V cache；K 在 q/k norm 和 RoPE 后写入。
- [ ] 注意力缩放为 `1/sqrt(head_dim)`，并按 4 个 Q head 共享一个 KV head 实现 GQA。
- [ ] 两次残差加法均存在：attention 后一次，MLP 后一次。
- [ ] 最终 RMSNorm 后才做 lm head，并在第一版仅提取最后位置的 logits。
- [ ] Z200 单卡 Q8_0 路径使用“激活 cast FP16 + 当前权重解量化 FP16 + hipBLAS FP32 累加”，没有把 Q8_0 指针直接传给普通 INT8 GEMM。
- [ ] Embedding 仅解量化 token 对应行；未把整个 `[V,H]` embedding 展开。
- [ ] 不分块 LM Head 的 FP16 权重 workspace 至少为 `151936 * 4096` 个元素，prefill/decode 均只以 `M=1` 计算最终 logits。
- [ ] FP32 KV cache 按实际 `max_seq_len` 做显存预算，单卡首版不因配置上限而盲目分配 40960 长度。
- [ ] 逐层复用初始化时分配的 workspace，forward 内无反复 `hipMalloc/hipFree`。
- [ ] 先完成 prefill/decode 等价性测试，再做融合或并行优化。
