# Qwen3-8B Runtime Notes Without GGML

这份文档给后续 agent 参考，用来实现一个不依赖 GGML 的 Qwen3-8B 推理 runtime。

前一份 `05_qwen3_gguf_loader_notes.md` 解决“权重、固定 config 校验和 tokenizer metadata 怎么从 GGUF 读出来”。这份文档解决“模型加载后，token ids 怎么跑 prefill，decode 怎么循环，KV cache 怎么维护，sampler/tokenizer 怎么接起来”。最终 HTTP 暴露方式见 `07_qwen3_http_server_contract.md`。

> **多并发实现覆盖说明：** 本文的 `Qwen3RuntimeState`、`reset() -> prefill() -> decode()` 和单份 sampler/KV cache 都是 B=1 的 bring-up 契约。实现多 HTTP 请求 continuous batching 时，必须改读 `09_continuous_batching_contract.md`：它定义 scheduler 是唯一 GPU 调用者、每请求 `SequenceState`/`slot_id`、batched runtime 接口和 kernel 改造要求。不要只移除 mutex 后并行调用本文的单序列 runtime。

目标是一个固定 Qwen3 dense text model 的小型 C++/HIP runtime，不做通用计算图框架。

## 1. Runtime 的职责

Runtime 负责：

- 持有固定 Qwen3-8B config 和对模型权重的非拥有引用。
- 创建 HIP stream、hipBLAS handle。
- 分配中间 activation buffers。
- 分配每层 KV cache。
- 生成 RoPE sin/cos table。
- 执行 prefill forward。
- 执行 decode forward。
- 维护初始化/有效状态、`current_pos`、KV cache 有效性和最后一轮 logits 是否可读。

Runtime 不负责：

- GGUF 二进制格式解析细节。
- tokenizer/chat template。
- greedy、top-k、top-p 等采样策略。
- HTTP 请求/响应。
- 通用 tensor graph。
- GGML backend scheduler。
- 多请求调度、paged KV、prefix cache。
- 训练、autograd。

## 2. 和现有代码的关系

当前旧代码的结构大概是：

```text
main.cu
  load_data()
  tokenizer
  LlamaInference engine
  prefill
  external sampler
  decode loop

llama_forward.cu
  LlamaInference::forward(...)

loadmodel.cu
  Config + float weights + RoPE table
```

新的 Qwen3 runtime 可以替代这些职责：

```text
05_qwen3_gguf_loader_notes.md
  -> Qwen3GgufModel: config + typed weights

06_qwen3_runtime_notes.md
  -> Qwen3Runtime: buffers + KV cache + prefill/decode + logits

qwen3_z200_kernels.hip.cpp
  -> actual kernels and hipBLAS Q8_0 linear wrapper

tokenizer.hpp / tokenizer.cpp
  -> existing Qwen3 byte-level BPE + single-turn chat prompt

07_qwen3_http_server_contract.md
  -> Qwen3Engine + sampler + OpenAI-compatible HTTP server
```

建议新增文件：

```text
qwen3_runtime.h
qwen3_runtime.cpp or qwen3_runtime.hip.cpp
qwen3_sampler.h
```

第一版也可以先把 runtime 写到一个 `.cpp/.hip.cpp` 里，跑通后再拆。

## 3. 总体流程

完整生成流程：

```text
load GGUF
read Qwen3 config
map weights
allocate runtime buffers
build RoPE table

prompt text
  -> tokenizer CPU, or directly provide token ids
  -> prefill(token_ids)
  -> logits for last prompt token
  -> sample next token

while not stop:
  token id
    -> decode(token id)
    -> logits
    -> sample next token
    -> detokenize and print
```

最小版可以跳过 tokenizer：

```text
std::vector<int32_t> prompt_ids = { ... };
runtime.prefill(prompt_ids.data(), prompt_ids.size(), &error);
Qwen3Sampler sampler(runtime.stream(), model.config().vocab_size);
CHECK_SAMPLER(sampler.initialize(&error));
const int32_t effective_max_new_tokens = std::min<int32_t>(
        max_new_tokens,
        runtime.max_seq_len() - runtime.current_pos());
for (int generated = 0; generated < effective_max_new_tokens; ++generated) {
    int32_t next = -1;
    CHECK_SAMPLER(sampler.sample(
            runtime, sparams, generated_ids, &next, &error));
    if (is_stop_token(next)) break;
    if (generated + 1 == effective_max_new_tokens) break;
    runtime.decode(next, &error);
}
```

## 4. Qwen3Runtime 建议接口

```cpp
struct RuntimeConfig {
    int32_t max_seq_len = 4096;
    int32_t max_prefill_tokens = 128;
};

struct Qwen3RuntimeState {
    int32_t current_pos = 0;
    int32_t n_prompt = 0;
    bool initialized = false;
    bool valid = false;
    bool has_logits = false;
};

class Qwen3Runtime {
public:
    // model must outlive Qwen3Runtime. Runtime does not own or copy weights.
    explicit Qwen3Runtime(Qwen3GgufModel& model, RuntimeConfig runtime_cfg);
    ~Qwen3Runtime();

    Qwen3Runtime(const Qwen3Runtime&) = delete;
    Qwen3Runtime& operator=(const Qwen3Runtime&) = delete;

    bool initialize(std::string* error);
    void reset();

    // Both functions only run model forward and update d_logits.
    // Sampling is an external CPU/GPU component.
    bool prefill(const int32_t* token_ids, size_t n_tokens, std::string* error);
    bool decode(int32_t token_id, std::string* error);

    // device_logits() is readable only when has_logits() is true.
    const float* device_logits() const { return d_logits_; }
    hipStream_t stream() const { return stream_; }
    int32_t current_pos() const { return state_.current_pos; }
    int32_t n_prompt() const { return state_.n_prompt; }
    int32_t max_seq_len() const { return runtime_cfg_.max_seq_len; }
    bool initialized() const { return state_.initialized; }
    bool valid() const { return state_.valid; }
    bool has_logits() const { return state_.has_logits; }

private:
    bool allocate_buffers(std::string* error);
    bool build_rope_table(std::string* error);
    bool forward_tokens(
            const int32_t* host_token_ids,
            int32_t n_tokens,
            int32_t start_pos,
            bool produce_logits,
            std::string* error);

    Qwen3GgufModel& model_;
    RuntimeConfig runtime_cfg_;
    Qwen3RuntimeState state_;
    hipStream_t stream_ = nullptr;
    float* d_logits_ = nullptr;
};
```

不要让构造函数按值接收 `Qwen3GgufModel`。`Qwen3Weights` 中保存了指向模型 tensor map 元素的指针，复制 model 后这些指针可能仍指向原对象。首版用非拥有引用最简单，并明确要求 `Qwen3GgufModel` 比 runtime 活得更久。

若希望 runtime 独占模型，应使用 move-only owner（例如 `std::unique_ptr<Qwen3GgufModel>`），并保证移动后重新建立或验证所有 tensor 指针，不能依赖默认复制。

Sampler 接口独立：

```cpp
struct SamplingParams {
    int32_t max_new_tokens = 128;
    int32_t top_k = 40;
    float top_p = 1.0f;
    float temperature = 0.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
    std::vector<int32_t> stop_token_ids;
};

int32_t sample_logits(
        const std::vector<float>& host_logits,
        const SamplingParams& params,
        const std::vector<int32_t>& generated_tokens);

class Qwen3Sampler {
public:
    Qwen3Sampler(hipStream_t stream, int32_t vocab_size);
    ~Qwen3Sampler();

    bool initialize(std::string* error); // allocate d_next_token once
    bool sample(
            const Qwen3Runtime& runtime,
            const SamplingParams& params,
            const std::vector<int32_t>& generated_tokens,
            int32_t* out_token,
            std::string* error);

private:
    hipStream_t stream_ = nullptr; // same stream as runtime.stream()
    int32_t vocab_size_ = 0;
    int* d_next_token_ = nullptr;
    std::vector<float> h_logits_; // only used by the CPU policy/debug path
};
```

`stream()` 是给同一进程内的外部 sampler 建立顺序依赖用的受控 accessor。Sampler 不创建第二条不相关的 stream，也不修改 runtime 状态；它只在 runtime stream 上消费 `device_logits()`。若实现者不希望公开裸 stream，也可以提供等价的 `greedy_sample()` / `copy_logits_to_host()` 受控方法，但公开接口必须闭合，不能在伪代码中使用未声明的 `stream`。

`Qwen3Sampler` 由 `Qwen3Engine` 初始化一次并跨请求复用；`sample()` 应校验 `stream_ == runtime.stream()`、`runtime.has_logits()`、`vocab_size_` 一致和 `out_token != nullptr`。不要在生成循环中重复分配 `d_next_token_` 或 `h_logits_`。

## 5. 必要状态

`Qwen3RuntimeState` 已在上一节的接口中定义，只保存 forward 必需状态。生命周期约束为：

```text
constructor: initialized=false, valid=false, has_logits=false
initialize success: initialized=true, valid=true, has_logits=false
prefill success: n_prompt=prompt length, current_pos=n_prompt, has_logits=true
decode success: current_pos += 1, has_logits=true
forward failure: valid=false, has_logits=false
reset after initialize: current_pos=0, n_prompt=0, valid=true, has_logits=false
```

`prefill()` 只接受一个新的逻辑请求，要求 `initialized && valid && current_pos == 0`。`decode()` 要求 `initialized && valid && has_logits && current_pos > 0`。这样可以阻止未初始化 forward、在旧请求后直接再次 prefill，以及读取上一轮残留 logits。

`current_pos` 是当前 decode 要写入 KV cache 的绝对位置。

例子：

```text
prompt length = 5
prefill writes KV positions 0,1,2,3,4
after prefill: current_pos = 5

decode token A writes KV position 5
after decode: current_pos = 6

decode token B writes KV position 6
after decode: current_pos = 7
```

不要把 `seq_len`、`n_tokens`、`current_pos` 混在一起：

```text
max_seq_len:  KV cache 总容量
n_tokens:     当前 forward 输入 token 数
current_pos:  当前新 token 在整段序列里的绝对位置
```

`generated_tokens` 属于上层 `Qwen3Engine`/sampler，不属于模型 forward 状态。`reset()` 至少执行：

```cpp
state_.current_pos = 0;
state_.n_prompt = 0;
state_.valid = true;
state_.has_logits = false;
```

`reset()` 只能在 `initialize()` 成功后调用，并保留 `state_.initialized=true`。不必清零整块 KV cache：后续 attention 只允许读取 `0 .. current_pos`，旧位置之外的数据不可见。调试模式可以填 NaN 帮助发现越界读取。

## 6. Config

当前任务只支持固定 Qwen3-8B dense，因此以下值是 runtime 的编译期/初始化真值，不是模糊 fallback：

```cpp
dim        = 4096;
hidden_dim = 12288;
n_layer    = 36;
n_head     = 32;
n_kv_head  = 8;
head_dim   = 128;
vocab_size = 151936;
rope_theta = 1000000.0f;
rms_eps    = 1e-6f;
```

Loader 仍然读取对应 GGUF metadata，但用途是逐项校验模型确实是目标 Qwen3-8B。任一关键值不一致就报错退出，不能悄悄把 runtime 改成另一个模型配置。

运行时 `max_seq_len` 不一定等于 GGUF 的训练 context。16GB 卡上先用：

```text
2048 or 4096
```

跑通后再放大。

## 7. Device Buffers

第一版使用 FP32 activation，权重按 GGUF 类型保存。

按 `max_tokens = max_prefill_tokens` 分配：

```text
d_token_ids     [max_tokens]
d_residual      [max_tokens, dim]
d_xb            [max_tokens, dim]
d_q             [max_tokens, n_head, head_dim]
d_k             [max_tokens, n_kv_head, head_dim]
d_v             [max_tokens, n_kv_head, head_dim]
d_q_norm        [max_tokens, n_head, head_dim]
d_k_norm        [max_tokens, n_kv_head, head_dim]
d_attn          [max_tokens, n_head, head_dim]
d_attn_proj     [max_tokens, dim]
d_gate          [max_tokens, hidden_dim]
d_up            [max_tokens, hidden_dim]
d_swiglu        [max_tokens, hidden_dim]
d_ffn_out       [max_tokens, dim]
d_logits        [vocab_size]
d_scores        [n_head, max_seq_len] for decode scratch
d_sin_table     [max_seq_len, head_dim / 2]
d_cos_table     [max_seq_len, head_dim / 2]
```

注意：不要按 `max_seq_len` 分配所有 activation。旧代码里很多 buffer 是 `max_seq_len * dim`，Qwen3-8B 会浪费很多显存。activation 只需要覆盖当前 forward 的 token 数。

朴素 prefill attention 是 `O(T^2)` 正确性 kernel。首版 bring-up 推荐先用：

```cpp
max_prefill_tokens = 64; // or 128
```

确认数值正确后再尝试 512/1024，并根据 activation 预算和实测耗时决定。prompt 更长时使用 chunked prefill，而不是按整个 `max_seq_len` 分配所有 activation。

## 8. KV Cache

Qwen3 是 GQA，KV cache 只需要 KV head，不需要 Q head。

```cpp
kv_dim = n_kv_head * head_dim;
```

每层 KV cache：

```text
K: [max_seq_len, n_kv_head, head_dim]
V: [max_seq_len, n_kv_head, head_dim]
```

总大小：

```cpp
kv_bytes = n_layer * 2 * max_seq_len * n_kv_head * head_dim * sizeof(float);
```

不要用：

```cpp
n_layer * 2 * max_seq_len * dim
```

因为 `dim = n_head * head_dim`，而 KV 只需要 `n_kv_head * head_dim`。对 Qwen3-8B：

```text
dim = 4096
kv_dim = 8 * 128 = 1024
```

用 `dim` 分配会多 4 倍。

第一版 KV cache 用 FP32，直接适配当前 attention kernels。后面要省显存，可以改 FP16 KV cache，同时修改 attention kernel 的读类型。

`max_seq_len=4096` 时的精确占用为：

```text
2 * 36 * 4096 * 8 * 128 * 4 bytes
= 1,207,959,552 bytes
= 1.125 GiB
```

Runtime 初始化时必须用 `size_t` 计算所有乘积，并调用 `hipMemGetInfo()` 打印模型、KV、workspace、activation 和剩余显存。任何配置超出安全预算时应在启动阶段拒绝，而不是等逐层 forward 才 OOM。

## 9. RoPE Table

Runtime 初始化时生成：

```cpp
build_rope_table(d_sin_table, d_cos_table, head_dim, max_seq_len, rope_theta);
```

Qwen3-8B 不要硬编码 `10000.0f`，要用：

```text
qwen3.rope.freq_base
```

常见是：

```cpp
rope_theta = 1000000.0f;
```

table shape：

```text
[max_seq_len, head_dim / 2]
```

调用现有 RoPE kernel 时必须显式使用 Qwen3 的 half-split 语义：

```cpp
rope_mode = ROPE_NEOX; // value 1 in the current kernel contract
q_row_stride = n_head * head_dim;       // 4096
k_row_stride = n_kv_head * head_dim;    // 1024
```

每次调用前检查：

```cpp
start_pos >= 0
n_tokens > 0
start_pos + n_tokens <= max_seq_len
start_pos + n_tokens <= rope_table_positions
```

## 10. Prefill Forward

输入：

```text
token_ids: [n_tokens]
start_pos: usually 0 for first prompt
```

公开接口接收 host token ids。`forward_tokens()` 首先把当前 chunk 精确复制到初始化时分配的 `d_token_ids`，然后再调用 embedding：

```cpp
hipMemcpyAsync(
    d_token_ids,
    host_token_ids,
    n_tokens * sizeof(int32_t),
    hipMemcpyHostToDevice,
    stream);
```

caller 的 host buffer 只需活到该异步复制被当前 stream 完成；首版 forward 末尾会同步 stream。不要把 host 指针误当 device pointer 直接传给 embedding kernel。

步骤：

```cpp
embedding_lookup(token_ids) -> residual [n_tokens, dim]

for layer in layers:
    rms_norm(xb, residual, attn_norm, n_tokens, dim, rms_eps)

    q = linear(xb, attn_q)       // [n_tokens, n_head * head_dim]
    k = linear(xb, attn_k)       // [n_tokens, n_kv_head * head_dim]
    v = linear(xb, attn_v)       // [n_tokens, n_kv_head * head_dim]

    per_head_rms_norm(q_norm, q, q_norm_weight)
    per_head_rms_norm(k_norm, k, k_norm_weight)

    rope(q_norm, start_pos, ROPE_NEOX)
    rope(k_norm, start_pos, ROPE_NEOX)

    kv_cache_write(k_norm, v, layer_cache, start_pos)

    prefill_gqa_attention(
        q_norm, K_cache, V_cache, attn,
        n_tokens, start_pos,
        scale=1.0f/sqrtf(head_dim))

    attn_proj = linear(attn, attn_output)
    residual += attn_proj

    rms_norm(xb, residual, ffn_norm, n_tokens, dim, rms_eps)

    gate = linear(xb, ffn_gate)
    up   = linear(xb, ffn_up)
    swiglu = silu(gate) * up
    ffn_out = linear(swiglu, ffn_down)

    residual += ffn_out

last_residual = residual + (n_tokens - 1) * dim
final_norm(xb, last_residual, output_norm, rows=1, dim=dim)
lm_head(xb, M=1) -> logits [vocab_size]
```

Qwen3 特别注意：

- Q/K projection 后有 per-head RMSNorm。
- Q/K norm 在 RoPE 前。
- Attention 是 GQA：`n_head > n_kv_head`。
- Attention scale 固定为 `1.0f / sqrtf(128.0f)`。
- 当前 norm kernel 按输入/输出不 alias 使用，`q_norm`、`k_norm` 需要独立 buffer。
- LM head 只需要最后一个 token 的 hidden state。`last_residual` 必须显式指向最终 layer 输出的最后一行；final RMSNorm 把这一行写入 `d_xb[0:dim]`，LM Head 再以 `d_xb` 为 `M=1` 输入，不能使用未定义的 `last_token_hidden`。

首版 runtime 明确选择分别执行 Q、K、V 三个 linear，以及分别执行 gate、up 两个 linear。这样直接得到 compact 多行 tensor，能使用当前 per-head norm、KV writer 和 SwiGLU kernel，不需要尚未实现的 fused split/pack/strided helper。代价是多几次整块反量化和 GEMM；这是首版正确性换性能的明确决策。

Prefill 完成后：

```cpp
state_.current_pos = start_pos + n_tokens;
state_.has_logits = produce_logits;
```

只有当前调用要求的工作和必要的 stream 错误检查都成功后，才能提交 `current_pos`：所有 chunk 都必须完成 36 层；`produce_logits=true` 的最后一个 chunk 还必须完成 final norm 和 LM Head。`produce_logits=false` 的非末 chunk 不执行这两个算子，因此不能把它们写成所有 chunk 的提交前置条件。

公开 `prefill()` 完成全部 chunk 后还要提交：

```cpp
state_.n_prompt = static_cast<int32_t>(total_prompt_tokens);
state_.has_logits = true; // final chunk produced logits
```

进入每次 forward 时先令 `state_.has_logits=false`，防止异步执行期间或失败后读取旧 logits。中途失败时保留旧 `current_pos`，同时设置：

```cpp
state_.valid = false;
state_.has_logits = false;
```

虽然重试可能覆盖部分 KV，但不同层可能已经写入不同进度；首版不要在失败状态上继续 decode，必须 `reset()` 后从 prompt 重新 prefill。

## 11. Decode Forward

输入：

```text
token_id: one token
start_pos = current_pos
n_tokens = 1
```

步骤和 prefill 类似，但 attention 用 decode kernel：

```cpp
embedding_lookup(token_id) -> residual [1, dim]

for layer in layers:
    rms_norm
    separate q/k/v linear
    q/k per-head rms_norm -> q_norm/k_norm
    rope(q_norm, current_pos, ROPE_NEOX)
    rope(k_norm, current_pos, ROPE_NEOX)
    kv_cache_write(k_norm, v, position=current_pos)
    decode_gqa_attention(
        q_norm, K_cache, V_cache, current_pos,
        scale=1.0f/sqrtf(head_dim))
    output projection
    residual add
    ffn
    residual add

final norm
lm_head -> logits
if the complete forward succeeded:
    current_pos += 1
    has_logits = true
else:
    state.valid = false
    state.has_logits = false
```

Decode 的 attention 是：

```text
Q: one token
K/V: positions [0 ... current_pos]
```

所以 decode 本质更像 GEMV/scan，不是大 GEMM。

## 12. Linear 调用策略

对 Q8_0 权重：

```cpp
qwen3_z200_q8_linear_fp32(
    handle,
    out,
    x,
    weight_q8,
    x_fp16_workspace,
    x_workspace_elements,
    weight_fp16_workspace,
    weight_workspace_elements,
    m,
    n,
    k,
    stream);
```

其中：

```text
x:      [m, k]
weight: GGML shape {k, n}, memory can be viewed as row-major W[n, k]
out:    [m, n]
```

常见参数：

```text
attn_q:        m=n_tokens, n=n_head*head_dim,    k=dim
attn_k:        m=n_tokens, n=n_kv_head*head_dim, k=dim
attn_v:        m=n_tokens, n=n_kv_head*head_dim, k=dim
attn_output:   m=n_tokens, n=dim,                k=n_head*head_dim
ffn_gate:      m=n_tokens, n=hidden_dim,         k=dim
ffn_up:        m=n_tokens, n=hidden_dim,         k=dim
ffn_down:      m=n_tokens, n=dim,                k=hidden_dim
lm_head:       m=1,        n=vocab_size,         k=dim
```

第一版可以用一个全局 workspace 复用：

```text
x_workspace_elements
    = max_prefill_tokens * max(dim, hidden_dim)
    = max_prefill_tokens * 12288 FP16 elements

weight_workspace_elements
    = vocab_size * dim
    = 151936 * 4096
    = 622,329,856 FP16 elements
```

显存提醒：

- 现在的 Q8_0 linear 是“整块权重反量化到 FP16 workspace 再 hipBLAS”。
- `lm_head/output.weight` 的 FP16 workspace 是 `1,244,659,712 bytes = 1187 MiB ~= 1.159 GiB`。
- 所有元素数和 byte 数使用 `size_t`，分配时再乘 `sizeof(__half)`，不要用 32-bit `int` 计算。
- 初始化阶段调用 `hipMemGetInfo()`；若完整 LM Head workspace 分配失败，再实现沿 vocab 行分块，不能静默分配更小 workspace 后继续调用。
- 如果 16GB 压力大，后面需要 fused Q8_0 GEMV/GEMM 或 tiled dequant GEMM，不能每次整块 dequant。

同一个 `weight_fp16_workspace` 只能串行复用。cast、dequant、hipBLAS GEMM 和消费 GEMM 输出的后续 kernel 必须进入同一条 HIP stream，或者用 event 建立显式依赖。首版只创建一条 stream，并执行：

```cpp
hipblasSetStream(handle, stream);
```

禁止两个 linear 并发使用共享 workspace，也不要在另一个线程同时修改同一 hipBLAS handle 的 stream 或 pointer mode。逐层路径中禁止 `hipMalloc/hipFree`；所有 workspace 在初始化时分配一次。

## 13. Sampler

当前 `qwen3_z200_kernels.hip.cpp` 已导出：

```cpp
extern "C" hipError_t qwen3_z200_launch_greedy_sample(
        const float* logits,
        int vocab_size,
        int* out_token,
        hipStream_t stream);
```

首版 `temperature == 0` 且没有 repetition penalty 等 logit processor 时，直接用这个 GPU argmax；确定性模式下忽略 `top_k/top_p`。Sampler 初始化时一次性分配 `d_next_token[1]`，析构时释放；每轮只把一个 `int32_t` 拷回 CPU：

```cpp
if (!runtime.has_logits()) {
    return set_error(error, "runtime has no readable logits");
}

CHECK_HIP(qwen3_z200_launch_greedy_sample(
        runtime.device_logits(),
        vocab_size,
        d_next_token,
        runtime.stream()));
CHECK_HIP(hipMemcpyAsync(
        &next_token,
        d_next_token,
        sizeof(int32_t),
        hipMemcpyDeviceToHost,
        runtime.stream()));
CHECK_HIP(hipStreamSynchronize(runtime.stream()));
```

该 kernel 对相同最大值选择更小 token id，适合作为确定性 greedy 基准。`temperature == 0` 不执行除零或 softmax。

以下情况仍走 CPU sampler：

- `temperature > 0` 时的随机采样，可再组合 top-k/top-p；
- `repetition_penalty != 1.0`，但尚未实现 GPU logit processor；
- 需要保存/检查完整 logits 的 correctness 调试。

CPU 路径显式使用 runtime 的 stream，不能引用一个未声明的局部 `stream`：

```cpp
CHECK_HIP(hipMemcpyAsync(
        h_logits.data(),
        runtime.device_logits(),
        vocab_size * sizeof(float),
        hipMemcpyDeviceToHost,
        runtime.stream()));
CHECK_HIP(hipStreamSynchronize(runtime.stream()));
int32_t next_token = sample_logits(h_logits, params, generated_tokens);
```

Qwen3 vocab 是 151936，完整复制每步为：

```text
151936 * sizeof(float) = 607744 bytes ~= 0.58 MiB
```

Sampler 只读取 logits 和生成历史，不修改 KV cache、`current_pos`、`n_prompt` 或 `has_logits`。GPU greedy 是一个外部采样算子，不改变“Runtime 只负责 forward”的边界。

Stop token 的第一来源是已经加载的 tokenizer：

```cpp
std::vector<int32_t> effective_stop_ids;
if (tokenizer.eos_token_id() >= 0) {
    effective_stop_ids.push_back(tokenizer.eos_token_id());
}
for (int32_t id : sparams.stop_token_ids) {
    append_unique(effective_stop_ids, id);
}
```

这份合并后的列表在生成循环中称为 `effective_stop_ids`。也可以从 HTTP/生成参数附加明确的 stop token id。不要要求 GGUF 一定存在 `tokenizer.ggml.eot_token_id`；当前 `Qwen3TokenizerData` 没有这个必需字段。

遇到 stop token 时不要把特殊 token 文本输出给用户：

```cpp
if (contains(effective_stop_ids, next_token)) {
    finish_reason = "stop";
    break;
}
```

## 14. Tokenizer 和 Detokenizer

Tokenizer 已由本目录的 `tokenizer.hpp` / `tokenizer.cpp` 实现，是 CPU 侧 `Qwen3Engine` 功能，不属于 GPU forward。GGUF loader 构造：

```text
tokenizer.ggml.model
tokenizer.ggml.pre
tokenizer.ggml.tokens
tokenizer.ggml.merges
tokenizer.ggml.token_type
tokenizer.ggml.bos_token_id
tokenizer.ggml.eos_token_id
tokenizer.ggml.padding_token_id
tokenizer.ggml.add_bos_token
```

并交给当前真实接口：

```cpp
Qwen3TokenizerData tokenizer_data = load_tokenizer_metadata(...);
Qwen3Tokenizer tokenizer;
std::string error;
if (!tokenizer.load(tokenizer_data, &error)) {
    throw std::runtime_error(error);
}
```

普通文本编码：

```cpp
Qwen3EncodeOptions options;
options.parse_special = true;
options.add_bos = false; // Qwen3 normally has no BOS
options.add_eos = false;

std::vector<int32_t> prompt_ids = tokenizer.encode(prompt, options);
```

单轮 chat：

```cpp
std::string formatted = tokenizer.format_chat_prompt(
        user_text,
        system_text,
        /* enable_thinking = */ false);
std::vector<int32_t> prompt_ids = tokenizer.encode(formatted, options);
```

流水线 C 的输出上限较短，因此 HTTP 首版默认 `enable_thinking=false`，避免 thinking 内容占满 `max_tokens` 后还没有产生最终答案。Tokenizer 当前只承诺最小单轮 `system + user` chat，不要伪装成完整多轮模板兼容。

Detokenizer 直接对已生成的 token id 序列调用：

```cpp
std::string text = tokenizer.decode(generated_ids, /* skip_special = */ true);
```

底层数值测试应绕过 tokenizer，直接传固定 `int32_t` token ids；HTTP/端到端测试才使用 chat template 和 decode。这样能把 tokenizer 错误与模型 forward 错误分开。

## 15. Generate Loop 伪代码

```cpp
std::string formatted = tokenizer.format_chat_prompt(
        user_text, system_text, /* enable_thinking = */ false);

Qwen3EncodeOptions encode_options;
encode_options.parse_special = true;
encode_options.add_bos = false;
encode_options.add_eos = false;

std::vector<int32_t> prompt_ids = tokenizer.encode(formatted, encode_options);

// sampler is an already initialized Qwen3Engine member.
runtime.reset();
CHECK_RUNTIME(runtime.prefill(
        prompt_ids.data(), prompt_ids.size(), &error));

std::vector<int32_t> effective_stop_ids = sparams.stop_token_ids;
if (tokenizer.eos_token_id() >= 0) {
    append_unique(effective_stop_ids, tokenizer.eos_token_id());
}

// Keep prompt + generated tokens inside the configured context contract.
const int32_t available_new_tokens =
        runtime.max_seq_len() - runtime.current_pos();
const int32_t effective_max_new_tokens = std::max<int32_t>(
        0,
        std::min(sparams.max_new_tokens, available_new_tokens));

std::vector<int32_t> generated_ids;
generated_ids.reserve(effective_max_new_tokens);
std::string finish_reason = "length";

for (int32_t generated = 0;
     generated < effective_max_new_tokens;
     ++generated) {
    int32_t next = -1;
    CHECK_SAMPLER(sampler.sample(
            runtime, sparams, generated_ids, &next, &error));

    if (contains(effective_stop_ids, next)) {
        finish_reason = "stop";
        break;
    }

    generated_ids.push_back(next);

    // The sampled token already counts as one generated token. Do not run
    // another decode when this was the requested final token.
    if (generated + 1 == effective_max_new_tokens) {
        break;
    }

    // Capacity was clamped before the loop; decode(next) writes current_pos.
    CHECK_RUNTIME(runtime.decode(next, &error));
}

std::string output = tokenizer.decode(
        generated_ids, /* skip_special = */ true);
```

这个循环最多产生 `min(max_new_tokens, max_seq_len - prompt_tokens)` 个 token，不是 `max_new_tokens + 1`。在进入循环前收紧上限，可以避免“最后一次 decode 已填满 context，下一轮仍从 logits 多采一个 token”的 off-by-one。若 `effective_max_new_tokens == 0`，直接以 `finish_reason="length"` 返回。`prefill()` 和 `decode()` 都只更新 logits；采样、stop 判断和 detokenize 在外层完成。

## 16. Chunked Prefill

第一版可以限制：

```cpp
prompt_ids.size() <= max_prefill_tokens
```

后面支持长 prompt 时做 chunked prefill：

```cpp
start_pos = 0;
while (start_pos < prompt_len) {
    chunk = prompt_ids[start_pos : start_pos + chunk_size]
    const bool is_last = start_pos + chunk.size() == prompt_len
    forward_tokens(
        chunk,
        start_pos,
        produce_logits=is_last)
    start_pos += chunk.size()
}
```

每个 chunk 都从该 chunk 自己的 embedding 开始，并完整执行全部 36 层。每一层都读取该层已有的历史 KV，并写入当前 chunk 的 KV。跨 chunk 保留的是每层 KV cache，不是上一个 chunk 的 residual/hidden。

前面的 chunk 可以跳过 final norm 和 LM Head；最后一个 chunk 才产生最终 logits。非末 chunk 完成全部 36 层并通过 stream 错误检查后即可提交 `current_pos`；最后一个 chunk 还必须成功完成 final norm 和 LM Head 才能提交。公开 `prefill()` 只有在最后一个 chunk 成功后才设置 `n_prompt=prompt_len` 和 `has_logits=true`。

chunked prefill 的 attention 要能看见历史 cache：

```text
chunk token at local t attends positions 0 ... start_pos + t
```

当前 `prefill_gqa_attention_kernel` 有 `start_pos` 参数，可以按这个方向接。

第一版在 chunked prefill 完成之前，可以明确拒绝：

```cpp
prompt_ids.size() > max_prefill_tokens
```

不能静默截断 prompt。

## 17. Error Handling

每个 HIP/hipBLAS 调用后检查返回值：

```cpp
#define CHECK_HIP(x) ...
#define CHECK_HIPBLAS(x) ...
```

Runtime 初始化时检查：

```cpp
dim % n_head == 0
head_dim == dim / n_head
n_head % n_kv_head == 0
max_seq_len > 0
max_prefill_tokens > 0
all required tensors exist
norm tensors exposed as float*
matrix tensors have expected shape
```

初始化和请求生命周期必须显式检查：

```text
initialize may run once
initialize success -> initialized=true, valid=true, has_logits=false
prefill requires initialized && valid && current_pos==0 && n_prompt==0
prefill success -> n_prompt=total prompt length && has_logits=true
decode requires initialized && valid && has_logits && current_pos>0
reset requires initialized and clears current_pos/n_prompt/has_logits
device_logits/sampler requires has_logits==true
```

每轮生成检查：

```cpp
token_id >= 0 && token_id < vocab_size
start_pos >= 0
n_tokens > 0
n_tokens <= max_prefill_tokens for one chunk
start_pos + n_tokens <= max_seq_len
start_pos + n_tokens <= rope_table_positions
state.valid == true
logits has no NaN/Inf, optional debug
```

边界检查必须在启动第一个 kernel 之前完成。当前 RoPE/KV kernel 的部分越界路径会直接 return，launch 本身仍可能返回成功；如果只在调用后看 `hipGetLastError()`，会得到“成功但 cache 没写完整”的状态。

错误处理采用提交语义：

```text
old_pos = state.current_pos
state.has_logits = false
run all kernels for this forward
check wrapper status
hipStreamSynchronize(stream) once at the forward boundary

success -> state.current_pos = old_pos + n_tokens; state.has_logits = produce_logits
failure -> state.current_pos = old_pos; state.valid = false; state.has_logits = false
```

首版为了可靠提交状态，在一次完整 prefill/decode 的末尾同步当前 stream 一次，而不是每层同步。这样可以在递增 `current_pos` 前捕获异步 HIP 错误；后续若做全异步 runtime，再用 event/future 表达提交完成。

`state.valid=false` 后，除 `reset()` 和析构外的 forward 都应拒绝执行。错误日志至少包含 `layer/op/M/N/K/start_pos/n_tokens`，方便区分 loader、GEMM、attention 和 cache 错误。

## 18. 最小实现顺序

建议按这个顺序做：

1. 写 `Qwen3Runtime` 类，只支持手写 token ids，不接 tokenizer。
2. 以非拥有引用从 `Qwen3GgufModel` 拿固定 config 和 weight pointers。
3. 分配 activation buffers、KV cache、RoPE table。
4. 实现 `prefill(prompt_ids)`，只支持 prompt 长度不超过 `max_prefill_tokens`。
5. 调用现有 `qwen3_z200_launch_greedy_sample()`，只回传一个 token id，实现确定性 GPU greedy。
6. 实现 `decode(token_id)` 单步循环。
7. 用固定 token ids 做 logits/KV 数值验证。
8. 接现有 `Qwen3Tokenizer` 和单轮 chat template。
9. 为 top-k/top-p、temperature、repetition penalty 和调试接完整 logits 的 CPU sampler。
10. 接 `07_qwen3_http_server_contract.md` 的 C++ HTTP server。
11. 再考虑 chunked prefill、GPU logit processors、FP16 KV、paged KV、性能优化。

## 19. 固定生成状态机

实现 Agent 不需要读取或复制 llama.cpp、vLLM 等现有推理框架。当前任务的最小状态机已经完全定义为：

```text
load_qwen3_gguf
Qwen3Tokenizer::load
Qwen3Runtime(model-reference)
runtime.initialize

per request:
  format chat prompt
  tokenizer.encode
  runtime.reset
  runtime.prefill(prompt_ids) -> logits

loop:
  next_token = external sampler
    greedy: GPU argmax -> copy one int32
    other policies/debug: copy full logits -> CPU sampler
  stop check
  append generated id
  if max/context limit: break
  runtime.decode(next_token) -> next logits

tokenizer.decode(all generated ids)
```

关键规则：

- prompt 先整体 prefill。
- 生成阶段每轮只 decode 一个新 token。
- sampler 在 logits 后面，不在 Transformer block 里面。
- detokenizer 在 CPU 侧，并对生成 id 序列调用当前 `decode()` 接口。
- 每轮都要检查 context/KV cache 容量。
- EOG/EOS token 要能停止生成。

本项目明确不引入：

- 不要用 `llama_context`。
- 不要用 `llama_batch` 数据结构，除非你想做兼容层。
- 不要 build GGML graph。
- 不要接 `ggml_backend_sched_graph_compute_async`。
- 不要引入 llama.cpp 的全套 backend/device/buffer 抽象。

当前 runtime 要显式做 llama.cpp 在 `llama_decode()` 里面隐藏起来的事情：

```text
embedding
per-layer attention/ffn forward
KV cache write/read
lm_head
logits
```

## 20. 不使用 GGML 时的责任拆分

这里不 build 通用计算图，也不接 backend scheduler。`Qwen3Runtime` 显式调用：

```text
rms_norm
linear
q/k norm
rope
kv write
attention
linear
add
ffn
lm_head
```

Runtime 输出 logits 后返回。上层 `Qwen3Engine` 再负责 sampler、stop token、detokenizer 和 HTTP response。不要在 Transformer block 内部采样，也不要让 HTTP 层直接修改 `current_pos` 或 KV cache。

好处是简单直接，适合固定 Qwen3-8B + Z200 bring-up。

坏处是：

- 不通用。
- 手动管理 buffer。
- 手动处理 shape 和 dtype。
- 性能优化要自己做。

## 21. 验收标准

第一版 runtime 的结构验收应满足：

```text
1. 输入一串 token ids。
2. prefill 能跑完所有 36 层。
3. 每层 KV cache 写入位置正确。
4. lm_head 得到 [vocab_size] logits。
5. Runtime 只输出 logits；外部 GPU greedy sampler 返回合法 token id，CPU sampler 可作为复杂策略/调试回退。
6. decode 可以连续跑多步，current_pos 正确递增。
7. 遇到 EOS、配置的 stop token 或 max_new_tokens 停止。
8. 不依赖 GGML tensor、GGML graph、GGML backend。
```

仅仅“能跑出 token”不算数值正确。必须增加以下 golden tests：

1. **固定 token ids 的 logits 对齐**：绕过 tokenizer，使用同一个 Q8_0 GGUF，比较最后一行 logits。记录最大绝对/相对误差，并比较 top-1；不要对浮点结果使用 `memcmp`。
2. **Prefill/Decode 等价性**：对 token 序列 `[t0..tN]`，比较一次完整 prefill 的最后 logits，与 `prefill(t0) + decode(t1)..decode(tN)` 的最后 logits。
3. **Chunk 等价性**：实现 chunked prefill 后，比较完整 prefill 与不同 chunk size 的最后 logits。
4. **KV 位置测试**：用递增小张量检查每层 K/V 的 `start_pos + local_t` 索引，确认 GQA 只分配 8 个 KV heads。
5. **Tokenizer 独立测试**：encode/decode/chat template 单独测试，不用生成 token 是否“看起来合理”替代 tokenizer golden。
6. **Sampler 独立测试**：构造已知 device logits，验证 GPU greedy 的 argmax、相同最大值取较小 id、只回传一个 `int32_t`，并验证 EOS/配置 stop 合并和 `max_new_tokens`/context 计数。

调试时先打印：

```text
prompt ids
current_pos before/after prefill
next token id
current_pos before/after each decode
first few logits
whether any buffer has NaN
```

端到端比较时必须使用同一份 Q8_0 GGUF、完全相同的 token ids 和 greedy 参数。先比较 logits/top-1，再比较生成 token；不要只用“同一句 prompt”，因为 chat template、thinking mode 或 BOS 差异会改变实际输入 token。

HTTP 层的额外验收见 `07_qwen3_http_server_contract.md`。
