# Qwen3-8B GGUF Loader Notes

这份文档给后续 agent 参考，用来在当前小型 C++/HIP 推理框架里实现一个最小可用的 Qwen3-8B GGUF loader。

先把 `Qwen3-8B-Q8_0.gguf` 的权重和必要配置读出来，接到现有 Qwen3/HIP forward 与 `qwen3_z200_kernels.hip.cpp`。

## 1. 实现范围

第一版只支持：

- 模型架构：dense Qwen3-8B，不支持 MoE，不支持 VL，不支持 Qwen3Next。
- GGUF 版本：优先支持 GGUF v3。
- tensor 类型：`F32`、`F16`、`Q8_0`。
- 权重加载：按 tensor name 建表，上传到 GPU。
- tokenizer：从同一个 GGUF 的 metadata 构造 `Qwen3TokenizerData`，交给 `tokenizer.cpp` 的 byte-level BPE 实现。

第一版不做：

- 完整 GGUF 写入器。
- 全部 quant 类型，例如 Q4_K、Q5_K、Q6_K。
- llama.cpp 的 backend/scheduler/graph 抽象。
- tokenizer/chat template 的完整兼容。

## 2. GGUF 和当前框架的关系

当前旧 loader 类似：

```cpp
read Config
read all weights as float
memory_map_weights(w, config, d_weight)
```

GGUF 不能这样读，因为每个 tensor 都有自己的名字、shape、type 和 offset：

```text
token_embd.weight              Q8_0 or F16/F32
blk.0.attn_q.weight            Q8_0
blk.0.attn_norm.weight         F32/F16
blk.0.attn_q_norm.weight       F32/F16
output_norm.weight             F32/F16
output.weight                  Q8_0, required for this fixed Qwen3-8B
```

所以 GGUF loader 的核心是：

```text
解析 metadata
解析 tensor info table
按 tensor.name 找权重
按 tensor.type 选择保存格式
按 tensor.offset 读取数据并上传 GPU
从 tokenizer metadata 构造 Qwen3TokenizerData
```

不要再生成或读取旧式 `tokenizer.bin`。Tokenizer 与权重 Loader 共用同一次 GGUF metadata 解析，Tokenizer 本身不重复打开 GGUF 文件。

## 3. GGUF 文件布局

GGUF 文件结构：

```text
magic:      "GGUF"   4 bytes
version:    uint32
n_tensors:  uint64/int64, 8 bytes
n_kv:       uint64/int64, 8 bytes

kv table:
  key string
  gguf_type int32
  value bytes

tensor info table:
  name string
  n_dims uint32
  dims[n_dims] uint64/int64
  ggml_type int32
  offset uint64, relative to data blob

tensor data blob:
  starts at aligned file offset
```

String 编码：

```text
uint64 length
length bytes, no trailing '\0'
```

默认 alignment 是 32，如果 metadata 里有 `general.alignment`，使用它：

```cpp
data_offset = align_up(current_file_offset, alignment);
real_tensor_file_offset = data_offset + tensor.offset;
```

## 4. 需要支持的类型

GGUF metadata 的 value type 是 `gguf_type`，tensor data 的类型是 `ggml_type`。不要混用这两个 enum。

第一版需要的 `ggml_type`：

```cpp
enum GgmlType : int32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q8_0 = 8,
};
```

第一版需要的 `gguf_type`：

```cpp
enum GgufType : int32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};
```

Bool 在 GGUF 里按 int8 存。

Array 编码：

```text
array element type: gguf_type int32
array length:       uint64
array data:         element bytes
```

如果 array element 是 string，每个元素仍按 `uint64 len + bytes` 读取。

## 5. Q8_0 存储格式

`Q8_0` 每 32 个原始元素一个 block：

```cpp
struct BlockQ8_0 {
    half d;        // FP16 scale, 2 bytes
    int8_t qs[32]; // 32 signed quants
};
```

每个 block 是 34 bytes，必须和 `qwen3_z200_kernels.hip.cpp` 里的 `BlockQ8_0` byte layout 一致。

Q8_0 tensor size 不能按 `numel` 算，要按 row 算：

```cpp
size_t ggml_row_size_q8_0(uint64_t ne0) {
    // ne0 是最快维，也就是矩阵的 K 维
    assert(ne0 % 32 == 0);
    return (ne0 / 32) * 34;
}

size_t tensor_nbytes_q8_0(const std::vector<uint64_t> & ne) {
    size_t rows = 1;
    for (size_t i = 1; i < ne.size(); ++i) {
        rows *= ne[i];
    }
    return rows * ggml_row_size_q8_0(ne[0]);
}
```

F32/F16:

```cpp
numel = product(ne)
F32 bytes = numel * 4
F16 bytes = numel * 2
```

## 6. 建议数据结构

```cpp
enum class TensorStorage {
    F32,
    F16,
    Q8_0,
};

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> ne; // GGML order: ne[0] fastest dim
    int32_t ggml_type;
    uint64_t offset;          // relative to data blob
    size_t nbytes;
    void * d_ptr = nullptr;
};

struct Qwen3Config {
    int dim;
    int hidden_dim;
    int n_layer;
    int n_head;
    int n_kv_head;
    int head_dim;
    int vocab_size;

    // Runtime KV cache cap. Can be smaller than GGUF train context length.
    int seq_len;

    float rope_theta;
    float rms_eps;
};

struct Qwen3LayerWeights {
    GgufTensor * attn_norm;
    GgufTensor * wq;
    GgufTensor * wk;
    GgufTensor * wv;
    GgufTensor * wo;
    GgufTensor * q_norm;
    GgufTensor * k_norm;
    GgufTensor * ffn_norm;
    GgufTensor * ffn_gate;
    GgufTensor * ffn_up;
    GgufTensor * ffn_down;
};

struct Qwen3Weights {
    GgufTensor * token_embd;
    GgufTensor * output_norm;
    GgufTensor * output; // required: tie_word_embeddings == false
    std::vector<Qwen3LayerWeights> layers;
};

struct Qwen3GgufModel {
    Qwen3Config config;
    Qwen3Weights weights;
    std::unordered_map<std::string, GgufTensor> tensors;

    Qwen3GgufModel() = default;
    ~Qwen3GgufModel();
    Qwen3GgufModel(const Qwen3GgufModel&) = delete;
    Qwen3GgufModel& operator=(const Qwen3GgufModel&) = delete;

    Qwen3GgufModel(Qwen3GgufModel&& other) noexcept;
    Qwen3GgufModel& operator=(Qwen3GgufModel&& other) noexcept;

    // Move operations must rebind all weight views after moving tensors.
    void rebind_weight_views();
};
```

`Qwen3Weights` 里的指针只是指向 `tensors` 元素的 view，不拥有 device memory。model 禁止默认复制；复制 unordered map 后，view 仍可能指向旧 model。填完全部 tensor 后再调用 `rebind_weight_views()`，之后不要继续插入导致 rehash。更稳妥的实现是让 weights 保存稳定索引/名称，在访问时查 tensor。

`Qwen3GgufModel` 是 GPU 权重 owner，其析构路径必须且只执行一次 `hipFree`。move 后要让源对象失去 device pointer 所有权，并在目标对象重新绑定 views，避免 double-free。`Qwen3Runtime` 只保存对 model 的非拥有引用，model 必须比 runtime 活得更久。

## 7. Qwen3-8B config

当前任务只支持固定 Qwen3-8B dense。Loader 读取 metadata 的目的，是验证输入文件和固定 runtime config 完全匹配；metadata 不是把 runtime 悄悄切换成其他 Qwen3 变体的配置入口。

需要读取的 key：

```text
general.architecture
qwen3.context_length
qwen3.embedding_length
qwen3.block_count
qwen3.feed_forward_length
qwen3.attention.head_count
qwen3.attention.head_count_kv
qwen3.attention.layer_norm_rms_epsilon
qwen3.rope.freq_base
qwen3.vocab_size
```

Qwen3-8B dense 固定值：

```cpp
config.dim        = 4096;
config.hidden_dim = 12288;
config.n_layer    = 36;
config.n_head     = 32;
config.n_kv_head  = 8;
config.head_dim   = 128;       // dim / n_head
config.vocab_size = 151936;
config.rope_theta = 1000000.0f;
config.rms_eps    = 1e-6f;
```

校验规则：

```text
general.architecture == "qwen3"
embedding_length      == 4096
feed_forward_length   == 12288
block_count           == 36
head_count             == 32
head_count_kv          == 8
vocab_size             == 151936
rope.freq_base          ~= 1000000.0
layer_norm_rms_epsilon  ~= 1e-6
```

整数必须精确相等；浮点 metadata 用小容差比较。关键字段缺失或不一致时打印实际值和期望值并失败，不使用另一个模型的 fallback。

`seq_len` 建议作为运行时参数，不要直接等于 GGUF 的训练 context。16GB 卡上先用 2048、4096 或 8192 跑通，否则 KV cache 会很大。

```cpp
config.seq_len = std::min(user_runtime_ctx, gguf_context_length);
```

## 8. 交给 Runtime 构造的 RoPE sin/cos table

RoPE table 不是 Qwen3 的大矩阵权重，不从 GGUF tensor data 里加载。Loader 只读取并校验 `qwen3.rope.freq_base`；`Qwen3Runtime::initialize()` 根据固定 config 和运行时 `max_seq_len` 在 CPU 端生成 table，再上传到 GPU。
参考代码：

```cpp
void build_rope_table(
        float * d_sin,
        float * d_cos,
        int head_dim,
        int seq_len,
        float rope_theta) {
    const int half = head_dim / 2;
    std::vector<float> sin_table((size_t) seq_len * half);
    std::vector<float> cos_table((size_t) seq_len * half);

    for (int pos = 0; pos < seq_len; ++pos) {
        for (int j = 0; j < half; ++j) {
            const float inv_freq = powf(rope_theta, -2.0f * j / (float) head_dim);
            const float angle = pos * inv_freq;
            sin_table[(size_t) pos * half + j] = sinf(angle);
            cos_table[(size_t) pos * half + j] = cosf(angle);
        }
    }

    hipMemcpy(d_sin, sin_table.data(), sin_table.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_cos, cos_table.data(), cos_table.size() * sizeof(float), hipMemcpyHostToDevice);
}
```

注意点：

- `n_head` 不参与 RoPE table 生成，table 只和 `seq_len`、`head_dim`、`rope_theta` 有关。
- 当前 `qwen3_z200::rope_kernel` 按 `pos * (head_dim / 2) + pair` 读取 cos/sin，所以 table shape 应该是 `[seq_len, head_dim / 2]`。
- prefill 时位置从 `start_pos` 开始；decode 时位置是当前 token 的绝对 pos。
- 如果后面支持 YaRN/LongRoPE，再扩展 `rope_freq_scale` 等参数；第一版 Qwen3-8B dense 先用基础 RoPE 公式。

## 9. Qwen3 tensor 名称

全局 tensor：

```text
token_embd.weight
output_norm.weight
output.weight          // required for fixed Qwen3-8B
```

每层 tensor，`%d` 是 layer index：

```text
blk.%d.attn_norm.weight
blk.%d.attn_q.weight
blk.%d.attn_k.weight
blk.%d.attn_v.weight
blk.%d.attn_output.weight
blk.%d.attn_q_norm.weight
blk.%d.attn_k_norm.weight
blk.%d.ffn_norm.weight
blk.%d.ffn_gate.weight
blk.%d.ffn_up.weight
blk.%d.ffn_down.weight
```

固定 Qwen3-8B 的 `tie_word_embeddings=false`。如果 `output.weight` 不存在，Loader 必须报错：

```cpp
throw std::runtime_error(
        "missing output.weight for Qwen3-8B with untied embeddings");
```

不能 fallback 到 `token_embd.weight`；两者数学含义和训练参数不同。

## 10. Tensor shape 规则

GGML tensor 维度顺序是 `ne[0]` fastest dim。对于 2D linear weight：

```text
tensor.ne = {K, N}
```

在内存里可以按 row-major `W[N, K]` 理解。调用现有 Q8_0 linear 时：

```cpp
k = tensor.ne[0]; // input dim
n = tensor.ne[1]; // output dim
m = n_tokens;
```

Qwen3 dense 主要 shape：

```text
token_embd.weight              {dim, vocab_size}
output_norm.weight             {dim}
output.weight                  {dim, vocab_size}

blk.i.attn_norm.weight         {dim}
blk.i.attn_q.weight            {dim, n_head * head_dim}
blk.i.attn_k.weight            {dim, n_kv_head * head_dim}
blk.i.attn_v.weight            {dim, n_kv_head * head_dim}
blk.i.attn_output.weight       {n_head * head_dim, dim}
blk.i.attn_q_norm.weight       {head_dim}
blk.i.attn_k_norm.weight       {head_dim}

blk.i.ffn_norm.weight          {dim}
blk.i.ffn_gate.weight          {dim, hidden_dim}
blk.i.ffn_up.weight            {dim, hidden_dim}
blk.i.ffn_down.weight          {hidden_dim, dim}
```

对应现有 forward 的线性层：

```text
Q:       x[*, dim]        * attn_q.weight       -> [*, n_head * head_dim]
K:       x[*, dim]        * attn_k.weight       -> [*, n_kv_head * head_dim]
V:       x[*, dim]        * attn_v.weight       -> [*, n_kv_head * head_dim]
O:       attn[*, dim]     * attn_output.weight  -> [*, dim]
gate:    x[*, dim]        * ffn_gate.weight     -> [*, hidden_dim]
up:      x[*, dim]        * ffn_up.weight       -> [*, hidden_dim]
down:    swiglu[*, hdim]  * ffn_down.weight     -> [*, dim]
lm_head: x[*, dim]        * output.weight       -> [*, vocab_size]
```

## 11. 类型分发策略

loader 必须保留 tensor 的 `ggml_type`。

```cpp
if (tensor.type == GGML_TYPE_Q8_0) {
    // Keep bytes as BlockQ8_0 on GPU.
    // Linear uses qwen3_z200_q8_linear_fp32.
}

if (tensor.type == GGML_TYPE_F32) {
    // Upload as float.
    // Norm weights can be used directly by RMSNorm kernels.
}

if (tensor.type == GGML_TYPE_F16) {
    // Option A: convert to float on CPU, upload float for norm kernels.
    // Option B: keep half and add F16-aware norm kernels later.
    // First version should convert 1D norm weights to float.
}
```

重要：`Qwen3-8B-Q8_0.gguf` 不是所有 tensor 都是 Q8_0。通常大矩阵是 Q8_0，norm 这类 1D tensor 是 F16/F32。不能把 norm 权重当 `BlockQ8_0` 读。

## 12. 加载流程建议

```cpp
Qwen3GgufModel load_qwen3_gguf(
        const std::string & path,
        int runtime_ctx,
        hipStream_t stream) {
    open file
    read header
    read kv metadata
    read tensor infos
    compute data_offset
    construct fixed Qwen3-8B config
    validate GGUF metadata exactly matches the fixed config
    fill Qwen3TokenizerData from tokenizer.ggml.* metadata
    tokenizer.load(tokenizer_data)
    validate architecture == "qwen3"
    validate config roughly matches Qwen3-8B
    leave RoPE table construction to Qwen3Runtime initialization

    for each tensor:
        compute nbytes by type and shape
        seek(data_offset + tensor.offset)
        read host bytes
        maybe convert F16 1D norm -> F32
        hipMalloc tensor.d_ptr
        hipMemcpy host -> device

    map_required_qwen3_tensors(model)
    rebind/validate all tensor views after the tensor map is stable
    validate all required tensors exist
    return model
}
```

建议先分两步写：

1. `parse_gguf_meta(path)`：只打印 metadata 和 tensor table，不上传 GPU。
2. `load_qwen3_gguf(path)`：在确认名字/shape/type 都对后，再做 GPU 上传和权重映射。

## 13. 验证 checklist

加载后先打印：

```text
architecture = qwen3
dim = 4096
n_layer = 36
n_head = 32
n_kv_head = 8
head_dim = 128
hidden_dim = 12288
vocab_size = 151936
rope_theta = 1000000
rms_eps = 1e-6
runtime seq_len = ...
```

每层检查：

```cpp
attn_norm.ne == {dim}
wq.ne        == {dim, n_head * head_dim}
wk.ne        == {dim, n_kv_head * head_dim}
wv.ne        == {dim, n_kv_head * head_dim}
wo.ne        == {n_head * head_dim, dim}
q_norm.ne    == {head_dim}
k_norm.ne    == {head_dim}
ffn_norm.ne  == {dim}
ffn_gate.ne  == {dim, hidden_dim}
ffn_up.ne    == {dim, hidden_dim}
ffn_down.ne  == {hidden_dim, dim}
```

类型检查：

```cpp
is_matrix(t) -> allow Q8_0/F16/F32
is_norm(t)   -> require F32 or F16, then expose as float*
```

RoPE 检查：

```cpp
head_dim == dim / n_head
rope_table_elements == runtime_seq_len * head_dim / 2
rope_theta == value from qwen3.rope.freq_base
```

## 14. 和现有 kernel 的对接

当前 `qwen3_z200_kernels.hip.cpp` 已经有：

```text
Q8_0 embedding lookup
Q8_0 dequant to FP16
Q8_0 linear wrapper: qwen3_z200_q8_linear_fp32
RMSNorm
Q/K per-head RMSNorm
RoPE
KV cache write
prefill GQA attention
decode GQA attention
SwiGLU
add/add_inplace
greedy argmax sampler primitive
```

当前 kernel 文件已经导出 `qwen3_z200_launch_greedy_sample()`。Loader 不调用采样器；runtime forward 产出 `[151936]` FP32 device logits 后，由外部 sampler 在同一条 HIP stream 上调用 GPU argmax，并只把一个 token id 拷回 CPU。完整 logits 复制只用于 top-k/top-p、repetition penalty 或 correctness 调试。

loader 只负责把 tensor 指针准备好。forward 根据 tensor type 选择：

```cpp
if (weight.type == Q8_0) {
    qwen3_z200_q8_linear_fp32(...);
} else if (weight.type == F16/F32) {
    // Later: hipBLAS GEMM directly, or convert once.
}
```

Norm 权重给 `qwen3_z200_launch_rms_norm` 和 `qwen3_z200_launch_per_head_rms_norm` 时，第一版最好统一成 device float pointer。

## 15. 常见坑

- 不要把 GGUF 的 `gguf_type` 当 tensor 的 `ggml_type`。
- 不要假设 `Q8_0.gguf` 里所有 tensor 都是 Q8_0。
- Q8_0 不能按 `numel` 字节读，要按 row size 读。
- tensor offset 是相对 data blob，不是相对文件开头。
- data blob 起点要按 `general.alignment` 对齐。
- `seq_len` 是运行时 KV cache 上限，不一定等于 `qwen3.context_length`。
- 固定 Qwen3-8B 的 `tie_word_embeddings=false`，`output.weight` 缺失必须报错，不能 fallback 到 `token_embd.weight`。
- GGML 2D shape `{K, N}` 在 GEMM 里对应 row-major `W[N, K]`。
- F16 norm 权重如果不转换，现有 float RMSNorm kernel 会读错。
- RoPE 不要硬编码 `10000.0f`；Qwen3-8B 要用 `qwen3.rope.freq_base`，通常是 `1000000.0f`。

## 16. Tokenizer metadata 交接

GGUF parser 读取完 metadata 后，应构造 `tokenizer.hpp` 定义的：

```cpp
Qwen3TokenizerData tokenizer_data;
tokenizer_data.model          = get_string("tokenizer.ggml.model");
tokenizer_data.pre_tokenizer  = get_string("tokenizer.ggml.pre");
tokenizer_data.tokens         = get_string_array("tokenizer.ggml.tokens");
tokenizer_data.merges         = get_string_array("tokenizer.ggml.merges");
tokenizer_data.token_types    = get_i32_array("tokenizer.ggml.token_type");
tokenizer_data.bos_token_id   = get_optional_i32("tokenizer.ggml.bos_token_id", -1);
tokenizer_data.eos_token_id   = get_optional_i32("tokenizer.ggml.eos_token_id", -1);
tokenizer_data.pad_token_id   = get_optional_i32("tokenizer.ggml.padding_token_id", -1);
tokenizer_data.add_bos_token  = get_optional_bool("tokenizer.ggml.add_bos_token", false);

Qwen3Tokenizer tokenizer;
std::string tokenizer_error;
if (!tokenizer.load(tokenizer_data, &tokenizer_error)) {
    throw std::runtime_error(tokenizer_error);
}
```

第一版必须校验 `tokenizer.ggml.model == "gpt2"`，并确保每个 token id 小于 embedding 的 `vocab_size=151936`。Tokenizer token 数可以小于 padded embedding 行数，不能要求二者严格相等。
