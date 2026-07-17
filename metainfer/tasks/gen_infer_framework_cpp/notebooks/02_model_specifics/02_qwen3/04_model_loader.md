# Qwen3 配置、Tokenizer 与 Safetensors 加载

先读：`00_contracts/qwen3_model_contracts.md`和
`00_contracts/weight_loading_contracts.md`。前者定义Qwen模型语义，后者定义原生
Checkpoint解析、TP切片和设备发布边界。

加载器的目标不是“把字节复制到 GPU”，而是在分配显存前证明：模型类型受支持、每个 Key/Shape/DType 正确、TP Slice 正确、Tokenizer 与 Checkpoint 匹配。

## 1. 分阶段加载流程

```text
读取 config.json
-> 解析/校验 Qwen3 Dense 或 MoE
-> 读取 tokenizer metadata/chat template
-> 读取 model.safetensors.index.json（若存在）
-> 读取每个 shard 的 Safetensors Header
-> 建立全局 Tensor Catalog
-> 根据语义 WeightSpec 检查 Key/Shape/DType
-> 计算每 Rank Slice 和内存预算
-> 分配 Device Storage
-> 按 Layer/Chunk 读取或 mmap、必要时转换 Layout
-> 异步上传并校验
-> 发布只读 ModelWeights
```

在 Catalog 和内存预算完成之前禁止批量 `hipMalloc`。

## 2. Qwen3Config 解析

可以使用经过许可的 C++ JSON Parser，但必须把动态 JSON 转成强类型结构：

```cpp
Result<Qwen3Config> ParseQwen3Config(const Json& root) {
  Qwen3Config cfg;
  ASSIGN_OR_RETURN(cfg.hidden_size, RequiredInt(root, "hidden_size"));
  ASSIGN_OR_RETURN(cfg.intermediate_size,
                   RequiredInt(root, "intermediate_size"));
  ASSIGN_OR_RETURN(cfg.num_hidden_layers,
                   RequiredInt(root, "num_hidden_layers"));
  ASSIGN_OR_RETURN(cfg.num_attention_heads,
                   RequiredInt(root, "num_attention_heads"));
  ASSIGN_OR_RETURN(cfg.num_key_value_heads,
                   RequiredInt(root, "num_key_value_heads"));
  ASSIGN_OR_RETURN(cfg.vocab_size, RequiredInt(root, "vocab_size"));
  ASSIGN_OR_RETURN(cfg.rms_norm_eps,
                   RequiredDouble(root, "rms_norm_eps"));
  ASSIGN_OR_RETURN(cfg.rope_theta,
                   OptionalDouble(root, "rope_theta", /*default=*/10000.0));

  cfg.head_dim = OptionalInt(root, "head_dim").value_or(
      cfg.hidden_size / cfg.num_attention_heads);
  ASSIGN_OR_RETURN(cfg.max_position_embeddings,
                   RequiredInt(root, "max_position_embeddings"));
  ASSIGN_OR_RETURN(cfg.tie_word_embeddings,
                   OptionalBool(root, "tie_word_embeddings", false));

  // 仅当相关字段存在时声明 MoE；不能凭模型目录名称猜测。
  ASSIGN_OR_RETURN(cfg.num_experts,
                   OptionalInt(root, "num_experts", 0));
  ASSIGN_OR_RETURN(cfg.num_experts_per_tok,
                   OptionalInt(root, "num_experts_per_tok", 0));

  RETURN_IF_ERROR(ValidateQwen3Config(cfg, /*tp_size=*/1));
  return cfg;
}
```

`rope_theta` 默认值只能在 Qwen3 官方配置语义允许时使用，优先读取真实字段。所有 JSON Helper 返回 `Result<T>`，错误统一通过 `ASSIGN_OR_RETURN` 传播。

## 3. Safetensors 文件结构

Safetensors 文件由以下部分组成：

```text
[8-byte little-endian uint64 header_size]
[header_size bytes JSON header]
[raw tensor data region]
```

Header 中的 `data_offsets` 相对于 Raw Data Region 起点。必须使用 Checked 64-bit Arithmetic。

```cpp
struct SafeTensorEntry {
  std::string name;
  DType dtype = DType::kUnknown;
  std::vector<std::int64_t> shape;
  std::uint64_t data_begin = 0;  // absolute file offset after validation
  std::uint64_t data_end = 0;
};

struct SafeTensorFile {
  std::filesystem::path path;
  std::uint64_t file_size = 0;
  std::uint64_t data_region_begin = 0;
  std::unordered_map<std::string, SafeTensorEntry> tensors;
};
```

安全读取 Header Size：

```cpp
Result<std::uint64_t> ReadHeaderSize(std::ifstream& input,
                                     std::uint64_t file_size) {
  std::array<std::uint8_t, 8> bytes{};
  if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
    return DataLoss("truncated safetensors header length");
  }
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
  }
  constexpr std::uint64_t kMaxHeaderBytes = 128ULL << 20;
  if (value == 0 || value > kMaxHeaderBytes || value > file_size - 8) {
    return DataLoss("invalid safetensors header length");
  }
  return value;
}
```

必须继续验证：JSON Shape 为非负整数、`begin <= end`、区间在文件内、`end-begin == NumElements(shape)*SizeOf(dtype)`、同一文件区间不非法重叠。

## 4. 多 Shard Index

当存在 `model.safetensors.index.json`：

```cpp
struct ShardLocation {
  std::filesystem::path file;
  std::string tensor_name;
};

Result<std::unordered_map<std::string, ShardLocation>>
LoadWeightMap(const std::filesystem::path& model_dir,
              const Json& index) {
  ASSIGN_OR_RETURN(const Json* map, RequiredObject(index, "weight_map"));
  std::unordered_map<std::string, ShardLocation> result;
  for (const auto& [name, file_json] : map->items()) {
    ASSIGN_OR_RETURN(std::string file_name, JsonString(file_json));
    std::filesystem::path relative(file_name);
    if (relative.is_absolute() || ContainsParentTraversal(relative)) {
      return InvalidArgument("weight shard escapes MODEL_DIR");
    }
    auto [it, inserted] = result.emplace(
        name, ShardLocation{model_dir / relative, name});
    if (!inserted) return DataLoss("duplicate tensor in weight_map: " + name);
  }
  return result;
}
```

打开文件后仍需使用 Canonical/Relative Check 确保路径位于 `MODEL_DIR` 内，并拒绝 Symbolic Link Escape（根据部署安全策略）。

## 5. Qwen3 WeightSpec 表

不要把 Key Mapping 分散在每个 Layer 构造函数中。建立一个可枚举、可测试的语义表：

```cpp
enum class WeightRole {
  kTokenEmbedding,
  kAttentionQ,
  kAttentionK,
  kAttentionV,
  kAttentionOutput,
  kAttentionQNorm,
  kAttentionKNorm,
  kInputNorm,
  kPostAttentionNorm,
  kMlpGate,
  kMlpUp,
  kMlpDown,
  kFinalNorm,
  kLmHead,
};

struct WeightSpec {
  WeightRole role;
  std::string checkpoint_key;
  std::vector<std::int64_t> expected_shape;
  ShardPolicy shard_policy;
  bool required = true;
};
```

Dense Layer 的典型 Key 模板：

```cpp
std::vector<WeightSpec> BuildDenseLayerSpecs(const Qwen3Config& cfg,
                                             std::int64_t layer) {
  const std::string p = "model.layers." + std::to_string(layer) + ".";
  return {
    {WeightRole::kAttentionQ, p + "self_attn.q_proj.weight", {/*...*/},
     ShardPolicy::kColumn, true},
    {WeightRole::kAttentionK, p + "self_attn.k_proj.weight", {/*...*/},
     ShardPolicy::kColumnOrReplicatedKv, true},
    {WeightRole::kAttentionV, p + "self_attn.v_proj.weight", {/*...*/},
     ShardPolicy::kColumnOrReplicatedKv, true},
    {WeightRole::kAttentionOutput, p + "self_attn.o_proj.weight", {/*...*/},
     ShardPolicy::kRow, true},
    {WeightRole::kAttentionQNorm, p + "self_attn.q_norm.weight", {/*...*/},
     ShardPolicy::kHeadLocal, true},
    {WeightRole::kAttentionKNorm, p + "self_attn.k_norm.weight", {/*...*/},
     ShardPolicy::kHeadLocal, true},
    {WeightRole::kMlpGate, p + "mlp.gate_proj.weight", {/*...*/},
     ShardPolicy::kColumn, true},
    {WeightRole::kMlpUp, p + "mlp.up_proj.weight", {/*...*/},
     ShardPolicy::kColumn, true},
    {WeightRole::kMlpDown, p + "mlp.down_proj.weight", {/*...*/},
     ShardPolicy::kRow, true},
  };
}
```

`{/*...*/}` 必须由 Config 计算真实 Shape，不能保留占位符。Agent 完成实现时应提供具体 Shape Function 和 Unit Test。

## 6. TP Slice

```cpp
Result<TensorSlice> ComputeContiguousShard(const SafeTensorEntry& source,
                                           int axis,
                                           int rank,
                                           int world_size) {
  if (axis < 0 || axis >= static_cast<int>(source.shape.size())) {
    return InvalidArgument("invalid shard axis");
  }
  const std::int64_t global = source.shape[axis];
  if (global % world_size != 0) {
    return InvalidArgument("tensor dimension is not divisible by TP size");
  }
  TensorSlice result;
  result.shape = source.shape;
  result.shape[axis] = global / world_size;
  result.axis_begin = rank * result.shape[axis];
  result.axis_end = result.axis_begin + result.shape[axis];
  return result;
}
```

只有当目标 Axis 在文件 Layout 中连续时，才能直接读取一个 Byte Range；否则需要 Row-wise Read 或显式 Pack Buffer，不能假设任意 Axis Slice 连续。

## 7. Upload 和发布

```cpp
class WeightLoader {
 public:
  Result<ModelWeights> Load(const ModelCatalog& catalog,
                            const Qwen3Config& config,
                            const RankContext& rank,
                            Backend& backend);

 private:
  Result<TensorStorage> LoadOne(const WeightSpec& spec,
                                const SafeTensorEntry& source,
                                const TensorSlice& slice,
                                BackendStream stream);
};
```

上传应分 Layer/Chunk，控制 Host Staging Buffer 峰值。所有 Upload Event 完成、Key 数量匹配、抽样值 Finite 后，才把 `ModelWeights` 发布给 ModelRunner。

## 8. Tokenizer 边界

推荐接口：

```cpp
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  virtual Result<std::vector<std::int32_t>> Encode(
      std::string_view utf8,
      bool add_special_tokens) const = 0;
  virtual Result<std::string> Decode(
      Span<const std::int32_t> token_ids,
      bool skip_special_tokens) const = 0;
  virtual Result<std::string> ApplyChatTemplate(
      Span<const ChatMessage> messages,
      bool add_generation_prompt) const = 0;
};
```

两种可接受方案：

- 链接许可兼容的 C/C++ Tokenizer Library 并加载 Checkpoint Asset；
- 通过离线 Python 转换为稳定 Native Artifact，服务进程只读取 Artifact。

Python 不得常驻 Serving。转换 Artifact 必须包含源 Asset Hash/Version，源变化后拒绝陈旧 Artifact。

## 9. 测试矩阵

```text
config 缺字段、错误类型、Dense/MoE 不匹配
Safetensors Header 截断/超大/字节序
data_offsets 越界/重叠/长度与 Shape 不符
Index 路径穿越、重复/缺失 Key
DType/Shape 错误
QKV/Gate-Up Pack Order
TP Slice Axis、非整除、非连续读取
Tokenizer 中文/英文/特殊 Token/Chat Template Round-trip
第 N 个 Weight Upload 失败后的 RAII 回滚
```
