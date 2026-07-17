# 原生权重加载契约

> 权威级别：Checkpoint 解析、TP 切片、DType 转换和设备发布的强制契约。

## 1. 边界

权重加载器只负责把目标 Checkpoint 中的张量转换为模型声明的
`WeightSpec`。它不得猜测模型结构、静默补零、跳过未知必需权重，或在请求
路径临时加载权重。模型配置、Tokenizer 和 Chat Template 必须先完成校验。

```cpp
struct WeightSpec {
  std::string checkpoint_name;
  std::string runtime_name;
  Shape global_shape;
  DType checkpoint_dtype;
  DType runtime_dtype;
  ShardRule shard;
  PackRule pack;
  bool required = true;
};

Result<ModelWeights> LoadModelWeights(const ModelConfig& config,
                                      const WeightManifest& manifest,
                                      const RankContext& rank,
                                      Backend& backend);
```

## 2. Safetensors 解析

- 先读取 Header Length，再在有界范围内解析 JSON Header；
- 每个 Tensor 的 `[begin, end)` 必须位于文件 Data Region 内；
- Offset、Shape 乘积和字节数使用溢出安全运算；
- Multi-shard Index 中每个必需 Key 必须恰好解析到一个文件；
- Memory Mapping 的生命周期必须覆盖所有 Host View，禁止返回悬空指针；
- 不支持的压缩、量化或 DType 必须在分配设备内存前失败。

不得把完整 Checkpoint 展开为 `std::vector<float>`。允许的Host内存包括
只读Mapping、受限的Pinned Staging Buffer和小型转换Workspace。

## 3. 名称映射与打包

名称映射必须由模型家族提供，而不是使用模糊后缀匹配：

```cpp
class WeightMapper {
 public:
  virtual Result<WeightTarget> Resolve(std::string_view checkpoint_name) const = 0;
  virtual Status ValidateComplete(Span<const LoadedWeight> loaded) const = 0;
};
```

Q/K/V和Gate/Up打包时必须声明顺序、目标Offset和Local Shape。写入每个
Slice前校验目标范围没有重叠；完成后验证整个目标Buffer无未初始化区间。

## 4. TP切片

切片只执行一次。Loader必须区分Checkpoint已经按Rank预切片和需要从Global
Tensor切片两种模式，并在Load Report中记录选择。

- Column Parallel：沿语义Output Feature切片；
- Row Parallel：沿语义Input Feature切片；
- Q/K/V分别计算Local Head范围后再打包；
- GQA必须处理KV Head Partition或Replication；
- Vocab Shard必须处理不可整除边界或明确拒绝；
- Norm、Bias、Router参数必须明确Replicated或Sharded。

任何Global Shape不能被所选Shard规则满足时，必须在Upload前返回错误。

## 5. DType与发布

Checkpoint DType、Runtime Weight DType和计算DType是三个独立字段。转换必须：

1. 有Backend能力Probe；
2. 有明确的舍入/量化规则；
3. 有小张量数值对比；
4. 在Load Report中记录源/目标DType；
5. 不得把不支持的类型静默改成FP32。

设备权重只在所有Header、Shape和映射验证成功后发布。单个Layer可采用事务式
加载：分配临时Storage、完成Upload和校验，再移动到Model所有权中。失败时释放
当前事务的所有资源，不留下半初始化模型。

## 6. 加载证据

`weight-load-report.json`至少记录模型ID、Checkpoint文件、每个Runtime Weight的
Global/Local Shape、DType、Shard规则、字节数、Rank、缺失/额外Key和总设备字节。
不得记录权重值或泄露任意文件内容。

## 7. 强制测试

- Header截断、Offset越界、Shape溢出和重复Key非零失败；
- Tiny Safetensors的值、Shape和DType逐项一致；
- QKV和Gate/Up打包顺序Golden Test；
- TP=1与TP=N的Local Slice重组后等于Global Tensor；
- 不支持量化格式在设备分配前失败；
- 注入Upload失败后无设备内存泄漏；
- 真实Checkpoint所有Required Key恰好加载一次。

