# C++ Attention 与 Paged KV Cache 强制契约

> 权威级别：`gen-infer-framework-cpp` 的强制契约。
>
> 实现指南：`01_framework_design/03_kv_cache.md`和`03_operators/01_attention_ops.md`。

本文只定义原生 C++ Runtime 必须满足的接口、所有权、Shape、同步和验证规则。不得把 Python、PyTorch、Python FlashAttention Wrapper 或逐请求子进程作为实现路径。

## 1. 适用范围

本契约覆盖：

- Qwen3 Dense/MoE 的 GQA Attention；
- Prefill 与单步/多步 Decode；
- Paged KV Cache 的物理布局和逻辑 Block Table；
- 单卡与 Tensor Parallel 下的本地 KV Head；
- HIP/CUDA/厂商 Attention Kernel 的统一 Backend 接口；
- 请求完成、取消、失败和进程退出时的显存回收。

模型数学语义必须来自 checkpoint 的 `config.json`，硬件能力必须来自 `hardware_profile.json`。不得根据营销型号猜测 Head、DType、Block Size 或 Kernel 支持情况。

## 2. Qwen3 Attention Shape 契约

设：

```text
Tq = 本次 Step 的 Query Token 数
Tk = 当前可见的 Key/Value Token 数
Hq = 本 Rank 的 Query Head 数
Hkv = 本 Rank 的 KV Head 数
D = head_dim
L = decoder layer 数
B = KV Block 数
S = 每个 KV Block 的 Token 数
```

Backend 边界上的逻辑 Shape 为：

```text
Q             [Tq, Hq, D]
K_current     [Tq, Hkv, D]
V_current     [Tq, Hkv, D]
slot_mapping  [Tq] int32/int64
block_table   [num_sequences, max_blocks_per_sequence] int32
context_lens  [num_sequences] int32
output        [Tq, Hq, D]
```

初始化时必须验证：

```cpp
struct AttentionConfig {
  std::int32_t num_attention_heads = 0;
  std::int32_t num_kv_heads = 0;
  std::int32_t head_dim = 0;
  std::int32_t tp_size = 1;
  DType activation_dtype = DType::kUnknown;
  DType kv_dtype = DType::kUnknown;
};

Status ValidateAttentionConfig(const AttentionConfig& cfg) {
  if (cfg.num_attention_heads <= 0 || cfg.num_kv_heads <= 0 ||
      cfg.head_dim <= 0 || cfg.tp_size <= 0) {
    return InvalidArgument("attention dimensions must be positive");
  }
  if (cfg.num_attention_heads % cfg.num_kv_heads != 0) {
    return InvalidArgument("Q heads must be divisible by KV heads");
  }
  return Status::Ok();
}
```

如果 `num_kv_heads` 不能被 TP Size 整除，必须在计划中明确采用 KV Head Replication 或经过验证的 Group Partition；不得通过整数除法静默丢弃 Head。

## 3. Block Size 与 Kernel 能力

`block_size` 是 Runtime 配置，不是固定常量。候选值必须同时满足：

- Addressing Kernel 支持；
- Paged Attention Kernel 支持；
- 显存预算和 Tail Waste 目标；
- 当前 DType、Head Dim 和 HIP 架构限制。

选择顺序：

```text
读取用户/默认候选
→ 查询已编译 Kernel Capability
→ 运行小规模正确性 Probe
→ 根据显存预算计算 Block Count
→ 将最终值写入 build/runtime manifest
```

不得因为某个 Python 库曾使用 256，就无条件把 256 作为海光或其他 Backend 的最低要求。

## 4. 物理布局必须显式化

实现必须选择一种布局，并将 Stride/Offset 规则写入 `KvLayout`；Kernel 不得根据 Allocation 大小猜布局。

```cpp
enum class KvMajorOrder {
  kLayerBlockHeadTokenDim,
  kLayerBlockTokenHeadDim,
};

struct KvLayout {
  std::int64_t num_layers = 0;
  std::int64_t num_blocks = 0;
  std::int64_t block_size = 0;
  std::int64_t local_kv_heads = 0;
  std::int64_t head_dim = 0;
  DType dtype = DType::kUnknown;
  KvMajorOrder order = KvMajorOrder::kLayerBlockHeadTokenDim;
  std::array<std::int64_t, 5> strides{};

  Result<std::size_t> ByteOffset(std::int32_t layer,
                                 std::int32_t block,
                                 std::int32_t kv_head,
                                 std::int32_t token_in_block,
                                 std::int32_t dim) const;
};
```

所有乘法和加法必须使用 Checked Arithmetic。Debug 构建必须检查 Layer、Block、Head、Token 和 Dim 边界；Release Kernel 入口必须在 Host 侧完成等价验证。

## 5. 所有权与生命周期

KV Storage 由 Engine 级 Pool 唯一拥有，请求只持有不可伪造的 Block Handle。

```cpp
struct BlockHandle {
  std::uint32_t id = 0;
  std::uint32_t generation = 0;
};

class PagedKvCache {
 public:
  static Result<PagedKvCache> Create(const KvLayout& layout,
                                     DeviceId device,
                                     BackendStream allocation_stream);

  Result<BlockReservation> Reserve(std::uint32_t count);
  Status Commit(RequestId request, BlockReservation&& reservation);
  Status Release(RequestId request, BackendStream last_use_stream);

  DeviceSpan<std::byte> key_storage() const;
  DeviceSpan<std::byte> value_storage() const;
  const KvLayout& layout() const;

 private:
  DeviceBuffer key_storage_;
  DeviceBuffer value_storage_;
  std::vector<BlockMetadata> metadata_;
  std::vector<std::uint32_t> free_list_;
  std::mutex allocator_mutex_;
};
```

强制规则：

- `Reserve` 不得立即修改请求可见的 Block Table；
- 全部 Rank/Layer 准备成功后才能 `Commit`；
- 未 Commit 的 Reservation 析构时自动 Rollback；
- Generation 不匹配必须报告 Stale Handle；
- Release 必须等待最后一次 GPU 使用完成，不能提前复用 Block；
- 取消和重复 Release 必须幂等，不能 Double Free；
- Steady-state Decode 禁止调用 `hipMalloc/cudaMalloc`。

## 6. Prefill 契约

Prefill 必须使用当前投影产生的 K/V 计算 Attention，并把同一份 K/V 写入 Paged Cache。不得先写 Cache 再通过未同步的 Cache 指针读取。

```cpp
struct PrefillAttentionArgs {
  TensorView q;                 // [Tq, Hq, D]
  TensorView k_current;         // [Tq, Hkv, D]
  TensorView v_current;         // [Tq, Hkv, D]
  DeviceSpan<const std::int32_t> cu_query_lengths;
  DeviceSpan<const std::int32_t> cu_kv_lengths;
  DeviceSpan<const std::int64_t> slot_mapping;
  MutableTensorView output;
  bool causal = true;
  float scale = 1.0F;
};

Status RunPrefillAttention(const PrefillAttentionArgs& args,
                           BackendStream stream);
Status ScatterKvToCache(const TensorView& k,
                        const TensorView& v,
                        DeviceSpan<const std::int64_t> slot_mapping,
                        PagedKvLayerView cache,
                        BackendStream stream);
```

Prefill 的完成条件是 Attention Output、KV 写入和请求元数据更新之间具有明确的 Stream/Event 顺序。Host 端 `kv_length` 只能在设备操作成功排队并建立依赖后推进。

## 7. Decode 契约

### 7.1 逻辑位置与物理 Cache 地址必须分离

每个请求的 RoPE Position 从 0 开始递增，与请求被分配到哪个 KV Block 或
Segment 无关。`logical_position` 进入 RoPE；`kv_offset/slot_mapping` 只进入
KV 读写地址计算。两个请求内容相同但物理 Segment 不同时，生成结果必须在
确定性模式下字节一致。

禁止使用 `kv_segment_base + logical_position` 作为 RoPE Position。必须有
跨两个不同物理 Segment 的相同请求回归测试。

Decode Step 必须先为新 Token 解析/预留 Slot，再写入本步 K/V，并让 Paged Attention 看到包含新 Token 的 Context。

```cpp
struct DecodeAttentionArgs {
  TensorView q;  // [num_sequences, Hq, D]
  DeviceSpan<const std::int32_t> block_table;
  DeviceSpan<const std::int32_t> context_lengths;
  PagedKvLayerView cache;
  MutableTensorView output;
  std::int32_t max_blocks_per_sequence = 0;
  float scale = 1.0F;
};

Status AppendDecodeKv(const TensorView& k,
                      const TensorView& v,
                      DeviceSpan<const std::int64_t> slot_mapping,
                      PagedKvLayerView cache,
                      BackendStream stream);
Status RunPagedDecodeAttention(const DecodeAttentionArgs& args,
                               BackendStream stream);
```

禁止每步把所有历史 KV 拼成连续 Tensor。Block Table 和 Context Length Buffer 应预分配并复用；只上传本轮发生变化的 Metadata。

## 8. Backend 与 Fallback

统一接口必须区分：

```text
Native Paged Attention Kernel
Vendor/Library Attention Kernel
Bounded Reference Kernel（仅正确性/调试）
Unsupported
```

Fallback 必须记录 Backend、Shape、DType 和原因。生产路径不得通过 Python 或 CPU 完成 Attention；GPU Backend 不支持某个 Shape 时应返回明确的 Unsupported，而不是静默得到不同语义。

## 9. Stream 与并发

以下依赖必须可在 Trace 中观察：

```text
metadata upload
→ KV append/scatter
→ paged attention read
→ output consumer
→ block release/reuse
```

可以使用独立 Metadata/Compute Stream，但必须通过 Event 建立顺序。禁止用 Device-wide Synchronize 掩盖缺失依赖，也禁止在 Kernel 尚未完成时修改或释放 Host/Device Metadata。

## 10. Tensor Parallel

每个 Rank 拥有本地 KV Storage，逻辑 Block 决策必须一致：

```text
Rank 0 产生 Step/Block 需求
→ 所有 Rank 本地 Reserve
→ Native Collective 汇总 can_reserve
→ 全部成功则 Commit
→ 任意失败则所有 Rank Rollback
```

TP 容量取所有 Rank 可用 Block 数的最小值。不得让某个 Rank 独自增加 Context Length 或跳过 KV 写入；否则后续 Collective/Attention 会产生错误或死锁。

## 11. 数值与 DType

- Q/K Norm、RoPE、Attention Scale 必须与 Qwen3 配置一致；
- BF16/FP16 累加精度由 Kernel Capability 明确声明；
- Softmax 的 Max/Sum Reduction 必须避免溢出；
- KV DType 与 Activation DType 不同时必须有显式转换和误差测试；
- Padding Token 不得生成 Slot；
- Prefill 最后一个 Token 与逐 Token Decode 的 Logits 必须在 DType 阈值内一致。
- 上述一致性必须由断言强制，最低 Cosine 阈值为 0.95；只打印数值不算通过。

不得用 `nan_to_num` 一类输出修补隐藏非法数值；出现 NaN/Inf 时应定位首个出错 Layer/Kernel。

## 12. 必测矩阵

```text
Block 边界前一 Token、边界 Token、边界后一 Token
空闲池耗尽与事务 Rollback
取消发生在 Prefill、Decode、响应写出阶段
重复 Release 与 Stale Generation
不同 Prompt Length 的混合 Batch
GQA 的 Q Head/KV Head 映射
BF16/FP16 KV DType
TP=1 与 TP=N Greedy Token/Logits 对齐
各 Rank 容量不一致时全局 Rollback
至少数千轮 Allocate/Decode/Release Soak 后 Block 全量回收
```

Acceptance 必须同时满足：无越界、无 Double Free、无 Steady-state Allocation、无 Python Runtime、无跨请求 KV 污染，并且 Reference 与 Native Kernel 在声明阈值内一致。
