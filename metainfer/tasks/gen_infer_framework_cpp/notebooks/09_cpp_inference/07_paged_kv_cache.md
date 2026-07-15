# Paged KV Cache 原生实现

先读：`00_contracts/cpp/cpp_memory_contracts.md`、共享的 `00_contracts/attention_kv_contracts.md`。共享文档提供 Attention/KV 语义，具体 C++ 所有权以本目录 Contract 为准。

Paged KV 的核心是把“请求逻辑 Token 位置”映射到“设备物理 Block”，并在分配失败、取消、TP Rank 失败时保持事务一致。

## 1. 逻辑状态与物理状态分离

```cpp
using PhysicalBlockId = std::uint32_t;

struct BlockRef {
  PhysicalBlockId id = 0;
  std::uint32_t generation = 0;
};

class BlockTable {
 public:
  Span<const BlockRef> blocks() const;
  std::uint32_t logical_tokens() const;
  Status Append(BlockRef block);
  void SetLogicalTokens(std::uint32_t value);

 private:
  std::vector<BlockRef> blocks_;
  std::uint32_t logical_tokens_ = 0;
};
```

Block Table 只保存 Block ID/Generation，不保存 Device Pointer。每个 Layer/Rank 根据相同 Block ID 访问自己的物理 KV Pool。

## 2. 物理布局

选择一种布局并写入 `KvLayout`，Kernel 不能根据分配大小猜布局：

```cpp
struct KvLayout {
  std::int64_t num_layers;
  std::int64_t num_blocks;
  std::int64_t block_size;
  std::int64_t num_kv_heads;
  std::int64_t head_dim;
  DType dtype;
};

// 一种可选语义布局：
// K/V [layer, block, kv_head, token_in_block, head_dim]
Result<std::size_t> KvByteOffset(const KvLayout& layout,
                                 int layer,
                                 PhysicalBlockId block,
                                 int kv_head,
                                 int token_in_block,
                                 int dim);
```

Offset 计算使用 Checked Arithmetic。K 和 V 可以分离 Storage，也可以在一个大 Allocation 中使用明确 Offset；两者都要有 Layout 测试。

## 3. 地址映射公式

```text
logical_position p
block_table_slot = p / block_size
token_in_block   = p % block_size
block_ref        = request.block_table[block_table_slot]
physical_address = pool(layer, block_ref.id, kv_head, token_in_block, dim)
```

Host 侧验证：

```cpp
Result<KvLocation> ResolveKvLocation(const BlockTable& table,
                                     std::uint32_t position,
                                     std::uint32_t block_size) {
  const std::uint32_t slot = position / block_size;
  const std::uint32_t offset = position % block_size;
  if (slot >= table.blocks().size()) {
    return OutOfRange("logical KV position has no physical block");
  }
  return KvLocation{table.blocks()[slot], offset};
}
```

Device Kernel 采用等价的 Bounds-check/Debug Assertion 策略。

## 4. Block Pool

```cpp
struct BlockMetadata {
  bool allocated = false;
  RequestId owner = 0;
  std::uint32_t generation = 0;
};

class KvBlockPool {
 public:
  static Result<KvBlockPool> Create(KvLayout layout,
                                    std::uint64_t budget_bytes,
                                    int device);

  Result<BlockReservation> Reserve(std::uint32_t count);
  Status Commit(BlockReservation&& reservation, RequestId owner);
  void Rollback(BlockReservation&& reservation) noexcept;
  Status FreeRequest(RequestId owner);

  std::uint32_t free_blocks() const;
  const KvLayout& layout() const;

 private:
  DeviceBuffer key_storage_;
  DeviceBuffer value_storage_;
  std::vector<BlockMetadata> metadata_;
  std::vector<PhysicalBlockId> free_list_;
  mutable std::mutex mutex_;
};
```

Free List 可以优化为无锁/分层结构，但第一版应使用容易验证的单 Owner + Mutex 设计。

## 5. 事务分配

```cpp
Result<BlockReservation> KvBlockPool::Reserve(std::uint32_t count) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (count > free_list_.size()) {
    return ResourceExhausted("insufficient KV blocks");
  }

  BlockReservation result(this);
  result.blocks.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    const PhysicalBlockId id = free_list_.back();
    free_list_.pop_back();
    BlockMetadata& meta = metadata_.at(id);
    if (meta.allocated) return Internal("free list contains allocated block");
    ++meta.generation;
    result.blocks.push_back(BlockRef{id, meta.generation});
  }
  return result;
}
```

`Reservation` 析构时若未 Commit，必须自动 Rollback。Rollback 不能重复加入 Free List。

```cpp
class BlockReservation {
 public:
  ~BlockReservation() {
    if (pool != nullptr && !committed) pool->Rollback(std::move(*this));
  }
  KvBlockPool* pool = nullptr;
  std::vector<BlockRef> blocks;
  bool committed = false;
};
```

生产代码应封装字段，示例用于说明所有权。

## 6. Prefill 分配与 Slot Mapping

```cpp
Result<PrefillKvPlan> PreparePrefillKv(RequestState& request,
                                       std::uint32_t prompt_tokens,
                                       KvBlockPool& pool) {
  const std::uint32_t required_blocks =
      CeilDiv(prompt_tokens, pool.layout().block_size);
  ASSIGN_OR_RETURN(auto reservation, pool.Reserve(required_blocks));

  PrefillKvPlan plan;
  plan.request_id = request.id;
  plan.reservation = std::move(reservation);
  plan.slot_mapping.reserve(prompt_tokens);
  for (std::uint32_t p = 0; p < prompt_tokens; ++p) {
    const BlockRef block = plan.reservation.blocks[p / pool.layout().block_size];
    plan.slot_mapping.push_back(
        PhysicalSlot{block, p % pool.layout().block_size});
  }
  return plan;
}
```

只有全部 Layer 写入成功后，Engine 才 Commit Reservation、更新 Block Table 和 `kv_length`。Padding 不生成 Slot。

## 7. Decode Append

```cpp
Result<DecodeKvPlan> PrepareDecodeKv(RequestState& request,
                                     KvBlockPool& pool) {
  const std::uint32_t position = request.kv_length;
  const bool needs_block = position % pool.layout().block_size == 0;

  std::optional<BlockReservation> reservation;
  if (needs_block) {
    ASSIGN_OR_RETURN(auto one, pool.Reserve(1));
    reservation.emplace(std::move(one));
  }

  const BlockRef target = needs_block
      ? reservation->blocks.front()
      : request.blocks.blocks().back();
  return DecodeKvPlan{
      request.id,
      position,
      PhysicalSlot{target, position % pool.layout().block_size},
      std::move(reservation)};
}
```

Decode Kernel 写入新 K/V 后，直接遍历现有 Block Table 计算 Attention，禁止每步重建连续 KV。

## 8. TP 跨 Rank 事务

每个 Rank 的物理地址不同，但逻辑 Block 数和 Block Table 决策必须一致：

```text
Rank 0 计算需要 block_count
-> 所有 Rank 本地 Reserve（暂不 Commit）
-> AllReduce(can_reserve, MIN)
-> 全部成功：Commit 相同逻辑 Block 序列
-> 任意失败：所有 Rank Rollback
```

实现可以由 Rank 0 分配 Logical Block ID，各 Rank 映射本地 Physical Block；无论哪种方案，都要验证 Collective 顺序与 Rollback。

## 9. 容量和 Fragmentation

```cpp
const std::uint64_t bytes_per_block =
    2ULL * num_layers * block_size * local_kv_heads * head_dim * SizeOf(kv_dtype);
const std::uint64_t block_count = kv_budget_bytes / bytes_per_block;
```

乘法必须 Checked。TP 使用所有 Rank 可承受的最小 Block Count。Block Size 是需要 Profile 的配置，不应复制其他框架常量。

指标：

```text
free/allocated/high-water blocks
allocation failures
tail waste tokens
per-request blocks
rollback count
stale generation detection
```

## 10. 取消与关闭

取消只在安全 Step Boundary 生效。Outstanding GPU Work 完成/排序后，Engine 调用 `FreeRequest`。函数幂等，重复 Cancellation/SIGTERM 不得 Double Free。

## 11. 测试矩阵

```cpp
TEST(KvPool, AllocatesAtExactBlockBoundary);
TEST(KvPool, AllocatesOneTokenPastBoundary);
TEST(KvPool, RollsBackOnExhaustion);
TEST(KvPool, DetectsStaleGeneration);
TEST(KvPool, FreeRequestIsIdempotent);
TEST(KvAddress, ResolvesLayerHeadTokenOffset);
TEST(KvLifecycle, CancelDuringPrefillAndDecode);
TEST(KvTp, RankCapacityMismatchRollsBackEverywhere);
```

再运行长时间 Allocate/Free Soak，最终 Free Block 数必须等于初始值。
