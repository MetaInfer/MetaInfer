# Paged KV Cache：C++/HIP 推理框架生成指南

> 本文用于指导 Agent 从零实现 Paged KV Cache，不依赖任何现有推理框架源码。文中的
> “必须”表示正确性或生命周期契约，“建议”表示适合第一版实现的工程选择。

相关主题：[Continuous Batching](continuous_batching.md) ·
[Tensor Parallel](../distributed/tensor_parallel.md)

## 1. 目标与范围

Paged KV Cache 需要解决四个问题：

1. 自回归 Decode 重用历史 Key/Value，避免重复计算完整前缀；
2. 多请求共享一个固定显存池，不为每个请求执行 `hipMalloc/hipFree`；
3. 请求可以动态加入、增长、取消、完成并安全复用物理块；
4. Prefill/Decode Attention 能通过每条序列的页表访问非连续物理块。

第一版应实现：

- 固定大小物理 KV block pool；
- 每条 sequence 独立 block table；
- Reserve/Commit/Rollback 事务；
- committed length 与 reserved capacity 分离；
- batch 原子推进；
- stale handle 检测；
- owning stream 同步后释放；
- CPU reference 和 HIP Paged Attention；
- 与 Chunked Prefill、Packed Decode、Continuous Batching 集成。

第一版不必实现：

- Prefix Cache；
- Copy-on-Write；
- CPU KV Swap；
- Request Preemption；
- LRU Eviction；
- 跨节点 KV；
- 超额分配。

这些高级能力应建立在正确的块所有权和生命周期之上，不能与基础版本同时混写。

## 2. KV Cache 的计算意义

自回归生成第 `t` 个 token 时，每层只需要计算新 token 的 Q/K/V：

```text
Q_t = X_t W_q
K_t = X_t W_k
V_t = X_t W_v
```

历史 K/V 已经缓存：

```text
K_cache = [K_0, K_1, ..., K_t]
V_cache = [V_0, V_1, ..., V_t]

Attention_t = softmax(Q_t K_cache^T / sqrt(head_dim)) V_cache
```

没有 KV Cache 时，每个 Decode step 都需要重新运行完整前缀；有 KV Cache 后，每层只新增
一个 token 的 K/V，Attention 仍随上下文长度增长，但 QKV Projection 不再重复处理历史
token。

## 3. 为什么不能只做连续大数组

朴素布局：

```text
K/V[layer][slot][max_context][kv_head][head_dim]
```

它的问题是：

- 每个 slot 必须按最大上下文预留，短请求浪费尾部；
- slot 数直接决定最大并发，不能灵活共享剩余容量；
- 请求长度不同，连续区间释放后容易产生外部碎片；
- batch row 和固定 slot 绑定，动态调度与内存所有权耦合；
- 扩容可能要求搬迁正在被 GPU 读取的历史 KV。

Paged KV Cache 把“逻辑 token 位置”和“物理显存地址”分离。

## 4. 分页地址模型

将 KV Pool 切成固定 token 数的物理块。对逻辑 token position `p`：

```text
logical_block = p / block_size_tokens
block_offset  = p % block_size_tokens
physical_block = block_table[logical_block]
```

物理地址：

```text
pool[layer][physical_block][block_offset][kv_head][head_dim]
```

不同 sequence 的物理块不要求连续：

```text
Sequence A: logical [0,1,2] -> physical [6,1,11]
Sequence B: logical [0,1]   -> physical [3,8]
```

Attention Kernel 必须通过当前 row 的 `block_table` 完成地址翻译，不能假设：

- physical block 连续；
- batch row 等于 sequence slot；
- 不同 sequence 的 past length 相同；
- position 等于 batch 内 token index。

## 5. 配置与显存公式

建议接口：

```cpp
enum class KvDType { kFp16, kBf16 };

struct PagedKvConfig {
  int64_t num_layers = 0;
  int64_t num_blocks = 0;
  int64_t block_size_tokens = 16;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
  KvDType dtype = KvDType::kFp16;

  Status Validate() const;
};
```

Z200/gfx906 的 Paged/Continuous scalable path 使用 FP16 KV。它需要 FP16 Paged writer 和
attention specialization；单序列 reference 中的 `float*` dense cache 不能直接复用。
BF16 只能在 Backend、Kernel 和实卡数值测试都明确支持时选择，不能因为两者都是
16-bit 存储就共用错误的指针类型或解释方式。

单个 K 或 V block 的字节数：

```text
block_bytes = block_size_tokens * num_kv_heads * head_dim * dtype_bytes
```

完整 KV Pool：

```text
total_bytes = num_layers * 2(K,V) * num_blocks * block_bytes
```

按单条最大上下文估算：

```text
blocks_per_sequence = ceil(max_context_tokens / block_size_tokens)
reserved_tokens_per_sequence = blocks_per_sequence * block_size_tokens

per_sequence_bytes = num_layers * 2 * reserved_tokens_per_sequence
                   * num_kv_heads * head_dim * dtype_bytes
```

这里必须使用向上取整后的 `reserved_tokens_per_sequence`；当 `max_context_tokens` 不是
`block_size_tokens` 的整数倍时，最后一个 block 未使用的 token 槽仍然占显存。

启动前必须使用 checked arithmetic，检测：

- 任意维度非正；
- element count 溢出；
- byte count 溢出；
- block index 超过 Kernel 使用的整数范围；
- 配置需要的显存大于可用显存预算。

不能依赖 `hipMalloc` 失败作为唯一容量验证。

### 5.1 示例，不得硬编码

假设：

```text
layers      = 36
kv_heads    = 8
head_dim    = 128
context     = 4096
dtype_bytes = 2
```

则单条 sequence 的 KV 约为：

```text
36 * 2 * 4096 * 8 * 128 * 2
= 603,979,776 bytes
≈ 576 MiB
```

该示例只用于说明容量数量级。实现必须从模型配置读取 shape。

### 5.2 冻结容量策略映射

表单字段与 Pool 配置必须一一对应：

```text
context              = resolved.resource_contract.max_context_per_request
max_active           = resolved.resource_contract.max_active_requests
block_size           = paged_kv_cache.block_size
blocks_per_request   = ceil(context / block_size)

full_context_per_request:
  total_blocks = blocks_per_request * max_active
  guaranteed_full_context_requests = max_active

shared_token_budget:
  total_blocks = ceil(max_total_cached_tokens / block_size)
  guaranteed_full_context_requests
    = min(max_active, floor(total_blocks / blocks_per_request))
```

`max_total_cached_tokens` 是 Pool 总容量承诺，不是“每个请求首次 admission 时一次性占满整段
context”的要求。实现可以按 step 增量 Reserve；但 Admission 必须立即拒绝
`prompt + max_new_tokens` 永远不可能装入 Pool 的请求，并保证冻结策略声明的并发上限可满足。
Pool 大小、分配时机和 committed logical length 是三个不同概念。

## 6. 推荐的设备内存布局

每层 K/V TensorView：

```text
[num_blocks, block_size_tokens, num_kv_heads, head_dim]
```

建议启动时只申请一个大 Buffer：

```text
| K layer 0 | K layer 1 | ... | alignment | V layer 0 | V layer 1 | ... |
```

要求：

- K/V layer 起始地址满足 Kernel 对齐要求；
- Layer offset 使用 checked `size_t` 计算；
- 请求稳态不产生新的设备 allocation；
- TensorView 不拥有内存，只引用 Pool Buffer；
- Pool 生命周期必须长于所有 Runner、Kernel 和 Sequence view。

也可以每层一个 allocation，但会增加 allocation 数量和析构复杂度。第一版优先单 Buffer。

## 7. Host 元数据结构

三份运行时契约共用同一个稳定 ID 类型：

```cpp
using SequenceId = uint64_t;  // 0 is invalid/reserved
```

HTTP 层的 request id 可以与 `SequenceId` 数值相同，但 KV Cache、Scheduler 和 TP Rank
之间必须传递明确的 `SequenceId`，不得用当前 batch row 或可复用 slot 代替。

### 7.1 BlockHandle

```cpp
struct BlockHandle {
  int32_t index = -1;
  uint64_t generation = 0;
};
```

`index` 是物理块编号；`generation` 防止旧 handle 在块被释放和复用后再次生效。

### 7.2 SequenceKvState

内部状态：

```cpp
struct SequenceKvState {
  SequenceId id = 0;
  std::vector<BlockHandle> blocks;
  int64_t committed_tokens = 0;
};
```

必须区分：

```text
reserved_capacity = blocks.size() * block_size_tokens
committed_tokens   = 已成功完成模型 step 的逻辑长度
```

Attention 只能读取 `[0, committed_tokens + current_step_tokens)` 中因当前图定义为有效的
位置，不能把仅预留但未提交的区域视为历史 KV。

### 7.3 SequenceKvView

对外返回快照：

```cpp
struct SequenceKvView {
  SequenceId id = 0;
  std::vector<BlockHandle> blocks;
  int64_t committed_tokens = 0;
};
```

释放时必须比较 view 与当前内部状态，拒绝 stale view。

### 7.4 KvReservation

```cpp
class KvReservation {
 public:
  KvReservation(KvReservation&&) noexcept;
  KvReservation& operator=(KvReservation&&) noexcept;
  KvReservation(const KvReservation&) = delete;

 private:
  SequenceId sequence_id_ = 0;
  uint64_t transaction_id_ = 0;
  std::vector<BlockHandle> blocks_;
};
```

Reservation 必须 move-only，避免同一事务被两个对象 Commit。

Continuous Batching 一次会为多个 sequence 扩容，因此还需要 batch transaction：

```cpp
struct SequenceReservationRequest {
  SequenceId id = 0;
  int64_t additional_tokens = 0;
};

class KvReservationBatch {
 public:
  KvReservationBatch(KvReservationBatch&&) noexcept;
  KvReservationBatch& operator=(KvReservationBatch&&) noexcept;
  KvReservationBatch(const KvReservationBatch&) = delete;

 private:
  std::vector<KvReservation> reservations_;
};
```

Batch reservation 要么整体挂接 capacity，要么整体回滚，不能在第 N 行失败时
留下前 N-1 行已挂接的新 blocks。

## 8. PagedKvCache 公共接口

推荐最小接口：

```cpp
class PagedKvCache {
 public:
  static Result<PagedKvCache> Create(
      const PagedKvConfig&, Backend&, Stream& owning_stream);

  Result<KvReservation> Reserve(
      SequenceId id, int64_t additional_tokens);
  Status Commit(KvReservation&&);
  Status Rollback(KvReservation&&);

  Result<KvReservationBatch> ReserveBatch(
      const std::vector<SequenceReservationRequest>& requests);
  Status CommitBatch(KvReservationBatch&&);
  Status RollbackBatch(KvReservationBatch&&);

  Status Advance(SequenceId id, int64_t tokens);
  Status AdvanceBatch(const std::vector<SequenceAdvance>& advances);

  Result<SequenceKvView> ViewSequence(SequenceId id) const;
  Status ReleaseSequence(const SequenceKvView&, Stream&);

  Result<TensorView> LayerKeyPool(int64_t layer);
  Result<TensorView> LayerValuePool(int64_t layer);

  size_t total_blocks() const noexcept;
  size_t free_blocks() const noexcept;
  size_t active_sequences() const noexcept;
  size_t high_water_blocks() const noexcept;
};
```

以上签名是项目默认 C++17 契约。C++20 实现可在不改变所有权和生命周期的
前提下改用 `std::span<const T>`。

`ReserveBatch()` 必须先验证全部请求并取得全部 blocks，再返回 move-only
transaction。`CommitBatch()` 在单 Scheduler 所有者模型下对一个已验证 transaction
必须是不会部分失败的状态转换；只有 Commit 后 BatchAssembler 才能读到覆盖当前
step 的完整 block table。`AdvanceBatch()` 仍然只在模型 step 成功后推进逻辑长度。

## 9. Reserve 算法

输入是“额外需要多少 token”，不是“额外需要多少 block”。

```text
committed = sequence exists ? sequence.committed_tokens : 0
target_tokens = committed + additional_tokens
target_blocks = ceil(target_tokens / block_size)
needed_blocks = max(0, target_blocks - current_blocks)
```

算法契约：

1. `id != 0`，`additional_tokens > 0`；
2. 该 sequence 没有 pending reservation；
3. 使用 checked addition；
4. 先计算 `needed_blocks`；
5. 如果 free blocks 不足，返回 ResourceExhausted，任何状态不变；
6. 取出全部需要的块并记录当前 generation；
7. 建立 transaction id 和 pending record；
8. 返回 move-only reservation。

伪代码：

```cpp
Result<KvReservation> Reserve(SequenceId id, int64_t additional) {
  validate(id, additional);
  auto needed = CalculateNeededBlocks(id, additional);
  if (needed > free_list.size()) return ResourceExhausted();

  std::vector<BlockHandle> acquired;
  for (size_t i = 0; i < needed; ++i) {
    int32_t index = free_list.back();
    free_list.pop_back();
    acquired.push_back({index, generations[index]});
  }
  uint64_t tx = next_transaction_id++;
  pending.emplace(tx, Pending{id, acquired});
  return KvReservation{id, tx, std::move(acquired)};
}
```

## 10. Commit 与 Rollback

Commit 必须验证：

- transaction id 存在；
- sequence id 匹配；
- reservation blocks 与 pending record 完全一致；
- reservation 尚未被消费。

成功后：

```text
sequence.blocks += reservation.blocks
erase pending transaction
invalidate reservation
update high-water mark
```

Rollback 执行：

```text
validate transaction
return blocks to free list
erase pending transaction
invalidate reservation
```

建议按获取顺序的逆序归还，使测试和多 Rank 确定性分配更容易保持。

## 11. AdvanceBatch 原子提交

```cpp
struct SequenceAdvance {
  SequenceId id;
  int64_t tokens;
};
```

`AdvanceBatch()` 必须先验证整个 batch：

- 非空；
- id 非零；
- tokens 为正；
- id 不重复；
- sequence 存在；
- `committed + tokens <= reserved_capacity`。

全部通过后再执行第二遍更新：

```cpp
for (const auto& a : advances) {
  states.at(a.id).committed_tokens += a.tokens;
}
```

禁止边验证边更新，否则第 N 行失败会留下前 N-1 行已推进的部分状态。

## 12. ReleaseSequence 安全语义

释放流程：

1. 验证 stream 是 Pool 的 owning stream/device；
2. 验证 view 与当前 sequence blocks、committed length 完全一致；
3. 同步 owning stream，确保没有 Kernel 继续读写这些 blocks；
4. 检查每个 block 的 generation；
5. generation 加一；
6. blocks 归还 free list；
7. 删除 sequence state。

必须先同步再复用。否则旧 Kernel 可能读取已经分配给新请求的物理块，造成跨请求数据
污染。

如果希望以后减少全 stream 同步，可以引入 per-sequence/per-block event，但不能直接删除
同步边界。

## 13. Packed Batch 元数据契约

Paged Attention 需要每个 batch row 的逻辑信息：

```cpp
struct PackedPagedBatch {
  std::vector<SequenceId> sequence_ids;  // [B]
  std::vector<int32_t> token_ids;        // [T]
  std::vector<int32_t> positions;        // [T]
  std::vector<int32_t> sequence_offsets; // [B + 1]
  std::vector<int32_t> token_rows;       // [T]
  std::vector<int32_t> past_lengths;     // [B]
  std::vector<int32_t> block_tables;     // [B * table_stride]
  int64_t block_table_stride = 0;
  int64_t max_total_tokens = 0;
};
```

定义：

- `B`：sequence rows；
- `T`：本 step 的真实 token 数；
- `sequence_offsets[i:i+2]`：row i 的 token 区间；
- `token_rows[t]`：token t 所属 row；
- `past_lengths[i]`：step 前的 committed length；
- `block_tables[i * stride + j]`：row i 的 logical block j 对应 physical block；
- 未使用页表项为 `-1`。

### 13.1 必须验证的不变量

在进入 HIP Kernel 前验证：

```text
sequence_ids.size == B > 0
token_ids.size == positions.size == token_rows.size == T > 0
sequence_offsets.size == B + 1
past_lengths.size == B
block_tables.size == B * stride
sequence_offsets[0] == 0
sequence_offsets[B] == T
```

对每行验证：

- sequence id 唯一；
- row token 区间非空；
- position 等于 `past_length + local_token_index`；
- token_rows 与 row 一致；
- block table 与 Cache view 一致；
- required blocks 已预留；
- physical index 在 Pool 范围内；
- required block 不重复；
- unused entries 为 `-1`。

通过验证的 batch 才可走 trusted Kernel path。

## 14. KV 写入 Kernel 契约

建议参数。`KvT` 是由 `PagedKvConfig::dtype` 选择的 Kernel specialization；
Z200 基线中 `KvT = __half`：

```cpp
template <typename KvT>
struct PagedKvWriteParams {
  const KvT* current_k;            // [T, kv_heads, head_dim]
  const KvT* current_v;            // [T, kv_heads, head_dim]
  KvT* key_pool;
  KvT* value_pool;
  const int32_t* positions;        // [T]
  const int32_t* token_rows;       // [T]
  const int32_t* block_tables;     // [B, stride]
  int32_t tokens;
  int32_t table_stride;
  int32_t block_size;
  int32_t kv_heads;
  int32_t head_dim;
};
```

对 token `t`：

```text
row = token_rows[t]
pos = positions[t]
logical = pos / block_size
offset = pos % block_size
physical = block_tables[row * stride + logical]
```

目标 index：

```text
(((physical * block_size + offset) * kv_heads + kv_head) * head_dim + dim)
```

Kernel 不得用 `t` 代替 position，也不得用 row 代替 physical block。

## 15. Paged Decode Attention 契约

Decode 每个 row 通常只有一个 query token，但历史长度不同：

```cpp
template <typename KvT>
struct PagedDecodeAttentionParams {
  const KvT* query;           // [B, q_heads, head_dim]
  const KvT* current_k;       // [B, kv_heads, head_dim]
  const KvT* current_v;       // [B, kv_heads, head_dim]
  KvT* key_pool;
  KvT* value_pool;
  const int32_t* block_tables;
  const int32_t* past_lengths;
  KvT* output;                // [B, q_heads, head_dim]
  int32_t batch;
  int32_t q_heads;
  int32_t kv_heads;
  int32_t head_dim;
  int32_t block_size;
  int32_t table_stride;
};
```

每行总有效长度：

```text
total_length = past_lengths[row] + 1
```

GQA Head 映射：

```text
queries_per_kv = q_heads / kv_heads
kv_head = q_head / queries_per_kv
```

Softmax reduction 建议 FP32；K/V 存储可为 BF16/FP16。必须处理 NaN/Inf 并返回可检测
状态，不能静默输出非法 token。

### 15.1 Fused Decode Attention 的 Kernel 映射

推荐先采用容易验证的映射：一个 workgroup 负责一个 `(row, q_head)`：

```text
grid.x = batch * q_heads
row    = block_id / q_heads
q_head = block_id % q_heads
```

workgroup 内线程协作遍历 `head_dim`，再按逻辑 position 顺序遍历历史。每个 position 都通过
页表翻译：

```text
logical_block = position / block_size
block_offset  = position % block_size
physical      = block_tables[row, logical_block]
kv_head       = q_head / queries_per_kv
```

该映射的优点是 row/head 完全隔离，没有跨 workgroup 同步；缺点是短上下文或小 head_dim 时
并行度有限。正确性版本完成后，可以让多个 workgroup 分段处理长 context，再执行第二阶段
归并，但必须正确合并 softmax 统计量。

### 15.2 Online Softmax，避免 score workspace

不要先物化 `[B, q_heads, max_context]` scores。可在读取分页 K 的同时维护在线统计量：

```text
m = -infinity              // running maximum
l = 0                      // running exp sum
acc[d] = 0                 // FP32 weighted V accumulator

for position in [0, total_length):
    score = dot(q, K[position]) * scale
    m_new = max(m, score)
    alpha = exp(m - m_new)
    beta  = exp(score - m_new)
    l     = l * alpha + beta
    acc   = acc * alpha + beta * V[position]
    m     = m_new

output[d] = acc[d] / l
```

实际 Kernel 由多个 lanes 共同计算 dot 和 `acc[d]`，`m/l` 需要 workgroup reduction。必须用
FP32 保存 score、maximum、sum 和 accumulator；最终才转换成目标 activation dtype。

如果按多个 context tiles 分段，每个 tile 输出 `(m_i, l_i, acc_i)`。两段的稳定合并公式：

```text
m = max(m_a, m_b)
l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
acc = acc_a * exp(m_a - m) + acc_b * exp(m_b - m)
```

不能只相加各 tile 已归一化的 output。

### 15.3 当前 K/V 的可见性

Decode 输入 token 的 K/V 可以：

1. 先由 KV Write Kernel 写入预留位置，再运行 Attention；
2. 由 Fused Attention 直接使用 `current_k/current_v`，并在同一 Kernel 写入 Pool。

方案 2 若同时读写 Pool，必须确保所有读取当前位置的线程看到一致数据。最简单的正确做法是
对当前 position 直接读寄存器/输入 tensor，而不是依赖跨 workgroup 写后可见性。逻辑长度仍
只能在整个模型 step 成功后由 Host `AdvanceBatch()` 提交。

## 16. Paged Prefill Attention 契约

Prefill 是 ragged token-major batch。每行可能有多个 current tokens：

```text
row i current range = [sequence_offsets[i], sequence_offsets[i+1])
past = past_lengths[i]
```

第 `local` 个 current token 的 causal 可见范围：

```text
[0, past + local]
```

实现可以选择：

1. 先把 current K/V 写入 Pool，再从 Pool 读取历史和 current；
2. 历史从 Pool 读取，current chunk 直接从 current K/V tensor 读取，结束后写 Pool；
3. Fused write + score + softmax + output。

无论选择哪条路径，都必须满足：

- 当前 token 不能看到未来 current token；
- row 之间不能读取彼此 block table；
- `committed_tokens` 在整个 batch 成功前保持旧值；
- batch 失败后请求进入失败清理，不把失败写入作为有效历史读取。

### 16.1 Prefill 的 Query 映射

Prefill 可把一个 workgroup 映射到 `(packed_token, q_head)`。对 packed token `t`：

```text
row         = token_rows[t]
local_query = t - sequence_offsets[row]
query_pos   = past_lengths[row] + local_query
visible     = query_pos + 1
```

然后使用与 Decode 相同的分页地址翻译和 online softmax，只遍历 `[0, visible)`。这一路径
天然支持 ragged rows，但长 Prompt 的总复杂度仍是 causal attention 的平方级。

高性能版本通常按 query tile 和 key tile 分块，让一个 workgroup 同时处理多个相邻 query，
复用 K/V tile。分块必须使用每行独立的 `past_length/q_len` 计算 causal mask，不能用 packed
token 全局索引判断可见性。

### 16.2 Prefill Workspace 上界

如果第一版使用两阶段 attention，workspace 应按本 tick 的真实 `T`、heads 和 tile 数从
预分配 arena 获取，而不是按：

```text
max_active_sequences * max_context_tokens * max_context_tokens
```

完整预留 score 矩阵会迅速形成显存墙。最终优化目标是 fused/online softmax，使 workspace
与 query rows 和归并 tiles 线性相关，而不是与 context 平方相关。

## 17. CPU Reference 必须先实现

在写 HIP trusted/fused Kernel 前，先实现标量 CPU 版本：

```text
PagedKvWriteReference
PagedDecodeAttentionReference
PagedPrefillAttentionReference
```

CPU 版本应显式执行 block translation、GQA mapping、causal mask 和 FP32 softmax。它不是
性能代码，而是所有 HIP 优化的数值 oracle。

建议比较：

- BF16/FP16 boundary 后的绝对/相对误差；
- 每个 row 独立输出；
- block 边界前后；
- 非连续 physical blocks；
- GQA 多个 Q Heads 共享一个 KV Head；
- past length 为 0、1、block_size-1、block_size、跨多个块。

## 18. 与 Scheduler 的生命周期集成

推荐 admission 流程：

```text
Queued request
  ↓
compute capacity policy
  ↓
Reserve
  ↓
Commit
  ↓
request becomes Active/Prefilling
```

容量策略有两种：

### 18.1 全请求预留

```text
reserve prompt_tokens + max_new_tokens at admission
```

优点：执行中不会因 KV OOM 中断；缺点：保守，未生成的上限也占块。

### 18.2 增量预留

```text
reserve prompt/chunk initially
reserve one or more blocks near capacity boundary
```

优点：池利用率更高；缺点：Decode 中途可能无法扩容，需要 preemption、失败或 admission
保证策略。

第一版建议全请求预留，待基础生命周期稳定后再实现增量预留。

## 19. 与 Continuous Batching 的接口

Continuous Batching 每个 tick 改变 batch membership，Paged KV 通过 SequenceId 保持历史
独立：

```text
Scheduler selects rows
      ↓
ReserveBatch + CommitBatch capacity
      ↓
BatchAssembler reads SequenceKvView
      ↓
PackedPagedBatch carries past lengths + block tables
      ↓
Runner executes one packed forward
      ↓
AdvanceBatch atomically commits logical lengths
```

Paged KV 不要求 sequence 固定在某个 batch row；row 只在当前 step 有意义。

## 20. 与 Tensor Parallel 的接口

Tensor Parallel 下，每个 Rank 只保存本地 KV Heads：

```text
local_kv_heads = global_kv_heads / tp_world_size
```

每个 Rank 创建独立 Paged KV Pool，配置中的 `num_kv_heads` 使用 local 值。各 Rank 接收
相同 SequenceId 和调度 step，因此 committed length、所需逻辑页数和 batch row 边界必须
一致。physical block index、block generation 和设备地址都是 Rank-local 状态，不要求数值
相同，也不能把一个 Rank 的 block table 直接交给另一个 Rank。

KV 不需要 AllReduce。Attention 在本地 Heads 上完成，O Projection 的 partial hidden 才
需要跨 Rank reduction。

## 21. 线程安全和所有权

推荐规则：

- Host allocator metadata 只由一个 Scheduler/Engine worker 修改；
- HTTP 线程只提交命令，不直接 Reserve/Release；
- GPU Pool 只由 owning Backend/Stream 使用；
- Runner 和 Cache 的生命周期长于所有请求；
- Release 必须发生在 terminal event 最终处理阶段；
- 不在持有 queue mutex 时同步 GPU；
- 不返回可修改内部 block table 的引用；
- 不允许多个 pending reservation 操作同一 sequence。

如果必须支持多 Host 线程直接访问 Cache，需要在 Cache 内部加锁，但锁不能跨 GPU
Synchronize 持有。第一版优先单所有者线程模型。

## 22. 错误语义

建议区分：

| 错误 | 状态 |
|---|---|
| 非法维度、重复 ID、stale view | InvalidArgument |
| 整数或地址范围溢出 | OutOfRange |
| KV Pool blocks 不足 | ResourceExhausted |
| HIP allocation/copy/kernel/sync 失败 | BackendError |
| generation/ownership 不变量破坏 | Internal |

错误必须携带上下文，但不得把 prompt、token 内容、权重或 KV 数据写入日志。

## 23. 分阶段实现顺序

### Phase A：Host Block Manager

- 配置和显存公式；
- free list、generation；
- Reserve/Commit/Rollback；
- ReserveBatch/CommitBatch/RollbackBatch capacity transaction；
- AdvanceBatch；
- View/Release；
- 纯 Host 单元测试。

### Phase B：Device Pool 和 CPU Paged Attention

- 单 Buffer K/V Pool；
- Layer TensorView；
- CPU KV write；
- CPU Decode/Prefill oracle；
- 非连续 block tests。

### Phase C：HIP KV Write 和 Decode Attention

- 上传 block tables/past lengths；
- Paged KV Write Kernel；
- Decode score/softmax/output；
- 与 CPU oracle 对齐；
- batch rows 隔离。

### Phase D：Ragged Prefill

- sequence offsets/token rows；
- Chunked Prefill；
- Paged Prefill Attention；
- batch 原子推进；
- mixed request lengths。

### Phase E：服务生命周期

- admission backpressure；
- cancel/failure/shutdown；
- terminal release；
- high-water metrics；
- steady-state allocation test。

### Phase F：性能优化

- trusted metadata path；
- fused decode attention；
- fused prefill attention；
- page table upload caching；
- vectorized BF16/FP16 load；
- block size tuning。

每个 Phase 必须保持前一阶段 oracle 可运行，不能用 fused Kernel 取代唯一正确性参考。

## 24. 必须测试的场景

### 24.1 Host allocator

- 配置非法和所有溢出分支；
- reserve 0、负数、重复 pending；
- capacity 已存在时不额外取块；
- pool exhaustion 状态不变；
- commit/rollback 只能消费一次；
- batch 任一 reservation 失败时 capacity 状态完全不变；
- CommitBatch 后所有行的 block table 同时可见；
- stale reservation；
- stale SequenceKvView；
- duplicate release；
- generation 跨复用递增；
- AdvanceBatch 任一行非法时全部不更新；
- 数万次申请/释放后所有 blocks 回收。

### 24.2 Attention correctness

- 单 block 和跨 block；
- physical blocks 打乱；
- 不同 row 使用不同 block table；
- past lengths 不同；
- ragged chunk lengths 不同；
- block boundary `15 -> 16 -> 17`；
- GQA head mapping；
- CPU/HIP BF16 容差；
- NaN/Inf 输入拒绝或标记。

### 24.3 生命周期

- Prefill 中取消；
- Decode 中取消；
- 模型 batch 失败；
- Sampling 失败；
- queue saturation；
- shutdown 时同时有 queued/active/in-flight；
- release 后 `free_blocks == total_blocks`；
- warmup 后 allocation count 不增长。

## 25. 性能测量方法

至少记录：

- block size：8/16/32；
- context：短、中、长；
- batch rows：1/2/4/最大；
- Decode Attention latency；
- Prefill Attention latency；
- page table upload time；
- KV Pool bytes；
- high-water blocks；
- steady-state allocation count；
- end-to-end TTFT、TPOT、throughput。

不要只测 Kernel。Paged KV 的价值还包括显存利用率、请求回收和动态 batching。

## 26. 常见错误

### 错误 1：只增加 slot 维度就称为 Paged KV

固定 `[slot][max_context]` 是多 slot contiguous KV，不是物理块页表。

### 错误 2：block table 只在 Host，Kernel 仍假设连续

如果 Attention 没有按 row 查表，分页只存在于元数据，没有进入执行语义。

### 错误 3：预留容量等同于 committed length

这会让 Attention 读取未初始化区域。

### 错误 4：batch 逐行 Advance

中途失败会产生部分提交，破坏 Scheduler 与 KV 一致性。

### 错误 5：释放不等待 GPU

新请求可能覆盖旧 Kernel 尚未读完的 block。

### 错误 6：没有 generation

旧 view 可能释放已经属于新请求的物理块。

### 错误 7：每请求 `hipMalloc`

失去 Pool 的稳态内存和延迟优势。

### 错误 8：把 Prefix Cache 当成基础分页的必要条件

Prefix sharing 是高级所有权模型，应在独占 block 生命周期正确后实现。

## 27. 完成定义

只有同时满足以下条件，才能声明 Paged KV Cache 已实现：

1. KV 由固定物理 block pool 管理，而非固定 sequence contiguous slot；
2. 每条 sequence 有独立 block table 和 committed length；
3. 单行与 batch Reserve/Commit/Rollback 都是原子 capacity transaction，且 generation 能拒绝 stale 操作；
4. Prefill/Decode Kernel 按 row 页表访问非连续 blocks；
5. batch logical length 原子推进；
6. 完成、取消、错误、shutdown 都能回收 blocks；
7. 稳态请求不产生设备 allocation；
8. CPU oracle 与 HIP 路径在 block boundary、ragged rows、GQA 下对齐；
9. metrics 能观测 current/high-water/free blocks；
10. 文档明确 Prefix Cache、Swap、Preemption 是否实现，不能混淆能力边界。
