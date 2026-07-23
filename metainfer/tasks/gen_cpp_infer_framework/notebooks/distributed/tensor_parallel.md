# Tensor Parallel：C++/HIP 多卡推理框架生成指南

> 本文用于指导 Agent 从零实现单机 Tensor Parallel，不依赖任何现有推理框架源码。
> 文中的“必须”表示正确性、通信顺序或生命周期契约，“建议”表示适合第一版实现的
> 工程选择。
>
> **适用边界：** Tensor Parallel、Paged KV Cache 和 Continuous Batching 是三个独立
> 能力。本文件的权重分片、Rank、Collective 和 group failure 规则属于 TP 基线；标有
> “组合能力”的章节只在对应能力同时被选中时生效。TP-only 使用每 Rank 本地的 contiguous
> FP32 KV，不得无条件创建 Paged allocator 或 Scheduler。

相关主题：[Paged KV Cache](../runtime/paged_kv_cache.md) ·
[Continuous Batching](../runtime/continuous_batching.md)

## 1. 目标与范围

Tensor Parallel（TP）把同一个 Transformer layer 的大矩阵分布到多张 GPU。每个 Rank
保存部分权重、计算部分结果，再通过 Collective 得到下一子层需要的逻辑完整 activation。

第一版建议限定为：

- 单机、单进程、`world_size=2`；
- 两个 HIP device 各有独立 Backend、Stream 和 RankContext；
- 使用非量化 GGUF 权重，Z200 首版以 F16 矩阵为基线；
- Attention heads 与 MLP intermediate channels 均匀切分；
- Q/K/V、Gate/Up 使用 Column Parallel；
- O、Down 使用 Row Parallel；
- 每层两次 AllReduce Sum；
- 每 Rank 独立保存本地 KV heads；TP-only 使用 contiguous KV，选择 Paged 后使用本地 Pool；
- Embedding、Norm 和 LM Head 先复制；
- Rank 0 产生唯一 Sampling 结果并广播 token；
- 自定义 HIP P2P TP2 Collective，或使用环境中可靠的 Collective 库；
- 任一 Rank 失败都能解除其他 Rank 的等待并整体退出。

第一版不必实现：

- 跨节点通信；
- `world_size > 2` 的高性能 Ring/Tree；
- Pipeline Parallel、Data Parallel 或 Expert Parallel；
- Sequence Parallel；
- Reduce-Scatter/AllGather activation 布局；
- Vocab Parallel；
- 通信计算重叠；
- 多进程 Rank 故障隔离。
- Q8_0 或其他 block-quantized 权重的 TP 分片与分布式反量化。

本契约中的“完整参数”表示非量化权重，不表示必须用 F32 存储。Z200 首版从
GGUF 读取 F16 矩阵并直接物化 Rank-local F16 shards；F32 权重可在加载时显式转换
为 F16。BF16 只在 Loader、Backend 和 gfx906 实卡测试都明确支持后启用。遇到
Q8_0 主矩阵时，基础 TP 实现必须返回清晰的 unsupported dtype，不得按 F16 字节
布局解释。更大模型的 Q8_0 TP 是后续独立能力。

当前能力编译器只接受 `tp_size=2`。先把 TP2 的数学、完整参数权重物化、通信顺序和服务
生命周期做正确；TP4/TP8 在 Collective、设备数量合同和验证矩阵补齐前必须前置拒绝。

当任务冻结了 `tp_size=P>1` 时，真实目标模型的权重加载、完整 Forward、生成和服务验收
必须始终使用 `P` 个 Rank。不得为了构造参考结果、调试或 parity，在一张卡上加载完整真实
权重。单 Rank 只用于显式限制尺寸的合成算子、缩小层或 Rank-local shard 测试；测试必须
给出其张量尺寸或显存上限。这样既避免把 TP1 fallback 误报为 TP 能力，也避免目标模型本来
就无法装入单卡时设计出不可执行的测试。

## 2. TP、DP 与 PP 的区别

| 策略 | 每张卡保存什么 | 一个请求如何执行 | 主要通信 |
|---|---|---|---|
| Data Parallel | 完整模型副本 | 只交给一个副本 | 请求分发或训练梯度 |
| Pipeline Parallel | 一部分连续 layers | 依次经过多个 stages | stage activation |
| Tensor Parallel | 每个 layer 的部分矩阵 | 同时经过所有 Ranks | layer 内 Collective |

Data Parallel 增加总吞吐，但通常不降低单请求显存需求。Pipeline Parallel 能容纳更深模型，
但单 microbatch 有 stage bubble。Tensor Parallel 同时降低每 Rank 的主要权重和 KV heads，
适合单模型无法放入一张卡，或希望多个设备共同完成同一 token 的场景。

TP 并不保证线性加速。Decode 的矩阵很小，切分后 GEMM 效率可能下降；每层新增同步；
Embedding、Norm、LM Head 等复制计算也限制加速比。因此必须先满足容量目标，再测性能。

## 3. Rank、World 与逻辑张量

```text
world_size P = 参与同一个 TP group 的 Rank 数
rank r       = [0, P) 中的逻辑编号
device       = Rank 对应的 HIP device ordinal
```

不要默认 `rank == device ordinal`。推荐显式映射：

```cpp
struct TensorParallelConfig {
  int32_t world_size = 1;
  int32_t rank = 0;
  int32_t device_ordinal = 0;
  int32_t coordinator_rank = 0;
  std::vector<int32_t> device_ordinals;
};
```

一个张量必须区分：

- global shape：数学模型中的完整尺寸；
- local shape：本 Rank 实际持有的尺寸；
- shard offset：本 Rank 对应 global tensor 的起始位置；
- replicated：每 Rank 都持有完整副本；
- partial：本 Rank 只有需要 Sum 的部分和。

类型系统或 descriptor 中应保留这些信息，避免把 local tensor 当成 global tensor。

## 4. 线性层存储约定

假设 checkpoint 中线性权重使用常见布局：

```text
W shape = [out_features, in_features]
Y = X W^T
X shape = [T, in_features]
Y shape = [T, out_features]
```

“Column Parallel”和“Row Parallel”描述逻辑矩阵乘法的切分，不是 C/C++ 内存中的行列名：

- Column Parallel 切 `out_features`，在 `[out, in]` 文件布局中是切连续行；
- Row Parallel 切 `in_features`，在 `[out, in]` 文件布局中是每行切一段列。

实现前必须把这个约定写入 `ShardSpec`，不能只凭函数名称猜 offset。

## 5. Column Parallel 数学

把输出维分成 `P` 份：

```text
W = concat(W_0, W_1, ..., W_{P-1}) along out_features
W_r shape = [out_features / P, in_features]
```

每 Rank 读取相同完整输入 `X`：

```text
Y_r = X W_r^T
Y_r shape = [T, out_features / P]
```

`Y_r` 是完整 `Y` 的一段，不是 partial sum，因此不需要立即通信。只要下一个算子也按相同
维度分片，就可以继续本地计算。

物理 `[out, in]` 中每 Rank 的 shard 是连续行：

```text
local_out = out_features / P
row_begin = rank * local_out
source_offset_elements = row_begin * in_features
local_elements = local_out * in_features
```

若 `out_features % P != 0`，第一版应拒绝加载，而不是悄悄丢行或构造不均匀 shard。

## 6. Row Parallel 数学

Row Parallel 接收已经按输入维分片的 activation：

```text
X = concat(X_0, X_1, ..., X_{P-1}) along in_features
W = concat(W_0, W_1, ..., W_{P-1}) along in_features
W_r shape = [out_features, in_features / P]
```

每 Rank 计算：

```text
Y_partial_r = X_r W_r^T
Y = sum_r(Y_partial_r)
```

所以 Row Parallel 输出必须执行 AllReduce Sum，之后每 Rank 都得到相同完整 `Y`。

在 `[out, in]` 布局中，输入列 shard 跨越每一行，不是一个连续大区间：

```cpp
for (int64_t out = 0; out < out_features; ++out) {
  const T* src = global + out * in_features + rank * local_in;
  T* dst = packed + out * local_in;
  std::copy_n(src, local_in, dst);
}
```

第一版可在 Host 权重加载阶段完成一次列打包。不要在每个 Forward 中切权重。

### 6.1 Bias 规则

若线性层有 bias：

- Column Parallel：bias 随输出 shard 切分，本地直接添加；
- Row Parallel：不能让每 Rank 都在 AllReduce 前添加完整 bias，否则结果是 `P * bias`；
- Row Parallel bias 应在 AllReduce 后添加一次，或只由一个 Rank 在 reduce 前添加。

即使目标模型没有 bias，通用接口也应明确该语义。

## 7. Transformer 层的标准切分

设模型配置：

```text
hidden_size       = H
intermediate_size = I
query_heads       = Nq
kv_heads          = Nkv
head_dim          = D
Nq * D            = Hq，常见情况下 Hq = H
world_size        = P
```

推荐分片：

| 权重 | Global shape `[out,in]` | 策略 | Rank-local shape |
|---|---:|---|---:|
| Q Projection | `[Nq*D, H]` | Column | `[Nq/P*D, H]` |
| K Projection | `[Nkv*D, H]` | Column | `[Nkv/P*D, H]` |
| V Projection | `[Nkv*D, H]` | Column | `[Nkv/P*D, H]` |
| O Projection | `[H, Nq*D]` | Row | `[H, Nq/P*D]` |
| Gate Projection | `[I, H]` | Column | `[I/P, H]` |
| Up Projection | `[I, H]` | Column | `[I/P, H]` |
| Down Projection | `[H, I]` | Row | `[H, I/P]` |

必须验证：

```text
Nq  % P == 0
Nkv % P == 0
I   % P == 0
```

如果 `Nkv < P` 或不能整除，可选择复制 KV heads、按 group 不均匀切分或专门的 head mapping，
但那是另一套明确设计。第一版应拒绝不支持的配置。

## 8. GQA 的本地 Head 映射

Grouped Query Attention 中：

```text
queries_per_kv_head = Nq / Nkv
```

均匀 TP 切分后每 Rank 有：

```text
local_query_heads = Nq / P
local_kv_heads    = Nkv / P
```

若全局 head 编号按连续范围分片：

```text
global_q_head_begin  = rank * local_query_heads
global_kv_head_begin = rank * local_kv_heads
```

本地 query head `qh` 对应本地 KV head：

```text
local_kv_head = qh / queries_per_kv_head
```

这要求 Q 和 KV shard 边界与 GQA group 对齐。不能只验证 `Nq % P` 而忽略 `Nkv % P`。

## 9. 每层数据流

一个标准 pre-norm Transformer layer：

```text
replicated hidden [T, H]
        │
        ├─ RMSNorm replicated
        ▼
Column Q/K/V
        │ local Q heads + local KV heads
        ├─ RoPE / QK Norm local
        ├─ write Rank-local KV (contiguous or paged by capability)
        ▼
local Attention [T, Nq/P, D]
        │ flatten local head dimension
        ▼
Row O Projection
        │ partial hidden [T, H]
        ▼
AllReduce Sum                         // Collective 1
        │ replicated attention output
        ├─ residual add
        ├─ RMSNorm replicated
        ▼
Column Gate/Up
        │ [T, I/P] + [T, I/P]
        ├─ local activation/SwiGLU
        ▼
Row Down Projection
        │ partial hidden [T, H]
        ▼
AllReduce Sum                         // Collective 2
        │ replicated MLP output
        └─ residual add -> replicated hidden [T, H]
```

每层恰好两次 `[T,H]` AllReduce 是基础实现的重要不变量。Q/K/V 和 Gate/Up 后没有
AllGather，因为其消费者 Attention 和 activation 都能在 local shard 上执行。

## 10. 为什么 Residual 流保持复制

Row Parallel AllReduce 后，每 Rank 获得相同 `[T,H]` hidden。这样：

- 下一层 RMSNorm 无需通信；
- Column Parallel 输入在所有 Rank 都可用；
- residual add 可以本地执行；
- 模型结构清晰，便于与单卡 oracle 对齐。

代价是 Norm、Residual 和部分 elementwise op 在每 Rank 重复。Sequence Parallel 可以减少
部分重复，但会引入 Reduce-Scatter/AllGather 和新的布局契约，不适合第一版混入。

## 11. Fused QKV 与 Gate/Up 的分片

checkpoint 可能分别存储 Q、K、V，但运行时希望拼为一次 Fused GEMM。GQA 下三段输出长度
不同，正确顺序是：

```text
for each rank:
  q_local = shard q_proj on its own output axis
  k_local = shard k_proj on its own output axis
  v_local = shard v_proj on its own output axis
  fused_local = concat(q_local, k_local, v_local)
```

不要先把 global Q/K/V 连接后再对总行数做等分，否则一个 Rank 可能拿到全部 Q 的尾部和
全部 K/V，head ownership 会错误。

Gate/Up 同理：

```text
gate_local = shard gate_proj outputs
up_local   = shard up_proj outputs
fused_local = concat(gate_local, up_local)
```

每个 local tensor 的 offset、shape 和拼接段界限都应保存在 descriptor 中供 Kernel 验证。

## 12. ShardSpec

```cpp
enum class ShardKind {
  kReplicated,
  kColumnParallel,
  kRowParallel,
};

struct TensorShape2D {
  int64_t rows;
  int64_t cols;
};

struct ShardSpec {
  std::string tensor_name;
  ShardKind kind;
  TensorShape2D global_shape;
  TensorShape2D local_shape;
  int64_t shard_axis;       // 0 for output, 1 for input, -1 replicated
  int64_t shard_begin;
  int64_t shard_length;
  int32_t rank;
  int32_t world_size;

  Status Validate() const;
};
```

`Validate()` 至少检查：

- Rank 和 world 合法；
- global/local 维度为正；
- replicated 的 local shape 等于 global shape；
- shard 范围不越界；
- 所有 Ranks 的 shard 无重叠且完整覆盖目标轴；
- dtype 与 byte size 乘法不溢出；
- tensor 实际 checkpoint shape 与声明一致。

不要把 Q/K/V 的 shard 规则散落在字符串判断中。建立模型参数到 `ShardSpec` 的集中映射表。

## 13. 权重物化 API

本节对主矩阵的基线契约是 GGUF `F16`。Column Parallel 取连续输出行，Row
Parallel 对每个输出行打包对应的输入列范围；物化后的 local shard 仍是普通 F16
矩阵，直接进入 local hipBLAS GEMM，不走 Q8_0 整矩阵反量化 workspace。Norm 等
F32/F16 小张量按 replicated 策略处理。

### 13.1 F16 local GEMM 的内存布局

`hipblasGemmEx` 使用 column-major 视图不代表输出需要再转置。对于数学上的
row-major `X[T,K] @ W[N,K]^T -> Y[T,N]`，标准调用为：

```cpp
hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
              N, T, K,
              &alpha, W, HIPBLAS_R_16F, K,
              X, HIPBLAS_R_16F, K,
              &beta, Y, HIPBLAS_R_32F, N,
              HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
```

这里 hipBLAS 看到的 `Y` 是 column-major `[N,T]`，其线性地址是
`token * N + feature`；这与下游 kernel 看到的 contiguous row-major `[T,N]`
完全相同。Q/K/V、O、Gate/Up、Down 和 LM Head 的独立 local GEMM 都遵循这条
规则。不得仅因 hipBLAS 是 column-major 就交换 `M/N`、插入转置 kernel 或再次
转置权重。只有实测 CPU reference 不一致且逐项地址推导确认失败后，才能修改
布局；诊断时必须记录 `(M,N,K,lda,ldb,ldc,transA,transB)`。

```cpp
struct TensorParallelModelConfig {
  int64_t hidden_size;
  int64_t intermediate_size;
  int64_t query_heads;
  int64_t kv_heads;
  int64_t head_dim;
  int64_t vocab_size;
  int64_t num_layers;

  Status ValidateForWorldSize(int32_t world_size) const;
};

class WeightArchive {
 public:
  Result<MaterializedWeights> MaterializeTensorParallel(
      Backend& backend,
      Stream& stream,
      const TensorParallelConfig& tp,
      const TensorParallelModelConfig& model) const;
};
```

推荐物化流程：

1. 从模型 config 读取所有 global shapes；
2. 验证 checkpoint tensor 名称、dtype、shape 和 byte range；
3. 为每个 tensor 生成 `ShardSpec`；
4. Column shard 直接取连续输出行；
5. Row shard 在 Host 端逐行打包输入列；
6. Replicated tensor 完整复制；
7. 按对齐要求计算一个或少量 Device allocation 布局；
8. 批量 H2D 到对应 Rank；
9. 同步加载 stream 后构造 TensorViews；
10. 运行 local shape 和 alias 验证。

所有尺寸必须从 config/checkpoint 读取。示例模型的层数、heads 或 intermediate size 不能
硬编码到加载器。

加载器必须在任何 device allocation 之前扫描所有主矩阵 dtype。基础 TP2 任务只有在
Q/K/V/O、Gate/Up/Down、Embedding 和 LM Head 等目标矩阵符合声明的非量化策略时
才能启动；不允许部分 Rank 用 F16、部分 Rank 意外走 Q8_0 路径。

## 14. Tied Embedding 与 LM Head

若模型的 LM Head 与 token embedding 权重共享：

```text
embedding.weight shape = [V, H]
lm_head.weight aliases embedding.weight
```

replicated MVP 中，每 Rank 只需物化一个 `[V,H]` allocation，并让两个 TensorView 指向同一
地址。不要在同一 Rank 重复复制两份。

如果后续只让 Rank 0 保存 LM Head，要注意 tied embedding 的影响：其他 Ranks 仍需要输入
embedding。可选方案是：

- 各 Rank 保留完整 embedding，只有 Rank 0 建立 LM Head alias；
- Rank 0 做 embedding lookup 后广播 hidden；
- 实现 Vocab Parallel embedding。

这三种方案的显存与通信不同，不能只删除其他 Rank 的 LM Head view 就认为完成优化。

## 15. 每 Rank 显存组成

粗略表示：

```text
rank_weight_bytes
  = sharded_qkvo_mlp_bytes / P
  + replicated_embedding_norm_other_bytes

rank_kv_bytes
  = num_layers
  * 2                         // K and V
  * num_blocks
  * block_size_tokens
  * (kv_heads / P)
  * head_dim
  * dtype_bytes

rank_runtime_bytes
  = activations + GEMM workspace + collective scratch
    + packed metadata + allocator overhead
```

因此 TP 权重占用通常大于理想的 `1/P`，因为 Embedding、Norm、可能的 LM Head 和 runtime
workspace 被复制。容量估算必须把 KV block pool 和 Collective scratch 算入，不能只看
checkpoint 文件大小。

## 16. Collective 抽象

```cpp
enum class ReduceOp {
  kSum,
};

struct CollectiveDescriptor {
  uint64_t sequence_number;
  ReduceOp op;
  DType dtype;
  int64_t element_count;
};

class Collective {
 public:
  virtual ~Collective() = default;

  virtual Status AllReduceInPlace(
      TensorView tensor,
      ReduceOp op,
      Stream& stream) = 0;

  virtual Status Broadcast(
      TensorView tensor,
      int32_t root,
      Stream& stream) = 0;

  virtual void Abort(Status reason) = 0;
};
```

每次 Collective 必须让所有 Ranks 对以下 descriptor 达成一致：

```text
sequence_number
operation
dtype
element_count / bytes
root（若有）
```

只验证指针非空不够。Rank 0 reduce 5 KiB、Rank 1 reduce 20 KiB 可能造成越界或永久等待。

## 17. RankContext

```cpp
struct RankContext {
  TensorParallelConfig config;
  std::unique_ptr<Backend> backend;
  std::unique_ptr<Stream> compute_stream;
  std::unique_ptr<Collective> collective;
  MaterializedWeights weights;
  std::unique_ptr<RankLocalKvStore> kv_cache; // dense or paged by frozen capability
};
```

创建顺序：

```text
validate group config
  -> create every device context
  -> validate/enable peer access or initialize collective library
  -> allocate fixed collective scratch
  -> materialize all Rank weights
  -> create Rank-local KV pools and runners
  -> group-wide readiness barrier
  -> begin serving
```

任一 Rank 初始化失败都必须销毁整个 group。不能让部分 Rank 进入服务循环。

## 18. TP2 HIP P2P 前提

自定义 TP2 P2P 路径启动时必须双向检查：

```cpp
hipDeviceCanAccessPeer(&can_0_to_1, device0, device1);
hipDeviceCanAccessPeer(&can_1_to_0, device1, device0);
```

两边都可访问后，在各自 device context 中启用 peer access。`already enabled` 应作为幂等
成功处理，其他错误必须中止 group。

还需验证：

- 两个 Rank 使用不同 device；
- 所有数据指针属于声明的 device；
- peer pointer 在 group 生命周期内保持有效；
- HIP runtime 支持跨 device stream wait event 的目标用法；
- topology 性能是否满足需求。

若 P2P 不可用，第一版应明确拒绝启动或选择受支持的 Collective backend，不要无提示地
通过 Host staging 实现极慢路径。

## 19. 为什么需要只读 scratch

每 Rank 的 Row Parallel GEMM 产生：

```text
partial_r [T,H]
```

错误的原地两卡求和：

```text
Rank 0 reads Rank 1 partial while Rank 1 overwrites it
Rank 1 reads Rank 0 partial while Rank 0 overwrites it
```

读取和写入重叠会产生时序相关结果。正确方法是每 Rank 先保存不可变输入：

```text
partial_r
  └─ D2D copy -> local_scratch_r

output_r = local_scratch_0 + local_scratch_1
```

两个 reduction kernel 各自写本地 output，但只读两块 scratch。scratch 只有在所有 peer
读取完成后才能被下一 generation 覆盖。

建议按最大 `[max_batched_tokens,H]` 在启动时分配 scratch，避免 Forward 中扩容导致 peer
pointer 失效或隐式同步。

## 20. TP2 P2P AllReduce 协议

一个安全的异步 generation 可按以下顺序实现：

```text
Host control plane:
  1. both Ranks submit identical descriptor
  2. descriptor exchange validates seq/dtype/count/op
  3. mismatch sets group-wide abort

On each Rank stream:
  4. wait until peer completed reading this Rank's previous scratch generation
  5. D2D copy partial -> local read-only scratch
  6. record local ready event

Host enqueue barrier:
  7. both Ranks confirm ready event has been enqueued

On each Rank stream:
  8. wait peer ready event
  9. launch reduce(local scratch, peer scratch -> local output)
 10. record local done event

Host enqueue barrier:
 11. both Ranks confirm done event has been enqueued
```

下一次复用 `local_scratch_r` 前，Rank r 必须等待 peer 对上一 generation 记录的 done event，
因为 peer kernel 才是该 scratch 的读者。

Event 建议使用禁用 timing 的轻量模式。不要在每次 AllReduce 中 `hipDeviceSynchronize()`；
同步关系应进入 stream/event DAG。

Host `Abort()` 可以解除尚未把 peer wait 提交到设备的线程等待；它不能可靠撤销一个已经
因设备致命错误而永久阻塞的 GPU stream。ready enqueue barrier 的意义之一，就是保证不会
在 peer 尚未 record event 时提交设备 wait。遇到不可恢复的 device fault，服务应退出进程
或重建整个设备运行时，不能继续复用该 TP group。

### 20.1 Reduction Kernel

第一版至少支持模型 activation dtype：

```text
BF16 input: convert to FP32, sum, cast to BF16
FP16 input: convert to FP32, sum, cast to FP16
F32 input:  FP32 sum
```

向量化加载前验证指针对齐和 element count 尾部。Kernel grid-stride loop 应覆盖任意合法
count，不能假设 `[T,H]` 总能整除 vector width。

### 20.2 为什么不能顺序执行两个 Rank

以下控制流会在第一个 Collective 死锁：

```cpp
rank0_runner.Forward();  // waits for rank1 in layer 0
rank1_runner.Forward();
```

两个 Rank 必须并发进入模型图，可使用：

- 两个常驻 Host Rank worker；
- 一个主线程 + 一个 peer worker；
- 多进程 Rank + Collective library。

第一版推荐常驻线程，避免每 tick 创建线程，同时便于 group-wide shutdown。

## 21. Barrier 与 Group-wide Abort

最危险的故障：

```text
Rank 0 在第 n 次 Collective 前返回错误
Rank 1 已在第 n 次 Collective 等待 Rank 0
```

普通 barrier 会永久阻塞。所有 descriptor exchange、Host barrier 和等待队列都必须观察
共享终止状态：

```cpp
class CollectiveGroupState {
 public:
  void Abort(Status first_error);
  bool aborted() const;
  Status abort_reason() const;

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool aborted_ = false;
  Status first_error_;
};
```

规则：

- 首个错误成为 group error；
- `Abort()` 幂等并 `notify_all()`；
- 所有 wait predicate 都包含 `aborted`；
- Abort 后 group 不再用于新请求；
- 协调器等待所有 Rank worker 退出后再销毁 event/scratch；
- 错误信息包含 rank、collective sequence 和 descriptor。

Host timeout 可以用于诊断，但不能把“超时后继续复用 group”当作恢复。设备执行已不对称时，
最安全的基础语义是整体失败。

## 22. `world_size > 2` 的边界

TP2 的“每 Rank 直接读所有 peer scratch 并求和”不应直接扩展到大量设备：

- peer 连接数是 `O(P^2)`；
- 每个 Rank 读取 `P` 份数据；
- topology 可能不支持全互联；
- barrier 和 event 管理迅速复杂化。

扩展 TP4/TP8 时建议引入经过验证的 Collective library，实现 Ring/Tree AllReduce，或明确
设计 Reduce-Scatter + AllGather。保留 `Collective` 抽象，使模型层不依赖 P2P 细节。

## 23. Paged KV Cache 的 Rank-local 语义（组合能力）

本节只在 `tensor_parallelism + paged_kv_cache` 时生效。TP-only 保留相同的 local head
ownership 和 committed length 规则，但地址是 contiguous `[position, local_kv_head, D]`，
没有 block id、block table 或 ReserveBatch。

Q/K/V heads 已分片，所以每 Rank 只缓存本地 KV heads：

```text
Rank 0 KV Pool: global KV heads [0, local_kv_heads)
Rank 1 KV Pool: next local_kv_heads
...
```

KV 不做 AllReduce，也不在 Ranks 间复制。每 Rank 的 Attention 使用本地 query heads 与对应
本地 KV heads。

同一逻辑 sequence 在每 Rank 有一个本地 cache state：

```text
logical sequence id     相同
committed token length  必须相同
block count             配置相同时通常相同
physical block index    只在本 Rank Pool 内有意义
device pointer          必然不同
```

不能把 Rank 0 的 block table 指针广播给 Rank 1。协调器广播逻辑 batch；每 Rank Batch
Assembler 查询自己的 block table。

### 23.1 KV 提交

一次 TP step 只有在所有 Ranks 的模型执行成功后才能逻辑提交：

```text
ReserveBatch on every Rank
  -> all Ranks report reservation success
  -> CommitBatch capacity on every Rank
  -> assemble each Rank-local block table
  -> all Rank forwards
  -> group success
  -> AdvanceBatch on every Rank
  -> Scheduler Apply once
```

任一 Rank Reserve 失败时，在任何 capacity commit 之前回滚所有 Rank reservations。
全部 reservation 成功后，各 Rank 在单所有者线程上执行已验证、不会部分失败的
`CommitBatch()`，然后才能组装 block table 和执行 Forward。任一 Rank Forward 失败时
整个请求组失败，不执行 `AdvanceBatch()`，并在 GPU 安全点后释放相关 sequences。

## 24. Continuous Batching 集成（组合能力）

本节只在 `tensor_parallelism + continuous_batching` 时生效。是否携带 block table 继续由
Paged KV 开关决定；Continuous-only 组合在每 Rank 使用相同 logical row 和不同的本地
dense sequence slot。

推荐一个逻辑 Scheduler 产生 `StepPlan`：

```text
Scheduler::BuildNext
        │ request ids, token ids, positions, q lengths
        ▼
TP Coordinator
        ├─ Rank 0 Assemble with local KV view
        ├─ Rank 1 Assemble with local KV view
        └─ ...
        ▼
concurrent Rank Forward
        ▼
single logical result
        ▼
Scheduler::Apply
```

必须保证所有 Ranks：

- 处理相同 request id 顺序；
- 输入相同 token ids 和 positions；
- Prefill/Decode row 边界相同；
- 进入相同数量和顺序的 Collectives；
- 使用相同 `T` 和 `[T,H]` reduce count；
- 观察到相同取消/失败线性化点。

不要让每个 Rank 的 Scheduler 独立基于本地时间或本地随机数选 batch。

## 25. TP Coordinator

```cpp
struct RankStepResult {
  int32_t rank;
  Status status;
  std::optional<TensorView> final_hidden;
};

class TensorParallelEngine {
 public:
  Result<ModelBatchResult> Execute(const StepPlan& plan);
  Status Shutdown();

 private:
  std::vector<std::unique_ptr<RankWorker>> ranks_;
  std::shared_ptr<CollectiveGroupState> group_;
};
```

一次 `Execute`：

1. 给所有 Rank worker 发布同一个 plan generation；
2. 每 Rank 通过 `PrepareCapacity()` 检查本地 KV，任一失败则全体回滚；
3. group preparation barrier 后在每 Rank提交 capacity；Paged 组合映射为
   `ReserveBatch/CommitBatch`，dense KV 只验证已拥有的 sequence slot；
4. 每 Rank 从本地 KV view 组装 dense 或 paged metadata；
5. 所有 Rank 并发 Forward；
6. 等待每个 Rank 的 completion event/fence，并收集实际设备执行 status；
7. 任一失败则 Abort、不 Advance 逻辑长度、清理 sequence 并返回 group error；
8. 全部成功后在每 Rank 执行 `AdvanceCommittedLength()`；
9. 从指定 Rank 获取 sampled tokens；
10. 返回一个稳定 row 顺序的逻辑 result。

Rank worker 命令队列应有 generation，拒绝重复、跳号或 shutdown 后的命令。
Forward 仅成功 enqueue kernels 不能作为 KV 提交依据；completion fence 必须覆盖最后一个
使用本 step KV、Collective scratch 和输出 buffer 的设备操作。

## 26. LM Head 与 Sampling 策略

Transformer 最后一层 AllReduce 后，各 Rank 理论上有相同 final hidden。接下来有三种方案。

### 26.1 Replicated LM Head，Rank 0 采样

每 Rank 保存完整 `[V,H]`，但只有 Rank 0 执行 LM Head 和 Sampling：

```text
all Ranks final hidden replicated
Rank 0: LM Head -> sample tokens
Rank 0: broadcast selected token ids
```

如果非 Rank 0 不需要本地 logits，这比每 Rank 重复巨大 LM Head 更合理。其缺点是每 Rank
仍可能因 tied embedding 保存完整 vocabulary weight。

最保守 MVP 也可以每 Rank 都执行 LM Head，但必须只采用一个 Rank 结果；调试模式可比较
Greedy token 一致性。随机采样绝不能每 Rank 独立执行后各自进入下一 step。

### 26.2 Rank 0-only Weight

只有 Rank 0 保存/执行 LM Head，可减少其他 Rank 权重和计算。若 embedding tied，需按第 14
节处理其他 Rank 的 embedding 需求。采样 token 广播 payload 很小。

### 26.3 Vocab Parallel

按 vocabulary rows 切 `[V,H]`：

```text
Rank r local logits [T, V/P]
```

Greedy 可以先求 local `(value, global_token_id)`，再对 `P` 个候选做全局 max。完整
temperature/top-p Sampling 更复杂，通常需要：

- 全局最大值与 softmax sum reduction；
- 分布式概率质量或候选集合；
- 唯一 RNG 决策；
- token id 广播。

不要把“local top-k 后随便合并”当作严格 top-p；候选截断可能丢失累计概率质量。

## 27. 输入 Embedding 策略

### 27.1 Replicated Embedding

每 Rank 保存完整 `[V,H]`，独立 lookup 相同 token ids，得到相同 hidden。实现最简单，没有
每 token embedding 通信。

### 27.2 Vocab Parallel Embedding

每 Rank 只保存 vocabulary 范围：

```text
if token belongs to rank:
    local_hidden = embedding[token - vocab_begin]
else:
    local_hidden = 0
hidden = AllReduceSum(local_hidden)
```

它降低复制权重，但在模型入口新增 Collective。可对 Prefill/Decode token batch 一次执行，
并与 Vocab Parallel LM Head 共享分片。第一版通常不需要。

## 28. RoPE、Norm、Residual 与 Elementwise

- RMSNorm 在复制 hidden 上由每 Rank本地执行；
- Q/K Norm 在 local heads 上执行，不通信；
- RoPE position ids 必须所有 Rank 相同；
- Attention causal/KV metadata 的逻辑长度必须相同；Paged 组合还要求 block-table view 有效；
- Residual add 应在 AllReduce 后用相同 residual 本地执行；
- SwiGLU 在 local intermediate shard 上执行；
- dropout 在推理中应关闭。

如果尝试 fusion，例如 `AllReduce + Residual + RMSNorm`，Collective 的输出仍是逻辑同步点。
融合不能改变“partial hidden 先跨 Rank 求和，再应用完整 residual/norm”的数学顺序。

## 29. 数值精度

TP 与单卡可能不是逐 bit 一致，因为归约顺序改变：

```text
single GEMM accumulation
vs.
sum(partial_0, partial_1, ...)
```

建议：

- GEMM 按后端推荐使用 FP32 accumulation；
- BF16/FP16 AllReduce 在 Kernel 内转 FP32 求和；
- 正确性测试使用绝对/相对容差；
- 同时比较最终 token，不能只比较 hidden；
- 避免在每 Rank 对同一个 bias/residual 重复计数；
- NaN/Inf 检测至少可在调试模式启用。

对 Greedy 而言，两个非常接近的 logits 可能因浮点顺序改变 token。测试应同时记录 logits
误差和 top-1 margin，合理区分数值容差与真实分片错误。

## 30. 服务生命周期

协议层应把 TP Engine 当成一个普通逻辑模型：

```text
HTTP request
  -> one logical Generate, or Scheduler Submit when Continuous Batching is selected
  -> one request step or one packed Scheduler step
  -> all Rank workers execute
  -> one token event
  -> JSON/SSE
```

客户端不需要知道 Rank 数量。设备列表、world size 和 collective backend 属于服务启动配置。

启动时建议记录：

```text
world_size
rank-to-device mapping
model global/local shapes
collective backend
peer access matrix
per-rank weight/KV/scratch bytes
```

不要记录模型权重内容、用户 Prompt 或敏感路径。

## 31. Shutdown 顺序

```text
1. stop admission
2. stop creating new Scheduler plans when Continuous Batching is active
3. mark group shutting down
4. finish or abort current collective generation
5. wake every Rank worker and Host barrier
6. join Rank workers
7. synchronize each owning stream
8. release Rank-local KV, workspace and weights
9. destroy peer events and disable/destroy collective resources
10. destroy streams/backends
```

设备错误路径也必须能执行第 5 步，否则一个 Rank 的 worker 可能永久等待。`Shutdown()` 和
`Abort()` 都应幂等，且不能在 Rank worker 内 join 自己。

## 32. 性能模型

粗略表示一层 TP 时间：

```text
T_layer_tp
  ≈ T_sharded_gemm(P)
  + T_local_attention(P)
  + 2 * T_allreduce(T * H * dtype_bytes, P)
  + T_replicated_ops
```

全模型：

```text
T_token_tp
  ≈ num_layers * T_layer_tp
  + T_embedding
  + T_final_norm
  + T_lm_head_sampling
```

AllReduce payload 每层每次约为：

```text
payload_bytes = T * H * dtype_bytes
```

Decode `T` 小，通信受固定延迟主导；Prefill `T` 大，更受带宽影响。即使字节数不大，每层
两次同步也可能产生 launch gap。

理论权重 FLOPs 约缩至 `1/P`，实际加速受以下因素限制：

- local GEMM 太小，GPU 利用率下降；
- 两次 layer AllReduce latency；
- Norm/Residual 等复制计算；
- Attention local head 数减少后的 Kernel 效率；
- replicated LM Head/Sampling；
- Rank 间负载或频率不一致；
- Host barrier 和 stream 同步；
- P2P topology。

## 33. 性能测量

### 33.1 Weight 与容量报告

对每 Rank 报告：

```text
sharded weight bytes
replicated weight bytes
KV pool bytes
collective scratch bytes
runner workspace bytes
peak device bytes
```

验证“能放下模型”时必须用运行期峰值，不是只看权重 allocation。

### 33.2 Collective microbenchmark

覆盖实际 payload：

```text
T = 1, 4, 16             // Decode batch
T = 32, 128, 512         // Prefill chunks
bytes = T * H * dtype
```

记录：

```text
enqueue latency
GPU elapsed latency
effective payload bandwidth
p50/p95/p99
warm/cold difference
连续多 generation 稳定性
```

带宽口径要说明是 payload bytes/time，还是按 AllReduce 算法统计总读写流量。

### 33.3 GEMM microbenchmark

对比 global 与 local shape：

```text
QKV:     [T,H] x [local_qkv,H]^T
O:       [T,local_q] x [H,local_q]^T
Gate/Up: [T,H] x [2*I/P,H]^T
Down:    [T,I/P] x [H,I/P]^T
LM Head: [T,H] x [V,H]^T
```

分别测 Decode 小 `T` 和 Prefill 大 `T`。TP 负提升常来自 local GEMM shape 效率，而不一定
来自 P2P 字节量。

### 33.4 End-to-end

在相同模型、精度、Prompt/Output、Sampling、batch workload 和测量边界下比较：

```text
单卡 vs TP2
TTFT
TPOT / ITL
prompt throughput
generation throughput
peak memory per Rank
```

同时抓 Kernel timeline，分解 GEMM、Attention、AllReduce、LM Head、Host gap。不能用一个
microbenchmark 推断完整服务瓶颈。

## 34. 分阶段实现顺序

### 阶段 A：Host 分片 oracle

- `TensorParallelConfig/ShardSpec`；
- 扫描 GGUF 主矩阵 dtype，验证基础路径为非量化 F16；
- Column 连续行切分；
- Row 逐行列打包；
- fused QKV/Gate-Up 分段规则；
- CPU linear partial sum 与完整 linear 对齐。

### 阶段 B：Collective oracle

- Host fake AllReduce；
- descriptor sequence validation；
- mismatch 和 Abort；
- 两个并发 Rank worker 控制流。

### 阶段 C：HIP TP2 P2P

- 双向 peer capability；
- 固定 scratch、ready/done events；
- BF16/FP16/F32 reduction；
- 多 generation 和故障解除测试；
- microbenchmark。

### 阶段 D：单 Layer TP2

- QKV local heads；
- local Attention/KV；
- O AllReduce；
- Gate/Up/Down AllReduce；
- 与 CPU reference 或缩小的合成 TP1 layer oracle 对齐；该 oracle 不加载完整真实权重。

### 阶段 E：完整模型

- 所有 layers 和 final norm；
- replicated Embedding/LM Head；
- Rank 0 Sampling + token broadcast；
- 真实目标模型始终按冻结的 `tp_size` 运行；
- 单请求 Prefill/Decode 使用有限值、确定性参考 fixture 和同一 TP 拓扑下不同执行模式对齐；
- 不构造完整真实权重的 TP1/单卡 reference。

### 阶段 F：服务与可选动态 batch

- 只有选择 Continuous Batching 时增加中央 Scheduler 和 Packed Decode/Ragged Prefill；
- 只有选择 Paged KV 时增加每 Rank local Pool 和 block tables；
- Cancel、error、shutdown；
- HTTP/SSE 与压力测试。

### 阶段 G：可选优化

- Rank 0-only LM Head；
- Vocab Parallel；
- Collective library 与 TP4；
- persistent kernels 或通信计算重叠；
- Sequence/Pipeline Parallel。
- 大模型 Q8_0 权重分片、block-aligned packing 与对应 local linear。

## 35. 测试矩阵

### 35.1 分片测试

- 每个 Column shard 与 global 对应行逐元素一致；
- 每个 Row shard 与 global 对应列逐元素一致；
- 所有 Ranks 合并后完整覆盖且不重叠；
- GQA Q/K/V 独立切分后 head ownership 正确；
- fused QKV 和 Gate/Up 段边界正确；
- tied weight 保持预期 alias；
- 非整除 shape 被明确拒绝；
- shape/dtype/byte overflow 被拒绝。
- F16 主矩阵被正确分片，Q8_0 主矩阵在未启用量化 TP 扩展时被明确拒绝。

### 35.2 线性代数测试

- concat(Column outputs) 与完整 GEMM 对齐；
- sum(Row partials) 与完整 GEMM 对齐；
- Row bias 只添加一次；
- BF16/FP16/F32 使用合理容差；
- `T=1` 和 ragged packed `T>1` 均覆盖。

### 35.3 Collective 测试

- 支持的 dtype 和任意合法 count；
- count 小于一个 wave 和存在 vector tail；
- 多 generation 连续复用；
- descriptor dtype/count/op/sequence mismatch 双侧失败；
- Rank 在 barrier 前、ready 后、kernel 后失败均能解除对端；
- Abort/Shutdown 幂等；
- scratch 在 peer 读完前不会被覆盖；
- 错误 device pointer 被拒绝。

### 35.4 模型正确性测试

- 缩小的合成单 Layer hidden 可与 CPU/TP1 oracle 对齐，并明确尺寸和显存上限；
- 真实模型 Prefill 完整 logits 在冻结 TP 拓扑下满足有限值和确定性 reference fixture；
- 多步 Decode KV 与 token 在任务实际选择的 contiguous/paged、单请求/packed 路径中对齐；
- 选择 Paged KV 时增加跨 block Paged Attention 对齐；
- GQA local head mapping 不串 Rank；
- final Greedy token 与冻结 TP 拓扑的 reference fixture 一致或有可解释的 near-tie；
- 每层 Collective 次数和顺序符合预期。

`tensor_parallel.numeric_parity` 分成两层证据：缩小的合成 TP1 与 TP-P 算子/Layer 数值
对齐，以及真实目标模型在 TP-P 下的有限 logits、确定性输出和同拓扑执行模式对齐。前者
验证分片与 Collective 数学，后者验证真实权重全链路；两者都不能被“完整模型 TP1”替代。

### 35.5 动态服务测试（仅 Continuous Batching）

- 多请求不同 Prompt/Output 长度；
- Packed Decode 和 Ragged Prefill 的 Rank batch metadata 一致；
- 请求动态加入退出时 token 不分叉；
- queued/active/in-flight cancel；
- 任一 Rank 模型失败导致 group-wide terminal；
- HTTP JSON/SSE 只发布一个逻辑 token stream；
- idle/in-flight/device error 下 shutdown 不死锁；
- 退出后所有 Rank 显存和 Host worker 被回收。

### 35.6 性能与稳定性测试

- Collective 实际 shape microbenchmark；
- local GEMM shape microbenchmark；
- 缩小合成 workload 可做 TP1/TP2 多轮基线；真实目标模型只测冻结 TP 拓扑；
- 长时间 Continuous Batching 压力；
- 无每 step `hipMalloc`、线程创建或 device synchronize；
- timeline 中 Ranks 的 Collective generation 对齐。

## 36. 常见错误

### 错误 1：把 `[out,in]` 的 Row/Column 切反

会得到 shape 看似合理但数值完全错误的权重。

### 错误 2：Global fused QKV 直接等分

GQA 的 Q/K/V 段长度不同，必须各自切分后再 local concat。

### 错误 3：Row Parallel 后没有 AllReduce

下一层拿到的只是 partial hidden，不是模型定义中的 activation。

### 错误 4：Row bias 每 Rank 添加

AllReduce 后 bias 被放大 `P` 倍。

### 错误 5：顺序调用 Rank Forward

Rank 0 在首个 Collective 等待尚未启动的 Rank 1，形成死锁。

### 错误 6：原地读取 peer 正在改写的 partial

缺少只读 scratch 和 done 保护，结果依赖时序。

### 错误 7：Collective 只匹配 bytes，不匹配 generation

一个 Rank 少进入一次 Collective 后，后续相同大小的操作可能错误配对。

### 错误 8：Rank 失败不唤醒 peer

服务错误路径永久挂起，无法 shutdown。

### 错误 9：跨 Rank 共享物理 KV block table

block index 和 device address 都是 Rank-local 的。

### 错误 10：每 Rank 独立随机采样

下一 token 不同后，所有 Collective shape 可能仍相同但语义已经分叉。

### 错误 11：把权重 `1/P` 当作总显存 `1/P`

复制权重、KV、workspace 和 Collective scratch 仍占显存。

### 错误 12：只看通信 microbenchmark

小 local GEMM、重复 LM Head 或 Host gap 也可能是主要瓶颈。

## 37. 完成定义

只有同时满足以下条件，才能声明基础 Tensor Parallel 已实现：

1. 所有 global/local shapes 与切分轴来自模型 config，未硬编码模型尺寸；
2. Q/K/V、Gate/Up、O、Down 的分片符合 Column/Row 数学；
3. GQA heads 和 intermediate channels 满足并验证 divisibility；
4. 每层 O 与 Down 后各有一次正确 AllReduce Sum；
5. 所有 Ranks 并发执行并严格匹配 Collective descriptor/generation；
6. 自定义 P2P 路径有只读 scratch、ready/done 依赖和 group-wide Abort；
7. 每 Rank KV 只保存本 Rank KV heads，逻辑长度在所有 Ranks 一致；选择 Paged 时再验证
   local block table 和 group reservation；
8. 选择 Continuous Batching 时由一个逻辑 Scheduler 驱动相同动态 batch，结果只 Apply 一次；
9. Sampling 产生唯一 token 并同步给所有 Ranks；
10. 分片、线性层、Collective、单 Layer、完整模型和服务测试均通过；
11. 任一 Rank 错误、取消和 shutdown 都不会造成死锁或提前释放；
12. 性能报告同时包含容量、Collective、local Kernel、端到端指标和 timeline；
13. 文档明确 replicated LM Head、Rank 0-only 或 Vocab Parallel 中实际采用哪一种；
14. 对 `world_size`、跨节点和高级并行能力的边界没有夸大。
15. 基础 TP 使用声明的非量化 GGUF 权重路径，未把 Q8_0 字节布局误当 F16；
    若量化 TP 未实现，能力边界和错误信息必须明确。
