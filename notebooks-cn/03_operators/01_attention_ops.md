# 注意力算子

## 概述

注意力是 LLM 推理中**对性能影响最大**的算子。不同阶段（prefill 与 decode）以及不同的 KV 缓存布局，需要不同的内核实现。本文档归纳三种主流路线在**内核层面**的要点。

## 注意力内核分类

```
注意力内核
├── Prefill / Extend（预填 / 延长）
│   ├── Flash Attention 变长（nano-vllm）
│   ├── Triton extend_attention（nano-sglang）
│   └── Flash Attention / FlashInfer 封装（mini-sglang）
├── Decode（解码）
│   ├── 带 KV 缓存的 Flash Attention（nano-vllm）
│   ├── Triton token_attention（nano-sglang）
│   └── FlashInfer 分页解码（mini-sglang）
└── KV Cache 写入
    ├── Triton store_kvcache（nano-vllm）
    ├── Python 层切片（nano-sglang）
    └── 自定义 store_cache 内核（mini-sglang）
```

## Prefill 注意力

### Flash Attention 变长（nano-vllm）

用于在**单个 batch** 内处理**变长**序列：

```python
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,                  # [total_q_tokens, num_heads, head_dim]
    k,                  # [total_kv_tokens, num_kv_heads, head_dim]
    v,                  # [total_kv_tokens, num_kv_heads, head_dim]
    cu_seqlens_q,       # [batch_size + 1]，Q 的累积序列长度
    cu_seqlens_k,       # [batch_size + 1]，KV 的累积序列长度
    max_seqlen_q,       # int，batch 内 Q 的最大序列长度
    max_seqlen_k,       # int，batch 内 KV 的最大序列长度
    causal=True,        # 因果掩码
)
# output: [total_q_tokens, num_heads, head_dim]
```

启用前缀缓存时，接口还可接受 `block_table`，以便从**分页内存**中读取已缓存的 KV。

### Triton Extend Attention（nano-sglang）

自定义**两阶段** Triton 内核：同时处理**前缀（已缓存）**与**延长（新 token）**：

```python
def extend_attention_fwd(
    Q_Extend,        # [total_extend_tokens, num_heads, head_dim]
    K_Extend,        # [total_extend_tokens, num_kv_heads, head_dim]
    V_Extend,        # [total_extend_tokens, num_kv_heads, head_dim]
    O_Extend,        # 输出: [total_extend_tokens, num_heads, head_dim]
    K_Buffer,        # KV 池: [pool_size, num_kv_heads, head_dim]
    V_Buffer,        # KV 池: [pool_size, num_kv_heads, head_dim]
    Req_to_tokens,   # [max_batch, max_ctx_len] — 映射到物理下标
    B_req_idx,       # [batch_size] — 在 Req_to_tokens 中的请求下标
    B_Seq_Len,       # [batch_size] — 总序列长度（前缀 + 延长）
    B_Start_Loc_Extend,  # [batch_size] — 在 Q_Extend 中的起始偏移
    B_Seq_Len_Extend,    # [batch_size] — 延长段长度
    sm_scale,            # 1/sqrt(head_dim)
    kv_group_num,        # num_q_heads // num_kv_heads
):
```

**内核逻辑**：

1. **阶段 1（前缀）**：对每个 query token，通过 `Req_to_tokens` 间接读取 `K_Buffer/V_Buffer`，与**全部已缓存前缀 token** 计算注意力分数。
2. **阶段 2（延长）**：与**新（延长）token** 计算注意力分数，并施加因果掩码。
3. **合并**：用在线 softmax 合并两部分的注意力结果。

**分块**：`BLOCK_M=128, BLOCK_N=128, num_warps=8`（当 head_dim > 64 时）。

### Flash Attention 后端（mini-sglang）

封装 `sgl-kernel` 的 Flash Attention 实现：

```python
class FlashAttentionBackend(BaseAttnBackend):
    def forward(self, q, k, v, layer):
        output = flash_attn_with_kvcache(
            q=q,
            k_cache=self.kv_cache.k_buffer(layer),
            v_cache=self.kv_cache.v_buffer(layer),
            page_table=self.page_table,
            cache_seqlens=self.cache_seqlens,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k_new=self.cu_seqlens_k_new,
            max_seqlen_q=self.max_seqlen_q,
            causal=True,
            softmax_scale=self.sm_scale,
        )
        return output
```

## Decode 注意力

### 带 KV 缓存的 Flash Attention（nano-vllm）

```python
from flash_attn import flash_attn_with_kvcache

output = flash_attn_with_kvcache(
    q,                   # [batch_size, 1, num_heads, head_dim]（每请求 1 个 token）
    k_cache,             # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,             # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table=block_table,  # [batch_size, max_blocks] — 物理块下标
    cache_seqlens=context_lens,  # [batch_size] — 实际序列长度
    causal=True,
)
```

### Triton Token Attention（nano-sglang）

面向**单 token query** 优化的**两阶段** Triton 内核：

```python
# 阶段 1：计算注意力 logits
# 网格: (batch, num_heads, num_blocks_per_seq)
def _fwd_kernel_stage1(Q, K_Buffer, Req_to_tokens, Att_Out, ...):
    """
    对每个 query token，与一块 KV cache token 计算 Q·K^T。
    输出: 部分注意力 logits [batch, num_heads, num_blocks, BLOCK_N]
    """

# 阶段 2：Softmax 并与 V 归约
# 网格: (batch, num_heads, 1)
def _fwd_kernel_stage2(Att_Out, V_Buffer, Req_to_tokens, Out, ...):
    """
    对所有块做 softmax，再与 V 加权求和。
    输出: 注意力结果 [batch, num_heads, head_dim]
    """
```

**为何分两阶段？** 单 token 查询时 KV 序列往往很长。按块并行计算 KV，再在归约阶段合并，更高效。

### FlashInfer 后端（mini-sglang）

```python
class FlashInferBackend(BaseAttnBackend):
    def __init__(self):
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer,  # 预分配 128MB 工作区
            kv_layout="NHD",
            use_tensor_cores=(gqa_group_size >= 4),
        )

    def forward_decode(self, q, layer):
        return self.decode_wrapper.forward(
            q, paged_kv_cache=(k_buffer, v_buffer),
        )
```

## KV Cache 写入内核

### Triton 写入 KV Cache（nano-vllm）

```python
@triton.jit
def store_kvcache_kernel(K, V, KVCache, SlotMapping, ...):
    """
    K: [num_new_tokens, num_kv_heads, head_dim]
    V: [num_new_tokens, num_kv_heads, head_dim]
    KVCache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    SlotMapping: [num_new_tokens] → 物理槽位下标

    对每个新 token i:
        slot = SlotMapping[i]
        block_id = slot // block_size
        offset = slot % block_size
        KVCache[0, layer, block_id, offset, :, :] = K[i]
        KVCache[1, layer, block_id, offset, :, :] = V[i]
    """
```

### 直接切片（nano-sglang）

```python
def store_kv_cache(self, k, v, out_cache_loc, layer_id):
    """在预分配池上用 Python 索引写入"""
    self.token_to_kv_pool.kv_data[layer_id][out_cache_loc] = torch.stack([k, v], dim=1)
```

## 注意力后端抽象（mini-sglang）

mini-sglang 提供了便于**切换后端**的抽象：

```python
class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(self, q, k, v, layer_id) -> Tensor:
        """对当前 batch 执行注意力"""
        pass

    @abstractmethod
    def prepare_metadata(self, batch) -> None:
        """前向前准备元数据（页表、序列长度等）"""
        pass

    @abstractmethod
    def init_capture_graph(self, batch_size) -> None:
        """为 CUDA Graph 捕获做准备"""
        pass

class HybridBackend(BaseAttnBackend):
    """Prefill 与 decode 使用不同后端"""
    def __init__(self, prefill_backend, decode_backend):
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id):
        if self.current_phase == "prefill":
            return self.prefill_backend.forward(q, k, v, layer_id)
        else:
            return self.decode_backend.forward(q, k, v, layer_id)
```

## 性能考量

| 方面 | Prefill | Decode |
|------|---------|--------|
| 瓶颈 | 计算受限 | 内存带宽受限 |
| Q 长度 | 多 token | 1 个 token |
| KV 长度 | prompt 长度 | 完整序列长度 |
| 较优内核 | Flash Attention（变长） | FlashInfer 或两阶段 Triton |
| 块大小 | 较大（128–256） | 较小（32–64） |
| Warp 数 | 8 | 2–4 |
| CUDA Graph | 通常不用（形状多变） | 常用（batch 大小固定时） |

## 设计模板

生成注意力相关算子时可按：

1. **选库**：Flash Attention（通用性较好）或 FlashInfer（decode 性能往往更好）。
2. **实现两条路径**：Prefill（变长）与 Decode（分页）。
3. **增加 KV 写入**：Triton 内核或直接索引。
4. **接口抽象**：采用 `BaseAttnBackend` 模式，便于日后替换实现。
5. **支持 CUDA Graph**：Decode 后端需能配合图捕获。

---

*译文对应英文原文：`notebooks/03_operators/01_attention_ops.md`。*
