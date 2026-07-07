# Attention & KV Cache — API 契约

> 关联 notebooks: `07_improvementPlan/improvement_plan.md` §P3-FA

## 概述

Paged KV cache (block_size=256) + flash_attn_varlen_func (prefill) + flash_attn_with_kvcache (decode)。
源实现文件: `engine/models/qwen.py`

---

## Paged KV Cache 格式

### 关键维度

```
_key_cache:  [num_blocks, 256, num_kv_heads, head_dim]  bf16
_value_cache: [num_blocks, 256, num_kv_heads, head_dim]  bf16
_block_table: [1, max_blocks]  int32
_slot_mapping_decode: [1]  int64 (预分配 register_buffer)
_kv_len_gpu: [1]  int32 (GPU tensor，严禁 .item() 在 forward_decode 内)
```

### Lazy Allocation

```python
if self._key_cache is None:
    max_blocks = (max_seq_len + 255) // 256
    self._key_cache = torch.zeros(max_blocks, 256, self.num_kv_heads, self.head_dim, dtype=torch.bfloat16, device=q.device)
    self._value_cache = torch.zeros_like(self._key_cache)
    self._block_table = torch.zeros(1, max_blocks, dtype=torch.int32, device=q.device)
```

### block_size 硬约束

`_kv_block_size = 256` — flash_attn_with_kvcache 的硬性最低要求（此值来自 flash_attn 库的约束，不同 flash_attn 版本可能有不同要求）。不能使用 16。

---

## Prefill Attention Path

### 关键规则
- K/V 来自**当前投影产出** (非从 KV cache 读取)
- 顺序: 投影 → flash_attn_varlen_func → index_copy_ 写入 cache
- causal=True
- B=1 时 cu_seqlens 自动构造 `[0, num_tokens]`

```python
def forward(self, hidden_states, positions, max_seq_len):
    B, S, H = hidden_states.shape  # B=1
    q, k, v = self.qkv_proj(hidden_states)
    q = q.view(B, S, self.num_heads, self.head_dim)    # [1,S,8,128]
    k = k.view(B, S, self.num_kv_heads, self.head_dim) # [1,S,2,128]
    v = v.view(B, S, self.num_kv_heads, self.head_dim) # [1,S,2,128]
    q = self.q_norm(q); k = self.k_norm(k)
    num_tokens = B * S
    q_flat = q.reshape(num_tokens, self.num_heads, self.head_dim)
    k_flat = k.reshape(num_tokens, self.num_kv_heads, self.head_dim)
    rotary_embedding(positions, q_flat, k_flat, self.head_dim, self._cos_sin_cache_gpu, is_neox=True)
    
    # KV cache lazy allocation
    if self._key_cache is None: ...
    
    # flash_attn_varlen_func (causal=True, K/V from current projection)
    cu = torch.tensor([0, num_tokens], dtype=torch.int32, device=q.device)
    v_flat = v.reshape(num_tokens, self.num_kv_heads, self.head_dim)
    out = flash_attn_varlen_func(q_flat, k_flat, v_flat, cu, cu, num_tokens, num_tokens, causal=True)
    
    # Write KV to paged cache (sequential block allocation)
    num_blocks = (num_tokens + 255) // 256
    self._block_table[0, :num_blocks] = torch.arange(num_blocks, dtype=torch.int32, device=q.device)
    slot_mapping = ...
    kc_flat.index_copy_(0, slot_mapping, k_flat)
    vc_flat.index_copy_(0, slot_mapping, v_flat)
    self._kv_len_gpu[0] = num_tokens
    
    out = out.view(B, S, self.q_size)
    return self.o_proj(out)
```

---

## Decode Attention Path

### 关键规则
- K/V 从 paged KV cache **读取** (非从当前投影)
- 先写当前 token 的 K/V 到 cache (index_copy_) → 再读
- causal=False (单 token，无未来信息)
- kv_len: GPU tensor `[1]` int32

```python
def forward_decode(self, hidden_states, positions, kv_len, max_seq_len):
    B, S, H = hidden_states.shape  # B=1, S=1
    q, k, v = self.qkv_proj(hidden_states)
    q = q.view(B, S, self.num_heads, self.head_dim)   # [1,1,8,128]
    k = k.view(B, S, self.num_kv_heads, self.head_dim) # [1,1,2,128]
    v = v.view(B, S, self.num_kv_heads, self.head_dim)
    q = self.q_norm(q); k = self.k_norm(k)
    
    # RoPE (flatten to 2D)
    q_flat = q.reshape(S, self.num_heads, self.head_dim)
    k_flat = k.reshape(S, self.num_kv_heads, self.head_dim)
    rotary_embedding(positions, q_flat, k_flat, self.head_dim, self._cos_sin_cache_gpu, is_neox=True)
    q = q_flat.view(B, S, self.num_heads, self.head_dim)
    k = k_flat.view(B, S, self.num_kv_heads, self.head_dim)
    
    # KV cache write (decode: write 1 token to slot=kv_len)
    self._slot_mapping_decode[0] = self._kv_len_gpu[0]
    k_write = k.reshape(1, self.num_kv_heads, self.head_dim)
    v_write = v.reshape(1, self.num_kv_heads, self.head_dim)
    kc_flat = self._key_cache.view(-1, self.num_kv_heads, self.head_dim)
    vc_flat = self._value_cache.view(-1, self.num_kv_heads, self.head_dim)
    kc_flat.index_copy_(0, self._slot_mapping_decode, k_write)
    vc_flat.index_copy_(0, self._slot_mapping_decode, v_write)
    self._kv_len_gpu[0] += 1
    
    # flash_attn_with_kvcache (read KV from paged cache)
    q_attn = q.reshape(1, 1, self.num_heads, self.head_dim)
    out = flash_attn_with_kvcache_op(q_attn, self._key_cache, self._value_cache,
                                      self._kv_len_gpu, self._block_table,
                                      self.scaling, causal=False)
    
    out = out.reshape(B, S, self.q_size)
    return self.o_proj(out)
```

---

## QwenAttentionTP 构造函数

```python
class QwenAttentionTP(nn.Module):
    def __init__(self, cfg):
        self.total_num_heads = cfg.num_attention_heads  # 32 (全量)
        self.total_num_kv_heads = cfg.num_key_value_heads  # 8 (全量)
        self.num_heads = cfg.num_attention_heads // tp_size  # 8 (per-rank)
        self.num_kv_heads = ...  # 2 or 1 (see KV head replication)
        self.head_dim = cfg.head_dim  # 128
        self.q_size = self.num_heads * self.head_dim  # 1024
        self.kv_size = self.num_kv_heads * self.head_dim  # 256
        self.scaling = self.head_dim ** -0.5  # 1/sqrt(128)
        self.qkv_proj = QKVColumnParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        
        # Buffers (persistent=False)
        self.register_buffer('_cu_q', torch.tensor([0,0], dtype=torch.int32), persistent=False)
        self.register_buffer('_cu_k', torch.tensor([0,0], dtype=torch.int32), persistent=False)
        self.register_buffer('_kv_len_gpu', torch.zeros(1, dtype=torch.int32), persistent=False)
        self.register_buffer('_slot_mapping_decode', torch.zeros(1, dtype=torch.int64), persistent=False)
        
        self._kv_block_size = 256
        self._key_cache = None; self._value_cache = None; self._block_table = None
        self._cos_sin_cache_cpu = _get_cos_sin_cache(...)
        self._cos_sin_cache_gpu = None  # lazy GPU transfer
```

**属性命名规范 (权重加载关键)**:
- `.qkv_proj` 非 `.q_proj`
- `.o_proj` 非 `.out_proj`
- `.q_norm` 和 `.k_norm` (per-head RMSNorm，Qwen3 特有)

---

## 陷阱与反模式

- **FM-008**: block_size 必须 ≥256
- **FM-007**: RoPE is_neox=True (Qwen3)，False 导致短句重复
- **FM-004**: CosSinCache shape `[max_pos, head_size]`，非 `[2*head_size]`
- **ATTN-001**: decode 路径 `causal=False`，prefill 路径 `causal=True`
- **ATTN-002**: Prefill K/V 来自当前投影，非从 cache 读取
- **ATTN-003**: block_table dtype=int32，非 int64
- **ATTN-004**: `_kv_len_gpu` 为 GPU tensor，decode 内不要 `.item()`
- **ATTN-005**: KV cache lazy alloc — 按需分配，严禁全量预分配

---

## GatedDeltaNet Linear Attention (Qwen3.5 Hybrid)

<!-- source: evolution/evo-001, timestamp=2026-07-03 -->

### 概述

GatedDeltaNet 是 Qwen3.5/3.6 混合架构中的线性注意力组件（48/64 层）。
替代标准 softmax attention，使用 recurrent state + delta rule 实现 O(N) 复杂度。
源实现文件: `engine/models/qwen3_5.py` (QwenGatedDeltaNetTP)

### 关键维度

```
in_proj_qkv output: [B, T, 2560] per rank (TP=4)
  Q: [B, T, 512] = 4 k_heads * 128 (per rank)
  K: [B, T, 512] = 4 k_heads * 128 (per rank)
  V: [B, T, 1536] = 12 v_heads * 128 (per rank)
State: [B, 12, 128, 128] = [B, v_heads_per_rank, Dk, Dv]
conv_state: [B, 2560, 4] = [B, total_per_rank, kernel_size]
```

### Prefill Forward (QwenGatedDeltaNetTP.forward)

```
1. in_proj_qkv → mixed_qkv [B, T, 2560]
2. transpose → [B, 2560, T]; causal conv1d (depthwise, kernel=4, groups=2560) + SiLU
3. Save _conv_state = mixed_qkv[:, :, -4:] (raw QKV, before conv)
4. split Q [B,T,4,128], K [B,T,4,128], V [B,T,12,128]
5. in_proj_a (Replicate): [B, T, 48] → 选择 local v_head range → [B, T, 12]
6. in_proj_b (Replicate): [B, T, 48] → 选择 local v_head range → [B, T, 12]
7. g = -exp(A_log) * softplus(a + dt_bias) → [B, T, 12]
8. beta = sigmoid(b) → [B, T, 12]
9. Q/K L2 normalize; repeat_interleave(×3) 匹配 V heads
10. Recurrent Gated Delta Rule → core_out [B, T, 12, 128], final_state [B, 12, 128, 128]
11. in_proj_z (ColumnParallel) → [B, T, 1536]; reshape → [B, T, 12, 128]
12. Qwen3_5RMSNormGated(core_flat, z_flat) → gated output
13. out_proj (RowParallel) → [B, T, 5120]
```

### Decode Forward (QwenGatedDeltaNetTP.forward_decode)

与 Prefill 的关键差异:
- T=1，使用缓存的 `_conv_state` 和 `_recurrent_state`
- conv1d: 拼接 `[_conv_state, current_qkv]` (dim=-1) → conv → 取最后 1 token
- Recurrent step: `initial_state=_recurrent_state`
- 更新 `_conv_state` 和 `_recurrent_state`

### Recurrent Gated Delta Rule 公式

```
for t in range(L):
    S = S * exp(g_t)                    # state decay
    kv_mem = sum(S * k_t, dim=-2)       # key-value memory
    delta = (v_t - kv_mem) * beta_t     # delta update
    S = S + k_t^T @ delta               # state update (POST-UPDATE)
    o_t = sum(S * q_t, dim=-2)          # output (使用POST-UPDATE state)
```

### 陷阱

- **DELTA-001**: State 维度 [Dk, Dv] = [128, 128] (Dk first, NOT Dv first)
- **DELTA-002**: conv_state 必须缓存 raw QKV（conv 之前的值）；zero-pad → 3000× 误差
- **DELTA-003**: 输出 o_t 使用 POST-update state，不可在 update 前计算
- **DELTA-004**: Recurrence 内部计算使用 fp32（bf16 → fp32 → bf16）
- **DELTA-005**: 数值溢出保护: `torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)`

---

## Qwen3.5 FullAttention (MRoPE + Output Gate)

<!-- source: evolution/evo-001, timestamp=2026-07-03 -->

### 与 Qwen3-8B FullAttention 的差异

| 维度 | Qwen3-8B | Qwen3.5/3.6 |
|------|---------|-------------|
| QKV 投影 | Merged qkv_proj (1 GEMM) | Separate q_proj, k_proj, v_proj (3 GEMMs) |
| Q 投影输出 | q_size | **2 × q_size** (Q + output gate) |
| 位置编码 | Standard RoPE (NeoX), full head_dim | **MRoPE**, partial_rotary_factor=0.25 |
| Output gate | 无 | attn_out * sigmoid(gate) |
| Q/K Norm | RMSNorm (standard) | RMSNorm (standard, 非 Qwen3_5RMSNorm) |
| KV cache | Paged, block_size=256 | Paged, block_size=256 (only these layers) |

### Prefill Forward (QwenFullAttentionTP.forward)

```
1. q_proj → q_double [B, S, 3072] (2 * q_size per rank)
2. q, gate = torch.chunk(q_double.view(B, S, 6, 512), 2, dim=-1) → q [B,S,6,256], gate [B,S,6,256]
3. k_proj → k [B, S, 256]; v_proj → v [B, S, 256] (kv_size per rank)
4. reshape k → [B, S, 1, 256], v → [B, S, 1, 256]
5. Q/K norms (standard RMSNorm, per-head)
6. MRoPE (NeoX-style): rotary_dim=64, pass-through dim=192
7. Paged KV cache lazy alloc + write (block_size=256)
8. flash_attn_varlen_func(causal=True) → [T, 6, 256]
9. out = out * sigmoid(gate.reshape(B, S, 1536))
10. o_proj (RowParallel) → [B, S, 5120]
```

### MRoPE 实现要点

- `partial_rotary_factor=0.25` → rotary_dim = 64 (head_dim=256)
- `mrope_section=[11, 11, 10]` → 三个维度各分配 22, 22, 20 (cos*2 + sin*2)
- Cos/sin cache shape: `[3, max_pos, 64]` (3D position)
- 对 1D text: 三个段合并为单一 cos/sin，退化为标准 RoPE
- NeoX-style rotation (前后半分)，与 Qwen3 系列一致
- **陷阱**: mrope_section 必须从 config.rope_parameters 读取，不能用 head_dim/3 推断
