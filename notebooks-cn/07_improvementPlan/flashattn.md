# Flash-Attention 集成方案

## 1. 背景

### 1.1 当前瓶颈

meta-infer/engine 的 attention 层全部使用 PyTorch 原生 `F.scaled_dot_product_attention`，存在以下性能问题：

| 问题 | 影响 |
|------|------|
| Decode 路径每步构造 `attn_mask` 张量 `[1,1,1,max_seq_len]` | 额外显存分配 + kernel launch 开销 |
| SDPA 对 `max_seq_len` 全量做 softmax（含 padding 位置） | 浪费计算，padding 越多越严重 |
| SDPA 为通用 kernel，未做 flash-attn 的 tiling/重计算优化 | HBM 访问量大，未利用 SRAM |
| 无 GQA 融合处理 | GQA 需要 `enable_gqa=True` 参数，内部可能有额外开销 |

当前 4 处调用位置：

| 文件 | 路径 | 行号 | 用途 |
|------|------|------|------|
| `deepseek_v2.py` | `DeepseekAttentionTP.forward()` | 289 | DeepSeek-V2 Prefill |
| `deepseek_v2.py` | `DeepseekAttentionTP.forward()` | 313 | DeepSeek-V2 Decode |
| `qwen.py` | `QwenAttentionTP.forward()` | 182 | Qwen3 Prefill |
| `qwen.py` | `QwenAttentionTP.forward()` | 202 | Qwen3 Decode |

### 1.2 Flash-Attention 优势

| 特性 | 说明 |
|------|------|
| IO-aware tiling | 将 Q/K/V 分块加载到 SRAM，减少 HBM 读写 |
| 重计算替代存储 | 反向传播时重计算 softmax，而非存储完整 attention 矩阵 |
| Varlen 模式 | `flash_attn_varlen_func` 支持变长序列，用 `cu_seqlens` 精确标记边界，无需 padding |
| GQA 原生支持 | 自动处理 `nheads % nheads_k == 0` 的 GQA 分组 |
| Fuse causal mask | causal 模式下自动跳过上三角区域，无需显式 mask |

### 1.3 环境约束

- **GPU**: A800 (SM80, Ampere) → 仅支持 **FA2**（FA3 需 SM90 Hopper，FA4 需 SM9.x+）
- **已安装**: conda env "meta"，flash-attn 2.8.3，PyTorch 2.9.1+cu128，CUDA 12.8
- **参考**: vllm 在 A800 上的 attention 优先级为 `[FLASH_ATTN, FLASHINFER, TRITON_ATTN]`，flash-attn 排第一

## 2. 集成方案

### 2.1 总体策略

分三个阶段，渐进式集成：

| 阶段 | 内容 | 改动量 | 收益 |
|------|------|--------|------|
| **Phase 1** | B=1 直接替换 SDPA → flash_attn_varlen_func | 小 | kernel 级加速，消除 attn_mask |
| Phase 2 | 多序列 batched prefill (cu_seqlens batch) | 中 | 消除 Python 循环，一次 kernel 处理全 batch |
| Phase 3 | Paged KV cache (block_table) | 大 | 动态内存管理，prefix sharing |

**当前实施 Phase 1**，保持现有 B=1 逐序列处理模式，只替换 attention kernel。

### 2.2 核心 API

```python
from flash_attn import flash_attn_varlen_func

flash_attn_varlen_func(
    q,                    # (total_q, nheads, headdim) — 必须是 3D ragged 格式
    k,                    # (total_k, nheads_k, headdim)
    v,                    # (total_k, nheads_k, headdim)
    cu_seqlens_q,         # (batch_size + 1,) int32 — Q 的累积序列长度
    cu_seqlens_k,         # (batch_size + 1,) int32 — K/V 的累积序列长度
    max_seqlen_q,         # int — Q 的最大序列长度
    max_seqlen_k,         # int — K/V 的最大序列长度
    dropout_p=0.0,        # 推理时为 0
    softmax_scale=None,   # 默认 1/sqrt(headdim)
    causal=False,         # prefill 用 True，decode 用 False
    window_size=None,     # 滑动窗口，None 表示全局注意力
    softcap=0.0,          # softcap，0 表示禁用
    alibi_slopes=None,    # ALiBi，None 表示不使用
    block_table=None,     # Phase 3 paged attention 用
    return_softmax_lse=False,  # 级联 attention 用
)
```

**关键**: flash-attn 接受 3D ragged 格式 `(total_tokens, nheads, headdim)`，不接受 4D `(B, S, H, D)`。当前代码需要 reshape。

### 2.3 KV Cache 布局

当前 engine 使用 contiguous 预分配 buffer：

```
Qwen:    (k_buf, v_buf, kv_len)           — [B, max_seq_len, num_kv_heads, head_dim]
DeepSeek: (k_nope_buf, v_buf, raw_k_pe_buf, kv_len) — [B, max_seq_len, heads, dim]
```

Phase 1 保持此布局不变。flash-attn 接受任意 contiguous tensor，通过 slicing `buf[0, :kv_len]` 得到 `(kv_len, heads, dim)` 的 3D view。

Phase 3 将改为 paged layout `(num_blocks, block_size, heads, dim)`，与 vllm 一致。

## 3. Phase 1 详细改动

### 3.1 QwenAttentionTP ([qwen.py](engine/models/qwen.py))

#### 3.1.1 Prefill 路径 (line 179-186)

**改动前**：
```python
q_p = q.permute(0, 2, 1, 3)                    # [B, H, S, D]
k_p = k_buf[:, :kv_len].permute(0, 2, 1, 3)    # [B, KV_H, S, D]
v_p = v_buf[:, :kv_len].permute(0, 2, 1, 3)    # [B, KV_H, S, D]
out = F.scaled_dot_product_attention(
    q_p, k_p, v_p, dropout_p=0.0, is_causal=True,
    scale=self.scaling, enable_gqa=use_gqa,
)
out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, self.q_size)
```

**改动后**：
```python
# flash-attn: reshape to (total, heads, headdim)
q_fa = q.reshape(seqlen, self.num_heads, self.head_dim)       # [S, H, D]
k_fa = k_buf[0, :kv_len]                                      # [kv_len, KV_H, D]
v_fa = v_buf[0, :kv_len]                                      # [kv_len, KV_H, D]
cu = torch.tensor([0, seqlen], dtype=torch.int32, device=q.device)
out = flash_attn_varlen_func(
    q_fa, k_fa, v_fa,
    cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=seqlen, max_seqlen_k=kv_len,
    causal=True, softmax_scale=self.scaling,
)
out = out.reshape(bsz, seqlen, self.q_size)
```

**变化说明**：
- 去掉 `.permute(0,2,1,3)` 和反向 permute（flash-attn 接受 `[S, H, D]` 原生布局）
- `enable_gqa` 不需要传——FA2 自动检测 `nheads % nheads_k == 0`
- `kv_len == seqlen` 时 `cu_seqlens_q` 和 `cu_seqlens_k` 相同

#### 3.1.2 Decode 路径 (line 194-206)

**改动前**：
```python
q_p = q.permute(0, 2, 1, 3)                                      # [1, H, 1, D]
k_p = k_buf.permute(0, 2, 1, 3)                                  # [1, KV_H, max_seq_len, D]
v_p = v_buf.permute(0, 2, 1, 3)                                  # [1, KV_H, max_seq_len, D]

attn_mask = torch.zeros(1, 1, 1, max_seq_len, device=hidden_states.device, dtype=q_p.dtype)
attn_mask[:, :, :, kv_len:] = float('-inf')

out = F.scaled_dot_product_attention(
    q_p, k_p, v_p, dropout_p=0.0, is_causal=False,
    scale=self.scaling, enable_gqa=use_gqa, attn_mask=attn_mask,
)
out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, self.q_size)
```

**改动后**：
```python
# flash-attn: 只传有效 KV 部分，无需 attn_mask
q_fa = q.reshape(1, self.num_heads, self.head_dim)               # [1, H, D]
k_fa = k_buf[0, :kv_len]                                         # [kv_len, KV_H, D]
v_fa = v_buf[0, :kv_len]                                         # [kv_len, KV_H, D]
cu_q = torch.tensor([0, 1], dtype=torch.int32, device=q.device)
cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=q.device)
out = flash_attn_varlen_func(
    q_fa, k_fa, v_fa,
    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    max_seqlen_q=1, max_seqlen_k=kv_len,
    causal=False, softmax_scale=self.scaling,
)
out = out.reshape(bsz, seqlen, self.q_size)
```

**变化说明**：
- **消除 `attn_mask` 分配**: 不再创建 `[1,1,1,max_seq_len]` 张量并填 `-inf`
- **只计算有效位置**: `k_buf[0, :kv_len]` 精确切片，kernel 不接触 padding
- `causal=False`: 单 query token 无需 causal mask
- `max_seq_len` buffer 保持不变（仍预分配用于 CUDA Graph 兼容），但不传给 flash-attn

### 3.2 DeepseekAttentionTP ([deepseek_v2.py](engine/models/deepseek_v2.py))

#### 3.2.1 Prefill 路径 (line 286-290)

**改动前**：
```python
q_cat = torch.cat([q_nope, q_pe], dim=-1).permute(0, 2, 1, 3)
k_cat = torch.cat([k_nope_valid, k_pe_rope], dim=-1).permute(0, 2, 1, 3)
v_perm = v_valid.permute(0, 2, 1, 3)
out = F.scaled_dot_product_attention(q_cat, k_cat, v_perm, dropout_p=0.0, is_causal=True, scale=self.scaling)
out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, self.local_heads * self.cfg.v_head_dim)
```

**改动后**：
```python
# flash-attn: 拼接 rope/nope 部分后 reshape 为 3D
q_fa = torch.cat([q_nope, q_pe], dim=-1).reshape(seqlen, self.local_heads, -1)   # [S, H, D_full]
k_fa = torch.cat([k_nope_valid[0], k_pe_rope[0]], dim=-1)                        # [kv_len, H, D_full]
v_fa = v_valid[0]                                                                 # [kv_len, H, D_v]
cu = torch.tensor([0, seqlen], dtype=torch.int32, device=q_fa.device)
out = flash_attn_varlen_func(
    q_fa, k_fa, v_fa,
    cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=seqlen, max_seqlen_k=kv_len,
    causal=True, softmax_scale=self.scaling,
)
out = out.reshape(bsz, seqlen, self.local_heads * self.cfg.v_head_dim)
```

**MLA 注意事项**：
- DeepSeek-V2 的 MLA 将 K 分为 `k_nope`（无位置编码）和 `k_pe`（RoPE 部分），拼接后形成完整 K
- FA2 不支持 `q_v` 参数（那是 FA3 的 MLA 原生支持），所以仍需拼接完整 Q/K 向量
- ⚠️ **FA2 要求 Q/K/V 三者 headdim 完全相同**，而 DeepSeek-V2-Lite 的 QK=192, V=128 不等，**FA2 无法使用**
- 验证：`flash_attn_varlen_func(q[192], k[192], v[128])` → 报错 "v must have shape (total_k, num_heads_k, head_size)"
- **DeepSeek-V2 改用 SDPA 切片优化**（去掉 attn_mask，只传有效 KV 部分），详见第 5 节

#### 3.2.2 Decode 路径 (line 305-314)

**改动前**：
```python
q_cat = torch.cat([q_nope, q_pe], dim=-1).permute(0, 2, 1, 3)  # [B, H, 1, D]
k_cat = torch.cat([k_nope_buf, k_pe_rope], dim=-1).permute(0, 2, 1, 3)  # [B, H, max_seq_len, D]
v_perm = v_buf.permute(0, 2, 1, 3)  # [B, H, max_seq_len, D]

attn_mask = torch.zeros(1, 1, 1, max_seq_len, device=hidden_states.device, dtype=q_cat.dtype)
attn_mask[:, :, :, kv_len:] = float('-inf')

out = F.scaled_dot_product_attention(q_cat, k_cat, v_perm, dropout_p=0.0, is_causal=False, scale=self.scaling, attn_mask=attn_mask)
out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, self.local_heads * self.cfg.v_head_dim)
```

**改动后**：
```python
q_fa = torch.cat([q_nope, q_pe], dim=-1).reshape(1, self.local_heads, -1)   # [1, H, D_full]
k_fa = torch.cat([k_nope_buf[0, :kv_len], k_pe_rope[0, :kv_len]], dim=-1)  # [kv_len, H, D_full]
v_fa = v_buf[0, :kv_len]                                                     # [kv_len, H, D_v]
cu_q = torch.tensor([0, 1], dtype=torch.int32, device=q_fa.device)
cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=q_fa.device)
out = flash_attn_varlen_func(
    q_fa, k_fa, v_fa,
    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    max_seqlen_q=1, max_seqlen_k=kv_len,
    causal=False, softmax_scale=self.scaling,
)
out = out.reshape(bsz, seqlen, self.local_heads * self.cfg.v_head_dim)
```

**注意**: Decode 路径中 `k_pe_rope` 的 RoPE 计算需要调整——当前对 `k_pe_buf` 全部 `max_seq_len` 位置做 RoPE（为 CUDA Graph 固定 shape），但 flash-attn 只需要有效部分。改为只对 `k_pe_buf[0, :kv_len]` 做 RoPE：

```python
# 改动前 (line 301-303): 对全部 max_seq_len 做 RoPE
all_positions = torch.arange(max_seq_len, device=hidden_states.device, dtype=torch.long)
k_pe_rope = _apply_rope_gptj(raw_k_pe_buf, all_positions, ...)
k_pe_rope = k_pe_rope.expand(-1, -1, self.local_heads, -1)

# 改动后: 只对有效 kv_len 做 RoPE
all_positions = torch.arange(kv_len, device=hidden_states.device, dtype=torch.long)
k_pe_rope = _apply_rope_gptj(raw_k_pe_buf[0, :kv_len].unsqueeze(0), all_positions, ...)
k_pe_rope = k_pe_rope[0].expand(-1, self.local_heads, -1)  # [kv_len, H, rope_dim]
```

这会破坏 CUDA Graph 的固定 shape 要求。如果需要保持 CUDA Graph 兼容，可以保留全量 RoPE 计算，然后切片 `k_pe_rope[0, :kv_len]` 传给 flash-attn。多算的 RoPE 只浪费少量算力，不影响正确性。

### 3.3 导入语句

在两个文件顶部添加：

```python
from flash_attn import flash_attn_varlen_func
```

### 3.4 保持不变的部分

| 组件 | 原因 |
|------|------|
| KV cache 分配与写入 | 保持 `[B, max_seq_len, heads, head_dim]` contiguous 布局 |
| Model Runner (`DeepseekTPModelRunner`, `QwenTPModelRunner`) | 保持 B=1 逐序列循环 |
| Scheduler | 无变化 |
| Block Manager / Memory Pool | Phase 1 不引入 paged attention |
| `torch.compile` | flash-attn kernel 不在 compile 图内，保持现有编译策略 |
| RoPE / RMSNorm / MLP / MoE | 不涉及 attention kernel |

## 4. 参考：vllm 的实现

### 4.1 vllm Flash-Attention 后端

vllm 的 flash-attn 集成在 `vllm/v1/attention/backends/flash_attn.py`，核心类：

- `FlashAttentionBackend`: 注册后端，声明支持的 dtype/块大小/计算能力
- `FlashAttentionMetadataBuilder`: 从 `CommonAttentionMetadata` 构建每 batch 的元数据张量
- `FlashAttentionImpl`: 实际 forward pass

**关键发现**: vllm 的 decode 路径也用 `flash_attn_varlen_func`（不是 `flash_attn_with_kvcache`），通过 `max_seqlen_q=1` + `block_table` + `seqused_k` 实现 paged decode。

### 4.2 vllm MLA 处理

vllm 有两种 MLA 后端：

| 后端 | Kernel | 计算能力 | 特点 |
|------|--------|----------|------|
| `FLASH_ATTN_MLA` | FA3 (`_vllm_fa3_C`) | SM90 only | 用 `q_v` 参数分离 rope/nope headdim |
| `FLASHMLA` | FlashMLA (`_flashmla_C`) | SM90/SM100 | DeepSeek 专用 kernel，decode only |

两者都需要 SM90+，**在 A800 上不可用**。因此我们使用 FA2 的标准 varlen 模式，拼接 rope/nope 部分。

### 4.3 vllm KV Cache 布局

```
(num_blocks, block_size, num_kv_heads, head_size)
```

通过 `block_table` (batch_size, max_blocks_per_seq) 映射逻辑位置到物理块。Phase 3 将参考此设计。

## 5. DeepSeek-V2 SDPA 优化方案

**所有 DeepSeek-V2 变体（Lite/全量/V3）都无法使用 FA2**，原因不是 head dim 超限，而是 FA2 要求 Q/K/V headdim 完全相同，而 MLA 的 QK headdim ≠ V headdim：

| 模型 | QK headdim | V headdim | FA2 |
|------|-----------|----------|-----|
| DeepSeek-V2-Lite | 192 | 128 | ❌ 不等 |
| DeepSeek-V2 全量 | 576 | 128 | ❌ 不等+超限 |
| DeepSeek-V3 | 576 | 128 | ❌ 不等+超限 |

### 方案 A: SDPA 去 mask 优化（推荐）

不使用 flash-attn，但消除 decode 的 `attn_mask` 开销：

```python
# Decode: 直接切片有效部分，SDPA 不需要 attn_mask
q_p = q.permute(0, 2, 1, 3)                          # [1, H, 1, D]
k_p = k_buf[:, :kv_len].permute(0, 2, 1, 3)          # [1, H, kv_len, D]
v_p = v_buf[:, :kv_len].permute(0, 2, 1, 3)          # [1, H, kv_len, D]
out = F.scaled_dot_product_attention(
    q_p, k_p, v_p, dropout_p=0.0, is_causal=False,
    scale=self.scaling,
)  # 无需 attn_mask，K/V 已切片到有效长度
```

**代价**: 失去 CUDA Graph 兼容性（tensor shape 每步变化）。可通过固定 `kv_len` 为 2 的幂次来缓解。

### 方案 B: 分解 Attention

将 `exp(QK^T) = exp(Q_nope * K_nope^T) * exp(Q_pe * K_pe^T)` 分解为两次 FA2 调用（各 headdim ≤ 256），然后合并 softmax 结果。实现复杂，数值稳定性需要 careful 处理，不推荐作为首选。

## 6. 验证计划

### 6.1 数值正确性

对比 flash-attn 与原 SDPA 的输出差异：

```python
# 在 QwenAttentionTP.forward() 中同时计算两种结果
out_sdpa = F.scaled_dot_product_attention(q_p, k_p, v_p, ...)
out_fa = flash_attn_varlen_func(q_fa, k_fa, v_fa, ...)

# fp16/bf16 下 max diff 应 < 1e-2
diff = (out_sdpa - out_fa).abs().max().item()
assert diff < 1e-2, f"Flash-attn output diff too large: {diff}"
```

### 6.2 端到端测试

```bash
conda activate meta

# DeepSeek-V2-Lite-Chat
torchrun --nproc_per_node=4 openai_tp_server.py --backend tp --model deepseek-v2-lite
# 另一个终端
python run_compare_metainfer_vllm.sh dsv2

# Qwen3-8B
torchrun --nproc_per_node=4 openai_tp_server.py --backend tp --model qwen3-8b
python run_compare_metainfer_vllm.sh qwen
```

对比优化前后：
- 生成结果一致性（temperature=0.0 时应完全一致）
- Prefill throughput (tokens/s)
- Decode throughput (tokens/s)
- GPU 显存占用

### 6.3 性能基线

| 指标 | 优化前 (SDPA) | 目标 (FA2) |
|------|--------------|------------|
| DeepSeek-V2 Prefill | 基线 | +20~40% |
| DeepSeek-V2 Decode | 基线 | +30~50%（消除 attn_mask） |
| Qwen3 Prefill | 基线 | +20~40% |
| Qwen3 Decode | 基线 | +30~50% |
| 显存 | 基线 | 减少（无 attn_mask 分配） |

## 7. 后续优化路线

### Phase 2: 多序列 Batched Prefill

当前 model runner 对每个 seq 逐个处理（B=1 循环）。Flash-attn 的 varlen 模式天然支持多序列 batch：

```python
# 构建 flat batch
all_q = torch.cat([q1, q2, q3], dim=0)  # [total_q, H, D]
cu_seqlens_q = torch.tensor([0, len1, len1+len2, len1+len2+len3], dtype=torch.int32)
# 一次 kernel 调用处理整个 batch
out = flash_attn_varlen_func(all_q, all_k, all_v, cu_seqlens_q, cu_seqlens_k, ...)
```

改动点：`ModelRunner.run()` 中 prefill 路径改为构建 flat batch，attention 层接收 flat tensor + cu_seqlens。

### Phase 3: Paged KV Cache

参考 vllm 的 block_table 模式：

- KV cache 改为 `(num_blocks, block_size, heads, dim)` block-granularity 分配
- `flash_attn_varlen_func` 传入 `block_table` + `seqused_k`（替代 `cu_seqlens_k`）
- engine 已有 `BlockManager`（`block_manager.py`），需接入 attention 路径
- 支持 prefix sharing、动态内存管理、KV cache eviction

### Phase 4: 其他 SOTA Kernel

| 技术 | 用途 | 前提 |
|------|------|------|
| FlashInfer | Decode 优化（paged KV、MLA） | 需评估 A800 兼容性 |
| DeepGEMM | FP8 grouped GEMM（MoE 加速） | 需 Hopper SM90 |
| DeepEP | MoE all-to-all dispatch | 需评估 TP 场景收益 |
| CUDA Graphs | 消除 CPU launch 开销 | 需固定 shape |
| ThunderKittens | 自定义高性能 kernel | 需 SM90+ |
