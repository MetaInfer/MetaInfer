# Triton MLA Decode 适配方案

## 1. 核心思路

### 1.1 当前方案 (FA2 + V-padding)

```
Decode 流程:
  q_nope [1,1,H,128]  ──┐
  q_pe   [1,1,H,64]   ──┼── cat → Q [1,1,H,192]
                          │
  k_nope_buf [1,max,H,128] ──┐
  k_pe_rope  [1,max,H,64]  ──┼── cat → K [1,max,H,192]
                               │
  v_buf [1,max,H,128] ──── F.pad → V [1,max,H,192]  ← 浪费 64 维零填充
                                                               │
  flash_attn_varlen_func(Q, K, V) → out [1,1,H,192]
                                         │
                                    trim → out [1,1,H,128]
```

### 1.2 Triton MLA 方案

```
Decode 流程:
  q_nope [1,1,H,128] ── @ W_UK_T → q_nope_proj [1,1,H,128]  ← 投影到潜空间
  q_pe   [1,1,H,64]  ──────────────────────────────────────┐
                                                             │
  q_proj = cat([q_nope_proj, q_pe]) → Q [1,1,H,192]  ◄─────┘
                                                         │
  kv_cache [1,max,1,192] = [c_kv(128) | k_pe(64)]  ────┤
  (c_kv 同时作为 K 和 V)                                 │
                                                         │
  Triton MLA kernel:
    Q @ K^T = Q_nope_proj @ c_kv^T + Q_pe @ k_pe^T  ← 192 维计算 attention scores
    P @ V   = P @ c_kv                                ← 只取前 128 维, 复用 K, 无零填充
    → out [1,1,H,128] (潜空间)
                                                         │
  out @ W_UV → out_expanded [1,1,H,v_head_dim]  ◄───────┘
```

### 1.3 关键差异

| 方面 | FA2 V-padding | Triton MLA |
|------|--------------|------------|
| KV cache | 3 个 buffer (k_nope, v, k_pe) | 1 个 buffer (c_kv \|\| k_pe) |
| V 计算 | 192 维 (64 维零填充) | 128 维 (复用 K 前 128 维) |
| Q 投影 | 直接 cat(q_nope, q_pe) | q_nope 先 @ W_UK 投影到潜空间 |
| 输出后处理 | trim [:128] | @ W_UV 矩阵乘 |
| 额外矩阵乘 | 无 | W_UK_T (decode Q) + W_UV (output) |
| Kernel | flash_attn_varlen_func (FA2) | 自定义 Triton split-K kernel |

## 2. 改动清单

### 2.1 新增文件

**`engine/kernels/triton_mla_decode.py`** — Triton MLA decode kernel

从 vllm 复制并简化:
- 源文件: `ref_projects/vllm/vllm/v1/attention/ops/triton_decode_attention.py`
- 只需要: `_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2` + Python wrapper
- 简化: 去掉 FP8 支持、logit_cap、非 MLA 路径
- 适配: 去掉 paged block_table，改用 contiguous buffer 索引

### 2.2 修改文件

**`engine/models/deepseek_v2.py`** — DeepseekAttentionTP

改动点:
1. `__init__`: 新增 W_UK_T 权重、W_UV 权重、统一 KV cache buffer
2. `forward` prefill: 保持现有 FA2 + V-padding（prefill 不用 Triton kernel）
3. `forward` decode: 改用 Triton MLA 路径
4. KV cache 格式: 从 `(k_nope_buf, v_buf, raw_k_pe_buf, kv_len)` 改为 `(kv_cache_buf, kv_len)`
5. 去掉FA2 wrapper 内部调用 `maybe_contiguous`（去掉检查 stride 后决定是否 `.contiguous()`）。

### 2.3 不改动

- Qwen 模型 (GQA, 不是 MLA)
- Prefill 路径 (仍用 FA2)
- Model Runner / Scheduler / Engine

## 3. 详细实现

### 3.1 KV Cache 格式变更

**当前**:
```python
past_key_values = (k_nope_buf, v_buf, raw_k_pe_buf, kv_len)
# k_nope_buf:    [B, max_seq_len, num_kv_heads, qk_nope_head_dim]  — 128 维
# v_buf:         [B, max_seq_len, num_kv_heads, v_head_dim]         — 128 维
# raw_k_pe_buf:  [B, max_seq_len, 1, qk_rope_head_dim]             — 64 维
# 总显存: max_seq_len * (128 + 128 + 64) * H * 2 bytes (bf16)
```

**改为**:
```python
past_key_values = (kv_cache_buf, kv_len)
# kv_cache_buf: [B, max_seq_len, 1, kv_lora_rank + qk_rope_head_dim]  — 192 维
# 总显存: max_seq_len * 192 * 1 * 2 bytes (bf16)
# 注意: MLA 的 KV heads = 1 (MQA 模式), 因为 c_kv 是共享的潜空间表示
```

**显存对比** (DeepSeek-V2-Lite, H=4, max_seq_len=256):
- 当前: 256 * (128+128+64) * 4 * 2 = 655,360 bytes = 640 KB per layer
- Triton MLA: 256 * 192 * 1 * 2 = 98,304 bytes = 96 KB per layer
- **节省 85% KV cache 显存**

### 3.2 权重变更

**新增权重** (从现有 kv_b_proj_with_mqa 中提取):

```python
# 当前 kv_b_proj_with_mqa: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
# 输出 split 为 k_nope [H, 128] 和 v [H, 128]

# Triton MLA 需要:
# 1. W_UK_T: 用于 decode 时将 q_nope 投影到潜空间
#    形状: [num_heads, qk_nope_head_dim, kv_lora_rank] = [H, 128, 128]
#    来源: kv_b_proj_with_mqa.weight 的 k_nope 部分转置
#
# 2. W_UV: 用于将 attention 输出从潜空间扩展回 v_head_dim
#    形状: [num_heads, kv_lora_rank, v_head_dim] = [H, 128, 128]
#    来源: kv_b_proj_with_mqa.weight 的 v 部分
```

**提取方式**:

```python
# kv_b_proj_with_mqa.weight shape: [num_heads * (qk_nope + v_head), kv_lora_rank]
# = [H * 256, 128] for Lite

W_full = self.kv_b_proj_with_mqa.weight.data
# reshape to [H, qk_nope + v_head, kv_lora_rank]
W_full = W_full.view(self.local_heads, self.cfg.qk_nope_head_dim + self.cfg.v_head_dim, self.cfg.kv_lora_rank)

# W_UK_T: k_nope 部分, shape [H, qk_nope, kv_lora_rank]
self.W_UK_T = W_full[:, :self.cfg.qk_nope_head_dim, :].transpose(1, 2).contiguous()
# shape: [H, kv_lora_rank, qk_nope] = [H, 128, 128]

# W_UV: v 部分, shape [H, kv_lora_rank, v_head_dim]
self.W_UV = W_full[:, self.cfg.qk_nope_head_dim:, :].transpose(1, 2).contiguous()
# shape: [H, v_head_dim, kv_lora_rank] → 需要转置为 [H, kv_lora_rank, v_head_dim]
```

注意: 这些权重**不需要额外显存**——它们是从现有 `kv_b_proj_with_mqa` 中切出来的 view 或转置。

### 3.3 Decode 路径改动

**当前 decode 流程**:
```python
# 1. QKV 投影 (不变)
q_nope, q_pe = split(q_b_proj(q_latent))  # [1,1,H,128], [1,1,H,64]
c_kv = kv_a_layernorm(kv_a_proj(hidden))   # [1,1,1,128] (潜空间)
k_pe = ...                                  # [1,1,1,64] (rope 部分)
k_nope, v = split(kv_b_proj(c_kv))         # [1,1,H,128], [1,1,H,128]

# 2. KV cache 写入 (改为统一 buffer)
kv_cache_buf[:, kv_len] = cat([c_kv, k_pe], dim=-1)  # [1,1,1,192]

# 3. Q 投影到潜空间 (新增)
q_nope_proj = bmm(q_nope, W_UK_T)  # [1,1,H,128] @ [H,128,128] → [1,1,H,128]

# 4. Q 拼接
q_proj = cat([q_nope_proj, q_pe], dim=-1)  # [1,1,H,192]

# 5. Triton MLA kernel (替换 FA2)
out = triton_mla_decode(
    q=q_proj,           # [1, H, 192]
    kv_cache=kv_cache_buf,  # [1, max_seq_len, 1, 192]
    kv_len=kv_len,
    ...
)  # out shape: [1, H, 128] (潜空间)

# 6. 输出扩展 (新增)
out = bmm(out, W_UV)  # [1,1,H,128] @ [H,128,128] → [1,1,H,128]

# 7. O 投影 (不变)
output = o_proj(out)
```

### 3.4 Prefill 路径

Prefill 保持不变（FA2 + V-padding），因为:
1. Prefill 每请求只调用 1 次，CPU 开销不重要
2. Prefill Q 长度 > 1，Triton split-K kernel 为 decode (Q=1) 优化，不适合长 Q
3. FA2 的 causal attention 对长序列更高效

Prefill 写入 KV cache 时，需要同时写入统一 buffer:
```python
# Prefill KV cache 写入
kv_cache_buf[:, :seqlen, :, :kv_lora_rank] = c_kv        # c_kv 部分
kv_cache_buf[:, :seqlen, :, kv_lora_rank:] = k_pe         # k_pe 部分
```

### 3.5 Triton Kernel 适配

从 vllm 复制 `_fwd_grouped_kernel_stage1`，需要的修改:

```python
# 去掉 paged block_table 索引，改用 contiguous 索引
# vllm: kv_loc = tl.load(Req_to_tokens + cur_batch * stride_req + split_kv_start + offs_n)
# 改为: kv_loc = split_kv_start + offs_n  (直接连续索引)

# 去掉 FP8 支持 (k_scale, v_scale)
# 去掉 logit_cap
# 保留 IS_MLA=True 路径 (v = tl.trans(k))
```

关键 kernel 代码 (IS_MLA 路径):
```python
# 加载 K (c_kv || k_pe, 192 维)
offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
k = tl.load(K_Buffer + offs_buf_k, ...)

# 计算 attention scores
qk = tl.dot(q, k.to(q.dtype))  # [BLOCK_H, BLOCK_N]
qk *= sm_scale
p = tl.exp(qk - n_e_max[:, None])

# MLA trick: 复用 K 的前 128 维作为 V
v = tl.trans(k)  # [BLOCK_N, 192] → [192, BLOCK_N], 只取前 128 维存入 acc
acc += tl.dot(p.to(v.dtype), v)  # [BLOCK_H, 128]
```

## 4. 实施步骤

### Step 1: 复制并简化 Triton kernel

```bash
cp ref_projects/vllm/vllm/v1/attention/ops/triton_decode_attention.py \
   engine/kernels/triton_mla_decode.py
```

简化内容:
- 去掉 `_fwd_kernel_stage1` (非 grouped 版本)
- 去掉 FP8 / logit_cap / 非 MLA 路径
- 改 contiguous 索引 (去掉 block_table)
- 只保留 `decode_attention_fwd` 入口函数

### Step 2: 修改 DeepseekAttentionTP.__init__

新增:
```python
# 从 kv_b_proj 提取 W_UK_T 和 W_UV
W = self.kv_b_proj_with_mqa.weight.data.view(
    self.local_heads, nope_d + v_d, kv_lora_rank
)
self.W_UK_T = W[:, :nope_d, :].transpose(1, 2).contiguous()  # [H, L, P]
self.W_UV = W[:, nope_d:, :].contiguous()                      # [H, P, V]

# 统一 KV cache buffer (延迟分配)
self._kv_cache_buf = None  # [B, max_seq_len, 1, kv_lora_rank + rope_dim]
```

### Step 3: 修改 decode 分支

替换 FA2 + V-padding 为 Triton MLA:
```python
# 1. 写入统一 KV cache
self._kv_cache_buf[0, kv_len-1, 0, :kv_lora_rank] = c_kv[0, 0, 0]
self._kv_cache_buf[0, kv_len-1, 0, kv_lora_rank:] = k_pe[0, 0, 0]

# 2. Q 投影到潜空间
q_nope_proj = torch.bmm(q_nope.squeeze(0), self.W_UK_T)  # [H, 1, L]

# 3. Q 拼接
q_mla = torch.cat([q_nope_proj.squeeze(1), q_pe.squeeze(0)], dim=-1)  # [H, 192]

# 4. Triton MLA kernel
out_latent = triton_mla_decode(q_mla, self._kv_cache_buf, kv_len, ...)  # [1, H, 128]

# 5. 输出扩展
out = torch.bmm(out_latent.squeeze(0), self.W_UV)  # [H, 1, V]

# 6. O 投影
output = self.o_proj(out.reshape(1, 1, -1))
```

### Step 4: 修改 prefill 的 KV cache 写入

Prefill 仍用 FA2，但 KV cache 写入统一 buffer:
```python
# 写入统一 buffer (替代原来的 3 个独立 buffer)
self._kv_cache_buf[:, :seqlen, :, :kv_lora_rank] = c_kv
self._kv_cache_buf[:, :seqlen, :, kv_lora_rank:] = k_pe
```

### Step 5: 正确性验证 + 清理

**Phase 5a: 双路径对比验证**

在 decode 分支中同时保留 FA2 和 Triton MLA 两条路径，比较输出差异:

```python
# forward decode 中, 同时计算两条路径
out_fa2 = ...  # FA2 V-padding 路径 (现有代码)
out_triton = ...  # Triton MLA 路径 (新增代码)

diff = (out_triton - out_fa2).abs().max().item()
if diff >= 1e-2:
    raise ValueError(f"Triton MLA output diff too large: {diff}")
# 验证通过后使用 Triton MLA 输出
out = out_triton
```

**Phase 5b: 删除 FA2 decode 路径**

验证通过后，删除 decode 分支中的 FA2 代码，只保留 Triton MLA 路径。需要删除的内容:

1. `DeepseekAttentionTP.__init__` 中不再需要的预分配缓冲区:
   - `self._k_cat_buf` — FA2 的 K 拼接缓冲区
   - `self._v_pad_buf` — FA2 的 V-padding 缓冲区
   - `self._q_cat_buf` — FA2 的 Q 拼接缓冲区

2. `forward` decode 分支中删除:
   - `k_nope_buf`, `v_buf`, `raw_k_pe_buf` 的写入和更新
   - `k_pe_rope` 的 RoPE 计算和 expand
   - `F.pad(v_buf[0], [...])` V-padding 操作
   - `flash_attn_varlen_func(...)` 调用
   - `out[:, :, :v_head_dim]` 输出 trim

3. KV cache 格式从 `(k_nope_buf, v_buf, raw_k_pe_buf, kv_len)` 简化为 `(kv_cache_buf, kv_len)`

4. Model runner (`DeepseekTPModelRunner.run`) 中:
   - Prefill 后 `seq.past_key_values` 存储新格式 `(kv_cache_buf, kv_len)`
   - Decode 时传入新格式

**删除后的 decode 分支应只包含**:
```python
# 1. KV cache 写入 (统一 buffer)
# 2. Q 投影到潜空间 (bmm with W_UK_T)
# 3. Q 拼接 (cat q_nope_proj + q_pe)
# 4. Triton MLA kernel
# 5. 输出扩展 (bmm with W_UV)
# 6. O 投影
```

**验证删除后正确性**:
```bash
# 运行端到端测试确认输出与基线一致
PYTHONPATH=$(pwd):$PYTHONPATH CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -c "
import os; os.environ['META_INFER_LOG_RANK0_ONLY']='1'
from llm_engine import LLMEngine; from pathlib import Path
e = LLMEngine(model_dir=Path('/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat'), inference_backend='deepseek_tp', max_num_seqs=4)
if int(os.environ.get('RANK','0'))==0: print(e.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0))
"
```

### Step 6: 性能测试

```bash
# 对比 P3-FA (V-padding) vs Triton MLA 的吞吐
SKIP_VLLM=1 CUDA_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 ROUNDS=10 STEPS=8 \
  bash run_compare_metainfer_vllm.sh dsv2
```

## 5. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Triton kernel 在 A800 上性能不如 FA2 | 收益不确定 | 先 benchmark 单层 kernel 耗时再决定 |
| W_UK_T / W_UV bmm 增加额外计算 | decode 每步多 2 次小矩阵乘 | bmm [H,1,128]@[H,128,128] 极快 (~1us) |
| KV cache 格式变更影响 prefill | prefill 需要适配 | prefill 仍用 FA2，只改 cache 写入 |
| torch.compile 与 Triton kernel 冲突 | 可能 graph break | Triton kernel 放在 compile 外或用 custom op |

## 6. 预期收益

| 场景 | FA2 V-padding | Triton MLA | 预期提升 |
|------|--------------|------------|---------|
| V 计算维度 | 192 (64 零填充) | 128 (复用 K) | ~33% 计算减少 |
| KV cache 显存 | 640 KB/layer | 96 KB/layer | 85% 节省 |
| CPU 开销 | F.pad + trim | bmm (极小) | 大幅减少 |
| Kernel 效率 | FA2 通用 kernel | 专为 MLA 优化 | 20-50% |

**总体预期**: decode 路径 1.3-2x 加速（相比 FA2 V-padding）。
