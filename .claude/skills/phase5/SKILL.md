# Phase 5：Attention + KV Cache

## 触发词

`/phase5`

## 角色

你是主 Agent。按 CLAUDE.md 的 spawn 协议执行完整串行路径（impl→spec→verify→抽查→汇总）。这是整个流水线中**错误密度最高**的阶段。

## 任务

构建 QwenAttentionTP，必须实现：

- `__init__`：KV head replication、num_heads=8 per-rank、num_kv_heads=2、`_kv_block_size=256`、block_table int32、`_kv_len_gpu` GPU tensor、KV cache 懒分配
- `forward(prefill)`：qkv_proj → Q/K norm → rotary → flash_attn_varlen_func(causal=True) → KV cache lazy alloc → block_table torch.arange → slot_mapping → index_copy_ 写入
- `forward_decode(decode)`：qkv_proj → rotary → slot_mapping=`_kv_len_gpu[0]` → index_copy_ → `_kv_len_gpu+=1` → flash_attn_with_kvcache(**全 keyword args**) → o_proj
- RMSNorm 始终返回 `(out, residual)` 2-tuple
- K/V reshape：`[num_tokens, kv_heads, head_dim]` → flat view → index_copy_

## Phase-Script 绑定

| 脚本 | 门禁 |
|------|------|
| `test_phase5_attention_init.py` | KV head replication + per-rank dimensions |
| `test_phase5_kv_cache_paged.py` | block_size=256 + slot_mapping + block_table |
| `test_phase5_flash_attn_prefill_decode.py` | prefill/decode 正确性 |

verif L2 跨 Phase 回归：重跑 Phase 1-4 全部 8 个脚本。

## 知识映射

- Blueprint：`qwen3_tp_model_interfaces.attention` → `class_hierarchy.QwenAttentionTP` → `paged_kv_cache_contract` → `flash_attention_integration_contract`
- ref_docs：`improvement_plan.md` §P0（⚠️ FG-3：旧 HF 方案矛盾，以蓝图为权威）、§P3-FA；`qwen3_effective_changes.md` #8 #10
- ref_code：蓝图 `decode_forward_pattern` 完整 forward_decode 方法体 pseudocode（可直接抄入）

## ⚠️ 五大高发错误

1. **block_size≠256**：nano-vllm 默认 16 → flash_attn_with_kvcache 要求 ≥256
2. **block_table dtype≠int32**：int64 会报错
3. **QKV reshape 用错 heads**：K/V 用 num_heads=8 而非 num_kv_heads_local=2
4. **prefill K/V 来源错误**：从 cache 读取而非从 qkv_proj 产出
5. **slot_mapping .item() 循环**：长 prompt 须向量化

## 关键约束

- KV cache 必须懒分配（num_blocks_needed = (num_tokens+255)//256，禁止 torch.zeros(max_blocks)）
- flash_attn_with_kvcache 必须全 keyword args（flash_attn 2.8.3+）
- prefill 时 K/V 来自投影输出，非 cache 读
- flash_attn_varlen_func 必须 `causal=True`
