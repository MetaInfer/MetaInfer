# Skill: Phase 5 编码（Attention + KV Cache）

## 触发词

`/inference:phase5` 或 `/phase5`

## 概述

构建 Phase 5: Attention + KV Cache —— 整个流水线中**错误密度最高**的阶段，请严格遵循每步协议。

---

## 环境

- 模型权重: `${MODEL_DIR}`
- Conda: `${PYTHON_PATH}/python`
- GPU: 4×A800
- **重要**: Phase 1-4 的代码已存在于 ./engine/ 下，不要重复生成。

## 你的角色

读取本目录的 CLAUDE.md，理解对抗子代理协作流。本次仅构建一个 Phase——Phase 5 是整个流水线中**错误密度最高**的阶段，请严格遵循每步协议。

## 本次任务

构建 Phase 5: Attention + KV Cache

必须实现:
- QwenAttentionTP.__init__（KV head replication、num_heads=8 per-rank、num_kv_heads=2、kv_block_size=256、block_table int32、_kv_len_gpu GPU tensor）
- QwenAttentionTP.forward(prefill): qkv_proj → Q/K norm → rotary_embedding → flash_attn_varlen_func(causal=True) → KV cache lazy alloc → block_table torch.arange → slot_mapping → index_copy_ 写入 cache
- QwenAttentionTP.forward_decode(decode): qkv_proj → rotary → slot_mapping=_kv_len_gpu[0] → index_copy_ → _kv_len_gpu+=1 → flash_attn_with_kvcache(causal=False) → o_proj

## 执行步骤（同 CLAUDE.md spawn 协议）

步骤 1: implementer → ./phase_report/PHASE5_IMPLEMENTER_REPORT.md（SUBMITTED）
步骤 2: spec-reviewer → ./phase_report/PHASE5_SPEC_REVIEW_REPORT.md（仅 ✅ 后进入步骤 3）
步骤 3: verification → ./phase_report/PHASE5_VERIFICATION_REPORT.md（L1+L2 跨 Phase 回归）
步骤 4: 主 Agent 汇总（含步骤 3.5 防假 PASS 抽查）→ ./phase_report/PHASE5_SUMMARY.md（必须验证 PID 互不相同）

## Phase Script 绑定

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 5 | test_phase5_attention_init.py + test_phase5_kv_cache_paged.py + test_phase5_flash_attn_prefill_decode.py |

verification L2 跨 Phase 回归：重跑 Phase 1-4 的全部 scripts/（共 8 个脚本）。

## Phase 5 知识映射

### 必读 JSON 路径

1. qwen3_tp_model_interfaces.attention — KV cache paged 格式 + block_table + slot_mapping
2. qwen3_tp_model_interfaces.class_hierarchy.QwenAttentionTP — **每个 attr 的精确名称和 shape 公式**
3. paged_kv_cache_contract — prefill/decode KV write + slot_mapping 公式（`block_table[0,i//256]*256+(i%256)`）+ index_copy_
4. flash_attention_integration_contract — prefill_path + decode_path kernel 调用

### 必读 ref_docs

- improvement_plan.md §P0（⚠️ FG-3：此 doc 描述旧 HF 方案，与 paged KV 架构矛盾，以**蓝图**为权威）
- improvement_plan.md §P3-FA
- qwen3_effective_changes.md #8 #10

### 必查 ref_code

- engine/models/qwen.py::QwenAttentionTP — 蓝图 full_method_body 给出了**完整 forward_decode 方法体（可直接抄入）**

## ⚠️ Phase 5 高发错误（Agent 最常犯的 5 个错误）

1. **block_size≠256**: nano-vllm 默认 16 → flash_attn_with_kvcache 要求 ≥256 → RuntimeError
2. **block_table dtype≠int32**: 用了 int64 → flash_attn_with_kvcache 报错
3. **QKV reshape 用错 heads**: K/V 用 num_heads=8 而非 num_kv_heads_local=2 → 8×128=1024≠kv_size=256 → shape mismatch
4. **prefill K/V 来源错误**: 从 cache 读取而非从 qkv_proj 产出 → 数值错误
5. **slot_mapping .item() 循环**: 长 prompt 用了 for-loop 而非向量化 → prefill >50ms overhead

## 关键约束

- implementer 不跑测试、不判 PASS
- spec 先审 → ✅ 才到 verif
- verif 做 L1（Phase 5 脚本）+ L2（Phase 1-4 回归）
- 主 Agent 禁止降级子代理结论
- PID 互不相同
