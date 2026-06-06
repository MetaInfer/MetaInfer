# Skill: Phase 5 编码（Attention + KV Cache）

## 触发词

`/inference:phase5` 或 `/phase5`

## 概述

构建 Phase 5: Attention + KV Cache —— 整个流水线中**错误密度最高**的阶段。通过 spawn phase-runner 子代理完成内部对抗审查，主 Agent 只做调度和抽查。

---

## 环境

- 模型权重: `${MODEL_DIR}`
- Conda: `${PYTHON_PATH}/python`
- GPU: 4×A800
- **重要**: Phase 1-4 的代码已存在于 ./engine/ 下，不要重复生成。

## 你的角色

你是**主 Agent**——只做高层调度和抽查，不亲自 orchestrate 三角色。通过 spawn phase-runner 子代理执行 Phase 5，你只看到结构化摘要，保持上下文轻量。

## 本次任务

构建 Phase 5: Attention + KV Cache

必须实现:
- QwenAttentionTP.__init__（KV head replication、num_heads=8 per-rank、num_kv_heads=2、kv_block_size=256、block_table int32、_kv_len_gpu GPU tensor）
- QwenAttentionTP.forward(prefill): qkv_proj → Q/K norm → rotary_embedding → flash_attn_varlen_func(causal=True) → KV cache lazy alloc → block_table torch.arange → slot_mapping → index_copy_ 写入 cache
- QwenAttentionTP.forward_decode(decode): qkv_proj → rotary → slot_mapping=_kv_len_gpu[0] → index_copy_ → _kv_len_gpu+=1 → flash_attn_with_kvcache(causal=False) → o_proj

## 执行方式

### 步骤 1：spawn phase-runner

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase 5 runner",
  prompt: """
Phase 5: Attention + KV Cache。
读取 .claude/skills/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phase5-coding.md 了解本 Phase 的任务细节。
执行完整 implementer→spec→verif 对抗审查链（模式 A：首次执行）。
"""
)
```

phase-runner 返回结构化摘要后，进入步骤 2。

### 步骤 2：主 Agent 防假 PASS 抽查

```bash
RANDOM_SCRIPT=$(ls scripts/test_phase5_*.py scripts/test_phase5_*.sh 2>/dev/null | shuf -n1)
ACTUAL_OUTPUT=$(python "${RANDOM_SCRIPT}" 2>&1 || bash "${RANDOM_SCRIPT}" 2>&1)
```

读取 `./phase_report/PHASE5_VERIFICATION_REPORT.md` 中该脚本的原始 stdout 比对：
- **一致** ✅ → Phase 5 交付，写 `./phase_report/PHASE5_SUMMARY.md`
- **不一致** ❌ → 写 `./phase_report/PHASE5_SPOT_CHECK_FAIL.md` → 回到步骤 1（重试模式）：

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase 5 runner (RETRY)",
  prompt: """
Phase 5 RETRY。
读取 ./phase_report/PHASE5_SPOT_CHECK_FAIL.md 了解失败原因。
读取 .claude/skills/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phase5-coding.md 了解任务细节。
执行完整 implementer→spec→verif 修复链（模式 B：重试修复，不得跳过任何环节）。
"""
)
```

重试后换一个脚本再次抽查。连续 5 次驳回 → 停止，向人类报告。

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

- 主 Agent 只做调度 + 抽查，不亲自 orchestrate 三角色
- phase-runner 内部 implementer/spec-reviewer/verification 三角色物理隔离（Shell claude -p）
- 审查串行：spec ✅ 才到 verif。spec ❌ 时 verif 不启动
- verif 做 L1（Phase 5 脚本）+ L2（Phase 1-4 回归）
- 主 Agent 抽查是最终裁定——不一致就驳回，连续 5 次才停止
- 主 Agent 禁止降级/修改子代理结论
- PID 互不相同
- scripts/ 不可修改。测试不过 → 改实现代码，不改脚本
