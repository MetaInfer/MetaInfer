# Phase 1-4：数值基元 → TP Embedding

## 触发词

`/phase1-4`

## 角色

你是主 Agent。按 CLAUDE.md 的 spawn 协议，依次完成 Phase 1→2→3→4，每个 Phase 走完整串行路径（impl→spec→verify→抽查→汇总）。不引入额外中间层。

## 任务

| Phase | 构建内容 |
|-------|---------|
| 1 | 7 个 vLLM kernel wrapper（rms_norm, fused_add_rms_norm, rotary_embedding, silu_and_mul, flash_attn_varlen_func, flash_attn_with_kvcache, gptq_gemm） |
| 2 | TP 通信：all_reduce_sum + all_gather_last_dim + CustomAR init |
| 3 | TP 线性层：ColumnParallel / RowParallel / MergedColumnParallel / QKVColumnParallel Linear |
| 4 | TP Embedding：VocabParallelEmbedding + ParallelLMHead |

## Phase-Script 绑定

| Phase | 门禁脚本 |
|-------|---------|
| 1 | `test_phase1_kernel_wrappers.py` + `.sh` |
| 2 | `test_phase2_tp_communication.py` + `test_phase2_custom_ar_init.sh` |
| 3 | `test_phase3_tp_linear.py` + `test_phase3_tp_linear_tp4.py` |
| 4 | `test_phase4_tp_embedding.py` + `test_phase4_tp_embedding_tp4.py` |

## 知识映射

按 AGENT_SKILL.md §2.0.1 三步链路，implementer 启动前必须读：

| Phase | Blueprint JSON | ref_docs | ref_code |
|-------|---------------|----------|----------|
| 1 | `qwen3_kernel_contracts` → `global_primitives_constraints.rmsnorm_precision_law` | `kernel_replacement_plan.md` §九 | `vllm/_custom_ops.py` |
| 2 | `tp_distributed_runtime` → `collectives.*` | — | `vllm/_custom_ops.py:640-680` |
| 3 | `tp_linear_layers` → `qwen3_8b_model_dims`（gate_up=[6144,4096] 非 [6400,4096]） | `qwen_dense_tp_implementation_guide.md`, `task10_tp_qwen_debug_experience.md` | — |
| 4 | `tp_embedding_and_lm_head` | — | — |

## ⚠️ 高发错误

- **Phase 3**：QKV cat 顺序必须是 Q-K-V（严禁 K-Q-V）；per-rank 维度必须用蓝图 verified_config 中的数字，不能脑补
- **Phase 4**：VocabParallelEmbedding 的 mask 必须是 `vocab_size // tp_size` 对齐；ParallelLMHead 需 all_gather 而非 all_reduce

## 关键约束

- 每个 Phase 独立走完整 impl→spec→verify→抽查→汇总 流程，通过后再进下一 Phase
- Phase 3 开始 verif 必须做跨 Phase 回归
- 代码写入 `./engine/` 下，不改 scripts/
- 主 Agent 禁止降级子代理结论
