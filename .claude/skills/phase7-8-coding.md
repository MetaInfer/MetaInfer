# Skill: Phase 7-8 编码（权重加载 + 框架外壳）

## 触发词

`/inference:phase7-8` 或 `/phase7-8`

## 概述

依次完成 Phase 7（权重加载）→ Phase 8（框架外壳）。每个 Phase 通过 spawn phase-runner 子代理完成内部对抗审查，主 Agent 只做调度和防假 PASS 抽查。

---

## 环境

- **必须先执行** `source .env_agent_infer` 加载 MODEL_DIR 和 PYTHON_PATH
- 模型权重: `${MODEL_DIR}`
- Conda: `${PYTHON_PATH}/python`
- GPU: 4×A800
- Phase 1-6 的代码已存在于 ./engine/ 下。

## 你的角色

你是**主 Agent**——只做高层调度和抽查，不亲自 orchestrate 三角色。每个 Phase 通过 spawn phase-runner 子代理执行，你只看到结构化摘要，保持上下文轻量。

## Phase 7: 权重加载

必须实现:
- QwenTPConfig dataclass: 从 config.json 动态读取所有字段（**严禁硬编码**）
- QwenForCausalLMTP.__init__: embed_tokens → layers[QwenDecoderLayerTP]×36 → norm → lm_head
- HF key mapping: QKV cat 顺序 **Q-K-V**（严禁 K-Q-V）；Gate-Up cat 顺序 **gate-up**
- load_weights(): safetensors.index.json → safe_open → double_shard_guard → load_weights 后 dist.barrier() + init_custom_ar()
- 每 rank 显存 < 全量模型/N（**~4.7GB/rank，非 ~15.7GB**）

### Phase 7 Scripts

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 7 | test_phase7_qwen_tp_config.py + test_phase7_hf_key_mapping.py + test_phase7_weight_loading.sh |

### Phase 7 知识映射

必读 JSON: class_hierarchy.QwenTPConfig（**_verified_config 标注 2026-05-27 物理验证**）→ class_hierarchy.QwenForCausalLMTP → construction_chain → qwen_hf_key_mapping（12 个 HF key→attr 映射+cat 顺序）→ qwen3_8b_model_dims
必读 ref_docs: kernel_replacement_plan.md §四
必查 ref_code: engine/tp_layers/linear.py::QKVColumnParallelLinear

## Phase 8: 框架外壳

Phase 8 的组件（Scheduler/Sequence/Sampler/BlockManager）与模型层物理独立——可从 nano-vllm 参考代码抄入，但必须按 AGENT_SKILL.md §2.2 的重载清单逐条修改。

必须实现:
- Sequence: input_ids/output_ids/block_table(list+Tensor 双轨)/status(WAITING→PREFILL→DECODE→FINISHED/REJECTED)
- Scheduler: schedule(num_free) 先 prefill 后 decode + REJECTED + _reserved_blocks。**删除 preempt() 逻辑**
- Sampler: temperature=0.0 greedy + top_p。TP 协议: rank 0 采样 + dist.broadcast
- BlockManager: TP 降级 no-op（_tp_degradation_fork_interface）。allocate/free 空方法

### Phase 8 Scripts

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 8 | test_phase8_sequence_scheduler.py + test_phase8_sampler_tp.py |

### Phase 8 知识映射

必读 JSON: components[0] Scheduler（完整 schedule+postprocess 伪代码）→ components[4] Sampler（TP 协议）→ components[5] Sequence（status+block_table 双轨）→ components[2] BlockManager（TP 降级）
必读 ref_docs: 02_scheduler.md, 03_kv_cache.md, 05_sampler.md, 01_architecture.md, 07_request_lifecycle.md
必查 ref_code: ref_projects/nano-vllm/nanovllm/engine/scheduler.py（⚠️ 删除 preempt L66-69）、block_manager.py、sequence.py、sampler.py

## ⚠️ Phase 7 高发错误

1. **QKV cat 顺序 K-Q-V**: Agent 频繁犯——必须 Q-K-V
2. **double_shard_guard 失效**: 预切片权重被二次切片 → shape mismatch
3. **每卡加载全量**: ~15.7GB 而非 ~4.7GB → OOM
4. **max_position_embeddings 硬编码 32768**: 实际 40960（FG-1）

## ⚠️ Phase 8 高发错误

1. **preempt() 未删除**: nano-vllm L66-69 保留 → running.pop() 序列丢失
2. **block_size 硬编码 16**: TP 路径未注入 256
3. **TP 各 rank 独立采样**: 未做 rank0+broadcast → KV 不同步 → NCCL 崩溃

## 执行方式

对 Phase 7 和 Phase 8，分别执行以下循环：

### 步骤 1：spawn phase-runner

```
Agent(
  subagent_type: "general-purpose",
  description: "Phase N runner",
  prompt: """
Phase N: [Phase名称]。
读取 .claude/skills/phase-runner.md 了解你的角色边界。
读取 .claude/skills/phase7-8-coding.md 了解本 Phase 的任务细节。
执行完整 implementer→spec→verif 对抗审查链（模式 A：首次执行）。
"""
)
```

phase-runner 返回结构化摘要后，进入步骤 2。

### 步骤 2：主 Agent 防假 PASS 抽查

```bash
RANDOM_SCRIPT=$(ls scripts/test_phase${N}_*.py scripts/test_phase${N}_*.sh 2>/dev/null | shuf -n1)
ACTUAL_OUTPUT=$(python "${RANDOM_SCRIPT}" 2>&1 || bash "${RANDOM_SCRIPT}" 2>&1)
```

读取 `./phase_report/PHASE${N}_VERIFICATION_REPORT.md` 中该脚本的原始 stdout 比对：
- **一致** ✅ → 该 Phase 交付，进入步骤 3
- **不一致** ❌ → 写 `./phase_report/PHASE${N}_SPOT_CHECK_FAIL.md` → 回到步骤 1（重试模式，模式 B）。连续 5 次驳回 → 停止，向人类报告。

### 步骤 3：写 Phase 汇总

抽查通过后，写 `./phase_report/PHASE${N}_SUMMARY.md`，含 PID 交叉验证和抽查结果。然后进入下一 Phase。

## 关键约束（每 Phase 均适用）

- 主 Agent 只做调度 + 抽查，不亲自 orchestrate 三角色
- phase-runner 内部 implementer/spec-reviewer/verification 三角色物理隔离（Shell claude -p）
- 审查串行：spec ✅ 才到 verif。spec ❌ 时 verif 不启动
- 主 Agent 抽查是最终裁定——不一致就驳回，连续 5 次才停止
- 主 Agent 禁止降级/修改子代理结论
- PID 互不相同
- scripts/ 不可修改。测试不过 → 改实现代码，不改脚本
