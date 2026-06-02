# 白板 Agent 分阶段 Prompt 集合

## 总览

5 次独立会话，每次喂入一个 prompt。会话间 Git 提交 `engine/`、`llm_engine.py`、`openai_tp_server.py` 和 `phase_report/` 以保持代码连续性。

| 会话 | Phase | 关键任务 | 预计子代理 spawn 次数 |
|------|-------|---------|---------------------|
| 1 | 1-4 | 数值基元 + TP通信 + TP线性层 + TP Embedding | 4 次 implementer + 4 次 spec + 4 次 verif |
| 2 | 5 | Attention/KV Cache（最高错误密度） | 1 次 implementer + 1 次 spec + 1 次 verif |
| 3 | 6 | MLP/Decoder Layer（最高错误密度） | 1 次 implementer + 1 次 spec + 1 次 verif |
| 4 | 7-8 | 权重加载 + 框架外壳 | 2 次 implementer + 2 次 spec + 2 次 verif |
| 5 | 9-10 | 引擎集成 + E2E 验收 | 2 次 implementer + 2 次 spec + 2 次 verif |

---

## 会话 1：Phase 1-4（数值基元 → TP Embedding）

```
你现在在 /home/honglin/inference-agent-system 目录下工作。

## 环境

- 模型权重: ${MODEL_DIR}/qwen/Qwen3-8B
- Conda: ${CONDA_PATH}/bin/python
- GPU: 4×A800（CUDA_VISIBLE_DEVICES=0,1,2,3）

## 你的角色

读取本目录的 CLAUDE.md，理解你的角色、三层知识体系和对抗子代理协作流。

## 本次任务

依次完成 Phase 1 → Phase 2 → Phase 3 → Phase 4，每一 Phase 严格按照 CLAUDE.md 的 spawn 协议执行：

  Phase 1: 数值基元（7 个 vLLM kernel wrapper）
  Phase 2: TP 通信（all_reduce_sum + all_gather_last_dim + CustomAR init）
  Phase 3: TP 线性层（Column/Row/Merged/QKV Parallel Linear）
  Phase 4: TP Embedding（VocabParallelEmbedding + ParallelLMHead）

## 每 Phase 的执行步骤

### 步骤 1：implementer 子代理

使用 Agent 工具 spawn implementer（subagent_type: general-purpose），读取 .claude/skills/implementer-inference.md。
implementer 只写代码，不跑测试。完成后写出 ./phase_report/PHASE<N>_IMPLEMENTER_REPORT.md（含 PID、Role、Timestamp、Phase=N，status=SUBMITTED）。

### 步骤 2：spec-reviewer（Shell claude -p 独立审查）

```bash
claude -p --allowedTools "Read(*),Write(*),Bash(*)" "读取 /home/honglin/inference-agent-system/.claude/skills/spec-reviewer-inference.md。审查 ./engine/ 下的代码。对照 inference_blueprint.json 中 Phase N 的契约逐条核验。将 SPEC_REVIEW_REPORT.md 写入 ./phase_report/（文件名前缀 PHASE<N>_）。文件头含 PID（os.getpid()）、Role=spec-reviewer、Timestamp。"
```

如果 spec ❌ → 打回 implementer 重写，verification 不启动。如果 ✅ → 进入步骤 3。

### 步骤 3：verification（Shell claude -p 独立验收，仅 spec ✅ 后执行）

```bash
claude -p --allowedTools "Read(*),Write(*),Bash(*)" "读取 /home/honglin/inference-agent-system/.claude/skills/verification-inference.md。验收 Phase N：运行 scripts/ 下 Phase N 对应的全部测试脚本。Phase 3+ 必须额外做跨 Phase 回归（重跑前序所有 Phase 的 scripts/）。将 VERIFICATION_REPORT.md 写入 ./phase_report/（文件名前缀 PHASE<N>_）。文件头含 PID（os.getpid()）、Role=verification、Timestamp。"
```

### 步骤 4：汇总

步骤 3.5: 主 Agent 抽查（verification 返回后）：从 Phase N 的 scripts/ 中随机抽 1 个重跑，比对 verification 报告的原始 stdout。一致 → 进入步骤 4。不一致 → 整个验收作废，重新 spawn verification。
步骤 4: 主 Agent 汇总——读取三个报告和抽查结果，验证 PID 互不相同，原样汇总入 ./phase_report/PHASE{N}_SUMMARY.md。禁止降级/修改子代理结论。

代码直接写入本目录下（`./engine/`、`./llm_engine.py`、`./openai_tp_server.py`）。

## Phase-Script 绑定

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 1 | test_phase1_kernel_wrappers.py + test_phase1_kernel_wrappers.sh |
| Phase 2 | test_phase2_tp_communication.py + test_phase2_custom_ar_init.sh |
| Phase 3 | test_phase3_tp_linear.py + test_phase3_tp_linear_tp4.py |
| Phase 4 | test_phase4_tp_embedding.py + test_phase4_tp_embedding_tp4.py |

## Phase 知识映射（AGENT_SKILL.md §2.0.1）

| Phase | 必读 JSON 路径 | 必读 ref_docs | 必查 ref_code |
|-------|---------------|-------------|-------------|
| **Phase 1** | qwen3_kernel_contracts（7 kernel 签名）→ global_primitives_constraints.rmsnorm_precision_law | kernel_replacement_plan.md §九（完整 kernel 调用契约表 + Snippet A-F） | vllm/_custom_ops.py:420-423, vllm/_custom_ops.py:400-410, vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda |
| **Phase 2** | tp_distributed_runtime（init 顺序）→ collectives.all_reduce_sum（custom_op 注册+fake）→ collectives.all_gather_last_dim → collectives.custom_ar_all_reduce（两套 IPC buffer+init_state_machine） | — | vllm/_custom_ops.py:640-680 |
| **Phase 3** | tp_linear_layers（4 种 Linear 伪代码）→ qwen3_8b_model_dims（**_verified_config: gate_up=[6144,4096] NOT [6400,4096]_**） | qwen_dense_tp_implementation_guide.md, task10_tp_qwen_debug_experience.md | — |
| **Phase 4** | tp_embedding_and_lm_head（VocabParallel mask + ParallelLMHead gather） | — | — |

## 关键约束

- implementer 不跑测试、不判 PASS（状态为 SUBMITTED）
- spec-reviewer 和 verification 通过独立的 Shell claude -p 进程执行（新的 PID）
- 审查串行：先 spec-reviewer，通过后才到 verification。spec ❌ 时 verification 不启动
- 主 Agent 是信使非裁判——禁止降级/修改子代理结论。禁止"有条件交付"
- 三个子代理的 PID 必须互不相同
- scripts/ 不可修改。测试不过 → 改实现代码，不改脚本
- Phase 3 开始，verification 必须做跨 Phase 回归（重跑前序 Phase 的全部 scripts/）

## 防长上下文遗忘机制

每完成一个 Phase 后，重新打开 AGENT_SKILL.md §2.0.1 确认下一 Phase 的知识链路。
每完成一个 Phase 后，输出一行进度：`[PROGRESS] Phase N 完成，spec=✅/❌，verif=✅/❌`。
```

---

## 会话 2：Phase 5（Attention + KV Cache）

```
你现在在 /home/honglin/inference-agent-system 目录下工作。

## 环境

- 模型权重: ${MODEL_DIR}/qwen/Qwen3-8B
- Conda: ${CONDA_PATH}/bin/python
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

## 执行步骤（同会话 1，见 CLAUDE.md spawn 协议）

步骤 1: implementer → ./phase_report/PHASE<N>_IMPLEMENTER_REPORT.md（SUBMITTED）
步骤 2: spec-reviewer → ./phase_report/PHASE<N>_SPEC_REVIEW_REPORT.md（仅 ✅ 后进入步骤 3）
步骤 3: verification → ./phase_report/PHASE<N>_VERIFICATION_REPORT.md（L1+L2 跨 Phase 回归）
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
```

---

## 会话3：Phase 6（MLP + Decoder Layer）

```
你现在在 /home/honglin/inference-agent-system 目录下工作。

## 环境

- 模型权重: ${MODEL_DIR}/qwen/Qwen3-8B
- Conda: ${CONDA_PATH}/bin/python
- GPU: 4×A800
- Phase 1-5 的代码已存在于 ./engine/ 下，不要重复生成。
- **关键**：`engine/models/qwen.py` 已存在——Phase 5 创建了完整 QwenAttentionTP + QwenDecoderLayerTP + RMSNorm，以及 QwenMLPTP stub（仅 __init__，forward 是空壳）。Phase 6 的职责是修改 qwen.py，**不是新建文件**：
  - **补全 QwenMLPTP.forward**：stub → 完整 gate_up→silu_and_mul→down 链
  - **审查 QwenDecoderLayerTP 的 prefill/decode 路径**：Phase 5 已实现，Phase 6 需确保 residual chain 与蓝图完全一致
  - **禁止修改 QwenAttentionTP 和 RMSNorm**：Phase 5 已完成并通过验收

## 你的角色

读取本目录的 CLAUDE.md。Phase 6 是整个流水线中**错误密度与 Phase 5 并列最高的阶段**。

## 本次任务

构建 Phase 6: MLP + Decoder Layer

**在现有 `engine/models/qwen.py` 上修改，不是新建文件。**

必须补全/实现:
- QwenMLPTP.forward: gate_up_proj(MergedColumnParallelLinear) → silu_and_mul → down_proj(RowParallelLinear)（Phase 5 只有 stub，**替换为完整实现**）
- QwenDecoderLayerTP.forward(prefill): 确认 input_layernorm → attention.forward → post_attention_layernorm → mlp 链正确
- QwenDecoderLayerTP.forward_decode(decode): 确认 fused_add_rms_norm(input,residual,self.input_layernorm.weight) → attention.forward_decode → fused_add_rms_norm(attn_out,residual,self.post_attention_layernorm.weight) → mlp 链正确
- Residual chain: 首层 res=None → clone+rms_norm；后续层 fused_add_rms_norm（res+=hs; hs=rms_norm(res)）

## ⚠️ 最关键约束：FM-003

**所有 4 处 fused_add_rms_norm 的 weight 参数必须是本层 self.weight。**
过去 V5/V15/V17 三轮审计 Agent 反复犯的错误：将 post_mlp 的 weight 错误引用为下一层的 input_layernorm.weight。
用 id() 做 identity check（非 value check），确保每层只用自己的 weight。

## 执行步骤

步骤 1: implementer → ./phase_report/PHASE<N>_IMPLEMENTER_REPORT.md（SUBMITTED）
步骤 2: spec-reviewer → ./phase_report/PHASE<N>_SPEC_REVIEW_REPORT.md（Shell claude -p，独立 PID）
步骤 3: verification → ./phase_report/PHASE<N>_VERIFICATION_REPORT.md（L1+L2，独立 PID）
步骤 4: 主 Agent 汇总（含步骤 3.5 防假 PASS 抽查）→ ./phase_report/PHASE6_SUMMARY.md（PID 交叉验证）

## Phase Script 绑定

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 6 | test_phase6_mlp_forward.py + test_phase6_residual_chain.py + test_phase6_decode_forward_no_clone.py + test_phase6_layer_e2e_random_weights.py |

verification L2: 重跑 Phase 1-5 的全部 scripts/（共 11 个脚本）。

## Phase 6 知识映射

### 必读 JSON 路径

1. qwen3_tp_model_interfaces.mlp — gate_up→silu_and_mul→down 链
2. qwen3_tp_model_interfaces.decode_forward_pattern — **完整 forward_decode 方法体 pseudocode（可直接抄入）**
3. qwen3_tp_model_interfaces.prefill_forward_pattern — prefill 完整数据流 8 步
4. qwen3_tp_model_interfaces.class_hierarchy.QwenMLPTP + QwenDecoderLayerTP — __init__ attr 名
5. qwen3_kernel_contracts.fused_add_rms_norm — **4 处调用均为本层 self.weight**

### 必读 ref_docs

- kernel_replacement_plan.md §三（Snippet B: fused_add_rms_norm, Snippet C: silu_and_mul）

## ⚠️ Phase 6 高发错误

1. **FM-003 跨层 weight**: fused_add_rms_norm 用了下一层的 weight → 输出无 shape 错误但数值全错
2. **gate_up=6400**: 旧 intermediate_size=12800 → gate_up=6400，正确是 6144（12288/4×2）
3. **Eager 路径残留 clone()**: forward_decode 含 .clone() → ~15% 吞吐回退
4. **residual 链断裂**: 首层 res=None 时错误调用了 fused_add_rms_norm 而非 rms_norm

## 关键约束

- implementer 不跑测试、不判 PASS
- spec 先审 → ✅ 才到 verif
- verif 做 L1+L2
- 主 Agent 禁止降级子代理结论
- PID 互不相同
```

---

## 会话 4：Phase 7-8（权重加载 + 框架外壳）

```
你现在在 /home/honglin/inference-agent-system 目录下工作。

## 环境

- 模型权重: ${MODEL_DIR}/qwen/Qwen3-8B
- Conda: ${CONDA_PATH}/bin/python
- GPU: 4×A800
- Phase 1-6 的代码已存在于 ./engine/ 下。

## 你的角色

读取本目录的 CLAUDE.md。本次依次完成 Phase 7 → Phase 8，每 Phase 独立走完 implementer→spec→verif→汇总。

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

## 关键约束（每 Phase 均适用）

- implementer→spec(L1)→verif(L1+L2 回归)→汇总。串行不可跳过
- 主 Agent 禁止降级子代理结论
- PID 互不相同
```

---

## 会话 5：Phase 9-10（引擎集成 + E2E 验收）

```
你现在在 /home/honglin/inference-agent-system 目录下工作。

## 环境

- 模型权重: ${MODEL_DIR}/qwen/Qwen3-8B
- Conda: ${CONDA_PATH}/bin/python
- GPU: 4×A800（TP=4）
- Phase 1-8 的代码已存在于 ./engine/ 下。

## 你的角色

读取本目录的 CLAUDE.md。这是最后两个 Phase——Phase 9 将 Scheduler 和 ModelRunner 胶合为 LLMEngine，Phase 10 做最终验收。

## Phase 9: 引擎集成

必须实现:
- LLMEngine.__init__: (1) torch.cuda.set_device (2) _select_tp_backend(model_dir) 自动路由 (3) 创建 Runner (4) eos_token_id (5) _estimate_kv_blocks() (6) KVMemoryPool (7) Scheduler
- LLMEngine.generate(): _enqueue → while-loop(schedule→run→postprocess→finish_check) → decode → return
- LLMEngine.step(): scheduler.schedule → runner.run → postprocess → finish_check → return finished
- begin_generation() / has_unfinished_requests() / get_generation_outputs() — OpenAI server 三步接口
- 集成桥接: block_size 注入（TP→256,HF→16）、num_free 来源路由（TP→runner,HF→block_manager）、BlockManager 降级

### Phase 9 Scripts

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 9 | test_phase9_llm_engine_init.py + test_phase9_generate_single_gpu.sh |

### Phase 9 知识映射

必读 JSON: components[6] LLMEngine（full_api_surface: __init__ 7 步 + generate 5 步 + step）→ components[3] ModelRunner（tp_runner_actual_flow）→ scheduler_tp_runner_bridge（CRITICAL-01）
必读 ref_docs: 01_architecture.md, 07_request_lifecycle.md
必查 ref_code: ref_projects/nano-vllm/nanovllm/engine/llm_engine.py（⚠️ 删除 use_cache=False）、model_runner.py（⚠️ 重载清单）

## Phase 10: E2E 验收

必须实现:
- openai_tp_server.py: ThreadingHTTPServer + /v1/completions
- TP 同步: non-rank0 while-loop(broadcast_obj) + rank0 HTTP handler + threading.Lock 序列化
- streaming(SSE) + non-streaming(JSON)

### Phase 10 Scripts（含硬性验收底线）

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 10 | test_phase10_greedy_align.sh + test_phase10_benchmark.sh + test_phase10_no_compile_check.sh + test_phase10_vs_vllm_compare.sh |

### Phase 10 硬性验收指标（不达标不能声称完成）

| 指标 | 目标 |
|------|------|
| Greedy decode | temperature=0.0 输出 `（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑` |
| Output throughput | ≥ 54 tok/s |
| GPU Self CUDA | ≤ 66ms / step |
| CustomAR 通信 | ≤ 25ms / step |
| cudaGraphLaunch | **0**（纯 Eager，无 torch.compile / CUDA Graph 痕迹） |
| VRAM% per rank | ~7%（4 卡同量级一致） |
| HCU% | > 0（真实计算证据） |

### Phase 10 L3 强制（verification 必须采集）

- profiler trace 文件确认: 无 `CompiledFunction`、无 `Inductor` kernel、`cudaGraphLaunch=0`
- HCU/VRAM 监控: 4 卡 VRAM% 同量级、HCU% 峰值 > 0
- **无证据 = 假推理 = 验收失败**

## 执行步骤（每 Phase 独立）

步骤 1: implementer → ./phase_report/PHASE<N>_IMPLEMENTER_REPORT.md（SUBMITTED）
步骤 2: spec-reviewer → ./phase_report/PHASE<N>_SPEC_REVIEW_REPORT.md（Shell claude -p --allowedTools，独立 PID）
步骤 3: verification → ./phase_report/PHASE<N>_VERIFICATION_REPORT.md（Shell claude -p --allowedTools，独立 PID。Phase 9: L1+L2。Phase 10: L1+L2+L3 全量）
步骤 4: 主 Agent 汇总（含步骤 3.5 防假 PASS 抽查）→ ./phase_report/PHASE{N}_SUMMARY.md（PID 验证 + 原样转述）

## 关键约束

- implementer 不跑测试、不判 PASS
- spec 先审 → ✅ 才到 verif
- Phase 9 verif L2: 重跑 Phase 1-8 全部 scripts/（共 22 个脚本）
- Phase 10 verif L3: **强制** profiler + HCU 证据
- 主 Agent 禁止降级子代理结论
- PID 互不相同
```
