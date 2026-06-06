# AGENT SKILL: TP/EP Inference Generation SOP

你是 `agent-infer` 的推理框架生成 Agent。你的唯一目标是：  
**基于 `inference_blueprint.json` 的架构知识图谱，生成单路径、可观测、可验证、可自愈的 TP/EP 推理实现。**

### 0.-2 路径兼容规则（开源分发保护）

本知识包中所有 `ref_docs` 路径已统一为 `notebooks-cn/`。若在任何文件（blueprint、AGENT_SKILL、外部引用）中遇到 `notebooks/`（不带 `-cn`）的路径，**自动替换为 `notebooks-cn/`** 后再解析。此规则确保本包在拷贝到不同环境或被人 fork 后，路径仍然有效。

### 0.-1 ref_docs 强制交叉验证（防信息孤岛）

`inference_blueprint.json` 中的 `ref_docs` / `source_refs` 不是"参考资料列表"，而是**蓝图信息的扩展存储**。蓝图中的伪代码、接口签名、维度参数可能只在 ref_doc 中有完整定义（例如 `kernel_replacement_plan.md §九` 包含 7 个 kernel 的精确调用契约）。

**强制规则**：
1. 在改动任何涉及 `ref_docs` 的组件前，必须先**实际打开并阅读**对应的 ref_doc 文件，确认蓝图引用的知识点确实存在于文档中。
2. 提交代码前，对改动涉及的每个 JSON 组件，逐条核对：
   - `ref_docs` 中引用的文档路径是否存在且可读？
   - 蓝图声称的签名/公式/约束在 ref_doc 中是否有对应内容？
   - 蓝图自身的 pseudocode 是否自包含（不依赖 ref_doc 中缺失的内容）？
3. 若 ref_doc 与蓝图信息冲突，以蓝图为准并记录差异。若 ref_doc 中找不到蓝图引用的知识点，视为**蓝图信息断裂**，标记为 🟡 Warning 并报告，不得脑补。
4. 审计类任务在检查 `ref_code` 指针精度的同时，**必须**抽样打开至少 3 个被引用的 ref_doc 文件，逐段验证至少 1 个蓝图引用的知识点确实存在于对应文档中。

### 0.0 主执行 Agent + 监控子代理（唯一例外）

1. 代码实现与改动由**当前主 Agent**顺序完成；禁止多个实现型 subagents 并行改同一仓库。  
2. 允许且仅允许开启 **1 个只读监控子代理**，职责仅限实时采集并汇报 HCU/VRAM 指标，**不得写代码**。  
3. 长任务仍按 **phase 分步**（见 blueprint `todo_generation_playbook`），但默认目标是一次会话跑通 `phase_1 ~ phase_9`（到引擎集成完成）。

### 0.0.1 反偷懒与反假输出（强制执行）

1. **禁止**打印硬编码、占位或演示用“假生成结果”冒充真实推理。  
2. **禁止**在未执行 `load_weights()` / 未从磁盘读入权重的情况下声称 TP 已跑通。  
3. 验收输出必须来自：`pytest` 真实跑过 或 `torchrun` 下真实前向；须能指出**对应代码路径**与**日志中的显存证据**（见下节）。  
4. 单卡 HF 调试用例**不得**替代 `torchrun --nproc_per_node=4` 的 TP 验收；两者可并存，但 TP 目标必须满足。

### 0.0.2 长上下文防”遗忘”（一次跑完 phase1-9 的前提）

模型在长对话中会丢失早期约束。**必须**在每完成一个 blueprint phase 或每轮大改前：**重新打开并扫一眼**  
`AGENT_SKILL.md` 与本任务相关的 `inference_blueprint.json` 节点（可用 `agent_navigation` 跳转）。可选：在仓库写 `agents-infer/PROGRESS.md` 记下当前 phase 与已验证命令。

### 0.0.3 逐 Phase scripts/ 门禁与一次完成 phase1-9

1. 开始前先输出 phase 清单与每 phase 的 scripts/ 测试命令（见 §2.0 表格）。
2. **每完成一个 Phase 的实现代码后，必须立即运行该 Phase 对应的全部 scripts/ 脚本**。全部 PASS 后才能进入下一 Phase。
3. phase1 通过后**不得结束任务**，必须继续 phase2/3/...，并最终执行 phase9。
4. 只有当 `phase_9_engine_integration` 的全部 scripts/ 通过 + E2E 命令全部通过，才允许输出”任务完成”。
5. Phase 10 为最终验收阶段——必须额外提供 profiler trace 和 HCU/VRAM 监控证据。

---

## 0. 启动前强制动作

1. 进入项目根目录：本仓库根目录（即 CLAUDE.md 所在目录）。
2. **物理块双轨制感知边界（CRITICAL-01 强制）**：
   - TP Runner 路径（`inference_backend="qwen_tp"/"deepseek_tp"`）：KV cache 由模型自管（`_kv_block_size=256`，`torch.arange` 顺序分配）。**严禁** BlockManager API 接入 TP Runner。
   - HF 兜底路径（`inference_backend="hf"`）：框架层 paging（BlockManager, block_size=16）正常生效。
   - **集成桥接接口**：LLMEngine.step() 中根据 `inference_backend` 注入 `block_size` (16/256) 到 Scheduler，并选择 `num_free_blocks` 来源（BlockManager 或 runner.get_num_free_blocks()）。`scheduler.schedule(num_free)` 接受 num_free 参数。TP 路径下 BlockManager.allocate/free 降级为纯计数 no-op。QwenTPModelRunner 必须暴露 `get_num_free_blocks()` 方法。KVMemoryPool 在 TP 路径仅做显存预算日志，实际 KV cache 由 QwenAttentionTP 内部 torch.zeros 创建。详见 `inference_blueprint.json > scheduler_tp_runner_bridge`。
3. 读取并同步以下上下文：
   - `inference_blueprint.json`
   - `notebooks-cn/04_parallel_strategies/*`（TP 切分策略）
   - `notebooks-cn/06_experience/*`（TP 调试经验）
   - `notebooks-cn/07_improvementPlan/improvement_plan.md`（框架层改动记录：P0 KV Cache、P2 torch.compile、P3-FA Flash Attention）
   - `notebooks-cn/07_improvementPlan/kernel_replacement_plan.md`（kernel 替换计划，后续阶段使用）
   - `CLAUDE.md`（当前项目状态总览）
3. 对输入模型目录先读取 `config.json`，提取：
   - `architectures`
   - `rope_scaling`
   - `num_attention_heads`、`num_key_value_heads`
   - `n_routed_experts`、`n_shared_experts`、`num_experts_per_tok`
4. 在开始写代码前，先输出“模型路由结论”：该模型属于 Dense 还是 MLA+MoE。

---

## 1. 执行铁律（Prime Directives）

1. **契约优先**：所有实现必须受 `inference_blueprint.json` 约束，禁止模型无关脑补。
2. **单路径优先**：只生成当前架构所需代码路径，禁止引入巨石型多分支。
3. **TDD 强制**：先写/改测试，再写实现。
4. **证据优先**：先打点和对齐证据，再猜根因。
5. **不重复切片**：权重加载链路必须规避双重切分。
6. **HF 对齐防 OOM**：多进程测试时 HF 基准模型禁止 `.to(device)` 常驻 GPU。
7. **日志可追溯**：改动后必须输出文件清单、原因、验证结果。
8. **自下而上 + scripts/ 门禁**：严格按 §2.0 的 Phase 顺序构建（数值基元→TP通信→线性层→Embedding→Attention→Decoder→权重加载→框架外壳→引擎集成）。每 Phase 的 scripts/ 测试全部 PASS 后，才能进入下一 Phase。Agent 自写 tests/ 不能替代 scripts/。

### 1.1 面向可观测性设计（Design for Observability）

1. **原生探针注入**  
   复杂算子（如 `DeepseekAttentionTP`、`ExpertParallelMoE`）必须预留 `DEBUG_MODE` 开关。
2. **数值健康检查**  
   每层 forward 末尾必须预留 NaN/Inf 检查路径，例如：
   - `assert not torch.isnan(out).any()`
   - `assert not torch.isinf(out).any()`
3. **维度白盒化**  
   关键中间变量必须支持打印 shape/device/dtype（受 debug 开关控制）。
4. **CUDA Graph 防御性探针**：Graph 捕获态内严禁 `.item()`、`torch.cuda.synchronize()`、`torch.cuda.get_rng_state()`。NaN 健康检查必须用 `is_current_stream_capturing()` 守卫。

### 1.2 显存与 Device 可观测性（TP 排障强制）

在 **TP 相关测试或首次成功 load_weights 后**，每个 rank 至少打印一次（可用 `META_INFER_LOG_RANK0_ONLY=1` 仅 rank0 详打，其它 rank 可简写）：

1. **进程与设备**
   - `RANK` / `LOCAL_RANK` / `WORLD_SIZE`
   - `torch.cuda.current_device()` 与 `torch.cuda.get_device_name(local_rank)`
2. **显存（证明非全量重复加载）**
   - `torch.cuda.memory_allocated()`、`torch.cuda.max_memory_allocated()`（单位 MB 即可）
3. **关键张量**
   - 首层与末层某权重或一次 forward 的 `logits` / `hidden_states` 的 `device`、`dtype`、`shape`

目的：快速发现「每卡加载全量权重」「张量留在 CPU」「设备不一致」等问题。

### 1.3 4 卡 TP 运行时监控门禁（防假推理）

在执行真实 TP 推理（Qwen3/DeepSeek-V2）期间，主 Agent 必须配合“只读监控子代理”输出如下证据：

1. **HCU 监控范围**  
   - 持续监控 `HCU 0,1,2,3` 的 HCU% 与 VRAM%（若机器实际设备数少于 5，需先打印可见设备清单并说明差异）。  
2. **4 卡 TP 一致性**  
   - `torchrun --nproc_per_node=4` 期间，参与 TP 的 4 张卡 VRAM% 需出现同量级且近似一致的区间。  
3. **目标区间（经验阈值）**  
   - Qwen3-8B 在 TP=4 推理时，每卡 VRAM% 约 `7%`（MODEL_DIR 指向 Qwen3-8B 目录）。  
   - DeepSeek-V2-Lite-Chat 在 TP=4 推理时，每卡 VRAM% 约 `14%`（MODEL_DIR 指向 DeepSeek-V2-Lite-Chat 目录）。  
4. **真实并行计算证据**  
   - 测试窗口内需出现 HCU% `> 0` 的时刻，且至少覆盖 4 张 TP 卡。  
5. **反作弊约束**  
   - 若仅有硬编码文本输出、无 HCU/VRAM 证据链，视为“假推理”，验收失败。

---

## 2. 开发流水线总览（自下而上 10 Phase）

> **构建范围**: nocompile, B=1, TP=4。绝对禁止 torch.compile 和 CUDA Graph。纯 Eager 模式。
> **Phase 顺序由物理依赖拓扑决定，不可重排。** 每 Phase 通过 TDD 门禁后才能进入下一 Phase。
> 完整实现在 `inference_blueprint.json > todo_generation_playbook`。本节仅保留关键参考信息。

```
Phase 1: 数值基元    → Phase 2: TP通信    → Phase 3: TP线性层
Phase 4: TP Embedding → Phase 5: Attention  → Phase 6: Decoder+MLP
Phase 7: 权重加载     → Phase 8: 框架外壳  → Phase 9: 引擎集成
                                                    ↓
                                            Phase 10: E2E验收
                                                    ↓
                                            Phase 11: 性能优化
```

### 编码入口：Skill 触发器

用户通过简短触发词启动各阶段编码，Agent 自动加载 `.claude/skills/phaseN-coding.md` 获得完整 prompt（含任务上下文、执行步骤、知识映射、高发错误清单）。

| 触发词 | Phase | Skill 文件 |
|--------|-------|-----------|
| `/phase1-4` | 1-4 | `phase1-4-coding.md` |
| `/phase5` | 5 | `phase5-coding.md` |
| `/phase6` | 6 | `phase6-coding.md` |
| `/phase7-8` | 7-8 | `phase7-8-coding.md` |
| `/phase9-10` | 9-10 | `phase9-10-coding.md` |
| `/phase11` | 11 | `phase11-coding.md` |

详见 `prompts.md`。

**Phase 8 可与 Phase 1-7 并行开发**（框架与模型在物理上互相独立），但 Phase 9 必须等双方完成。

### 2.0 Phase-脚本硬绑定与逐 Phase 交付门禁（scripts/ 铁律）

每个 Phase 必须运行对应 scripts/ 下的**所有**测试脚本，**全部 PASS** 后该 Phase 才算交付。Agent 不能跳过任何 Phase 的脚本检查直接进入下一 Phase。

| Phase | 交付前提（全部 PASS 才算完成） | 脚本数 |
|-------|------------------------------|--------|
| Phase 1 数值基元 | `test_phase1_kernel_wrappers.py` + `test_phase1_kernel_wrappers.sh` | 2 |
| Phase 2 TP 通信 | `test_phase2_tp_communication.py` + `test_phase2_custom_ar_init.sh` | 2 |
| Phase 3 TP 线性层 | `test_phase3_tp_linear.py` + `test_phase3_tp_linear_tp4.py` | 2 |
| Phase 4 TP Embedding | `test_phase4_tp_embedding.py` + `test_phase4_tp_embedding_tp4.py` | 2 |
| Phase 5 Attention/KV Cache | `test_phase5_attention_init.py` + `test_phase5_kv_cache_paged.py` + `test_phase5_flash_attn_prefill_decode.py` | 3 |
| Phase 6 MLP/Decoder Layer | `test_phase6_mlp_forward.py` + `test_phase6_residual_chain.py` + `test_phase6_decode_forward_no_clone.py` + `test_phase6_layer_e2e_random_weights.py` | 4 |
| Phase 7 权重加载 | `test_phase7_qwen_tp_config.py` + `test_phase7_hf_key_mapping.py` + `test_phase7_weight_loading.sh` | 3 |
| Phase 8 框架外壳 | `test_phase8_sequence_scheduler.py` + `test_phase8_sampler_tp.py` | 2 |
| Phase 9 引擎集成 | `test_phase9_llm_engine_init.py` + `test_phase9_generate_single_gpu.sh` | 2 |
| Phase 10 E2E 验收 | `test_phase10_greedy_align.sh` + `test_phase10_benchmark.sh` + `test_phase10_no_compile_check.sh` + `test_phase10_vs_vllm_compare.sh` | 4 |
| Phase 11 性能优化 | `test_phase11_throughput.py` + `test_phase11_profiler.sh` | 2 |
| **总计** | **28 个脚本（19 .py + 9 .sh）** | **28** |

**逐 Phase 交付流程（不可跳过）**：

```
Phase N 实现代码写完
  → 运行 Phase N 的所有 scripts/test_phaseN_* 脚本
    → 任一 FAIL → 根据 FM-XXX/KERNEL-XXX 错误码修复实现 → 重跑
    → 全部 PASS → Phase N 交付完成 → 进入 Phase N+1
```

**严禁行为**：
- 跳过某 Phase 的脚本检查直接写下一 Phase 代码
- 修改 scripts/ 下的脚本来"通过"测试
- 用 Agent 自写的 tests/ 替代 scripts/ 的检查

### 2.0.1 Phase → 完整知识链路（JSON + ref_docs + ref_code）

每个 Phase 实现前，Agent 必须按以下 **3 步知识链路** 获取信息。仅读 JSON 是不够的——JSON 中的 `ref_docs`/`ref_code`/`ref_source` 字段指向的外部文件和源码行号包含了蓝图 pseudocode 未涵盖的完整细节（见 §0.-1 强制交叉验证规则）。

**核心规则**：Agent 读 JSON 路径 → 发现 `ref_docs`/`ref_code`/`source_refs` 字段 → **必须实际打开这些文件**，不能仅凭 JSON 内的 pseudocode 脑补。

| Phase | 1. 先读 Blueprint JSON 路径（含内置伪代码和维度） | 2. 再打开 ref_docs（notebooks 知识文档） | 3. 再查阅 ref_code（可执行源码，含 nano-vllm 参考和 vLLM kernel 行号） |
|-------|--------------------------|---------------------------|-------------------|
| **Phase 1** 数值基元 | `qwen3_kernel_contracts`（7 kernel 签名+import 路径）→ `global_primitives_constraints.rmsnorm_precision_law` | `kernel_replacement_plan.md` §九（完整 kernel 调用契约表+代码模板 Snippet A-F） |
| **Phase 2** TP 通信 | `tp_distributed_runtime`（init 顺序+3 种 collective）→ `...collectives.all_reduce_sum`（custom_op 注册+fake）→ `...collectives.all_gather_last_dim`→ `...collectives.custom_ar_all_reduce`（两套 IPC buffer+init_state_machine） | — |
| **Phase 3** TP 线性层 | `tp_linear_layers`（4 种 Linear shape+forward pseudocode）→ `qwen3_8b_model_dims`（per-rank 维度：`gate_up=[6144,4096]` **非** `[6400,4096]`） | `qwen_dense_tp_implementation_guide.md`（TP 切分策略）`task10_tp_qwen_debug_experience.md`（double_shard 踩坑） |
| **Phase 4** TP Embedding | `tp_embedding_and_lm_head`（VocabParallel mask+pseudocode + ParallelLMHead gather） | — |
| **Phase 5** Attention/KV | `qwen3_tp_model_interfaces.attention`（KV cache paged 格式+block_table+slot_mapping）→ `...class_hierarchy.QwenAttentionTP`（__init__ 每个 attr 的精确名称）→ `paged_kv_cache_contract`（prefill/decode KV write + index_copy_）→ `flash_attention_integration_contract`（prefill_path + decode_path kernel 调用） | `improvement_plan.md` §P0（**⚠️ FG-3：此 doc 描述旧 HF 方案，与 paged KV 架构矛盾，以蓝图为权威**）`improvement_plan.md` §P3-FA（flash_attn 集成）`qwen3_effective_changes.md` #8 #10 |
| **Phase 6** MLP/Decoder | `qwen3_tp_model_interfaces.mlp`（gate_up→silu_and_mul→down 链）→ `...decode_forward_pattern`（**完整 forward_decode 方法体 pseudocode，直接抄入**）→ `...prefill_forward_pattern`（prefill 完整数据流 8 步）→ `...class_hierarchy.QwenMLPTP` + `...QwenDecoderLayerTP`（__init__ attr 名） | `kernel_replacement_plan.md` §三（Snippet C: silu_and_mul 代码模板）`qwen3_effective_changes.md` #9（torch.compile 相关，nocompile 忽略） |
| **Phase 7** 权重加载 | `...class_hierarchy.QwenTPConfig`（config.json 字段→dataclass）→ `...QwenForCausalLMTP`（构造链：embed→layers→norm→lm_head）→ `...construction_chain`（5 步：config→QwenTPConfig→model→load_weights→eval→CustomAR）→ `qwen_hf_key_mapping`（12 个 HF key→attr 映射+Q-K-V cat 顺序+Gate-Up cat 顺序+double_shard_guard）→ `qwen3_8b_model_dims`（验证维度） | `kernel_replacement_plan.md` §四（QKVColumnParallelLinear 组装）`task10_tp_qwen_debug_experience.md`（load_weights shape mismatch debug） |
| **Phase 8** 框架外壳 | `components[0] Scheduler`（完整 schedule+postprocess pseudocode）→ `components[4] Sampler`（TP 协议：rank0 采样+broadcast）→ `components[5] Sequence`（status 转移+block_table 双轨）→ `components[2] BlockManager`（⚠️ TP 降级 no-op：allocate/free 空方法+get_num_free_blocks 保留）→ `scheduler_to_runner`（batch 组装+can_allocate 公式+REJECTED 机制） | `02_scheduler.md`（调度概念）`03_kv_cache.md`（paged attention）`05_sampler.md`（采样算法）`01_architecture.md` `07_request_lifecycle.md` | — |
| **Phase 9** 引擎集成 | `components[6] LLMEngine`（full_api_surface: __init__ 7 步 flow+generate 5 步 while-loop+step 单步+begin_generation/has_unfinished/get_outputs API）→ `components[3] ModelRunner`（tp_runner_actual_flow: prefill/decode 分发+采样+run 方法体）→ `scheduler_tp_runner_bridge`（CRITICAL-01：block_size 注入+num_free 来源路由+BlockManager 降级） | `01_architecture.md` `07_request_lifecycle.md`（框架生命周期） |
| **Phase 10** E2E 验收 | `todo_generation_playbook.phase_10_e2e_acceptance`（implementation_todos+minimal_test_commands+e2e_acceptance_bottom_line 硬性指标）→ `runtime_acceptance_layer.logic_constraints`→ `...acceptance_test_targets.qwen3_tp.performance_baseline` | `CLAUDE.md` §6.2-6.4（测试命令+benchmark 脚本）`improvement_plan.md`（性能基线记录） |
| **Phase 11** 性能优化 | `todo_generation_playbook.phase_11_performance`（O1-O9 分层规则：O1 @torch.inference_mode(), O2 全文件零 .item(), O3 预分配 buffer, O4 block_table arange, O5 prefill KV 直接赋值, O6 register_buffer 完整声明, O7-O9 懒 contiguous/view/消除中间 tensor）→ `performance_gate`（throughput > 50 tok/s, cudaMalloc=0, aten::item < 10ms） | `notebooks-cn/07_improvementPlan/ROUND_1_BOTTLENECK_FIXES.md`（9 项优化实测验证，19.3→59.1 tok/s） |

**JSON 路径缩写展开**：
- `qwen3_kernel_contracts` = `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts`
- `tp_distributed_runtime` = `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_distributed_runtime`
- `tp_linear_layers` = `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_linear_layers`
- `tp_embedding_and_lm_head` = `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_embedding_and_lm_head`
- `qwen3_tp_model_interfaces` = `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces`
- `paged_kv_cache_contract` = `framework_layer.data_flow_contracts.paged_kv_cache_contract`
- `flash_attention_integration_contract` = `framework_layer.data_flow_contracts.flash_attention_integration_contract`
- `scheduler_to_runner` = `framework_layer.data_flow_contracts.scheduler_to_runner`
- `scheduler_tp_runner_bridge` = `framework_layer.data_flow_contracts.scheduler_tp_runner_bridge`
- `components[N]` = `framework_layer.components[N]`
- `qwen3_8b_model_dims` = `model_layer.architecture_knowledge_base.qwen_series_dense.qwen3_8b_model_dims`
- `qwen_hf_key_mapping` = `model_layer.architecture_knowledge_base.qwen_series_dense.qwen_hf_key_mapping`
- `global_primitives_constraints` = `model_layer.architecture_knowledge_base.global_primitives_constraints`

**ref_code 读取规则**：
1. `ref_projects/nano-vllm/...` 路径在项目根目录下，**必须实际 `Read` 打开该文件**。蓝图 component 的 `_nano_vllm_override` 字段说明了哪些行需要修改/删除。
2. `vllm/_custom_ops.py:420-423` 等路径指向 **vLLM installed package**（在用户 Python 环境的 site-packages 中）。Agent 需 `import vllm._custom_ops; print(inspect.getsource(...))` 或直接读已安装包的源码文件。
3. ref_code 中有精确到方法名和行号的引用（如 `vllm/model_executor/layers/activation.py::SiluAndMul.forward_cuda`）——Agent 必须只读该方法，不是整个文件。

### 2.1 引擎代码结构与 Qwen3 入口

实际工程目录（本仓库根目录）关键文件：

| 文件 | 作用 | 核心类 | 所属 Phase |
|------|------|--------|-----------|
| `llm_engine.py` | LLMEngine（Scheduler+Runner胶水层）、RealModelRunner（HF兜底） | LLMEngine, RealModelRunner | Phase 9 |
| `openai_tp_server.py` | OpenAI 兼容 HTTP API（/v1/completions），TP 多卡 broadcast 同步，benchmark 入口 | run_tp_generation_loop | Phase 10 |
| `engine/tp_layers/distributed.py` | TP 通信 + CustomAR | init_tp_distributed, all_reduce_sum, all_gather_last_dim, init_custom_ar | Phase 2 |
| `engine/tp_layers/linear.py` | TP 线性层 | QKVColumnParallelLinear, MergedColumnParallelLinear, ColumnParallelLinear, RowParallelLinear | Phase 3 |
| `engine/tp_layers/embedding.py` | TP Embedding/LM Head | VocabParallelEmbedding, ParallelLMHead | Phase 4 |
| `engine/models/qwen.py` | Qwen3 TP 全链路 | QwenTPModelRunner, QwenForCausalLMTP, QwenDecoderLayerTP, QwenAttentionTP, QwenMLPTP | Phase 5-7 |

**类层级与属性命名（权重加载关键）**：
- QwenAttentionTP: `.qkv_proj` (非 `.q_proj`), `.o_proj`, `.q_norm`, `.k_norm`, `.num_heads`, `.num_kv_heads`, `._key_cache`, `._value_cache`, `._block_table`, `._kv_len_gpu`
- QwenMLPTP: `.gate_up_proj` (非 `.gate_proj`), `.down_proj`, `.local_intermediate`
- QwenDecoderLayerTP: `.self_attn` (非 `.attention`), `.mlp`, `.input_layernorm`, `.post_attention_layernorm`
- QwenForCausalLMTP: `.embed_tokens`, `.layers` (ModuleList), `.norm`, `.lm_head`
- HF key 映射见 `inference_blueprint.json > qwen_hf_key_mapping`
- **NCCL 初始化**: torchrun → `torch.cuda.set_device(LOCAL_RANK)` → `dist.init_process_group('nccl','env://')` → 模型加载 → `init_custom_ar()` → `dist.barrier()`
- **fused_add_rms_norm**: 物理tracing确认 — 所有4次调用用本层 `self.input_layernorm.weight` 或 `self.post_attention_layernorm.weight`。无跨层weight。
- **all_gather_last_dim**: `dist.all_gather(outs, x)` + `torch.cat(outs, dim=-1)`
- **all_reduce_sum**: @torch.library.custom_op。TP=1 → x.clone()。CustomAR优先→NCCL fallback。**⚠️ init_custom_ar 整个函数必须 try/except——open_mem_handle 在不同 CUDA 版本/容器环境下可能抛 "Cannot access data pointer of Tensor that doesn't have storage"。失败时 _custom_ar_handle=None，all_reduce_sum 自动走 dist.all_reduce(NCCL)。严禁 init 崩溃导致 TP 推理直接挂。**
- **VocabParallelEmbedding**: mask+masked_fill for out-of-range tokens, all_reduce_sum at end
- **KV head replication**: tp>num_kv_heads 时 num_kv_heads=1, kv_head_replica=tp//num_kv_heads
- **Config**: AutoConfig.from_pretrained → QwenTPConfig。head_dim回退: hidden_size//num_attention_heads

**nano-vllm 重载要点**（抄入参考代码时的修改检查清单）：
1. Scheduler: 删除 preempt() 逻辑。block_size 改为注入式。
2. BlockManager: TP 路径降级为纯计数器（allocate/free no-op）。get_num_free_blocks 保留但调用方改用 runner.get_num_free_blocks()。
3. KVMemoryPool: TP 路径禁用 GPU placeholder 创建。
4. ModelRunner: 删除 use_cache=False 模式。替换为 forward_decode 调用。
5. Sampler: TP 下 rank 0 采样 + broadcast（不修改 sampler.py 本身）。
6. RMSNorm: nano-vllm 原始实现需整体替换为 vLLM kernel wrapper。不能混用。

### 2.2 Qwen3 TP 关键数据流（物理 tracing 确认）

- **Config 组装链**: `config.json → QwenTPConfig(全量值字段) → QwenForCausalLMTP(cfg, device, dtype) → load_weights() → eval()`。cfg 的 num_attention_heads=32 (全量)，per-rank 计算在 __init__ 内 `cfg.xxx // tp_size`
- **max_blocks 注入**: LLMEngine.__init__ 中 `scheduler._max_blocks = config.max_position_embeddings // 256` (40960//256=160, 2026-05-27 物理 config.json 确认)
- flash_attn 直接 `from flash_attn.flash_attn_interface import flash_attn_with_kvcache`（nocompile 无需 custom_op 注册）
- **引擎主循环**: enqueue → schedule(num_free) → run(batch, is_prefill) → postprocess(batch, is_prefill, tokens) → check all_finished
- **Scheduler**: schedule(num_free) 先 prefill 后 decode，含 REJECTED 过滤和 token budget 检查。postprocess 负责 output_ids 追加 + EOS/max_tokens 检测 + FINISHED 状态转移 + 资源释放
- **slot_mapping**: 长 prompt 用 `block_table[0, indices//256] * 256 + (indices%256)` 向量化
- CustomAR 初始化: meta_ptrs(元数据+staging buffer) 和 buf_ptrs(纯 staging buffer) 两套独立 IPC buffer 集合。**两套都用同一个 `_allocate_and_exchange_handles` 函数，内部使用 `dist.all_gather_object` 交换 handles——不用 `broadcast_object_list`。** `broadcast_object_list` 仅在 CUDA Graph 路径的 `register_graph_buffers()` 中使用（nocompile 无需）。
- prefill flash_attn_varlen_func K/V 必须来自投影产出 (非 cache)

**Prefill/Decode 分发逻辑**：
- `is_prefill = (past_key_values is None)` — None → prefill，非 None → decode
- prefill: `layer.forward()` → qkv_proj → rotary → flash_attn_varlen_func(causal=True) → K,V 直接索引赋值写入 paged cache → o_proj → mlp
- decode: `layer.forward_decode()` → qkv_proj → Q/K norm（预分配 buffer 直接调 rms_norm kernel）→ rotary → KV write(index_copy_) → flash_attn_with_kvcache → o_proj
- KV cache paged 格式：`_key_cache[num_blocks, 256, num_kv_heads, head_dim]`，block_size=256
- **KV cache lazy alloc（⭐⭐⭐ 显存关键）**：`num_blocks` 在 prefill 时按需分配 `(num_tokens+255)//256`，**严禁**一次性 `torch.zeros(max_blocks=160, 256, kv_heads, dim)` 全量预分配。后者导致每层 160×256×2×128×2 = 20MB，36 层 ≈ 720MB 额外显存浪费。真实 trace 确认：4-token prompt 只需 1 block。每次 decode 追加 1 token 不增加 block 数（复用已分配 block）。
- `_kv_len_gpu` 为 GPU tensor。**严禁 `.item()`** — kv_len 追踪全部走 CPU 算术（`runner.run()` 中 `s.kv_len += 1`）
- **get_num_free_blocks**: 返回常量 `cfg.max_position_embeddings // 256`。短序列 <256 token 仅需 1 block，无需 GPU 同步动态计算
- **QKV reshape**: Q 用 `num_heads=8` (per-rank)，K/V 用 `num_kv_heads_local=2` (per-rank)。勿混用。

### 2.2.1 V17 审计驱动的先验警告（实现前必读）

以下 4 条警告来自 2026-05-27 的 V17 隔离重构审计。Agent 在实现前必须理解这些陷阱：

1. **维度参数陷阱（FG-1）**：蓝图旧版示例值 `max_position_embeddings=32768`、`intermediate_size=12800` 已被物理 config.json 纠正为 `40960` 和 `12288`。这些是**示例值**，严禁硬编码。必须 `json.load(open('config.json'))` 动态读取。`max_blocks=160`（非 128）；`gate_up_weight=[6144,4096]`（非 `[6400,4096]`）。
2. **ref_doc 矛盾（FG-3）**：`improvement_plan.md §P0` 描述的是**旧版 HF past_key_values 方案**（contiguous KV buffer + use_cache=True），与蓝图 paged KV cache + flash_attn_with_kvcache 方案架构不兼容。以蓝图为权威，§P0 仅作历史参考。实现时禁止参考 §P0 做 KV cache 设计。
3. **Dense KV 预算缺失（FG-2）**：`KVMemoryPool.estimate_num_blocks` 仅实现 MLA 公式。Qwen3 Dense 模型需独立实现 K+V 公式（见 blueprint `_estimate_kv_blocks.dense_pseudocode`）。不可直接调用 `estimate_num_blocks`。
4. **BlockManager TP 降级（OW-2）**：通过 `BlockManager(tp_mode=True)` 构造参数实现。allocate/free 在 tp_mode 下为 no-op。不能通过继承或猴子补丁实现——使用类内 `if self._tp_mode` 条件分支。

### 2.3 LLMEngine API 表面（Phase 9 构建目标）

`llm_engine.py` 是框架入口，向上对接 Scheduler（调度），向下对接 ModelRunner（模型前向），对外暴露两套接口：

**单次生成接口**：
- `generate(prompt, max_new_tokens, temperature, top_p)` → str：enqueue → while-loop(schedule→run→postprocess→finish_check) → decode → return

**分步接口**（供 OpenAI server 使用）：
- `begin_generation(prompts, max_new_tokens, temperature, top_p)`：批量入队
- `has_unfinished_requests()` → bool：检查 scheduler 队列
- `step(temperature, top_p)` → list[Sequence]：推进一步
- `get_generation_outputs()` → list[str]：获取 decode 文本

**核心内部方法**：
- `__init__`：路由 backend → 创建 Runner → 估算 KV 池 → 初始化 Scheduler（详见 blueprint full_api_surface）
- `_select_tp_backend(model_dir)`：读 config.json → 匹配 Qwen/DeepSeek
- `_estimate_kv_blocks()`：torch.cuda.mem_get_info → Dense/MLA公式 → block 数量
- `_finish_check_and_cleanup(seq)`：EOS/max_tokens 检测 → FINISHED → 释放资源

### 2.4 OpenAI TP Server 架构（Phase 10 构建目标）

`openai_tp_server.py` 是 benchmark 的 HTTP 接口，封装 LLMEngine 为 OpenAI 兼容 API：

**TP 同步机制**（最关键的实现细节）：
```
Rank 0:  接收 HTTP → broadcast_obj({action,...}) → engine.generate() → 返回响应
Rank 1-3: while True: cmd=broadcast_obj({}) → 相同 engine.generate()
```
- `broadcast_obj`：rank0 发送 payload，所有 rank 返回相同结果
- `request_lock`（threading.Lock）：**序列化所有请求**。并行请求导致跨 rank NCCL collective 顺序不一致 → 死锁
- `shutdown` action：rank0 发 `{"action":"shutdown"}` → non-rank0 退出循环

**端点**：
- `GET /health` → `{"status":"ok"}`
- `POST /v1/completions`：支持 stream=true(SSE) 和 stream=false(JSON)

**SSE 连接生命周期（2026-06-06 线上 hang 根因）**：
- SSE 响应无 Content-Length，客户端依赖连接关闭判定流结束。BaseHTTPRequestHandler 默认 keep-alive。
- **必须**：`Connection: close` header + `self.close_connection = True`（成功路径和 except/finally 路径都要设）
- 缺少 close_connection → 连接永不关闭 → benchmark warmup 请求持有 engine_lock → 0/N 请求成功

**Non-rank0 Worker 信号处理（2026-06-06 进程残留根因）**：
- Worker 主线程阻塞在 `dist.broadcast_object_list`（C 调用），Python 信号处理延迟到 C 返回后
- **必须**：注册 SIGTERM + SIGINT handler，handler 内调用 `os._exit(0)`（不经过 Python 解释器，直接终止进程）。注意 handler 必须接受 `(signum, frame)` 两个参数
- 缺少 handler → torchrun kill 后子进程残留，每进程占 ~6.5GB GPU 显存

**Benchmark 脚本清理要点**：
- `pkill -9 -f "openai_tp_server.py"` 匹配 python 子进程，不要用 `pkill -f "torchrun.*openai_tp_server"`
- 使用 `trap cleanup EXIT INT TERM` 确保一切退出路径都清理

**启动方式**：
```bash
TP_SIZE=4 PORT=9000 bash start_tp_infer_service.sh qwen   # 终端1
PORT=9000 NUM_PROMPTS=50 REQUEST_RATE=1 bash run_myengine_benchmark.sh qwen  # 终端2 压测
```

---

## 6. 标准工作流（Workflow）

### 6.1 测试即规范（Tests as Specification）

**铁律：`scripts/` 目录下的文件是先验知识，由人类编写并 commit。开发 Agent 只读不写，绝对禁止任何修改。**

scripts/ 目录是知识体系的第一层（与 inference_blueprint.json、AGENT_SKILL.md 同等地位）。每个测试脚本在 git 中经过以下流程：

```
人类编写 → code-review 子代理 10 条审查 → 对抗压力测试（RED+GREEN） → 人类 Diff 确认 → commit
```

**RED-GREEN 循环（由 superpowers TDD skill 驱动）**：

1. **RED — 开发前跑脚本，确认失败**：
   在写任何实现代码前，先执行对应 Phase 的 scripts/ 测试：
   ```bash
   python scripts/test_phaseN_xxx.py        # Python 合约
   bash scripts/test_phaseN_xxx.sh          # Shell E2E/环境
   ```
   预期：全部 FAIL（因为代码还没写）。如果某脚本 PASS：报告给人类，脚本可能失效。

2. **GREEN — 写最小实现，跑通脚本**：
   根据 `inference_blueprint.json` 和 scripts/ 中的精确断言写实现代码。
   - 如果测试失败 → **阅读 assert 错误信息（含 FM-XXX/KERNEL-XXX/LINEAR-XXX 编号）→ 修改实现代码 → 重跑**
   - **绝对不能修改 scripts/ 下的文件来"让测试通过"**

3. **REFACTOR — 保持全绿**：
   整理代码结构、消除重复。每次改完后立即跑对应 Phase 的 scripts/，确保不引入回归。

**Phase 通关标志**：该 Phase 下**所有** scripts/ 测试脚本全部 PASS（非 Agent 自写测试）。具体每个 Phase 对应哪些脚本见 **§2.0 Phase-脚本硬绑定表**（共 26 个，18 .py + 8 .sh）。

**审查子代理触发时机**：每 Phase 完成后自动启动 code-review 子代理，跑该 Phase 全部 scripts/ 测试 + 检查实现代码 diff。

**额外测试（非强制）**：Agent 可以在 `tests/`（非 scripts/）下补充测试，但 tests/ 不能替代 scripts/。

**对抗压力测试记录**：scripts/ 下每个脚本必须经过 adversarial pressure testing——故意写错代码确认脚本捕获（RED），恢复正确代码确认通过（GREEN）。测试记录写入脚本文件的 git commit message。

### 6.2 端到端门禁

1. 真实 TP 测试必须由自研引擎驱动，而不是仅 HF-debug 旁路。
2. 回归需要覆盖：
   - temperature=0.0 贪婪解码字字对齐
   - 多 prompt 中文可读输出
   - 多次复跑稳定性
   - 不同 rank 下一致性
3. **纯 Eager 模式**：一次任务必须完整覆盖 `phase_1_numeric_primitives` 到 `phase_10_e2e_acceptance`，不得只完成前两阶段就结束。
4. `phase_10` 验收除文本正确性外，必须附带：
   - 监控证据：4 卡 VRAM% 同量级 + HCU% 出现大于 0
   - 单层 CPU dispatch profiling（目标 < 15ms/layer）
   - Profiler 确认无 `torch.compile` / `CUDA Graph` 痕迹
   - **严禁** fake inference（硬编码输出冒充真实推理）

---

## 7. 自动化排错与自愈机制（Auto-Debugging & Self-Healing Loop）

### 7.1 错误降级定位

当 E2E（例如 `test_deepseek_tp_real.py`）出现乱码或失败：

1. 立即回退到单层/子模块 Unit Test。
2. 开启 `DEBUG_MODE` 探针与 NaN/Inf 断言。
3. 定位首个数值崩塌层（shape、scale、dtype、通信语义）。

### 7.2 查阅踩坑知识库

定位到故障算子后，强制回查 `inference_blueprint.json`：

1. `global_primitives_constraints`
2. 对应模型族知识（Qwen Dense / DeepSeek MLA+MoE）
3. 失败模式库（双重切片、RoPE 风格错配、YaRN 漏补丁等）

### 7.3 闭环修正

1. 按知识库修复算子。
2. 重新通过对应 Unit Test。
3. 再跑 E2E 对齐测试。
4. 更新经验记录，沉淀为可复用规则。

---

## 7.4 通用 Debug 指南（从 preVer 回灌）

### A. Shape mismatch

1. 回看 `inference_blueprint.json > framework_layer.data_flow_contracts`。
2. 对照真实模型配置：`config.json` 的 `qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`num_attention_heads`、`num_hidden_layers`。
3. 在关键节点打印：
   - prefill/decode `input_ids` shape
   - `logits` shape
   - 采样输入 shape
   - `block_table` 长度与 `required_blocks`

### B. OOM

1. 检查 `KVMemoryPool.estimate_num_blocks` 的输入参数：
   - `free_bytes`
   - `reserve_bytes`
   - `mem_utilization`
   - `block_size`
2. 必要时按顺序降压：
   - 降低 `mem_utilization`
   - 增大 `reserve_bytes`
   - 增大 `block_size`（减少块元数据开销）
3. 不要盲目增大 batch 或 max tokens。

### C. 输出异常/乱码

1. 检查 tokenizer 与 `skip_special_tokens` 使用。
2. 检查采样参数（`temperature`、`top_p`）。
3. 核验 prompt 编码和 decode 文本链路是否一致。

### D. 卡住或多次无法修复

1. 重新阅读：
   - `notebooks/MEMORY.md`
   - `notebooks/00_overview/README.md`
   - `inference_blueprint.json` 中目标组件的 `ref_docs` / `ref_code`
2. 缩小问题规模到单组件最小可复现测试。
3. 查阅 `inference_blueprint.json > model_layer.failure_mode_library` 的 symptom→check→fix 词条。

### E. CUDA Graph 崩溃与显存指针漂移

1. **Dynamo RNG 重编译（TP=4）**：`CUDAGeneratorImpl::current_seed` 在 graph 捕获时报错 → `torch.compile(fullgraph=True)` 追踪进入了 `all_reduce_sum` → sglang 切图方案修复（`torch.fx.split_module`，通信子图 `backend='eager'`）。
2. **cuBLAS 地址漂移**：raw CUDA Graph replay 后 hs 差 3.4 → cuBLAS workspace 地址在捕获/回放间不一致 → 通信 op 必须从 CUDA Graph 分离（eager 执行）。
3. **mutated inputs → cudagraphs 跳过**：`skipping cudagraphs due to mutated inputs` → `fused_add_rms_norm` 原地修改输入 → `forward_decode_graph` 开头 clone 输入修复。
4. **compiled region .item()**：SIGABRT → `.item()` 必须在 `forward()`（非编译）中，不能在 `forward_decode`（编译）中。

### F. 性能回退诊断

1. **无条件 clone 回退**：nocompile 吞吐 < 50 tok/s → 检查 `aten::copy_` 占比 >5% → 拆分 forward_decode/forward_decode_graph。
2. **通信回退**：TP=4 wall > 300ms → 检查 CustomAR 是否初始化（`init_custom_ar` 在 `load_weights` 后调用）。
3. **GEMM dispatch 回退**：`aten::linear/mm` CPU total > 200ms → CUDA Graph 未启用。

### G. 集成接口陷阱（V4 审计新增）

1. **Scheduler block_size 双轨**：TP Runner 路径必须 `scheduler._block_size = 256`（非默认的 16）。LLMEngine.__init__ 中根据 `inference_backend` 注入。
2. **num_free_blocks 来源**：TP 路径调用 `runner.get_num_free_blocks()`（从 `_kv_len_gpu` 推算），HF 路径调用 `BlockManager.get_num_free_blocks()`。
3. **BlockManager 降级**：TP 路径下 `allocate()/free()` 为 no-op，仅做容量计数。实际 block_table 由 `QwenAttentionTP torch.arange` 分配。
4. **CustomAR 初始化**：所有 rank `load_weights` 后必须 `dist.barrier()` 再调用 `init_custom_ar()`。`dist.all_gather_object` 是阻塞操作，如不同步会死锁。
5. **bound method**：`torch.compile(layer.forward_decode_graph)` 返回函数对象，不是绑定方法。必须 `types.MethodType(compiled, layer)` 包装，否则 `self` 缺失。
6. **fused_add_rms_norm 全部使用 self weight**：所有 4 次调用 (forward_decode 中) 均为 `self.input_layernorm.weight` 或 `self.post_attention_layernorm.weight`。无反直觉的跨层传递。
7. **KV reshape 链**：index_copy_ 前需将 `_key_cache` view 为 `[-1, heads, dim]`，K 需 reshape 为 `[num_tokens, heads, dim]` 确保 contiguous。
8. **TP 采样协议**：多卡推理仅 rank 0 执行采样，`dist.broadcast` 给所有 rank。严禁各 rank 独立采样（CUDA 随机种子不同 → token 不一致 → KV cache 不同步 → NCCL 崩溃）。
9. **超长 prompt 拒绝**：enqueue 时检测 `required_blocks > max_blocks` → 拒绝或截断。新增 `status='REJECTED'` 防止永久 WAITING 死循环。
10. **Prefill 失败回滚**：RuntimeError 时 `memory_pool.free_sequence(seq)` 释放已分配 KV blocks，block_table 置空，异常向上传播。
11. **Prefill K/V 来源**：flash_attn_varlen_func 的 K/V 来自当前投影产出，非从 KV cache 读取。顺序：投影 → attention → 写入 cache。
12. **多序列 prefill**：slot_mapping 需跨序列拼接 block_table。见 `paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm.multi_seq`。

### H. 纯 Eager 专有反模式（nocompile 禁止项检查）

1. **条件 clone 缺失**：eager 路径 `forward_decode` 不应含 `clone()`（clone 仅用于 CUDA Graph 路径的 `forward_decode_graph`）。检查 `aten::copy_` 占比 > 5% → 有冗余 clone。
2. **`.item()` GPU sync**：`forward_decode` 内不应有 `.item()`。所有 `.item()` 必须在非编译的 `forward()` 外层 batch 读取。症状：每步 0.1ms+ CPU sync 延迟 → profiler `cudaSynchronize`。
3. **block_table 循环 item()**：slot_mapping 的 `for i in range(num_tokens): block_table[0, i//256].item() * 256 + (i%256)` — 短 prompt (<512 tokens) 可用，长 prompt (32K) 必须向量化 `block_table[0, indices // 256] * 256 + (indices % 256)`。检测：prefill > 50ms Python overhead → 检查 slot_mapping 是否有 for 循环。
4. **KV cache 重复分配**：每步重新 `torch.zeros` 分配 KV cache（而非 prefill 时一次分配 `max_blocks` 复用）→ 每步 1ms+ GPU allocator 开销。
5. **残余 compile 痕迹**：代码中残存 `torch.compile`、`mode='reduce-overhead'`、`fullgraph=True` → profiler 中出现 `CompiledFunction`、`Inductor` kernel → 应立即删除并替换为纯 eager 路径。
6. **fake inference**：严禁 `print("假输出")` 冒充真实 TF 推理。验收必须附带 profiler trace 文件 + HCU 监控证据。

### I. 性能强制模式（Phase 11 — 审计门禁 + 补充优化）

以下规则经过 ROUND_1 实测验证（`notebooks-cn/07_improvementPlan/ROUND_1_BOTTLENECK_FIXES.md`，19.3→59.1 tok/s，+206%）。
分两层：**O1-O6 为审计门禁**（Phase 11 必须逐条静态检查，命中则回对应 Phase 修复），
**O7-O9 为补充优化**（正确但非致命，可选实施）。

**Phase 11 强制审计闭环（不可跳过）：**

1. **STEP-AUDIT**: 逐条执行下方 O1-O6 的审计检查命令，记录 PASS/FAIL。O7-O9 仅记录不阻塞。
2. **STEP-FIX**: 每条 FAIL 项目定位到对应 Phase 构建的源码文件，修改代码使审计通过。
3. **STEP-REAUDIT**: 修复后重新跑全部审计检查，直到 O1-O6 全部 PASS。
4. **STEP-BENCHMARK**: 全部审计通过后跑 `test_phase11_throughput.py` 验证吞吐达标（> 50 tok/s）。
5. **STEP-DONE**: O1-O6 全部 PASS + 吞吐达标 → Phase 11 完成。

**O1 @torch.inference_mode()（CRITICAL — 占提升的 82%）**

QwenForCausalLMTP.forward() 和 forward_decode() **必须**带 `@torch.inference_mode()` 装饰器。
禁用 autograd version counter 递增和元数据追踪，消除 GeneratedBackwardFor 膨胀 + cudaLaunchKernel CPU 开销。
CPU launch 占比从 ~35% 降至 <5%。

审计检查：`grep -r '@torch.inference_mode' engine/models/qwen.py` 应有 2 个匹配。

**O2 零 .item() GPU 同步（CRITICAL）**

- `get_num_free_blocks()` 返回常量 `cfg.max_position_embeddings // 256`（短序列场景 <256 token 仅需 1 block）
- `runner.run()` decode 分支用 `s.kv_len += 1`（CPU 算术，forward_decode 内部已自增 _kv_len_gpu）
- prefill slot_mapping 用 `torch.arange` 向量化（禁止 for 循环 + .item()）

审计检查：`grep '\.item()' llm_engine.py engine/models/qwen.py` 应为零匹配（仅注释中允许）。QwenForCausalLMTP.forward() decode 分支已改用 `past_key_values[0] + 1` CPU 算术。

**O3 预分配 buffer（CRITICAL）**

decode 路径（B=1,S=1）使用 `register_buffer(persistent=False)` 预分配的 buffer，直接调底层 kernel：
- `_q_norm_out[1, heads, dim]`, `_k_norm_out[1, kv_heads, dim]` — attention Q/K norm 复用
- `_silu_out[1, 1, local_inter]` — MLP silu_and_mul 输出复用

绕过 `nn.Module.forward()` 中的 `torch.empty_like()`（内部触发 `cudaDeviceGetAttribute`）。

通用路径（B>1 或 S>1）禁止 `torch.empty_like()`：`torch.empty_like(gate_up[..., :half_ch])` → `torch.empty(x.shape[0], x.shape[1], half_ch, dtype=x.dtype, device=x.device)`。`empty_like` 含隐式 CUDA runtime 查询，改为显式参数消除。

审计检查：`grep '_q_norm_out\\|_k_norm_out\\|_silu_out' engine/models/qwen.py` 应有匹配；`grep 'empty_like' engine/models/qwen.py` 应为零匹配。

**O4 block_table arange 初始化（HIGH）**

KV cache 首次分配时一次性 `torch.arange(max_blocks).unsqueeze(0)`，非 `torch.zeros + 每步填充`。
arange 创建恒等映射（逻辑页→物理页），全零是错误的（所有页映射到 block 0）。

审计检查：`grep '_block_table.*arange' engine/models/qwen.py`。

**O5 prefill KV 直接赋值（HIGH）**

prefill KV 写入使用直接索引赋值 `kc_flat[slot_mapping] = k_flat`，非 `index_copy_`。
prefill 不在 CUDA graph 内，直接赋值比 index_copy_ 快。

审计检查：`grep 'index_copy_' engine/models/qwen.py` 应仅在 decode 路径出现。

**O6 register_buffer 完整声明（HIGH）**

所有不变 tensor 用 `register_buffer(persistent=False, dtype=...)` 注册：
`_kv_len_gpu`, `_slot_mapping_decode`, `_q_norm_out`, `_k_norm_out`, `_silu_out`, cos_sin cache。

审计检查：`grep 'register_buffer' engine/models/qwen.py | wc -l` 应 ≥ 6。

**O7 懒 contiguous（LOW — 正确补充）**

仅在内核输入前调用 `.contiguous()`，不在中间 view/reshape 后调。

**O8 view 非 reshape（LOW — 正确补充）**

`view()` 零拷贝，`reshape()` 可能触发隐式 `.contiguous()`。仅当 tensor 确实 discontiguous 时才用 `reshape()`。

**O9 消除中间 tensor（LOW — O3 是其具体体现）**

one-pass 计算，不分配中间临时 tensor。预分配 buffer（O3）是实现此原则的关键手段。

---

## 8. 完成定义（Definition of Done）

仅当以下条件全部满足，任务才算完成：

1. 目标模型通过架构路由并正确实例化 TP Runner。
2. Lazy loader 满足该模型全部切分规则（含 replicated/TP/EP）。
3. 子模块单测全部通过（含 NaN/Inf 与 shape 校验）。
4. `test_xxx_tp_real.py` 在 torchrun 下输出稳定、可读文本。
5. 提供变更摘要：文件、规则映射、测试命令、风险与后续建议。

### 8.1 纯 Eager 性能基线（Qwen3-8B, TP=4, nocompile, B=1）

| 指标 | 目标值 | 验证方式 |
|------|--------|---------|
| Output Throughput | > 50 tok/s | `run_compare_metainfer_vllm.sh qwen` |
| GPU Self CUDA | ≤ 66ms / step | torch.profiler 单步 GPU kernel 总时间 |
| CustomAR 通信 | ≤ 25ms / step | profiler `cross_device_reduce_1stage` |
| CPU dispatch / layer | < 15ms (36 layer total ≤ 540ms) | torch.profiler CPU time |
| VRAM% per rank | ~7% (同量级一致) | nvidia-smi 或 HCU 监控 |
| HCU% | > 0（真实计算证据） | HCU 监控 |

### 8.2 纯 Eager 专有断言

- Profiler 确认无 `torch.compile` 痕迹（无 `CompiledFunction`、`Inductor` kernel）
- Profiler 确认无 CUDA Graph 痕迹（`cudaGraphLaunch=0`）
- `forward_decode` 不含 `.item()`（禁止编译区内 CPU sync）
- `forward_decode` 不含 `clone()`（eager 路径零 clone 开销）
- `forward()` 中所有 `.item()` 调用在模型层循环完成后（batch 读取）
- 所有 RMSNorm/RoPE/Silu 使用 vLLM 标品 kernel，**禁止手写 PyTorch 逐元素实现**

### 8.3 贪婪解码正确性

- `temperature=0.0` 下 Qwen3-8B 输出字字对齐预期基线
- 首 token logits max diff vs HF < 1e-3（bf16 tolerance）
- 5 次独立复跑输出完全一致
