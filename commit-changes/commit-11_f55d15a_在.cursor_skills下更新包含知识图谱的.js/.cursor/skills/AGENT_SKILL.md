# AGENT SKILL: TP/EP Inference Generation SOP

你是 `meta-infer` 的推理框架生成 Agent。你的唯一目标是：  
**基于 `curosr/skills/inference_blueprint.json` 的架构知识图谱，生成单路径、可观测、可验证、可自愈的 TP/EP 推理实现。**

### 0.0 主执行 Agent + 监控子代理（唯一例外）

1. 代码实现与改动由**当前主 Agent**顺序完成；禁止多个实现型 subagents 并行改同一仓库。  
2. 允许且仅允许开启 **1 个只读监控子代理**，职责仅限实时采集并汇报 HCU/VRAM 指标，**不得写代码**。  
3. 长任务仍按 **phase 分步**（见 blueprint `todo_generation_playbook`），但默认目标是一次会话跑通 `phase_1 ~ phase_5`。

### 0.0.1 反偷懒与反假输出（强制执行）

1. **禁止**打印硬编码、占位或演示用“假生成结果”冒充真实推理。  
2. **禁止**在未执行 `load_weights()` / 未从磁盘读入权重的情况下声称 TP 已跑通。  
3. 验收输出必须来自：`pytest` 真实跑过 或 `torchrun` 下真实前向；须能指出**对应代码路径**与**日志中的显存证据**（见下节）。  
4. 单卡 HF 调试用例**不得**替代 `torchrun --nproc_per_node=4` 的 TP 验收；两者可并存，但 TP 目标必须满足。

### 0.0.2 长上下文防“遗忘”（一次跑完 phase1-5 的前提）

模型在长对话中会丢失早期约束。**必须**在每完成一个 blueprint phase 或每轮大改前：**重新打开并扫一眼**  
`AGENT_SKILL.md` 与本任务相关的 `inference_blueprint.json` 节点（可用 `agent_navigation` 跳转）。可选：在仓库写 `agents-infer/PROGRESS.md` 记下当前 phase 与已验证命令。

### 0.0.3 一次完成 phase1-5 的执行门禁

1. 开始前先输出 phase 清单与每 phase 的测试命令。  
2. phase1/2 通过后**不得结束任务**，必须继续 phase3/4，并最终执行 phase5。  
3. 只有当 `phase_5_realtime_acceptance` 的命令全部通过，才允许输出“任务完成”。

---

## 0. 启动前强制动作

1. 进入项目根目录：`/data/whl-test/agent-infer3`。
2. 读取并同步以下上下文（**Qwen3 / DeepSeek-V2 TP 知识入口**：先看 `inference_blueprint.json` 顶部的 `agent_navigation`，再按需展开 `model_layer` 与 `framework_layer.data_flow_contracts.tp_layer_interface_contracts`）：
   - `curosr/skills/inference_blueprint.json`
   - `notebooks/tasks.md`
   - `notebooks/MEMORY.md`
   - `notebooks/00_overview/README.md`
   - `notebooks-cn/01_framework_design/*`
   - `notebooks-cn/02_model_specifics/*`
   - `notebooks-cn/08_history_prompt/*`
   - `notebooks-cn/06_experience/*`
   - `notebooks-cn/07_agentPlan/*`
   - `notebooks-cn/04_parallel_strategies/*`
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
8. **先框架后模型**：必须先实现 `inference_blueprint.json.framework_layer.components`，并逐组件单测全部通过，才允许进入 Qwen3/DeepSeek-V2 的 TP 适配。

### 1.1 面向可观测性设计（Design for Observability）

1. **原生探针注入**  
   复杂算子（如 `DeepseekAttentionTP`、`ExpertParallelMoE`）必须预留 `DEBUG_MODE` 开关。
2. **数值健康检查**  
   每层 forward 末尾必须预留 NaN/Inf 检查路径，例如：
   - `assert not torch.isnan(out).any()`
   - `assert not torch.isinf(out).any()`
3. **维度白盒化**  
   关键中间变量必须支持打印 shape/device/dtype（受 debug 开关控制）。

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
   - `/data/xinference/cache/Qwen3-8B` 在 TP=4 推理时，每卡 VRAM% 约 `7%`。  
   - `/data/xinference/cache/deepseek-v2-chat-pytorch-16b` 在 TP=4 推理时，每卡 VRAM% 约 `14%`。  
4. **真实并行计算证据**  
   - 测试窗口内需出现 HCU% `> 0` 的时刻，且至少覆盖 4 张 TP 卡。  
5. **反作弊约束**  
   - 若仅有硬编码文本输出、无 HCU/VRAM 证据链，视为“假推理”，验收失败。

---

## 2. 阶段 1：解析与路由（Routing & Analysis）

当用户给出一句话需求（例如“支持某模型目录 TP 推理”）时，执行：

1. 读取模型 `config.json`。
2. 根据 `architectures[0]` 在 `inference_blueprint.json` 查询：
   - 模型家族（Dense / MLA+MoE）
   - Attention 路由（标准/MLA）
   - MLP/MoE 路由（TP/EP）
   - RoPE 风格（Neox 或 GPT-J）
3. 形成“本次唯一执行路径”说明，明确：
   - 必须 replicated 的权重
   - 必须 TP/EP 切分的权重
   - 必须启用的数值补丁（如 YaRN mscale）

---

## 3. 阶段 2：单路径组装（Single-Path Assembly）

1. 严禁复制 vLLM 巨型分支结构与运行时 if-else 树。
2. 根据检索到的算子约束，直接合成单路径模型实现：
   - Dense：Qwen 路径（QKV/O + gate/up/down）
   - MLA+MoE：DeepSeek 路径（q_a/kv_a replicated + q_b/kv_b/o TP + routed EP）
3. 禁止“先写通用再临时 patch”；应直接产出架构定制代码。

---

## 3.1 阶段 2.5：框架组件优先落地（Framework First Gate）

在任何 TP 模型代码开始前，必须先完成以下 gate：

1. 先实现并验证框架组件（按依赖顺序）：
   - `Sequence`
   - `Sampler`
   - `BlockManager`
   - `KVMemoryPool`
   - `Scheduler`
   - `LLMEngine`（仅框架闭环，不含特定模型 TP 逻辑）
2. 每个组件都要有对应单测并通过。
3. 只有在框架层单测全绿后，才能开始：
   - `QwenTPModelRunner`
   - `DeepseekTPModelRunner`
   - 模型权重惰性加载与 TP/EP 适配

硬门禁：若框架组件单测存在失败，禁止修改模型 TP 代码。

---

## 4. 阶段 3：惰性装载合成（Lazy Loader Synthesis）

1. 统一使用 `safetensors.safe_open` + `model.safetensors.index.json`。
2. 生成按架构定制的惰性加载策略：
   - Dense：按 dim0/dim1 常规切片。
   - DeepSeek MLA+MoE：
     - `q_a_proj`、`kv_a_proj_with_mqa` 全量读取。
     - `q_b_proj`、`kv_b_proj_with_mqa`、`o_proj` 走 TP 切片读取。
     - routed experts 仅加载本 rank `local_expert_ids`，禁止全量读专家。
3. 线性层 `load_weight_shard` 必须兼容“输入已是 local shard”的直拷逻辑。

---

## 5. 阶段 4：引擎接入与数值对齐（Engine Integration & Numerics Alignment）

1. 把新模型注册到引擎路由（读取 `config.json.architectures` 自动分发 Runner）。
2. 编写 `test_xxx_tp_real.py` 接入 `LLMEngine + Scheduler`：
   - 批量添加 prompt
   - `while eng.has_unfinished_requests(): eng.step()`
   - 输出 `eng.get_generation_outputs()`
3. 与 HF 进行 `temperature=0.0` 对齐（用于验证）：
   - 文本一致性
   - logits 或分层 hidden diff（至少首 token）
4. 确认测试日志写在项目目录下，禁止默认写 `/tmp`。

---

## 6. 标准工作流（Workflow）

### 6.1 TDD 与细粒度单元测试（Granular Unit Testing）

1. 放弃仅依赖单一 E2E 测试。
2. 执行顺序必须是：
   - 先写算子单测（例如 `moe.py`、`linear.py`）。
   - 实例化单层算子，喂随机 Tensor，触发 shape 校验与 NaN/Inf 检查。
   - 单测通过后再接入 `ModelRunner`。
   - 最后跑 `LLMEngine` 端到端回归。
3. 任一子模块未通过，不得推进下一层集成。

### 6.2 端到端门禁

1. 真实 TP 测试必须由自研引擎驱动，而不是仅 HF-debug 旁路。
2. 回归需要覆盖：
   - 多 prompt 中文可读输出
   - 多次复跑稳定性
   - 不同 rank 下一致性
3. 一次任务必须完整覆盖 `phase_1_framework_components` 到 `phase_5_realtime_acceptance`，不得只完成前两阶段就结束。
4. `phase_5` 验收除文本正确性外，必须附带监控证据：4 卡 VRAM% 同量级 + HCU% 出现大于 0。

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

---

## 8. 完成定义（Definition of Done）

仅当以下条件全部满足，任务才算完成：

1. 目标模型通过架构路由并正确实例化 TP Runner。
2. Lazy loader 满足该模型全部切分规则（含 replicated/TP/EP）。
3. 子模块单测全部通过（含 NaN/Inf 与 shape 校验）。
4. `test_xxx_tp_real.py` 在 torchrun 下输出稳定、可读文本。
5. 提供变更摘要：文件、规则映射、测试命令、风险与后续建议。
