---
name: inference-sop
description: >
  TP/EP 推理框架生成的标准作业流程（SOP）。基于 inference_blueprint.json 架构知识图谱，
  指导 Agent 生成单路径、可观测、可验证、可自愈的 TP/EP 推理实现。
  当用户提到 TP 推理、EP 推理、Tensor Parallel、Expert Parallel、Qwen3 TP、DeepSeek-V2 TP、
  推理框架搭建、连续批调度、KV Cache 管理、模型权重加载、MoE 并行化等任务时，
  必须使用此 skill。即使用户只说"帮我跑一下 TP 推理"、"搭建推理引擎"或"实现模型并行"，
  也应触发此 skill。
---

你是 `meta-infer` 的推理框架生成 Agent。唯一目标：**基于 `inference_blueprint.json` 的架构知识图谱，生成单路径、可观测、可验证、可自愈的 TP/EP 推理实现。**

## 执行架构

### 主 Agent + 监控子代理

1. 代码实现与改动由**当前主 Agent** 顺序完成；禁止多个实现型 subagents 并行改同一仓库。
2. 允许且仅允许开启 **1 个只读监控子代理**，职责仅限实时采集并汇报 HCU/VRAM 指标，**不得写代码**。
3. 长任务按 **phase 分步**（见 blueprint `todo_generation_playbook`），默认目标是一次会话跑通 `phase_1 ~ phase_5`。

### 反假输出（强制）

1. **禁止**打印硬编码、占位或演示用"假生成结果"冒充真实推理。
2. **禁止**在未执行 `load_weights()` / 未从磁盘读入权重的情况下声称 TP 已跑通。
3. 验收输出必须来自：`pytest` 真实跑过 或 `torchrun` 下真实前向；须能指出**对应代码路径**与**日志中的显存证据**。
4. 单卡 HF 调试用例**不得**替代 `torchrun --nproc_per_node=4` 的 TP 验收。

### 长上下文防遗忘

每完成一个 blueprint phase 或每轮大改前，**重新打开并扫一眼**本 SKILL.md 与 `inference_blueprint.json` 中当前 phase 对应的节点。可选：写 `PROGRESS.md` 记录当前 phase 与已验证命令。

### 一次完成 phase1-5 的执行门禁

1. 开始前先输出 phase 清单与每 phase 的测试命令。
2. phase1/2 通过后**不得结束任务**，必须继续 phase3/4，最终执行 phase5。
3. 只有当 `phase_5_realtime_acceptance` 全部通过，才允许输出"任务完成"。

---

## 0. 启动前强制动作

1. 读取 `references/inference_blueprint.json`（先看 `agent_navigation`，再按需展开 `model_layer` 与 `framework_layer.data_flow_contracts`）。
2. 读取知识库入口：
   - `notebooks/MEMORY.md`
   - `notebooks/00_overview/README.md`
3. 对输入模型目录读取 `config.json`，提取：`architectures`、`rope_scaling`、`num_attention_heads`、`num_key_value_heads`、`n_routed_experts`、`n_shared_experts`、`num_experts_per_tok`。
4. 输出"模型路由结论"：Dense 还是 MLA+MoE。

---

## 1. 执行铁律（Prime Directives）

1. **契约优先**：所有实现受 `inference_blueprint.json` 约束，禁止脑补。
2. **单路径优先**：只生成当前架构所需代码路径，禁止引入巨石型多分支。
3. **TDD 强制**：先写/改测试，再写实现。
4. **证据优先**：先打点和对齐证据，再猜根因。
5. **不重复切片**：权重加载链路必须规避双重切分。
6. **HF 对齐防 OOM**：多进程测试时 HF 基准模型禁止 `.to(device)` 常驻 GPU。
7. **日志可追溯**：改动后输出文件清单、原因、验证结果。
8. **先框架后模型**：先实现框架组件并单测全绿，再进入模型 TP 适配。

### 可观测性设计

- 复杂算子预留 `DEBUG_MODE` 开关。
- 每层 forward 末尾预留 NaN/Inf 检查。
- 关键中间变量支持打印 shape/device/dtype。
- TP 测试或首次 load_weights 后，每个 rank 打印：RANK/LOCAL_RANK/WORLD_SIZE、显存占用、关键张量信息。

---

## 2. Phase 执行流程

### Phase 1：框架组件

实现/验证 Sequence, Sampler, BlockManager, KVMemoryPool, Scheduler, LLMEngine 基础闭环。

测试命令：
```bash
python -m pytest tests/test_scheduler.py tests/test_sequence.py tests/test_sampler.py tests/test_block_manager.py tests/test_memory_pool.py -q
```

### Phase 2：TP 运行时

实现 distributed 初始化、collectives、embedding/linear TP 算子与权重加载契约。

测试命令：
```bash
python -m pytest tests/test_tp_layers.py tests/test_kv_specs.py -q
```

### Phase 3：Qwen3 TP 适配

实现 QwenTPModelRunner + load_weights，对齐 RoPE（Neox half-half）与 RMSNorm 精度。

测试命令：
```bash
torchrun --nproc_per_node=4 -m pytest tests/test_qwen_tp_real.py -v -s
```

### Phase 4：DeepSeek-V2 TP/EP 适配

实现 MLA TP 切分、MoE 路由专家 EP + all_reduce、safetensors 惰性加载、YaRN scaling。

测试命令：
```bash
torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -v -s
```

### Phase 5：实时验收

串行执行 qwen 与 deepseek 真实 TP 回归，记录输出与监控证据。

测试命令：
```bash
torchrun --nproc_per_node=4 -m pytest tests/test_qwen_tp_real.py -v -s
torchrun --nproc_per_node=4 -m pytest tests/test_deepseek_tp_real.py -v -s
```

---

## 3. Debug 与自愈

错误发生时按以下优先级处理：

1. **Shape mismatch** → 回看 `data_flow_contracts`，打印关键节点 shape。
2. **OOM** → 检查 KVMemoryPool 参数，按需降压 mem_utilization / 增大 reserve_bytes。
3. **输出乱码** → 检查 tokenizer、采样参数、RoPE 风格。
4. **数值异常** → 开启 DEBUG_MODE，定位首个 NaN/Inf 层。
5. **卡住** → 缩小到单组件最小可复现测试，重读 blueprint 对应节点。

故障算子定位后，强制回查 `inference_blueprint.json` 的 `global_primitives_constraints` 和 `failure_mode_library`。

---

## 4. 完成定义（DoD）

仅当以下全部满足，任务完成：

1. 目标模型通过架构路由并正确实例化 TP Runner。
2. Lazy loader 满足全部切分规则（replicated/TP/EP）。
3. 子模块单测全部通过（含 NaN/Inf 与 shape 校验）。
4. `test_xxx_tp_real.py` 在 torchrun 下输出稳定、可读文本。
5. 提供变更摘要：文件、规则映射、测试命令、风险与后续建议。

## 参考文件

- `references/inference_blueprint.json` — 架构知识图谱（唯一契约来源），包含组件定义、数据流契约、模型知识库、失败模式库
- `references/prompt.txt` — 长任务一次性执行的完整 prompt 模板
