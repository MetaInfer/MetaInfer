# AGENT SKILL: Inference Framework Closed-Loop SOP

你是用于 `meta-infer` 项目的代码生成子 Agent。你的唯一目标是：**严格依据 `inference_blueprint.json` 完成可测试、可复现、可调试的推理框架开发闭环**。

---

## 0. 启动前强制动作

1. 进入工作根目录：`meta-infer`。
2. 若不存在目录 `meta-infer/agents-infer`，立即创建。
3. 后续新生成的中间产物（草稿、调试脚本、阶段性实验）默认写入 `meta-infer/agents-infer`。
4. 每次开始任务前，必须先读取：
   - `notebooks/tasks.md`
   - `notebooks/MEMORY.md`
   - `notebooks/00_overview/README.md`
   - `curosr/skills/inference_blueprint.json`

---

## 1. 执行铁律（Prime Directives）

1. **TDD 必须执行**：先写测试，再写实现。
2. **契约优先**：所有张量形状、dtype、device 以 `inference_blueprint.json` 为准，禁止脑补。
3. **单组件封口**：当前组件单测未 100% 通过，禁止改动其他组件或做整体串联。
4. **变更可追溯**：每次改动后，输出你修改的文件列表与原因。
5. **日志必须充分**：保留关键 print（shape、block 使用、命中率、调度步延迟、结束原因）。

---

## 2. 标准工作流（Workflow）

### Step 1: 契约对齐

1. 解析 `inference_blueprint.json`：
   - `components`
   - `data_flow_contracts`
   - `logic_constraints`
2. 抽取当前任务涉及组件的契约（例如 `runner_decode_tensors`、`deepseek_v2_mla_kv_contract`）。
3. 形成“待验证清单”：每一条契约都要在代码或测试中可见。

### Step 2: 参考溯源

1. 使用 Bash 子智能体查阅 `inference_blueprint.json` 的 `ref_code`。
2. 只抽取必要实现模式，不复制复杂非主干逻辑。
3. 对照 `ref_docs` 校验是否遗漏关键步骤（调度、块分配、采样、生命周期）。

### Step 3: 编写与测试

1. 先在`meta-infer/agents-infer/tests/` 下实现组件代码。
2. 后新增 `meta-infer/agents-infer/tests/` 下对应测试。
3. 立即运行最小相关测试：
   - 单组件：`pytest meta-infer/agents-infer/test_xxx.py -q`
   - 端到端：`python -m pytest meta-infer/agents-infer/test_real_inference.py -s -q`

### Step 4: 即时单测门禁

1. 任一组件改动后，必须立刻运行该组件测试。
2. 测试失败时，只允许在该组件范围内修复。
3. 该组件未通过前，不得推进下一组件，不得做跨组件集成。

---

## 3. 错误排查指南（Troubleshooting）

### A. Shape mismatch

1. 回看 `inference_blueprint.json > data_flow_contracts`。
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

## 4. 完成定义（Definition of Done）

任务仅在以下条件全部满足时视为完成：

1. `python -m pytest tests/test_real_inference.py -s -q` 成功通过。
2. 终端日志中可见真实 16B 模型输出文本（如 “The capital of France is” 的合理续写）。
3. 所有新增/修改组件测试通过。
4. 输出最终变更摘要（文件、契约点、测试结果、残余风险）。

---

## 5. 强制日志规范（Human-in-the-loop Debug）

每个关键阶段必须打印：

1. **调度层**：step、phase、waiting/running 数量、free_blocks。
2. **内存层**：`num_blocks`、`bytes_per_token`、`bytes_per_block`、logical KV bytes。
3. **模型层**：prefill/decode 输入 shape、logits shape、next_tokens。
4. **生命周期**：每个请求的 finish reason（`eos` / `max_tokens`）与耗时。

日志缺失视为任务未达标。
