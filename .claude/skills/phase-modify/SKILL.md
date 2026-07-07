# Phase Modify：增量修改模式

## 触发词

`/phase-modify` — 在已有引擎代码基础上应用策略变更，非全量重建。

## 触发场景

master 循环 iter-002 及后续轮次使用此模式。首轮（iter-001）仍使用 `/phase-all` 全量构建。

## 概述

与 `/phase-all` 不同，`/phase-modify` **不清理引擎代码**。已有代码是从上一轮成功迭代继承的。子 agent 只做四件事：读策略 → 定位文件 → 修改代码 → 验证受影响 Phase。

## 执行流程

### Step 1: 读取策略文件

读 `iterations/strategy-<ITER_ID>.json`。提取 `changes[]` 数组。

### Step 2: 对每个 change 执行修改

对 `changes[]` 中的每个条目：

1. **定位目标文件**：根据 `change.file` 字段，在项目根目录下找到对应文件
2. **理解当前代码**：读取目标文件中与修改相关的代码段
3. **执行修改**：使用 Edit 工具进行精确修改，不重写整个文件
4. **记录变更**：记录修改前后的代码差异

**changes[] 条目结构**：
```json
{
  "file": "llm_engine.py",
  "action": "change_constructor_arg",
  "param": "block_size",
  "from": 256,
  "to": 128,
  "description": "LLMEngine.__init__ 中 block_size 参数从 256 改为 128"
}
```

**action 类型与修改方式**：

| action | 修改方式 |
|--------|---------|
| `change_constructor_arg` | 修改对应文件中构造函数/初始化方法的参数默认值或传入值 |
| `add_constructor_param` | 在构造函数中新增参数并在方法签名和调用处同步 |
| `algorithmic` | 修改算法实现逻辑（如算子替换、kernel 选择），按 description 描述执行 |
| `architectural` | 修改架构级决策（如切换 attention 后端），可能涉及多文件联动 |

### Step 3: 确定受影响 Phase

根据修改的文件和参数，确定需要重新验证的 Phase：

| 修改目标 | 受影响 Phase |
|---------|-------------|
| `engine/kernels/*` | Phase 1 |
| `engine/tp_layers/distributed.py` | Phase 2 |
| `engine/tp_layers/linear.py` | Phase 3 |
| `engine/tp_layers/embedding.py` | Phase 4 |
| `engine/models/*` (attention 部分) | Phase 5 |
| `engine/models/*` (mlp/decoder 部分) | Phase 6 |
| `engine/models/*` (weight_loading 部分) | Phase 7 |
| `engine/framework/*` | Phase 8 |
| `llm_engine.py` | Phase 9 |
| `llm_engine.py` (benchmark 相关参数) | Phase 10, Phase 11 |

如果 `changes[]` 涉及多个 Phase，需要依次验证所有受影响的 Phase。

### Step 4: 运行受影响 Phase 的 scripts/ 门禁

只运行受影响 Phase 的 `scripts/` 测试（不运行全量 Phase 1-11）：

```bash
source .env_agent_infer
# 示例：只跑 Phase 3 的测试（因为改了 block_size 相关参数）
bash scripts/test_phase3_tp_linear.py
bash scripts/test_phase3_tp_linear_tp4.py
```

- 如果全部 PASS → 进入 Step 5
- 如果任一 FAIL → 回退本轮所有修改，写 FAILURE_REPORT.md，退出

### Step 5: 运行完整验证（可选，变更影响面大时）

如果修改涉及 Phase 7+ 的文件，建议运行全量 regression 验证：

```bash
# 跨 Phase 回归：重跑 Phase 8 之前的全部 scripts/
# 至少跑 Phase 9 的 E2E 验证
bash scripts/test_phase9_generate_single_gpu.sh
```

### Step 6: 写结果

修改全部通过后，写输出到 `iterations/<ITER_ID>/`：

1. `code_changes.txt` — 列出所有修改的文件、变更内容
2. `phase_report/benchmarks.jsonl` — KPI 数据（运行 benchmark 获取）
3. `phase_report/AGGREGATE_REPORT.md` — 受影响 Phase 的验证结果汇总

## 回退机制

如果修改导致测试失败：
1. 从 `master/results/<上一轮_iter_id>/code/` 恢复被修改的文件
2. 写 FAILURE_REPORT.md 说明失败原因
3. 退出码非零

## 硬约束

| 约束 | 细节 |
|------|------|
| 不清理引擎代码 | 已有代码保留，只在上面增量修改 |
| 不运行全量 /phase-all | 只运行受影响 Phase 的 scripts/ |
| 变更可回退 | 所有修改前记录原始状态，失败时完整回退 |
| scripts/ 不可变 | 测试不过就回退修改，绝不改测试 |
| 策略即指令 | changes[] 是精确的修改指令，不需要子 agent 自行发挥 |
