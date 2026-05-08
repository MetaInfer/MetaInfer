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

### 主 Agent + 子代理协作

协同规则：
1. **主 Agent** 负责调度子代理、汇总输出、推进 phase。
2. **子代理调用顺序**（每个组件/每轮实现）：
   - `contract-checker` → 产出开发前检查清单（硬门禁标注）。
   - `ref-tracer` → 仅在新组件首次实现或重大契约变更时调用，提取最小可迁移路径。
   - `tdd-test-writer` → 先写测试草案，声明"已满足先测后写"。
   - `impl-coder` → 按契约 + 测试实现；测试失败仅在本组件内迭代。
   - `integration-verifier` → 端到端闭环验证，审计日志与 DoD。
3. 禁止多个实现型子代理并行修改同一仓库。
4. 允许且仅允许额外开启 **1 个只读监控子代理**（`tp-gpu-monitor`），职责仅限实时采集 `nvidia-smi` 数据（显存占用/GPU 利用率），**不得写代码**，不得干预主流程。

### 子代理门禁矩阵

| 子代理 | 前置条件 | 产出 | 失败处理 |
|--------|---------|------|---------|
| `contract-checker` | 已读取 blueprint + AGENT_SKILL + 本轮任务 | 结构化检查清单（含硬门禁） | 硬门禁未满足 → 停止开发 |
| `ref-tracer` | blueprint + 指定组件 | 最小可迁移路径 + 禁止迁移项 | 路径不清 → 回查 blueprint |
| `tdd-test-writer` | blueprint + AGENT_SKILL + 目标组件 | pytest 测试草案 | 测试草案未产出 → 禁止实现 |
| `impl-coder` | contract-checker 清单 + tdd 测试草案 + 当前测试已执行 | 改动文件 + 实现点 + 测试结果 | 测试失败 → 仅当前组件内迭代 |
| `integration-verifier` | 所有单测通过 | pass/fail + 证据 + 风险 + 修复建议 | 失败 → 定位组件并回退 |
| `tp-gpu-monitor` | 指定 GPU 编号 + 测试命令 | TP 监测验证清单 | 显存/计算硬门禁失败 → TP 验证不通过 |

### 反假输出（强制）

1. **禁止**打印硬编码、占位或演示用"假生成结果"冒充真实推理。
2. **禁止**在未执行 `load_weights()` / 未从磁盘读入权重的情况下声称 TP 已跑通。
3. 验收输出必须来自：`pytest` 真实跑过 或 `torchrun` 下真实前向；须能指出**对应代码路径**与**日志中的显存证据**。
4. 单卡 HF 调试用例**不得**替代 `torchrun --nproc_per_node={TP_SIZE}` 的 TP 验收。

### 长上下文防遗忘

每完成一个 blueprint phase 或每轮大改前，**重新打开并扫一眼**本 SKILL.md 与 `inference_blueprint.json` 中当前 phase 对应的节点。可选：写 `PROGRESS.md` 记录当前 phase 与已验证命令。

### 一次完成 phase1-5 的执行门禁

1. 开始前先输出 phase 清单与每 phase 的测试命令。
2. phase1/2 通过后**不得结束任务**，必须继续 phase3/4，最终执行 phase5。
3. 只有当 `phase_5_realtime_acceptance` 全部通过，才允许输出"任务完成"。

---

## 0. 启动前强制动作

1. 读取 `.claude/skills/inference-sop/references/inference_blueprint.json`（先看 `agent_navigation`，再按需展开 `model_layer` 与 `framework_layer.data_flow_contracts`）。
2. 读取知识库入口：
   - `notebooks/MEMORY.md`
   - `notebooks/00_overview/README.md`
3. 对输入模型目录读取 `config.json`，提取：`architectures`、`rope_scaling`、`num_attention_heads`、`num_key_value_heads`、`n_routed_experts`、`n_shared_experts`、`num_experts_per_tok`。
4. 输出"模型路由结论"：Dense 还是 MLA+MoE。
5. 从用户 prompt 中解析 TP_SIZE 与 GPU ID 列表（用户通过 `CUDA_VISIBLE_DEVICES=` 或 `GPU_IDS=` 指定）。执行 `nvidia-smi` 确认目标 GPU 可用且 VRAM 未被大量占用，然后设置 `CUDA_VISIBLE_DEVICES` 约束可见设备。后续所有 `torchrun` 命令均使用用户指定的参数动态构造：`--nproc_per_node={TP_SIZE}`、`CUDA_VISIBLE_DEVICES={GPU_IDS}`。

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

## 1.5 编码铁律与防遗忘守则

**编码前必读**：以下规则来自多轮调试的惨痛教训（详见 `notebooks-cn/06_experience/04_task13_tp_correctness_fix_rmsnorm.md` 和 `05_task13_one_shot_tp_build_guide.md`）。违反任何一条都会导致输出完全错误，且难以通过表面现象定位根因。

### 精度铁律（Precision Laws）

| # | 规则 | 正确做法 | 错误做法 | 后果 |
|---|------|---------|---------|------|
| P1 | RMSNorm weight 乘法在 fp32 | `(x * self.weight.float()).to(dtype)` | `x.to(dtype).mul_(weight)` 或 `(x * weight).to(dtype)` | bf16 乘法放大误差 3x → MoE 路由错误 → 输出乱码 |
| P2 | all_reduce 在 fp32 | `tmp = x.float(); dist.all_reduce(tmp, SUM)` | `dist.all_reduce(x, SUM)` 直接在 bf16 | 数值溢出/精度丢失 |
| P3 | RoPE cos/sin 转回 input dtype | `cos, sin = cos.to(input_dtype), sin.to(input_dtype)` | cos/sin 保持 fp32 不转 | 小角度精度差异级联 36 层 |
| P4 | Router logits 在 fp32 | `F.linear(x.float(), gate.weight.float())` | `F.linear(x, gate.weight)` 在 bf16 | softmax 精度不足 → 路由抖动 |
| P5 | Router softmax 在 fp32 | `torch.softmax(logits, dim=-1, dtype=torch.float32)` | 默认 bf16 softmax | top-k 选择不稳定 |

### TP 切分铁律（Tensor Parallel Laws）

| # | 规则 | 正确做法 | 错误做法 | 后果 |
|---|------|---------|---------|------|
| T1 | ColumnParallel 切 dim=0 | `weight[rank*local : end, :]` | 切 dim=1 | 权重错乱 |
| T2 | RowParallel 切 dim=1 | `weight[:, rank*local : end]` | 切 dim=0 | 权重错乱 |
| T3 | load_weight_shard 防双切片 | 先检查 `shape == self.weight.shape` | 直接按 rank 切片 | 已切片的权重再切 → shape=0 或错乱 |
| T4 | MLA 低秩投影不切片 | `q_a_proj`, `kv_a_proj_with_mqa` 全量复制 | 传 split_dim 切片 | 注意力计算完全错误 |
| T5 | Embedding 防双切片 | 检查 `shape[0] == local_vocab_size` | 直接按 vocab_start:end 切 | rank>0 权重为空 |

### RoPE 铁律（Rotary Position Embedding Laws）

| # | 模型 | `_rotate_half` 风格 | cos/sin dtype |
|---|------|-------------------|---------------|
| R1 | Qwen3 | 前一半/后一半 split: `x1=x[...,:d], x2=x[...,d:]` | bf16 |
| R2 | DeepSeek-V2 | 奇偶交错 (GPT-J): `x1=x[...,::2], x2=x[...,1::2]` | bf16 |
| R3 | 两者共用 | cos/sin 在 fp32 下计算后 **必须** `.to(input_dtype)` | — |

**严禁混用**：Qwen3 用奇偶交错 → 输出乱码；DeepSeek-V2 用前一半/后一半 → RoPE 完全错误。

### GQA 铁律（Grouped Query Attention Laws）

| # | 规则 | 正确做法 | 错误做法 |
|---|------|---------|---------|
| G1 | KV head repeat 使用 repeat_interleave | `k.repeat_interleave(n_rep, dim=2)` | `k.unsqueeze(2).expand(...).reshape(...)` |
| G2 | expand+reshape 产生错误布局 | — | 交错 [kv0,kv1,kv0,kv1] 而非分组 [kv0,kv0,kv1,kv1] |
| G3 | num_kv_heads < tp_size 时 KV 复制 | `allow_kv_replication=True` | 直接切片导致 shape 不整除 |

### MoE 铁律（Mixture of Experts Laws）

| # | 规则 | 正确做法 | 错误做法 |
| M1 | ExpertParallelMoE 期望 3D 输入 | `hidden_states.unsqueeze(0)` 转 3D | 直接传 2D `[N, H]` |
| M2 | Router 全量复制 | gate weight 不做 TP/EP 切分 | 切分 gate → 各 rank 路由不一致 |
| M3 | Routed experts EP + all_reduce | 各 rank 只算自己的 expert，all_reduce 求和 | 全量加载所有 expert → OOM |
| M4 | Shared experts TP | gate/up ColumnParallel, down RowParallel | 用 EP 处理共享专家 |
| M5 | routed_scaling_factor | topk 后乘以 | 遗漏 → FFN 信号衰减 |

### 模型构建铁律（Model Construction Laws）

| # | 规则 | 正确做法 | 错误做法 |
|---|------|---------|---------|
| C1 | CPU-first 构建 | CPU 创建 → load_weights → `.cuda()` | `set_default_device('cuda')` 直接在 GPU 创建 |
| C2 | dtype 管理 | `set_default_dtype(bf16)` 创建后立即恢复 `float32` | 不恢复 → 后续代码意外使用 bf16 |
| C3 | Safetensors 惰性加载 | `safe_open` + `get_slice` | `load_file` 全量加载 → CPU OOM |
| C4 | 权重映射表 | 维护完整的 HF key → TP component 映射 | 硬编码路径或遗漏权重 |

### YaRN 铁律（DeepSeek-V2 专用）

| # | 规则 | 说明 |
|---|------|------|
| Y1 | 必须实现 `_compute_inv_freq` 中的 YaRN 频率插值 | `inv_interp * (1-inv_mask) + inv_freq * inv_mask` |
| Y2 | 必须实现 `_yarn_get_mscale` | attention scaling 修正 |
| Y3 | 必须实现 `_yarn_find_correction_range` 和 `_yarn_linear_ramp_mask` | 频率混合边界计算 |
| Y4 | 遗漏任何一项 → 长序列位置编码错误 → 输出语义漂移 | |

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

**编码时必须检查**（防遗忘清单）：
- RoPE `_rotate_half`：前一半/后一半 split（规则 R1），**不是**奇偶交错
- cos/sin：fp32 计算后 `.to(input_dtype)` 转 bf16（规则 P3）
- RMSNorm：`(x * self.weight.float()).to(dtype)`（规则 P1）
- GQA：`repeat_interleave`（规则 G1），**不是** expand+reshape
- KV head replication：`allow_kv_replication=True`（规则 G3）
- 模型构建：CPU-first（规则 C1），safetensors 惰性加载（规则 C3）

测试命令：
```bash
CUDA_VISIBLE_DEVICES={GPU_IDS} torchrun --nproc_per_node={TP_SIZE} -m pytest tests/test_qwen_tp_real.py -v -s
```

### Phase 4：DeepSeek-V2 TP/EP 适配

实现 MLA TP 切分、MoE 路由专家 EP + all_reduce、safetensors 惰性加载、YaRN scaling。

**编码时必须检查**（防遗忘清单）：
- RoPE `_rotate_half`：奇偶交错 GPT-J style（规则 R2），**不是**前一半/后一半
- YaRN：完整实现 `_compute_inv_freq`（规则 Y1-Y4），遗漏任何一项 → 输出乱码
- MLA 投影：`q_a_proj`/`kv_a_proj_with_mqa` 全量复制（规则 T4），**严禁切片**
- MoE：输入 3D（规则 M1），Router fp32（规则 P4），routed_scaling_factor（规则 M5）
- MoE 权重：Router replicated（规则 M2），routed experts EP（规则 M3），shared experts TP（规则 M4）
- RMSNorm：`(x * self.weight.float()).to(dtype)`（规则 P1）
- 模型构建：CPU-first（规则 C1），safetensors 惰性加载（规则 C3）

测试命令：
```bash
CUDA_VISIBLE_DEVICES={GPU_IDS} torchrun --nproc_per_node={TP_SIZE} -m pytest tests/test_deepseek_tp_real.py -v -s
```

### Phase 5：实时验收

串行执行 qwen 与 deepseek 真实 TP 回归，记录输出与监控证据。

测试命令：
```bash
CUDA_VISIBLE_DEVICES={GPU_IDS} torchrun --nproc_per_node={TP_SIZE} -m pytest tests/test_qwen_tp_real.py -v -s
CUDA_VISIBLE_DEVICES={GPU_IDS} torchrun --nproc_per_node={TP_SIZE} -m pytest tests/test_deepseek_tp_real.py -v -s
```

---

## 3. Debug 与自愈

错误发生时按以下优先级处理：

1. **Shape mismatch** → 回看 `data_flow_contracts`，打印关键节点 shape。
2. **OOM** → 检查 KVMemoryPool 参数，按需降压 mem_utilization / 增大 reserve_bytes。检查是否在 GPU 上直接创建模型。
3. **输出乱码** → 按以下流程排查（见 3.1 节）。
4. **数值异常** → 开启 DEBUG_MODE，定位首个 NaN/Inf 层。
5. **卡住** → 缩小到单组件最小可复现测试，重读 blueprint 对应节点。

故障算子定位后，强制回查 `inference_blueprint.json` 的 `global_primitives_constraints` 和 `failure_mode_library`。

### 3.1 输出乱码排查流程（按优先级）

**第一步：首 token 对比**
- 打印 logits 的 top-5 tokens 和 values
- 与 HF 真值对比 `max_diff` 和 `mean_diff`

**第二步：按误差大小定位**
- `max_diff > 0.1`：检查 RMSNorm weight 乘法是否在 fp32（规则 P1）
- `max_diff 0.01~0.1`：检查 RoPE cos/sin dtype（规则 P3）、rotate_half 风格（规则 R1/R2）
- `max_diff < 0.01` 但首 token 不同：检查 GQA KV head 布局（规则 G1/G2）

**第三步：逐层误差追踪**
- 如果第二步未定位，逐层对比 hidden_states 的 max_diff/mean_diff
- 找到首个发散层后，检查该层的 attention/MLP/MoE 实现

**常见根因速查**：
| 现象 | 最可能根因 | 检查项 |
|------|-----------|--------|
| DeepSeek 首 token 差异巨大 (如 12350 vs 185) | RMSNorm bf16 weight 乘法 | P1 |
| Qwen3 输出乱码/拉丁字符 | GQA expand+reshape 布局错误 | G1/G2 |
| Qwen3 首 token 偏差 | cos/sin dtype 不匹配 | P3 |
| DeepSeek 长序列输出漂移 | 遗漏 YaRN 频率插值 | Y1-Y3 |
| 输出重复/复读机 | RoPE rotate_half 风格错误 | R1/R2 |
| MoE 输出不稳定 | Router bf16 精度不足 | P4/P5 |
| 模型创建时 OOM | 在 GPU 上直接创建模型 | C1 |
| RuntimeError shape 不匹配 | MoE 接收 2D 输入 | M1 |
| rank>0 输出全零 | Embedding 双切片 | T5 |

### 3.2 编码阶段防遗忘检查清单

**每次写完一个组件后，对照检查**：

写 RMSNorm 时：
- [ ] `self.weight.float()` 在乘法前调用
- [ ] `x.float()` 在方差计算前调用
- [ ] `.to(input_dtype)` 在返回前调用

写 ColumnParallelLinear 时：
- [ ] `load_weight_shard` 有 shape 匹配检查
- [ ] 切分维度是 dim=0（输出维度）

写 RowParallelLinear 时：
- [ ] `load_weight_shard` 有 shape 匹配检查
- [ ] 切分维度是 dim=1（输入维度）
- [ ] forward 中 all_reduce 后才加 bias

写 RoPE 时：
- [ ] 确认目标模型的 rotate_half 风格（Qwen3 vs GPT-J）
- [ ] cos/sin 在 fp32 下计算
- [ ] cos/sin `.to(input_dtype)` 转回 bf16

写 Attention 时：
- [ ] GQA 使用 `repeat_interleave`（不是 expand+reshape）
- [ ] KV head replication 时使用 `allow_kv_replication`

写 MoE 时：
- [ ] 期望 3D 输入，添加维度处理
- [ ] Router 在 fp32 下计算
- [ ] 乘以 `routed_scaling_factor`

写权重加载时：
- [ ] MLA 低秩投影全量复制（不传 split_dim）
- [ ] Embedding 检查 `shape[0] == local_vocab_size`
- [ ] 使用 `safe_open` + `get_slice`（不用 `load_file`）

写模型 Runner 时：
- [ ] CPU 上创建模型
- [ ] `set_default_dtype` 后恢复 `float32`
- [ ] load_weights 后再 `.cuda()`

---

## 4. 完成定义（DoD）

仅当以下全部满足，任务完成：

1. 目标模型通过架构路由并正确实例化 TP Runner。
2. Lazy loader 满足全部切分规则（replicated/TP/EP）。
3. 子模块单测全部通过（含 NaN/Inf 与 shape 校验）。
4. `test_xxx_tp_real.py` 在 torchrun 下输出稳定、可读文本。
5. 首 token 与 HF 真值一致（Qwen3-8B: 220, DeepSeek-V2-Lite: 185）。
6. 所有精度铁律（P1-P5）、TP 铁律（T1-T5）、RoPE 铁律（R1-R3）、GQA 铁律（G1-G3）、MoE 铁律（M1-M5）、构建铁律（C1-C4）均已遵守。
7. 提供变更摘要：文件、规则映射、测试命令、风险与后续建议。

## 参考文件

- `references/inference_blueprint.json` — 架构知识图谱（唯一契约来源），包含组件定义、数据流契约、模型知识库、失败模式库
- `references/prompt.txt` — 长任务一次性执行的完整 prompt 模板
- `notebooks-cn/06_experience/04_task13_tp_correctness_fix_rmsnorm.md` — RMSNorm + GQA 修复的完整故障链分析
- `notebooks-cn/06_experience/05_task13_one_shot_tp_build_guide.md` — 一次性 TP 框架搭建的完整参考（含所有组件的精确实现模式）

## 附录：编码阶段自检速查卡

**每次写代码前看一眼**：

```
RMSNorm:  (x * self.weight.float()).to(dtype)  ← weight 必须 .float()
all_reduce: tmp = x.float(); all_reduce(tmp)    ← bf16 必须先转 fp32
RoPE Qwen3: x1=x[...,:d], x2=x[...,d:]        ← 前一半/后一半
RoPE DS-V2: x1=x[...,::2], x2=x[...,1::2]     ← 奇偶交错
cos/sin:    cos.to(input_dtype)                  ← 必须转回 bf16
GQA:        k.repeat_interleave(n_rep, dim=2)    ← 不是 expand+reshape
MoE input:  hidden_states.unsqueeze(0) if 2D    ← 必须 3D
Router:     F.linear(x.float(), gate.weight.float()) ← fp32
构建:       CPU 创建 → load_weights → .cuda()   ← 不是 GPU 上直接创建
加载:       safe_open + get_slice                ← 不是 load_file
MLA 投影:   q_a_proj, kv_a_proj 全量复制        ← 严禁 split_dim
Embedding:  check shape[0] == local_vocab_size   ← 防双切片
```
