# 测试脚本作为先验知识的分析与改造计划 v2

## 1. 当前问题本质

现在的流程是：

```
Agent 读 blueprint.json → Agent 写代码 → Agent 写测试 → Agent 跑测试 → 人检查
                                 ↑                    ↑
                          两处都可能犯错            两处都需要审查
```

Agent 生成的测试和 Agent 生成的代码**出自同一个模型**。如果模型对某个接口有误解（比如它"认为" fused_add_rms_norm 应该用跨层 weight），它会同时把错误写进代码和测试，测试自然通过，但实际是错的。

## 2. 改造目标

把测试**提前到 Agent 之外**：

```
人类写好测试脚本（已commit） → Agent 读 blueprint + 测试 → Agent 只写代码 → 跑人类写的测试 → 不通过就自查自修
                                    ↑                                                      ↑
                              Agent只能读测试，不能改测试                            人类审查过的标尺
```

**核心区别：测试不再是 Agent 的"作品"，而是 Agent 的"裁判"。**

## 3. 与 superpowers 的关系

superpowers 的 TDD skill（`test-driven-development`）的 RED-GREEN-REFACTOR 循环，假设的是**人类写测试，Agent 写实现**。当前做法把 RED 步骤交给了 Agent，违反了 TDD 的核心假设：**测试必须由"不相信代码正确的人"来写**。

`subagent-driven-development` 的对抗结构可以部分缓解，但如果 spec（测试文件）也是 Agent 生成的，对抗就失去了根基。

此外，`superpowers/CLAUDE.md` 对代码质量提出了硬性约束，这些约束**直接适用于本项目的测试脚本编写**：

### 3.1 superpowers CLAUDE.md 强制映射到测试脚本的 6 条铁律

| # | superpowers 规则 | 映射到 scripts/ 测试脚本的含义 |
|---|-----------------|------------------------------|
| 1 | **Fabricated content 零容忍**（"PRs containing invented claims...will be closed immediately"） | 每个测试必须引用**真实发生过的 bug**（session、error、commit hash）。禁止"这个测试理论上能防止 X"的假想测试 |
| 2 | **Speculative fixes 不收**（"must solve a real problem someone actually experienced"） | 禁止前摄性测试。测试用例必须能追溯到：V17 审计报告的具体 FG/OW 项，或物理 tracing 的 data_ptr/调用栈，或过往 agent 会话中实际犯过的错误 |
| 3 | **One problem per PR** | 每个测试脚本只锁定**一个不变性**。禁止在一个 test_xxx.py 中捆绑多个不相关的检查点 |
| 4 | **Human review required**（"show your human partner the complete diff"） | 每个测试脚本提交前必须经过人类 diff 审查。审查子代理的结论是辅助，最终决定权在人类 |
| 5 | **Skill = executable code**（"Skills are not prose — they are code that shapes agent behavior"） | `scripts/` 下的测试文件就是"可执行的 skill 约束"。必须能在多 session 下重复运行且输出确定性结果。必须经过 adversarial pressure testing（故意写错代码后确实会失败） |
| 6 | **Describe problem, not change** | 每个测试脚本文件头必须写清楚：**Why this test exists**（什么 bug），**When it was added**（日期+commit），**What failure looks like**（Agent 什么错误会触发失败） |

```
测试脚本头模板（必须填写）：
# [脚本名]
# Why: 防止 [具体错误描述]（发现于 [日期, commit/issue]）
# What failure: Agent 如果 [具体错误行为]，此测试将 [具体失败信息]
# Superpowers gate: 此脚本对应 superpowers CLAUDE.md rule 2（Speculative fixes 不收）
```

## 4. 三层知识体系（脚本扩展版）

```
第一层：先验知识（人类写，Agent 只读）
  ├── inference_blueprint.json    # 架构知识图谱
  ├── AGENT_SKILL.md              # 执行 SOP
  └── scripts/                    # 【新增】固定测试合约（.py 或 .sh，不做限制）
        │
        ├── 单元级（.py）           # 随机 Tensor 喂入，校验输出 shape/dtype/数值
        ├── 集成级（.py）           # 多组件联合，校验接口契约
        └── 端到端级（.sh）         # torchrun 拉起真实推理，校验贪婪解码字字对齐

第二层：生成产物（Agent 写，受第一层约束）
  ├── engine/**/*.py              # 推理框架代码
  └── tests/test_*.py             # Agent 可以补充额外测试（非强制）

第三层：验收证据（Agent 运行，不能伪造）
  ├── profiler trace
  ├── HCU/VRAM 监控
  └── benchmark JSON
```

### 脚本语言选择原则

| 测试类型 | 推荐格式 | 原因 |
|---------|---------|------|
| 纯 Python 函数级（随机 Tensor 喂入，校验 shape/value） | `.py` | pytest fixture 丰富，assert 精确 |
| TP 多卡集成（需 torchrun 环境） | `.py` | 可通过 `torchrun --nproc_per_node=N -m pytest` 运行 |
| 端到端贪婪解码 + 性能基线 | `.sh` | 涉及环境变量、conda 激活、多命令串联，shell 更自然 |
| Profiler 痕迹检查（确认无 compile/graph） | `.sh` | 需要对 profiler 输出做 grep/awk，shell 更高效 |

## 5. 每个模块的固定测试接口

每个组件对外暴露的接口通过预写测试脚本来锁定。以 Phase 6 Decoder Layer（最容易犯错的组件）为例：

```python
# scripts/test_phase6_decoder_layer.py
# 由人类写，commit 到 git。Agent 不能修改。
# 审查子代理：脚本每次修改后自动触发 code-review 子代理检查

def test_fused_add_rms_norm_weight_is_self_layer():
    """
    固定合约：所有 fused_add_rms_norm 的 weight 参数必须是本层 weight。
    过去最频繁的错误点——Agent 经常误用下一层 weight。
    Agent 写错 → 测试不通过 → Agent 自查 → 修正。
    """

def test_decode_residual_chain_no_cross_layer_reference():
    """Prevent cross-layer weight reference."""

def test_decode_forward_has_no_clone_in_eager_mode():
    """Eager path must not clone."""

def test_decode_forward_has_no_item_call():
    """.item() forbidden inside forward_decode."""

# 每个测试文件头部有明确的注释说明：
# - 这个测试保证什么不变性
# - 过去什么错误触发了这个测试
# - Agent 如果失败应该检查哪里
```

## 6. 细化后的脚本清单（每 Phase 至少一个，高发区多个）

```
scripts/
  # === Phase 1: 数值基元 (2 个脚本) ===
  test_phase1_kernel_wrappers.py        # 7 个 vLLM kernel 的随机 Tensor 正确性合约
  test_phase1_kernel_wrappers.sh        # 确认 vLLM import 路径 + flash_attn 可用性

  # === Phase 2: TP 通信 (2 个脚本) ===
  test_phase2_tp_communication.py       # all_reduce/all_gather shape + tp=2/4 一致性
  test_phase2_custom_ar_init.sh         # torchrun 拉起，确认 CustomAR 初始化不超时

  # === Phase 3: TP 线性层 (2 个脚本) ===
  test_phase3_tp_linear.py             # 4 种 Linear shape + double_shard_guard 合约
  test_phase3_tp_linear_tp4.py         # torchrun tp=4 下 Linear 输出一致性

  # === Phase 4: TP Embedding (2 个脚本) ===
  test_phase4_tp_embedding.py          # VocabParallel mask + LM Head gather 合约
  test_phase4_tp_embedding_tp4.py      # torchrun tp=4 下 Embedding 一致性

  # === Phase 5: Attention + KV Cache (3 个脚本，错误高发区) ===
  test_phase5_attention_init.py        # __init__ shape: num_heads/kv_heads/kv_block_size
  test_phase5_kv_cache_paged.py        # KV cache lazy alloc + slot_mapping + index_copy_
  test_phase5_flash_attn_prefill_decode.py  # flash_attn_varlen_func + flash_attn_with_kvcache

  # === Phase 6: MLP + Decoder Layer (4 个脚本，最高发区) ===
  test_phase6_mlp_forward.py           # gate_up_proj→silu_and_mul→down_proj shape
  test_phase6_residual_chain.py        # fused_add_rms_norm 本层 weight + residual 累积
  test_phase6_decode_forward_no_clone.py   # eager 路径无 clone（aten::copy_ 检查）
  test_phase6_layer_e2e_random_weights.py  # 随机权重下层完整 prefill+decode

  # === Phase 7: 权重加载 (3 个脚本) ===
  test_phase7_qwen_tp_config.py        # config.json → QwenTPConfig 动态读取
  test_phase7_hf_key_mapping.py        # QKV cat 顺序 + GateUp cat 顺序
  test_phase7_weight_loading.sh        # 真实 Qwen3-8B 权重加载 + per-rank 显存检查

  # === Phase 8: 框架外壳 (2 个脚本) ===
  test_phase8_sequence_scheduler.py    # Sequence 状态转移 + Scheduler 无 preempt
  test_phase8_sampler_tp.py            # TP 采样 rank0+broadcast 协议

  # === Phase 9: 引擎集成 (2 个脚本) ===
  test_phase9_llm_engine_init.py       # LLMEngine 路由 + _select_tp_backend
  test_phase9_generate_single_gpu.sh   # 单卡 generate() 端到端可读文本

  # === Phase 10: E2E 验收 (4 个脚本) ===
  test_phase10_greedy_align.sh         # temperature=0 字字对齐（核心验收底线）
  test_phase10_benchmark.sh            # TP=4 吞吐基准（≥54 tok/s）
  test_phase10_no_compile_check.sh     # Profiler 确认无 torch.compile/CUDA Graph 痕迹
  test_phase10_vs_vllm_compare.sh     # 新生成框架 vs vLLM TP=4 Qwen3 对比（含 CUDA Graph 开/关）

总计：26 个脚本（18 个 .py + 8 个 .sh）

### 6.1 Phase 10 vLLM 对比脚本详细规格

`test_phase10_vs_vllm_compare.sh` 是新生成框架的最终验收关卡。它独立拉起新框架和 vLLM，在同一批 GPU 上跑相同的 prompt 集合，输出对比报告。

```
脚本执行流程：
1. 检查 GPU 可用性（nvidia-smi 确认 4 卡空闲）
2. 启动新生成框架 OpenAI server（TP=4, PORT=9000）
3. 启动 vLLM server (TP=4, PORT=8001, CUDA Graph 关闭)
4. 对同一批 prompt 分别打 benchmark（REQUEST_RATE=1, MAX_CONCURRENCY=1）
5. 记录三组指标：新框架 / vLLM no-CUDA-Graph / vLLM CUDA-Graph
6. 输出对比表
7. 清理进程
```

参考实现：

| 参考源 | 作用 |
|--------|------|
| `run_compare_metainfer_vllm.sh` | meta-infer vs vLLM 对比流程（含 SKIP_VLLM/SKIP_MYENGINE 开关） |
| `ref_projects/vllm/examples/offline_inference/simple_profiling.py` | vLLM LLM() 实例化 + SamplingParams + profiler_config |
| `ref_projects/vllm/benchmarks/benchmark_serving_structured_output.py` | vLLM 官方 serving benchmark 脚本 |

对比指标：

| 指标 | 新框架目标 | vLLM 参考（CUDA Graph 开） | vLLM 参考（CUDA Graph 关） |
|------|-----------|--------------------------|--------------------------|
| Output Throughput (tok/s) | ≥ 54 | ~**166.8** | ~**52** |
| Mean TTFT (ms) | < 50 | ~30 | ~**50** |
| GPU Self CUDA / step (ms) | ≤ **66** | ~15 | ~**60** |
| GPU Mem / rank (GB) | ~6 | ~6 | ~6 |

*注：vLLM 参考值为 Qwen3-8B TP=4 B=1 实测值。新框架为 nocompile 纯 Eager 模式。*

审查子代理额外检查项：
- 脚本是否正确处理了 vLLM serve 的启动等待（health check 轮询）
- 脚本是否在对比失败时输出可操作的差异报告（而非仅 "FAILED"）
- 环境变量 VLLM_GPU_MEM_UTIL 是否可配置
```

## 7. 审查子代理门禁（superpowers CLAUDE.md 驱动的硬约束）

每完成一个测试脚本的编写，必须启动一个**审查子代理**进行 10 条检查。其中 6 条（S1-S6）映射自 `superpowers/CLAUDE.md`，4 条（G1-G4）是工程级别的通用检查。

```
审查子代理检查清单（10 条，必须全部通过）：

[S1] 脚本是否引用了真实发生的 bug？（superpowers: No fabricated content）
     → 检查文件头是否有 Why/When/What failure 三段
     → 检查引用的 commit/issue/session 是否可查证

[S2] 脚本是否防止的是实际经历过的问题？（superpowers: No speculative fixes）
     → 检查测试逻辑是否对应 V17 审计报告中的 FG/OW 项，或物理 tracing 证据
     → 无对应真实问题的测试函数标记为 "speculative — reject"

[S3] 脚本是否只锁定一个不变性？（superpowers: One problem per PR）
     → 如果一个 .py 文件有多个 test_ 函数，每个函数必须聚焦于同一 Phase 的不同侧面
     → 跨 Phase 的断言出现在同一文件 → reject，需拆分

[S4] 脚本是否已经过人类的 diff 审查确认？（superpowers: Human review required）
     → gate 条件：人类在 commit 消息中确认 "reviewed: scripts/test_phaseX_xxx.py"

[S5] 脚本是否可重复、确定性运行？（superpowers: Skill = executable code）
     → 脚本在相同环境/相同 commit 下跑 3 次，3 次结果必须相同
     → 随机 seed 不固定导致偶尔失败 → reject，必须 torch.manual_seed(42)

[S6] 脚本文件头是否完整？（superpowers: Describe problem, not change）
     → 检查 # Why / # When / # What failure / # Superpowers gate 四行是否全部填写
     → 缺任何一行 → reject

[G7] 脚本能否独立运行（不依赖未提交的外部文件）
[G8] 错误信息是否足够指导 Agent 定位问题（不能只写 assert False）
[G9] 边界条件是否覆盖（空输入、极端 shape、tp=1 vs tp=4）
[G10] .sh 脚本是否正确设置了 PYTHONPATH、CUDA_VISIBLE_DEVICES 等环境变量
```

审查结果为 ❌ 的脚本不得进入 `scripts/` 目录。必须修改后重新审查通过全部 10 条。

### 7.1 Adversarial Pressure Testing（对抗压力测试）

每完成一个 Phase 的全部脚本后，必须进行一次对抗压力测试：

1. 故意在实现代码中引入对应的错误（如把 `self.input_layernorm.weight` 改为 `next_layer.input_layernorm.weight`）
2. 跑该 Phase 的测试脚本
3. 确认测试**确实失败**（而非误报 PASS）
4. 恢复正确代码，确认测试**确实通过**
5. 记录测试结果

如果某测试脚本在错误代码下仍然 PASS，该脚本**无效**，必须重写。

## 8. AGENT_SKILL.md 的 TDD 策略修改

当前 §6.1 说的是：
> "先写算子单测，实例化单层算子，喂随机 Tensor..."

这仍然是让 Agent 自己写测试。需要改为：

```markdown
### 6.1 测试即规范（Tests as Specification）

**铁律：`scripts/` 下的文件是先验知识，Agent 只读不写。**

1. 在写任何实现代码前，先跑对应 Phase 的测试脚本 → 确认 RED（预期失败）
2. 写最小实现代码 → 跑测试 → 确认 GREEN
3. 如果测试不通过：检查实现，**不能修改 scripts/ 下的任何文件**
4. 测试脚本可以是 .py（pytest）或 .sh（bash），Agent 按 Phase 顺序执行
5. Agent 可以在 `tests/`（非 scripts/）下补充额外测试，但不能替代 scripts/ 测试
6. 每个 Phase 通过标志：该 Phase 下**所有** scripts/ 脚本全部 PASS
```

## 9. inference_blueprint.json 需要加什么

每个 Phase 增加 `test_interface_contracts` 字段：

```json
"phase_6_mlp_decoder_layer": {
  "test_interface_contracts": {
    "scripts": [
      {
        "file": "scripts/test_phase6_residual_chain.py",
        "format": "pytest",
        "guards_against": "Agent 使用跨层 weight 或 fused_add_rms_norm 参数错误",
        "past_error_triggered_by": "kernel_replacement_plan.md §9.2 跨层 weight 文档错误"
      },
      {
        "file": "scripts/test_phase6_decode_forward_no_clone.py",
        "format": "pytest",
        "guards_against": "Eager 路径误留 CUDA Graph 的 clone()，导致 ~15% 吞吐回退",
        "past_error_triggered_by": "Stage C clone 回归"
      }
    ],
    "review_gate": "每个 script 写入前必须通过 code-review 子代理审查（见 AGENT_SKILL.md §6.1）"
  }
}
```

## 10. 细化后的实施计划

| 步骤 | 内容 | 产出 | 工作量 |
|------|------|------|--------|
| 1 | 创建 `scripts/` 目录和 `scripts/__init__.py` | 目录结构 | 2 分钟 |
| 2 | 写 Phase 6 的 4 个脚本（最高发区优先） | 4 个文件 | 25 分钟 |
| 3 | 审查子代理检查 Phase 6 脚本 | 审查报告 | 5 分钟/脚本 |
| 4 | 写 Phase 1-3 的 6 个脚本 | 6 个文件 | 25 分钟 |
| 5 | 审查子代理检查 Phase 1-3 脚本 | 审查报告 | 5 分钟/脚本 |
| 6 | 写 Phase 5 的 3 个脚本（KV cache 高发区） | 3 个文件 | 20 分钟 |
| 7 | 审查子代理检查 Phase 5 脚本 | 审查报告 | 5 分钟/脚本 |
| 8 | 写 Phase 4、7-9 的 10 个脚本 | 10 个文件 | 30 分钟 |
| 9 | 审查子代理检查 Phase 4、7-9 脚本 | 审查报告 | 3 分钟/脚本 |
| 10 | 写 Phase 10 的 4 个脚本（含 vLLM 对比） | 4 个文件 | 25 分钟 |
| 11 | 审查子代理检查 Phase 10 脚本 | 审查报告 | 4 分钟/脚本 |
| 12 | 对抗压力测试：每 Phase 故意写错代码验证脚本确实能捕获 | 压力测试报告 | 30 分钟 |
| 13 | 更新 `inference_blueprint.json`：每 Phase 加 `test_interface_contracts` + superpowers gate 引用 | JSON 增量 | 15 分钟 |
| 14 | 更新 `AGENT_SKILL.md`：重写 §6.1 + 新增审查子代理门禁（含 superpowers CLAUDE.md 6 条映射） | MD 增量 | 15 分钟 |
| 15 | 用已知正确实现验证全部 26 个脚本通过 | 全绿报告 | 15 分钟 |
| 16 | `git add scripts/` 并 commit | commit | 2 分钟 |

预计总工作量：约 5 小时（含审查子代理来回 + 对抗压力测试）

## 11. 最关键的设计决策

**Q: 脚本用 .py 还是 .sh？**
A: 不限制。单元级/集成级优先 .py（pytest 精确断言），端到端级优先 .sh（环境变量 + 多命令串联）。同 Phase 可以既有 .py 又有 .sh。

**Q: 测试文件谁来写？**
A: 人写初版，审查子代理把关，人做最终 commit。后续如果发现新的重复性错误，人新增脚本"锁住"正确行为。Agent 永远不能修改 `scripts/` 下的文件。

**Q: 审查子代理怎么工作？**
A: 每写一个脚本 → 启动 code-review 子代理 → 检查 6 条门禁 → 通过则进入 scripts/，不通过则修改后重审。审查子代理的结论人可以看到并做最终决策。

**Q: Agent 发现脚本"不合理"怎么办？**
A: Agent 报告给人，人来决定是否修改脚本。Agent 不能自己改脚本来"通过"。

**Q: 这和 subagent 对抗结构怎么配合？**
A: 对抗结构中的 spec reviewer 现在是**跑脚本**而非读 JSON。脚本全绿 = spec 符合。同时，spec reviewer 在每 Phase 完成后自动触发，不需要人提醒。

**Q: Agent 不知道怎么写测试暴露接口怎么办？**
A: `scripts/` 下的文件本身就是模板。Agent 如果需要补充测试（在 `tests/` 下），可以直接参考 scripts/ 文件的格式：同样的 fixture 模式、同样的 assert 风格。
