# Phase Evolve 任务卡：知识库进化

## 触发词

`/evolve <model_id>` — 启动知识库进化路径，让系统学会生成新模型的推理框架。

## 触发场景

| 场景 | 说明 |
|------|------|
| 用户手动触发 | `/evolve deepseek-v3` — 直接指定模型 |
| 主 Agent 自动触发 | master 循环检测到模型不在知识库覆盖范围 |
| 首次生成全部失败 | `/phase-all` 的 one_pass_rate < 阈值且目标模型未知 |

## 构建目标

让知识库（`notebooks-cn/`）通过进化循环，学会独立（不依赖开源代码）生成新模型的推理框架。

## 进化循环

```
无开源尝试 → 失败 → 开源辅助 → 成功 → 固化知识 → 无开源重验 → 成功/继续
```

## 知识映射

| 进化组件 | 契约文件 | 关联文档 |
|----------|---------|----------|
| 模型发现 | `00_contracts/model_specs.md` | `02_model_specifics/` |
| 架构差异分析 | `00_contracts/attention_kv_contracts.md` + `mlp_decoder_contracts.md` | `03_operators/` |
| TP 策略适配 | `00_contracts/tp_communication_contracts.md` + `tp_linear_contracts.md` | `04_parallel_strategies/` |
| 权重映射 | `00_contracts/weight_loading_contracts.md` | — |
| 引擎集成 | `00_contracts/engine_contracts.md` | `01_framework_design/` |

## 脚本门禁

进化成功标准（所有 Phase scripts 在企业模式下通过）：

| Phase | 门禁脚本 |
|-------|---------|
| Phase 1 | `test_phase1_kernel_wrappers.py` + `.sh` |
| Phase 2 | `test_phase2_tp_communication.py` + `test_phase2_custom_ar_init.sh` |
| Phase 3 | `test_phase3_tp_linear.py` + `test_phase3_tp_linear_tp4.py` |
| Phase 4 | `test_phase4_tp_embedding.py` + `test_phase4_tp_embedding_tp4.py` |
| Phase 5 | `test_phase5_attention_init.py` + `test_phase5_kv_cache_paged.py` + `test_phase5_flash_attn_prefill_decode.py` |
| Phase 6 | `test_phase6_mlp_forward.py` + `test_phase6_residual_chain.py` + `test_phase6_decode_forward_no_clone.py` + `test_phase6_layer_e2e_random_weights.py` |
| Phase 7 | `test_phase7_qwen_tp_config.py` + `test_phase7_hf_key_mapping.py` + `test_phase7_weight_loading.sh` |
| Phase 8 | `test_phase8_sequence_scheduler.py` + `test_phase8_sampler_tp.py` |
| Phase 9 | `test_phase9_llm_engine_init.py` + `test_phase9_generate_single_gpu.sh` |
| Phase 10 | `test_phase10_greedy_align.sh` + `test_phase10_benchmark.sh` + `test_phase10_no_compile_check.sh` + `test_phase10_vs_vllm_compare.sh` |
| Phase 11 | `test_phase11_throughput.py` + `test_phase11_profiler.sh` |

## 高发错误

1. **Explorer 搜索偏差**：论文信息与 HF config.json 不一致时，以 config.json 为准
2. **开源代码版本不匹配**：vLLM/SGLang 的实现可能针对特定版本，需验证与目标模型版本的兼容性
3. **知识固化不充分**：Consolidator 写入了"从 config.json 可推断"的参数而没有写入"不开源无法发现"的关键差异
4. **过度依赖开源代码**：implementer 在有开源参考时直接复制实现模式，导致无开源重验时失败

## 开关控制

| 开关状态 | Explorer 可访问 | Implementer 可访问 |
|----------|----------------|-------------------|
| OFF | WebSearch + WebFetch + notebooks-cn/ | 仅 notebooks-cn/ + exploration_report.md |
| ON | 上述 + knowledge/vllm/ + knowledge/sglang/ | 上述 + exploration_report.md（含开源分析） |

## 工作流（CLAUDE.md 中的进化编排器）

进化采用 impl→spec→verify 三层对抗串行（首次构建也走完整串行路径）。主 Agent 通过 `evolution/EVOLUTION.md` 编排 Explorer → Implementer → Consolidator → Verify 循环。

## 目录结构

```
evolution/
├── EVOLUTION.md              ← 进化编排器行为定义
├── state.json                ← 进化状态
├── decision-log.jsonl        ← 裁决日志
├── strategies/               ← 策略文件
├── results/<EVO_ID>/         ← 每轮结果
└── scripts/
    └── call-evo-agent.sh     ← 子 agent 启动脚本
```
