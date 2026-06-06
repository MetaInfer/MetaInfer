# Skill: Phase 9-10 编码（引擎集成 + E2E 验收）

## 触发词

`/inference:phase9-10` 或 `/phase9-10`

## 概述

这是最后两个 Phase——Phase 9 将 Scheduler 和 ModelRunner 胶合为 LLMEngine，Phase 10 做最终验收。

---

## 环境

- 模型权重: `${MODEL_DIR}`
- Conda: `${PYTHON_PATH}/python`
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
- **SSE 响应必须** `Connection: close` + `self.close_connection = True`（SSE 无 Content-Length，keep-alive 导致客户端永久 hang）
- **Non-rank0 worker 必须注册 SIGTERM handler → os._exit(0)**（主线程阻塞在 NCCL collective 时 Python 信号被延迟，os._exit 直接终止进程）

### Phase 10 Scripts（含硬性验收底线）

| Phase | 必须全部 PASS 的 scripts/ |
|-------|--------------------------|
| Phase 10 | test_phase10_greedy_align.sh + test_phase10_benchmark.sh + test_phase10_no_compile_check.sh + test_phase10_vs_vllm_compare.sh |

### Phase 10 硬性验收指标（不达标不能声称完成）

| 指标 | 目标 |
|------|------|
| Greedy decode | temperature=0.0 输出正确对齐 |
| Output throughput | > 50 tok/s |
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
