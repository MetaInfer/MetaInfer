# Phase 9-10：引擎集成 + E2E 验收

## 触发词

`/phase9-10`

## 角色

你是主 Agent。按 CLAUDE.md 的 spawn 协议，依次完成 Phase 9 → Phase 10，每 Phase 独立走完整串行路径。这是最后两个 Phase——Phase 9 胶合引擎，Phase 10 最终验收。

---

## Phase 9：引擎集成

### 任务

- LLMEngine.__init__：(1) torch.cuda.set_device (2) _select_tp_backend 自动路由 (3) 创建 Runner (4) eos_token_id (5) _estimate_kv_blocks() (6) KVMemoryPool (7) Scheduler
- LLMEngine.generate()：_enqueue → while-loop(schedule→run→postprocess→finish_check) → decode → return
- LLMEngine.step()：scheduler.schedule → runner.run → postprocess → finish_check
- begin_generation() / has_unfinished_requests() / get_generation_outputs()
- 集成桥接：block_size 注入（TP→256, HF→16）、num_free 来源路由、BlockManager 降级

### Phase 9 Scripts

`test_phase9_llm_engine_init.py` + `test_phase9_generate_single_gpu.sh`

### 知识映射

- Blueprint：`components[6] LLMEngine` → `components[3] ModelRunner` → `scheduler_tp_runner_bridge`（CRITICAL-01）
- ref_docs：`01_architecture.md`, `07_request_lifecycle.md`
- ref_code：`ref_projects/nano-vllm/nanovllm/engine/llm_engine.py`（⚠️ 删除 use_cache=False）

---

## Phase 10：E2E 验收

### 任务

- openai_tp_server.py：ThreadingHTTPServer + /v1/completions
- TP 同步：non-rank0 while-loop(broadcast_obj) + rank0 HTTP handler + threading.Lock
- streaming(SSE) + non-streaming(JSON) — 详见 `inference_blueprint.json > OpenAITPServer.streaming.critical_constraints`

### Phase 10 Scripts

| 脚本 | 门禁 |
|------|------|
| `test_phase10_greedy_align.sh` | temperature=0 greedy 字字对齐基线 |
| `test_phase10_benchmark.sh` | 吞吐对齐 physical trace 基线 |
| `test_phase10_no_compile_check.sh` | cudaGraphLaunch=0、无 torch.compile |
| `test_phase10_vs_vllm_compare.sh` | vs vLLM 对比流程可运行 |

### L3 强制证据（verif 必须采集）

- profiler trace：无 CompiledFunction、无 Inductor kernel、cudaGraphLaunch=0
- HCU/VRAM：4 卡 VRAM% 同量级、HCU% > 0
- **无证据 = 假推理 = 验收失败**

### 知识映射

- Blueprint：`todo_generation_playbook.phase_10_e2e_acceptance` → `runtime_acceptance_layer.logic_constraints`
- 乱码诊断：AGENT_SKILL.md §7.4.C（6 症状 → Phase → FM/Bug → fix）
- 自愈先例：`notebooks-cn/07_improvementPlan/bugfix.md`

---

## 关键约束

- Phase 9 verif L2：重跑 Phase 1-8 全部 22 个脚本
- Phase 10 verif L3：**强制** profiler + HCU 证据
- Phase 10 遇到乱码 → 按 AGENT_SKILL.md §7.4.C 回溯流程定位源 Phase
- 主 Agent 禁止降级子代理结论
