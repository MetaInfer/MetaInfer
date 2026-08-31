# DCU Kernel Auto-Opt 工作流速查（给接手 AI 的快速上手）

> 维护说明：本文档由人工/AI 维护，内容基于 2026-08 的实际代码与现场任务。
> 代码演进后请同步更新；涉及版本、容器名、具体数值的地方会标注“以实际为准”。
> 遇到不确定的信息，先查代码/任务现场，不要凭记忆下结论。

## 0. 这个功能是什么

MetaInfer 的一个 task 插件：在一个节点（worker29，4×K500SM_AI/gfx928）上用多 Agent +
4 GPU 对 DCU 算子做“生成/优化 → 可信正确性+性能基准 → 串行回归验证”的闭环。
当前主算子是 **INT8 W8A8 GEMM**（DeepSeek-V4 TP4/TP8 的六个逻辑 GEMM），内核语言 HIP C++（DUMMA Tensor Core）。

一句话流程：`新建任务表单 → 解析配置并冻结 API 契约 → 固定 Triton Graph 基线 → 每个 GPU 一个 Agent worker 迭代优化 → 合并技能 → 串行验证全部 API shape（含回归） → 出报告`。

## 1. 目录地图（先认路）

```
metainfer/tasks/dcu_kernel_auto_opt/
├── form.yaml                      # 新建任务表单定义（所有可配置项）
├── api/                           # 接入的算子 API 文件（plugin 本地权威契约）
│   └── int8w8a8gemm/int8_w8a8_gemm_api.py
├── variant/                       # 参考变体 HIP 代码
│   └── w8a8_gemm_variants.hip
├── assets/
│   ├── smoke_harness.cpp          # infra smoke 模式的向量 kernel
│   ├── w8a8_baseline/             # W8A8 基线扩展模板（bindings/w8a8_gemm_hip/profile_pmc.sh…）
│   └── w8a8_bench.py              # 可信 harness（任务仓库里会拷贝一份）
├── orchestrator/
│   ├── config.py                  # load_config：解析表单 + shape/GPU 分配校验
│   ├── api_contracts.py           # 解析权威 API、冻结快照、default_optimization_shapes
│   ├── w8a8_baselines.py          # 固定 Triton Graph 基线表（缺条目直接报错）
│   ├── w8a8_pipeline.py           # 真实 worker 生命周期：W8A8Runner/benchmark/PMC/验收
│   ├── gen_and_opt_pipeline.py    # Generate 模式 + 最终 synthesis + 串行验证
│   ├── real_pipeline.py           # 真实（非 mock）流程骨架
│   ├── worker.py / phases.py      # mock worker / 状态机阶段
│   ├── guidance.py / skill_store.py / result_store.py / gpu_binding.py / pmc_profile.py
│   ├── cli.py                     # 命令行入口：dcu-kernel-auto-opt run requirements.json ...
│   └── adapters/                  # mock kernel adapter
├── server/                        # Web 插件路由（summary/iterations/guidance…）
├── static/                        # 前端（dkao-shape-input.js 里硬编码了 shape 常量！）
├── bridge/                        # agent bridge（控制面↔agent）
└── tests/                         # 单元测试（改行为后必须跑）
```

## 2. 端到端工作流

阶段状态机见 `orchestrator/phases.py`：
`prepare → generate_kernel_repo → baseline → parallel_explore → skill_synthesis → serial_validate → report → finished`。

### 2.1 新建任务（表单字段，见 form.yaml）

关键字段：`operator`（Quantized GEMM）、`kernel_language`（HIP C++）、`target_hardware`（K500SM_AI/gfx928）、
`dtype`（INT8 W8A8）、`claude_model`（Opus/Sonnet）、`execution_mode`（Mock / Real INT8 W8A8 GEMM /
Generate & optimize / Infra smoke）、`target_repo_path`、`shape_assignment_mode`（AI automatic / Manual by GPU）、
`shape_scope`（All API shapes / Selected shapes only）、`shape_config`、`max_iterations`、`minimum_improvement_percent`、`extra_notes`。

`shape_config` 是 YAML（config.py 解析，最多 4 个 worker、每 GPU 一个、shape 必须恰好分配一次）：

```yaml
shapes:
  - {id: tp4_wqkv_a_m4096, tp_size: 4, operator: wqkv_a, M: 4096, N: 1536, K: 4096}
  # ... 其余 shape
assignments:
  worker_0: {gpu: 0, shapes: [tp4_wqkv_a_m4096]}
```

提交后 `load_config` 会调用 `api_contracts.validate_contract_shapes`，用**冻结契约**校验每个 shape 的
M/N/K 是否合法（M 范围、K%32==0、N%16==0、(K,N) 是否在 TP4/TP8 表内）。

### 2.2 固定接口（先读这三个文件）

1. **权威 API 契约**：`metainfer/tasks/dcu_kernel_auto_opt/api/int8w8a8gemm/int8_w8a8_gemm_api.py`
   （orchestrator 从这里 resolve；`METAINFER_OPERATOR_API_ROOT` 可覆盖，测试用临时目录走覆盖路径）。
   参考变体 HIP 代码放在 `metainfer/tasks/dcu_kernel_auto_opt/variant/w8a8_gemm_variants.hip`。
2. **任务内冻结快照**：任务仓库 `kernel-repos/<repo>/int8_w8a8_gemm_api.py`，其 sha256 记录在
   `scaffold_manifest.json` 的 `control_plane_files` 中，`gen_and_opt_pipeline._task_local_api_contract`
   每次运行都校验 digest。**改权威 API 只影响新任务**（控制面会重新 staging 新 digest）；**不要手改旧任务仓库里的快照**。
3. **固定调用面**：`w8a8_gemm_out(x_q[M,K] int8, packed_weight, x_scale[M,1] fp32,
   packed_weight_scale[N,1], out[M,N] bf16, workspace) -> out`，底层是 `torch.ops.zth_w8a8.gemm_out`；
   可选 `pack_weight`。语义：

   `out[m,n] = bf16( int32_dot(x_q[m,:], weight[:,n]) * x_scale[m] * weight_scale[n] )`

   计时区只包含 `w8a8_gemm_out`；`prepare_weight`/`allocate_workspace` 在 Graph capture 之前、不计时。

### 2.3 Baseline（固定表 + 可自测）

`w8a8_baselines.py::fixed_triton_graph_baseline(shape_id, shape)` 按 `(tp, M, N, K)` 查表，
**查不到就抛 ValueError**（baseline 阶段直接失败），所以新 shape 必须先补表。
表值是 Triton Graph 基线（µs），TP4 M=4096 条目是 2026-08-06 在 worker29 用
`baseline/int8_utils.py`（lmslim）实测的（graph replay median：wqkv_a 13247、wq_b 20590、
wo_b 19882、gate_up 8790、down 5546；wq_b 与 indexer.wq_b 共用一条）。

要自己测 Triton baseline：用 `matmul_int8`（即 SGLang/lmslim 实际调用路径），M>1024 默认 config
是 `BM256/BN256/BK64/GROUP8/SPLIT_K1/warps8`，GPU event、预分配 out（排除分配）、
热缓存协议建议 `warmups=10, samples=20, launches_per_sample=5`；可参考
`zth_infer/baseline/bench_triton_tp4_m4096.py`。

### 2.4 Parallel explore（worker 生命周期，w8a8_pipeline.py）

- 每个 assignment 一个 worker（`worker_N ↔ GPU N`，最多 4 个）；每个 worker 负责若干 shape。
- **Agent 每轮只能改 `csrc/w8a8_gemm_hip.hip` 和 `proposal.json`**（控制面拥有其余文件）。
- 阶段：bootstrap（iteration 0，正确性优先；大 M 要求直接上 DUMMA tile kernel，标量只做
  unmatched/small-M fallback）→ 多轮迭代。
- 每轮：控制面用 `W8A8Runner.benchmark` 跑 `w8a8_bench.py`
  （CPU int64 exact reference + CUDA Graph replay 计时，median/P90）→
  `evaluate_candidate_acceptance`（median 提升 ≥1%（`ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT=1.0`）
  且无 P90 回归）→ 可选 `profile_pmc.sh`（hipprof PMC）。
- 接受的 artifact 落在 `workers/<w>/accepted/<shape>/`（`kernel.hip`、`kernel.cuda.o`、
  `manifest.json`、`gfx928.co`、`isa.txt`），`result.json` 里记录 `source_sha256/object_sha256`（后续会校验）。

### 2.5 Synthesis + Serial validate（gen_and_opt_pipeline._synthesize_final_candidate）

1. 用 worker 的 accepted objects 重建 `final/source`：`git restore` → 拷贝 object/HIP →
   `llvm-objcopy` 符号命名（`mi_<shape_id>_` 前缀）→ 渲染 `csrc/w8a8_dispatch.cpp` →
   写 `artifact_manifest.json`。
2. **dispatch 只按 (N,K) 路由、忽略 M**（见 w8a8_dispatch.cpp），所以回归 shape 也会走自定义 object。
3. 验证集合 = 6 个优化 shape（默认 protocol）+ 全部 API fallback shapes（`warmups=2, samples=3`）。
   优化 shape 还有 `final ≤ 1.05× worker best` 的性能门。
4. 任一 shape 正确性失败 → `Final prebuilt-object W8A8 validation failed: correctness failed for <shape>: <json>`。
5. 全部通过 → 提交 final worktree、cherry-pick 回 seed、写 `final_report.json`，任务 finished/success。

### 2.6 主 agent 与子 agent 的职责边界（严格区分，不是靠自觉）

**是严格区分的**，由“提示词 + agent 运行时 + 控制面代码”三层强制，任何越界都会被控制面判失败并重试：

| 角色 | 允许写 | 禁止 |
|---|---|---|
| 主协调 agent（Generate 阶段，`kernel_coordinator`） | 只写 `proposal.json` | 任何 HIP/C++/CUDA/Python backend、setup/build/test/benchmark/API 文件；不得编译或跑 harness |
| 子 worker agent（bootstrap + 每轮迭代） | 只创建/修改 `csrc/w8a8_gemm_hip.hip`（+ `proposal.json`） | 其它一切；API/backend/bindings/setup/harness 都是控制面所有 |
| Final synthesis agent（主侧合并） | 只改 `csrc/w8a8_gemm_hip.hip` + `proposal.json` | 其它一切；不得自己编译/跑 benchmark |

三层强制：

1. **提示词层**（`orchestrator/prompts.py`）：
   - `main_coordinator_prompt`：`Hard role boundary: You are not a kernel implementation agent... Write only proposal.json`；
   - `bootstrap_worker_prompt`：`You own and must create csrc/w8a8_gemm_hip.hip`；
   - synthesis prompt：`edit only csrc/w8a8_gemm_hip.hip and proposal.json`。
2. **运行时层**：所有 agent 都以
   `_SOURCE_ONLY_AGENT_ARGS = ["--disallowedTools", "Bash,Skill,WebFetch,WebSearch"]`
   启动 → 不能执行 shell/编译/benchmark/联网，只能用文件工具编辑源码。
3. **控制面层（最终裁决，`gen_and_opt_pipeline.py` / `w8a8_pipeline.py`）**：
   - 主协调返回后：API 契约与参考文件 digest 校验 + `git status --porcelain --untracked-files=all`，
     除 `proposal.json` 外任何改动 → `generate_role_violation`，该 attempt 失败；
   - worker bootstrap：控制面文件 digest 锁定 + `git status`，检出除 `csrc/w8a8_gemm_hip.hip`
     外改动 → `bootstrap Agent changed files outside its HIP ownership`；
   - worker 迭代：契约 digest + `git diff --name-only`，只允许 `csrc/w8a8_gemm_hip.hip`
     → `agent changed control-plane-owned extension infrastructure`。

控制面（非 agent）负责：任务 staging、API 快照、编译构建、`w8a8_bench` 正确性/性能、
hipprof PMC、验收门槛、final synthesis（符号命名 + dispatch 渲染）、串行验证、报告。
子 agent 的结果互不直接合并——合并由控制面/synthesis agent 完成。

## 3. 关键常量（2026-08 现状，改前先读 API 文件）

TP4 (K,N)：

| operator | K | N |
|---|---:|---:|
| wqkv_a | 4096 | 1536 |
| wq_b / indexer.wq_b | 1024 | 8192 |
| wo_b | 2048 | 4096 |
| shared_gate_up_proj | 4096 | 1024 |
| shared_down_proj | 512 | 4096 |

TP8 另有 wq_b/wo_b (1024,4096)、gate_up (4096,512)、down (256,4096)、indexer (1024,8192) 等。

- `DEFAULT_OPTIMIZATION_M_VALUES = (2, 16, 3072)`；`TP4_EXTRA_OPTIMIZATION_M_VALUES = (4096,)`
  （TP4 专属，2026-08-06 加）→ 默认 42 个 shape（TP4 24 + TP8 18）。
- `MIN_M=1, MAX_M=4096`；`WORKSPACE_BUDGET_BYTES=16MB`。
- **M=4096 时大部分 (N,K) 的 split-K workspace 容量为 0** → 大 M kernel 必须走 2D M-tile 路径，
  不能依赖 split-K workspace。

## 4. 环境（重要）

- **本机就是 worker29**（hostname=worker29，10.18.17.80，4×K500SM_AI/gfx928，DTK 26.04）。
- **MetaInfer 相关容器一律 `zth_meta`**：挂载 `zth_infer → /workspace`，PID1=`serve.py --port 8765`。
  GPU python 需要：
  `source /opt/dtk/env.sh` + `HIP_VISIBLE_DEVICES=<n>` + `PYTHONPATH=/workspace/MetaInfer`；
  任务工作区在容器里是 `/workspace/MetaInfer/nodes/worker29/workspaces/<task_id>/`。
- serving/benchmark 容器 `zth1-sglang-deepseek-v4-flash-tracing`（torch2.9/triton3.3/lmslim）
  只用于 Triton 基线测量等独立用途，**不要改它代码、不要在里面跑 MetaInfer 任务流程**。
  （容器名历史上变过，以 `docker ps` 为准。）
- GPU 绑定：`gpu_binding.py` 只设 `HIP_VISIBLE_DEVICES`，不要同时设 `ROCR_VISIBLE_DEVICES`。

## 5. 常见坑（都是真实踩过的）

1. **布局一致性（最容易翻车）**：object 的 `pack_weight` 如果重排了 B（例：down_proj (512,4096)
   打成 N-blocked 64 列），它内部**所有按 M 分发的路径必须读同一个布局**。dispatch 按 (N,K) 忽略 M，
   回归 M（2/16/3072）会走进通用路径；通用 DUMMA 若按 row-major 读 N-blocked B 就全错。
   修复示例：`dumma_eligible && (n,k)==(4096,512)` 时改用 N-blocked-aware 的 64x64 tile kernel，
   精确 M=4096 的 128x64 路径保持不变。
2. **worker 只验证分配到的 M** → 回归 bug 拖到 serial validate 才暴露。
   建议：worker 验收时对每个 (K,N) 额外跑 M=2/16/3072 正确性（M-sweep）。
3. **不要手改任务仓库的 API 快照**（digest mismatch）；改
   `metainfer/tasks/dcu_kernel_auto_opt/api/` 下的权威契约后新建任务。
4. **CPU int64 reference 在 M≥3072 很慢**（分钟级）→ 用 reference 缓存目录
   （`final/cache/references/`，key 是 m-n-k），w8a8_bench 用 `--reference-cache-dir` 命中。
5. **serial validate 失败后的恢复流程**（有先例 `zth_infer/recover_c9_serial_validate.py`、
   `recover_cc86e2b2_serial_validate.py`）：
   patch `workers/<w>/accepted/<shape>/kernel.hip` → 用 hipcc 重编 `kernel.cuda.o`
   （`-O3 --offload-arch=gfx928 -std=c++17 -fPIC -fno-gpu-rdc`）→ 更新
   `result.json`/`manifest.json` 里的 `source_sha256/object_sha256` → 写恢复驱动
   （`GenAndOptPipeline._phase(VALIDATE,...) → _synthesize_final_candidate → _phase(REPORT)`）。
   注意：容器内后台跑长任务用 `docker exec -d`，否则进程会随 exec 会话结束被杀。
6. **前端 shape 常量** `static/dkao-shape-input.js` 与 API 默认 shape 要同步
   （TP4 专属 M=4096 是分别维护的）。
7. **角色越界会自动判失败**：`generate_role_violation` / `changed files outside its HIP ownership`
   / `agent changed control-plane-owned extension infrastructure` 都是控制面在 agent 返回后
   用 git diff/digest 检出的。修复方式是恢复被改文件、重试该 attempt，**不要绕过检查**。

## 6. 新 AI 快速开始 checklist

1. 先读：`form.yaml` → `api_contracts.py` → 权威 `int8_w8a8_gemm_api.py` →
   `w8a8_pipeline.py`（W8A8Runner/worker loop）→ `gen_and_opt_pipeline.py`
   （`_synthesize_final_candidate`/serial validate）。
2. 跑测试：`python3 -m pytest metainfer/tasks/dcu_kernel_auto_opt/tests/`。
3. 看一个真实任务现场（已成功的例子：`dcu-kernel-auto-opt-cc86e2b2`）：
   `plan.json`（配置）、`shared_baseline/results.json`（基线）、`workers/*/accepted/*`（kernel）、
   `final/source/csrc/w8a8_dispatch.cpp`（路由）、`final_report.json`（结果）。
4. 内核级工作参考 skill：`int8-w8a8-quantized-gemm-optimization`、`dcu-kernel-tuning`、
   `hygon-dcu-kernel`、`hygon-gfx928-memory-isa`、`sglang-custom-kernel-integration`；
   环境/SSH/容器细节参考 `remote-dcu-env`。
5. 改动任何行为后：更新本文件相关段落 + 跑 tests + 用真实任务验证（优先在 zth_meta 里）。

## 7. 不确定性标注

- 版本号、容器名、基线数值均为 2026-08 观察值，动手前以实际代码/`docker ps`/现场数据为准。
- “恢复流程”是手工驱动（复用 `_synthesize_final_candidate`），不是 UI 一键重试；UI 是否提供重试以
  `server/routes.py` 实际实现为准。
- 本文档不替代 skill 里的性能调优细节（tile 选择、LDS、DUMMA API、hipprof 用法），那些看对应 skill。
