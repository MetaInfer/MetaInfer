# MetaInfer → DSH agent（dcu-kernel-auto-opt）

把 dcu-kernel-auto-opt 任务的 agent 执行链路从 Claude Code 换成 **DeepSeek
Harness（DSH）**：主 agent（kernel coordinator）与子 agent（worker / repair /
synthesis）全部由 DSH Python SDK 驱动，MetaInfer 的编排与验证 harness 语义
不变。

## 原理

MetaInfer 的 `SubAgentManager` 把每个 agent 当作一个 CLI 进程调用：

```
<claude_bin> -p --output-format stream-json --input-format text --verbose \
  --permission-mode bypassPermissions --add-dir <workdir> [--add-dir <extra>...] \
  [--model <m>] [--effort <e>] [--resume <sid> | --session-id <sid>] [extra...]
```

prompt 从 stdin 传入，stdout 输出逐行 stream-json 事件
（`system` / `assistant` / `result`），退出码 0 + `result` 事件 = 成功。

本目录提供：

| 文件 | 作用 |
|---|---|
| `dsh_agent.py` | ccb 兼容的 CLI 包装器：解析上述参数、stdin 读 prompt、经 DSH Python SDK 跑一个 DSH agent、把结果重新输出为 stream-json 事件流 |
| `cordis.yml` | DSH runtime 的自定义组合：agent spine + tool-fs + tool-bash + tool-subagent + tool-todo + skills + 会话持久化 |
| `run_dsh_task.sh` | 便捷启动脚本：把 `METAINFER_CLAUDE_BIN` 指向本包装器后直接跑 orchestrator CLI |
| `tests/smoke_sdk.py` | SDK 冒烟测试（runtime 启动 / 模型调用 / 会话 resume / 文件工具） |

## 本机接入方式（worker29）

WebUI new-task 表单中 dcu-kernel-auto-opt 增加 **Agent framework** 选择：

- `ccb` → Claude Code（模型 `Sonnet` / `Opus`）
- `dsh` → DeepSeek Harness（模型 `deepseek-v4-flash`，即
  `deepseek/deepseek-v4-flash-0731`）

orchestrator CLI 在 `agent_framework=dsh` 时自动把 `claude_bin` 指向本目录的
`dsh_agent.py`（`--claude-bin` 显式传入时优先），无需改动 MetaInfer 共享代码。

命令行直接运行（headless）：

```bash
python3 -m metainfer.tasks.dcu_kernel_auto_opt.orchestrator.cli run \
  <requirements.json> --state-dir ... --workspace-dir ...
```

requirements.json 中 `answers.agent_framework = "dsh"` 即可。

## 模型端点（本机）

本机 DSH profile（`~/.dsh/settings.yaml`）使用 TokenHub 网关，dev-checkout
runtime 只内置 `deepseek-official` adapter（`llm-deepseek`），它通过
`DEEPSEEK_BASE_URL` 环境变量覆盖端点——所以用官方 adapter 指向同一网关：

- provider：`deepseek-official`（runtime adapter；TokenHub 经 base_url 覆盖）
- baseURL：`https://tokenhub.tencentmaas.com/plan/v3`
- 模型：`deepseek/deepseek-v4-flash-0731`（默认）
- API key：`~/.dsh/.credentials.yaml` 的 `TENCENT_API_KEY`

`dsh_agent.py` 的环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DSH_AGENT_PROVIDER` | `deepseek-official` | DSH runtime provider（dev runtime 仅内置此 adapter） |
| `DSH_AGENT_MODEL` | `deepseek/deepseek-v4-flash-0731` | 模型覆盖 |
| `DSH_AGENT_BASE_URL` | `https://tokenhub.tencentmaas.com/plan/v3` | 模型端点（`DEEPSEEK_BASE_URL` 优先） |
| `TENCENT_API_KEY` / `DEEPSEEK_API_KEY` | 凭据文件兜底 | API key |
| `DSH_AGENT_CORDIS` | 本目录 `cordis.yml` | 自定义组合路径 |
| `DSH_AGENT_SESSION_ROOT` | `{最后一个 --add-dir}/.dsh-sessions` | 会话 JSONL 持久化根（跨迭代 resume 需要稳定路径） |
| `DSH_AGENT_MAX_TOKENS` | 65536 | 每次请求输出上限 |
| `DSH_AGENT_DEBUG` | — | 保留 runtime 日志 |

## 会话连续性

`SubAgentManager` 用 `--session-id`（首轮）与 `--resume <id>`（后续轮）延续
agent 会话；DSH runtime 按 session id 把会话 JSONL 持久化在
`DSH_AGENT_SESSION_ROOT` 下，同一个 id 再次 `run()` 即恢复上下文。默认
session root 取 orchestrator 传入的 workspace_dir（`--add-dir` 的最后一个），
跨迭代稳定。

**已知限制（0.1.0rc6 runtime）**：小会话 resume 可用（SDK 冒烟测试验证），但
大会话（如一次完整 kernel 编辑迭代，~400KB zstd）resume 时 runtime 报
`corrupt session log` / `unsupported flat-file layout` 并返回空回复。
`dsh_agent.py` 已做兜底：resume 失败（finish_reason != completed/max-tokens）
时自动改用全新 session 重跑一次（prompt 自带全部上下文，效果等价），并把最终
session id 透出给编排器。代价是每个 resume 失败的迭代多一次快速失败尝试，不影响
正确性。

## 已知注意点

- `dsh_agent.py` 需要有执行位（`chmod 755`），否则 `SubAgentManager` 的
  Popen 会报 `PermissionError`。
- SDK 为 dev checkout（`/root/deepseek-harness/python/sdk`），provider
  `tencent` + TokenHub 端点在 runtime 侧解析；改 `cordis.yml` 时以 runtime
  报错为准逐步对齐。
- 迭代循环在 agent 失败时会把 `attempt_limit` 递增（`w8a8_pipeline.py` 的
  `replacement_iteration` 逻辑），失败多会显著拉长任务；resume 回退已消除
  空回复型失败。
- 最终验证（serial_validate）对 GPU 争抢敏感：本机其他容器跑重负载时会把
  benchmark 拉高数倍，导致 `performance regressed` 误报。clean 复测（GPU
  空闲窗口）同一 prebuilt object 可测到与 worker best 一致甚至更优的延迟。
