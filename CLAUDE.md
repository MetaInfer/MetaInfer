## 设计哲学

- 纯 Python，无编译依赖
- 文件系统即数据库；server 与 orchestrator 解耦，通过文件系统传递状态
- 多节点通过共享文件系统协同，每个节点只写自己的 `nodes/<node_id>/`

## 数据一致性：单一数据源（Single Source of Truth）

**文件系统即数据库**这一选择的代价是：失去数据库内置的一致性约束。任何"同一份事实"被存到多个文件，都会在并发/重启/部分写入下漂移，最终表现为难以排查的功能 bug。下列原则**强制执行**：

### 原则

1. **每份事实有且只有一个权威文件**（source of truth）。其他文件需要这份信息时，要么从权威源读取后派生（运行时计算），要么显式声明为"不可回读的历史快照"（写完只用于展示/审计，不再驱动逻辑）。
2. **严禁双向同步**。如果 A 是权威、B 是缓存，B 只能由 A 单向派生；绝不存在"B 改了回写 A"或"A、B 互相更新"的路径。
3. **冷重启路径必须重新走权威源**。任何在内存/进程里持有的状态（limit、pid、status）一旦进程退出就丢失；重启时只能从权威文件读，不能从 requirements.json / form 副本读"为了方便"。
4. **新增字段时先问"谁是权威"**。不要图省事把值复制到第二个文件——短期的省事会变成长期的 bug 工厂。
5. **历史快照必须标注**。某文件如果只是建任务时的表单记录（之后不再驱动运行时），必须在 schema 注释里写明："historical record, runtime reads from <other_file>"。

### 已确立的权威源（参考）

| 数据 | 权威源 | 历史快照 / 派生 |
|---|---|---|
| 预算阈值 | `token_budget.json::config.max_cost_usd` | `requirements.json::token_budget_max_cost_usd`（建任务时表单值，运行时不再读） |
| 预算累计 | `token_budget.json::totals` | `timeline.jsonl` 的 `token_usage` 事件（展示用，从权威派生） |
| 运行时状态 | `run.json`（phase / iteration / **finished / final_status**） | `registry.json`（**仅身份**：id/type/label/state_dir/workspace_dir/created_at/launcher。**绝不**缓存进程状态） |
| 任务规格 | `requirements.json`（task_type / form / label / created_at） | `registry.json::type/label`（缓存）；run.json 不再存 task_type |
| 进程存活 | OS 进程表（`/proc/<pid>`）+ `orchestrator.pid`（pid / started_at / finished_at / exit_hint） | `runtime.json::tasks.<id>`（仅 WebUI session 用 boot_id 标记归属，不作为状态查询源）；**registry.json 不存进程状态** |
| 进程死亡清理 | `launcher._reap_dead_pid_file()`（单一 reap 路径） | reconcile / liveness / kill 都**调它**，禁止另写 `_write_pid_file_finished` 这种只更新部分文件的简化版 |

### 已知反模式（**禁止**）

- **双写**：同一字段被两个文件各持一份，且都被运行时读取 → 必然漂移。
  - 已修复的例子：`requirements.json::token_budget_max_cost_usd` 和 `token_budget.json::config.max_cost_usd` 曾经都被读，导致 WebUI 调整预算后冷重启失效（commit 待补）。
  - 已修复的例子：`task_type` 曾经同时存在 requirements.json / run.json / registry.json，已从 run.json 移除（orchestrator 加载时 load_run 过滤未知字段，兼容旧文件）。
  - 已修复的例子：`created_at` 曾经同时存在 registry.json / run.json，已从 run.json 移除（registry.json::created_at 是唯一权威源）。
  - 已修复的例子：进程状态 (pid / started_at / finished_at) 曾经**三处存储** —— `orchestrator.pid` / `runtime.json::tasks.<id>` / `registry.json::tasks[]`。registry 那份名义上是"派生缓存",实际**没有任何派生函数**,reconcile / _reap_dead_pid_file / kill 各自选择性同步;`tasks.update_task` 里 `if v is None: continue` 还静默吞掉了 `pid=None` 的清除语义,导致死任务的 registry 永远显示 stale pid,liveness 用它做 pre-filter 时直接走错路。已**从 registry 移除 pid/started_at/finished_at 字段**,所有进程状态查询只走 `launcher.status()` 读 `orchestrator.pid`;旧 registry.json 通过 `_strip_legacy` 兼容。
- **多条 reap 路径效果不一致**：reconcile 原来用自己的 `_write_pid_file_finished`(只碰 orchestrator.pid),而 liveness 用 `launcher._reap_dead_pid_file`(还会刷 run.json + 写 timeline)。两条路径 → 同样的死亡事件,UI 拿到的信号不一致。**任何"清理死亡任务"的代码都必须调 `launcher._reap_dead_pid_file`**,禁止另写简化版。
- **构造函数参数压过文件**：构造函数从 A 文件读值传入，`_load()` 看到"非 None"就跳过 B 文件——这等价于把 A 钉死为权威。正确做法是构造函数只传"env override"，文件值由 `_load()` 单独决定。
- **多 task 包复制同一份解析逻辑**：每个 task orchestrator 自己实现一遍 cascade → 修一个 bug 要改 N 处。共享逻辑下沉到 `metainfer/orchestrator/` 公共层。
- **字段别名 + 多 reader 各写一份 fallback 链**：例如 requirements.json 曾经既支持扁平 `target_model` 又支持嵌套 `answers.target_model` / `form.target_model`，每个 reader 自己写 `req.get("x") or (req.get("answers") or {}).get("x")` —— 12+ 处复制，每处 null 处理略有不同。已加 `metainfer.orchestrator.requirements.req_field()` 统一读取，所有 task 包的读取都应走这个 helper。

### requirements.json 扁平化规约

WebUI 的 `create_task` 把表单 answers **扁平展开**到顶层（`{"task_id":..., "target_model":..., "max_iterations":"50", ...}`），没有 `answers` 或 `form` 子键。

- **写**：只写扁平。新代码不要在 requirements.json 里塞 `answers` / `form` 子字典。
- **读**：用 `metainfer.orchestrator.requirements.req_field(req, key)` / `req_field_int` / `req_field_float`。helper 内部保留对历史嵌套形式的兼容（旧文件、test fixture），但 production 路径只走扁平。
- **新加字段**：在 task 的 `form.yaml` 里声明 → WebUI 自动写入扁平顶层 → reader 用 `req_field` 读。不需要改 requirements.json 的 schema 文档。

### Code review 检查清单

提交前自问：
- [ ] 我新增/修改的字段，是否已经有别的文件存了？如果是，谁是权威？
- [ ] 我的代码读这个字段时，读的是权威源，还是某个缓存？
- [ ] 冷重启后，我的逻辑还能拿到正确值吗？（写一个测试覆盖 restart 场景）
- [ ] 我有没有把"派生量"当"权威量"写到磁盘？（派生量应每次计算，不持久化）

## 运行时目录结构

每个 task 占用 **两个并列子树**，挂在 `$METAINFER_ROOT/nodes/<node_id>/` 下：

```
$METAINFER_ROOT/                    (默认 <cwd>)
└── nodes/
    └── <node_id>/                  (默认 hostname；$METAINFER_NODE_ID 可覆盖。该层级存在是为了后续单平台管控多节点，多节点共享NFS存储。)
        ├── workspaces/             ← 迭代生成产物（结构由 task 包定义）
        │   └── <task_id>/
        └── .metainfer/             ← 元数据 + 日志
            ├── registry.json       (含 workspace_dir)
            ├── registry.lock
            ├── runtime.json
            ├── runtime.lock
            └── tasks/<task_id>/
                ├── requirements.json   {"task_id", "task_type", "created_at", "form": {...}, ...}
                ├── run.json            RunStatus (current_phase, current_iteration, …)
                ├── timeline.jsonl      每行: {"ts": float, "type": str, "payload": dict}
                ├── orchestrator.{pid,log}
                ├── agents.json
                ├── token_budget.json
                ├── iterations/<NNN>.json
                └── logs/<NNN>/
```

关键不变性：
- workspace_dir 由 orchestrator 写、用户读；.metainfer 由 orchestrator + WebUI 协同写。
- WebUI 重置时同时清两个目录：`reset_state_dir(state_dir, workspace_dir, task_id, task_type)`。
- Orchestrator CLI 必须接受 `run <req.json> --state-dir … --workspace-dir …`。

## 项目测试

- 每个修改都需要测试用例。Agent 操作使用 Mock 进行测试。

## 添加新任务类型

所有 task 类型是对等 plugin（包括 shell 自身 `sys-shell`）。新增一个类型 **只改 `metainfer/tasks/<your_task>/` 下的文件**。

**完整骨架和详细注释见：** `metainfer/tasks/example/`

核心步骤：
1. 复制 `metainfer/tasks/example/` → `metainfer/tasks/<your_task>/`
2. 全局替换 `X-type-id` / `X` / `example` 为你自己的名字
3. 取消 `register()` 调用的注释
4. 实现 `orchestrator/pipeline.py` 的迭代逻辑
5. 写测试

### 公共层契约摘要

| 层 | 文件 | 关键约束 |
|---|---|---|
| 文件 | `timeline.jsonl` | `{"ts": float, "type": str, "payload": dict}`；shell 不解释 `type` |
| 文件 | `requirements.json` | `task_type` 必须与 `TaskPlugin.task_type` 和 `WebPlugin.type` 一致 |
| 文件 | `run.json` | shell 读取 11 个字段（见 `state_reader.read_run` defaults）；task 字符串是 opaque token |
| CLI | orchestrator argv | 必须接受 `run <req.json> --state-dir … --workspace-dir …` |
| Web | `build_router(plugin)` | 返回相对路径 `APIRouter`；shell 挂载到 `/api/{type}/{task_id}` |
| Web | `_state_readers.py` | task 专属读取；**不往公共 `state_reader.py` 加 task 逻辑** |
| QA | `qa_config.resolve_target` | `(state_dir, payload) -> {events_file, target_workdir, target_label}` |

### URL 架构

- `sys-shell` → `/api/sys-shell`（无 `{task_id}`）
- task plugin → `/api/{type}/{task_id}`
- 前端静态资源 → `/static/plugins/{type}/`

### 验证

```bash
python -c "from metainfer.server.registry import all_plugins; import metainfer.tasks; print([p.type for p in all_plugins()])"
python -c "from metainfer.orchestrator.tasks import all_tasks; import metainfer.tasks; print([p.task_type for p in all_tasks()])"
python -m pytest
```
