# Profiling 接口契约

> 关联 notebooks: `06_profiling/01_pytorch_profiler.md`, `05_inference_service/01_long_running_inference_service.md`

## 概述

每一次迭代产出的推理框架代码 **MUST** 预留 profiling 接口。E 阶段的 perf oracle 在测量吞吐 / 延迟之外，会通过环境变量触发该接口，捕获一份 Chrome tracing 格式的 `.json.gz` 文件，作为 retrospective 与 F (perf plan) 的输入。

接口设计原则：

1. **默认关闭** —— 不设环境变量时，框架行为与未启用 profiler 完全一致；不能因为「留了 profiler hook」就引入稳态开销。
2. **环境变量开关** —— 框架自己 **不** 决定何时开始 profiling；由调用方（perf oracle / 开发者手工）通过环境变量驱动。
3. **输出格式标准化** —— 一律输出 Chrome tracing JSON（`chrome://tracing`、`perfetto.dev` 可直接打开），gzip 压缩，扩展名 `.json.gz`。
4. **输出路径可控** —— 不写死路径；通过环境变量接收，默认落到 server 工作目录。
5. **零额外依赖** —— 优先使用 `torch.profiler`（PyTorch 自带）。可选 `py-spy` 等 OS 级工具，但框架代码本身不强制引入。

---

## 环境变量契约

框架在启动时读取下列变量。所有变量都是 **可选的**；缺失 = 关闭。

| 变量 | 含义 | 取值 | 默认 |
|---|---|---|---|
| `METAINFER_PROFILE` | 总开关 | `1` / `0` / 未设 | 未设 = 关闭 |
| `METAINFER_PROFILE_OUTDIR` | tracing 文件输出目录（必须存在并可写） | 任意路径 | `.` (server CWD) |
| `METAINFER_PROFILE_DURATION_S` | 单次捕获时长（秒）。到时自动 stop+flush | 正整数 | `60` |
| `METAINFER_PROFILE_ACTIVITIES` | 选哪些 activity | `CPU,CUDA` / `CPU` / `CUDA` | `CPU,CUDA` |
| `METAINFER_PROFILE_SCHEDULE_WAIT` | 每个 profiler step 的 wait 步数 | 正整数 | `1` |
| `METAINFER_PROFILE_SCHEDULE_WARMUP` | warmup 步数 | 正整数 | `1` |
| `METAINFER_PROFILE_SCHEDULE_ACTIVE` | active 采样步数 | 正整数 | `3` |
| `METAINFER_PROFILE_SCHEDULE_REPEAT` | 重复采样次数 | 正整数 | `1` |

### 命名约定

- **进程内只启动一个 profiler 实例**。多个 worker 进程时，每个进程独立写自己的 trace 文件，文件名带 rank / pid 后缀避免冲突。
- 输出文件名模板：`metainfer-profile-{rank|pid}-{start_ts}.json.gz`。
- 框架在 server 启动日志中 **打印一行** 标明 profiler 是否启用、输出路径是什么，便于排查。格式固定为：
  ```
  [metainfer-profile] enabled=1 outdir=/path activities=CPU,CUDA duration_s=60
  ```
  或
  ```
  [metainfer-profile] enabled=0
  ```

### 触发模型

框架的 profiler 满足以下两个触发方式之一（实现任一即可，推荐都做）：

1. **进程内自动启停**（主路径）：profiler 在 server 启动时创建并 start，在 `METAINFER_PROFILE_DURATION_S` 秒后由内部定时器自动 stop + export。这种方式让 perf oracle 可以「设好环境变量 → 启 server → 跑 benchmark → kill server」时不需要发额外信号。
2. **SIGUSR1 触发**（可选辅助路径）：收到 SIGUSR1 时如果 profiler 还没启动则启动；再次收到则 stop + export。用于「在稳态运行时窗口式采样」。

无论哪种触发，stop 时都要确保 `.json.gz` 文件完整 flush 后再退出 server。SIGTERM 处理流程里 **必须** 包含 `profiler.stop()` + export。

---

## 接口实现骨架（PyTorch profiler）

最小骨架，符合上面的契约：

```python
# profile.py
import os
import signal
import threading
from pathlib import Path
from typing import Optional

import torch.profiler as tp


def profile_enabled() -> bool:
    return os.environ.get("METAINFER_PROFILE", "0") == "1"


class ProfileContext:
    """Singleton-ish: one profiler per process. None when disabled."""
    def __init__(self) -> None:
        self.enabled = profile_enabled()
        self.profiler: Optional[tp.profiler.profile] = None
        self._timer: Optional[threading.Timer] = None
        self.outdir: Optional[Path] = None
        self.rank: int = int(os.environ.get("LOCAL_RANK", "0"))

    def start(self) -> None:
        if not self.enabled:
            print("[metainfer-profile] enabled=0", flush=True)
            return
        outdir = Path(os.environ.get("METAINFER_PROFILE_OUTDIR", "."))
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        activities_str = os.environ.get("METAINFER_PROFILE_ACTIVITIES", "CPU,CUDA").upper()
        activities = []
        if "CPU" in activities_str:
            activities.append(tp.ProfilerActivity.CPU)
        if "CUDA" in activities_str:
            activities.append(tp.ProfilerActivity.CUDA)

        wait = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_WAIT", "1"))
        warmup = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_WARMUP", "1"))
        active = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_ACTIVE", "3"))
        repeat = int(os.environ.get("METAINFER_PROFILE_SCHEDULE_REPEAT", "1"))

        self.profiler = tp.profile(
            activities=activities,
            schedule=tp.schedule(wait=wait, warmup=warmup,
                                 active=active, repeat=repeat),
            on_trace_ready=tp.tensorboard_trace_handler(str(outdir)),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.profiler.start()
        print(f"[metainfer-profile] enabled=1 outdir={outdir} "
              f"activities={activities_str} duration_s="
              f"{os.environ.get('METAINFER_PROFILE_DURATION_S', '60')}", flush=True)

        duration = int(os.environ.get("METAINFER_PROFILE_DURATION_S", "60"))
        self._timer = threading.Timer(duration, self.stop)
        self._timer.daemon = True
        self._timer.start()

        # Optional SIGUSR1 toggle path
        signal.signal(signal.SIGUSR1, self._on_sigusr1)

    def _on_sigusr1(self, signum, frame) -> None:
        # Toggle behavior: if running, stop; if stopped, do nothing.
        try:
            self.stop()
        except Exception:
            pass

    def stop(self) -> None:
        if self.profiler is None:
            return
        try:
            self.profiler.stop()
        except Exception:
            pass
        self.profiler = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
```

注意：`tp.tensorboard_trace_handler(outdir)` 默认输出文件名包含 host/time/rank，已经足够唯一；如果你想要更受控的命名，可以替换成自定义 handler：

```python
def _trace_handler(p):
    out = outdir / f"metainfer-profile-rank{self.rank}-{int(time.time())}.json.gz"
    p.export_chrome_trace(str(out))
```

### 集成到 server

server 进程在加载完模型、进入 serve 循环前调用一次：

```python
ctx = ProfileContext()
ctx.start()
# ... serve loop ...
# 在 SIGTERM handler / atexit 里：
ctx.stop()
```

绝对不要 wrap 单个 `forward()`；profiler 是 **进程级生命周期**，包住整个 serve 窗口。

---

## 禁止事项

- ❌ 在 `METAINFER_PROFILE` 未设时启动 profiler（会引入 ~10%+ 的稳态开销）
- ❌ 把 profiler 写死为始终启用、用环境变量关闭的反向逻辑
- ❌ 把 `.json.gz` 写到 `/tmp` 不带任务标识，多任务并发时互相覆盖
- ❌ 多 rank 时所有进程写同一个文件名（在 TP > 1 时必然冲突）
- ❌ 用 `record_shapes=True` + `with_stack=True` —— 在 long-running server 下内存爆炸；只在调试具体 kernel 时短时启用
- ❌ profiler 还在 active 时直接 `os._exit()` —— 会导致 `.json.gz` 不完整

---

## Perf oracle 端的对接

Perf oracle（`metainfer/orchestrator/oracles/perf/harness.py`）在 spawn `serve.sh` 之前会：

1. 为本任务创建 `~/.metainfer/tasks/<task_id>/iterations/<n>/profile/`
2. 设置 `METAINFER_PROFILE=1`
3. 设置 `METAINFER_PROFILE_OUTDIR=<that dir>`
4. 设置 `METAINFER_PROFILE_DURATION_S=<perf benchmark wall time + 5s 缓冲>`
5. 启 server、跑 benchmark、kill server
6. benchmark 结束后，读取 `profile/` 下的 `.json.gz`，在 perf-report.json 的 `profile_artifacts` 字段记录文件路径与大小

profiling 输出 **不** 参与 regression 判定（避免引入噪声），只作为 retrospective / F-step perf-plan 的诊断素材。
