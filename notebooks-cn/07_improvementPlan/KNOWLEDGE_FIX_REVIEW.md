# Knowledge Fix Review — SSE 连接生命周期 + Worker 信号处理

| 字段 | 值 |
|------|-----|
| 审查日期 | 2026-06-06 |
| 审查对象 | inference_blueprint.json / AGENT_SKILL.md / prompts.md 的知识改动 vs openai_tp_server.py 实际代码修复 |
| Bug 1 | SSE 流式响应连接不关闭（benchmark hang） |
| Bug 2 | Non-rank0 Worker 进程残留（GPU 不释放） |

---

## 总体结论：✅ PASS

三个知识文件的改动**准确描述了实际 bug 根因和修复方案**，与 `openai_tp_server.py` 中的代码修复完全一致。JSON 语法有效。无关键遗漏。有 2 处微小差异（不影响正确性），详见下文。

---

## 逐文件审查

### 1. inference_blueprint.json — ✅ PASS

**审查范围**：`framework_layer.components[7].OpenAITPServer.architecture` 下的 `streaming.critical_constraints` 和 `worker_signal_handling` 两个新增子节点。

#### 1.1 streaming.critical_constraints

| 字段 | 代码对应 | 判定 |
|------|---------|------|
| `_why`: "SSE 无 Content-Length，浏览器/客户端依赖连接关闭来判定流结束。BaseHTTPRequestHandler 默认 keep-alive" | `openai_tp_server.py:352-358` SSE headers 注释 + `Connection: close` | ✅ 准确 |
| `connection_header`: "Connection: close（不是 keep-alive）" | `openai_tp_server.py:356` `self.send_header('Connection', 'close')` | ✅ 一致 |
| `close_connection_flag`: "self.close_connection = True 必须在 try 成功路径和 except/finally 错误路径都设置" | `openai_tp_server.py:440`（成功路径）+ `openai_tp_server.py:469`（finally 路径） | ✅ 一致 |
| `past_bug_20260606`: "benchmark 0/5 请求成功，warmup 请求持有 engine_lock 永久不释放" | 症状描述与背景一致 | ✅ 准确 |

**遗漏检查**：`_why` 字段同时解释了"只设 header 不设 flag → 连接仍不关闭"的机制，覆盖了修复的两个必要条件。无遗漏。

#### 1.2 worker_signal_handling

| 字段 | 代码对应 | 判定 |
|------|---------|------|
| `_why`: "non-rank0 worker 主线程阻塞在 dist.broadcast_object_list（NCCL/Gloo collective）。Python 默认信号处理在 C 调用返回后才执行" | `openai_tp_server.py:137-141` 注释："When torchrun sends SIGTERM, os._exit(0) is the only reliable way to terminate because the main thread may be blocked in NCCL/Gloo collectives ... Python's default signal handling defers signals until the C call returns" | ✅ 一致 |
| `solution`: "signal.signal(SIGTERM, handler) → handler 中调用 os._exit(0)。os._exit() 不经过 Python 解释器，直接终止进程" | `openai_tp_server.py:142-148` 定义 `_worker_signal_handler(signum, frame)` → `os._exit(0)` + `signal.signal(signal.SIGTERM, ...)` | ✅ 一致 |
| `past_bug_20260606`: "torchrun kill 后 4 个 python 子进程残留，各占 ~6.5GB GPU 显存" | 与背景描述一致 | ✅ 准确 |

**遗漏检查**：覆盖了根因（C 调用阻塞 + Python 信号延迟）、方案（os._exit 直接终止）、症状（进程残留 + GPU 占用）。无遗漏。

#### 1.3 JSON 语法

```bash
python -c "import json; json.load(open('inference_blueprint.json')); print('JSON valid')"
# 输出: JSON valid
```

✅ 语法有效，无解析错误。

---

### 2. AGENT_SKILL.md §2.4 — ✅ PASS（含 2 处微小备注）

**审查范围**：§2.4 "OpenAI TP Server 架构" 新增的三个小节。

#### 2.1 "SSE 连接生命周期（2026-06-06 线上 hang 根因）"

| 描述 | 代码对应 | 判定 |
|------|---------|------|
| "SSE 响应无 Content-Length，客户端依赖连接关闭判定流结束" | HTTP SSE 规范 + `openai_tp_server.py:356` | ✅ 准确 |
| "必须：`Connection: close` header + `self.close_connection = True`（成功路径和 except/finally 路径都要设）" | 行 356 + 行 440 + 行 469 | ✅ 一致 |
| "缺少 close_connection → 连接永不关闭 → benchmark warmup 请求持有 engine_lock → 0/N 请求成功" | 症状描述 | ✅ 准确 |

#### 2.2 "Non-rank0 Worker 信号处理（2026-06-06 进程残留根因）"

| 描述 | 代码对应 | 判定 |
|------|---------|------|
| "Worker 主线程阻塞在 `dist.broadcast_object_list`（C 调用），Python 信号处理延迟到 C 返回后" | `openai_tp_server.py:137-141` 注释 | ✅ 准确 |
| "必须：`signal.signal(SIGTERM, lambda: os._exit(0))`" | `openai_tp_server.py:142-148` | ⚠️ 微小差异（见下） |
| "缺少 handler → torchrun kill 后子进程残留，每进程占 ~6.5GB GPU 显存" | 症状描述 | ✅ 准确 |

**⚠️ 微小差异 #1 — lambda 签名**：

AGENT_SKILL.md 写 `signal.signal(SIGTERM, lambda: os._exit(0))`，但 Python signal handler 必须接受两个参数 `(signum, frame)`，无参 lambda 会运行时报错。实际代码用的是：
```python
def _worker_signal_handler(signum, frame):
    ...
    os._exit(0)
```

**影响**：低。这是文档简写而非可执行代码，读者理解意图即可。但建议修正为 `lambda signum, frame: os._exit(0)` 避免误导。

**⚠️ 微小差异 #2 — 仅提 SIGTERM，遗漏 SIGINT**：

AGENT_SKILL.md 只提 SIGTERM，但实际代码同时注册了 SIGINT（`openai_tp_server.py:147`）。SIGINT 处理对开发调试（Ctrl+C）很重要。

**影响**：低。SIGTERM 是生产环境 torchrun kill 的主要信号，是核心修复。SIGINT 是开发便利性补充。

#### 2.3 "Benchmark 脚本清理要点"

- `pkill -9 -f "openai_tp_server.py"` — 实用运维建议，非代码修复描述，无问题。
- `trap cleanup EXIT INT TERM` — 同上。

---

### 3. prompts.md 会话 5 Phase 10 — ✅ PASS

**审查范围**：Phase 10 部分新增的两条约束（行 392-393）。

| 约束 | 代码对应 | 判定 |
|------|---------|------|
| "SSE 响应必须 `Connection: close` + `self.close_connection = True`（SSE 无 Content-Length，keep-alive 导致客户端永久 hang）" | 行 356 + 行 440 + 行 469 | ✅ 一致 |
| "Non-rank0 worker 必须注册 SIGTERM handler → os._exit(0)（主线程阻塞在 NCCL collective 时 Python 信号被延迟，os._exit 直接终止进程）" | 行 142-148 | ✅ 一致 |

两行约束简洁准确地概括了修复要点，括号内补充了根因。无遗漏。

---

## 代码修复完整性核实

### Bug 1: SSE 连接不关闭

| 修复项 | 代码位置 | 知识文件覆盖 |
|--------|---------|-------------|
| `Connection: keep-alive` → `Connection: close` | 行 356 | blueprint ✅ + SKILL ✅ + prompts ✅ |
| 成功路径 `self.close_connection = True` | 行 440 | blueprint ✅ + SKILL ✅ + prompts ✅ |
| except/finally 路径 `self.close_connection = True` | 行 469 | blueprint ✅ + SKILL ✅ + prompts ✅ |

### Bug 2: Worker 进程残留

| 修复项 | 代码位置 | 知识文件覆盖 |
|--------|---------|-------------|
| SIGTERM handler → `os._exit(0)` | 行 142-148 | blueprint ✅ + SKILL ✅ + prompts ✅ |
| SIGINT handler → `os._exit(0)` | 行 147 | 仅代码有，三个知识文件均未提及 |

---

## 发现的问题汇总

| # | 严重性 | 文件 | 描述 |
|---|--------|------|------|
| 1 | ⚪ 低 | AGENT_SKILL.md §2.4 | `lambda: os._exit(0)` 无参 lambda 不可用作 signal handler（缺 signum/frame 参数）。建议修正为 `lambda signum, frame: os._exit(0)` |
| 2 | ⚪ 低 | AGENT_SKILL.md §2.4 + prompts.md | 仅提及 SIGTERM，遗漏代码中同时注册的 SIGINT handler。对生产环境无影响（torchrun 发 SIGTERM），但对开发调试有参考价值 |

**无中/高严重性问题。**

---

## 结论

三个知识文件（blueprint JSON / AGENT_SKILL.md / prompts.md）对两个 bug 的根因、修复方案、症状描述**均与实际代码修复一致**。JSON 语法有效。两处微小差异（lambda 签名简写、遗漏 SIGINT）均为文档简写层面，不影响知识传递的正确性和完整性。

**审查结果：✅ PASS**
