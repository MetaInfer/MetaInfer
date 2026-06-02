# 推理框架服务化：常驻进程与多次 Prompt（参考 ref_projects）

本文说明如何把「每次 `python` 冷启动都要加载模型」的用法，改成**长驻服务**：**框架（含模型权重、KV 池、调度器）只初始化一次**，新 prompt **仅作为新请求进入调度**，无需重新 `from_pretrained`。

路径均相对于仓库根目录下的 **`meta-infer/ref_projects/`**。

---

## 1. 核心区别

| 模式 | 行为 | 典型代价 |
|------|------|----------|
| **脚本/测试单次进程** | 进程内 `LLMEngine()` → `generate()` → 退出 | 每次进程启动都加载权重、建池 |
| **服务化** | 进程常驻；HTTP/ZMQ/IPC 收请求；内部循环 `step()` 或 `run_forever()` | **仅首次**加载；后续请求只增加 `Sequence`/消息 |

实现要点：

1. **单进程最小服务**：`engine = LLMEngine(...)` 放在模块级或 `main` 里只执行一次，用 **FastAPI/`while True`** 多次调用 `generate(prompt)`（需注意**并发与线程安全**，简易版可串行）。
2. **多进程服务**：**API 进程**只做收发；**Worker 进程**内持 `ModelRunner`/`Engine`，通过 **队列 / ZMQ / Pipe** 投递已 tokenize 的请求或 `GenerateReqInput`。
3. **通信原语**：同步 RPC、ZeroMQ（nano-sglang / mini-sglang）、共享内存 + Event（nano-vllm 多卡 `ModelRunner` worker）。

---

## 2. nano-vllm：引擎常驻、`generate` 可多次调用

### 2.1 设计要点

- **`LLMEngine`** 在 `__init__` 中创建 **`ModelRunner`**（及 TP 时多个子进程）、**`Scheduler`**、**`tokenizer`**（见 `nanovllm/engine/llm_engine.py`）。
- **`generate()`**（约 60–90 行）向调度器 **`add_request`**，然后 **`while not self.is_finished(): self.step()`**，直到本批 prompt 全部完成；**不**在每次 `generate` 里重新构造 `ModelRunner`。
- 因此：若在**同一进程**内先后两次调用 **`llm.generate(batch_a)`**、**`llm.generate(batch_b)`**，只要调度器语义支持「多轮批次」或清空状态（nano-vllm 的 `generate` 是一次性跑完当前 `is_finished` 循环，**需确认 scheduler 是否在批次结束后清空**——从代码看是一次 `generate` 内跑完所有已 `add_request` 的 seq）。**服务化**常见做法是：**常驻 `LLM` 实例**，每来一批用户请求就调一次 **`generate`** 或暴露 **`add_request` + 外层 `while` `step()`**。

### 2.2 参考源码路径与行号

| 内容 | 文件 | 行号（约） |
|------|------|------------|
| `LLMEngine.__init__`：创建 `ModelRunner`、`Scheduler`、tokenizer | `nano-vllm/nanovllm/engine/llm_engine.py` | 17–35 |
| `add_request`、`step`、`is_finished`、`generate` 主循环 | 同上 | 43–90 |
| 对外别名 `LLM` | `nano-vllm/nanovllm/llm.py` | 1–5 |
| 单次脚本：一次 `LLM(...)` 一次 `generate` | `nano-vllm/example.py` | 6–24 |
| 多卡时 rank0 与 worker 通过 `SharedMemory` + `Event` 调 `ModelRunner.call` | `nano-vllm/nanovllm/engine/model_runner.py` | 61–89（`loop`/`read_shm`/`write_shm`/`call`） |

**服务化思路（最小改动）**：进程内全局 **`llm = LLM(model_path, ...)`** 初始化一次；每个 HTTP 请求解析 body 得到 `prompt`，调用 **`llm.generate([prompt], sampling_params)`**（若调度器支持连续多轮；否则需查阅 `Scheduler.is_finished` 与批次边界）。更细粒度可仿 **`add_request` + `step`** 自行驱动循环（同文件 43–55 行）。

---

## 3. nano-sglang：FastAPI + 多进程 + Router 常驻循环

### 3.1 设计要点

1. **`launch_server`**（`server.py`）：分配端口 → **`TokenizerManager`** → 启动 **`start_router_process`**（Router + 模型 RPC）与 **`start_detokenizer_process`** → 等待 pipe 收到 **`"init ok"`** → 线程里 **`uvicorn.run(app)`**（约 73–137 行）。
2. **HTTP 层**：**`POST /generate`** 调用 **`tokenizer_manager.generate_request(obj)`**（约 39–53 行）；模型**已在子进程加载**，请求只序列化 **`GenerateReqInput`**。
3. **Router**：**`RouterManager.loop_for_recv_requests`** 持续 **`recv_pyobj`** 收请求；**`loop_for_forward`** 调 **`model_client.step`** 并把结果发给 detokenizer（`managers/router/manager.py` 约 33–78 行）。

### 3.2 参考源码路径与行号

| 内容 | 文件 | 行号（约） |
|------|------|------------|
| FastAPI 路由 `/generate`、`/v1/completions`、`launch_server` | `nano-sglang/python/sglang/srt/server.py` | 27–71，73–137 |
| `TokenizerManager`：ZMQ 与 router/detokenizer 连接、`generate_request` | `nano-sglang/python/sglang/srt/managers/tokenizer_manager.py` | 64–119（`__init__` 与 `generate_request` 起始） |
| Router 双循环：收请求 + `model_client.step` | `nano-sglang/python/sglang/srt/managers/router/manager.py` | 14–78，`start_router_process` 56–78 |

---

## 4. mini-sglang：ZMQ 前端 + 子进程 `Scheduler.run_forever`

### 4.1 设计要点

1. **`launch_server`**（`server/launch.py`）：**`run_api_server`** 先建 **`FrontendManager`（全局状态）**，再 **`start_backend()`** 拉起多进程：**每 TP rank 一个 `_run_scheduler`**，内联 **`Scheduler(...).run_forever()`**（约 16–37，40–113 行）；另起 tokenizer/detokenizer 进程。
2. **`Scheduler.run_forever`**（`scheduler/scheduler.py`）：**无限循环** `overlap_loop` / `normal_loop`，内部 **`receive_msg`** 从 ZMQ 取 **`UserMsg`** 等，再 **`_forward`**；**`Engine` 在 `Scheduler.__init__` 创建一次**（约 45–49 行）。
3. **`run_api_server`**（`server/api_server.py`）：设置 **`_GLOBAL_STATE`** → **`start_backend()`** → **`uvicorn.run(app)`**（约 403–442 行）。
4. **HTTP**：**`POST /generate`** 里 **`state.new_user()`** + **`send_one(TokenizeMsg(...))`**，不重新加载模型（约 229–247 行）。

### 4.2 参考源码路径与行号

| 内容 | 文件 | 行号（约） |
|------|------|------------|
| 子进程入口：`_run_scheduler` → `scheduler.run_forever()` | `mini-sglang/python/minisgl/server/launch.py` | 16–37，59–113 |
| `Scheduler.__init__` 创建 `Engine`；`run_forever` 无限循环 | `mini-sglang/python/minisgl/scheduler/scheduler.py` | 45–76，120–131 |
| `FrontendManager`、`/generate`、`run_api_server` | `mini-sglang/python/minisgl/server/api_server.py` | 100–159，229–247，403–442 |
| **离线单进程**：`LLM` 继承 `Scheduler`，`generate` 内 `run_forever()` 直到结束（非 HTTP，但展示「一次 Engine」） | `mini-sglang/python/minisgl/llm/llm.py` | 28–98 |

---

## 5. 映射到当前 `meta-infer/llm_engine.py` 的落地建议

当前 **`LLMEngine`** 已在 **`__init__`** 中加载 HF 模型与 **`KVMemoryPool`/`Scheduler`**。要做「服务化」：

1. **最小**：单独写一个 **`server_minimal.py`**（或 FastAPI），启动时 **`engine = LLMEngine(...)`** 一次；每个请求调用 **`engine.generate(prompt, ...)`**。注意：**多请求并发**时 `Scheduler`/`Sequence` 是否线程安全——若未设计锁，应 **单 worker 串行** 或 **请求队列 + 单线程执行**。
2. **对齐 ref**：多进程时让**仅 worker** 持有 GPU 上的 `model`，前端只做 tokenize/HTTP，通过 **ZMQ/队列** 传 `prompt` 或 `input_ids`（仿 nano-sglang / mini-sglang）。

---

## 6. 小结表

| 项目 | 常驻核心对象 | 新 prompt 入口 | 典型 IPC/协议 |
|------|----------------|----------------|---------------|
| nano-vllm | `LLMEngine` / `ModelRunner` | `add_request` 或 `generate` | 单机多进程：SharedMemory（model_runner） |
| nano-sglang | Router + ModelRpcClient + TokenizerManager | HTTP → `generate_request` | ZMQ、Pipe |
| mini-sglang | `Scheduler` + `Engine` | HTTP → `TokenizeMsg` → ZMQ | ZMQ、多进程 Queue ack |

---

*文档用于对照 `ref_projects` 实现常驻推理服务；行号随上游变更可能漂移，请以类名/函数名为准检索。*
