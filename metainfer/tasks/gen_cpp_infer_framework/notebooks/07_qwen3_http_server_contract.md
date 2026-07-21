# Qwen3-8B C++ HTTP Server 与 `serve.sh` 契约

> 用途：给实现 Agent 一份可以直接接入 MetaInfer 流水线 C 的服务层契约。本文只负责把已经完成的 Qwen3 tokenizer、runtime 和 sampler 暴露成 HTTP API，不改变模型计算。

## 1. 最终目标

生成目录中必须提供：

```text
CMakeLists.txt
build.sh                         # 流水线生成并拥有，不得修改
serve.sh                         # Agent 编写，前台启动服务
include/http_server.hpp
include/openai_api.hpp
src/main.cpp
src/http_server.cpp
src/openai_api.cpp
src/engine.cpp
src/model_loader.cpp
src/tokenizer.cpp
src/sampler.cpp
...
```

构建后必须产生：

```text
build/metainfer_cpp_server
```

流水线 C 不会直接调用 C++ 类。它只通过 `serve.sh` 启动程序，再通过 HTTP 检查框架。

```text
MetaInfer C oracle
  -> bash build.sh
  -> MODEL_DIR=<real-model> bash serve.sh <free-port>
       -> exec build/metainfer_cpp_server --port <port> --model <real-model>
            -> load GGUF + tokenizer + Qwen3Runtime
            -> GET  /v1/models
            -> POST /v1/chat/completions
  -> extract choices[0].message.content
  -> judge responses
  -> SIGTERM server process group
```

当前流水线契约的源码参考路径：

```text
metainfer/tasks/gen_cpp_infer_framework/orchestrator/prompts.py
metainfer/tasks/gen_cpp_infer_framework/orchestrator/oracles/correctness.py
metainfer/tasks/gen_cpp_infer_framework/orchestrator/oracles/perf.py
```

## 2. 流水线 C 的真实启动过程

流水线按以下顺序执行：

1. 检查 `build.sh` 和 `serve.sh` 存在。
2. 执行系统生成的 `bash build.sh`。
3. 检查构建是否成功；失败时不会启动服务器。
4. 选择一个空闲的 localhost TCP 端口。
5. 设置真实模型路径：

   ```bash
   export MODEL_DIR=/path/to/model
   ```

6. 执行：

   ```bash
   bash serve.sh <port>
   ```

7. 轮询 `GET /v1/models`；模型未就绪时允许暂时连接失败或返回 `503`。
8. 服务就绪后，逐个发送 `POST /v1/chat/completions`。
9. 从响应的 `choices[0].message.content` 取出模型文本。
10. 测试结束后向整个服务进程组发送 `SIGTERM`，超时才使用 `SIGKILL`。

因此 `serve.sh` 必须阻塞在前台，不能 daemonize，不能使用 `&` 启动后立即退出。

## 3. 服务层和模型层的边界

C++ HTTP 层只做：

- TCP/HTTP 请求接收；
- JSON 解析与校验；
- OpenAI 请求转换为内部 `GenerateRequest`；
- 调用 tokenizer、runtime 和 sampler；
- 把生成文本封装成 JSON；
- 错误码、信号和资源生命周期管理。

HTTP 层不要重新实现：

- GGUF tensor 解析；
- Q8_0 反量化；
- Transformer forward；
- KV cache 算法；
- HIP kernel。

建议内部接口：

```cpp
struct GenerateRequest {
    std::string system_text;
    std::string user_text;
    int32_t max_new_tokens = 128;
    float temperature = 0.0f;
    uint64_t seed = 0;
    bool enable_thinking = false;
};

struct GenerateResult {
    std::string text;
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
    bool stopped_by_eos = false;
};

class Qwen3Engine {
public:
    bool initialize(const std::string& model_path, std::string* error);
    bool generate(
            const GenerateRequest& request,
            GenerateResult* result,
            std::string* error);
};
```

`Qwen3Engine` 初始化一次并长期持有模型权重。每个请求重新设置生成状态和 KV cache 位置，但不能重新加载权重。

## 4. C++ 主程序参数

建议二进制支持：

```text
metainfer_cpp_server \
  --host 127.0.0.1 \
  --port 18080 \
  --model /path/to/Qwen3-8B-Q8_0.gguf
```

至少支持：

```text
--host <ip>       default: 127.0.0.1
--port <1..65535> required
--model <path>    required after fallback resolution
--help
--version
```

模型参数可以是 GGUF 文件，也可以是模型目录。若是目录，必须使用确定性的规则选择模型：

1. 优先寻找任务配置指定的 GGUF 文件名；
2. 否则目录中必须恰好只有一个 `.gguf`；
3. 存在零个或多个候选时打印清晰错误并退出，不能随便选第一个文件。

启动顺序建议：

```text
parse args
validate model path
create/load Qwen3GgufModel
create tokenizer
create Qwen3Runtime and allocate GPU buffers
create sampler
bind/listen HTTP socket
enter blocking accept loop
```

首版可以同步加载完模型后再监听端口。流水线允许加载期间连接失败，并会等待最多数分钟。这样比“先监听、后台加载、管理 ready 状态”更简单。

## 5. 必须实现的 HTTP 路由

### 5.1 `GET /v1/models`

模型加载完成后返回 `200 application/json`：

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-8b",
      "object": "model",
      "created": 0,
      "owned_by": "metainfer"
    }
  ]
}
```

流水线把这个接口当成低开销 readiness 检查。不要在这个请求中执行一次模型生成。

### 5.2 `POST /v1/chat/completions`

这是流水线 C 真正使用的接口。实际请求形状为：

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 16,
  "temperature": 0.0,
  "stream": false
}
```

健康探测可能把 `model` 写成 `probe`，正式测试通常写成 `default`。固定 Qwen3-8B 服务应接受这两个别名，不能因为它们和 `qwen3-8b` 不同而拒绝请求。

首版必须解析：

- `messages[].role`；
- `messages[].content`；
- `max_tokens`；
- `temperature`；
- `stream`；
- 可选 `seed`。

流水线只发送 `stream: false`。首版遇到 `stream: true` 可以返回 `400`，但不能返回伪造的非流式结果并声称是 SSE。

最小成功响应：

```json
{
  "id": "chatcmpl-metainfer-1",
  "object": "chat.completion",
  "created": 0,
  "model": "qwen3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Paris"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 1,
    "total_tokens": 13
  }
}
```

流水线判定成功的关键字段是：

```text
choices[0].message.content
```

它必须存在且必须是 JSON string。HTTP `200` 但 `choices` 为空，仍然会被判为无输出。

`finish_reason` 建议：

```text
stop    -> EOS/EOG/stop token
length  -> 达到 max_tokens 或上下文容量
```

### 5.3 `POST /v1/completions`

流水线 C 当前不依赖它，可以第二阶段再实现。若实现，返回：

```json
{
  "choices": [
    {"index": 0, "text": "...", "finish_reason": "stop"}
  ]
}
```

## 6. Chat 请求到 Qwen3 Runtime 的调用流程

单个流水线请求应执行：

```text
parse JSON
  -> extract optional system message and user message
  -> Qwen3Tokenizer::format_chat_prompt(user, system, enable_thinking=false)
  -> tokenizer.encode(prompt, parse_special=true, add_bos=false)
  -> runtime.reset()
  -> runtime.prefill(prompt_token_ids)
  -> sample logits
  -> repeat runtime.decode(previous_token)
  -> stop on EOS/EOG, max_tokens, or context capacity
  -> tokenizer.decode(generated_ids, skip_special=true)
  -> OpenAI JSON response
```

流水线的测试输出上限较短。为避免 Qwen3 thinking 内容耗尽 `max_tokens` 后还没有给出最终答案，首版服务建议默认：

```cpp
enable_thinking = false;
```

如果后面需要 thinking mode，可以增加非标准可选字段或启动参数，但不能改变流水线默认行为。

首版 tokenizer 只完整支持单轮 `system + user`。若收到超出当前能力的多轮 `assistant/user` 历史，应返回清晰 `400`；不要错误地把所有 role 文本直接拼接成普通字符串。

## 7. 单卡 Runtime 的并发规则（B=1 基线）

当前 `Qwen3Runtime` 只有一套：

- `current_pos`；
- KV cache；
- activation workspace；
- FP16 权重反量化 workspace；
- hipBLAS handle 和 HIP stream。

这些状态不能被两个请求同时使用。

首版最安全的实现是整个服务串行处理请求：

```cpp
accept connection
read request
handle complete generation
write response
close connection
```

也可以让 HTTP 层使用工作线程，但调用 `engine.generate()` 时必须持有覆盖完整生成周期的全局 mutex：

```cpp
std::lock_guard<std::mutex> lock(engine_mutex);
engine.generate(...);
```

不能只锁单个 decode step，否则请求 A/B 会交替覆盖同一套 KV cache。

流水线 C 顺序发送请求，因此串行首版可以通过正确性测试。性能流水线 E 会并发发送请求；串行服务仍能工作，但吞吐较低。

> **continuous batching 实现时，以上“全程 engine mutex”模式不再适用。** 必须以 `09_continuous_batching_contract.md` 为唯一并发规范：HTTP worker 只向有界队列提交请求；一个可 join 的 scheduler 线程独占 runtime/HIP stream；每请求独占 KV slot；scheduler 每个 decode tick 动态合并活跃 slot。不要仅删除 mutex，也不要让多个 HTTP worker 直接调用 runtime。

## 8. 最小 HTTP/1.1 传输契约

可以使用系统已有的 HTTP/JSON 库，也可以编写一个小型 POSIX socket server。禁止运行时下载依赖。

若自己实现，至少保证：

1. 使用 `socket`、`bind`、`listen`、`accept`。
2. 设置 `SO_REUSEADDR`。
3. 读取到 `\r\n\r\n` 后再解析 header。
4. header 名大小写不敏感。
5. 根据 `Content-Length` 循环读取完整 body，不能假设一次 `recv()` 就收到完整 JSON。
6. 设置 header/body 大小上限，例如 64 KiB/1 MiB，超限返回 `413`。
7. 首版可以只支持 `Content-Length`，不支持 chunked request 时返回 `400` 或 `501`。
8. 响应必须包含：

   ```text
   HTTP/1.1 200 OK
   Content-Type: application/json; charset=utf-8
   Content-Length: <exact byte count>
   Connection: close
   ```

9. 循环 `send()`，直到完整响应发送完或连接失败。
10. 每个连接处理一个请求后关闭；首版不必实现 keep-alive。

JSON parser 至少要正确处理 object、array、string、number、boolean、null，以及字符串中的：

```text
\"  \\  \n  \r  \t  \uXXXX
```

JSON serializer 必须转义模型输出中的双引号、反斜杠和控制字符。不能直接用字符串拼接把原始模型文本塞进 JSON，否则代码输出中的 `"` 或换行会产生非法响应。

## 9. 参数校验和 HTTP 错误

建议返回统一错误结构：

```json
{
  "error": {
    "message": "messages must contain one user message",
    "type": "invalid_request_error",
    "code": "invalid_messages"
  }
}
```

状态码建议：

| 情况 | 状态码 |
| --- | ---: |
| 非法 JSON、字段类型错误 | 400 |
| 不支持 `stream:true` | 400 |
| 未知路由 | 404 |
| 路由存在但方法错误 | 405 |
| body 过大 | 413 |
| 模型仍在加载 | 503 |
| HIP/hipBLAS/生成失败 | 500 |

输入限制至少包括：

```cpp
1 <= max_tokens && max_tokens <= runtime_limit
temperature >= 0.0f
messages is a non-empty array
content is a string
prompt_tokens + max_tokens <= max_seq_len
```

上下文超过容量时可以减少实际生成上限并用 `finish_reason="length"`，也可以返回 `400`；必须避免 KV cache 越界。

## 10. `serve.sh` 契约与模板

`serve.sh` 的模型路径解析顺序必须是：

1. 第二个位置参数 `$2`；
2. 环境变量 `$MODEL_DIR`；
3. 从任务 requirements 写入的固定默认路径。

不能在真实模型缺失时静默切换成 mock、echo 或随机输出。

参考模板：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-}"
DEFAULT_MODEL_PATH="/absolute/path/from/task-requirements"
MODEL_PATH="${2:-${MODEL_DIR:-${DEFAULT_MODEL_PATH}}}"

if [[ -z "${PORT}" || ! "${PORT}" =~ ^[0-9]+$ ]] ||
   (( PORT < 1 || PORT > 65535 )); then
    echo "usage: $0 <port> [model-path]" >&2
    exit 2
fi

if [[ -z "${MODEL_PATH}" || ! -e "${MODEL_PATH}" ]]; then
    echo "model path does not exist: ${MODEL_PATH}" >&2
    exit 3
fi

SERVER="${ROOT_DIR}/build/metainfer_cpp_server"
if [[ ! -x "${SERVER}" ]]; then
    bash "${ROOT_DIR}/build.sh"
fi

exec "${SERVER}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --model "${MODEL_PATH}"
```

关键点：

- 使用 `exec`，让 C++ 进程继承 `serve.sh` 的 PID 和信号；
- 不使用 `nohup`、`setsid` 或末尾 `&`；
- 不吞掉 stdout/stderr，流水线会保存日志；
- 不覆盖 `ROCR_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES` 等硬件绑定环境变量；
- 文件需要提交可执行权限：`chmod +x serve.sh`。

## 11. SIGTERM 和资源清理

流水线完成测试后会终止服务。服务器必须在大约十秒内退出。

信号 handler 只能执行最小操作：

```cpp
volatile std::sig_atomic_t g_stop = 0;

void on_signal(int) {
    g_stop = 1;
}
```

accept loop 可以使用 `poll/select` 的短超时定期检查 `g_stop`，或者使用 self-pipe 唤醒。不要在 signal handler 里调用 `hipFree`、C++ iostream、mutex 或复杂析构逻辑。

### 11.1 禁止停止状态脱节（2026-07-20 回归记录）

曾出现过如下实现：signal handler 只设置全局 `g_stop`，而 HTTP accept loop 检查的是 `SimpleHttpServer::stop_`。两者没有任何同步路径，因此 SIGTERM 虽然被捕获，服务却永远不会退出，smoke test 最终阻塞在 `wait`，同时长期占用端口和 GPU 显存。

禁止这种“双停止变量”实现：

```cpp
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

// BUG: loop 从不读取 g_stop，且没有代码调用 server.stop()。
while (!stop_) {
    accept(listen_fd, nullptr, nullptr);
}
```

停止状态必须只有一条可证明的传播链。推荐使用以下任一方案：

1. `poll/select` 或 self-pipe：signal handler 只设置 `g_stop` 并写入 self-pipe；accept loop 被唤醒后直接读取同一个 `g_stop`。
2. 专用 `sigwait` 线程：主线程先 block SIGTERM/SIGINT，等待线程通过 `sigwait` 收到信号后调用 `server.stop()`；`stop()` 设置 `std::atomic<bool>` 并 `shutdown()` 监听 socket，以解除阻塞的 `accept()`。复杂 C++ 清理由正常控制流执行，不在 signal handler 内执行。

无论采用哪种方案，都必须满足：

- `accept()` 返回错误后先检查停止状态；已停止时立即 `break`，不得重试；
- 不依赖 `SO_RCVTIMEO` 自动解决 shutdown；必须显式证明 loop 会观察到停止状态；
- SIGTERM 后十秒内进程退出 0 或约定的受控退出码；
- 析构顺序释放 socket、scheduler/worker、HIP stream/handle、KV cache 与模型权重；
- Continuous batching 模式还必须唤醒 condition variable、完成或取消 pending/active 请求，并 join scheduler，不能留下 detached thread。

Smoke test 必须同时验证“请求成功”和“能够退出”：

```bash
bash serve.sh "${PORT}" >server.stdout.log 2>server.stderr.log &
server_pid=$!

# health / generation checks ...

kill -TERM "${server_pid}"
for _ in $(seq 1 100); do
    kill -0 "${server_pid}" 2>/dev/null || break
    sleep 0.1
done
if kill -0 "${server_pid}" 2>/dev/null; then
    kill -KILL "${server_pid}"
    wait "${server_pid}" || true
    echo "server ignored SIGTERM" >&2
    exit 1
fi
wait "${server_pid}"
```

进程检查不得使用裸 `pgrep -f metainfer_cpp_server` 作为成功判据；`-f` 会匹配检查命令自身或包含该字符串的父 shell，产生看似“不断重生 PID”的假阳性。优先保存并检查 `$!`，需要扫描时使用精确的 `/proc/<pid>/exe`、`ps` 字段或排除检查进程本身。测试应杀 `serve.sh` 通过 `exec` 暴露的真实 server PID；若引入 `timeout`/wrapper，则必须明确杀整个测试进程组，并在 TERM 宽限期后升级到 KILL。

退出 accept loop 后再按正常 C++ 生命周期：

```text
stop accepting requests
finish or abort active request
destroy HTTP resources
destroy Qwen3Runtime buffers/KV cache
destroy hipBLAS handle and HIP stream
free model device weights
exit 0
```

## 12. CMake 和 `build.sh`

Agent 只维护 `CMakeLists.txt`；`build.sh` 由流水线和硬件 profile 生成，不得修改。

`CMakeLists.txt` 必须定义准确目标名：

```cmake
add_executable(metainfer_cpp_server
    src/main.cpp
    src/http_server.cpp
    src/openai_api.cpp
    src/engine.cpp
    src/model_loader.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    # HIP sources...
)
```

要求：

- C++17 或更高；
- include path 指向项目 `include/`；
- 链接 HIP runtime、hipBLAS 和必要的线程库；
- 不使用 `FetchContent`、`git clone` 或运行时下载；
- `bash build.sh` 成功后，`./build/metainfer_cpp_server --help` 必须返回 0。

## 13. 本地 Smoke Test

交给流水线 C 前至少完成：

```bash
bash build.sh
./build/metainfer_cpp_server --help
```

然后在一个空闲端口启动：

```bash
MODEL_DIR=/path/to/model bash serve.sh 18080 \
    >server.stdout.log 2>server.stderr.log &
SERVER_PID=$!
```

健康检查：

```bash
curl -fsS http://127.0.0.1:18080/v1/models
```

生成检查：

```bash
curl -fsS \
  -H 'Content-Type: application/json' \
  -d '{
        "model":"default",
        "messages":[{"role":"user","content":"What is the capital of France?"}],
        "max_tokens":16,
        "temperature":0,
        "stream":false
      }' \
  http://127.0.0.1:18080/v1/chat/completions
```

响应需要能被标准 JSON parser 解析，并且：

```text
choices[0].message.content is a non-empty string
```

测试后：

```bash
kill -TERM "${SERVER_PID}"
wait "${SERVER_PID}"
```

不要把后台测试进程留给下一次运行，否则可能占用端口和 Z200 显存。

## 14. 首版验收清单

- [ ] `bash build.sh` 产生 `build/metainfer_cpp_server`。
- [ ] `--help` 和 `--version` 不加载模型并返回 0。
- [ ] `serve.sh <port>` 前台阻塞，并使用 `exec` 启动 C++ server。
- [ ] `$2 -> $MODEL_DIR -> 固定默认值` 的模型路径优先级正确。
- [ ] 模型路径错误时打印真实路径并非零退出，不进入 mock 模式。
- [ ] 模型和 GPU buffer 只在启动时加载/分配一次。
- [ ] `GET /v1/models` 返回合法 JSON 和 HTTP 200。
- [ ] `POST /v1/chat/completions` 接受 `model=probe/default`。
- [ ] 支持 `stream:false`、`temperature:0` 和 `max_tokens`。
- [ ] `choices[0].message.content` 是非空字符串。
- [ ] 模型输出经过正确 JSON escaping。
- [ ] B=1 基线中，每个请求开始前 reset runtime/KV 状态。
- [ ] B=1 基线中，单卡串行执行完整请求，不交叉覆盖 KV cache；continuous batching 实现改按 `09_continuous_batching_contract.md` 的 slot/KV 隔离与并发验收执行。
- [ ] EOS/EOG、`max_tokens` 和 context limit 都能停止生成。
- [ ] SIGTERM 能让服务在十秒内退出并释放 GPU 资源；测试同时覆盖空闲 `accept()` 和请求执行/排队期间的 shutdown。
- [ ] 停止状态只有一条可证明的传播链，不存在 handler 写 `g_stop`、loop 却只读另一个 `stop_` 的脱节实现。
- [ ] 本地 smoke test 后没有残留进程；使用保存的真实 PID/进程组验证，不以裸 `pgrep -f` 结果作为判据。
