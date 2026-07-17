# 原生 OpenAI HTTP/SSE 服务契约

> 权威级别：C++ Server 与 Immutable HTTP Oracle 的强制契约。

## 1. 进程契约

`bash serve.sh PORT` 最终必须 `exec` 原生 Rank-0 Server/Launcher，并在前台阻塞。服务必须遵守：

```text
MODEL_DIR
CUDA/HIP/ROCR_VISIBLE_DEVICES
METAINFER_PROFILE
METAINFER_PROFILE_OUTDIR
SIGTERM
```

核心推理不得由 Python Child Process 提供。

## 2. HTTP 接口

必需 Endpoint：

```text
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

建议将 Transport DTO 与 Engine Request 分开：

```cpp
struct ChatMessage {
  std::string role;
  std::string content;
};

struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::uint32_t max_tokens = 1;
  float temperature = 0.0f;
  float top_p = 1.0f;
  bool stream = false;
  std::optional<std::uint64_t> seed;
};

Result<GenerateRequest> ValidateAndTranslate(
    const ChatCompletionRequest& request,
    const ModelMetadata& model);
```

必须校验 JSON Type、Required Field、Model、Token Limit、Sampling Range、Stop 和 Stream。客户端错误返回结构化 4xx；内部/设备错误返回 5xx，但不得暴露任意内存或文件内容。

### HTTP/1.1 字节流规则

- Header 读取到 `\r\n\r\n` 后解析 `Content-Length`，继续读取完整 Body；
- 单次 `read/recv` 不保证包含完整 Header 或 Body；
- 单次 `write/send` 不保证写完响应，必须循环处理 Partial Write；
- 客户端提前断开不得终止服务进程，使用 `MSG_NOSIGNAL` 或等价的
  `SIGPIPE` 策略，并把断开传递到请求取消路径；
- Body/Headers 必须有独立上限，超限返回 4xx，禁止固定栈缓冲区越界；
- 必须测试 Header/Body 分段到达、零长度 GET、Malformed Length 和
  写响应前断开连接。

Chat Messages 交给 Engine 前必须通过 Checkpoint 的 Chat Template 和
Special Token 配置渲染。禁止使用 `user: ... assistant:` 之类手工模板。

## 3. 非流式响应

### Sampling 参数语义

- `temperature=0` 必须执行确定性 Greedy，同一请求重复执行结果字节一致；
- `temperature>0` 必须执行真实随机采样，不能回退到 Argmax；
- 相同 Prompt、Sampling 参数和 Seed 必须复现；不同 Seed 在非退化分布上应
  能产生不同 Token 序列；
- `top_p` 必须限制候选累计概率，范围错误返回结构化 4xx；
- API 接受但 Engine/Sampler 忽略任何上述字段均视为正确性失败。

响应至少包含稳定的：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "...",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

`choices[0].message.content` 必须是字符串。

## 4. SSE Streaming

`stream=true` 时必须：

1. 返回正确的 SSE Content Type；
2. 按顺序发送 `data: {...}\n\n`；
3. Flush 增量 Token；
4. 只发送一次终态 Finish Event；
5. 最后发送 `data: [DONE]\n\n`；
6. 客户端断开时通过 Engine Cancel Path 释放请求。

建议抽象：

```cpp
class EventSink {
 public:
  virtual Status Push(const EngineEvent& event) = 0;
  virtual bool IsDisconnected() const = 0;
  virtual void Close() noexcept = 0;
};
```

慢客户端的输出 Queue 必须有界，不能阻塞 Engine Thread 或无限持有 KV。

## 5. 生命周期规则

- **SVC-001**：模型 Ready 之前 `/v1/models` 可以报告 Loading，但不得接受真实推理。
- **SVC-002**：SIGTERM 后停止 Admission，在有界时间 Drain/Cancel，Flush Profile，关闭 TP Rank。
- **SVC-003**：端口被占用时直接报告并退出，禁止终止端口所有者。
- **SVC-004**：日志包含 Request ID/Rank ID，默认不记录完整 Prompt。
- **SVC-005**：测试模式固定 Sampling Seed 和参数。
- **SVC-006**：Signal Handler 只做 Async-signal-safe 通知，不直接 Join/Lock/Flush。

## 6. 验收用例

必须覆盖：

- Ready 前后 `/v1/models`；
- Completion/Chat 正常响应；
- 错误 JSON、超大 Body、不支持参数；
- Deterministic Greedy；
- Seeded Stochastic Sampling：同 Seed 复现、不同 Seed 差异、`top_p` 生效；
- SSE 顺序、Finish、`[DONE]`；
- 客户端断开和 Cancellation；
- Header/Body 分段、Partial Write 和 SIGPIPE；
- 并发、Backpressure、Overload；
- 端口冲突不杀进程；
- Idle/Loading/Decode 阶段 SIGTERM；
- 进程树中没有 Python 推理 Worker。
