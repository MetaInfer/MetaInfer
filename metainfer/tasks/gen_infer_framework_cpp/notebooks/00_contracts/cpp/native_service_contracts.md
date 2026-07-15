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

## 3. 非流式响应

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
- SSE 顺序、Finish、`[DONE]`；
- 客户端断开和 Cancellation；
- 并发、Backpressure、Overload；
- 端口冲突不杀进程；
- Idle/Loading/Decode 阶段 SIGTERM；
- 进程树中没有 Python 推理 Worker。
