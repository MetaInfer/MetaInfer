# 原生 C++ HTTP 与 SSE 服务实现

先读：`00_contracts/cpp/native_service_contracts.md`。

HTTP Library 可以使用经过许可的小型 C++ Transport，但只能处理 Socket/HTTP；Tokenizer、Scheduler、KV、Model Execution 必须属于本项目 Engine。

## 1. Transport 与 Engine 分层

```cpp
class HttpHandler {
 public:
  HttpResponse HandleModels(const HttpRequest& request);
  HttpResponse HandleCompletions(const HttpRequest& request);
  HttpResponse HandleChatCompletions(const HttpRequest& request);

 private:
  Result<GenerateRequest> ParseCompletion(const HttpRequest& request);
  Result<GenerateRequest> ParseChatCompletion(const HttpRequest& request);

  InferenceEngine* engine_;   // engine outlives handler
  const Tokenizer* tokenizer_;
  const ModelMetadata* model_;
};
```

HTTP Handler 不得调用 `Qwen3DecoderLayer::Forward` 或访问 `KvBlockPool`。

## 2. 启动顺序

```text
解析 CLI/Environment
-> 加载 hardware_profile.json 并检查 blocker
-> 初始化 Rank/Backend/Collective
-> 加载 Config/Tokenizer/Weights
-> 创建 Engine/KV/Scheduler
-> 执行 Warmup/Health Probe
-> Bind/Listen
-> 标记 Ready
```

端口 Bind 失败时直接报告并退出，禁止查找/终止占用端口的进程。

`main.cpp` 骨架：

```cpp
int main(int argc, char** argv) {
  auto config = ParseServerConfig(argc, argv);
  if (!config.ok()) return PrintErrorAndExit(config.status());

  auto runtime = NativeRuntime::Create(config.value());
  if (!runtime.ok()) return PrintErrorAndExit(runtime.status());

  SignalNotifier signals;
  auto signal_status = signals.Install();
  if (!signal_status.ok()) return PrintErrorAndExit(signal_status);

  HttpServer server(config.value().listen_address,
                    config.value().port,
                    runtime.value().handler());
  auto run_status = server.RunUntil(signals.shutdown_event());

  Status shutdown = runtime.value().Shutdown(config.value().shutdown_timeout);
  if (!run_status.ok()) LogError(run_status);
  if (!shutdown.ok()) LogError(shutdown);
  return run_status.ok() && shutdown.ok() ? 0 : 1;
}
```

Signal Handler 只向 `SignalNotifier` 写 Pipe/EventFD/Atomic Flag，复杂 Shutdown 在普通线程执行。

## 3. 请求体限制与 JSON 校验

Transport 在解析前限制 Body Bytes，例如配置项 `max_request_body_bytes`。JSON 解析后逐字段校验：

```cpp
Result<ChatCompletionRequest> ParseChatRequest(const Json& root) {
  ChatCompletionRequest request;
  ASSIGN_OR_RETURN(request.model, RequiredString(root, "model"));
  ASSIGN_OR_RETURN(const Json& messages, RequiredArray(root, "messages"));
  if (messages.empty()) return InvalidArgument("messages must not be empty");

  for (const Json& message : messages) {
    ChatMessage parsed;
    ASSIGN_OR_RETURN(parsed.role, RequiredString(message, "role"));
    ASSIGN_OR_RETURN(parsed.content, RequiredString(message, "content"));
    RETURN_IF_ERROR(ValidateRole(parsed.role));
    request.messages.push_back(std::move(parsed));
  }

  ASSIGN_OR_RETURN(request.max_tokens,
                   OptionalUInt(root, "max_tokens", 16));
  ASSIGN_OR_RETURN(request.temperature,
                   OptionalFloat(root, "temperature", 0.0f));
  ASSIGN_OR_RETURN(request.top_p,
                   OptionalFloat(root, "top_p", 1.0f));
  ASSIGN_OR_RETURN(request.stream,
                   OptionalBool(root, "stream", false));
  RETURN_IF_ERROR(ValidateSampling(request));
  return request;
}
```

禁止把错误类型/范围静默替换成默认值。例如字符串 `"max_tokens": "100"` 必须返回 4xx。

## 4. Chat Template 与 Tokenize

```cpp
Result<GenerateRequest> HttpHandler::Translate(
    const ChatCompletionRequest& http) const {
  if (http.model != model_->served_name) {
    return NotFound("requested model is not served");
  }
  ASSIGN_OR_RETURN(std::string prompt,
                   tokenizer_->ApplyChatTemplate(http.messages,
                                                 /*add_generation_prompt=*/true));
  ASSIGN_OR_RETURN(std::vector<std::int32_t> tokens,
                   tokenizer_->Encode(prompt, /*add_special_tokens=*/false));
  RETURN_IF_ERROR(ValidateContext(tokens.size(), http.max_tokens, *model_));
  return GenerateRequest::From(http, std::move(tokens));
}
```

日志默认不打印 `prompt` 或完整 Messages，只记录 Request ID、Token Count、Sampling 摘要。

## 5. 非流式请求

```cpp
HttpResponse HttpHandler::HandleChatCompletions(const HttpRequest& raw) {
  auto request = ParseAndTranslateChat(raw);
  if (!request.ok()) return ToOpenAiError(request.status());

  auto submitted = engine_->Submit(std::move(request.value()));
  if (!submitted.ok()) return ToOpenAiError(submitted.status());

  const RequestId id = submitted.value();
  auto result = engine_->WaitFinal(id, request_timeout_);
  if (!result.ok()) {
    engine_->Cancel(id);
    return ToOpenAiError(result.status());
  }
  return JsonResponse(200, BuildChatCompletionJson(result.value()));
}
```

HTTP Timeout 后调用幂等 Cancel。`WaitFinal` 不得持有 Engine Scheduler Lock。

## 6. SSE Writer

```cpp
class SseWriter final : public EventSink {
 public:
  Status Push(const EngineEvent& event) override {
    if (connection_.Disconnected()) return Cancelled("client disconnected");
    ASSIGN_OR_RETURN(std::string payload, SerializeSseEvent(event));
    RETURN_IF_ERROR(connection_.Write("data: "));
    RETURN_IF_ERROR(connection_.Write(payload));
    RETURN_IF_ERROR(connection_.Write("\n\n"));
    return connection_.Flush();
  }

  Status Finish() {
    if (finished_) return Status::Ok();
    finished_ = true;
    RETURN_IF_ERROR(connection_.Write("data: [DONE]\n\n"));
    return connection_.Flush();
  }

  bool IsDisconnected() const override { return connection_.Disconnected(); }

 private:
  HttpConnection connection_;
  bool finished_ = false;
};
```

实际实现应通过有界 Queue 把 Engine Event 交给 Socket Writer，不能让 `Push` 在 Engine Thread 中执行阻塞网络 I/O。

## 7. 有界 Streaming Queue

```cpp
class StreamingSession {
 public:
  Status Enqueue(EngineEvent event);
  void WriterLoop();
  void Disconnect();

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<EngineEvent> events_;
  std::size_t pending_bytes_ = 0;
  const std::size_t max_pending_bytes_;
  bool disconnected_ = false;
};
```

达到 `max_pending_bytes` 时按策略取消请求或返回 Resource Exhausted，禁止无限增长。

## 8. OpenAI Error Shape

```json
{
  "error": {
    "message": "max_tokens exceeds model limit",
    "type": "invalid_request_error",
    "param": "max_tokens",
    "code": "context_length_exceeded"
  }
}
```

将内部 `StatusCode` 映射到 HTTP Status/Error Type，但不要把 Backend Stack、文件内容或任意内存地址返回给客户端。

## 9. `/v1/models` 与 Ready

```cpp
HttpResponse HandleModels() {
  if (!runtime_->ready()) {
    return JsonResponse(503, BuildLoadingResponse(runtime_->load_stage()));
  }
  return JsonResponse(200, BuildModelList(runtime_->metadata()));
}
```

Oracle 会轮询该接口。Ready 只能在模型权重、Backend、KV Pool 和必要 Warmup 成功后设置。

## 10. Shutdown

```cpp
Status NativeRuntime::Shutdown(std::chrono::milliseconds timeout) {
  admission_.Stop();
  RETURN_IF_ERROR(engine_.Shutdown(timeout));
  RETURN_IF_ERROR(server_sessions_.CloseAll());
  RETURN_IF_ERROR(tp_launcher_.ShutdownOwnedRanks(timeout));
  RETURN_IF_ERROR(profiler_.Flush());
  return Status::Ok();
}
```

每一步要有 Deadline，避免一个 Rank/客户端无限阻塞退出。

## 11. 测试矩阵

```text
GET /v1/models Loading/Ready
Completion/Chat 正确 JSON Schema
缺字段、错误 JSON Type、超大 Body
Model 不匹配、Context 超限、Sampling 越界
Greedy Deterministic
SSE Event 顺序、Finish、[DONE]
客户端中途断开 -> Engine Cancel -> KV 释放
慢客户端 -> 有界 Backpressure
并发请求、Queue Full
端口冲突但不终止端口 Owner
SIGTERM：Idle/Loading/Active Decode
进程树无 Python 推理 Worker
```
