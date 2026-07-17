# SSE增量响应

先读：`00_contracts/native_service_contracts.md`和`02_openai_http_api.md`。

## 1. 协议

`stream=true`返回`Content-Type: text/event-stream`。每个事件使用
`data: <json>\n\n`，至少包含OpenAI兼容的`choices[0].delta`。正常结束发送带
`finish_reason`的事件和`data: [DONE]\n\n`。

## 2. 真正增量

Engine产生Token/文本增量后立即入有界Response Queue；Writer循环处理Partial
Write并Flush。把完整生成结果切成多个字符串后一次性发送不属于流式。Tokenizer
可能跨Token持有不完整UTF-8/Byte序列，Detokenizer只有在形成有效增量时发布。

## 3. 生命周期

```cpp
enum class StreamEventKind { kRole, kContent, kFinish, kError };
struct StreamEvent { RequestId request; StreamEventKind kind; std::string data; };
```

每请求最多一个Finish和一个`[DONE]`。错误发生在Header前可返回普通结构化HTTP
错误；Header后只能发送SSE Error/Finish并关闭。客户端断开触发取消，不发送到
已失效Socket。

## 4. 背压和SIGPIPE

Queue满时暂停该请求的输出或取消该请求，不阻塞Engine全局线程。Socket Write
处理EINTR、EAGAIN和Partial Write，并使用`MSG_NOSIGNAL`或进程级SIGPIPE策略。
只能由Connection所有者关闭FD，避免重用FD被旧Writer误写。

## 5. 测试

- 至少两个非空Content Chunk后`[DONE]`；
- Role、Content、Finish顺序和JSON Schema；
- 多字节Unicode跨Token增量；
- 慢客户端Queue上限；
- Header后错误；
- 写入中断开不终止服务；
- `stream=false`与流式最终文本一致；
- 取消后不再产生内容事件。

