# 改进计划：原生服务、SSE与取消闭环

状态：proposed  
来源：旧任务曾出现空`NextEvent()`和HTTP直接驱动Step；SRC-LLAMA、SRC-VLLM、
SRC-SGLANG。  
前置Contract：`00_contracts/native_service_contracts.md`、
`01_framework_design/07_request_lifecycle.md`、
`05_inference_service/03_sse_streaming.md`。

## 1. 目标

原生C++ Server只负责协议、连接和Response Sink；Engine通过事件接口提供Token、文本、
完成或错误。`stream=false`聚合相同事件，`stream=true`按SSE增量写出，两条路径共享
同一个请求状态机和生成语义。

非目标：第一阶段不实现HTTP/2、TLS终止、WebSocket或分布式Gateway。生产部署可由
外部Reverse Proxy处理TLS，但不能由Python Sidecar承载模型服务。

## 2. 目标接口

```text
SubmitRequest -> RequestHandle
NextEvent(handle, deadline) -> Token | TextDelta | Finish | Error
Cancel(handle, reason)
ResponseSink::TryWrite(bytes)
ConnectionState { reading, submitted, streaming, closing }
```

HTTP不得调用ModelRunner、Scheduler Step或读取可变RequestState。Engine不知道Socket
类型，只向有界Response Queue发布事件。

## 3. 实施阶段

### H0：协议基线

- 增量读取直到完整Header，再按`Content-Length`循环读取Body；
- 限制Header、Body、JSON深度、Message数、Prompt Token和输出Token；
- 处理Partial Read/Write、EINTR、EAGAIN、Peer Close和SIGPIPE；
- 每个错误映射稳定HTTP Status和OpenAI-compatible Error Body；
- `/v1/models`只有在Checkpoint、Backend、KV Pool和Warmup就绪后报告Ready。

### H1：真实Engine Event

- 删除空事件Stub和HTTP内部Polling Step；
- RequestHandle携带不可复用Generation ID，防止旧连接消费新请求事件；
- Finish只发送一次，Cancel幂等，Error后禁止继续Token事件；
- 非流式响应使用同一事件流聚合，不维护第二套Decode Loop；
- 请求对象在Engine和Connection均释放后才销毁。

### H2：SSE

- Header为`text/event-stream`，每个事件是完整UTF-8/JSON Frame；
- 首个Role、Content Delta、Finish Reason和`[DONE]`顺序固定；
- Detokenizer保留不完整UTF-8和Tokenizer Byte边界，不逐Token强行构造字符；
- Header发送后发生错误时发送SSE Error/Finish或关闭，不能再改HTTP Status；
- `stream_options`等未支持字段明确拒绝或按Capability处理。

### H3：背压和取消

- 每连接Pending Byte和Event数有上限；
- 慢客户端只暂停/取消自身请求，不阻塞全局Engine；
- Disconnect立即发送Cancel Command，Engine在安全Step边界回收KV；
- Response Queue满、Deadline、Server Shutdown和Backend Error使用不同Reason；
- 取消后的已完成GPU工作可以丢弃，但不能Commit新Token或RNG Counter。

### H4：负载和生命周期

- Admission Queue有上限并返回明确Overload响应；
- Keep-alive、Idle Timeout和最大每连接请求数可配置；
- SIGTERM先停止监听，再停止Admission，Drain或Cancel，Flush Profile并Join；
- Server PID、Port和子进程归属由启动脚本精确记录；
- 不使用Detach Thread处理无限连接。

## 4. 本地验收

协议：Header和Body分多次写入、超限、Malformed JSON、未知字段策略、Partial Response、
客户端提前断开、SIGPIPE和Keep-alive连续请求。

语义：

- 同一Seed下Stream和Non-stream最终文本、Usage和Finish Reason一致；
- 首个SSE事件、多个Delta、Finish和`[DONE]`顺序正确；
- UTF-8跨Token边界不会输出损坏字符；
- Cancel后无新增内容且KV/Request Slot被回收；
- 一个慢客户端不使其他短请求P99无界增长；
- Backend错误只结束受影响请求或触发有界全局失败，不产生假200。

耐久：固定并发运行一千个短请求，混入Disconnect、Timeout和非法请求，检查FD、Thread、
Host Memory、Device Memory和Active Request都回到稳定范围。

## 5. 性能验收

分开记录Parse、Tokenize、Queue Wait、TTFT、Inter-token Latency、Write Stall、P50/P99和
完成率。Streaming不要求提高总吞吐，但不能因每TokenFlush、锁竞争或无界小Write导致
明显回退。合并Frame必须服从Latency上限。

## 6. 风险与回滚

- Event Queue引入的新锁不能包住Model Forward或Socket Write。
- Disconnect检测具有平台差异，使用真实Socket测试而不是只调用Cancel API。
- 外部框架的OpenAI字段覆盖面不是本卡片承诺；Capability和错误必须诚实。
- 如果异步I/O库需要新依赖，先评估License、ABI和目标服务器可用性；保留最小POSIX基线。
