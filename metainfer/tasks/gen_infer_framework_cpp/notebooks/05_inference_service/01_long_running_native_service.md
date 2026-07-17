# 长驻原生推理服务

## 1. 服务与一次性程序的区别

长驻服务只加载一次模型和设备资源，随后接收多个并发请求。请求结束只释放请求
状态、KV和临时Workspace，不销毁模型或Backend。服务必须支持健康、背压、取消、
优雅退出和持续错误观测。

## 2. 组件

```text
Listener/Event Loop
-> HTTP Parser + OpenAI Validation
-> bounded Admission Queue
-> Tokenizer Workers
-> InferenceEngine/Scheduler
-> Response Event Queue
-> JSON or SSE Writer
```

I/O层不得阻塞等待整个模型Forward；Engine通过Request ID发送Token/Finish/Error
事件。所有Queue有上限和关闭语义。

## 3. Ready与Live

Live表示进程事件循环仍可响应；Ready表示真实Checkpoint加载、目标Device Probe、
Warmup和必要Rank Barrier全部完成。加载期间不接受生成请求。运行时不可恢复的
Backend错误使Ready变False并开始Drain/Stop。

## 4. 并发与背压

限制连接数、Header/Body字节、等待请求、活跃Sequence、总Token预算和每连接
Streaming Queue。超过限制返回结构化4xx/429/503，不分配无界线程或内存。慢
客户端通过有界Queue施加背压或取消其请求，不得阻塞所有Decode。

## 5. 请求隔离

每请求拥有Sampling State、取消令牌、KV映射和Response Sink。连接断开只取消
对应请求。一个请求的非法JSON、OOM或Stop条件不能改变其他请求的状态。

## 6. 测试

连续多Prompt不重复加载；并发短/长请求；Queue满；慢客户端；连接断开；单请求
取消；Backend错误；加载中健康；SIGTERM Drain；一千次请求后内存稳定。

