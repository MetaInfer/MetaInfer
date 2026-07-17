# 请求生命周期：从HTTP到资源回收

先读：`00_contracts/framework_contracts.md`、`00_contracts/engine_contracts.md`和
`05_inference_service/01_long_running_native_service.md`。

## 1. 状态机

```cpp
enum class RequestPhase {
  kAccepted,
  kTokenizing,
  kWaitingPrefill,
  kPrefillReserved,
  kPrefillRunning,
  kWaitingDecode,
  kDecodeReserved,
  kDecodeRunning,
  kFinishing,
  kCompleted,
  kCancelled,
  kFailed,
};
```

状态只由Engine线程提交。HTTP、Tokenizer Worker、Backend Callback和Writer通过
Command/Event请求转换，不能同时直接修改RequestState。

## 2. RequestState

```cpp
struct RequestState {
  RequestId id;
  RequestPhase phase;
  std::vector<std::int32_t> prompt_tokens;
  std::vector<std::int32_t> generated_tokens;
  std::int64_t committed_kv_length = 0;
  std::vector<BlockId> blocks;
  SamplingConfig sampling;
  SamplingState sampling_state;
  CancellationToken cancellation;
  ResponseSinkHandle response;
};
```

Request只持逻辑Block ID和Response Handle，不持Owning Device Pointer或Socket裸
指针。`committed_kv_length`只在成功Step后更新。

## 3. Admission

HTTP层完成JSON、模型名、Token Limit和Sampling范围校验，再提交Admission。Engine
检查服务Ready、Queue/Token/KV预算和Feature支持，接受或返回结构化错误。被拒绝
请求不得分配KV或进入Scheduler。

## 4. Tokenization

Tokenizer Worker应用Checkpoint Chat Template并返回Token ID。结果通过Request ID
回到Engine；如果期间取消，则丢弃结果。Tokenizer失败只结束对应请求。Prompt
Token数超过模型/用户限制时，在设备资源分配前拒绝。

## 5. Prefill事务

```text
WaitingPrefill
-> reserve KV blocks and workspace
-> PrefillReserved
-> enqueue device step
-> PrefillRunning
-> completion/error
-> commit KV length and logits, or rollback reservation
```

成功后Sampler产生第一个生成Token，再将Request转入WaitingDecode或Finishing。
采样失败不能提交KV/Token不一致状态。

## 6. Decode事务

每步先预留一个物理Slot，Runner使用当前已提交Token和逻辑Position执行Decode。
完成后采样下一Token；Engine原子提交Token、KV Length和RNG Counter。EOS、Stop、
最大长度和取消进入Finishing。

## 7. 输出事件

非流式请求在完成后构造一次响应；流式请求发布Role/Content/Finish事件。Engine只
发布有效Token/文本增量，Writer拥有Socket。Writer失败提交Cancel Command，不能
直接释放KV或Request对象。

## 8. Cancellation

取消是幂等的：等待请求直接从Queue移除；已预留未运行请求回滚；运行中请求设置
标志并等待当前Device Event边界；完成后释放KV、Workspace、Sampling State和
Response Handle。不能释放仍被Kernel读取的Storage。

## 9. Failure

请求输入错误只影响请求；OOM可拒绝/延后请求；不可恢复Backend或Collective错误
使服务退出Ready并进入全局Drain。Request Error保留首个根因和Step/Layer信息，
后续清理错误不能覆盖它。

## 10. Shutdown

停止Admission后，按配置Drain或Cancel活跃请求。Engine等待有界Device Event，
完成所有Response Sink，释放请求资源，再交给Process Lifecycle释放Model和Backend。
多Rank由Rank 0广播停止，所有Rank按相同Step/Collective边界退出。

## 11. 测试

- 每条合法状态转换和非法转换；
- Tokenization期间取消；
- Prefill/Decode分配后Kernel失败回滚；
- Streaming断开只取消一个请求；
- EOS、Stop、Max Token同时触发的确定优先级；
- Batch重排后Request/KV/RNG不串线；
- Shutdown时Waiting和Running请求资源全部回收；
- 一千次请求后状态表、KV和Workspace计数归零。
