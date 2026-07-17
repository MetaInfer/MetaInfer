# Model Runner：Prefill与Decode执行

先读：`00_contracts/engine_contracts.md`、`00_contracts/attention_kv_contracts.md`
和`01_framework_design/08_tensor_ownership.md`。

## 1. 职责

ModelRunner把不可变`StepPlan`转换为设备输入，调用模型并返回Logits/Event。它不
选择请求、不修改Scheduler队列、不采样Token，也不直接发送HTTP响应。

```cpp
struct StepPlan {
  std::uint64_t step_id;
  StepKind kind;
  std::vector<RequestStep> requests;
  std::vector<SlotMapping> slots;
};

struct StepResult {
  TensorStorage logits;
  BackendEvent completion;
  std::vector<RequestId> request_order;
};

Result<StepResult> RunStep(const StepPlan& plan);
```

## 2. 输入准备

Host元数据先进行边界验证，再复制到预分配Device Metadata Buffer。Prefill至少
构造Token IDs、逻辑Positions、CuSeqLens或等价边界、Slot Mapping和Block
Table；Decode构造每请求的最新Token、逻辑Position、KV Length和物理Slot。

逻辑Position只进入RoPE；物理Slot只进入KV地址。不得把Block Base或Segment
Offset加到RoPE Position。

## 3. Prefill

Prefill允许不同长度请求形成Ragged Batch。Runner必须保证：

- Padding不占KV Slot；
- 每个请求最后有效Token的Logits可定位；
- KV写入和Attention可见范围一致；
- Step失败时预留Block可以回滚；
- Commit前Scheduler看不到新的KV Length。

Reference Path可使用可调试的分阶段Kernel，但生产服务仍必须在目标加速器上
执行完整热路径。

## 4. Decode

Decode每请求追加一个输入Token并产生下一Token Logits。执行顺序固定为：预留
Slot、准备Metadata、模型Forward、等待Step Event、返回Logits、由Engine采样、
最后提交Request/KV状态。任何中间失败不得让Request的逻辑长度提前增长。

## 5. Stream与Workspace

Runner从Workspace Pool取得本Step的Lease。所有依赖Buffer在Completion Event
前不可归还。Compute和Communication可使用不同Stream，但必须用Event表达依赖；
禁止用全局`hipDeviceSynchronize()`掩盖缺少的同步。

## 6. TP执行

Rank 0广播相同Step ID、Request Order和Metadata摘要。所有Rank按照模型层顺序
调用相同Collective。最终Logits可以Gather到Rank 0或执行Distributed Top-K，
但Token只采样一次并广播。单Rank失败必须触发Collective Abort和全局退出。

## 7. 错误边界

Runner错误包含Step、Layer、Operator、Rank、Device和Backend错误。Kernel Launch
错误立即读取；异步错误在Event/Step边界观察。错误后不得返回部分Logits，也不
得以NaN清洗、零填充或CPU重算继续服务。

## 8. 测试

- Tiny Model单Layer与Host Reference对比；
- Prefill最后Token与逐Token Decode Logits Cosine不低于0.95；
- 两个相同请求分配不同物理KV Segment时Greedy输出一致；
- Ragged Prefill、Block边界、取消和OOM回滚；
- 同一Step重复执行无未初始化读取；
- TP=1与TP=N的Top-K和Greedy Token一致；
- Workspace在Completion前不复用，失败后无泄漏。

