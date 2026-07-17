# Sampler：确定性与随机采样

先读：`00_contracts/framework_contracts.md`和
`03_operators/04_sampling_ops.md`。

## 1. 请求配置

```cpp
struct SamplingConfig {
  float temperature = 0.0f;
  float top_p = 1.0f;
  std::int32_t top_k = 0;
  std::uint64_t seed = 0;
  float repetition_penalty = 1.0f;
  std::vector<std::int32_t> stop_token_ids;
};
```

参数在进入Engine前校验：`temperature >= 0`、`0 < top_p <= 1`、Top-K范围有效，
Token ID位于Vocab内。服务接受的字段必须传入Sampler并产生对应语义。

## 2. 两种模式

`temperature == 0`严格执行Greedy Argmax。相同Logits的Tie规则固定为最小Token
ID，结果不依赖线程调度或未定义的并行归约顺序。

`temperature > 0`执行真实随机采样：温度缩放、可选Penalty、Top-K、Top-P、
归一化和Categorical Sample。禁止在随机模式仍调用Argmax。

## 3. RNG状态

每个Request拥有独立RNG状态，建议Key由`seed + request_sequence_id`导出，Counter
由生成步推进。Batch重新排序、其他请求加入/退出和不同Scheduler批次不得改变
某个请求在相同输入下的随机序列。

```cpp
struct SamplingState {
  std::uint64_t key;
  std::uint64_t counter;
};

Result<SampleResult> Sample(const TensorView& logits,
                            const SamplingConfig& config,
                            SamplingState& state,
                            BackendStream stream);
```

只有成功提交Token后才推进Counter；取消或失败Step不得消耗随机数状态。

## 4. Top-P

Top-P必须按照概率降序选择最小前缀，使累计概率达到阈值，并至少保留一个
Token。数值计算使用稳定Softmax；NaN、Inf或总概率为零必须返回错误。不能只
解析`top_p`而忽略。

## 5. Batch与TP

Batch内每行使用自己的Config和State。实现可以按参数分桶，但输出必须恢复原
Request顺序。TP模式由Rank 0对完整Logits采样，或使用经过验证的Distributed
Sampling；其他Rank不得独立使用本地Logits采样。

## 6. Stop与Detokenize

Sampler只返回Token和可选概率信息。EOS、Stop Token、最大长度和Stop String的
最终状态由Engine处理；Detokenizer负责字节边界。不得为了让文本看似合理而
修改已采样Token。

## 7. 测试

- Greedy三次字节一致，Tie选择固定；
- 同Seed、相同请求和配置重复运行完全复现；
- 不同Seed在非退化分布上至少产生两种结果；
- 同Seed下`top_p=0.01`与`top_p=0.95`可观测不同；
- Batch排序变化不改变单Request序列；
- 失败/取消不推进Counter；
- 极大/极小Logits无NaN，非法参数返回结构化4xx；
- TP=1和TP=N的Greedy结果一致。

