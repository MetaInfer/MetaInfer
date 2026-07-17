# GPU Sampling算子

本文描述Sampler的设备实现；请求级RNG与状态提交见
`01_framework_design/05_sampler.md`。

## 1. 算子流水线

```text
logits validation
-> repetition/frequency penalties（声明支持时）
-> temperature scaling
-> top-k candidate selection
-> stable softmax
-> top-p prefix selection
-> counter-based categorical sample
```

Greedy使用独立Argmax路径，并固定Tie为最小Token ID。随机路径必须真实消费Seed和
Counter，禁止调用Argmax伪装Sampling。

## 2. 接口

```cpp
struct SamplingRow {
  float temperature;
  float top_p;
  int top_k;
  std::uint64_t rng_key;
  std::uint64_t rng_counter;
};

Status SampleRows(TensorView logits, Span<const SamplingRow> configs,
                  TensorView token_ids, TensorView next_counters,
                  const OperatorContext& context);
```

输出Counter只有在Engine提交Token后写回Request State。Kernel不直接修改Host请求。

## 3. 数值和性能

Softmax使用FP32归约并检测NaN/Inf。Top-P至少保留一个候选。Vocab很大时可先
Top-K再归一化，但若用户没有Top-K限制，优化不能改变Top-P语义。Batch行参数可以
不同；分桶后恢复原顺序。

## 4. TP

基线为Gather完整Logits到Rank 0后采样。Distributed Top-K需要证明全局候选、概率
归一化和Seed语义与基线一致后才可启用。最终Token由Rank 0广播。

## 5. 测试

Argmax Tie、不同Vocab尾部、同Seed复现、不同Seed差异、Top-P极值、Batch重排、
NaN/Inf、Counter提交/回滚，以及CPU Reference概率分布统计测试。

