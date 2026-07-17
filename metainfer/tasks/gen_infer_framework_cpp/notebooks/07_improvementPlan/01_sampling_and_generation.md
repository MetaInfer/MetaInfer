# 改进计划：真实随机采样与生成语义

状态：proposed  
来源：本地旧任务`gen-infer-framework-cpp-f2efebab`的Sampler在
`temperature > 0`时仍回退到Greedy；SRC-LLAMA、SRC-TRT。  
前置Contract：`00_contracts/engine_contracts.md`、
`01_framework_design/05_sampler.md`、`03_operators/04_sampling_ops.md`。

## 1. 问题和目标

只实现`temperature=0`可以满足确定性C Oracle，但不能代表完整采样能力。目标是同时
支持可证明的Greedy和Seeded Random路径，并让每个OpenAI请求字段真正影响输出。

目标能力：

- `temperature=0`固定Tie规则的Argmax；
- `temperature>0`的Temperature、Top-K、Top-P和Categorical Sample；
- 请求级Seed、Counter和Batch重排不变性；
- 可选Repetition Penalty和Logprobs必须有独立Capability；
- TP下由拥有全局Logits的Rank采样并广播Token。

非目标：第一阶段不做Beam Search、Speculative Decoding、Grammar Sampling，也不追求
一次Kernel完成所有策略。

## 2. 接口改动范围

```text
SamplingConfig         用户参数与校验后的Canonical值
SamplingState          request_id、seed/key、counter
SamplingWorkspace      候选、概率、排序和归约临时区
Sampler::Prepare       Penalty、Temperature、候选筛选
Sampler::Draw          Counter-based RNG和Categorical Sample
Sampler::Commit        Token成功提交后推进Counter
SamplingCapabilities  Backend支持的策略与限制
```

Parser不得静默裁剪非法参数。未实现的字段返回结构化4xx或明确Capability错误，不能
接受后忽略。

## 3. 实施阶段

### S0：冻结语义和Host Reference

- 为每个字段定义默认值、范围、组合顺序和错误响应；
- 实现只用于测试的C++ FP64/FP32 Reference；
- 固定Argmax Tie、Stable Softmax、Top-P最小前缀和至少一个候选；
- 明确Penalty应用在Temperature之前，Stop判断发生在采样之后。

### S1：请求级RNG

- 使用Counter-based RNG，状态属于Request而非Thread或Global Sampler；
- `seed + request_sequence_id`导出独立Key，Counter按成功Token提交推进；
- Cancel、Retry、失败Step和KV事务回滚不能错误消耗Counter；
- 同Seed结果可复现，不同请求加入Batch不能改变已有请求随机序列。

### S2：设备随机路径

- 先实现可验证的多Kernel路径：Scale、Top-K、Softmax/Top-P、Draw；
- FP32累加并检测NaN/Inf和零概率；
- 对大Vocab预分配Workspace，禁止每Token分配；
- 与Host Reference比较候选集合、概率和固定随机数对应Token。

### S3：批内异构与性能

- Batch每行可有不同Temperature、Top-K、Top-P和Seed；
- 参数相同的行可以分桶，但输出和RNG状态必须恢复请求顺序；
- Profile确认排序、D2H和同步成本，再决定融合或分层Top-K；
- 优化前后运行同一统计测试，不能只比较一个固定Prompt。

### S4：TP和高级策略

- 基线为Gather全局Logits到Rank 0采样并Broadcast Token；
- Distributed Top-K只有在全局候选与归一化等价证明后启用；
- Repetition/Frequency/Presence Penalty分别声明，不把它们混为一个参数；
- Logprobs返回值与实际过滤前后定义保持一致。

## 4. 本地验收

正确性：

- Greedy重复三次字节一致，Equal Logits选择最小Token ID；
- 同Seed、同输入、同配置跨Batch布局完全一致；
- 不同Seed在非退化分布上产生多种结果；
- Top-K候选永不越界，Top-P保留的概率前缀满足定义；
- Cancel/失败Step后Retry与未失败基线产生同一序列；
- 非法参数返回错误，未实现策略不静默Fallback；
- 真实Qwen3 Checkpoint同时通过Greedy C Oracle和Seeded Sampling Oracle。

统计验证：对固定人工分布执行足够样本，使用置信区间或适合的Goodness-of-fit检验，
确认频率与目标概率一致。测试Seed固定，阈值和样本数写入测试报告，禁止“看起来随机”。

性能：记录Sampler GPU Event、每Token临时字节、Host Sync次数和全请求吞吐。只有在随机
模式相对端到端Decode成为可测瓶颈时才做融合。

## 5. 风险与停止条件

- 上游框架支持某策略不表示本卡片必须同轮实现；请求Schema与Capability要同步。
- Full Vocab Sort可能吞噬Decode收益，先保证语义再选择Select算法。
- 设备RNG实现变化会改变Seed序列，Build Manifest必须记录Sampling ABI版本。
- 若DTK缺少所需Primitive，保留Native多Kernel实现；不得回到Greedy冒充随机。
