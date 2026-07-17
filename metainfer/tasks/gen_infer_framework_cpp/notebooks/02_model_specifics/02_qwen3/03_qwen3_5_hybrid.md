# Qwen3.5/3.6 Hybrid原生Runtime

Hybrid模型不是Qwen3 Dense的参数变化，而是状态空间/线性Attention层与完整
Attention层的组合。只有Checkpoint明确声明相应Architecture时才使用本文。

## 1. 配置与层调度

Loader必须解析每层类型、Full Attention间隔、GatedDeltaNet参数、Conv/SSM State
Shape、MRoPE Section和模型特有Norm/Gate字段。层类型数组长度必须等于Layer数；
未知类型不得按Dense Layer继续。

```cpp
enum class HybridLayerKind { kGatedDeltaNet, kFullAttention };

struct HybridLayerSpec {
  HybridLayerKind kind;
  std::int64_t layer_index;
  GatedDeltaConfig delta;
  FullAttentionConfig attention;
};
```

## 2. Persistent State

每个活跃请求和每个GatedDeltaNet层拥有独立Conv/Recurring State。State由Engine
请求生命周期管理，物理Storage由专用Pool管理；Batch重排时通过Request ID映射，
不能按Batch Row永久绑定。

Prefill产生初始State，Decode原地或双缓冲更新。Step失败时State必须回滚或仅在
Completion后提交。取消请求释放全部Layer State。

## 3. GatedDeltaNet路径

执行顺序和Shape完全由Checkpoint定义，至少包括输入Projection、Conv State、
Decay/Delta参数、Recurrent Update、Gate和Output Projection。Reference实现可分步
执行并以FP32累加；生产Kernel必须与多个序列长度和Batch重排场景逐项对比。

不得复用标准Attention KV Cache替代Recurring State，也不得在每个Decode Step
从全部历史Token重算State。

## 4. Full Attention与MRoPE

Full Attention层遵循Checkpoint定义的Q/K/V、Q/K Norm、Output Gate和MRoPE。
MRoPE由多个Position Section组成，必须验证Section总维度和Layout。逻辑Position
来自Request，不能混入KV Block或State Pool Offset。

## 5. Norm与权重映射

Hybrid Checkpoint可能有不同Norm变体和嵌套权重前缀。Weight Mapper必须使用完整
Architecture前缀和Layer Kind匹配；Required Key被静默跳过通常会产生全零/NaN，
因此Load Report必须按Layer列出已加载权重。

## 6. TP

Full Attention可按Qwen Dense TP切分；GatedDelta参数、Conv Channel和State需要
独立Shard规则。任何参数不能仅在Rank 0加载后让其他Rank使用零值。启用TP前先
验证TP=1全部中间结果，再比较每类Layer的TP输出和State。

## 7. 测试

- Dense模型被Hybrid Loader拒绝，Hybrid模型不走Dense Runtime；
- Prefill后逐步Decode State与Reference一致；
- 两请求交错和Batch重排不交换State；
- 取消/OOM回滚后Pool统计守恒；
- MRoPE边界和不同Section配置；
- 每种Layer单层、完整交替层序列和TP等价；
- 无NaN/Inf，错误权重前缀导致非零失败。

