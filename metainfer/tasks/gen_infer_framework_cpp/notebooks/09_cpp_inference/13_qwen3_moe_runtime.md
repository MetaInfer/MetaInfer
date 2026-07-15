# Qwen3 MoE 原生 Runtime

先读：`00_contracts/cpp/cpp_qwen3_runtime_contracts.md`、目标 Checkpoint 的 `config.json`。Qwen3 MoE 不能通过“忽略 Expert 字段并使用 Dense MLP”实现。

## 1. 配置与权重

必须从目标配置读取并验证：

```cpp
struct MoeConfig {
  std::int64_t num_experts = 0;
  std::int64_t experts_per_token = 0;
  std::int64_t expert_intermediate_size = 0;
  bool normalize_topk_weights = false;
  float router_scale = 1.0f;
  bool has_shared_expert = false;
};
```

具体字段名和 Router Policy 由目标模型元数据决定，禁止套用其他 MoE Family。加载器还要映射：

```text
router/gate weight
每个 expert 的 gate/up/down weight
可选 shared expert weight
```

在分配显存前计算每 Rank Expert Bytes，加上 Attention、Norm、Workspace、通信 Buffer 和 KV。总权重字节刚好小于四卡聚合显存，不代表模型可运行。

## 2. Router CPU Reference

首先实现一个慢但确定性的 Host Reference，用来验证 GPU Router：

```cpp
struct RoutedExpert {
  std::int32_t expert_id;
  float weight;
};

Result<std::vector<RoutedExpert>> SelectTopKReference(
    Span<const float> logits,
    const MoeConfig& config) {
  if (config.experts_per_token <= 0 ||
      config.experts_per_token > config.num_experts ||
      logits.size() != static_cast<std::size_t>(config.num_experts)) {
    return InvalidArgument("invalid MoE router shape/config");
  }

  std::vector<std::pair<float, int>> values;
  values.reserve(logits.size());
  for (int expert = 0; expert < config.num_experts; ++expert) {
    if (!std::isfinite(logits[expert])) {
      return DataLoss("non-finite router logit");
    }
    values.emplace_back(logits[expert], expert);
  }

  // 同分时按 expert_id 稳定处理，保证测试确定性。
  std::partial_sort(values.begin(),
                    values.begin() + config.experts_per_token,
                    values.end(),
                    [](const auto& a, const auto& b) {
                      return a.first != b.first ? a.first > b.first
                                                : a.second < b.second;
                    });

  std::vector<RoutedExpert> selected;
  for (int i = 0; i < config.experts_per_token; ++i) {
    selected.push_back({values[i].second,
                        ApplyCheckpointRouterScore(values[i].first, config)});
  }
  if (config.normalize_topk_weights) NormalizeWeights(selected);
  return selected;
}
```

`ApplyCheckpointRouterScore` 必须按真实 Qwen3 MoE 配置实现，不能把示例中的 Logit 直接当最终 Weight。

## 3. Token Assignment

每个 Token 产生 `top_k` 个 Assignment：

```cpp
struct ExpertAssignment {
  std::int32_t token_index;
  std::int32_t expert_id;
  float route_weight;
};

struct ExpertBatchPlan {
  TensorStorage sorted_token_indices;
  TensorStorage sorted_expert_ids;
  TensorStorage sorted_route_weights;
  TensorStorage expert_offsets;  // [num_experts + 1]
  std::int32_t assignment_count = 0;
};
```

构建过程：

```text
(token, expert, route_weight)
-> 统计每个 Expert Assignment 数
-> Prefix Sum 得到 expert_offsets
-> Stable Scatter 到 Expert-contiguous 顺序
-> Gather Hidden Rows
-> Expert GEMM
-> Route Weight Multiply
-> Scatter-add 回原 Token 顺序
```

所有临时 Buffer 上限为 `scheduled_tokens * experts_per_token`，必须 Checked Multiply，禁止按请求数量无界扩张。

## 4. Expert 执行接口

```cpp
class MoeExperts {
 public:
  Result<void> Forward(const TensorView& hidden,
                       const ExpertBatchPlan& plan,
                       TensorView output,
                       WorkspaceLease& workspace,
                       BackendStream stream) const;

 private:
  Result<void> RunOneExpert(int expert_id,
                            TensorView grouped_input,
                            TensorView grouped_output,
                            BackendStream stream) const;
};
```

正确性基线可以按 Expert 逐个调用 GEMM。性能版本可使用 Grouped GEMM，但必须保留同一 `ExpertBatchPlan` 并与基线对比。

## 5. Scatter-add 注意事项

每个 Token 会收到多个 Expert 结果：

```cpp
for (assignment : assignments_for_token) {
  output[token] += assignment.route_weight * expert_output[assignment];
}
```

GPU 实现可使用分组后归约，避免无序 Atomic 导致不可控数值差异。需要显式定义 Accumulation DType 和 Tie/Order。

## 6. 并行放置策略

必须只选择并记录一种：

### TP-sharded Experts

每个 Expert 分布在所有 TP Rank，复用 Dense MLP 的 Column/Row Parallel Collective。实现相对直接，但通信量可能较高。

### Expert Parallel

不同 Rank 拥有 Expert 子集，需要 All-to-All 风格 Token Exchange、Capacity、Ordering 和 Failure Protocol。不能只把 Expert Weight 分到不同卡而不交换 Token。

### Replicated Experts

仅当每 Rank 内存预算明确可容纳时允许，通常不适合显存紧张的大模型。

```cpp
enum class ExpertPlacement { kTensorParallel, kExpertParallel, kReplicated };

struct ExpertPlacementPlan {
  ExpertPlacement kind;
  std::vector<int> owner_rank_by_expert;
};
```

Router 决策必须在需要的 Rank 上一致。即使某个 Expert 本步没有 Token，所有 Rank 的 Collective 顺序仍必须一致。

## 7. Shared Expert

如果配置声明 Shared Expert，必须单独执行其 MLP 并按模型语义合并：

```cpp
RETURN_IF_ERROR(routed_experts_.Forward(..., routed_output, ...));
if (config_.has_shared_expert) {
  RETURN_IF_ERROR(shared_expert_.Forward(hidden, shared_output, ...));
  RETURN_IF_ERROR(ops_->Add(routed_output, shared_output, output, ctx));
}
```

不得把 Shared Expert 当作 Router Top-k 中的普通 Expert，除非目标模型明确定义如此。

## 8. 测试与 Profiling

必须覆盖：

```text
已知 Router Logits 的 Top-k、Tie、归一化
一个 Token 选择多个 Expert
多个 Token、空 Expert、严重倾斜 Expert
Gather/Scatter 保留原 Token 顺序
Shared Expert On/Off
Expert Key/Shape/Placement
Reference Per-expert 与 Grouped Execution
TP=1/TP=N 的 Expert ID、Weight、Output
显存不足在分配前拒绝
Rank 失败时所有通信退出
```

Profile 分开记录 Router、Prefix Sum/Permutation、Expert GEMM、Scatter/Reduction、Collective、每 Expert Token Count。只有 Grouped Path 与 Reference 一致后才开始优化。
