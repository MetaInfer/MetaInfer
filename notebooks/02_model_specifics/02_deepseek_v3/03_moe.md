# DeepSeek MoE (Mixture of Experts)

## 概述

DeepSeek V3 使用创新的 MoE 架构，包含共享专家(Shared Experts)和路由专家(Routed Experts)两种类型，通过 Grouped Top-K 路由实现高效的专家选择。

## 架构设计

### 专家结构

```
DeepseekV2MoE
├── gate (MoEGate)                    # 路由门控
│   └── weight [n_routed_experts, hidden_size]
│
├── experts (SharedFusedMoE)          # 路由专家
│   └── 256 experts, each:
│       ├── gate_proj [hidden_size, intermediate_size]
│       ├── up_proj [hidden_size, intermediate_size]
│       └── down_proj [intermediate_size, hidden_size]
│
└── shared_experts (DeepseekV2MLP)    # 共享专家
    └── 1 expert (always active)
```

### 参数配置

```python
# MoE 核心参数
n_routed_experts = 256          # 路由专家数量
n_shared_experts = 1            # 共享专家数量
num_experts_per_tok = 8         # 每个token激活的专家数
moe_intermediate_size = 2048    # 专家中间层维度
n_group = 8                     # 专家分组数
topk_group = 4                  # 每组选择的top-k

# 计算激活参数
active_params = num_experts_per_tok * moe_intermediate_size * 3  # 每token
# = 8 × 2048 × 3 = 49,152 参数/token

# 如果是 Dense:
dense_params = n_routed_experts * moe_intermediate_size * 3
# = 256 × 2048 × 3 = 1,572,864 参数

# 激活率 = 49,152 / 1,572,864 ≈ 3.1%
```

## Grouped Top-K 路由

### 路由流程

```
输入 hidden_states [num_tokens, hidden_size]
    │
    ▼
Gate Linear: [num_tokens, n_routed_experts]
    │
    ▼
Grouped Top-K Selection:
  1. 分成 n_group=8 组，每组 32 个专家
  2. 每组选 topk_group=4 个专家
  3. 总共选择 8 × 4 = 32 候选专家
  4. 从候选中选最终 top-k=8 个
    │
    ▼
Expert Computation + Weighted Sum
```

### Grouped Top-K 实现

```python
def grouped_topk(hidden_states, router_logits):
    """
    Grouped Top-K 路由选择
    
    Args:
        hidden_states: [num_tokens, hidden_size]
        router_logits: [num_tokens, n_routed_experts]
    
    Returns:
        topk_weights: [num_tokens, top_k]
        topk_ids: [num_tokens, top_k]
    """
    num_tokens = hidden_states.shape[0]
    
    # 1. 分组
    scores = torch.softmax(router_logits, dim=-1)
    group_scores = scores.view(num_tokens, n_group, -1)  # [B, 8, 32]
    
    # 2. 每组选择 top-k
    group_topk = group_scores.topk(topk_group, dim=-1)  # 每组选4个
    
    # 3. 组装候选专家
    # 候选数 = n_group × topk_group = 8 × 4 = 32
    
    # 4. 从候选中选最终 top-k
    candidate_scores = ...
    topk_weights, topk_ids = candidate_scores.topk(num_experts_per_tok)
    
    # 5. 归一化权重
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights, topk_ids
```

### NoAux_TC 路由方法

DeepSeek V3 使用 `noaux_tc` (no auxiliary, top-k correction) 路由方法：

```python
class MoEGate(nn.Module):
    def __init__(self, config, quant_config):
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        
        if config.topk_method == "noaux_tc":
            # 添加修正偏置
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(n_routed_experts, dtype=torch.float32)
            )
```

## 共享专家融合

### 融合策略

为了提高效率，可以将共享专家融合到 MoE kernel 中：

```python
# 判断是否融合共享专家
num_fused_shared_experts = n_shared_experts if can_fuse else 0

# 融合后的专家数
total_experts = n_routed_experts + num_fused_shared_experts  # 256 + 1 = 257

# 融合后的 top-k
total_topk = num_experts_per_tok + num_fused_shared_experts  # 8 + 1 = 9
```

### DeepEP 融合

在 DeepEP 后端中，共享专家作为本地槽位融合：

```python
# DeepEP 融合配置
if is_deepep_fusion:
    num_experts_for_moe = n_routed_experts + moe_ep_size  # 256 + EP_size
    top_k_for_moe = num_experts_per_tok + 1  # 8 + 1 = 9
```

## 前向传播

### 标准前向传播

```python
def forward(hidden_states):
    num_tokens, hidden_dim = hidden_states.shape
    
    # 1. 共享专家计算（可并行）
    if shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
    
    # 2. 路由计算
    router_logits = self.gate(hidden_states)  # [num_tokens, 256]
    
    # 3. Top-K 选择
    topk_output = self.topk(hidden_states, router_logits)
    
    # 4. 专家计算
    routed_output = self.experts(hidden_states, topk_output)
    
    # 5. 加权组合
    # output = routed_scaling_factor * routed_output + shared_output
    final_output = routed_output * routed_scaling_factor
    if shared_output is not None:
        final_output += shared_output
    
    return final_output
```

### 双流优化 (SGLang)

```python
def forward_dual_stream(hidden_states):
    """使用双 CUDA 流并行计算共享专家和路由专家"""
    current_stream = torch.cuda.current_stream()
    alt_stream = torch.cuda.Stream()
    
    # 主流：共享专家
    shared_output = self._forward_shared_experts(hidden_states)
    
    # 备流：路由专家
    with torch.cuda.stream(alt_stream):
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_output = self.experts(hidden_states, topk_output)
    
    # 同步并合并
    current_stream.wait_stream(alt_stream)
    final_output = shared_output + routed_output * routed_scaling_factor
    
    return final_output
```

## MoE 后端实现

### 后端选择

| 后端 | 适用场景 | 特点 |
|------|----------|------|
| DeepGEMM | NVIDIA Hopper/Blackwell | FP8 优化，DeepSeek 官方 |
| DeepEP | 大规模 EP | 专家并行优化 |
| FlashInfer MoE | 通用 | FlashInfer 库实现 |
| Cutlass MoE | NVIDIA GPU | Cutlass kernel |
| Triton MoE | 兼容性 | 纯 Triton 实现 |
| Marlin MoE | INT4/INT8 | 量化推理 |

### DeepGEMM 集成

```python
# DeepGEMM 配置
use_deep_gemm = (
    is_cuda and
    device_capability >= 90 and  # Hopper+
    weight.dtype == torch.float8_e4m3fn
)

if use_deep_gemm:
    from deep_gemm import grouped_gemm
    output = grouped_gemm(hidden_states, expert_weights, topk_ids, topk_weights)
```

## 专家并行 (EP)

### EP 实现原理

```python
# EP 配置
ep_size = 16  # 16个GPU
num_local_experts = n_routed_experts // ep_size  # 256 / 16 = 16 experts/GPU

# 每个 GPU 存储部分专家
local_expert_ids = ep_rank * num_local_experts + torch.arange(num_local_experts)

# Token 分发
def dispatch(hidden_states, topk_ids):
    # 1. 根据 topk_ids 确定 token 目标专家
    # 2. All-to-All 通信分发 token
    # 3. 每个GPU只计算本地专家
    ...

# Token 合并
def combine(expert_outputs, topk_weights):
    # 1. All-to-All 通信收集结果
    # 2. 按权重加权求和
    ...
```

### DeepEP A2A Backend

```python
class DeepEPDispatcher:
    """DeepEP All-to-All 通信优化"""
    
    def dispatch(self, hidden_states, topk_ids):
        # 1. 计算 token 分发目标
        send_counts, recv_counts = self.compute_comm(topk_ids)
        
        # 2. 执行 All-to-All 通信
        dispatched = self.all_to_all(hidden_states, send_counts, recv_counts)
        
        return dispatched
    
    def combine(self, expert_output, topk_weights):
        # 1. 反向 All-to-All
        combined = self.all_to_all(expert_output, ...)
        
        # 2. 加权求和
        output = (combined * topk_weights).sum(dim=-1)
        
        return output
```

## 序列并行 MoE

### 实现

```python
# 启用序列并行 MoE
is_sequence_parallel = parallel_config.use_sequence_parallel_moe

if is_sequence_parallel:
    # 1. 对 hidden_states 分块
    hidden_states = sequence_parallel_chunk(hidden_states)
    
    # 2. 每个 TP rank 只处理部分 token
    local_output = self.experts(hidden_states, ...)
    
    # 3. All-Gather 合并结果
    final_output = tensor_model_parallel_all_gather(local_output)
```

## EPLB (Expert Parallelism Load Balancing)

### 动态负载均衡

```python
class EPLBState:
    """专家并行负载均衡"""
    
    def __init__(self, num_experts, ep_size):
        self.expert_distribution = torch.zeros(num_experts)
        self.physical_to_logical = torch.arange(num_experts)
    
    def update_distribution(self, expert_counts):
        """更新专家使用统计"""
        self.expert_distribution += expert_counts
    
    def rebalance(self):
        """重新分配专家以均衡负载"""
        # 1. 分析当前分布
        # 2. 计算新的专家映射
        # 3. 迁移专家权重
        ...
```

## 量化支持

### FP8 MoE

```python
# FP8 量化配置
weight_block_size = [128, 128]  # 块级量化

# FP8 权重格式
weight.dtype = torch.float8_e4m3fn

# FP8 计算
output = fp8_grouped_gemm(
    hidden_states,  # BF16/FP16 输入
    weight,         # FP8 权重
    scale,          # Per-block scale
    topk_ids,
    topk_weights
)
```

### INT4/INT8 MoE

```python
# INT8 量化
weight.dtype = torch.int8

# W4A8 混合精度
weight.dtype = torch.uint8  # INT4 packed
activation_cast = torch.float8_e4m3fn
```

## 性能优化

### 预取优化

```python
# Expert 权重预取
class ExpertPrefetcher:
    def prefetch_next_layer(self, layer_id, topk_ids):
        """预取下一层的专家权重"""
        next_layer_experts = predict_experts(topk_ids)
        prefetch_weights(next_layer_experts)
```

### CUDA Graph 支持

```python
# MoE 支持 CUDA Graph
# 需要：
# 1. 固定 top-k 选择模式
# 2. 预分配输出缓冲区
# 3. 避免动态形状

can_use_cuda_graph = (
    batch_size in captured_batch_sizes and
    use_static_shape
)
```

## 权重加载

### 专家权重映射

```python
def make_expert_params_mapping(num_experts, num_redundant_experts):
    """创建专家权重参数映射"""
    mapping = []
    for expert_id in range(num_experts + num_redundant_experts):
        mapping.extend([
            ("experts.w13_weight", "gate_proj", expert_id, 0),
            ("experts.w13_weight", "up_proj", expert_id, 1),
            ("experts.w2_weight", "down_proj", expert_id, None),
        ])
    return mapping
```

### 权重加载流程

```python
def load_weights(weights):
    for name, weight in weights:
        if "experts" in name:
            # 解析专家ID
            expert_id = parse_expert_id(name)
            
            # 加载到对应位置
            param.weight_loader(param, weight, expert_id=expert_id)
        elif "shared_experts" in name:
            # 加载共享专家
            ...
        elif "gate" in name:
            # 加载路由权重
            ...
```
