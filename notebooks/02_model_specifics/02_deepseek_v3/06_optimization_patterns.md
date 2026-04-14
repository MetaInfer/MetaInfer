# DeepSeek V3 推理优化模式

## 概述

本文档总结了 vLLM 和 SGLang 中针对 DeepSeek V3 的核心推理优化技术。

## 1. MLA 权重吸收

### 原理

在 Decode 阶段，将 KV 恢复投影与注意力计算融合：

```
传统:
  K = W_k @ kv_c    →    Attention(Q, K, V)
  V = W_v @ kv_c

优化后:
  Q' = Q @ W_k      →    Attention(Q', kv_c, kv_c) @ W_v
```

### 效果

- 减少内存访问
- 避免显式恢复完整 KV
- 特别适合小 batch decode 场景

## 2. DeepGEMM FP8 GEMM

### 原理

DeepSeek 官方的 FP8 矩阵乘法优化：

```python
# 配置
weight_block_size = [128, 128]  # 块级量化
weight_dtype = torch.float8_e4m3fn

# DeepGEMM 调用
output = deep_gemm.grouped_gemm(
    hidden_states,    # BF16
    weight,           # FP8
    scale,            # Per-block scale
    topk_ids,
    topk_weights
)
```

### 启用条件

```python
use_deep_gemm = (
    is_cuda and
    device_capability >= 90 and  # Hopper/Blackwell
    weight.dtype == torch.float8_e4m3fn and
    SGLANG_ENABLE_JIT_DEEPGEMM  # 默认启用
)
```

### 预编译

```bash
# 预编译 DeepGEMM kernels (推荐)
python -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8
```

## 3. 共享专家融合

### 原理

将共享专家作为"第257个专家"融合到 MoE kernel：

```python
# 融合配置
num_fused_shared_experts = 1  # 融合1个共享专家

# 专家布局
total_experts = n_routed_experts + num_fused_shared_experts  # 257

# Top-K
total_topk = num_experts_per_tok + num_fused_shared_experts  # 9
```

### 条件

```python
can_fuse = (
    is_cuda and
    device_capability >= 80 and
    n_routed_experts == 256 and
    n_shared_experts == 1 and
    not is_sbo_enabled() and
    not is_tbo_enabled()
)
```

### DeepEP 融合

在 DeepEP 后端，共享专家作为本地专家槽位：

```python
# DeepEP 融合
num_experts = n_routed_experts + ep_size  # 256 + 16 = 272 (EP=16)
top_k = num_experts_per_tok + 1           # 8 + 1 = 9
```

## 4. 双流并行

### 原理

使用两个 CUDA 流并行计算共享专家和路由专家：

```python
def forward_dual_stream(hidden_states):
    current_stream = torch.cuda.current_stream()
    alt_stream = self.alt_stream  # 预分配的备流
    
    # 主流：共享专家
    shared_output = self._forward_shared_experts(hidden_states)
    
    # 备流：路由专家
    with torch.cuda.stream(alt_stream):
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_output = self.experts(hidden_states, topk_output)
    
    # 同步
    current_stream.wait_stream(alt_stream)
    
    return shared_output + routed_output * routed_scaling_factor
```

### 效果

- 共享专家和路由专家并行计算
- 充分利用 GPU SM
- 减少整体延迟

## 5. Data Parallel Attention

### 原理

在 Attention 层使用数据并行，减少 KV Cache 冗余：

```
传统 TP=8:
  每个 TP rank 存储完整 KV → 8x 冗余

DP Attention:
  只在 DP rank 0 存储 KV → 1x 存储
  其他 rank 独立计算（无 KV Cache）
  MoE 前同步
```

### 启用

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --enable-dp-attention \
    --tp 8 \
    --dp 8
```

### 效果

- KV Cache 大小减少 8x
- 支持更大 batch size
- 高吞吐场景显著提升

## 6. CUDA Graph 优化

### 支持范围

DeepSeek V3 的以下组件支持 CUDA Graph：

- MLA Attention
- MoE (需要静态路由模式)
- MTP 层

### 配置

```bash
# 指定捕获的 batch size
--cuda-graph-bs 1 2 4 8 16 32

# 大 batch MTP 需要调整
--max-running-requests 128
```

### 条件

```python
can_use_cuda_graph = (
    batch_size in captured_batch_sizes and
    not dynamic_sequence_length and
    not use_custom_routing
)
```

## 7. Chunked Prefill

### 原理

将长 prefill 序列分块处理：

```python
chunk_size = 8192

for chunk_start in range(0, seq_len, chunk_size):
    chunk_end = min(chunk_start + chunk_size, seq_len)
    chunk_hidden = hidden_states[chunk_start:chunk_end]
    chunk_positions = positions[chunk_start:chunk_end]
    
    # 处理块
    output = self.forward(chunk_hidden, chunk_positions, ...)
    
    # 合并状态
    ...
```

### 好处

- 减少峰值显存
- 支持更长序列
- 更好的调度灵活性

## 8. TBO (Two-Batch Overlap)

### 原理

重叠两个 batch 的计算：

```python
# Batch 1: MoE 阶段
# Batch 2: Attention 阶段
# 两个阶段并行执行
```

### 启用条件

```python
can_run_tbo = (
    num_layers > first_k_dense_replace and
    batch_size_meets_threshold
)
```

## 9. SBO (Single-Batch Overlap)

### 原理

在单个 batch 内重叠不同操作：

```python
# 共享专家与 MoE dispatch 重叠
# MoE down GEMM 与 MoE combine 重叠
```

### 配置

```python
# SBO 相关环境变量
SGLANG_ENABLE_SBO = 1
SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = 1
```

## 10. Large-Scale EP

### 配置示例

```bash
# 96 GPU 大规模 EP
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 1 \
    --ep 96 \
    --disable-shared-experts-fusion \
    --enable-dp-attention
```

### PD Disaggregation

Prefill-Decode 分离部署：

```bash
# Prefill 节点
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --port 30000 \
    --pd-separate "prefill" \
    ...

# Decode 节点
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --port 30001 \
    --pd-separate "decode" \
    ...
```

## 性能基准

### H200 (8x) 性能

| 配置 | 吞吐量 (tokens/s) | 延迟 (ms) |
|------|-------------------|-----------|
| BF16 | 52,000 | 15 |
| FP8 | 85,000 | 10 |
| FP8 + MTP | 130,000 | 8 |

### MI300X (8x) 性能

| 配置 | 吞吐量 (tokens/s) |
|------|-------------------|
| FP8 | 75,000 |
| BF16 | 48,000 |

## 优化选择指南

| 场景 | 推荐优化组合 |
|------|--------------|
| 单请求低延迟 | CUDA Graph + MTP |
| 高吞吐服务 | DP Attention + DeepGEMM |
| 长序列 | NSA + CP + Chunked Prefill |
| 多节点部署 | EP + PD Disaggregation |
| 资源受限 | 量化 (INT8/W4A8) |

## 环境变量参考

```bash
# DeepGEMM
SGLANG_ENABLE_JIT_DEEPGEMM=1          # 启用 DeepGEMM

# CUDA Graph
SGLANG_DISABLE_CUDA_GRAPH=0           # 启用 CUDA Graph

# MTP
SGLANG_ENABLE_SPEC_V2=1               # 启用 overlap scheduler

# DP Attention
SGLANG_ENABLE_DP_ATTENTION=1          # 启用 DP Attention

# 调试
SGLANG_DEEPEP_BF16_DISPATCH=1         # DeepEP BF16 调度
```
