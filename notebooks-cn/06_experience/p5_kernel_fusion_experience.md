# P5 Kernel Fusion 实践经验

## 概览

| 子项 | 模型 | 吞吐变化 | 方法 |
|------|------|---------|------|
| P5a | Qwen3-8B | 12.58 → 12.76 (+1.4%) | gate_up_proj 合并 GEMM |
| P5b | DeepSeek-V2-Lite | 13.08 → 13.20 (+0.9%) | GPU-side expert_map hybrid |

---

## P5a: Qwen gate_up_proj 合并 GEMM

### 动机

Qwen MLP 前向:
```
x → gate_proj(x)  [GEMM 1]
x → up_proj(x)    [GEMM 2]
h = silu(gate) * up  [2 elementwise + 1 mul]
```

每次 decode 多 1 次 GEMM kernel launch。merge 为单次 GEMM 省掉这个 launch。

### 实现

**`engine/tp_layers/linear.py`**: 新增 `MergedColumnParallelLinear`

权重组装为 `[2 * local_intermediate, hidden_size]`，一次 `F.linear` 出 `[B, S, 2*I]`。前半是 gate，后半是 up。

关键: `load_weight_shard(gate_weight, up_weight)` 分别从 safetensors 读 `gate_proj.weight` 和 `up_proj.weight`（TP shard），拼接为合并权重。

**`engine/models/qwen.py`**: `QwenMLPTP` forward

```python
# BEFORE: 2 GEMM + 3 elementwise
gate = self.gate_proj(x)
up = self.up_proj(x)
h = F.silu(gate) * up

# AFTER: 1 GEMM + 3 elementwise
gate_up = self.gate_up_proj(x)
d = gate_up.shape[-1] // 2
h = F.silu(gate_up[..., :d]) * gate_up[..., d:]
```

### 尝试过的失败方案

**Triton `silu_and_mul` kernel**: 将 `F.silu(gate) * up` 融合为单个 Triton kernel。

| 版本 | 吞吐 (tok/s) | 变化 |
|------|-------------|------|
| baseline | 12.58 | — |
| Triton fused | 7.08 | **-43.7%** |

**失败原因**: Triton kernel 对 [1, 1, 3072]（decode single token）的小 tensor launch overhead (~10μs) 远大于 PyTorch 原生 elementwise 的 launch (~2μs)。batch=1 下 Triton 的开销吃掉所有收益反而倒亏。

### 经验

1. **Triton kernel 不适合极小的 tensor**（< 10K 元素）。PyTorch 原生 elementwise 有极低的 launch overhead。
2. 合并 GEMM 总是有利的（省一次 launch），激活融合不一定。
3. 基准测试一定要跑——单 GPU 看似正确，实际 TP=4 benchmark 才显真相。

---

## P5b: DeepSeek MoE GPU-side expert mapping

### 动机

DeepSeek-V2-Lite 60 层 MoE，每层 top_k=6。原代码:

```python
for token_i in range(flat.shape[0]):      # Python 循环
    for k in range(self.cfg.top_k):        # Python 循环
        expert_id = int(topk_idx_flat[token_i, k].item())  # GPU→CPU sync!
        if str(expert_id) not in self.experts:
            continue
        local_out += self.experts[key](x) * w
```

每步 decode: 60 × 6 = **360 次 `.item()`** GPU→CPU 同步。

### 核心矛盾分析

#### .item() 的开销

每次 `.item()` 触发：
1. GPU 完成所有 outstanding CUDA work（pipeline flush）
2. GPU→CPU 内存拷贝（D2H transfer）
3. CPU 侧读取值

单次 ~5μs。decode 阶段 360 次总计 ~1.8ms。在端到端 ~76ms/step 中占 **~2.4%**。

#### Batched nonzero 的开销

GPU 侧方案: 为每个 local expert 做 `mask = (local_eids == local_idx)` 然后 `nonzero()`。

每次 mask 操作 = 一次 CUDA kernel launch（~3μs）。40 个 local expert = 40 次 mask × 3μs = **120μs/层**。60 层 = 7.2ms，还不算 nonzero、index_add_ 的额外 launch。

```
核心矛盾:
  旧方案: 6 sync × 5μs  = 30μs/层   (按需付费)
  新方案: 40 kernel × 3μs  = 120μs/层  (每 expert 固定成本)
  
  30μs < 120μs → decode 旧方案胜
```

#### 为什么 prefill 反转

Prefill 的 seqlen=5（5 tokens）改变了游戏规则：

**旧方案 prefill**:
- 1800 次 `.item()` sync（5×6×60）
- 每个 expert 被调用时只处理 **1 个 token** → 5 次独立的 GEMV [1, H]×[H, I]
- GEMV 的 tensor core 利用率极低（128×128 tile 只用了 1 行）

**新方案 prefill**:
- 40 mask kernel（fix cost，不随 token 增长）
- `flat[token_indices]` 一次传 [5, H] → GEMM [5, H]×[H, I]
- tensor core tile 利用率大幅提升

```
GEMV×5:  5 kernel launch × 每个只算 1 行 → ~500 us
GEMM×1:  1 kernel launch × 一次算 5 行    → ~100 us
```

GEMM batching 收益（~400μs）> mask kernel 开销（~120μs）。prefill 新方案胜。

### 实现

**GPU-side expert_map** (参考 vLLM 模式):

```python
# 初始化: 映射 global_id → local_idx or -1
m = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
for local_idx, global_eid in enumerate(self.local_expert_ids):
    m[global_eid] = local_idx
self._expert_map = m  # 备于 GPU，随 .to(device) 移动
```

**Hybrid forward** (根据 token 数量选策略):

```python
if num_tokens > 4:  # prefill
    local_eids = expert_map[topk_idx_flat.long()]  # GPU tensor 索引，零 sync
    for local_idx, global_eid in enumerate(self.local_expert_ids):
        mask = (local_eids == local_idx)
        if not mask.any():
            continue
        token_idx, k_idx = mask.nonzero(as_tuple=True)
        expert_out = self.experts[str(global_eid)](flat[token_idx])  # batched!
        local_out.index_add_(0, token_idx, expert_out * weights)
else:  # decode: .item() 循环 (已存在，不改)
    for token_i in range(num_tokens):
        for k in range(self.cfg.top_k):
            expert_id = int(topk_idx_flat[token_i, k].item())
            ...
```

阈值 `num_tokens > 4` 对应 seqlen ≥ 5 的正常 prefill。

### 为什么不会 TP 死锁

vLLM 核心 insight: **所有 rank 永远执行相同的 GPU kernel grid**。

| Rank | expert_map 内容 | mask 命中 | 行为 |
|------|----------------|----------|------|
| 0 | expert 0-39=0..39, 其余=-1 | 只有 0-39 命中 | nonzero→forward→index_add |
| 1 | expert 40-79=0..39, 其余=-1 | 只有 40-79 命中 | nonzero→forward→index_add |

每个 rank 都遍历 40 个 local expert、做 40 次 mask、跑 40 次 nonzero。只是不同 rank 的 mask 命中不同的 expert。

`all_reduce_sum` 在最后被所有 rank 以相同 shape 的张量调用。非本地 expert 的 local_out 保持零，被 harmless sum 进去。

### 尝试过的失败方案

| 方案 | 结果 | 原因 |
|------|------|------|
| `.cpu()` 批量同步 | TP prefill 挂死 | CPU sync 点在 GPU pipeline 的位置因 rank 而异，时序分歧 |
| 全部 batched nonzero | -15.0% (11.12 tok/s) | decode 时 40 mask kernel > 6 .item() |
| **Hybrid (prefill:batched, decode:.item())** | **+0.9% (13.20 tok/s)** | 各取所长 |

### 经验

1. **Kernel launch 不是免费的** — 40 个 elementwise mask 的开销可以大于 360 次 GPU sync。
2. **GEMM batching 是真正的收益源** — 把 N 次 GEMV 合成 1 次 GEMM 远比消除 `.item()` 本身重要。
3. **Hybrid 策略在 token 数两极分化时最优** — decode 几乎永远是 1 token，prefill 几乎永远 >4 tokens。硬要统一路径反而不如分别处理。
4. **vLLM 的 expert_map 模式是正确的** — GPU 侧 tensor 索引既高效又不会死锁。核心是"所有 rank 执行相同 kernel grid"这一约束。
5. **完整的 Fused MoE kernel 在 batch=1 时收益有限** — P1 两次尝试 -16.8%/-25%，需要等 batch>1（P4/P2 之后）才有意义。

---

## 文件变更清单

| 文件 | 变更 |
|------|------|
| `engine/tp_layers/linear.py` | 新增 `MergedColumnParallelLinear` |
| `engine/models/qwen.py` | `QwenMLPTP` 改用合并 GEMM + 权重加载适配 |
| `engine/kernels/triton_activation.py` | 新建 (Triton kernel 保留但未启用) |
| `engine/tp_layers/moe.py` | 新增 `_expert_map` + hybrid forward |
