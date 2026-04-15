# Qwen3 MoE — 混合专家模型结构

## 整体结构

Qwen3 MoE 与 Dense 变体共享 Attention 层（包括 QK Norm），区别在于 MLP 被替换为 MoE 层。

```
Qwen3MoeForCausalLM
├── model: Qwen3MoeModel
│   ├── embed_tokens: VocabParallelEmbedding(vocab_size, hidden_size)
│   ├── layers: ModuleList[Qwen3MoeDecoderLayer × num_hidden_layers]
│   │   └── Qwen3MoeDecoderLayer
│   │       ├── input_layernorm: RMSNorm(hidden_size, eps)
│   │       ├── self_attn: Qwen3Attention              ← 与 Dense 完全一致（含 QK Norm）
│   │       │   ├── qkv_proj, q_norm, k_norm, rotary_emb, attn, o_proj
│   │       ├── post_attention_layernorm: RMSNorm(hidden_size, eps)
│   │       └── mlp: Qwen3MoeSparseMoeBlock             ← 替换为 MoE
│   │           ├── gate: Linear(hidden_size → num_experts, bias=False)
│   │           └── experts: FusedMoE
│   │               ├── gate_up_proj: [num_experts, 2*moe_intermediate_size, hidden_size]
│   │               └── down_proj: [num_experts, hidden_size, moe_intermediate_size]
│   └── norm: RMSNorm(hidden_size, eps)
└── lm_head: ParallelLMHead(hidden_size → vocab_size)
```

## 与 Dense 变体的对比

```
Dense DecoderLayer:                    MoE DecoderLayer:
├── input_layernorm                    ├── input_layernorm        (相同)
├── self_attn (含 QK Norm)             ├── self_attn (含 QK Norm) (相同)
├── post_attention_layernorm           ├── post_attention_layernorm(相同)
└── mlp: Qwen3MLP                     └── mlp: Qwen3MoeSparseMoeBlock
    ├── gate_up_proj [H→2I]                ├── gate [H→E]  (Router)
    ├── SiluAndMul                         └── experts (FusedMoE)
    └── down_proj [I→H]                       ├── gate_up_proj [E, 2I', H]
                                               ├── SiluAndMul (per expert)
                                               └── down_proj [E, H, I']
```

**关键区别**：
- Dense：一个大 MLP，`intermediate_size` 可能是 12288
- MoE：多个小 MLP（专家），`moe_intermediate_size` 可能是 1536，但有 128 个专家

## MoE 层前向传播

```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        self.num_experts = config.num_experts          # 如 128
        self.top_k = config.num_experts_per_tok        # 如 8
        self.norm_topk_prob = config.norm_topk_prob    # True

        # Router: 简单的线性层
        self.gate = Linear(config.hidden_size, config.num_experts, bias=False)

        # 所有专家的权重打包在一起（高效 fused kernel）
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )

    def forward(self, hidden_states):
        # Step 1: Router 计算路由分数
        router_logits = self.gate(hidden_states)      # [B, num_experts]

        # Step 2: FusedMoE 执行专家计算
        # 内部流程:
        #   a. softmax(router_logits) → routing weights
        #   b. topk(weights, k=top_k) → 选出 top_k 个专家
        #   c. (可选) 如果 norm_topk_prob: 重归一化 top_k weights 使其和为 1
        #   d. 对每个 token, 将其发送到被选中的专家
        #   e. 每个专家: gate_up_proj → SiLU*Mul → down_proj
        #   f. 加权求和各专家输出
        output = self.experts(hidden_states, router_logits)

        return output
```

### Router 详解

```python
def route(hidden_states, gate_weight, top_k, norm_topk_prob):
    # 计算路由分数
    router_logits = hidden_states @ gate_weight.T      # [B, num_experts]

    # Softmax 得到概率
    routing_weights = softmax(router_logits, dim=-1)   # [B, num_experts]

    # 选择 Top-K 专家
    topk_weights, topk_ids = routing_weights.topk(top_k, dim=-1)
    # topk_weights: [B, top_k]，topk_ids: [B, top_k]

    # 重归一化（norm_topk_prob=True）
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
```

### Fused MoE Kernel

所有专家的计算被融合到一个 kernel 中，避免逐专家执行的开销：

```python
class FusedMoE:
    def __init__(self, num_experts, top_k, hidden_size, intermediate_size, renormalize):
        # 所有专家的权重打包为 3D 张量
        # gate + up 合并: [num_experts, 2 * intermediate_size_per_tp, hidden_size]
        self.gate_up_proj = Parameter(torch.empty(
            num_experts, 2 * intermediate_size // tp_size, hidden_size
        ))
        # down: [num_experts, hidden_size, intermediate_size_per_tp]
        self.down_proj = Parameter(torch.empty(
            num_experts, hidden_size, intermediate_size // tp_size
        ))

    def forward(self, hidden_states, router_logits):
        # 使用 Triton fused_moe kernel
        return fused_moe(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,       # gate_up 合并权重
            w2=self.down_proj,          # down 权重
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation="silu",
        )
```

## Shared Expert（共享专家，可选）

某些 Qwen3 MoE 模型具有共享专家（类似 DeepSeek V3）：

```python
class Qwen3MoeSparseMoeBlockWithShared(nn.Module):
    def __init__(self, config):
        # 路由专家（同上）
        self.gate = Linear(hidden_size, num_experts, bias=False)
        self.experts = FusedMoE(...)

        # 共享专家：一个普通的 MLP，对所有 token 都执行
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
            )
            self.shared_expert_gate = Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        # 路由专家输出
        router_logits = self.gate(hidden_states)
        routed_output = self.experts(hidden_states, router_logits)

        # 共享专家输出
        if hasattr(self, 'shared_expert'):
            shared_output = self.shared_expert(hidden_states)
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = routed_output + shared_output * shared_gate
        else:
            output = routed_output

        return output
```

## decoder_sparse_step（部分稀疏）

vllm 实现中支持 `decoder_sparse_step` 配置：只有部分层是 MoE，其余是 Dense MLP：

```python
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = Qwen3Attention(config)  # 所有层 Attention 相同

        # 根据 decoder_sparse_step 决定 MLP 类型
        is_moe_layer = (layer_idx + 1) % config.decoder_sparse_step == 0
        if is_moe_layer:
            self.mlp = Qwen3MoeSparseMoeBlock(config)  # MoE
        else:
            self.mlp = Qwen3MLP(config)                 # Dense
```

**注意**：当前主流 Qwen3 MoE 模型（如 Qwen3-235B-A22B）的所有层都是 MoE（`decoder_sparse_step=1`），但代码框架应支持混合配置。

## HuggingFace Config 字段（MoE 特有）

```python
moe_config_fields = {
    # === 基础字段（与 Dense 共享） ===
    "hidden_size": int,
    "num_hidden_layers": int,
    "num_attention_heads": int,
    "num_key_value_heads": int,
    "head_dim": int,
    "vocab_size": int,
    "max_position_embeddings": int,
    "rms_norm_eps": float,
    "rope_theta": float,
    "hidden_act": str,
    "attention_bias": bool,         # Qwen3: False
    "tie_word_embeddings": bool,

    # === MoE 特有字段 ===
    "num_experts": int,              # 总专家数，如 128
    "num_experts_per_tok": int,      # 每 token 激活的专家数，如 8
    "moe_intermediate_size": int,    # 每个专家的 intermediate size，如 1536
    "intermediate_size": int,        # Dense 层的 intermediate size（如果有 dense 层）
    "norm_topk_prob": bool,          # 是否重归一化 top-k 权重，通常 True
    "decoder_sparse_step": int,      # MoE 层间隔，1=全部 MoE
    "shared_expert_intermediate_size": int,  # 共享专家 intermediate size，0=无共享专家
}
```

## 权重名称映射

### MoE 层权重（与 Dense 层的区别）

```
# Dense MLP 权重（如果存在 dense 层）
model.layers.{i}.mlp.gate_proj.weight              [intermediate_size, hidden_size]
model.layers.{i}.mlp.up_proj.weight                [intermediate_size, hidden_size]
model.layers.{i}.mlp.down_proj.weight              [hidden_size, intermediate_size]

# MoE 层权重
model.layers.{i}.mlp.gate.weight                   [num_experts, hidden_size]
model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [moe_intermediate_size, hidden_size]
model.layers.{i}.mlp.experts.{j}.up_proj.weight    [moe_intermediate_size, hidden_size]
model.layers.{i}.mlp.experts.{j}.down_proj.weight  [hidden_size, moe_intermediate_size]

# 共享专家权重（如果有）
model.layers.{i}.mlp.shared_expert.gate_proj.weight  [shared_intermediate_size, hidden_size]
model.layers.{i}.mlp.shared_expert.up_proj.weight    [shared_intermediate_size, hidden_size]
model.layers.{i}.mlp.shared_expert.down_proj.weight  [hidden_size, shared_intermediate_size]
model.layers.{i}.mlp.shared_expert_gate.weight       [1, hidden_size]
```

### 权重打包

推理框架加载时将逐专家权重打包为 3D 张量：
```python
# HF: experts.{j}.gate_proj.weight [I, H] × num_experts
# HF: experts.{j}.up_proj.weight   [I, H] × num_experts
# → 推理: gate_up_proj [num_experts, 2*I, H]

# HF: experts.{j}.down_proj.weight [H, I] × num_experts
# → 推理: down_proj [num_experts, H, I]
```

加载逻辑：
```python
def load_expert_weights(model, checkpoint):
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            gate_w = checkpoint[f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"]
            up_w = checkpoint[f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"]
            down_w = checkpoint[f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"]

            # TP 切分
            gate_w_shard = gate_w[tp_rank * shard_size : (tp_rank + 1) * shard_size]
            up_w_shard = up_w[tp_rank * shard_size : (tp_rank + 1) * shard_size]

            # 打包 gate+up
            model.layers[layer_idx].mlp.experts.gate_up_proj[expert_idx] = \
                torch.cat([gate_w_shard, up_w_shard], dim=0)

            # down 切分
            model.layers[layer_idx].mlp.experts.down_proj[expert_idx] = \
                down_w[:, tp_rank * shard_size : (tp_rank + 1) * shard_size]
```

## TP 切分方式（MoE 特有）

| 权重 | 切分方式 | 通信 |
|------|---------|------|
| `gate` (Router) | **完整复制**（所有 rank 需要完整路由） | 无 |
| `experts.gate_up_proj` | 每个专家沿 intermediate 维切分 | 无 |
| `experts.down_proj` | 每个专家沿 intermediate 维切分 | **all-reduce** |
| `shared_expert` | 同 Dense MLP 切分 | **all-reduce** |

## 代码生成模板

```python
class Qwen3MoEMLP(nn.Module):
    def __init__(self, config):
        # Router
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)

        # Fused experts
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )

    def forward(self, x):
        router_logits = self.gate(x)
        return self.experts(x, router_logits)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)   # 与 Dense 完全相同
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MoEMLP(config)            # 替换为 MoE
```

## 源码参考

| 项目 | 文件 |
|------|------|
| mini-sglang | `minisgl/models/qwen3_moe.py` + `minisgl/models/utils.py:MoEMLP` + `minisgl/layers/moe.py` |
| vllm | `vllm/model_executor/models/qwen3_moe.py` |
| sglang | `srt/models/qwen3_moe.py` |
