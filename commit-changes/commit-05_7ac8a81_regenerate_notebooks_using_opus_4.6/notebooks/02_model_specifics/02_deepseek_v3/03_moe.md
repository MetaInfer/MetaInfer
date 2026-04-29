# DeepSeek V3 - Mixture of Experts (MoE)

## Architecture

DeepSeek V3 uses a **fine-grained MoE** with shared experts, which differs from simpler MoE architectures like Mixtral.

### Comparison

| Feature | Mixtral | DeepSeek V3 |
|---------|---------|-------------|
| Total experts | 8 | 256 |
| Active experts per token | 2 | 8 (of 256 routed) |
| Shared experts | None | 1-2 (always active) |
| Expert granularity | Coarse (each expert = full FFN) | Fine (each expert = small FFN) |
| Routing | Simple top-k | Grouped top-k |

## Structure

```
Input hidden_state [B, H]
    ↓
[Router/Gate: Linear(H → N_experts)]  → routing scores
    ↓
[Grouped Top-K Selection]  → select K experts from N
    ↓
┌──────────────────┐
│ Shared Expert(s)  │ → always compute, no routing
│ gate_up → SiLU*x  │
│ down → output     │
└──────────────────┘
        +
┌──────────────────┐
│ Routed Experts   │ → only selected K experts compute
│ For each active:  │
│   gate_up → SiLU  │
│   down → output   │
│ Weighted sum      │
└──────────────────┘
    ↓
Final output = shared_output + routed_output
```

## Grouped Top-K Routing

Instead of simple top-k over all experts, DeepSeek groups experts and selects from each group:

```python
def grouped_topk(scores, n_groups, topk_group, top_k):
    # 1. Reshape scores into groups
    grouped = scores.reshape(batch, n_groups, experts_per_group)

    # 2. Select top groups first
    group_scores = grouped.max(dim=-1).values  # Best score per group
    top_groups = group_scores.topk(topk_group).indices

    # 3. Zero out non-selected groups
    mask = zeros_like(scores)
    mask[top_groups] = 1
    scores = scores * mask

    # 4. Select top-k from remaining
    topk_indices = scores.topk(top_k).indices
    topk_weights = softmax(scores[topk_indices])

    return topk_indices, topk_weights
```

This ensures diversity across expert groups rather than concentrating on similar experts.

## Fused MoE Kernel

For efficiency, MoE computation is fused into a single kernel that:
1. Performs token-to-expert routing
2. Applies the gate+up projection for each active expert
3. Applies SiLU activation
4. Applies the down projection
5. Accumulates weighted results

### Triton Fused MoE Pattern (mini-sglang)
```python
def fused_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """
    w1: [num_experts, 2*intermediate_size, hidden_size]  (gate+up merged)
    w2: [num_experts, hidden_size, intermediate_size]     (down)
    """
    # 1. Sort tokens by expert assignment
    sorted_token_ids, expert_ids, num_tokens_per_expert = sort_by_expert(topk_ids)

    # 2. Fused kernel: for each expert's token block
    #    a. Multiply by w1 (gate_up)
    #    b. Apply SiLU activation on gate portion
    #    c. Multiply by w2 (down)
    #    d. Accumulate with expert weight

    fused_moe_kernel[grid](
        hidden_states, w1, w2,
        sorted_token_ids, expert_ids, num_tokens_per_expert,
        topk_weights, ...
    )
```

## Expert Parallelism (EP)

For very large MoE models, experts can be distributed across GPUs:

### Option 1: TP within experts
Each expert's weights are sharded across TP ranks (same as dense model):
```
Rank 0: Expert[i].w1[:, :half], Expert[i].w2[:half, :]
Rank 1: Expert[i].w1[:, half:], Expert[i].w2[half:, :]
```

### Option 2: EP across experts
Different experts live on different GPUs:
```
Rank 0: Expert[0..63]
Rank 1: Expert[64..127]
Rank 2: Expert[128..191]
Rank 3: Expert[192..255]
```
Requires all-to-all communication to route tokens to the right GPU.

## Impact on Inference Framework

1. **Memory**: Expert weights are large (256 experts × FFN params). Must plan GPU memory carefully
2. **Compute**: Only K experts are active per token → effective compute similar to dense model
3. **Communication**: EP requires all-to-all; TP+MoE requires careful design
4. **Kernel**: Fused MoE kernels are essential for performance; naive per-expert computation is prohibitively slow
5. **Load balancing**: Token distribution across experts affects throughput
