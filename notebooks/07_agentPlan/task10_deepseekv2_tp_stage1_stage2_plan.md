# Task10: DeepSeekV2 TP Adaptation Plan (Phase 1 + Phase 2)

This document only defines two things:
- Phase 1: `engine/tp_layers/moe.py` `ExpertParallelMoE` skeleton (simplified EP via AllReduce).
- Phase 2: `engine/models/deepseek_v2.py` `DeepseekAttentionTP` MLA structure definition.

Does not include Phase 3 weight loading implementation, does not modify existing source code.

---

## 0. Goals and Boundaries

### 0.1 Current Goals (Must Implement)
- Prioritize supporting **DeepSeekV2** TP (target first run `tp_size=4`).
- Structurally预留 for **DeepSeekV3** extensibility (not implementing MTP/NSA/DeepEP in this phase).
- Only output "code skeleton design document", not落地 implementation.

### 0.2 Explicitly Not Doing (Prohibited This Round)
- Not implementing Phase 3 (`load_weights` lazy slicing/OOM protection).
- Not introducing full EP all-to-all communication (only simplified local expert + all_reduce aggregation).
- No fused kernel/FlashMLA/FP8 optimization.

---

## 1. Overall Design Principles (Inherited from Qwen TP Experience)

Hard constraints from `notebooks-cn/06_experience/01_task10_tp_qwen_debug_experience.md`, directly applied to DeepSeekV2:

1. **Model source alignment priority**
   Attention/RoPE/Norm must use official implementation semantics, no "approximate implementation".
2. **No double slicing**
   Parameter loading layer and module internal layer must约定 "input is full or local shard", avoid double slicing causing empty tensors.
3. **Correctness before optimization**
   First use verifiable path (single sample/low concurrency) to run through, then do batched/fused optimization.
4. **RoPE rotation style must be explicitly标注**
   This task fixes Neox style (first-half/second-half), prohibits mixing even-odd interleaving style.
5. **Norm numerical path must be stable**
   Key numerical paths in attention/normalization prefer fp32 computation then cast back.

---

## 2. Phase 1: `engine/tp_layers/moe.py` Skeleton

## 2.1 File and Export Goals
- New file: `engine/tp_layers/moe.py`
- Exports:
  - `ExpertParallelMoEConfig`
  - `ExpertParallelMoE`
  - (Optional) `partition_experts_for_rank`

## 2.2 Config Structure (Suggested)

```python
@dataclass
class ExpertParallelMoEConfig:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    tp_size: int
    tp_rank: int
    # DeepSeekV2 doesn't implement shared experts details first,保留 field for V3 extension
    num_shared_experts: int = 0
    score_function: str = "softmax"  # or sigmoid
    route_scale: float = 1.0
```

Notes:
- `num_experts` based on model config (document can use 160 as example, but implementation must not hardcode).
- `top_k` default read from DeepSeekV2 config.
- `num_shared_experts`保留 extension位, Phase 1 can not enable.

## 2.3 Module Topology

### Router (Gate, Replicated)
- Each rank保持同一份 router weights.
- Computes `router_logits[B, T, E]` for input `hidden_states[B, T, H]`.
- Per-token top-k: gets `topk_idx[B, T, K]` and `topk_weight[B, T, K]`.

### Experts (Expert IDs evenly distributed to TP ranks)
- Evenly split `[0, num_experts)` into `tp_size` parts.
- Current rank only instantiates `local_expert_ids` corresponding expert modules.
- Expert suggested structure (skeleton):
  - `w1/gate_up` (can merge) + activation
  - `w2/down`

## 2.4 Forward Semantics (Simplified EP via AllReduce)

Input: `hidden_states: [B, T, H]`
Output: `moe_out: [B, T, H]`

Steps:
1. Router computes top-k on this rank (same on all ranks).
2. Initialize `local_out = zeros_like(hidden_states)`.
3. Iterate token's top-k experts:
   - If expert belongs to this rank: execute expert forward, multiply routing weight, accumulate to `local_out`.
   - If expert doesn't belong to this rank: skip (this rank contributes 0).
4. `global_out = all_reduce_sum(local_out)`, get complete MoE output.
5. Return `global_out`.

This approach优点:
- Simple, easy to verify, and consistent with current TP library (only depends on all_reduce).
- No need for all-to-all or complex token dispatch.

Cost:
- Effective compute utilization不如 complete EP; can evolve to all-to-all later.

## 2.5 Skeleton Interface Definition (Suggested)

```python
class ExpertParallelMoE(nn.Module):
    def __init__(self, cfg: ExpertParallelMoEConfig):
        ...
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()
        self.local_expert_ids = partition_experts_for_rank(...)
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)  # replicated
        self.experts = nn.ModuleDict({...})  # only local experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 1) router -> topk
        # 2) local experts accumulate
        # 3) all_reduce_sum
        return out
```

## 2.6 Essential Error Prevention Points (Phase 1)

1. **top-k consistency**
   All ranks must base on same input to compute same `topk_idx`, otherwise all_reduce result meaningless.
2. **Expert归属 boundary**
   `expert_id` to `rank` mapping function must be unit test covered (first/last/remainder scenarios).
3. **Empty expert load**
   Some rank may have no token hitting local experts in某步, still must correctly participate in all_reduce.
4. **dtype unification**
   Router logits, topk weights, expert output dtype strategy must be consistent, avoid rank间 numerical deviation amplification.

---

## 3. Phase 2: `engine/models/deepseek_v2.py` MLA Structure Definition

## 3.1 File and Export Goals
- New file: `engine/models/deepseek_v2.py`
- First define (structure only):
  - `DeepseekV2TPConfig`
  - `DeepseekAttentionTP`
  - `DeepseekDecoderLayerTP` (skeleton only)
  - `DeepseekForCausalLMTP` (skeleton only, `load_weights` left empty for Phase 3)

## 3.2 Config Fields (Minimum Set)

```python
@dataclass
class DeepseekV2TPConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rms_norm_eps: float
    num_hidden_layers: int
    # MoE related (for Decoder layer assembly)
    n_routed_experts: int
    num_experts_per_tok: int
```

Notes:
- These fields are minimum dimensions for MLA/MoE, should not reuse Qwen Dense config assumptions.
- For V3 compatibility,保留 extension field interface (e.g., grouped_topk parameter).

## 3.3 `DeepseekAttentionTP`: Sharding Rules (Hard Constraints)

### Absolutely Cannot Shard (Replicated)
- `q_a_proj`
- `kv_a_proj_with_mqa`

Reason: They are `hidden_size -> lora_rank` dimensionality reduction bottlenecks, sharding by head dimension destroys latent vector semantics and subsequent cache structure.

### Must Shard by TP
- `q_b_proj`: `ColumnParallelLinear` (output head dimension sharding)
- `kv_b_proj_with_mqa`: `ColumnParallelLinear` (output head dimension sharding)
- `o_proj`: `RowParallelLinear` (input sharded by rank, output all_reduce)

## 3.4 MLA Forward Data Flow (Structure Definition)

For input `x[B, T, H]`:

1. **Q path (two segments)**
   - `q_latent = q_a_proj(x)` (replicated)
   - `q_full_local = q_b_proj(q_latent)` (column parallel, get this rank's local heads)
   - Split into `q_nope` and `q_pe`.

2. **KV path (compressed representation)**
   - `kv_latent_plus_pe = kv_a_proj_with_mqa(x)` (replicated)
   - Split:
     - `c_kv` (latent, dimension `kv_lora_rank`)
     - `k_pe` (rope token part, dimension `qk_rope_head_dim` or derived)
   - `kv_full_local = kv_b_proj_with_mqa(c_kv)` (column parallel)
   - From it take `k_nope` and `v_local`.

3. **Decoupled RoPE (only on pe segment)**
   - Rotate `q_pe` and `k_pe`.
   - RoPE uses Neox (first-half/second-half) rotation style.
   - Final concatenation:
     - `q_local = concat(q_nope, q_pe_rot)`
     - `k_local = concat(k_nope, k_pe_rot)`

4. **Attention + output projection**
   - This rank executes local heads attention to get `attn_out_local`.
   - `o_proj(attn_out_local)` aggregated via `RowParallelLinear`.

## 3.5 KV Cache Definition (Structure Only)

MLA cache only retains compressed state:
- `c_kv` (latent)
- `k_pe` (small rope component)

Does not cache:
- Expanded full `K` / `V` heads.

Suggested预留 data structure:
```python
class DeepseekMLAKVCache:
    # per layer per token:
    # latent: [kv_lora_rank]
    # k_pe:   [qk_rope_head_dim]
    ...
```

## 3.6 Assembly Relationship with Phase 1 `ExpertParallelMoE`

- `DeepseekDecoderLayerTP`'s MLP position改为:
  - Dense layer: retain conventional branch interface (for fallback)
  - MoE layer: connect `ExpertParallelMoE` (Phase 1 artifact)
- After Phase 2 complete, model structure can express:
  - MLA Attention (TP)
  - MoE (simplified EP via all_reduce)

---

## 4. Forward Compatibility for DeepSeekV3 (Only Interface预留)

Though this round only does V2, need预留 V3 extension points:

1. **MoE routing strategy pluggable**
   - `route_impl: Literal["topk", "grouped_topk"]`
2. **Shared expert interface**
   - `num_shared_experts` field and independent branch
3. **Attention mode enum**
   - `attn_impl: Literal["mla", "mla_absorb"]`
4. **Optional MTP/NSA slot**
   - Decoder output留 hook, not implementing logic in this phase

---

## 5. Implementation Order (Phase 1+2 Only)

1. New `engine/tp_layers/moe.py`: first let `ExpertParallelMoE` run through forward and all_reduce with random input.
2. New `engine/models/deepseek_v2.py`: first define `DeepseekAttentionTP` and related config/class skeleton.
3. Wire `DeepseekDecoderLayerTP` to connect Attention + MoE module graph (not connected to real weights).
4. Only do structure-level smoke test (shape/device/communication reachable), not real model quality alignment.

---

## 6. This Document Deliverable Checklist

- [x] Only covers Phase 1 and Phase 2.
- [x] Clear `moe.py` `ExpertParallelMoE` communication strategy (local expert + all_reduce).
- [x] Clear MLA "cannot shard / must shard" matrix.
- [x] Clear Decoupled RoPE and Neox style requirements.
- [x] Clear KV cache only stores latent + rope component.
- [x] Inherited Qwen TP error prevention rules, avoid repeated mistakes.
