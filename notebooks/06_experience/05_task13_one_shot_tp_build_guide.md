# One-Shot TP Inference Framework Build Complete Guide

## Date
2026-05-07

## Goal

This document is the ultimate reference for **one-shot, zero-error** TP inference framework construction. Combined with `inference_blueprint.json` and `SKILL.md`, Agent should be able to directly generate correct implementation from this document without multi-round debugging.

**Reference implementation**: `/home/honglin/meta-infer/engine/` (verified TP=2/4 all correct)

**Acceptance targets**:
- Qwen3-8B TP=2: Chinese prompt outputs readable Chinese
- DeepSeek-V2-Lite TP=2: Chinese prompt outputs readable Chinese

---

## I. Architecture Overview

### 1.1 File Structure

```
engine/
├── tp_layers/
│   ├── __init__.py          # Export all TP components
│   ├── distributed.py       # Process group management + all_reduce/all_gather
│   ├── linear.py            # ColumnParallelLinear + RowParallelLinear + RMSNorm
│   ├── embedding.py         # VocabParallelEmbedding + ParallelLMHead
│   ├── moe.py               # ExpertParallelMoE + _ExpertMLP
│   └── attention.py         # PagedAttention (optional, for continuous batch scheduling)
├── models/
│   ├── __init__.py          # Export QwenTPModelRunner, DeepseekTPModelRunner
│   ├── qwen.py              # Qwen3 Dense TP complete implementation
│   └── deepseek_v2.py       # DeepSeek-V2 MLA+MoE TP/EP complete implementation
├── sampler.py               # Sampling logic
├── structs.py               # Sequence data structures
└── scheduler.py             # Scheduler (optional)
```

### 1.2 Execution Order (Phase Dependency Chain)

```
Phase 1: distributed.py → linear.py → embedding.py → moe.py  (Framework foundation)
Phase 2: qwen.py  (Qwen3 TP adaptation)
Phase 3: deepseek_v2.py  (DeepSeek-V2 TP/EP adaptation)
Phase 4: End-to-end acceptance
```

**Iron rule**: Each component in Phase 1 must pass unit tests before entering Phase 2.

---

## II. Component Implementation Contracts (Per File)

### 2.1 distributed.py — TP Communication Primitives

**5 functions that must be implemented**:

```python
def get_tp_rank() -> int:
    """Get from dist.get_rank() first, fallback to env var RANK (default 0)."""

def get_tp_size() -> int:
    """Get from dist.get_world_size() first, fallback to env var WORLD_SIZE (default 1)."""

def is_tp_enabled() -> bool:
    return get_tp_size() > 1

def init_tp_distributed() -> str:
    """Initialize process group. Uses init_method="env://", auto-selects nccl/gloo.
    Must call torch.cuda.set_device(LOCAL_RANK). Returns backend name."""

def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """For bf16/fp16 tensors: convert to fp32 first, all_reduce, then convert back.
    For other dtypes: direct in-place all_reduce."""

def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """all_gather along last dimension and concatenate."""
```

**Key implementation details**:

```python
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not is_tp_enabled():
        return x
    if x.dtype in (torch.float16, torch.bfloat16):
        tmp = x.float()                          # ← Must! bf16 accumulation overflows
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
        return tmp.to(dtype=x.dtype)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x
```

**Common errors**:
- ❌ Using custom `_TP_GROUP` global variable → can be None, causing AttributeError
- ❌ bf16 direct all_reduce → numerical overflow/precision loss
- ❌ Not calling `torch.cuda.set_device(LOCAL_RANK)` → wrong device allocation with multiple GPUs

**Correct approach**: Directly use `dist.init_process_group(backend, init_method="env://")`, don't create custom process groups.

---

### 2.2 linear.py — Parallel Linear Layers + RMSNorm

#### 2.2.1 ColumnParallelLinear

**Sharding dimension**: weight `[output_size, input_size]` dim=0 (output dimension)

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=False, gather_output=False):
        self.tp_size = get_tp_size()
        ensure_divisible(output_size, self.tp_size)
        self.local_output_size = output_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.local_output_size, input_size))

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        return all_gather_last_dim(y) if self.gather_output else y

    def load_weight_shard(self, full_weight):
        # Anti-double-slice: if input already matches local shape, directly copy
        if full_weight.shape == self.weight.shape:
            self.weight.data.copy_(full_weight.to(...))
            return
        rank = get_tp_rank()
        start = rank * self.local_output_size
        self.weight.data.copy_(full_weight[start:end, :].to(...))
```

**Use cases**: q_proj, k_proj, v_proj, gate_proj, up_proj (dim=0 sharding, `gather_output=False`)

#### 2.2.2 RowParallelLinear

**Sharding dimension**: weight `[output_size, input_size]` dim=1 (input dimension)

```python
class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=False):
        self.local_input_size = input_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(output_size, self.local_input_size))

    def forward(self, x):
        y = F.linear(x, self.weight, None)
        y = all_reduce_sum(y)                    # ← Partial sum → global sum
        if self.bias is not None:
            y = y + self.bias                     # bias added after all_reduce (bias is fully replicated)
        return y

    def load_weight_shard(self, full_weight):
        if full_weight.shape == self.weight.shape:
            self.weight.data.copy_(full_weight.to(...))
            return
        rank = get_tp_rank()
        start = rank * self.local_input_size
        self.weight.data.copy_(full_weight[:, start:end].to(...))
```

**Use cases**: o_proj, down_proj (dim=1 sharding)

#### 2.2.3 RMSNorm — **Precision Critical**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()                                            # ← Convert to fp32
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)                      # ← fp32 normalization
        return (x * self.weight.float()).to(input_dtype)          # ← weight also to fp32 then multiply!
```

**Fatal errors**:
```python
# ❌ Wrong: convert to bf16 first then multiply weight
return (x.to(input_dtype)).mul_(self.weight)

# ❌ Wrong: weight multiplication in bf16 precision
return (x * self.weight).to(input_dtype)    # self.weight is bf16 Parameter
```

**Why this is fatal**:
1. RowParallelLinear's all_reduce produces ~0.008 element-level error in bf16
2. bf16 multiplication `a_bf16 * w_bf16` is sensitive to small input perturbations, amplifying error ~3x (0.008 → 0.023)
3. 0.023 difference is sufficient to change MoE gate's top-k expert selection
4. Error cascades through 26 MoE layers, final output completely deviates

**Correct approach**: `self.weight.float()` ensures multiplication in fp32 precision.

---

### 2.3 embedding.py — Vocab Parallel Embedding

#### 2.3.1 VocabParallelEmbedding

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.local_vocab_size = num_embeddings // self.tp_size
        self.vocab_start = get_tp_rank() * self.local_vocab_size
        self.vocab_end = self.vocab_start + self.local_vocab_size
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim))

    def forward(self, input_ids):
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill((~mask).unsqueeze(-1), 0)   # Zero out non-local rank tokens
        return all_reduce_sum(out)                          # Sum contributions from all ranks

    def load_weight_shard(self, full_weight):
        # Anti-double-slice: if already local vocab size, directly copy
        if int(full_weight.shape[0]) == self.local_vocab_size:
            shard = full_weight
        else:
            shard = full_weight[self.vocab_start : self.vocab_end, :]
        self.weight.data.copy_(shard.to(...))
```

**Common errors**:
- ❌ Not checking `full_weight.shape[0] == self.local_vocab_size`, causing rank>0 to re-slice already sliced weights (shape=0 error)
- ❌ Not zeroing out non-local rank tokens, causing embedding values to double after all_reduce

#### 2.3.2 ParallelLMHead

```python
class ParallelLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, gather_output=True):
        # Note parameter order: hidden_size, vocab_size (not vocab_size, hidden_size)
        self.local_vocab_size = vocab_size // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, hidden_size))

    def forward(self, hidden_states):
        local_logits = F.linear(hidden_states, self.weight)
        if not self.gather_output:
            return local_logits
        logits = all_gather_last_dim(local_logits)
        return logits[..., :self.vocab_size]   # Truncate to original vocab_size (handle remainder)
```

---

### 2.4 moe.py — Expert Parallelism

```python
class ExpertParallelMoE(nn.Module):
    """Simplified EP: router fully replicated; each rank only computes local experts; all_reduce sum."""

    def __init__(self, cfg: ExpertParallelMoEConfig):
        self.local_expert_ids = partition_experts_for_rank(cfg.num_experts, self.tp_size, self.tp_rank)
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)  # replicated
        self.experts = nn.ModuleDict(
            {str(i): _ExpertMLP(cfg.hidden_size, cfg.intermediate_size) for i in self.local_expert_ids}
        )

    def _router_topk(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, hidden).float(), self.gate.weight.float(), None)
        #                            ↑ fp32 computation ↑
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.cfg.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.cfg.routed_scaling_factor
        return topk_idx, topk_weight.to(hidden_states.dtype)

    def forward(self, hidden_states):  # hidden_states: [B, T, H]
        bsz, seqlen, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        topk_idx, topk_weight = self._router_topk(hidden_states)
        local_out = torch.zeros_like(flat)
        # Per-token, per-top-k loop (prioritize correctness, not throughput)
        for token_i in range(flat.shape[0]):
            x = flat[token_i : token_i + 1]
            for k in range(self.cfg.top_k):
                expert_id = int(topk_idx_flat[token_i, k].item())
                key = str(expert_id)
                if key not in self.experts:
                    continue  # Non-local expert, contribution is zero
                local_out[token_i] += self.experts[key](x) * topk_weight_flat[token_i, k]
        global_out = all_reduce_sum(local_out)  # Sum contributions from all ranks
        return global_out.view(bsz, seqlen, hidden)
```

**Key design**:
- Router (gate) fully replicated on all ranks, ensuring routing decisions一致
- Each rank only instantiates experts it负责
- `all_reduce_sum` aggregates expert outputs from all ranks
- Input must be 3D `[B, T, H]`, not 2D `[N, H]`

**Common errors**:
- ❌ Receiving 2D input `[N, H]` → `hidden_states.shape` unpacking fails
- ❌ Router not computed in fp32 → insufficient softmax precision causing routing抖动
- ❌ Not multiplying `routed_scaling_factor` → FFN signal衰减

---

### 2.5 qwen.py — Qwen3 Dense TP

#### 2.5.1 RoPE Implementation (**Precision Critical**)

```python
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen3 uses first-half/second-half split, not even/odd interleaving!"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(q, k, positions, theta):
    dim = q.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = positions.to(torch.float32)
    freqs = torch.outer(t, inv_freq)
    with torch.autocast(device_type='cuda', enabled=False):
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    cos, sin = cos.to(input_dtype), sin.to(input_dtype)  # ← Convert back to bf16!
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k
```

**Two RoPE styles (cannot be mixed)**:

| Model | `_rotate_half` Style | cos/sin dtype |
|-------|---------------------|---------------|
| Qwen3 | First-half/second-half split | bf16 |
| DeepSeek-V2 | Even/odd interleaving (GPT-J style) | bf16 |

**Fatal errors**:
- ❌ Qwen3 using even/odd interleaving `_rotate_half` → output garbled/repeated
- ❌ cos/sin staying fp32 without converting to bf16 → small angle cos(θ) difference in fp32≈1.0 vs bf16≈1.0 cascades through 36 layers
- ❌ DeepSeek-V2 using first-half/second-half split → RoPE position encoding completely wrong

#### 2.5.2 GQA KV Head Repeat (**Layout Critical**)

```python
# In attention forward:
if self.num_kv_heads != self.num_heads:
    repeat = self.num_heads // self.num_kv_heads
    k = k.repeat_interleave(repeat, dim=2)   # ← Correct: [kv0, kv0, kv1, kv1]
    v = v.repeat_interleave(repeat, dim=2)
```

**Fatal error**:
```python
# ❌ Wrong: expand+reshape produces interleaved layout
k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(bs, seq, -1, head_dim)
# Result: [kv0, kv1, kv0, kv1] — Q head 0 attends to kv0, Q head 1 attends to kv1, Q head 2 attends to kv0...
# Correct should be: Q head 0,1 both attend to kv0, Q head 2,3 both attend to kv1
```

**Why expand+reshape is wrong**:
- `expand` copies entire tensor along new dimension
- `reshape` merges dimensions with interleaved排列
- Result: `[kv0, kv1, kv0, kv1]` (interleaved) not `[kv0, kv0, kv1, kv1]` (grouped)
- DeepSeek-V2 unaffected (num_heads == num_kv_heads, no GQA)

#### 2.5.3 KV Head Replication (when num_kv_heads < tp_size)

```python
# When total_num_kv_heads < tp_size (e.g., 4 KV heads, 8 GPUs):
if self.total_num_kv_heads >= self.tp:
    self.num_kv_heads = self.total_num_kv_heads // self.tp
    self.kv_head_replica = 1
else:
    self.num_kv_heads = 1  # Each rank only has 1 KV head
    self.kv_head_replica = self.tp // self.total_num_kv_heads
    self.kv_group_rank = self.rank // self.kv_head_replica
```

Weight loading:
```python
# allow_kv_replication branch in _load_tensor:
if allow_kv_replication and split_size < tp and tp % split_size == 0:
    replicas = tp // split_size
    shard_rank = rank // replicas
    start = shard_rank
    end = start + 1
```

#### 2.5.4 Complete Attention Flow

```python
class QwenAttentionTP(nn.Module):
    def forward(self, hidden_states, positions):
        bsz, seqlen, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)  # per-head RMSNorm
        k = self.k_norm(k)
        q, k = _apply_rope(q, k, positions, self.rope_theta)

        # GQA → MHA
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        # [B, T, H, D] → [B, H, T, D] for SDPA
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, self.q_size)
        return self.o_proj(out)  # RowParallel → all_reduce
```

#### 2.5.5 MLP Flow

```python
class QwenMLPTP(nn.Module):
    def forward(self, x):
        gate = self.gate_proj(x)   # ColumnParallel
        up = self.up_proj(x)       # ColumnParallel
        h = F.silu(gate) * up
        return self.down_proj(h)   # RowParallel → all_reduce
```

#### 2.5.6 Weight Loading Mapping Table

| HF Weight Name | TP Component | split_dim | Notes |
|----------------|--------------|-----------|-------|
| `model.embed_tokens.weight` | VocabParallelEmbedding | 0 | load_weight_shard |
| `model.layers.{i}.input_layernorm.weight` | RMSNorm | None | Full replication |
| `model.layers.{i}.post_attention_layernorm.weight` | RMSNorm | None | Full replication |
| `model.layers.{i}.self_attn.q_norm.weight` | RMSNorm | None | Full replication |
| `model.layers.{i}.self_attn.k_norm.weight` | RMSNorm | None | Full replication |
| `model.layers.{i}.self_attn.q_proj.weight` | ColumnParallelLinear | 0 | |
| `model.layers.{i}.self_attn.k_proj.weight` | ColumnParallelLinear | 0 | allow_kv_replication=True |
| `model.layers.{i}.self_attn.v_proj.weight` | ColumnParallelLinear | 0 | allow_kv_replication=True |
| `model.layers.{i}.self_attn.o_proj.weight` | RowParallelLinear | 1 | |
| `model.layers.{i}.mlp.gate_proj.weight` | ColumnParallelLinear | 0 | |
| `model.layers.{i}.mlp.up_proj.weight` | ColumnParallelLinear | 0 | |
| `model.layers.{i}.mlp.down_proj.weight` | RowParallelLinear | 1 | |
| `model.norm.weight` | RMSNorm | None | Full replication |
| `lm_head.weight` | ParallelLMHead | 0 | load_weight_shard |

---

### 2.6 deepseek_v2.py — DeepSeek-V2 MLA+MoE TP/EP

#### 2.6.1 MLA Attention Architecture

DeepSeek-V2 uses Multi-head Latent Attention (MLA), KV compressed to low-rank space:

```
Input hidden_states
  ├─ q_a_proj (replicated) → q_a_layernorm → q_b_proj (TP shard dim=0) → q
  │                         ↓
  └─ kv_a_proj_with_mqa (replicated) → split → c_kv (kv_lora_rank) + k_pe (qk_rope_head_dim)
                                         ↓              ↓
                                    kv_a_layernorm    RoPE (GPT-J style)
                                         ↓
                                    kv_b_proj_with_mqa (TP shard dim=0) → split → k_nope + v
                                         ↓
                              cat([k_nope, k_pe]) → k
```

**Key constraints**:
- `q_a_proj` and `kv_a_proj_with_mqa` are **fully replicated**, **must NOT be sliced**
- `q_b_proj` and `kv_b_proj_with_mqa` are **TP sharded** (ColumnParallel, dim=0)
- `o_proj` is **TP sharded** (RowParallel, dim=1)

#### 2.6.2 MLA Weight Loading Mapping Table

| HF Weight Name | TP Component | split_dim | Notes |
|----------------|--------------|-----------|-------|
| `model.layers.{i}.self_attn.q_a_proj.weight` | nn.Linear (replicated) | None | **Must NOT slice** |
| `model.layers.{i}.self_attn.q_a_layernorm.weight` | RMSNorm | None | Full replication |
| `model.layers.{i}.self_attn.q_b_proj.weight` | ColumnParallelLinear | 0 | |
| `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight` | nn.Linear (replicated) | None | **Must NOT slice** |
| `model.layers.{i}.self_attn.kv_a_layernorm.weight` | RMSNorm | None | Full replication |
| `model.layers.{i}.self_attn.kv_b_proj.weight` | ColumnParallelLinear | 0 | |
| `model.layers.{i}.self_attn.o_proj.weight` | RowParallelLinear | 1 | |

#### 2.6.3 DeepSeek-V2 RoPE (YaRN + GPT-J style)

```python
def _rotate_half_gptj(x: torch.Tensor) -> torch.Tensor:
    """GPT-J style even/odd interleaving, not Qwen3's first-half/second-half!"""
    x1 = x[..., ::2]    # Even dimensions
    x2 = x[..., 1::2]   # Odd dimensions
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope_gptj(x, positions, theta, rope_scaling):
    dim = x.shape[-1]
    inv_freq = _compute_inv_freq(dim, theta, rope_scaling, x.device)  # Contains YaRN interpolation
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    if rope_scaling and rope_scaling.get("type") == "yarn":
        # YaRN mscale compensation
        factor = float(rope_scaling.get("factor", 1.0))
        mscale = float(rope_scaling.get("mscale", 1.0))
        mscale_all_dim = float(rope_scaling.get("mscale_all_dim", 1.0))
        rope_mscale = _yarn_get_mscale(factor, mscale) / _yarn_get_mscale(factor, mscale_all_dim)
        cos = cos * rope_mscale
        sin = sin * rope_mscale
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2).to(dtype=x.dtype)
    return x * cos + _rotate_half_gptj(x) * sin
```

**YaRN frequency interpolation** (`_compute_inv_freq`):
```python
def _compute_inv_freq(dim, theta, rope_scaling, device):
    base_freq = theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    inv_freq = 1.0 / base_freq
    if not rope_scaling or rope_scaling.get("type") != "yarn":
        return inv_freq
    # YaRN: linear blending between interpolation and extrapolation frequencies
    factor = float(rope_scaling.get("factor", 1.0))
    inv_interp = 1.0 / (factor * base_freq)
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, theta, original_max_pos)
    inv_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, device=device)
    return inv_interp * (1.0 - inv_mask) + inv_freq * inv_mask
```

**Fatal errors**:
- ❌ Missing YaRN frequency interpolation → long sequence position encoding error, output semantic drift
- ❌ Using Neox style rotate_half (first-half/second-half) → RoPE completely wrong
- ❌ cos/sin staying fp32 without converting to bf16 → precision mismatch

#### 2.6.4 MoE Layer (EP + Shared Expert TP)

```python
class DeepseekMoETP(nn.Module):
    def __init__(self, cfg):
        self.routed = ExpertParallelMoE(ExpertParallelMoEConfig(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            num_experts=cfg.n_routed_experts,
            top_k=cfg.num_experts_per_tok,
            routed_scaling_factor=cfg.routed_scaling_factor,
        ))
        # Shared experts use TP (not EP)
        self.shared_experts = DeepseekMLPTP(
            intermediate_size=cfg.moe_intermediate_size * cfg.n_shared_experts
        )

    def forward(self, x):
        out = self.routed(x)       # EP: all_reduce aggregation
        out = out + self.shared_experts(x)  # TP: Column/Row parallel
        return out
```

**MoE weight loading**:
```python
# Router gate: replicated (full copy)
mlp.routed.gate.weight.data.copy_(full_weight.to(...))

# Routed experts: only load local rank's expert (EP)
for eid_str, expert in mlp.routed.experts.items():
    e = int(eid_str)
    expert.gate_proj.weight.data.copy_(self._load_tensor(f"{pfx}.mlp.experts.{e}.gate_proj.weight"))
    expert.up_proj.weight.data.copy_(self._load_tensor(f"{pfx}.mlp.experts.{e}.up_proj.weight"))
    expert.down_proj.weight.data.copy_(self._load_tensor(f"{pfx}.mlp.experts.{e}.down_proj.weight"))

# Shared experts: TP sharding
mlp.shared_experts.gate_proj.load_weight_shard(self._load_tensor(..., split_dim=0))
mlp.shared_experts.up_proj.load_weight_shard(self._load_tensor(..., split_dim=0))
mlp.shared_experts.down_proj.load_weight_shard(self._load_tensor(..., split_dim=1))
```

#### 2.6.5 DeepSeek-V2 Complete Attention Flow

```python
class DeepseekAttentionTP(nn.Module):
    def forward(self, hidden_states, positions):
        bsz, seqlen, _ = hidden_states.shape

        # Q path
        if self.cfg.q_lora_rank is None:
            q_full = self.q_b_proj(hidden_states)
        else:
            q_latent = self.q_a_proj(hidden_states)      # replicated
            q_latent = self.q_a_layernorm(q_latent)
            q_full = self.q_b_proj(q_latent)              # TP shard
        q = q_full.view(bsz, seqlen, self.local_heads, self.cfg.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim], dim=-1)

        # KV path
        kv_latent_plus_pe = self.kv_a_proj_with_mqa(hidden_states)  # replicated
        c_kv, k_pe = torch.split(kv_latent_plus_pe, [self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)
        kv_full = self.kv_b_proj_with_mqa(c_kv)                     # TP shard
        kv_full = kv_full.view(bsz, seqlen, self.local_heads, self.cfg.qk_nope_head_dim + self.cfg.v_head_dim)
        k_nope, v = torch.split(kv_full, [self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], dim=-1)

        # RoPE
        q_pe = _apply_rope_gptj(q_pe, positions, self.cfg.rope_theta, self.cfg.rope_scaling)
        k_pe = k_pe.view(bsz, seqlen, 1, self.cfg.qk_rope_head_dim)
        k_pe = _apply_rope_gptj(k_pe, positions, self.cfg.rope_theta, self.cfg.rope_scaling)
        k_pe = k_pe.expand(-1, -1, self.local_heads, -1)  # MQA → MHA broadcast

        # Concatenate nope + rope
        q_cat = torch.cat([q_nope, q_pe], dim=-1).permute(0, 2, 1, 3)
        k_cat = torch.cat([k_nope, k_pe], dim=-1).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q_cat, k_cat, v, is_causal=True, scale=self.scaling)
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, self.local_heads * self.cfg.v_head_dim)
        return self.o_proj(out)  # RowParallel → all_reduce
```

#### 2.6.6 MoE 2D/3D Compatibility

Decoder layer passes hidden_states as 2D `[N, H]`, but ExpertParallelMoE expects 3D `[B, T, H]`.

```python
class DeepseekV2MoE(nn.Module):
    def forward(self, hidden_states):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # [N, H] → [1, N, H]
            routed_out = self.routed(hidden_states).squeeze(0)
            shared_out = self.shared_experts(hidden_states.squeeze(0))
            return routed_out + shared_out
        # 3D path...
```

---

## III. Model Building and Weight Loading

### 3.1 CPU-First Building (Anti-OOM)

```python
class QwenTPModelRunner:
    def __init__(self, model_dir, tp_size, tp_rank, ...):
        init_tp_distributed()
        torch.set_default_dtype(torch.bfloat16)
        self.model = QwenForCausalLMTP(cfg)    # ← Create on CPU
        torch.set_default_dtype(torch.float32)  # ← Restore default dtype
        self.model.load_weights()               # ← Load weights on CPU
        self.model.cuda()                       # ← Move to GPU
```

**Fatal errors**:
- ❌ Using `torch.set_default_device(f"cuda:{tp_rank}")` → creates 8B+ params on GPU → OOM
- ❌ `.cuda()` first then `load_weights()` → GPU memory full, cannot load

### 3.2 Safetensors Lazy Loading

```python
def _load_tensor(self, key, *, split_dim=None, allow_kv_replication=False):
    weight_map = self._resolve_weight_map()
    fname = weight_map.get(key)
    fp = self.cfg.model_dir / fname

    with safe_open(str(fp), framework="pt", device="cpu") as f:
        if split_dim is None:
            return f.get_tensor(key)          # Full read
        sl = f.get_slice(key)
        shape = list(sl.get_shape())
        split_size = int(shape[split_dim])
        tp = get_tp_size()
        rank = get_tp_rank()
        if split_size % tp == 0:
            part = split_size // tp
            start = rank * part
            end = start + part
        elif allow_kv_replication and split_size < tp and tp % split_size == 0:
            replicas = tp // split_size
            shard_rank = rank // replicas
            start = shard_rank
            end = start + 1
        else:
            raise ValueError(...)
        index = [slice(None)] * len(shape)
        index[split_dim] = slice(start, end)
        return sl[tuple(index)]               # Lazy slice, no full load
```

**Key advantage**: Using `safe_open` + `get_slice` instead of `load_file`, avoids loading full weights to CPU memory.

---

## IV. Complete Error Checklist (Aggregated from Tasks 10-13)

### 4.1 Fatal Errors (Cause completely wrong output)

| # | Error | Symptom | Root Cause | Fix |
|---|-------|---------|------------|-----|
| 1 | RMSNorm bf16 multiplication | DeepSeek output garbled | `x.to(dtype).mul_(weight)` amplifies error 3x in bf16 | `(x * self.weight.float()).to(dtype)` |
| 2 | GQA expand+reshape | Qwen3 output garbled | Produces interleaved KV head layout | Use `repeat_interleave` |
| 3 | RoPE rotate_half style error | Output repeated/garbled | Qwen3 uses first-half/second-half, DeepSeek uses even/odd interleaving | Strictly choose by model |
| 4 | Missing YaRN frequency interpolation | DeepSeek long sequence output drift | `rope_scaling.type=yarn` not handled | Implement complete `_compute_inv_freq` |
| 5 | MLA low-rank projection sliced | DeepSeek output wrong | `kv_a_proj_with_mqa` is replicated, not TP | Don't pass `split_dim` when loading |
| 6 | Embedding double slicing | rank>0 output all zeros | `load_weight_shard` re-slices already sliced weights | Check `shape[0] == local_vocab_size` |
| 7 | MoE receives 2D input | RuntimeError | `ExpertParallelMoE` expects `[B, T, H]` | Add `unsqueeze(0)` / `squeeze(0)` |

### 4.2 Precision Errors (Output "roughly correct" but first token mismatch)

| # | Error | Symptom | Root Cause | Fix |
|---|-------|---------|------------|-----|
| 8 | cos/sin dtype mismatch | Qwen3 first token deviation | cos/sin computed in fp32 but not converted to bf16 | `.to(input_dtype)` |
| 9 | bf16 all_reduce overflow | Numerical accumulation error | bf16 direct SUM overflows | Convert to fp32 first then all_reduce |
| 10 | Model created on CUDA | Weight loading inconsistency | Uninitialized CUDA memory + CPU→GPU copy | CPU create → load → `.cuda()` |
| 11 | Router logits computed in bf16 | MoE routing抖动 | softmax sensitive to bf16 precision | `.float()` for logits and softmax |

### 4.3 Engineering Errors (Runtime crash)

| # | Error | Symptom | Root Cause | Fix |
|---|-------|---------|------------|-----|
| 12 | `_TP_GROUP` is None | AttributeError | Custom process group not initialized | Use `dist.init_process_group` directly |
| 13 | GPU OOM | CUDA out of memory | Other processes占用 GPU / model created on GPU | Check `nvidia-smi`, CPU-first building |
| 14 | Weight file path error | FileNotFoundError | safetensors index parsing error | Check `model.safetensors.index.json` |

---

## V. Coding Rules (Agent Must Follow)

### 5.1 Precision Rules

1. **RMSNorm**: Normalization and weight multiplication both in fp32
2. **RoPE**: cos/sin computed in fp32, **result converted back to input dtype (bf16)** then applied
3. **MoE Router**: logits and softmax computed in fp32
4. **all_reduce**: bf16/fp16 tensors converted to fp32 first then reduce
5. **Attention matmul**: `q @ k.T` computed in bf16 (consistent with HF), softmax in fp32

### 5.2 TP Sharding Rules

1. **ColumnParallelLinear**: Shard weight dim=0 (output dimension), return local shard when `gather_output=False`
2. **RowParallelLinear**: Shard weight dim=1 (input dimension), add bias after all_reduce in forward
3. **load_weight_shard**: Must check `full_weight.shape == self.weight.shape` to prevent double slicing
4. **MLA low-rank projections** (`q_a_proj`, `kv_a_proj_with_mqa`): **Must NOT slice**, fully replicate
5. **KV head replication**: Use `allow_kv_replication=True` when `num_kv_heads < tp_size`

### 5.3 RoPE Rules

1. **Qwen3**: `_rotate_half` = first-half/second-half split, cos/sin to bf16
2. **DeepSeek-V2**: `_rotate_half` = even/odd interleaving (GPT-J style), YaRN frequency interpolation, cos/sin to bf16
3. **Both must**: cos/sin computed in fp32 then `.to(input_dtype)`

### 5.4 MoE Rules

1. **Router (gate)**: Fully replicated, not sharded in TP/EP
2. **Routed experts**: EP (each rank only instantiates its own experts)
3. **Shared experts**: TP (ColumnParallel/RowParallel)
4. **Aggregation method**: EP uses `all_reduce_sum`, TP uses RowParallel's all_reduce
5. **Input dimensions**: ExpertParallelMoE expects 3D `[B, T, H]`

### 5.5 Model Building Rules

1. **CPU-first**: Create model on CPU → load weights → `.cuda()`
2. **Safetensors lazy loading**: Use `safe_open` + `get_slice`, not `load_file`
3. **dtype management**: `torch.set_default_dtype(torch.bfloat16)` for model creation, immediately restore `torch.float32`
4. **Weight mapping**: Maintain complete HF key → TP component mapping table

### 5.6 Verification Rules

1. **First token comparison**: TP output's first token must exactly match HF ground truth
2. **Logits statistics**: Print mean, std, top-5 tokens, compare with HF
3. **Per-layer error tracking**: If first token doesn't match, per-layer compare hidden_states max_diff/mean_diff
4. **GPU memory monitoring**: Print `torch.cuda.memory_allocated()` after loading

---

## VI. One-Shot Generation Checklist

Before generating code, confirm each item:

### Phase 1: Framework Components

- [ ] `distributed.py`: `get_tp_rank/get_tp_size` fallback from env, `all_reduce_sum` reduce in fp32
- [ ] `linear.py`: `RMSNorm.forward` uses `(x * self.weight.float()).to(dtype)`
- [ ] `linear.py`: `ColumnParallelLinear.load_weight_shard` has shape match check
- [ ] `linear.py`: `RowParallelLinear.load_weight_shard` has shape match check
- [ ] `embedding.py`: `VocabParallelEmbedding.load_weight_shard` checks `shape[0] == local_vocab_size`
- [ ] `embedding.py`: `ParallelLMHead` parameter order is `(hidden_size, vocab_size)`
- [ ] `moe.py`: `ExpertParallelMoE` expects 3D input, router computed in fp32

### Phase 2: Qwen3 TP

- [ ] `_rotate_half`: first-half/second-half split (not even/odd interleaving)
- [ ] `cos/sin`: computed in fp32 then `.to(input_dtype)` to bf16
- [ ] GQA: uses `repeat_interleave` (not `expand+reshape`)
- [ ] KV head replication: `allow_kv_replication` when `num_kv_heads < tp_size`
- [ ] Weight loading: `k_proj` and `v_proj` use `allow_kv_replication=True`
- [ ] Model building: CPU-first, safetensors lazy loading

### Phase 3: DeepSeek-V2 TP/EP

- [ ] `_rotate_half_gptj`: even/odd interleaving (not first-half/second-half)
- [ ] YaRN: complete implementation of `_compute_inv_freq` (with `_yarn_find_correction_range` and `_yarn_linear_ramp_mask`)
- [ ] MLA projections: `q_a_proj`/`kv_a_proj_with_mqa` fully replicated, `q_b_proj`/`kv_b_proj` TP sharded
- [ ] MoE: router replicated, routed experts EP, shared experts TP
- [ ] MoE input: 2D/3D compatible (`unsqueeze(0)` / `squeeze(0)`)
- [ ] `routed_scaling_factor`: multiplied after router topk

### Phase 4: Acceptance

- [ ] Qwen3-8B TP=2: first token = 220, output readable Chinese
- [ ] DeepSeek-V2-Lite TP=2: first token = 185, output readable Chinese
- [ ] GPU memory: Qwen3 ~8 GiB/card, DeepSeek-V2 ~16 GiB/card
- [ ] No NaN/Inf: logits statistics mean/std reasonable

---

## VII. Debug Flow (When Output is Incorrect)

### 7.1 First Token Mismatch

1. Print logits top-5 tokens and values
2. Compare with HF ground truth `max_diff` and `mean_diff`
3. If `max_diff > 0.1`: check RMSNorm weight multiplication is in fp32
4. If `max_diff > 0.01`: check RoPE cos/sin dtype, rotate_half style
5. If `max_diff < 0.01` but first token different: check GQA KV head layout

### 7.2 Output Garbled/Repeated

1. Check RoPE `_rotate_half` style matches model
2. Check GQA `repeat_interleave` vs `expand+reshape`
3. Check YaRN frequency interpolation implemented

### 7.3 OOM

1. `nvidia-smi` check GPU memory usage
2. Confirm model created on CPU
3. Confirm `safe_open` + `get_slice` lazy loading (not `load_file`)
4. DeepSeek-V2: check EP configuration (larger `ep_size`, fewer experts per card)

### 7.4 NCCL Errors

- Error code 2/3: usually async abort logs during process exit, not computation failure
- Ensure all ranks called `dist.init_process_group`
- Ensure `destroy_process_group()` called before exit

---

## VIII. Reference File Index

| File | Purpose |
|------|---------|
| `inference_blueprint.json` | Architecture knowledge graph (sole contract source) |
| `SKILL.md` | Execution flow and gates |
| `01_task10_tp_qwen_debug_experience.md` | Qwen TP=4 debugging experience |
| `02_task11_deepseek_tp_real_debug_experience.md` | DeepSeek TP debugging experience |
| `03_task12_deepseek_ep_memory_fix.md` | EP memory optimization |
| `03_task12_qwen_tp_debug_bf16_precision.md` | bf16 precision cascade issues |
| `04_task13_tp_correctness_fix_rmsnorm.md` | RMSNorm + GQA fix |
