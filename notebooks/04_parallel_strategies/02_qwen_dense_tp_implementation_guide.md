# Qwen Dense Model TP Sharding Implementation Guide (For Current Minimal Framework)

Goal of this document: Extract **Qwen Dense (non-MoE) TP implementation knowledge** from `nano-sglang` and `mini-sglang` that can be directly migrated to the current `meta-infer/engine`, and provide source code paths and key code snippets.
Scope: `tp_size = N`, `N ∈ {1,2,4,8}` (same code parameterized, no code forking by N).

---

## 1. Overall TP Design (Establish Protocol First, Then Modify Model)

### 1.1 Unified Conclusions
- TP is not about "wrapping the model in a layer", but simultaneously modifying 4 things:
  - Process group initialization (rank/size/group)
  - Sharding shape of linear layers and embedding/lm_head
  - Slicing weights by the same rules during loading
  - Communication in forward pass (`all_reduce` / `all_gather`)
- Qwen Dense can directly reuse the general TP pattern of LLaMA/Qwen2/Qwen3:
  `QKV parallel + O/Down row parallel + Embedding/LMHead vocab parallel`.

### 1.2 Key Reference Paths
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/parallel_state.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/engine/engine.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/info.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/impl.py`

### 1.3 Key Code (Initialization)
```python
# nano-sglang: model_runner.py
torch.distributed.init_process_group(
    backend="nccl",
    world_size=self.tp_size,
    rank=self.tp_rank,
    init_method=f"tcp://127.0.0.1:{self.nccl_port}",
)
initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
```

```python
# mini-sglang: engine.py
set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
torch.distributed.init_process_group(
    backend="nccl",  # or gloo + pynccl
    rank=config.tp_info.rank,
    world_size=config.tp_info.size,
    init_method=config.distributed_addr,
)
```

---

## 2. Qwen Dense Layer Sharding Rules (Most Critical)

Qwen2/Qwen3 in mini-sglang directly uses unified operators:
- `LinearQKVMerged`
- `LinearOProj`
- `LinearColParallelMerged` (gate/up)
- `LinearRowParallel` (down)
- `VocabParallelEmbedding`
- `ParallelLMHead`

Reference paths:
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen2.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen3.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/utils.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/linear.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/embedding.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/attention.py`

### 2.1 Attention: QKV and KV Head Replication/Sharding
```python
# minisgl/layers/linear.py
local_num_qo = div_even(num_qo_heads, tp_info.size)
local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
```

```python
# minisgl/layers/attention.py
self.num_qo_heads = div_even(num_qo_heads, tp_size)
self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
```

Knowledge points:
- `num_qo_heads` must be divisible by `tp_size`.
- If `num_kv_heads` is less than `tp_size`, replication is used (common in GQA scenarios); otherwise, heads are sharded.

### 2.2 MLP: gate/up Column Parallel, down Row Parallel
```python
# minisgl/models/utils.py
self.gate_up_proj = LinearColParallelMerged(
    config.hidden_size, [config.intermediate_size, config.intermediate_size], has_bias=False
)
self.down_proj = LinearRowParallel(config.intermediate_size, config.hidden_size, has_bias=False)
```

Knowledge points:
- Column parallel (output sharding) is commonly used for FFN expansion branches;
- Row parallel (input sharding) is commonly used for projecting back to hidden_size, with results requiring cross-rank summation.

### 2.3 O Projection and RowParallel Communication
```python
# minisgl/layers/linear.py
y = F.linear(x, self.weight, self.bias)
if self._tp_size > 1:
    y = self._comm.all_reduce(y)
```

Knowledge points:
- `o_proj` and `down_proj` typically follow "each card computes a partial result, then all_reduce to get the complete output".

---

## 3. Embedding / LM Head Vocab Parallelism

Reference paths:
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/embedding.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/logits_processor.py`

### 3.1 Embedding: Shard by vocab dimension + all_reduce merge
```python
# minisgl/layers/embedding.py
self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
start_idx = self.num_embeddings_tp * tp_rank
finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
self.vocab_range = (start_idx, finish_idx - start_idx)
...
return self._comm.all_reduce(y) if self.tp_size > 1 else y
```

### 3.2 LM Head: Each card computes local logits, then all_gather to concatenate
```python
# minisgl/layers/embedding.py
logits = F.linear(x, module.weight, self.bias)
output_tensor = self._comm.all_gather(logits)
...
return output_tensor[:, : self.num_embeddings]
```

```python
# nano-sglang/logits_processor.py
last_logits = torch.matmul(last_hidden, weight.T)
if self.tp_size > 1:
    last_logits = tensor_model_parallel_all_gather(last_logits)
last_logits = last_logits[:, : self.config.vocab_size]
```

Knowledge points:
- Under TP, before sampling, global vocab aggregation (`all_gather`) must be performed first, then `[:vocab_size]` to trim padding.

---

## 4. Weight Loading: Sharding Rules Must Be Exactly Consistent with Layer Parallel Rules

Reference paths (most critical):
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/weight.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/llama2.py`

### 4.1 mini-sglang: Unified sharding + merge
```python
# minisgl/models/weight.py
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]
...
if any(key.count(sub) for sub in _SPLIT_DIM_0):
    return value.chunk(n, dim=0)[r].clone()
elif any(key.count(sub) for sub in _SPLIT_DIM_1):
    return value.chunk(n, dim=1)[r].clone()
```

```python
# minisgl/models/weight.py: fused loading
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
```

### 4.2 nano-sglang: Parameter objects with `weight_loader`
```python
# nano-sglang/llama2.py
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
...
weight_loader = param.weight_loader
weight_loader(param, loaded_weight, shard_id)
```

Knowledge points:
- "How layers are parallelized" and "how weights are sliced" must correspond one-to-one, otherwise shape or semantics will be wrong.
- For Qwen Dense, it is recommended to directly reuse this pattern:
  - q/k/v -> merged into `qkv_proj`
  - gate/up -> merged into `gate_up_proj`
  - o/down -> input dimension sharding
  - embed/lm_head -> vocab sharding

---

## 5. KV Cache and TP Size Coupling (Required for Dense Too)

Reference paths:
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/engine/engine.py`

Key code:
```python
# nano-sglang/model_runner.py
head_num = self.model_config.num_key_value_heads // self.tp_size
...
self.token_to_kv_pool = TokenToKVPool(
    ...,
    head_num=self.model_config.num_key_value_heads // self.tp_size,
)
```

```python
# mini-sglang/engine.py
cache_per_page = (
    2 * head_dim
    * div_even(num_kv_heads, tp_size, allow_replicate=True)
    * page_size * dtype_size * num_layers
)
```

Knowledge points:
- After TP, per-card KV head count changes, directly affecting memory estimation and pool sizing;
- If `num_kv_heads < tp_size`, estimate per-card KV size using replication logic.

---

## 6. Minimum Actionable Modification Checklist for Current `meta-infer` (Qwen Dense TP)

Recommended modification order:

1) **Parallel context**
- Add `tp_size/tp_rank` parameters and process group initialization (`nccl`).
- Provide `all_reduce/all_gather` wrapper functions (single-card bypass).

2) **TP linear layers**
- Add to `meta-infer/engine`:
  - `LinearQKVMerged`
  - `LinearColParallelMerged`
  - `LinearRowParallel`
  - `LinearOProj`
- Forward communication rules aligned with mini/nano.

3) **Qwen Dense model definition**
- Create QwenDense model class (first align with Qwen3 structure):
  - Attention uses `qkv_proj + o_proj`
  - MLP uses `gate_up_proj + down_proj`
  - Embedding / LMHead uses vocab parallelism.

4) **Weight loader**
- Fusion mapping by `q/k/v` and `gate/up`;
- Rank slicing by dim0/dim1/vocab;
- Add replication slicing logic for `num_kv_heads < tp_size`.

5) **KV Pool and Logits**
- KV pool capacity estimation changed to depend on local kv-head;
- Logits undergo `all_gather` before sampling when TP>1.

---

## 7. High-Frequency Pitfall Checks During Implementation

- `num_attention_heads % tp_size != 0`: Error directly, no silent degradation.
- The relationship between `num_kv_heads` and `tp_size` must follow the "shard/replicate" binary logic.
- Sampling without `all_gather` on `lm_head`: Causes incomplete vocabulary and sampling errors.
- Inconsistent order of merge then shard / shard then merge: Wrong shape or wrong semantics.
- KV cache estimation still uses global `num_kv_heads`: Overestimates memory and causes scheduling errors.

---

## 8. Qwen Dense TP Reference Source Code Index

### nano-sglang
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/parallel_state.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/linear.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/logits_processor.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/llama2.py`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/mixtral.py` (mainly for TP communication and KV/GQA handling patterns)

### mini-sglang
- `meta-infer/ref_projects/mini-sglang/python/minisgl/engine/engine.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/info.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/impl.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/linear.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/embedding.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/attention.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen2.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen3.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/utils.py`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/weight.py`

---

If you want me to directly modify `meta-infer/engine` to add Qwen Dense TP next, I will follow the order in Section 6 of this guide, and prioritize running `tp=1 -> tp=2` consistency tests first before expanding to `tp=4/8`.
