# DeepSeekV2 Adaptation Knowledge in vLLM (TP Perspective)

This document organizes the core modifications in `vllm/model_executor/models/deepseek_v2.py` relative to direct HF loading into 4 categories, and provides:

- Knowledge point descriptions
- Reference source code paths
- Key code snippets directly related to TP (brief)

For subsequent implementation of DeepSeekV2's TP version (especially TP=4) in the minimal framework.

---

## 1) Parallel Layers Replaced with vLLM Parallel Layers

## 1.1 Knowledge Points

- Ordinary `nn.Linear` is replaced with layers with clear TP semantics:
  - `QKVParallelLinear`
  - `MergedColumnParallelLinear`
  - `RowParallelLinear`
  - `VocabParallelEmbedding` / `ParallelLMHead`
- These layers have built-in:
  - Parameter shape sharding by `tp_size`
  - Forward `all_reduce`/`all_gather` protocols
  - `weight_loader` sharded loading interface

## 1.2 Reference Source Code Paths

- Model usage side:
  - `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py`
- Layer definition side:
  - `vllm-v0.15.1-dev/vllm/model_executor/layers/linear.py`
  - `vllm-v0.15.1-dev/vllm/model_executor/layers/vocab_parallel_embedding.py`
- Parameter loading protocol:
  - `vllm-v0.15.1-dev/vllm/model_executor/parameter.py`

## 1.3 Key Code (TP)

```python
# deepseek_v2.py
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
self.gate_up_proj = MergedColumnParallelLinear(...)
self.down_proj = RowParallelLinear(...)
```

```python
# deepseek_v2.py
tp_size = get_tensor_model_parallel_world_size()
self.num_heads = self.total_num_heads // tp_size
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

```python
# layers/linear.py (RowParallelLinear semantics)
if self.reduce_results and self.tp_size > 1:
    output_ = tensor_model_parallel_all_reduce(output_parallel)
```

---

## 2) Attention/KV Cache Changed to vLLM Runtime Protocol

## 2.1 Knowledge Points

- Does not use HF's direct `past_key_values` usage, but interfaces with vLLM runtime attention and KV cache abstraction.
- Key points:
  - `Attention(...)` uniformly connects to vLLM attention backend
  - `KVCacheSpec` / `MLAAttentionSpec` declares cache layout
  - Special structure for MLA path (DeepSeekV2/3 specific)

## 2.2 Reference Source Code Paths

- `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py`
- Related interfaces:
  - `vllm-v0.15.1-dev/vllm/v1/kv_cache_interface.py`

## 2.3 Key Code (TP + KV)

```python
# deepseek_v2.py
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    return MLAAttentionSpec(...)
```

```python
# deepseek_v2.py
self.attn = Attention(
    self.num_heads,
    self.head_dim,
    self.scaling,
    num_kv_heads=self.num_kv_heads,
    cache_config=cache_config,
    ...
)
```

```python
# deepseek_v2.py
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
attn_output = self.attn(q, k, v)
output, _ = self.o_proj(attn_output)
```

---

## 3) MoE Path Changed to FusedMoE + EP/TP Group Coordination

## 3.1 Knowledge Points

- MoE is no longer pure Python per-expert forward concatenation, but uses `SharedFusedMoE` (fused kernel path).
- Simultaneously introduces EP (Expert Parallel) and TP group information:
  - `tp_rank/tp_size`
  - `ep_group/ep_rank/ep_size`
- This enables MoE to be scalable and high-performance in large model inference services.

## 3.2 Reference Source Code Paths

- `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py`
- Related MoE layers:
  - `vllm-v0.15.1-dev/vllm/model_executor/layers/fused_moe/`
- Distributed group interfaces:
  - `vllm-v0.15.1-dev/vllm/distributed/parallel_state.py`

## 3.3 Key Code (TP/EP + MoE)

```python
# deepseek_v2.py
self.tp_size = get_tensor_model_parallel_world_size()
self.tp_rank = get_tensor_model_parallel_rank()
self.ep_group = get_ep_group().device_group
self.ep_rank = get_ep_group().rank_in_group
self.ep_size = self.ep_group.size()
```

```python
# deepseek_v2.py
self.gate = GateLinear(...)
self.experts = SharedFusedMoE(...)
```

```python
# deepseek_v2.py
expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
    self,
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=self.config.n_routed_experts + ...,
    num_redundant_experts=self.num_redundant_experts,
)
```

---

## 4) Weight Loading is Not `from_pretrained`, But Mapping + Sharding + Expert-Specific Loader

## 4.1 Knowledge Points

- vLLM model layer parameters are not simply `copy_` by HF name one-to-one, but through:
  1. Name mapping (e.g., q/k/v fused to qkv)
  2. shard_id guided slicing
  3. Expert parameter mapping (MoE)
  4. Parameter object's `weight_loader(...)` executes final sharded loading
- This mechanism is key to TP working; HF's default loading path lacks such a unified protocol.

## 4.2 Reference Source Code Paths

- Model-specific loading logic:
  - `vllm-v0.15.1-dev/vllm/model_executor/models/deepseek_v2.py` (`load_weights`)
- Generic loaders:
  - `vllm-v0.15.1-dev/vllm/model_executor/model_loader/default_loader.py`
  - `vllm-v0.15.1-dev/vllm/model_executor/model_loader/sharded_state_loader.py`
  - `vllm-v0.15.1-dev/vllm/model_executor/model_loader/weight_utils.py`
- Parameter-side sharding behavior:
  - `vllm-v0.15.1-dev/vllm/model_executor/parameter.py`

## 4.3 Key Code (Mapping + Sharding)

```python
# deepseek_v2.py
stacked_params_mapping = [
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
]
```

```python
# deepseek_v2.py
for param_name, weight_name, shard_id in stacked_params_mapping:
    if weight_name not in name:
        continue
    name = name.replace(weight_name, param_name)
    param = params_dict[name]
    weight_loader = param.weight_loader
    weight_loader(param, loaded_weight, shard_id)
```

```python
# deepseek_v2.py (MoE expert weight specific loading)
weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
success = weight_loader(
    param, weight_to_load, name_mapped,
    shard_id=shard_id, expert_id=expert_id, return_success=True,
)
```

```python
# model_loader/weight_utils.py
def sharded_weight_loader(shard_axis: int):
    def loader(param, loaded_weight):
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        return default_weight_loader(param, loaded_weight)
```

---

## 5) Summary of Essential Differences from Direct HF Loading

- HF direct loading:
  - Goal is "model semantics correct + generally usable", default does not handle TP/EP serving details.
- vLLM DeepSeekV2 adaptation:
  - Goal is "parallelizable sharding + servable scheduling + high-throughput execution", so must have:
    - TP-aware layer definitions
    - KV/cache runtime protocols
    - FusedMoE and EP/TP group coordination
    - Unified `weight_loader` mapping/sharding/expert loading mechanism

---

## 6) Direct Suggestions for Subsequent "Minimal Framework TP=4 Integration"

Implementation priority order:
1. First align `layers/linear.py` TP semantics (QKV/Row/MergedColumn).
2. Then implement `parameter.py`-like `weight_loader` protocol.
3. Reference `deepseek_v2.py::load_weights` for name mapping and expert loading.
4. Finally consider kernel-side replacement optimization (Triton/Cutlass/FlashInfer).

This ensures TP correctness is established first, then performance is gradually optimized.
