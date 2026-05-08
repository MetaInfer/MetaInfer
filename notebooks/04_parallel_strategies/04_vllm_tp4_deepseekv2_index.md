# vLLM TP Sharding and Model Loading Knowledge Index (For DeepSeekV2 `TP=4`)

Goal: Based on the directories you specified:
- `meta-infer/ref_projects/vllm/vllm/model_executor/model_loader`
- `meta-infer/ref_projects/vllm/vllm/model_executor/layers`
- `meta-infer/ref_projects/vllm/vllm/model_executor/kernels`
and their directly imported TP-related dependencies, consolidate an index document that can directly guide subsequent implementation of DeepSeekV2 `TP=4` in the minimal framework.

---

## 1. Responsibility Boundaries of the Three Directory (Build Mental Model First)

- **`model_loader/`**: Solves "from checkpoint to per-rank parameter tensor" (download, traverse shards, slice by TP rank, `weight_loader` dispatch).
- **`layers/`**: Solves "how model structure is sharded and communicated by TP" (`ColumnParallel`, `RowParallel`, `QKVParallel`, `VocabParallel`, MoE reduction, etc.).
- **`kernels/`**: Solves "how operators execute efficiently on this rank" (Triton/Cutlass/FlashInfer linear kernels); typically **does not directly manage TP communication groups**.

---

## 2. `model_executor/model_loader`: Files That Must Be Mastered

## 2.1 General Loading and `weight_loader`

- `model_loader/base_loader.py`
  - Abstract base class and `load_weights` entry (approx. `37-64`).

- `model_loader/default_loader.py`
  - Default loading path, ultimately `model.load_weights(...)` (approx. `368-381`).
  - Rank flattening logic for EP/DP/PC/TP mixed scenarios (approx. `334-357`).

- `model_loader/weight_utils.py`
  - `default_weight_loader` (`1361+`)
  - `row_parallel_weight_loader` (`1382-1394`)
  - `sharded_weight_loader(shard_axis)` (`1400-1410`)
  This is the most direct template for TP shard slicing: `tp_rank * shard_size` + `narrow(...)`.

## 2.2 Pre-Sharded Checkpoints (Best for Large Model TP)

- `model_loader/sharded_state_loader.py`
  - Documentation: each worker only reads its own shard (`29-34`).
  - `load_weights` assembles file patterns by `get_tensor_model_parallel_rank()` (`110-123`).
  - Only supports pre-sharded checkpoints (`130-135`).

## 2.3 TP Sharding Logic in Quantized Loading

- `model_loader/bitsandbytes_loader.py`
  - Imports `get_tensor_model_parallel_rank/world_size` (file header).
  - Classifies by module type: row/column/unsharded (`492+`).
  - Actual shard index calculation (`338-417` section).

## 2.4 Other TP-Strongly-Related Files

- `model_loader/tensorizer_loader.py` and `tensorizer.py`
  - When `tensor_parallel_size > 1`, requires tensorizer URI/shard constraints (e.g., `tensorizer.py` approx. `312+`).

- `model_loader/ep_weight_filter.py`
  - Expert weight filtering for DP+EP (comments clearly state cooperation with `FusedMoE.weight_loader`).

---

## 3. `model_executor/layers`: TP Structure Definitions and Weight Loading Protocols

## 3.1 TP Linear Layer Main File (Core)

- `layers/linear.py`
  - `ColumnParallelLinear`: `406+`
  - `MergedColumnParallelLinear`: `603+`
  - `QKVParallelLinear`: `965+`
  - `RowParallelLinear`: `1371+`

Key points:
- Column parallel shards output dimension, row parallel shards input dimension and `all_reduce` when necessary.
- `QKVParallelLinear` has built-in KV head replication handling (when `num_kv_heads < tp_size`).
- These layers depend on parameter objects' `load_*` methods in the `weight_loader` flow.

## 3.2 Vocab Sharding

- `layers/vocab_parallel_embedding.py`
  - `VocabParallelEmbedding`: `192+`
  - `ParallelLMHead`: `503+`
  - Vocab range sharding, masking, `tensor_model_parallel_all_reduce/all_gather` protocols.

## 3.3 DeepSeekV2 (Target Model) Direct Reference

- `model_executor/models/deepseek_v2.py` (although not in your three listed directories, it is the convergence point of these layers/loaders)
  - Header imports show dependencies (`39-84`):
    `vllm.distributed`, `layers.linear`, `vocab_parallel_embedding`, `weight_utils`, `SharedFusedMoE`, etc.
  - Attention TP dimension logic and QKV/OProj (`130-191`).
  - MLP `MergedColumnParallelLinear` + `RowParallelLinear` (`211-238`).
  - MoE `tp_size/tp_rank` and EP groups (`250-258`).
  - `load_weights` (`1470-1687`) is the most critical DeepSeekV2 TP loading implementation:
    - `stacked_params_mapping` fusion mapping (`1474-1499`)
    - `expert_params_mapping` (`1503-1515`)
    - Generic stack params `weight_loader(param, loaded_weight, shard_id)` (`1536-1571`)
    - Expert params loaded via expert-aware loader (`1620-1655`)
    - Fallback to `default_weight_loader` (`1679-1683`).

---

## 4. Import Dependencies: TP Files That Must Be Read

## 4.1 Distributed TP API (Shared Dependency of layers/model_loader)

- `vllm/distributed/parallel_state.py`
  - `get_ep_group()`: `1254-1260`
  - `get_tensor_model_parallel_world_size()`: `1827-1829`
  - `get_tensor_model_parallel_rank()`: `1832-1834`

- `vllm/distributed/communication_op.py`
  - `tensor_model_parallel_all_reduce`: `12-14`
  - `tensor_model_parallel_all_gather`: `17-21`

## 4.2 Parameter Objects and weight_loader Protocol

- `model_executor/parameter.py`
  - `BasevLLMParameter` binds `tp_rank/tp_size`: `41-67`
  - `_ColumnvLLMParameter.load_column_parallel_weight`: `148-154`
  - `_ColumnvLLMParameter.load_merged_column_weight`: `156-177`
  - `_ColumnvLLMParameter.load_qkv_weight`: `178-201`
  - `RowvLLMParameter` (row parallel parameter): `204+`

This is the bridge that implements "layer shard semantics" into "parameter slice copying".

---

## 5. `model_executor/kernels`: Real Relationship with TP

- `kernels/linear/__init__.py`: Linear kernel registration and selection (`22+` large import section, `301+` selection function).
- `kernels/linear/scaled_mm/ScaledMMLinearKernel.py`: Kernel interface abstraction (`55+`).
- `kernels/linear/scaled_mm/triton.py`: Specific kernel implementation (mainly concerned with per-rank tensor processing, does not manage TP groups).

Conclusion:
- Kernel files generally don't write `get_tp_group().all_reduce()` style group communication;
- TP semantics are mainly in `layers`/`parameter`/`distributed`; kernels only consume already-sharded weights and inputs.

---

## 6. Minimum Implementation Mapping for DeepSeekV2 `TP=4` (For Minimal Framework)

When connecting the current minimal framework to TP=4, copy from the following mapping:

1. **Distributed interface layer**
   Reference: `distributed/parallel_state.py` + `communication_op.py`
   Implement `get_tp_rank/get_tp_size/all_reduce/all_gather`.

2. **Parallel linear layers**
   Reference: `layers/linear.py` (`Column/Merged/QKV/Row`)
   At least implement:
   - QKV sharding (with KV replication)
   - O/down projection all-reduce.

3. **Vocab parallelism**
   Reference: `layers/vocab_parallel_embedding.py`.

4. **Weight loading protocol**
   Reference: `parameter.py` + `weight_utils.py` + `deepseek_v2.py::load_weights`.
   Core:
   - `stacked_params_mapping` (q/k/v, gate/up merge)
   - `expert_params_mapping` (MoE expert parameter mapping)
   - `weight_loader(param, weight, shard_id, ...)`.

5. **MoE processing**
   Reference: `deepseek_v2.py` `SharedFusedMoE` path (`1503+` and `1620+`).
   First version can do TP + local expert mapping + necessary all-reduce, then gradually add EP/advanced scheduling.

---

## 7. Recommended Reading Order (Fastest Path to TP=4)

1. `model_executor/models/deepseek_v2.py` (first read `load_weights` and Attention/MLP/MoE structure)
2. `model_executor/layers/linear.py`
3. `model_executor/parameter.py`
4. `model_loader/weight_utils.py` + `sharded_state_loader.py`
5. `layers/vocab_parallel_embedding.py`
6. `distributed/parallel_state.py` + `communication_op.py`
7. `kernels/linear/*` (read last, confirm kernel backend)

---

## 8. Conclusion

If the goal is "subsequently implement DeepSeekV2 `TP=4` directly in the minimal framework", the most critical thing is not to modify kernels first, but to打通 these three layers first:

- **TP communication and rank state (distributed)**
- **Parallel layer semantics and parameter loading protocols (layers + parameter)**
- **DeepSeekV2-specific mapping loading (models/deepseek_v2.py::load_weights)**

These are all clearly indexed in the paths above and can serve as a direct implementation checklist.
