# DeepSeekV2 (MoE) TP=N Implementation Knowledge Guide in Minimal Framework (N∈{1,2,4,8})

Goal: Consolidate **directly transferable** TP/MoE key mechanisms from nano-sglang and mini-sglang to the current minimal inference framework, supporting subsequent implementation of **DeepSeekV2 Chat 16B TP=N sharding** (`N ∈ {1,2,4,8}`).

---

## 1. Define Boundaries: What Must Be Done in TP=N (N∈{1,2,4,8}) vs. What Can Wait

### Must Do (Phase 1)
- Initialize distributed (4 processes, 1 rank 1 card), obtain `tp_rank/tp_size`.
- Linear layer sharding: `ColumnParallel`, `RowParallel`, `QKVParallel`.
- Embedding/LM Head vocab dimension sharding (forward all-reduce / all-gather).
- Weight loading sharded by TP (including QKV fused and MoE expert parameters).
- MoE router output reduction within TP group (all-reduce).

### Can Wait (Phase 2)
- PyNCCL optimized communication, CUDA Graph, FlashInfer specialized paths.
- More complex parallelism (DP/PP/EP combinations) and hierarchical scheduling.

---

## 2. nano-sglang: TP and MoE Core Implementation Points

## 2.1 TP Process Groups and Communication Primitives

### Key Source Code
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/parallel_state.py`
  - `initialize_model_parallel()`: `19-85`
  - `get_tensor_model_parallel_world_size/rank`: `111-128`
  - `tensor_model_parallel_all_reduce`: `188-199`
  - `tensor_model_parallel_all_gather`: `201-227`
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/parallel_utils/utils.py`
  - `divide`: `17-21`
  - `split_tensor_along_last_dim`: `24-49`

### Transferable Conclusions
- The minimal TP framework interface is: `tp_rank/tp_size + all_reduce + all_gather + split_last_dim`.
- Minimal engineering can start with only tensor-parallel group, without launching pipeline group.

---

## 2.2 Linear Layer TP Sharding (Most Important)

### Key Source Code
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/linear.py`
  - `ColumnParallelLinear`: `138-232`
  - `MergedColumnParallelLinear`: `235-333`
  - `QKVParallelLinear`: `336-470`
  - `RowParallelLinear`: `473-595`

### Transferable Rules
- **ColumnParallel**: Shards output dimension; forward defaults to no communication (optional gather).
- **RowParallel**: Shards input dimension; `all_reduce` after forward.
- **QKVParallel**:
  - `num_heads = total_q_heads // tp_size`
  - When `total_kv_heads < tp_size`, KV head replication (`num_kv_head_replicas` logic, `380-385`).
  - Weight loading via `loaded_shard_id in ["q","k","v"]` segment slicing (`441-463`).

---

## 2.3 Vocab Sharding (Embedding/LM Head)

### Key Source Code
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`
  - `VocabParallelEmbedding`: `35-109`
  - `ParallelLMHead`: `112-150`

### Transferable Rules
- Embedding: Each rank retains only vocab shard, forward masks out-of-bounds tokens to 0, then `all_reduce` to merge.
- LM Head: Each rank only computes local vocab logits, final `all_gather` restores full vocabulary logits (or only gather on rank0).

---

## 2.4 MoE and TP Coupling

### Key Source Code
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/models/mixtral.py`
  - `MixtralMoE` initialization and expert allocation: `73-115`
  - MoE forward + `tensor_model_parallel_all_reduce`: `116-137`
  - Attention QKV/OProj TP: `140-215`
  - `load_weights` QKV shard and expert filtering: `339-380`

### Transferable Rules
- Experts partitioned by rank (`np.array_split(..., tp_size)[rank]`, `91-93`).
- Router (gate) can be replicated; expert FFN only computes local experts, then TP all-reduce on results.
- When loading weights, must **skip non-local rank experts** (`375-376`), otherwise parameter name mismatch or VRAM waste.

---

## 2.5 TP Initialization Entry (Manager Side)

### Key Source Code
- `meta-infer/ref_projects/nano-sglang/python/sglang/srt/managers/router/model_runner.py`
  - `init_process_group` + warmup all_reduce + `initialize_model_parallel`: `225-237`
  - Model build and load entry: `247-294`
  - KV pool head_num scaled by TP: `303-320`

### Transferable Rules
- Init distributed first, then build model and load shard.
- In KV cache capacity estimation, `num_kv_heads` must use `// tp_size` (or do mapping when replication is allowed).

---

## 3. mini-sglang: More "Engineering-Oriented" TP/MoE Reference

## 3.1 TP State and Communication Abstraction

### Key Source Code
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/info.py`
  - `DistributedInfo` / `set_tp_info` / `get_tp_info`: `6-31`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/distributed/impl.py`
  - `TorchDistributedImpl`: `25-41`
  - `DistributedCommunicator`: `63-70`
  - `enable_pynccl_distributed`: `73-90`

### Transferable Rules
- Encapsulate communication as `comm.all_reduce/all_gather`, business layer doesn't directly write `dist.*`, making future PyNCCL replacement easier.

---

## 3.2 TP Linear Layers (Concise Version)

### Key Source Code
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/linear.py`
  - `_LinearTPImpl`: `13-33`
  - `LinearColParallelMerged`: `56-69`
  - `LinearQKVMerged`: `71-89`
  - `LinearOProj`: `91-107`
  - `LinearRowParallel`: `109-127`

### Transferable Rules
- mini-sglang writes dimension mappings very clearly, suitable as a minimal engineering template.
- `LinearQKVMerged` supports `allow_replicate=True` for KV head replication (`83`).

---

## 3.3 MoE Layer and Post-Router Reduction

### Key Source Code
- `meta-infer/ref_projects/mini-sglang/python/minisgl/layers/moe.py`
  - `MoELayer`: `9-43`
  - Forward post `all_reduce`: `45-59`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/moe/fused.py`
  - `FusedMoe.forward`: `230-257`

### Transferable Rules
- MoE parameters sharded by TP on intermediate dimension (`intermediate_size_per_partition`, `33-43`).
- After each rank runs fused MoE backend, all-reduce to aggregate hidden states.

---

## 3.4 Qwen3-MoE Assembly Pattern (Mappable to DeepSeekV2)

### Key Source Code
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/qwen3_moe.py`
  - Model assembly entry: `18-80`
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/utils.py`
  - `MoEMLP` (gate replicated + experts): `53-76`
  - `RopeAttn` (`LinearQKVMerged` + `LinearOProj`): `79-124`

### Transferable Rules
- DeepSeekV2 can follow the same pattern: Attention uses QKV/O TP; MoE uses gate + experts + reduce.

---

## 3.5 Weight Sharding/Merge Loading (Very Critical)

### Key Source Code
- `meta-infer/ref_projects/mini-sglang/python/minisgl/models/weight.py`
  - Sharding rule constants: `13-31`
  - `_shard_tensor` (dim0/dim1/vocab/QKV-KV replication): `34-53`
  - Merge rules (`q_proj/k_proj/v_proj -> qkv_proj`, `gate/up -> gate_up_proj`): `16-30`, `55-60`
  - Streaming load main loop: `75-124`
  - MoE expert stack: `111-119`

### Transferable Rules
- Most valuable for minimal engineering: Complete **sharding + fusion** during loading, inference modules only consume "ready" parameters.
- DeepSeekV2 TP=N (`N ∈ {1,2,4,8}`) should also prioritize this "reshape at load time" strategy, rather than frequent reshape/cat during forward.

---

## 4. Direct Implementation Blueprint for DeepSeekV2 TP=N (N∈{1,2,4,8})

Below is the minimum viable plan based on the above source code, directly applicable to `meta-infer/engine`.

1. **Distributed startup**
   - Launch `N` processes (`N ∈ {1,2,4,8}`), set `LOCAL_RANK/RANK/WORLD_SIZE`.
   - In engine initialization, do `init_process_group("nccl", world_size=N, rank=rank)`.
   - Build TP group API following nano-sglang `parallel_state.py` (can skip PP for now).

2. **Replace linear layer abstractions**
   - Replace current projection layers with:
     - `ColumnParallelLinear`
     - `QKVParallelLinear` (supports KV replication)
     - `RowParallelLinear`
   - Rules copied from nano/mini dimension sharding.

3. **Embedding/LMHead TP-ization**
   - vocab sharded by rank.
   - embedding forward: mask + all_reduce.
   - lm_head forward: local logits + all_gather (or rank0 gather).

4. **MoE TP path**
   - gate remains replicated (simplest and most stable).
   - expert FFN parameters sharded by TP (prefer intermediate dimension).
   - After local expert computation, `all_reduce` to aggregate.

5. **Weight loading (most critical)**
   - Reference mini `models/weight.py`:
     - dim0/dim1 slicing by key rules;
     - q/k/v merged to qkv;
     - gate/up merged;
     - MoE experts stack;
   - Reference nano mixtral `load_weights`:
     - `weight_loader(param, tensor, shard_id)`;
     - Skip non-local rank experts.

6. **KV cache parameters**
   - `num_kv_heads` uses `div_even(..., tp_size, allow_replicate=True)` logic.
   - Cache bytes/token estimation synchronized with per-rank kv_heads.

7. **Verification strategy (TP=N)**
   - Under `tp ∈ {1,2,4,8}`, same prompt output length and eos behavior should be consistent.
   - Minor numerical differences allowed, but top-1/sampling statistics should be stable.
   - Start with single batch, then multi-batch, then long context.

---

## 5. File List to Copy First During Implementation (By Priority)

1. `nano-sglang/.../layers/linear.py`
2. `mini-sglang/.../models/weight.py`
3. `nano-sglang/.../models/mixtral.py`
4. `mini-sglang/.../layers/moe.py` + `.../moe/fused.py`
5. `nano-sglang/.../parallel_utils/parallel_state.py`
6. `mini-sglang/.../distributed/impl.py`
7. `mini-sglang/.../models/utils.py` (RopeAttn / MoEMLP assembly)

---

## 6. Notes (For DeepSeekV2)

- DeepSeekV2 is MoE + GQA/MLA structure, **KV heads and Q heads are asymmetric**: KV replication must be handled.
- Not recommended to implement complex EP (expert parallel) in the first version; first run MoE within TP and all-reduce.
- If the current minimal framework still uses HF `AutoModelForCausalLM` for direct forward, the above TP layers cannot be directly inserted; need to enter the "controllable model layer" path (similar to mini/nano's custom model assembly + load_weights).

---

*Note: This document only consolidates TP sharding knowledge from the specified directories and their direct import associations, aimed at supporting subsequent implementation of DeepSeekV2 TP=N (`N ∈ {1,2,4,8}`) in the current project.*
