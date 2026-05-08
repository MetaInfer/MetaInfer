# Task10 TP Debugging Experience (Qwen TP=4)

This document records the problems, root causes, and fixes encountered during this round of `Qwen TP` real joint debugging, for reuse by subsequent Agents when generating TP components. Among them, **§3.2/§3.3** (RoPE `rotate_half`, RMSNorm fp32) are the key reasons for this round's **short-sentence repetition/text degradation** even after weights and sharding were correct; **§6** is the checklist for migrating to **DeepSeek-like MoE** and "one-shot full framework generation".

## 1. Error 1: Embedding Shard Loading shape=0

- **Phenomenon**
  - Error: `The size of tensor a (...) must match the size of tensor b (0)`.
  - Occurred in `VocabParallelEmbedding.load_weight_shard`.
- **Root Cause**
  - In `qwen.py`, `_load_tensor(..., split_dim=0)` already took the local vocab shard by `tp_rank`.
  - `load_weight_shard` sliced again by `vocab_start:vocab_end`, causing rank>0 to produce empty tensors.
- **Fix**
  - `load_weight_shard` added branch:
    - If the input tensor's first dimension already equals `local_vocab_size`, directly copy;
    - Otherwise slice by full vocab range.
- **Experience**
  - For TP parameter loading, must be clear whether "input is full weight or local shard"; don't slice at two levels repeatedly.

## 2. Error 2: Qwen Config Triggers DeepSeek MLA Field Access Exception

- **Phenomenon**
  - Error: `AttributeError: 'QwenTPConfig' object has no attribute 'qk_nope_head_dim'`.
  - Occurred in KV budget estimation path.
- **Root Cause**
  - `hf_deepseek_v2_kv_bytes_per_token` was hardcoded with DeepSeek MLA formula fields early on.
  - Qwen is Dense/GQA config, doesn't have `qk_nope_head_dim/qk_rope_head_dim/v_head_dim`.
- **Fix**
  - In `engine/kv_specs.py`, added Dense/GQA fallback:
    - Use `num_key_value_heads` + `head_dim` to estimate K/V bytes.
  - In `RealModelRunner` config printing, distinguish MLA vs Dense/GQA log formats.
- **Experience**
  - KV estimation functions should use "model family branching" rather than "single model hardcoding".

## 3. TP Output Short Sentence/Fragment Repetition, Text Degradation

Short sentence repetition, synonymous fragments flooding the screen, is common on TP and **multi-source**: both data path (mask/position) issues and **subtle inconsistencies between custom forward and HuggingFace's same model family in "local operators"**. Below are two sub-experience records.

### 3.1 Left Padding Batch + Inconsistent Position/Mask Semantics with HF

- **Phenomenon**
  - TP output shows many repeated phrases (e.g., "苏州园林的特色" repeating).
- **Root Cause**
  - Early TP runner used left padding batch processing, but custom Qwen forward didn't fully align with HF's `attention_mask/position_ids` semantics.
  - Causing attention context pollution, manifesting as repeated, degraded text.
- **Fix**
  - In `QwenTPModelRunner.run`, changed to **per-sequence forward** (each sample input individually, no left padding batch concatenation), then sample from last position logits.
  - Though slower, can first ensure semantic consistency with HF baseline.
- **Experience**
  - During TP correctness phase, prioritize "per-sequence no padding" conservative path, performance optimization (batched + mask) in later phases.

### 3.2 RoPE `rotate_half` Inconsistency with HF (Qwen3 / Llama "first-half/second-half" vs Wrong "even/odd dimensions")

- **Phenomenon**
  - After mask/aligned and weight loading without double slicing, **TP=4 vs HF single card** still shows significant difference: same prompt output contains **short-sentence level repetition, nonsensical parallelism**, inconsistent with single card fluent answer.
- **Root Cause**
  - In `transformers`, **Qwen3**'s `rotate_half` is: last dimension **split from the middle** into `x1, x2`, then `cat((-x2, x1))` (consistent with Llama/most open-source implementations).
  - If custom version wrongly uses **even/odd dimension interleaving** (`x[...,::2]`, `x[...,1::2]` then stack/flatten), it doesn't match Qwen3's `cos/sin` layout `cat([freqs, freqs], dim=-1)`, equivalent to **phase/position encoding entire layer misalignment** throughout attention, won't simply manifest as "overall logits one constant difference", but often manifests as **degenerate distribution → repeatedly sampling local high-frequency fragments**.
- **Fix**
  - In `engine/models/qwen.py`, aligned `_rotate_half` line by line with HF `modeling_qwen3`; `_apply_rope` computes `emb` in float precision with autocast disabled, then `to` back to activation dtype; if 2D `position_ids` exist, need to be consistent with `Qwen3RotaryEmbedding`'s `inv_freq.expand` + matmul behavior (see implementation in same file).
- **Experience**
  - **When doing custom forward, RoPE must be consistent with target checkpoint's HF class implementation**: `rotate_half` is "split-half rotate" or "interleave rotate" varies by model family; don't copy a RoPE from another repo and assume it's universal.
  - When debugging, prioritize cross-referencing **`transformers.models.<model_name>.modeling_***'s `apply_rotary_pos_emb` and `RotaryEmbedding.forward`, compare `apply_rope` tensor on CPU with same random `q`, rather than just盯着 TP all_reduce.

### 3.3 RMSNorm Uses Full Half-Precision Variance Under bf16 (Inconsistent with Qwen3RMSNorm)

- **Phenomenon**
  - When叠加 with 3.2, further amplifies distribution shift; alone can also cause logits drift.
- **Root Cause**
  - HF's `Qwen3RMSNorm` computes `variance` and `rsqrt` in **float32**, then multiplies with `weight`, finally converts back to `hidden_states` dtype.
- **Fix**
  - Custom `RMSNorm` same as HF: `input_dtype` → `float32` → normalize → `* weight` → `to(input_dtype)`.
- **Experience**
  - For "existing weights" inference path, **Norm's numerical path should be as consistent as possible with official inference**; TP only slices matrices, don't擅自 change Norm's precision assumptions, unless with complete benchmark testing.

## 4. NCCL Error Code 2/3 Explanation and Handling Suggestions

- **Phenomenon**
  - At process exit, see RCCL/NCCL `socket... error code 2/3`, `Abort COMPLETE`, `Proxy Service` etc. logs.
- **Explanation**
  - Mostly communicator async abort tail logs during test process exit, not necessarily indicating computation failure.
  - If test summary is `PASSED`, usually can be treated as exit noise.
- **Suggestion**
  - Prioritize looking at pytest results and main exception stack, don't judge failure solely by NCCL tail logs.
  - If need to reduce noise, can do process group cleanup at test tail; but avoid repeated destroy/recreate in parameterized multi-case causing额外 network instability.

## 5. This Round's Verification Conclusion

- `torchrun --nproc_per_node=4 -m pytest tests/test_qwen_tp_real.py -v -s` passes on Dense Qwen3 (historically reported `2 passed`; subject to current cases and model path).
- Notes:
  - TP shard loading chain works;
  - Key probes (weight shape/device, logits device/shape, per-rank memory) output normally;
  - **HF single card** should still serve as **ground truth reference**, per-layer or at least first layer/first token logits diff, then proceed with batched+mask optimizations.

## 6. Transferable: DeepSeek-like MoE TP and "One-Shot Full Inference Framework" Checklist

The reader of this section is the **subsequent Agent/person**: the pitfalls踩 on Dense Qwen3 will reappear on **MoE (like DeepSeek)** in the form of "more branches + more weight names + per-token multiple experts"; if using Cursor to one-shot generate **TP-containing full-stack inference framework**, please treat the below as **mandatory answer items**, to avoid only implementing Column/Row linear but forgetting model-family-specific operators.

### 6.1 Habits Directly Inherited from Qwen3 Dense

- **Use HF/official `modeling_*` as single source of truth**
  - For each type of attention, RoPE, RMS/LayerNorm, expert routing, KV cache layout, **name the file and function** in the repo (e.g., `Qwen3Attention`, `apply_rotary_pos_emb`), custom forward **line-by-line comparison**, not from memory copying RoPE from "generic Llama" articles.
- **Norm / softmax / scale dtype strategy**
  - Consistent with **official implementation** of pre-training/inference weights (e.g., Norm in fp32, attn_softmax in fp32 then cast back); TP doesn't replace this contract.
- **First run through no-padding single sequence**, then open batch and long context optimization.

### 6.2 Points That **Must** Be Thought Through on MoE

- **Expert Parallel (EP) and Tensor Parallel (TP) dimensions**
  - Dense only has "column/row parallel" slicing `hidden` and `ffn`; MoE also has **router, gate, expert weights** sharded by expert or hidden dimension. Need clarity: which operators **follow TP rank**, which **follow EP rank**, whether same batch of processes reuses multiple `process_group`s.
- **Routing and all_to_all / all_gather**
  - Wrong implementations often manifest as **some tokens never go through experts** or **expert outputs repeatedly accumulated**; harder to visually verify than Dense's `all_reduce`, need to prepare **small batch, fixed random seed, compare single-card per-token routing indices**.
- **Same type of "short sentence repetition/garbled" symptoms**
  - On Dense, root cause is often **RoPE/Norm/position**; on MoE, also check: **shared expert, device expert, whether training items (like aux loss) were误 introduced in inference**. May still manifest as repeated text, need to **排除 by subsystem**.

### 6.3 Suggestions for "Cursor One-Shot Complete TP Framework" Prompt/Structure

If want less返工, one-shot generate **joint-debuggable** skeleton, recommend explicitly requesting generated artifacts include (can directly write into `AGENTS.md` or design doc):

1. **Model family source of truth**: Path in `transformers` of corresponding class + list of classes to align (Attention, RMSNorm, Rotary, MoE layers, etc.).
2. **Precision conventions**: Each operator's input/intermediate/output dtype (consistent with HF `forward` `autocast` behavior, or mark **sections where autocast is prohibited**).
3. **TP/EP grouping and communication primitives table**: Which of `all_reduce` / `all_gather` / `reduce_scatter` / `all_to_all` is used after each module, tensor shape and row/column major order.
4. **Weight loading contract**: `safetensors` key names, whether already sharded, **prohibition of double slicing** convention (see §1).
5. **KV and multi-model family**: `kv_specs` or equivalent layer's table-driven **MLA / GQA / MHA branching** (see §2), avoiding one formula for all checkpoints.
6. **Minimum acceptance**: `tp=1` custom forward vs HF **same prompt first token or full segment logits alignment**; then `tp=n` vs `tp=1` **text or logits consistent**; finally throughput.

> **One-sentence memo**: TP sharding correct, communication no deadlock, still possible "looks runnable, output like broken model" — **RoPE/Norm/position differ from HF by one convention, output will look like disaster**; add **routing and expert communication** on MoE, without single-card/single-process ground truth alignment habit, debugging cost rises **exponentially**.
