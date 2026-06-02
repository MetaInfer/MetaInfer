# Phase 7 Spec Review Report

**PID**: 964266
**Role**: spec-reviewer
**Timestamp**: 2026-05-30T07:25:23Z
**Phase**: 7

---

## Spec Compliance: ✅ PASS

---

## Evidence Chain (逐条核验的 JSON Path)

### 1. class_hierarchy.QwenTPConfig — 12 Fields + from_model_dir

| # | Blueprint Field | Code Evidence | Verdict |
|---|----------------|---------------|---------|
| 1 | `model_dir: Path` | `engine/models/qwen.py:405` | ✅ |
| 2 | `hidden_size: int` | `engine/models/qwen.py:406` | ✅ |
| 3 | `intermediate_size: int` | `engine/models/qwen.py:407` | ✅ |
| 4 | `num_hidden_layers: int` | `engine/models/qwen.py:408` | ✅ |
| 5 | `num_attention_heads: int` | `engine/models/qwen.py:409` | ✅ |
| 6 | `num_key_value_heads: int` | `engine/models/qwen.py:410` | ✅ |
| 7 | `head_dim: int` | `engine/models/qwen.py:411` | ✅ |
| 8 | `vocab_size: int` | `engine/models/qwen.py:412` | ✅ |
| 9 | `rms_norm_eps: float` | `engine/models/qwen.py:413` | ✅ |
| 10 | `rope_theta: float` | `engine/models/qwen.py:414` | ✅ |
| 11 | `max_position_embeddings: int` | `engine/models/qwen.py:415` | ✅ |
| 12 | `tie_word_embeddings: bool = False` | `engine/models/qwen.py:416` | ✅ |

- `from_model_dir` factory: `engine/models/qwen.py:418-440` — reads `config.json` dynamically, all values from file, no hardcoded dims. ✅
- `head_dim` fallback: `engine/models/qwen.py:426` — `cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])` ✅

> 🟡 **Note**: Contract requirement stated "13 fields", but the blueprint `inference_blueprint.json` at `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenTPConfig.fields` lists exactly 12 fields. Code implements all 12. The extra expected field is ambiguous — either a counting error in the task spec or a missing blueprint field. Code is compliant with the blueprint.

### 2. class_hierarchy.QwenForCausalLMTP — 4 Core Attributes

| Blueprint Attr | Code Evidence | Verdict |
|---------------|---------------|---------|
| `self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)` | `engine/models/qwen.py:465` | ✅ |
| `self.layers = nn.ModuleList([QwenDecoderLayerTP(cfg) for _ in range(cfg.num_hidden_layers)])` | `engine/models/qwen.py:466-468` | ✅ |
| `self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)` | `engine/models/qwen.py:469` | ✅ |
| `self.lm_head = ParallelLMHead(...)` | `engine/models/qwen.py:470` | ⚠️ see below |

> 🟡 **Note — lm_head constructor argument order**: Blueprint `class_hierarchy` entry reads `ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)` (hidden_size first). Code uses `ParallelLMHead(cfg.vocab_size, cfg.hidden_size)` (vocab_size first). The **actual** `ParallelLMHead.__init__` signature at `engine/tp_layers/embedding.py:131` is `__init__(self, num_embeddings, embedding_dim, ...)` — i.e., `(vocab_size, hidden_size)`. The code's argument order matches the actual class API. This is a **blueprint documentation error** in `class_hierarchy`, not a code defect.

### 3. construction_chain

Blueprint: `config → model → load_weights → eval → init_custom_ar`

| Step | Code Evidence | Verdict |
|------|--------------|---------|
| `QwenTPConfig.from_model_dir(model_dir)` | `engine/models/qwen.py:418-440` | ✅ |
| `QwenForCausalLMTP(cfg, device, dtype)` | `engine/models/qwen.py:462-472` | ✅ |
| `model.load_weights()` | `engine/models/qwen.py:522-560` | ✅ |
| `model.eval()` | Not in model code — caller's responsibility | ✅ (correct separation) |
| `init_custom_ar(device=...)` inside `load_weights()` at line 560 | `engine/models/qwen.py:558-560` | ✅ (matches `load_weights_pseudocode` line 2477) |

> Note: Blueprint `load_weights_pseudocode` (line 2477) explicitly places `dist.barrier(); init_custom_ar(device=device)` inside `load_weights()`. The `construction_chain` is a high-level summary; the pseudocode is authoritative. Code matches pseudocode.

### 4. qwen_hf_key_mapping — 12 HF Key → Attr Mappings

| # | Blueprint HF Key | Code Dispatch Case | Evidence | Verdict |
|---|-----------------|-------------------|----------|---------|
| 1 | `model.layers.N.self_attn.q_proj.weight` | Buffer for QKV merge (Q) | `qwen.py:609-610` | ✅ |
| 2 | `model.layers.N.self_attn.k_proj.weight` | Buffer for QKV merge (K) | `qwen.py:611-612` | ✅ |
| 3 | `model.layers.N.self_attn.v_proj.weight` | Buffer for QKV merge (V) | `qwen.py:613-614` | ✅ |
| 4 | `model.layers.N.self_attn.o_proj.weight` | `o_proj.load_weight_shard(full)` | `qwen.py:615-616` | ✅ |
| 5 | `model.layers.N.mlp.gate_proj.weight` | Buffer for gate-up merge (gate) | `qwen.py:617-618` | ✅ |
| 6 | `model.layers.N.mlp.up_proj.weight` | Buffer for gate-up merge (up) | `qwen.py:619-620` | ✅ |
| 7 | `model.layers.N.mlp.down_proj.weight` | `down_proj.load_weight_shard(full)` | `qwen.py:621-622` | ✅ |
| 8 | `model.layers.N.input_layernorm.weight` | `layer.input_layernorm.weight.data.copy_(full)` | `qwen.py:623-624` | ✅ |
| 9 | `model.layers.N.post_attention_layernorm.weight` | `layer.post_attention_layernorm.weight.data.copy_(full)` | `qwen.py:625-626` | ✅ |
| 10 | `model.embed_tokens.weight` | `embed_tokens.load_weight_shard(full)` | `qwen.py:598-599` | ✅ |
| 11 | `lm_head.weight` | `lm_head.load_weight_shard(full)` | `qwen.py:602-603` | ✅ |
| 12 | `model.norm.weight` | `norm.weight.data.copy_(full)` | `qwen.py:600-601` | ✅ |

### 5. load_weights_pseudocode Compliance

Blueprint 5-step flow vs code:

| Blueprint Step | Code Evidence | Verdict |
|---------------|---------------|---------|
| 1. Read `model.safetensors.index.json` → `weight_map` | `qwen.py:533-536` | ✅ |
| 2. Group by safetensors file `{filename: [hf_keys]}` | `qwen.py:539-541` | ✅ |
| 3. For each file: `safe_open` → iterate keys → dispatch | `qwen.py:547-551` | ✅ |
| 4. After all files: merge QKV + merge Gate-Up | `qwen.py:553-555` | ✅ |
| 5. `dist.barrier()` + `init_custom_ar(device=...)` | `qwen.py:558-560` | ✅ |

> Architecture note: Code uses a **two-phase accumulate-then-merge** pattern (buffering Q/K/V and gate/up shards before merging) rather than the blueprint's single-pass `_load_one_weight`. This is a correct optimization: Q/K/V weights may reside in different safetensors files, so merge must happen after all files are read. The blueprint's `_load_one_weight` pseudocode implicitly assumes all three QKV weights arrive in order — the accumulate pattern eliminates this assumption. ✅

---

## ⚠️ 4 高发错误逐条核验

### 🔴 Error 1: QKV Cat Order — Q-K-V (严禁 K-Q-V)

| Aspect | Evidence |
|--------|----------|
| Merge code | `engine/models/qwen.py:667` — `torch.cat([q_shard, k_shard, v_shard], dim=0)` |
| Verdict | ✅ **Q-K-V order confirmed. No violation.** |

### 🔴 Error 2: double_shard_guard — 是否使用已有 load_weight_shard 方法

| Weight Target | Dispatch Method | double_shard_guard Present? | Evidence |
|--------------|----------------|---------------------------|----------|
| `embed_tokens` | `self.embed_tokens.load_weight_shard(full)` | ✅ `embedding.py:94-101` | `if shard.shape == self.weight.shape: copy_; else: slice` |
| `lm_head` | `self.lm_head.load_weight_shard(full)` | ✅ `embedding.py:168-175` | Same pattern |
| `o_proj` (RowParallel) | `layer.self_attn.o_proj.load_weight_shard(full)` | ✅ `linear.py:129-135` | `if shard.shape == self.weight.shape: copy_; else: slice[:, start:end]` |
| `down_proj` (RowParallel) | `layer.mlp.down_proj.load_weight_shard(full)` | ✅ `linear.py:129-135` | Same RowParallel pattern |
| `qkv_proj` (merged) | `self.layers[...].qkv_proj.load_weight_shard(merged)` | ✅ `linear.py:212-239` | Pre-sliced check + full-weight slicing with Q/K/V sections |
| `gate_up_proj` (merged) | `self.layers[...].gate_up_proj.load_weight_shard(merged)` | ✅ `linear.py:300-316` | Pre-sliced check + gate/up section slicing |
| `norm` (RMSNorm) | `self.norm.weight.data.copy_(full)` | N/A (1D weight, no sharding) | ✅ |
| `input_layernorm` | `layer.input_layernorm.weight.data.copy_(full)` | N/A (1D weight, no sharding) | ✅ |
| `post_attention_layernorm` | `layer.post_attention_layernorm.weight.data.copy_(full)` | N/A (1D weight, no sharding) | ✅ |

Verdict: ✅ **All sharded weights correctly routed through `load_weight_shard` methods with double_shard_guard.**

### 🔴 Error 3: 维度动态读取 (严禁硬编码 40960/12288/32768)

Full-text scan for hardcoded dimension literals (`40960`, `12288`, `32768`, `128` as head_dim, `32` as num_heads, `8` as num_kv_heads, `36` as num_layers) in the Phase 7 code (lines 394-700):

| Search | Result |
|--------|--------|
| Hardcoded `40960` | ❌ Not found in Phase 7 code |
| Hardcoded `12288` | ❌ Not found in Phase 7 code |
| Hardcoded `32768` | ❌ Not found in Phase 7 code |
| `hidden_size` source | `cfg['hidden_size']` from config.json (line 429) |
| `intermediate_size` source | `cfg['intermediate_size']` from config.json (line 430) |
| `num_hidden_layers` source | `cfg['num_hidden_layers']` from config.json (line 431) |
| `head_dim` source | `cfg.get('head_dim', fallback)` from config.json (line 426) |
| `inter_per_rank` | `self.cfg.intermediate_size // tp_size` (line 686) — dynamic |

Verdict: ✅ **All dimensions dynamically read from `config.json` via `QwenTPConfig`. No hardcoded dimensions.**

### 🔴 Error 4: KV Head Replication 分支 (tp > num_kv_heads 时复制)

| Aspect | Code Evidence | Verdict |
|--------|--------------|---------|
| `kv_heads_local` calculation | `qwen.py:652` — `kv_heads_local = max(1, self.cfg.num_key_value_heads // tp_size)` | ✅ |
| `kv_size_per_rank` | `qwen.py:653` — `kv_heads_local * self.cfg.head_dim` | ✅ |
| Normal case (`num_kv_heads >= tp_size`) | `qwen.py:659-661` — per-rank slice of K/V | ✅ |
| Replication case (`num_kv_heads < tp_size`) | `qwen.py:662-664` — `k_shard = k_full; v_shard = v_full` (all ranks get full) | ✅ |

For Qwen3-8B with TP=4: `num_kv_heads=8 >= tp_size=4` → normal case, per-rank slice. ✅
For edge case (e.g. tp_size=16 > num_kv_heads=8): `max(1, 8//16) = 1` → `kv_heads_local=1`, replication activates. ✅

Verdict: ✅ **KV head replication correctly implemented.**

---

## Phase 1-6 Code Integrity Check

Phase 7 code starts at line 390 with clear comment `# Phase 7: QwenTPConfig`. Pre-existing Phase 1-6 classes:

| Class | Lines | Verified Unmodified? |
|-------|-------|---------------------|
| `RMSNorm` | 52-79 | ✅ (Phase 1-5, unchanged) |
| `QwenAttentionTP` | 86-279 | ✅ (Phase 5, unchanged) |
| `QwenMLPTP` | 286-312 | ✅ (Phase 5/6, unchanged) |
| `QwenDecoderLayerTP` | 318-387 | ✅ (Phase 5/6, unchanged) |

No modifications to any Phase 1-6 code detected. All four classes retain their original signatures, attribute names, and logic.

---

## Issues Found

None. All 5 contract areas pass against the blueprint:

1. ✅ `QwenTPConfig`: 12/12 fields present (blueprint lists 12), `from_model_dir` dynamic, `head_dim` fallback
2. ✅ `QwenForCausalLMTP`: All 4 attrs correct (`embed_tokens`/`layers`/`norm`/`lm_head`)
3. ✅ `construction_chain`: config→model→load_weights (含 barrier+init_custom_ar); eval 由调用方负责
4. ✅ `qwen_hf_key_mapping`: 12/12 mappings present and correctly dispatched
5. ✅ `load_weights_pseudocode`: All 5 steps implemented, two-phase accumulate-then-merge is architecturally sound
6. ✅ QKV cat order Q-K-V
7. ✅ double_shard_guard on all sharded weights
8. ✅ All dims dynamic from config.json
9. ✅ KV head replication branch correct

---

## Blueprint Information Gaps

### 🟡 `class_hierarchy.QwenForCausalLMTP` — lm_head argument order mismatch

- **Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenForCausalLMTP.attrs[3]`
- **Blueprint says**: `ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)`
- **Code says** (`qwen.py:470`): `ParallelLMHead(cfg.vocab_size, cfg.hidden_size)`
- **Actual API** (`embedding.py:131`): `ParallelLMHead(num_embeddings, embedding_dim)` — i.e., `(vocab_size, hidden_size)`
- **Resolution**: Code is correct. Blueprint `class_hierarchy` entry should be updated to `ParallelLMHead(cfg.vocab_size, cfg.hidden_size)`.

### 🟡 Field count discrepancy

- Task spec says "13 fields 全部存在" but blueprint lists 12 fields for QwenTPConfig. Code implements all 12 blueprint fields.

---

## Conclusion

**Spec 审查通过，代码与蓝图契约一致，可移交 verification。**
