# Phase 7 Spec Review Report — QwenTPConfig + QwenForCausalLMTP (Weight Loading)

| Field | Value |
|-------|-------|
| **PID** | 3901543 |
| **Role** | spec-reviewer |
| **Timestamp** | 2026-06-08T19:45:59Z |
| **Phase** | 7 |
| **Spec Compliance** | **PASS** |

---

## Evidence Chain (逐条核对)

### 1. Blueprint: QwenTPConfig class_hierarchy fields
**Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenTPConfig`

| Blueprint Field | Code Location | Status |
|----------------|---------------|--------|
| model_dir: Path | `qwen.py:585` | **PASS** |
| hidden_size: int | `qwen.py:586` | **PASS** |
| intermediate_size: int | `qwen.py:587` | **PASS** |
| num_hidden_layers: int | `qwen.py:588` | **PASS** |
| num_attention_heads: int | `qwen.py:589` | **PASS** |
| num_key_value_heads: int | `qwen.py:590` | **PASS** |
| head_dim: int | `qwen.py:591` | **PASS** |
| vocab_size: int | `qwen.py:592` | **PASS** |
| rms_norm_eps: float | `qwen.py:593` | **PASS** |
| rope_theta: float | `qwen.py:594` | **PASS** |
| max_position_embeddings: int | `qwen.py:596` | **PASS** (default 32768, overwritten by config.json) |
| tie_word_embeddings: bool | — | **MISSING (minor)** — blueprint lists this field; code omits it. Never used downstream, no functional impact. |

Blueprint factory: `_load_qwen_tp_config(model_dir): AutoConfig.from_pretrained(...)` vs code: `QwenTPConfig.from_config(config_source)` which reads raw JSON. Naming differs but dynamic read is preserved. **PASS** (functional equivalent).

### 2. Blueprint: QwenForCausalLMTP class_hierarchy constructor
**Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenForCausalLMTP`

| Blueprint Attr | Code | Status |
|---------------|------|--------|
| `self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)` | `qwen.py:675` | **PASS** |
| `self.layers = nn.ModuleList([QwenDecoderLayerTP(cfg) for _ in range(cfg.num_hidden_layers)])` | `qwen.py:676-678` (36 layers) | **PASS** |
| `self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)` | `qwen.py:679` | **PASS** |
| `self.lm_head = ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)` | `qwen.py:680` — `ParallelLMHead(cfg.vocab_size, cfg.hidden_size)` | **PASS** (see note below) |

Blueprint `lm_head` signature says `(cfg.hidden_size, cfg.vocab_size, gather_output=True)` but actual `ParallelLMHead` constructor is `(num_embeddings, embedding_dim, bias, tp_size)`. Code correctly passes `(cfg.vocab_size, cfg.hidden_size)` matching the real constructor. The blueprint's parameter order is swapped relative to the actual `ParallelLMHead` implementation. `gather_output=True` does not exist as a constructor parameter (all_gather is always-on in LM head forward). This is a **blueprint documentation error**, not a code defect.

### 3. Blueprint: construction_chain (5 steps)
**Path**: `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.construction_chain`

| Step | Blueprint | Code | Status |
|------|-----------|------|--------|
| 1 | `cfg = _load_qwen_tp_config(model_dir)` | `QwenTPConfig.from_config(model_dir)` (see docstring `qwen.py:657`) | **PASS** |
| 2 | `model = QwenForCausalLMTP(cfg, device=device, dtype=torch.bfloat16)` | `qwen.py:669` constructor signature matches | **PASS** |
| 3 | `model.load_weights()` | `qwen.py:686` method implemented | **PASS** |
| 4 | `model.eval()` | NOT in code — this is a caller-side call, model does not call eval() internally | **PASS** (external step, not model's responsibility) |
| 5 | `init_custom_ar(device=device)` | `qwen.py:751-754` — called inside `load_weights()` after `dist.barrier()` | **PASS** |

### 4. Blueprint: Qwen dense loader (split_dim rules)
**Path**: `model_layer.architecture_knowledge_base.lazy_loader_synthesis_rules.qwen_dense_loader`

| Rule | Code Verification | Status |
|------|-------------------|--------|
| split_dim_0: q_proj, k_proj, v_proj, gate_proj, up_proj | `QKVColumnParallelLinear.load_weight_shard` (`linear.py:332-375`) shards dim=0; `MergedColumnParallelLinear.load_weight_shard` (`linear.py:224-252`) shards dim=0; qwen.py merge code passes full weight → slicing happens inside load_weight_shard | **PASS** |
| split_dim_1: o_proj, down_proj | `RowParallelLinear.load_weight_shard` (`linear.py:148-165`) shards dim=1 | **PASS** |

### 5. Blueprint: HF key mapping (all 12+2 mappings)
**Path**: `model_layer.architecture_knowledge_base.lazy_loader_synthesis_rules.qwen_hf_key_mapping`

| HF Key Pattern | Target | Code (`_dispatch_weight`) | Status |
|---------------|--------|---------------------------|--------|
| `model.embed_tokens.weight` | `embed_tokens` | `qwen.py:781` | **PASS** |
| `model.norm.weight` | `norm` | `qwen.py:783` | **PASS** |
| `lm_head.weight` | `lm_head` | `qwen.py:785` | **PASS** |
| `model.layers.N.input_layernorm.weight` | `layers[N].input_layernorm` | `qwen.py:792` | **PASS** |
| `model.layers.N.post_attention_layernorm.weight` | `layers[N].post_attention_layernorm` | `qwen.py:794` | **PASS** |
| `model.layers.N.self_attn.q_proj.weight` | `layers[N].self_attn.qkv_proj` (Q, cat Q-K-V) | `qwen.py:797-799` | **PASS** |
| `model.layers.N.self_attn.k_proj.weight` | `layers[N].self_attn.qkv_proj` (K) | `qwen.py:800-802` | **PASS** |
| `model.layers.N.self_attn.v_proj.weight` | `layers[N].self_attn.qkv_proj` (V) | `qwen.py:803-805` | **PASS** |
| `model.layers.N.self_attn.o_proj.weight` | `layers[N].self_attn.o_proj` | `qwen.py:807` | **PASS** |
| `model.layers.N.self_attn.q_norm.weight` | `layers[N].self_attn.q_norm` | `qwen.py:809` | **PASS** |
| `model.layers.N.self_attn.k_norm.weight` | `layers[N].self_attn.k_norm` | `qwen.py:811` | **PASS** |
| `model.layers.N.mlp.gate_proj.weight` | `layers[N].mlp.gate_up_proj` (gate, cat gate-up) | `qwen.py:814-816` | **PASS** |
| `model.layers.N.mlp.up_proj.weight` | `layers[N].mlp.gate_up_proj` (up) | `qwen.py:817-819` | **PASS** |
| `model.layers.N.mlp.down_proj.weight` | `layers[N].mlp.down_proj` | `qwen.py:821` | **PASS** |

merge_order: Q-K-V **PASS**; Gate-Up **PASS** (verified below).

### 6. Blueprint: Qwen3-8B model dims
**Path**: `model_layer.architecture_knowledge_base.qwen_series_dense.qwen3_8b_model_dims`

All dimensions are read dynamically from `config.json` via `from_config()`. No hardcoded values found in QwenTPConfig or QwenForCausalLMTP. **PASS**.

---

## Eight Mandate Checks (四大高发错误 + 4 扩展)

### Check 1: QKV cat order (Q-K-V, NOT K-Q-V)

| Location | Code | Order |
|----------|------|-------|
| `_try_merge_qkv` | `qwen.py:828` | `torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)` | **Q-K-V PASS** |
| `load_weights` finalize | `qwen.py:737-738` | `torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)` | **Q-K-V PASS** |
| `QKVColumnParallelLinear.forward` split | `linear.py:329` | `y.split([q_size, kv_size, kv_size], dim=-1)` | **Q-K-V PASS** |
| `QKVColumnParallelLinear.load_weight_shard` | `linear.py:374` | `torch.cat([q_shard, k_shard, v_shard], dim=0)` | **Q-K-V PASS** |

No K-Q-V anywhere. **PASS**.

### Check 2: double_shard_guard

| Class | File:Line | Guard Pattern |
|-------|-----------|---------------|
| `ColumnParallelLinear` | `linear.py:84-92` | `if weight.shape == self.weight.shape: copy_(); else: slice dim=0` | **PASS** |
| `RowParallelLinear` | `linear.py:148-165` | `if weight.shape == self.weight.shape: copy_(); else: slice dim=1` | **PASS** |
| `MergedColumnParallelLinear` | `linear.py:240-252` | `if weight.shape == self.weight.shape: copy_(); else: slice gate+up shards dim=0` | **PASS** |
| `QKVColumnParallelLinear` | `linear.py:350-375` | `if weight.shape == self.weight.shape: copy_(); else: slice Q/K/V shards dim=0` | **PASS** |
| `VocabParallelEmbedding` | `embedding.py:94-100` | `if weight.shape == self.weight.shape: copy_(); else: slice dim=0` | **PASS** |
| `ParallelLMHead` | `embedding.py:184-190` | `if weight.shape == self.weight.shape: copy_(); else: slice dim=0` | **PASS** |
| `RMSNorm` | `qwen.py:88-90` | `copy_()` only (RMSNorm always replicated; shape always matches by design) | **PASS** |

All `load_weight_shard` methods that may receive pre-sliced weights have shape-equality guard before slicing. **PASS**.

Note: In `qwen.py`'s `_try_merge_qkv`, the full-weight cat [6144, 4096] is passed to `load_weight_shard`, which detects shape mismatch (self.weight is [1536, 4096]) and applies the correct per-rank slicing. The intermediate full cat is a temporary GPU tensor freed after `load_weight_shard` returns. **PASS**.

### Check 3: Each rank loads only its shard (no torch.zeros full pre-allocation)

| Component | Per-rank weight shape | Proof |
|-----------|----------------------|-------|
| `QKVColumnParallelLinear.weight` | `[1536, 4096]` (not [6144, 4096]) | `linear.py:308-309` — `torch.empty(out_features_per_rank, hidden_size)` |
| `MergedColumnParallelLinear.weight` | `[6144, 4096]` (not [24576, 4096]) | `linear.py:206-207` — `torch.empty(gate_up_out_dim, hidden_size)` |
| `RowParallelLinear.weight` (o_proj) | `[4096, 1024]` (not [4096, 4096]) | `linear.py:128-129` — `torch.empty(out_features, in_features_per_rank)` |
| `RowParallelLinear.weight` (down_proj) | `[4096, 3072]` (not [4096, 12288]) | Same pattern |
| `VocabParallelEmbedding.weight` | `[37984, 4096]` (rank 0, not [151936, 4096]) | `embedding.py:60-61` — `torch.empty(local_vocab_size, embedding_dim)` |
| `ParallelLMHead.weight` | `[37984, 4096]` (rank 0) | `embedding.py:151-152` — `torch.empty(local_vocab_size, embedding_dim)` |

No `torch.zeros(全量)` pre-allocation anywhere. All parameters are initialized at per-rank size. **PASS**.

### Check 4: q_norm / k_norm dispatch

| HF Key | Target | Code |
|--------|--------|------|
| `model.layers.N.self_attn.q_norm.weight` | `layer.self_attn.q_norm` | `qwen.py:808-809` | **PASS** |
| `model.layers.N.self_attn.k_norm.weight` | `layer.self_attn.k_norm` | `qwen.py:810-811` | **PASS** |

Both dispatched, RMSNorm `load_weight_shard` copies full weight (replicated). **PASS**.

### Check 5: Gate-Up cat order (gate-up, NOT up-gate)

| Location | Code | Order |
|----------|------|-------|
| `_try_merge_gate_up` | `qwen.py:837` | `torch.cat([parts["gate"], parts["up"]], dim=0)` | **gate-up PASS** |
| `load_weights` finalize | `qwen.py:745-746` | `torch.cat([parts["gate"], parts["up"]], dim=0)` | **gate-up PASS** |
| `MergedColumnParallelLinear.load_weight_shard` | `linear.py:251` | `torch.cat([gate_shard, up_shard], dim=0)` | **gate-up PASS** |

No up-gate anywhere. **PASS**.

### Check 6: QwenTPConfig dynamic read from config.json (no hardcoded dims)

**Path**: `qwen.py:598-642` (`from_config`)

All 11 fields populated directly from the loaded dict `d`. No hardcoded numeric values. The `head_dim` fallback (`qwen.py:627`) dynamically computes `hidden_size // num_attention_heads` using values from the config dict. **PASS**.

Minor: `max_position_embeddings` has default `32768` in the dataclass field declaration (`qwen.py:596`), but `from_config` always overwrites it with the config.json value at line 641. The default is never relied upon in practice when `from_config` is used. For direct dataclass instantiation without `from_config`, the default would apply. **PASS** (no hardcoded dims reach inference path).

### Check 7: construction_chain 5 steps

Blueprint 5-step sequence matched to code locations:

1. config: `QwenTPConfig.from_config(model_dir)` — `qwen.py:598-642` **PASS**
2. model construct: `QwenForCausalLMTP(cfg, device, dtype)` — `qwen.py:669-680` **PASS**
3. load_weights: `model.load_weights()` — `qwen.py:686-754` **PASS**
4. eval: `model.eval()` — caller-side, not model's internal concern **PASS**
5. init_custom_ar: `init_custom_ar(device=device)` — `qwen.py:751-754` (inside `load_weights`) **PASS**

Step 5 is folded into `load_weights` rather than being a separate post-load step, but it executes at the right moment: after all weights on GPU, before first forward. **PASS**.

### Check 8: QwenForCausalLMTP module tree construction order

| Order | Component | Code | Status |
|-------|-----------|------|--------|
| 1 | `self.embed_tokens` | `qwen.py:675` — `VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)` | **PASS** |
| 2 | `self.layers` | `qwen.py:676-678` — `nn.ModuleList([QwenDecoderLayerTP(cfg) for _ in range(36)])` | **PASS** |
| 3 | `self.norm` | `qwen.py:679` — `RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)` | **PASS** |
| 4 | `self.lm_head` | `qwen.py:680` — `ParallelLMHead(cfg.vocab_size, cfg.hidden_size)` | **PASS** |

The attribute names match blueprint: `.embed_tokens`, `.layers` (ModuleList), `.norm`, `.lm_head`. Module tree order: embed → layers[0..35] → norm → lm_head. **PASS**.

---

## Encoding Iron Laws (AGENT_SKILL.md §1) Scan

| Law | Check | Status |
|-----|-------|--------|
| fused_add_rms_norm uses self.weight (not cross-layer) | `qwen.py:507,510` use `self.input_layernorm.weight`; `qwen.py:518,559,563` use `self.post_attention_layernorm.weight` | **PASS** |
| KV head replication | `qwen.py:120-124` — `if num_kv_heads >= tp_size: divide; else: num_kv_heads=1` | **PASS** |
| all_gather_last_dim = dist.all_gather + torch.cat | Deferred to `engine/tp_layers/distributed.py` (Phase 2 artifact) | **PASS** (not in this file) |
| QKV cat order Q-K-V | Verified above (Check 1) | **PASS** |
| Gate-Up cat order gate-up | Verified above (Check 5) | **PASS** |
| block_size=256 | `qwen.py:157` — `self._kv_block_size = 256` | **PASS** |
| block_table dtype=int32 | `qwen.py:201` — `torch.zeros(1, num_blocks, dtype=torch.int32)` | **PASS** |
| 维度值来自 config.json 动态读取 | Verified above (Check 6) | **PASS** |
| rmsnorm_precision_law | `qwen.py:56-57` — `self.weight * x_f.to(input_dtype)`, NOT `.float()` first | **PASS** (uses vLLM kerenel, not hand-written PyTorch) |
| tp_linear_load_no_double_shard | Verified above (Check 2) | **PASS** |

---

## Issues Found

None that warrant a FAIL. One minor observation:

- **Blueprint QwenTPConfig field `tie_word_embeddings: bool` missing from code dataclass** (`qwen.py:578-596`). The field is not used anywhere in the inference pipeline. No functional impact. Equivalent to **PASS**.

- **Blueprint `lm_head` constructor disagrees with actual `ParallelLMHead` interface**: blueprint says `ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)` but the actual constructor is `ParallelLMHead(num_embeddings, embedding_dim, bias, tp_size)`. The code correctly uses the real constructor `(cfg.vocab_size, cfg.hidden_size)`. This is a **blueprint documentation discrepancy**, not a code defect.

---

## Blueprint Information Gaps

- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.construction_chain`: blueprint references `_load_qwen_tp_config(model_dir)` as the config factory, but the code implements `QwenTPConfig.from_config(config_source)`. Naming differs; functional semantics preserved.

- `framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_tp_model_interfaces.class_hierarchy.QwenForCausalLMTP.attrs[3]`: blueprint says `ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)` but the actual `ParallelLMHead` class (`engine/tp_layers/embedding.py:129`) takes `(num_embeddings, embedding_dim, bias, tp_size)`. The blueprint swaps the first two arguments and references a non-existent `gather_output` parameter. The code correctly matches the real interface.

---

## Verdict

**Spec Compliance: PASS**

Spec review passes. The code correctly implements:
- QKV cat in Q-K-V order (all 4 locations)
- double_shard_guard in all 7 `load_weight_shard` implementations
- Per-rank (not full) weight allocation
- q_norm and k_norm dispatch both handled
- Gate-Up cat in gate-up order (all 3 locations)
- Dynamic config.json read via `from_config()` (no hardcoded dims)
- 5-step construction chain (config -> model -> load_weights -> eval -> CustomAR)
- Correct module tree order (embed_tokens -> layers[36] -> norm -> lm_head)

All 14 HF key mappings present and routed correctly. No blocking issues. Code is ready for verification.
