# Phase 4 Spec Review Report — TP Embedding

| 字段 | 值 |
|------|-----|
| PID | 3823466 |
| Role | spec-reviewer |
| Timestamp | 2026-06-09T00:00:00Z |
| Phase | 4 |
| Target File | `engine/tp_layers/embedding.py` |
| Blueprint Nodes Reviewed | `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_embedding_and_lm_head`, `framework_layer.todo_generation_playbook.phase_4_tp_embedding`, `model_layer.architecture_knowledge_base.global_primitives_constraints.tp_linear_load_no_double_shard`, `class_hierarchy.qwen3_tp_model_interfaces.QwenForCausalLMTP` |
| Review Method | Independent — code and blueprint only; prior reviewer output ignored |

## Spec Compliance: PASS

---

## Evidence Chain

### VocabParallelEmbedding

- **`tp_embedding_and_lm_head.vocab_parallel_embedding.forward_pseudocode`** (blueprint lines 862-868)
  : PASS @ `embedding.py:64-82` — Line-by-line match with blueprint pseudocode. `mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)` (line 71), `local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)` (line 74), `out = F.embedding(local_ids, self.weight)` (line 76), `out = out.masked_fill((~mask).unsqueeze(-1), 0)` (line 79), `return all_reduce_sum(out)` (line 82).

- **`tp_embedding_and_lm_head.vocab_parallel_embedding.input_ids`** (blueprint line 857: `[B, T] int64`)
  : PASS @ `embedding.py:64` — `def forward(self, input_ids: torch.Tensor) -> torch.Tensor:`. Accepts int64 input_ids.

- **`tp_embedding_and_lm_head.vocab_parallel_embedding.output_after_all_reduce`** (blueprint line 860: `[B, T, hidden_size]`)
  : PASS @ `embedding.py:76,82` — `F.embedding` produces `[B, T, embedding_dim]`; `all_reduce_sum` preserves shape. `embedding_dim` = `hidden_size`.

- **vocab_start/end range calculation — covers non-divisible vocab_size**
  : PASS @ `embedding.py:54-58` — `per_rank = num_embeddings // tp_size; remainder = num_embeddings % tp_size; self.local_vocab_size = per_rank + (1 if self.tp_rank < remainder else 0); self.vocab_start = self.tp_rank * per_rank + min(self.tp_rank, remainder); self.vocab_end = self.vocab_start + self.local_vocab_size`.
  Independently verified with python3:
  - vocab_size=151936, tp=4: `[(0,37984), (37984,75968), (75968,113952), (113952,151936)]` — correct, no gaps/overlaps
  - vocab_size=151937, tp=4: `[(0,37985), (37985,75969), (75969,113953), (113953,151937)]` — correct, remainder=1 handled
  - vocab_size=10, tp=4: `[(0,3), (3,6), (6,8), (8,10)]` — correct, remainder=2 handled
  - vocab_size=11, tp=4: `[(0,3), (3,6), (6,9), (9,11)]` — correct, remainder=3 handled
  Formula verified: `vocab_start = i*per_rank + min(i, remainder)` correctly places start after first `min(i, remainder)` ranks (each bearing `per_rank+1`).

- **Mask correctness: `masked_fill(~mask, 0)` before `all_reduce_sum`**
  : PASS @ `embedding.py:71,74,79` — Mask created via `(input_ids >= self.vocab_start) & (input_ids < self.vocab_end)` (line 71). Out-of-range `local_ids` zeroed via `masked_fill(~mask, 0)` (line 74). Out-of-range embeddings zeroed via `out.masked_fill((~mask).unsqueeze(-1), 0)` (line 79). After `all_reduce_sum`, exactly one rank has non-zero contribution per token.

- **`tp_linear_load_no_double_shard` guard** (blueprint `global_primitives_constraints`, lines 2379-2381: "若传入权重 shape == self.weight.shape，则直接 copy_。仅当传入权重是全量张量时，才可按 tp_rank 执行切片")
  : PASS @ `embedding.py:94-100` — `if weight.shape == self.weight.shape: self.weight.data.copy_(weight)` (already pre-sliced, line 95-96); `else: shard = weight[self.vocab_start:self.vocab_end, :].contiguous(); self.weight.data.copy_(shard)` (full weight sliced by tp_rank, lines 99-100).

- **Class name: `VocabParallelEmbedding`**
  : PASS @ `embedding.py:22` — `class VocabParallelEmbedding(nn.Module):`. Matches blueprint line 1891 (`phase_4_tp_embedding.implementation_todos[0]`).

- **Weight shape: `[local_vocab_size, embedding_dim]`**
  : PASS @ `embedding.py:60-62` — `nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim, dtype=torch.float32))`. Matches blueprint line 858: `[vocab_size/tp, hidden_size]`.

- **Forward signature: `forward(input_ids)`**
  : PASS @ `embedding.py:64` — `def forward(self, input_ids: torch.Tensor) -> torch.Tensor:`. Accepts `[B,T] int64` as contracted.

### ParallelLMHead

- **`tp_embedding_and_lm_head.parallel_lm_head.forward_pseudocode`** (blueprint lines 876-880)
  : PASS @ `embedding.py:159-172` — `local_logits = F.linear(hidden_states, self.weight, self.bias)` (line 166) — blueprint says `F.linear(hidden_states, self.weight)`; code adds optional `self.bias` (defaults to `None`, `F.linear` treats `bias=None` as no-bias mode — compatible extension). `logits = all_gather_last_dim(local_logits)` (line 171) — exact match. `return logits` (line 172).

- **CRITICAL: Uses `all_gather_last_dim`, NOT `all_reduce`** (blueprint line 879)
  : PASS @ `embedding.py:171` — `logits = all_gather_last_dim(local_logits)`. Confirmed: all_gather (concatenation along last dim), not all_reduce_sum. Each rank computes logits for a different vocab shard; summing would corrupt the result. Correct operation.

- **Output shape: `[B, T, vocab_size]`** (blueprint line 874)
  : PASS @ `embedding.py:163,171-172` — Docstring: `returns: [B, T, num_embeddings]`. After `all_gather_last_dim`, all local vocab slices concatenated to full vocabulary — output is `[B, T, num_embeddings]` = `[B, T, vocab_size]`. Verified: sum of `local_vocab_size` across all ranks always equals `num_embeddings` (range calculation guarantees this).

- **`all_gather_last_dim` implementation** (blueprint-collective contract via AGENT_SKILL.md line 241)
  : PASS @ `engine/tp_layers/distributed.py:220-234` — uses `dist.all_gather(outs, x)` (line 233), NOT `all_gather_into_tensor`. Then `torch.cat(outs, dim=-1)` (line 234). Correct.

- **`tp_linear_load_no_double_shard` guard**
  : PASS @ `embedding.py:184-190` — Identical guard to VocabParallelEmbedding: `if weight.shape == self.weight.shape: self.weight.data.copy_(weight)` (line 185-186); `else: shard = weight[self.vocab_start:self.vocab_end, :].contiguous(); self.weight.data.copy_(shard)` (lines 189-190).

- **Class name: `ParallelLMHead`**
  : PASS @ `embedding.py:107` — `class ParallelLMHead(nn.Module):`. Matches blueprint line 1892 (`phase_4_tp_embedding.implementation_todos[1]`).

- **Forward signature: `forward(hidden_states)`**
  : PASS @ `embedding.py:159` — `def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:`. Accepts `[B,T,hidden_size]` as contracted.

### General / Cross-Cutting

- **Import from `engine.tp_layers.distributed` (not external pip package)**
  : PASS @ `embedding.py:15` — `from engine.tp_layers.distributed import all_reduce_sum, all_gather_last_dim, _get_world_size`. All collectives imported from the local engine module. No external pip package imports for TP communication.

- **No hardcoded dimension values — all via constructor args**
  : PASS @ `embedding.py:39-43` (VocabParallelEmbedding ctor), `embedding.py:129-134` (ParallelLMHead ctor) — `num_embeddings`, `embedding_dim`, `tp_size` all parameterized. No magic numbers.

- **`phase_4_tp_embedding.implementation_todos` fully addressed** (blueprint lines 1891-1892)
  : PASS @ Both items implemented:
  1. VocabParallelEmbedding: mask -> masked_fill(~mask,0) -> all_reduce_sum @ `embedding.py:64-82`
  2. ParallelLMHead: F.linear(hidden, weight) -> local logits -> all_gather_last_dim -> full vocab @ `embedding.py:159-172`

- **Constructor parameter count matches `phase_4_tp_embedding` contract**
  : PASS — The `tp_embedding_and_lm_head.parallel_lm_head.forward_pseudocode` contract does not constrain constructor parameters beyond weight shape `[vocab_size/tp, hidden_size]`. The code's `__init__(num_embeddings, embedding_dim, bias=False, tp_size=None)` produces this weight shape correctly: `self.weight` is `[local_vocab_size, embedding_dim]` = `[vocab_size/tp, hidden_size]`.

---

## Issues Found: None

No contract violations detected. All 9 key verification items pass independently.

---

## Blueprint Information Gaps

- **`class_hierarchy.qwen3_tp_model_interfaces.QwenForCausalLMTP.attrs[3]`** (blueprint line 1201):
  : YELLOW_FLAG — Constructor calling convention in `class_hierarchy` is inconsistent with code constructor.
  Blueprint says: `self.lm_head = ParallelLMHead(cfg.hidden_size, cfg.vocab_size, gather_output=True)` — args: `(hidden_size, vocab_size, gather_output=True)`.
  Code constructor: `ParallelLMHead(num_embeddings, embedding_dim, bias=False, tp_size=None)` — args: `(num_embeddings=vocab_size, embedding_dim=hidden_size)`.
  The positional argument order is inverted between the blueprint class_hierarchy snippet and the actual code. Additionally, `gather_output=True` does not exist as a parameter in the code (the code always gathers unconditionally). This is a Phase 7 (QwenForCausalLMTP assembly) concern — the Phase 4 contract (`tp_embedding_and_lm_head`) does not specify constructor signatures. Phase 7 implementer must reconcile: either align the blueprint or use keyword arguments (`ParallelLMHead(num_embeddings=cfg.vocab_size, embedding_dim=cfg.hidden_size)`).

---

## Conclusion

Spec审查通过。代码与蓝图 `tp_embedding_and_lm_head` 契约及 `phase_4_tp_embedding` 实现清单一致。可移交 verification。
