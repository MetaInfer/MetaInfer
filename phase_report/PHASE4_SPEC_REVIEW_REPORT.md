# PHASE4_SPEC_REVIEW_REPORT.md

- **PID**: 912497
- **Role**: spec-reviewer
- **Timestamp**: 2026-05-30T00:00:00Z
- **Phase**: 4
- **Review Target**: `./engine/tp_layers/embedding.py`
- **Blueprint Contract Source**: `inference_blueprint.json` → `framework_layer.data_flow_contracts.tp_layer_interface_contracts.tp_embedding_and_lm_head`

---

## Spec Compliance: ✅ PASS

---

## Evidence Chain (逐条核验)

### vocab_parallel_embedding — Forward Pseudocode (5 步)

- [x] **Step 1: mask creation** — `mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)`
  - ✅ @ `embedding.py:73` — 与蓝图逐字一致，operator、operand 完全匹配

- [x] **Step 2: local_ids with masked_fill** — `local_ids = (input_ids - self.vocab_start).masked_fill(~mask, 0)`
  - ✅ @ `embedding.py:76` — 与蓝图逐字一致

- [x] **Step 3: F.embedding lookup** — `out = F.embedding(local_ids, self.weight)  # [B, T, embedding_dim]`
  - ✅ @ `embedding.py:79` — 使用 `F.embedding`（非 `nn.Embedding` 的 forward wrapper），shape 注释 `[B, T, embedding_dim]` 与蓝图一致

- [x] **Step 4: masked_fill zero-out** — `out = out.masked_fill((~mask).unsqueeze(-1), 0)`
  - ✅ @ `embedding.py:82` — `.unsqueeze(-1)` 将 [B,T] 的 ~mask 扩展为 [B,T,1] 以广播到 [B,T,embedding_dim]，与蓝图逐字一致

- [x] **Step 5: all_reduce_sum** — `return all_reduce_sum(out)`
  - ✅ @ `embedding.py:85` — 调用 `all_reduce_sum`，其内部实现（`distributed.py:107-130`）按蓝图 contract：CustomAR P2P > NCCL fallback > TP=1 返回 clone()

### vocab_parallel_embedding — Vocab Partition Math

- [x] **vocab_start 公式**: `tp_rank * (vocab_size // tp_size)`
  - ✅ @ `embedding.py:59` — `self.tp_rank * self.local_vocab_size`，其中 `self.local_vocab_size = num_embeddings // self.tp_size`（line 58）
  - ✅ @ `embedding.py:60` — `self.vocab_end = self.vocab_start + self.local_vocab_size`，与蓝图注释 `vocab_end = vocab_start + local_vocab_size` 一致

### vocab_parallel_embedding — Shape/Dtype Contract

- [x] **input_ids**: `[B, T] int64`
  - ✅ @ `embedding.py:64` docstring 声明 `[B, T] int64 — token ids in range [0, num_embeddings)`，PyTorch embedding 自动接受 int64/long

- [x] **local_weight**: `[vocab_size/tp, hidden_size]`
  - ✅ @ `embedding.py:62` — `nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim))`，local_vocab_size = num_embeddings // tp_size

- [x] **local_embedding**: `[B, T, hidden_size] (masked local vocab)`
  - ✅ @ `embedding.py:79` — `F.embedding(local_ids, self.weight)` 输出 [B, T, embedding_dim]，蓝色注释 `[B, T, embedding_dim]`

- [x] **output_after_all_reduce**: `[B, T, hidden_size]`
  - ✅ @ `embedding.py:85` — `all_reduce_sum` 不改变 shape，输出 [B, T, embedding_dim]

### parallel_lm_head — Forward Pseudocode (2 步)

- [x] **Step 1: local_logits** — `local_logits = F.linear(hidden_states, self.weight)  # [B, T, vocab_size/tp]`
  - ✅ @ `embedding.py:155` — 使用 `F.linear(hidden_states, self.weight, self.bias)`，当 `bias=False`（默认）时 `self.bias is None`，`F.linear` 行为与无 bias 一致。shape 注释 `[B, T, local_vocab_size]` = `[B, T, vocab_size/tp]`

- [x] **Step 2: all_gather_last_dim** — `logits = all_gather_last_dim(local_logits)  # [B, T, vocab_size]`
  - ✅ @ `embedding.py:158` — 调用 `all_gather_last_dim(local_logits)`，其内部实现（`distributed.py:143-161`）使用 `dist.all_gather(outs, x)` + `torch.cat(outs, dim=-1)`，**符合编码铁律禁止 `all_gather_into_tensor`** 的要求

### parallel_lm_head — Shape/Dtype Contract

- [x] **input_hidden**: `[B, T, hidden_size]`
  - ✅ @ `embedding.py:148` docstring `[B, T, embedding_dim]`

- [x] **local_logits**: `[B, T, vocab_size/tp]`
  - ✅ @ `embedding.py:155` — weight shape `[local_vocab_size, embedding_dim]` → F.linear 输出 `[B, T, local_vocab_size]`

- [x] **output_logits_gather**: `[B, T, vocab_size]`
  - ✅ @ `embedding.py:158` — `all_gather_last_dim` 沿 dim=-1 拼接 tp_size 份，local_vocab_size × tp_size = num_embeddings

### Class Names & Attribute Names

- [x] **VocabParallelEmbedding** → ✅ @ `embedding.py:28` — 类名与蓝图 `class_hierarchy` 中 `QwenForCausalLMTP.attrs` 的 `self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size)` 一致

- [x] **ParallelLMHead** → ✅ @ `embedding.py:109` — 类名与蓝图 `model_forward_pseudocode` 中 `self.lm_head(hidden_states)` 调用一致

### Encoding Iron Laws (AGENT_SKILL.md §1)

- [x] **all_gather_last_dim = dist.all_gather + torch.cat**（非 all_gather_into_tensor）
  - ✅ @ `distributed.py:159-161` — `dist.all_gather(outs, x)` + `torch.cat(outs, dim=-1)`

- [x] **维度值来自参数动态读取**（禁止硬编码）
  - ✅ `num_embeddings`、`embedding_dim`、`tp_size` 均为 `__init__` 参数，未硬编码任何数值

- [x] **all_reduce_sum 实现 CustomAR P2P > NCCL fallback**
  - ✅ @ `distributed.py:107-130` — 三层 fallback: CustomAR → NCCL → clone()

---

## Minor Observations (不阻塞 PASS)

These are not violations of the Phase 4 forward contract but are noted for completeness:

1. **`padding_idx` parameter** @ `embedding.py:50` — VocabParallelEmbedding 接受 `padding_idx` 参数但未使用。蓝图 `tp_embedding_and_lm_head.vocab_parallel_embedding` 未定义此参数。实现以 API 兼容性接受（docstring 声明 "accepted for API compatibility"），不影响 forward contract。

2. **`bias` parameter in ParallelLMHead** @ `embedding.py:131` — `bias=False` 默认值确保与蓝图伪代码行为一致。`bias=True` 时 `F.linear(hidden_states, self.weight, self.bias)` 会额外加 bias，此路径不属于蓝图定义范围但也不违反蓝图（蓝图未禁止 bias）。

3. **`load_weight_shard` methods** @ `embedding.py:87-101,161-175` — 此方法属于 Phase 7（权重加载）范畴，在 Phase 4 中出现属于提前脚手架。含有 double_shard_guard 模式（检查 shape 是否预切片 → 直接 copy_ 或从完整权重切片）。此方法不参与 forward 数据流，不影响 Phase 4 验收。

---

## Verdict

**Spec 审查通过，代码与蓝图契约一致，可移交 verification。**

所有蓝图 `tp_embedding_and_lm_head` 中定义的 forward 步骤、shape、dtype、vocab partition math、collective 选择均被精确实现。未发现任何契约偏差。
