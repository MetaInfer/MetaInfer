# 蓝图重构完备性漏洞报告

**审计对象**：`inference_blueprint.json` + `AGENT_SKILL.md`
**审计身份**：推理引擎独立闭环审计官（Isolated Reconstructability Auditor）
**审计前提**：假设 `engine/` 目录源码被物理擦除，Agent 仅持有上述两份文件
**审计范围**：调度层 + Qwen3-8B TP=4（排除 `deepseek_v2_v3_mla_moe` 节点）
**审计日期**：2026-05-26

---

## 1. 🔴 可重构性阻断项（Reconstructability Gaps）

以下空洞规约将导致全新 Agent 必定或极大概率写出崩溃/死锁代码。

---

### GAP-1：Scheduler 双轨 block_size 选择机制缺失

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_to_runner.max_num_batched_tokens.block_size_selection`
- **严重等级**：🔴 阻断

**问题**：`max_num_batched_tokens.formula = max(1, num_free_blocks * block_size)` 中的 `block_size` 在 RealModelRunner(16) 与 TP_Runner(256) 之间不同，但 `scheduler_to_runner` 节点内部没有定义 Scheduler 如何根据当前 backend 选择 block_size。图谱分别在以下位置提及双轨制：

- `components.Scheduler._dual_track_note`："LLMEngine block_size=16 仅对 RealModelRunner 有效。TP Runner 硬编码 _kv_block_size=256。"
- `scheduler_tp_runner_bridge.llm_engine_block_size_injection`："LLMEngine.__init__ 中根据 self.inference_backend 向 Scheduler 注入 block_size"（给出了伪代码）

**但存在问题**：`block_size_selection` 节点位于 `scheduler_to_runner` 内部，而注入逻辑在 `scheduler_tp_runner_bridge` 中，两处信息分离。`scheduler_to_runner` 未引用 `scheduler_tp_runner_bridge`。Agent 若从 `scheduler_to_runner` 开始编码（自然的搜索路径），看到的只有 `block_size_selection.RealModelRunner: 16, TP_Runner: 256`，但不知**如何将 TP_Runner 这个 key 映射到具体的 `inference_backend` 字符串**（`'qwen_tp'` / `'deepseek_tp'`），且 `llm_engine_block_size_injection` 的伪代码只列了 `inference_backend in (None, 'hf')` 和 `inference_backend in ('qwen_tp', 'deepseek_tp')`，与 `block_size_selection` 中的 `RealModelRunner` / `TP_Runner` 命名不一致。

**必须补充的技术规格契约**：在 `scheduler_to_runner.max_num_batched_tokens` 中增加对 `scheduler_tp_runner_bridge` 的显式引用，或统一两处的命名空间。

---

### GAP-2：TP Runner 路径 `num_free_blocks` 的获取公式未闭合

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner`
- **严重等级**：🔴 阻断

**问题**：`num_free_blocks_source` 给出公式 `runner.get_num_free_blocks() -> int  # max_blocks - current_allocated (从 _kv_len_gpu 推算: (max_model_len - kv_len) // 256)`。存在三个细节缺失：

1. **`kv_len` 从哪层获取**：`_kv_len_gpu` 是 per-layer 的 tensor（36 层各一个），Agent 需要知道取任意一层（如 `self.layers[0].self_attn._kv_len_gpu[0].item()`）即可，因为所有层在 decode 步开始时值相同。但图谱未说明。
2. **prefill 调度时的时序差**：`prefill_kv_len_semantics` 说明 "prefill 前 `_kv_len_gpu=0`；写入后 = prefill_token_count"。Scheduler 在 prefill 前调用 `runner.get_num_free_blocks()` 时，`_kv_len_gpu` 还是旧值（未反映即将进行的 prefill 分配），导致 `num_free_blocks` 高估。图谱未说明 Scheduler 需要独立跟踪 `total_allocated_tokens` 而非依赖 runner 的实时查询。
3. **整数除法的取整方向**：`(max_model_len - kv_len) // 256` 在 kv_len 不是 256 整数倍时，会少算一个正在部分使用的 block。实际公式应为 `max_blocks - ceil(kv_len / 256)`。

**必须补充的技术规格契约**：

```
QwenTPModelRunner.get_num_free_blocks() 实现:
    # kv_len 从任意层读取，所有层同步
    kv_len = self.layers[0].self_attn._kv_len_gpu[0].item()
    max_blocks = self.config.max_position_embeddings // 256  # 128 for Qwen3-8B
    allocated_blocks = (kv_len + 255) // 256  # ceil division
    return max_blocks - allocated_blocks

LLMEngine 侧: Scheduler 维护独立的 _reserved_blocks 计数器。
prefill 前更新 _reserved_blocks → schedule() 使用 max(0, max_blocks - _reserved_blocks)
prefill 完成后由 postprocess 确认实际分配量。
```

---

### GAP-3：多序列 Prefill batch 的 block_table 拼接逻辑缺失

- **JSON Path**：`framework_layer.data_flow_contracts.paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm`
- **严重等级**：🔴 阻断

**问题**：`slot_mapping_algorithm.pseudocode` 和 `full_reshape_chain.pseudocode` 都假设单序列 prefill——`block_table[0,:num_blocks] = torch.arange(num_blocks)`，`slot_mapping[i]` 直接从 0 开始分配。但 `batch_assembly_contract.prefill_ragged` 明确说 batch 使用 ragged concatenation（`torch.cat([seq.input_ids_tensor() for seq in batch], dim=1)`），这意味着**多个序列的 token 被拼接成一个 ragged tensor**。

当 batch 中有 2 个序列（A: 800 tokens = ceil(800/256)=4 blocks, B: 300 tokens = ceil(300/256)=2 blocks），所有 token 被 cat 为 `[1, 1100]`。`slot_mapping` 需要为序列 A 分配 4 个 block（slot 0-799），为序列 B 分配 2 个 block（slot 800-1099），且 B 的 block 编号应接着 A 的最大 block 编号。当前伪代码的 `torch.arange(num_blocks)` 只适用于单序列。

**必须补充的技术规格契约**：

```python
# 多序列 prefill batch 的 block_table + slot_mapping 构造:
slot_mapping = torch.zeros(total_tokens, dtype=torch.int64, device=device)
offset = 0
next_block_id = 0
for seq in batch:
    n = seq.seq_len()
    nb = (n + 255) // 256
    blk = torch.arange(next_block_id, next_block_id + nb, dtype=torch.int32, device=device)
    bt_padded = F.pad(blk.unsqueeze(0), (0, max_blocks - nb))
    seq.block_table = bt_padded
    for i in range(n):
        slot_mapping[offset + i] = blk[i // 256].item() * 256 + (i % 256)
    offset += n
    next_block_id += nb
```

---

### GAP-4：Sampler 在 TP 配置下的 Rank 安全执行未定义

- **JSON Path**：`framework_layer.components[4]` (Sampler) — 无 TP 相关字段
- **严重等级**：🔴 阻断

**问题**：Sampler 组件描述仅提及 `temperature/top_p`，无 TP 相关约束。但 `ParallelLMHead` 的 all_gather 在每个 rank 上都产生完整 logits `[B, T, vocab_size]`。如果每个 rank 独立调用 `torch.multinomial`，由于各 rank 的 CUDA 随机种子不同，会产出不同的 token。这些 token 参与 KV cache 写入和后续 decode，导致 rank 间 `_kv_len_gpu` 值和 KV cache 内容不一致，NCCL 集体操作时产生不确定的错误或 shape mismatch。

正确行为是 **rank 0 执行所有采样，然后广播 token 给所有 rank**，但 Agent 无法从图谱中获知这一点——Sampler 组件只说 "将 logits 转换为 token ids"，完全没有 TP 上下文。

**必须补充的技术规格契约**：

```
LLMEngine.step() 中 runner.run() 内部:
  # 只在 rank 0 采样:
  if dist.get_rank() == 0:
      tokens = [sample_next_tokens(logits[i, -1, :], temperature, top_p) for i in range(batch_size)]
  else:
      tokens = [0] * batch_size
  # 广播给所有 rank:
  if dist.is_initialized() and dist.get_world_size() > 1:
      token_tensors = [torch.tensor([t], dtype=torch.long, device=device) for t in tokens]
      for tt in token_tensors:
          dist.broadcast(tt, src=0)
      tokens = [tt.item() for tt in token_tensors]
```

---

### GAP-5：Prefill 调度时 Scheduler 内部 `_reserved_blocks` 与 Runner 实际分配之间的同步缺口

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm`
- **严重等级**：🔴 阻断（多步循环场景）

**问题**：Scheduler 在多步 decode 循环中需要知道当前还有多少可用 block。图谱给出了两种数据来源：

- `BlockManager.get_num_free_blocks()`（HF 路径）
- `runner.get_num_free_blocks()`（TP 路径）

但 **prefill 调度发生在 runner forward 之前**——此时 `_kv_len_gpu` 仍为旧值。如果 Scheduler 在调度 prefill batch 时从 runner 查询 `get_num_free_blocks()`，得到的是**上一步结束后的可用 block 数**，但 Scheduler 在这一步中可能可能连续调度了多个 prefill 序列——每个序列的 `can_allocate` 判定会累加消耗。Scheduler 必须在内部独立维护一个**当前 step 内的 block 消耗计数器**，而不是反复查询 runner（runner 的计数器不会在 forward 中途更新）。

图谱中的 `schedule_algorithm` 未描述 Scheduler 内部是否需要维护 `_step_reserved_blocks` 变量。

**必须补充的技术规格契约**：

```python
# Scheduler.schedule() 伪代码:
def schedule(self, num_free_blocks):
    reserved = 0
    batch = []
    for seq in self.waiting:
        req = ceil(len(seq.input_ids) / self._block_size)
        if reserved + req <= num_free_blocks and self._check_max_tokens(batch, seq):
            batch.append(seq)
            reserved += req
    return batch, True if batch else self._select_decode(num_free_blocks - reserved)
```

---

### GAP-6：CUDA Graph TP=4 状态仍为“阻塞”，split_graph 方案缺少可执行验证路径

- **JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.cuda_graph_execution_contract.current_status.tp4`
- **严重等级**：🔴 阻断（如 Agent 的目标包含 TP=4 CUDA Graph）

**问题**：`current_status.tp4` 明确标注为"阻塞"。`split_graph_implementation` 提供了 SGLang 方案的三步骤和伪代码，但缺少以下关键信息使 Agent 无法独立实现：

1. **无 `split_graph` 函数的具体位置**：只说 `ref_projects/sglang/python/sglang/srt/compilation/backend.py:214-257`，但 Agent 无法访问外部 ref_projects（按审计假设）。
2. **无具体的 `torch.compile` backend 注册代码**：`SGLangBackend.compile()` 的伪代码展示了逻辑，但 `custom_backend` 如何注册到 `torch.compile`（`torch.compile(backend='sglang_custom')`）没有说明。
3. **`all_reduce_sum` custom_op 注册后，正常 eager 模式和 split_graph 编译模式如何切换**：环境变量 `CUDA_GRAPH=1` 触发 swap_mechanism，但 split_graph 是另一个维度——两者如何组合？

**必须补充的技术规格契约**：将 `split_graph_implementation` 扩展为可独立编码的完整模板，包括 backend 注册、与 `swap_mechanism` 的组合逻辑、以及单 GPU 验证到 TP=4 的渐进测试路径。

---

## 2. 🟡 信息熵不足警告项（Ambiguous Descriptions）

---

### WARN-1：TP Runner decode batch 的迭代方式存在歧义

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_to_runner.batch_assembly_contract.decode_single.note` + `components[3].tp_runner_actual_flow.decode`
- **严重等级**：🟡 警告

**问题**：`decode_single.note` 说 "batch 中多序列分别调用 forward_decode"，暗示 per-sequence for-loop。但 `tp_runner_actual_flow.decode` 说 "模型 forward(input_ids, past_key_values=kv_lens) → 逐层 layer.forward_decode(hidden_states, pos, kv_len, residual=residual)"，其中 `input_ids` 可能是 `[B, 1]` 的 batch tensor。两种描述不一致。如果采用 batch 处理（`[B, 1, hidden_size]`），则 flash_attn_with_kvcache 的 `_block_table` 和 `_kv_len_gpu` 需要从 `[1, max_blocks]` 和 `[1]` 扩展到 `[B, max_blocks]` 和 `[B]`——当前所有 KV cache 形状描述都是单序列的。

**风险**：Agent 必须在 for-loop 和 batch 间做选择。for-loop 正确但慢；batch 处理需要改动所有 attention 层的数据结构。

---

### WARN-2：Scheduler 超长序列无拒绝机制导致死循环

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm.empty_return`
- **严重等级**：🟡 警告

**问题**：`empty_return = "([], False) — 调用方忙等重试或检查 all_finished 退出"`。如果用户提交一个 `input_ids` 远超 `max_position_embeddings` 的 prompt（如 100K tokens），`required_blocks > max_blocks` 恒成立，`can_allocate` 永远返回 False。该序列永远停留在 WAITING 状态，`all_finished` 永远为 False，LLMEngine 的 step-loop 进入无限循环。

图谱未定义 `UNRUNNABLE` 状态或超长序列的拒绝/截断逻辑。

**风险**：Agent 的实现在遇到超长 prompt 时死循环。

---

### WARN-3：Prefill 异常时 KV cache 回滚机制缺失

- **JSON Path**：`framework_layer.data_flow_contracts.request_level.status_transitions`
- **严重等级**：🟡 警告

**问题**：`status_transitions` 只定义了正向三条路径（WAITING→PREFILL→DECODE→FINISHED），无任何回退边。如果 `runner.run(batch, is_prefill=True)` 期间发生 CUDA OOM、NCCL 超时或 NaN，已部分写入的 KV cache（`index_copy_` 已完成的部分 token）和 block_table 条目需要释放。否则后续调度中的 `num_free_blocks` 将永久减少，最终资源耗尽。

图谱未定义 prefill 失败时的清理协议。

**风险**：显存紧张时单次 prefill 失败导致 KV cache 泄漏，后续所有请求饿死。

---

### WARN-4：`fused_add_rms_norm` 跨层 weight 传递的物理访问路径不清晰

- **JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.fused_add_rms_norm.constraint`
- **严重等级**：🟡 警告

**问题**：`residual_chain_pseudocode` 和 `last_layer_note` 已经说明了语义上的正确行为——最后一层 post_mlp 不调用 `fused_add_rms_norm`，由外部 `model.norm(hidden_states, residual)` 处理。但 `forward_decode` 的统一签名中 `residual_chain_pseudocode` 显示 `post_attention_layernorm` 的 `fused_add_rms_norm(attn_out, res, post_attn_norm_weight, eps)` 使用了本层的 `post_attn_norm_weight`，而 `post_mlp` 的 `fused_add_rms_norm` 按照 vLLM 语义应使用**下一层**的 `input_layernorm.weight`。然而 `forward_decode` 方法定义在 `QwenDecoderLayerTP`（单层）内部，无法访问 `self.next_layer.input_layernorm.weight`。

如果按 `residual_chain_pseudocode` 字面实现（post_mlp 的 fused_add_rms_norm 发生在当前层内），它只能访问本层的 `post_attn_norm_weight`（而非下一层 `input_layernorm.weight`）。vLLM 的实际做法是在**下一层**的 `input_layernorm` 处才做 `fused_add_rms_norm`（即 residual 传递到下一层，下一层入口处 fused），而不是当前层出口处。`residual_chain_pseudocode` 将两次 `fused_add_rms_norm` 都放在同一层内描述，与 vLLM 和实际物理布局不一致。

**风险**：Agent 可能按伪代码在当前层出口调用 `fused_add_rms_norm`，使用 `self.post_attention_layernorm.weight` 作为第二处 norm 的 weight，导致与 vLLM 语义差异（最后 35 层使用的是错误的 weight）。

---

### WARN-5：`init_state_machine` workspace_size 计算基于错误的 per-rank 维度

- **JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce.constraint.init_state_machine.workspace_size`
- **严重等级**：🟡 低

**问题**：workspace_size 注释 "hidden_size=4096 × bf16 = 8KB per tensor, 4 ranks = 32KB" 使用的是全量 hidden_size。但 RowParallelLinear 的 TP=4 输出是每 rank 的 partial output `[B, T, hidden_size]`（hidden_size=4096 是输出维度，RowParallel 不切分输出维），所以确实是 8KB。这不是错误，但注释中的 "per tensor" 可能被误解。此外 36 layers × 2 RowParallel × 8KB = 576KB，16MB 足够，计算偏差不影响功能。

**风险**：低。Agent 可能被误导分配更大 buffer，仅浪费显存。

---

### WARN-6：`flash_attn_varlen_func` prefill 中 K/V 来源未明确

- **JSON Path**：`framework_layer.data_flow_contracts.flash_attention_integration_contract.prefill_path`
- **严重等级**：🟡 警告

**问题**：prefill 路径描述为 `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)`，其中 `kv_source` 说明为 "paged KV cache 中已写入的 prefill 位置"。但 prefill attention 的 K 和 V 实际上来自**当前 projection 产出**（刚算出的 K、V），而非从 cache 中读取，因为：

1. 当前 prefill 的 K、V 要同时用于 attention 计算和写入 cache 供后续 decode 使用
2. 如果从 cache 读取（此时 cache 中相应位置还是 zeros 或上一条序列的残留数据），attention 结果错误

正确的顺序是：先做投影得到 K、V → 用 K、V 做 flash_attn_varlen_func → 再将 K、V 写入 cache。`kv_source` 的描述暗示 `index_copy_` 发生在 attention 之前，是顺序错误。

**风险**：Agent 按字面理解，可能先 `index_copy_` 写入 cache，再从 cache 读取做 attention，或直接从 cache 读取做 attention（忽略刚投影的 K、V）。

---

## 3. 🟢 最终图纸判决（Verdict）

### 量化结论

| 维度 | 完备度 | 关键缺口 |
|------|--------|---------|
| 框架调度层（Scheduler/BlockManager/KVMemoryPool） | 65% | GAP-1, GAP-2, GAP-3, GAP-5, WARN-2, WARN-3 |
| 框架层增量修改（P0/P2/P3-FA） | 80% | GAP-4, WARN-1, WARN-6 |
| Kernel 层 7 大标品（数据契约 + 残差链 + IPC） | 75% | WARN-4, WARN-5 |
| CUDA Graph 静态执行契约 | 55% | GAP-6（TP=4 路径仍标注阻塞） |
| **加权综合完备度** | **≈70%** | |

### 判决

**当前图谱可在有经验的推理引擎开发者监督下开工，但不适合零推理背景的 Agent 独立闭环。**

图谱的架构骨架已基本完整——`scheduler_tp_runner_bridge` 桥接层解决了调度层与 TP Runner 之间的主要断层，`init_state_machine` 提供了 CustomAR 的完整初始化序列，`full_reshape_chain` 补全了 KV cache 写入的重塑步骤。但仍有 6 个阻断项主要集中在"多序列 batch 场景下的细节契约"和"异常路径处理"两个维度。

### 必须修复的 Top 3 阻断项

1. **GAP-2（num_free_blocks 闭合）+ GAP-5（Scheduler _reserved_blocks）合并修复** — 明确 `get_num_free_blocks()` 的完整实现、prefill 调度前的时序同步机制、以及 Scheduler 内部的 `_step_reserved_blocks` 变量。

2. **GAP-3（多序列 prefill block_table）+ GAP-4（Sampler TP 广播）合并修复** — 补充多序列 prefill 的 block_table 拼接伪代码，以及在 Sampler 组件或 `scheduler_tp_runner_bridge` 中定义 TP 下 rank-0-only 采样 + broadcast 的职责分配。

3. **GAP-6（CUDA Graph TP=4 split_graph 可执行性）** — 将 `split_graph_implementation` 从"伪代码概要"升级为可编码的完整模板（backend 注册 + swap_mechanism 组合逻辑 + 单 GPU→TP=4 渐进验证路径），或明确标注为"下一 milestone 的独立设计任务，当前蓝图不覆盖"。

---

**总结**：图谱已逼近"可施工"水平。本轮的主要改进（`scheduler_tp_runner_bridge`、`init_state_machine`、`full_reshape_chain`、多边界情况覆盖）将架构完整性从前次的 ~48% 提升到 ~70%。剩余缺陷不再构成系统性的架构崩塌，而是体现在多序列并发调度、TP 采样的跨 rank 一致性、以及异常路径健壮性这三个"生产就绪"维度的缺失。补充上述 3 项后，完备度可提升至 ~85%，届时可交付有经验的 Agent 独立实现。
