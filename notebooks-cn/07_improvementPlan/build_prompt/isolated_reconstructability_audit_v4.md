# 蓝图重构完备性漏洞报告（V4 更新后审计）

**审计对象**：`inference_blueprint.json` (v2.3.0, 更新后) + `AGENT_SKILL.md` (更新后)
**审计身份**：推理引擎独立闭环审计官（Isolated Reconstructability Auditor）
**审计前提**：假设 `engine/` 目录源码被物理擦除，Agent 仅持有上述两份文件
**审计范围**：调度层 + Qwen3-8B TP=4（排除 `deepseek_v2_v3_mla_moe` 节点）
**审计日期**：2026-05-26
**上次审计**：V4 初版（同目录 `isolated_reconstructability_audit.md`）— 综合完备度 48%

---

## 0. 与上次审计的差异摘要

上次审计（V4 初版）报告了 **6 个阻断项 + 6 个警告项**。经审阅更新后的图谱，**全部 12 项已修复**：

| 旧编号 | 描述 | 修复节点 | 状态 |
|--------|------|---------|------|
| GAP-1 | Scheduler 双轨 block_size 选择 | 新增 `scheduler_tp_runner_bridge.llm_engine_block_size_injection` | ✅ |
| GAP-2 | num_free_blocks 数据来源 | 新增 `scheduler_tp_runner_bridge.num_free_blocks_source` | ✅ |
| GAP-3 | BlockManager 与 prefix caching 冲突 | 新增 `scheduler_tp_runner_bridge.block_manager_role_in_tp_path` + `prefix_caching_in_tp_path` + `interface_boundary` | ✅ |
| GAP-4 | KV cache index_copy_ 重塑链 | 新增 `paged_kv_cache_contract.prefill_kv_write.full_reshape_chain` | ✅ |
| GAP-5 | CustomAR init 状态机 | 新增 `custom_ar_all_reduce.constraint.init_state_machine` (7 步伪代码 + workspace 公式) | ✅ |
| GAP-6 | CUDA Graph TP=4 split_graph | 新增 `all_reduce_sum_custom_op_template.split_graph_implementation` (3 steps + SGLangBackend 伪代码) | ✅ |
| WARN-1 | Python hash 跨进程一致性 | 新增 `compute_hash.hash_policy` (单进程假设 + hashlib 备选) | ✅ |
| WARN-2 | fused_add_rms_norm 最后一层 | 新增 `fused_add_rms_norm.constraint.last_layer_note` | ✅ |
| WARN-3 | prefill/decode 混批竞态 | 新增 `schedule_algorithm.state_transition_safety` | ✅ |
| WARN-4 | torch.compile bound method | 修正 `swap_mechanism.implementation` 为 `types.MethodType` | ✅ |
| WARN-5 | flash_attn dynamic num_blocks | 新增 `custom_op_registration.complete_template.dynamic_shape_note` | ✅ |
| WARN-6 | QKV num_heads 歧义 | 补充 `qkv_merged_projection.constraint`: "num_heads = config.num_attention_heads (全量值)" | ✅ |

另外，`AGENT_SKILL.md` §0 新增集成桥接接口说明，§7.4.G 新增 7 条"集成接口陷阱"。

---

## 1. 🔴 可重构性阻断项（Reconstructability Gaps）

更新后的图谱已消除系统性架构断层。以下为剩余阻断项，均为**细节缺失**而非架构缺陷。

---

### GAP-1（新）：TP Runner `get_num_free_blocks` 公式中 `kv_len` 来源未指定

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner`
- **严重等级**：🔴 阻断

**问题**：`scheduler_tp_runner_bridge` 定义了 TP Runner 的 `get_num_free_blocks()` 公式为 `(max_model_len - kv_len) // 256`，但未指定 `kv_len` 从哪个 layer 获取。当前描述暗示从 `_kv_len_gpu` 推算，但 `_kv_len_gpu` 是 per-layer 的 tensor——虽然所有层的值在 decode 步开始时相同，Agent 必须知道取任意层（如 `self.layers[0].self_attn._kv_len_gpu[0].item()`）均可。此外，prefill 阶段 prefill 完成后 `_kv_len_gpu` 才更新（见 `prefill_kv_len_semantics`），因此 prefill 调度时 `kv_len` 仍为旧值——Agent 需要在 scheduler 中独立跟踪 `total_allocated_tokens` 而非从 `_kv_len_gpu` 读取。

**必须补充的技术规格契约**：

```
QwenTPModelRunner.get_num_free_blocks() 实现:
  def get_num_free_blocks(self):
      allocated = self.layers[0].self_attn._kv_len_gpu[0].item()  # 所有层 kv_len 同步
      max_tokens = self.config.max_position_embeddings  # 32768
      total_blocks = max_tokens // self._kv_block_size  # 128
      allocated_blocks = (allocated + self._kv_block_size - 1) // self._kv_block_size
      return total_blocks - allocated_blocks

注意: prefill 调度时 _kv_len_gpu 尚未更新 (prefill 写入在 forward 内部完成)，
因此 get_num_free_blocks() 在 prefill forward 前返回的是基于旧 kv_len 的容量，
调度器必须在 prefill 完成后调用 postprocess 更新内部跟踪。
```

---

### GAP-2（新）：Prefill batch 中多序列的独立 block_table 管理未说明

- **JSON Path**：`framework_layer.data_flow_contracts.paged_kv_cache_contract.prefill_kv_write.slot_mapping_algorithm`
- **严重等级**：🔴 High

**问题**：当前 `slot_mapping_algorithm.pseudocode` 和 `full_reshape_chain.pseudocode` 假设单序列的 prefill — 使用 `torch.arange(num_blocks)` 分配 block_table。但当 batch 中有多个序列（如 sequence A 有 5 个 block, sequence B 有 3 个 block），每个序列需要独立的 block_table 和独立的 slot_mapping。`cu_seqlens_q` 能处理 ragged 拼接的 attention，但 **KV cache 写入时，sequence A 和 B 的 K/V 被 cat 成 `[total_tokens, heads, dim]` 的单一 ragged tensor，然后通过 slot_mapping 写回各自独立的 block_table 区域**。当前伪代码未展示多序列时 block_table 的拼接和 slot_mapping 的偏移计算。

**必须补充的技术规格契约**：

```python
# 多序列 prefill batch 的 block_table + slot_mapping 构造:
all_block_tables = []
slot_mapping = torch.zeros(total_tokens, dtype=torch.int64, device=device)
offset = 0
for seq in batch:
    num_blocks = (seq.seq_len() + 255) // 256
    bt = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)  # [1, num_blocks]
    padded_bt = F.pad(bt, (0, max_blocks - num_blocks))  # 填充到统一 max_blocks
    all_block_tables.append(padded_bt)  # [1, max_blocks]
    for i in range(seq.seq_len()):
        slot_mapping[offset + i] = bt[0, i // 256].item() * 256 + (i % 256)
    offset += seq.seq_len()
```

---

### GAP-3（新）：Sampler 在 TP 配置下的执行职责未定义

- **JSON Path**：`framework_layer.components[4]` (Sampler) — 缺失 TP 集成信息
- **严重等级**：🔴 High

**问题**：`ParallelLMHead` 输出的 `[B, T, vocab_size]` 经 all_gather 后在每个 rank 上是一致的。但 **sampling 是否需要所有 rank 都执行**？如果每个 rank 都独立调用 `sample_next_tokens(logits[:, -1, :], temperature, top_p)`，由于 `torch.multinomial` 的随机种子在 rank 间不同，各 rank 会产出不同的 token——这会导致 batch 中的 KV cache 和后续解码在 rank 间不一致，最终 NCCL 通信时 shape 或内容不匹配。解决方案通常是指定 rank=0 执行采样后广播结果。

**必须补充的技术规格契约**：

```
LLMEngine.step() 中:
  tokens = runner.run(batch, is_prefill, temperature, top_p)
  # runner.run() 内部: 只在 rank 0 执行 sampling，然后 broadcast 结果给所有 rank:
  if dist.is_initialized():
      token_list = [torch.tensor(t, device=device) for t in tokens] if rank == 0 else [torch.zeros(1, dtype=torch.long, device=device) for _ in tokens]
      for t in token_list:
          dist.broadcast(t, src=0)
      tokens = [t.item() for t in token_list]
```

---

## 2. 🟡 信息熵不足警告项（Ambiguous Descriptions）

---

### WARN-1（新）：TP Runner decode batch 的迭代方式模糊

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_to_runner.batch_assembly_contract.decode_single.note`
- **严重等级**：🟡 警告

**问题**：`decode_single.note` 说 "decode 每步只处理最新 token，batch 中多序列分别调用 forward_decode"。`_tp_runner_actual_path` 又说 "model.forward(...) → 逐层 layer.forward_decode(hidden_states, pos, kv_len, residual=residual)"。目前 B=1（单个 token），但 batch 中有多个序列时，是 Python for-loop 逐序列调用 `model.forward` 还是将多个序列的 token 拼接为 `[1, batch_size]` batch 处理？如果逐序列 for-loop，KV cache 写入各自独立（没有问题但性能低）。如果 batch 处理，multiple sequences 共享同一个 block_table 形状 `[batch, max_blocks]`，但每个序列的 `_block_table` 和 `_kv_len_gpu` 是独立的——batch 处理需要更复杂的索引逻辑。

**风险**：Agent 可能在 decode 时尝试 batch 处理（将多个 [1,1] token stack 为 [B,1]），但未意识到各序列的 `_kv_len_gpu` 和 `_block_table` 需要独立索引，导致 KV cache 写入错乱。

---

### WARN-2（新）：Scheduler 超长序列的死锁问题

- **JSON Path**：`framework_layer.data_flow_contracts.scheduler_to_runner.schedule_algorithm.empty_return`
- **严重等级**：🟡 警告

**问题**：`empty_return` 定义返回 `([], False)` 时调用方"忙等重试或检查 all_finished 退出"。但如果一个序列的 `required_blocks > total_blocks`（例如 prompt 远超 `max_position_embeddings`），该序列永远无法被调度，而 Scheduler 会一直返回空 batch（因为有一条 WAITING 序列卡住了队列，即使总有 decode 序列可运行）。如果 Agent 以 `all_finished` 为退出条件，这个序列永远不会 finish，导致死循环。图谱未定义超时或 UNRUNNABLE 拒绝机制。

**风险**：Agent 实现的 LLMEngine step-loop 在遇到超长 prompt 时无限循环。

---

### WARN-3（新）：Prefill → Decode 状态转换时 Forward 失败的回滚机制缺失

- **JSON Path**：`framework_layer.data_flow_contracts.request_level.status_transitions`
- **严重等级**：🟡 警告

**问题**：序列状态转换 `RUNNING_PREFILL → RUNNING_DECODE` 发生在 `postprocess` 中。但如果在 `runner.run(batch, is_prefill=True)` 期间发生异常（如 OOM、CUDA error），已部分写入的 KV cache 和 block_table 处于不一致状态。当前图谱没有定义任何异常回滚路径（如将序列回退到 WAITING 并释放已分配 block）。

**风险**：Agent 实现的系统在显存紧张时可能因一次 prefill 失败导致 KV cache 泄漏，后续所有调度均因 `num_free_blocks` 错误而失败。

---

### WARN-4（新）：`init_state_machine` workspace_size 计算基于全量 hidden_size

- **JSON Path**：`framework_layer.data_flow_contracts.tp_layer_interface_contracts.qwen3_kernel_contracts.custom_ar_all_reduce.constraint.init_state_machine.workspace_size`
- **严重等级**：🟡 Low

**问题**：workspace_size 注释说 "hidden_size=4096 × bf16 = 8KB per tensor"，但 TP=4 下每 rank 的 hidden 维度是 1024，RowParallelLinear 输出 2KB。逻辑上 16MB 仍然足够（36 layers × 2 RowParallel × 2KB = 144KB），但 Agent 可能被误导去分配更大的 buffer。非功能性缺陷。

**风险**：低。Agent 可能分配过大 buffer 浪费少许显存。

---

### WARN-5（重新开放）：Prefill ragged batch 中 `cu_seqlens_k` 在 paged KV 场景下的语义歧义

- **JSON Path**：`framework_layer.data_flow_contracts.flash_attention_integration_contract.prefill_path.cu_seqlens_construction`
- **严重等级**：🟡 警告

**问题**：`cu_seqlens_k` 定义为 `cu_seqlens_q.clone()`，意味着 K 的有效长度等于 Q 的有效长度。这适用于 prefill 之后（所有 KV 已写入 cache）。但 prefill 调用 `flash_attn_varlen_func` 时，K 和 V 的来源是什么？是从 paged KV cache 中读取（K/V 在 `index_copy_` 之后），还是直接从投影产出？如果从 cache 中读取，K 的实际长度应该是 `kv_len_gpu`（此时为 0 或旧值），而不是 `seq.seq_len()`。当前描述假设 prefill 时 Q 和 K 来自相同的 token 序列（即标准的 causal self-attention prefill），这在写入 KV cache 之前是正确的（使用刚投影的 K，不是从 cache 读取的）。但这一点未明确说明。

**风险**：Agent 可能在 prefill attention 时错误地从 KV cache 读取（此时 cache 中可能还保留着旧序列的残留数据），而不是使用刚投影产出的 K 和 V。

---

## 3. 🟢 最终图纸判决（Verdict）

### 量化结论

| 维度 | 上次完备度 (V4 初版) | 本次完备度 (更新后) | 变化 |
|------|---------------------|-------------------|------|
| 框架调度层（Scheduler/BlockManager/KVMemoryPool） | 40% | **75%** | +35% |
| 框架层增量修改（P0/P2/P3-FA） | 65% | **85%** | +20% |
| Kernel 层 7 大标品（数据契约 + 残差链 + IPC） | 55% | **85%** | +30% |
| CUDA Graph 静态执行契约 | 30% | **65%**（TP=4 split_graph 有伪代码但仍待验证） | +35% |
| **加权综合完备度** | **48%** | **78%** | **+30%** |

### 判决

**当前图谱已具备有限条件下开工的能力，但需 Agent 具备中级推理引擎开发经验。** 

上次审计的核心问题——Scheduler 与 TP Runner 之间的架构断层——已通过新增 `scheduler_tp_runner_bridge` 节点完整修复。所有 6 个 GAP 和 6 个 WARN 均已闭合。

剩余 3 个阻断项（GAP-1: `kv_len` 来源、GAP-2: 多序列 block_table、GAP-3: Sampler TP 集成）均为局部细节缺失，Agent 可以通过合理的工程推断解决（但不保证首次正确）。修复这 3 项后，完备度可提升至 ~88%。

### 与上次判决的关键差异

| 维度 | V4 初版 | 更新后 |
|------|---------|--------|
| 架构完整性 | Scheduler ↔ TP Runner 之间存在断层 | `scheduler_tp_runner_bridge` 定义了完整桥接接口 |
| KV cache 重塑链 | 只有 `index_copy_` 一行 | `full_reshape_chain` 提供完整伪代码 |
| CustomAR 初始化 | 只有散落的约束 | `init_state_machine` 7 步状态机 + barrier + workspace |
| 边界情况 | 6 个信息熵警告 | last_layer, hash, batch_safety, MethodType, dynamic_shape 全部闭合 |
| AGENT_SKILL 配合 | 仅引用 JSON 节点 | §7.4.G 新增 7 条集成接口陷阱，与 JSON 互为补充 |

### 必须修复的 Top 3 阻断项（更新后）

1. **GAP-1（kv_len 来源 + prefill 调度同步）** — 在 `scheduler_tp_runner_bridge.num_free_blocks_source.TP_Runner` 中补充 `get_num_free_blocks()` 的完整实现伪代码，明确 kv_len 从 layer[0] 读取，并说明 prefill 调度与 kv_len 更新之间的时序差。

2. **GAP-2（多序列 prefill block_table 管理）** — 在 `paged_kv_cache_contract.prefill_kv_write` 中补充多序列 batch 场景下的 block_table 拼接和 slot_mapping 偏移计算伪代码。

3. **GAP-3（Sampler TP 集成）** — 在 Sampler 组件或 `scheduler_tp_runner_bridge` 中补充 TP 配置下 rank-0-only 采样 + broadcast 的职责分配。

---

**总结**：图谱从上次审计的"不可开工"（48%）升级为"有限条件下可开工"（78%）。更新有效修复了上次审计识别的全部系统性架构缺陷。当前 Agent 仅依赖这两份文件，**可以产出结构正确的 Qwen3-8B TP=4 推理引擎框架代码**，但在多序列 prefill batch 处理、TP 采样广播、以及 prefill 调度同步三个细节上需要自行推断（大概率在集成测试中发现并修正）。补充上述 3 个阻断项后，图谱将具备生产级可重构性（~88%）。
