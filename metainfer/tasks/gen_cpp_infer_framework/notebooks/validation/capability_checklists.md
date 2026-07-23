# 能力实现与证据检查表

本文件把冻结能力编译成最小实现和证据清单。Agent 只执行已选能力，但 Baseline 永远必需。

## 1. Baseline

### 实现

- [ ] CMake 构建真实 C++ Server 和 `qwen3_numeric_tests`；
- [ ] GGUF 使用对齐 data base 和相对 Tensor offset，所有 range 有界；
- [ ] Qwen3 Config、Tokenizer、Chat Template 来自真实文件；
- [ ] Prefill、Decode、RoPE、GQA、SwiGLU、LM Head、Sampling 可执行；
- [ ] 单序列 KV 的 logical length 只在成功 Step 后推进；
- [ ] `serve.sh PORT` 前台阻塞、继承 `MODEL_DIR`、无静默 Mock；
- [ ] OpenAI Chat Response 至少包含 `choices[0].message.content`。

### Numeric case ID

```text
cast_fp32_to_fp16
rms_norm
per_head_rms_norm
rope_neox
kv_write
prefill_gqa
swiglu
greedy
```

F16 增加 `f16_linear`。Q8_0 增加 `dequant_q8_0`、`q8_embedding`、`q8_linear`。
每个 case 必须运行真实 Kernel、读取结果并与独立 CPU Reference 比较。

### 行为证据

- [ ] Build/Server lifecycle；
- [ ] 真实 Model load 和 Tensor fingerprint；
- [ ] 有限且输入相关的 logits/output；
- [ ] Tokenizer/Chat Template；
- [ ] 多个 OpenAI 请求和有界关闭。

## 2. Paged KV Cache

### 实现

- [ ] 固定 Block Size 的 Rank-local K/V Pool；
- [ ] 每 Sequence 独立 Block Table、Generation、reserved capacity 和 committed length；
- [ ] Reserve/Commit/Rollback/Release 原子且 stale handle 可拒绝；
- [ ] Paged KV Write 和 Paged Attention 使用 row 自己的 Block Table；
- [ ] Pool exhausted 时状态不变，Release 后容量完全恢复。

### Numeric case ID

```text
paged_attention
kv_capacity_contract
```

`kv_capacity_contract` 必须实例化冻结的 Context、Concurrency、Block Size 和 Capacity
Policy，至少证明：承诺上限可 admission；不可能请求立即拒绝；失败 Reserve 不减少 Block；
Batch rollback 原子；Release 恢复全部容量。

### Metadata 与行为证据

- [ ] `capabilities` 包含 `paged_kv_cache`；
- [ ] `kv_block_size`、`kv_capacity_policy`、`max_context_length` 来自 Runtime；
- [ ] 两次跨 Block 长上下文请求完成且 Block 可复用。

## 3. Continuous Batching

### 实现

- [ ] 单一 Scheduler owner 和有界队列；
- [ ] Queued/Admitted/Prefill/Decode/Terminal 状态机；
- [ ] Packed rows 带独立 SequenceId、position、past length、KV view 和 Sampling state；
- [ ] Chunked Prefill 不提前提交请求状态；
- [ ] Cancel、Disconnect、Failure 最终只释放一次；
- [ ] `max_observed_batch_size` 在真实 Runner Step 更新。

### Numeric case ID

```text
packed_sequence_isolation
kv_capacity_contract
```

### Metadata 与行为证据

- [ ] `max_concurrency` 等于冻结值；
- [ ] 不同 Prompt 的并发输出与各自 deterministic baseline 一致；
- [ ] `max_observed_batch_size >= 2`，不是 HTTP socket 计数。

## 4. Tensor Parallelism

### 实现

- [ ] `tp_size` 个 Rank 绑定不同 Device；
- [ ] Attention/MLP Weight 依据 column/row semantics 分片；
- [ ] `num_attention_heads` 和 `num_kv_heads` 可被 `tp_size` 整除；
- [ ] 每层 Attention O 和 MLP Down 后 Collective 顺序一致；
- [ ] Rank 0 Sampling，Token 广播，所有 Rank 使用同一逻辑 Step；
- [ ] 任一 Rank 错误触发 group-wide abort，无 TP1 回退。

### Numeric case ID

```text
tp_collective
tp_sharded_linear
```

### Metadata 与行为证据

- [ ] `tp_size`、`world_size`、`rank`、`device_ids`；
- [ ] `weight_sharding` 和真实 `collective_backend`；
- [ ] reduced synthetic TP 数值与 Reference 一致；
- [ ] 真实模型仅使用冻结 TP topology 加载和生成。

## 5. Paged KV + Continuous Batching

- [ ] 同一 Tick 的所有 Sequence Reserve 全部成功后才 Commit；
- [ ] BatchAssembler 只读取已 Commit capacity 的 Block Table snapshot；
- [ ] Forward 成功后才统一 Advance committed length；
- [ ] 任一失败回滚整个 Tick 的未提交 reservation；
- [ ] 并发序列不会读取或释放彼此 Block。

详细状态顺序见 [联合状态机](../runtime/paged_continuous_state_machine.md)。

## 6. TP + Paged KV

- [ ] 每 Rank 只保存 local KV heads 和自己的 Physical Block Table；
- [ ] 同一逻辑 Sequence 的 committed length 在所有 Rank 一致；
- [ ] 所有 Rank Prepare 成功后才一起 Commit capacity；
- [ ] 任一 Rank exhaustion 导致所有 Rank rollback；
- [ ] 单请求跨 block 长上下文在冻结 TP topology 下完成。

详细关系见 [TP 与 Rank-local Paged KV 合同](../distributed/tp_paged_kv_contract.md)。

## 7. TP + Continuous Batching

- [ ] 一个逻辑 Scheduler 冻结所有 Rank 共用的 row order 和 token metadata；
- [ ] 未选择 Paged 时，每 Rank 使用本地 contiguous sequence slots；
- [ ] 所有 Rank 完成同一 Step 后才推进 committed length 和 Apply；
- [ ] Rank 0 Sampling 后广播唯一 token；
- [ ] 两个并发请求在 TP2 下实际进入 packed Runner。

详细关系见 [TP 与 Continuous Batching 合同](../distributed/tp_continuous_batching_contract.md)。

## 8. TP + Paged KV + Continuous Batching

- [ ] Rank 共享逻辑 SequenceId、row 和 token count；
- [ ] 每 Rank 独立计算和保存本地 KV Block Table；
- [ ] 所有 Rank Prepare 成功后 group barrier，再 Commit capacity；
- [ ] 任一 Rank Reserve/Forward/Collective 失败时 group-wide rollback/abort；
- [ ] 两个并发长上下文请求在冻结 TP topology 下完成。

详细关系见 [TP 与 Rank-local Paged KV 合同](../distributed/tp_paged_kv_contract.md)。

## 9. 禁止用作证据

- Header、Stub、固定 JSON Metadata 或未调用的方法；
- 只检查 API return code、不读取输出的 Numeric case；
- HTTP 并发成功但 Runner batch size 始终为 1；
- 完整真实模型 TP1 参考或单卡 fallback；
- Mock、固定答案、Echo、CPU LM-head；
- 仅日志声称 PASS，但 Report 缺少精确 case ID；
- Detailed advisory capacity 数字与实现不一致时的猜测值。
