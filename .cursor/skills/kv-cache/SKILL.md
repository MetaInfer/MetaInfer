---
name: kv-cache
description: >-
  Designs prefix-aware KV cache and radix-tree caching for LLM inference engines.
  Use when implementing or refactoring block tables, paged KV, radix prefix cache,
  cache hit/miss, eviction, or when the user mentions nano-vllm BlockManager,
  nano-sglang/mini-sglang RadixCache, or Agent-generated inference frameworks.
---

# KV Cache（逻辑层）

## 与 Memory Pool 的分工

- **KV Cache 组件**：管「**哪些 token 的 KV 已存在、能否复用、如何索引**」——块表、前缀匹配、Radix 树、引用计数、驱逐策略、与调度器的交互。
- **Memory Pool 组件**：管「**GPU 上 KV 张量缓冲区从哪来、槽位/页如何分配**」——见姊妹技能 `memory-pool`。

二者通过 **物理槽位索引**（block id、token slot、`indices` 张量）衔接：KV Cache 命中返回已有下标；未命中向 Memory Pool 申请新槽位并写回树/块表。

## 源码对照（ref_projects）

| 项目 | 主文件 | 职责摘要 |
|------|--------|----------|
| nano-vllm | `nanovllm/engine/block_manager.py` | 块级分配 + **xxhash 块内容哈希**前缀复用；`hash_to_block_id`；`ref_count` |
| nano-sglang | `python/sglang/srt/managers/router/radix_cache.py` | Radix 树：`match_prefix` / `insert` / `evict`；`inc_ref_counter` / `dec_ref_counter`；`evictable_size_` |
| mini-sglang | `python/minisgl/kvcache/radix_cache.py` | `RadixPrefixCache`：`match_prefix`、`insert_prefix`（按 `page_size` 对齐）、`lock_handle`、LRU 叶节点驱逐 |
| mini-sglang（无前缀） | `python/minisgl/kvcache/naive_cache.py` | `NaivePrefixCache`：永不命中、无驱逐；占位实现 |

## nano-vllm：BlockManager 要点

- **`Block`**：`block_id`、`ref_count`、`hash`、`token_ids`；`reset()` 将 `ref_count` 置 1。
- **`allocate(seq)`**：按序列块迭代；满块用 `compute_hash(token_ids, prefix_hash)` 查 `hash_to_block_id`；命中则 `ref_count += 1` 且可更新 `num_cached_tokens`；未命中从 `free_block_ids` 取块并 `update` 哈希表。
- **`deallocate(seq)`**：逆序遍历 `block_table`，`ref_count--`，为 0 时归还空闲块。
- **`may_append(seq)`**：序列长度跨块边界时挂新块；块填满时用前缀块 hash 计算当前块 hash 并登记。

与 **ModelRunner** 协同：`prepare_prefill` / `prepare_decode` 用 `block_table` 生成 `slot_mapping` 与 `block_tables`（见 `model_runner.py`），供 attention 写入/读取分页 KV。

## nano-sglang：RadixCache 要点

- **`TreeNode`**：`children`（边键为 token 序列）、`value`（该边对应 KV 槽位索引列表，常为 tensor 片段）、`ref_counter`、`last_access_time`。
- **`match_prefix(key)`**：沿树匹配；部分匹配时 `_split_node` 分裂边。
- **`insert(key, value)`**：插入剩余后缀为新叶或延伸路径。
- **`evict(num_tokens, evict_callback)`**：叶节点最小堆（按 `last_access_time`）；`ref_counter > 0` 跳过；`evict_callback` 负责把 `node.value` 对应槽位还给 **TokenToKVPool**。
- **`inc_ref_counter` / `dec_ref_counter`**：沿父链更新；`ref_counter` 0↔1 时维护 `evictable_size_`。

## mini-sglang：RadixPrefixCache 要点

- 边上存 **`torch.Tensor` key/value**（value 为物理 KV 池下标）；`get_match_len` 可用 `fast_compare_key`。
- **`insert_prefix`**：`insert_len = align_down(len, page_size)`，与 **MHAKVCache 页**对齐。
- **`lock_handle`**：请求持有前缀时沿路径 `ref_count++/--`，维护 `evictable_size` / `protected_size`。
- **`evict`**：叶节点 LRU（`timestamp`），`ref_count==0` 才可驱逐；删除叶后若父成叶且可驱逐则重新入堆。

## Agent 生成推理框架时的检查清单

1. **是否分页**：块表（nano-vllm）还是 token 槽位 + radix（sglang 系）？
2. **前缀语义**：仅完整块哈希（nano-vllm）还是任意前缀（Radix）？
3. **引用生命周期**：请求开始/结束/抢占时何时 `inc`/`dec`？驱逐前是否保证 `ref_count==0`？
4. **与 runner 的接口**：输出是否为 `block_tables` + `slot_mapping` 或 `req_to_token` 行？
5. **MLA / GQA**：逻辑层不变，但 **每 token KV 字节数** 影响池容量（由 memory-pool 技能估算）。

## 延伸阅读

- 项目内：`notebooks-cn/01_framework_design/03_kv_cache.md`
- 项目内：`notebooks-cn/01_framework_design/01_architecture.md`（源码对照表）
