---
name: memory-pool
description: >-
  Pre-allocates GPU KV buffers and implements slot/page allocation with optional
  reference counting. Use when implementing ReqToTokenPool, TokenToKVPool,
  MHAKVCache, nano-vllm allocate_kv_cache, profiling GPU memory for KV capacity,
  or Agent-generated inference frameworks.
---

# Memory Pool（物理层）

## 与 KV Cache 的分工

- **Memory Pool**：**预分配**大 KV 张量；**O(1) 或批量**分配/回收**槽位或页**；`mem_state` / 空闲栈；可选**引用计数**以支持前缀共享；向 kernel 暴露 `kv_data` / `get_key_buffer` 视图。
- **KV Cache**：谁占用哪些槽位、Radix/块表逻辑；见姊妹技能 `kv-cache`。

## 源码对照（ref_projects）

| 项目 | 主文件 | 职责摘要 |
|------|--------|----------|
| nano-vllm | `nanovllm/engine/model_runner.py` → `allocate_kv_cache` | 单张量 `[2, layers, num_blocks, block_size, kv_heads, head_dim]`；按层把 `k_cache`/`v_cache` 绑到模块 |
| nano-sglang | `python/sglang/srt/memory_pool.py` | `ReqToTokenPool` + `TokenToKVPool`；请求槽与 KV 槽分离 |
| mini-sglang | `python/minisgl/kvcache/mha_pool.py` | `MHAKVCache`：页级 buffer + `store_kv` → `store_cache` kernel |

## nano-vllm：allocate_kv_cache

1. **显存预算**：`mem_get_info` + `memory_stats` peak/current；  
   `block_bytes = 2 * num_layers * block_size * num_kv_heads_tp * head_dim * dtype.itemsize`（TP 下 `num_kv_heads // world_size`）。
2. **`num_kvcache_blocks`**：`(total * gpu_memory_utilization - used - peak + current) // block_bytes`。
3. **`kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)`**。
4. **层绑定**：遍历 `model.modules()`，对含 `k_cache`/`v_cache` 的层：`module.k_cache = kv_cache[0, layer_id]`（view）。

块 ID 的分配/释放由 **`BlockManager`**（kv-cache 技能）维护，与物理张量行一一对应。

## nano-sglang：两级池

### ReqToTokenPool

- `req_to_token`: `[max_reqs, max_context_len]` int32，**每请求一行**，存各 position 对应的**全局 KV 槽下标**。
- `mem_state`: `[max_reqs]` bool，**1=该请求槽可用（未占用）**，`alloc` 时置 0；与常见「1=占用」直觉相反，以源码为准。
- `alloc(need_size)`：`nonzero(mem_state)` 取前 `need_size` 个请求槽索引；`free` 时恢复为 1。

### TokenToKVPool

- `kv_data[layer]`: `[size, 2, head_num, head_dim]`（每层一张量，适配 Triton/FlashInfer）。
- `mem_state`: **int16 引用计数**（0 表示可分配）；`alloc` → `add_refs`；`free` → `decrease_refs`。
- `alloc_contiguous(need_size)`：在空闲下标中找**连续**区间，满足部分 kernel 偏好。
- `get_kv_data_flashinfer`：reshape 为 FlashInfer NHD 页格式。

**数据流**：Radix `match_prefix` 得到已有 value 张量 → 新 token `alloc` 新槽 → `req_to_token[req, pos] = slot_idx`；attention 按 slot scatter 写入 `kv_data`。

## mini-sglang：MHAKVCache

- **`_kv_buffer`**: `[2, num_layers, num_pages, page_size, local_kv_heads, head_dim]`；TP 下 `local_kv_heads = div_even(...)`。
- **`k_cache(i)` / `v_cache(i)`**：第 `i` 层的 K/V 子张量视图。
- **`store_kv(k, v, out_loc, layer_id)`**：调用 `store_cache` Triton kernel，按 `out_loc` 写入扁平化 `view(num_pages*page_size, heads, dim)`。

与 **RadixPrefixCache** 对齐：`insert_prefix` 长度按 `page_size` 对齐，保证索引与页边界一致。

## 容量估算模板（Agent 可用）

```text
bytes_per_token = 2 * num_layers * num_kv_heads_after_tp * head_dim * dtype_bytes
# 分页时：
bytes_per_block = bytes_per_token * block_size   # nano-vllm
# 或页：
bytes_per_page = bytes_per_token * page_size     # mini-sglang MHA
available = total_gpu_bytes * utilization - model_and_activation_peak
num_slots = available // bytes_per_block_or_page
```

GQA：用 **KV head 数** 而非 Q head 数。

## Agent 生成推理框架时的检查清单

1. 池是否在 **模型加载 + warmup** 后按 peak 内存计算？
2. dtype/device 是否与模型一致？TP 下 head 维度是否除 `world_size`？
3. 前缀共享时是否用 **引用计数** 而非简单 bool？
4. 是否暴露 runner 所需的 **buffer 视图**（按层 / 按页）？
5. `ReqToTokenPool` 与 `TokenToKVPool` 的 **free** 顺序：先释放请求行再减 KV 引用，避免悬空索引。

## 延伸阅读

- [06_memory_pool.md](../06_memory_pool.md)
- [01_architecture.md](../01_architecture.md)（源码对照表）

**Cursor 技能副本**：`meta-infer/.cursor/skills/memory-pool/SKILL.md`（便于 Agent 发现）
