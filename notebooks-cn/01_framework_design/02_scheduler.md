# 调度器 — 连续批处理与请求调度

## 核心概念

调度器是推理框架的「大脑」。它在每一步决定**处理哪些请求**、处于**哪个阶段**（prefill 或 decode）。相对朴素批处理，关键创新是**连续批处理（continuous batching）**：请求可在任意时刻进出 batch，而不必等整批结束。

## 两阶段执行模型

LLM 推理有两个本质不同的阶段：

### Prefill（Extend）阶段

- **输入**：新请求的多个 prompt token
- **计算**：所有 prompt token 并行计算（类似训练）
- **输出**：所有 prompt 位置的 KV 已写入，并产生第一个生成 token
- **特点**：计算密集、算术强度高

### Decode 阶段

- **输入**：每个请求一个新 token
- **计算**：对全部历史 KV 做 attention + 当前位置的 MLP
- **输出**：每个请求一个新 token
- **特点**：内存带宽受限、算术强度低

### 对调度为何重要

Prefill 与 decode 的资源画像不同。好的调度器要平衡：

- **Prefill 延迟**：新请求多快拿到首 token（TTFT）
- **Decode 吞吐**：所有在跑请求合计每秒生成多少 token
- **内存预算**：KV 缓存有限，每个运行中请求都占内存

## 调度算法

### 1. Prefill 优先（nano-vllm 风格）

```
def schedule():
    if waiting_queue 非空:
        # 尽量多调度 prefill
        batch = select_from_waiting(budget=max_num_batched_tokens)
        return batch, is_prefill=True
    else:
        # 没有等待中的新请求，再做 decode
        batch = running_queue
        return batch, is_prefill=False
```

**逻辑**：优先启动新请求；仅当没有新请求等待时才 decode。

**权衡**：TTFT 好，但若新请求持续涌入可能饿死 decode。

### 2. Prefill/Decode 交错（nano-sglang 风格）

```
def forward_step():
    if can_schedule_new_requests() and num_decode_steps >= threshold:
        batch = get_new_fill_batch()  # prefill
        num_decode_steps = 0
    else:
        batch = running_batch  # decode
        num_decode_steps += 1
```

**逻辑**：在两次 prefill batch 之间固定跑若干步 decode（例如 10 步），降低调度开销、摊薄上下文切换成本。

### 3. 重叠调度（mini-sglang 风格）

```
def overlap_loop():
    next_batch = schedule_next_batch()
    while next_batch:
        current_batch = next_batch
        # GPU 执行 current_batch
        future = engine.forward_batch(current_batch)
        # GPU 忙时 CPU 准备下一批
        next_batch = schedule_next_batch()
        # 等待 GPU
        future.wait()
        postprocess(current_batch)
```

**逻辑**：CPU 调度与 GPU 计算重叠，隐藏调度延迟；元数据准备可走独立 CUDA 流。

## 关键调度决策

### 批大小预算

调度器通常限制 token 预算：

- `max_num_batched_tokens`：单次 prefill batch 最大 token 数（如 16384）
- `max_num_seqs`：最大并发序列数（如 512）

### 分块 Prefill

prompt 太长无法一次装入 batch 时：

```
if prompt_len > remaining_budget:
    chunk_size = remaining_budget
    只调度 chunk_size 个 token
    标记请求为「部分 prefill 完成」
    # 下一步 prefill 继续
```

**nano-vllm 规则**：每个 batch 至多一个序列可被分块（最后加入的那一个）。

**mini-sglang 规则**：`PrefillAdder` 管理分块逻辑，用 `ChunkedReq` 跟踪已处理了多少 prompt。

### 抢占（nano-vllm）

decode 时 KV 满：

```
def handle_oom_during_decode():
    # 将最低优先级序列移回等待
    victim = running_queue.pop()
    deallocate_blocks(victim)
    victim.status = WAITING
    waiting_queue.appendleft(victim)
```

### 前缀感知调度（nano-sglang）

三种启发式重排队列以提高缓存命中：

1. **FCFS**：先来先服务（默认）
2. **LPM（最长前缀匹配）**：优先处理与 radix 缓存共享最长前缀的请求
3. **Weight**：按树打分，考虑每个缓存节点被多少待处理请求共享——偏向能让最多请求受益的分支

## 内存预算计算

调度前须确认 KV 是否够用：

```python
# nano-vllm 风格（按块）
needed_blocks = ceil(prompt_len / block_size) - cached_blocks
can_schedule = block_manager.num_free_blocks >= needed_blocks

# nano-sglang 风格（按 token）
needed_tokens = prompt_len - prefix_cached_len
can_schedule = token_pool.available() >= needed_tokens
```

### Decode 内存预留（mini-sglang）

decode 每步每请求还要再占 1 个 token 的空间，调度器需**预留**：

```python
# 每个运行中请求预留一整页，避免 decode 中途 OOM
inflight_tokens = num_running_reqs * page_size
available = total_pages - used_pages - inflight_reserved
```

## 后处理

一步前向之后，调度器更新状态：

```python
def postprocess(batch, sampled_tokens):
    for req, token in zip(batch.reqs, sampled_tokens):
        req.append_token(token)
        if token == eos_token or req.output_len >= max_tokens:
            req.status = FINISHED
            free_kv_cache(req)
            # radix 缓存系统：把 token 序列插入缓存供以后复用
            cache.insert(req.token_ids, req.cache_indices)
```

## 设计模板

最小调度器需要：

1. **两个队列**：`waiting`（新请求）、`running`（生成中）
2. **schedule()**：在内存与 token 预算内选出下一批
3. **postprocess()**：更新状态、检测结束、释放资源
4. **内存检查**：向 KV 管理器查询剩余容量

可按需求增加：

- 分块 prefill（超长 prompt）
- 抢占（超订系统）
- 前缀感知调度（使用 radix 缓存时）
- 重叠调度（追求极限吞吐）

