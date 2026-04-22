# 请求生命周期 - 端到端流程

## 概述

本文追踪单个请求从 HTTP 输入到生成输出的完整路径，展示推理框架中各组件如何协同工作。

## 生命周期阶段

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. API 接收        │ HTTP POST /generate {prompt, params}           │
├──────────────────┼──────────────────────────────────────────────────┤
│ 2. 分词            │ 使用 HF tokenizer：text → token_ids            │
├──────────────────┼──────────────────────────────────────────────────┤
│ 3. 入队            │ 创建 Req/Sequence，加入 waiting 队列            │
├──────────────────┼──────────────────────────────────────────────────┤
│ 4. 调度            │ Scheduler 选取请求并检查内存预算                 │
├──────────────────┼──────────────────────────────────────────────────┤
│ 5. 缓存匹配        │ （可选）Radix cache 查找共享前缀                 │
├──────────────────┼──────────────────────────────────────────────────┤
│ 6. 内存分配        │ 为新内容分配 KV cache 块/词元                    │
├──────────────────┼──────────────────────────────────────────────────┤
│ 7. Prefill         │ 一次性处理所有（未缓存）prompt 词元              │
├──────────────────┼──────────────────────────────────────────────────┤
│ 8. Decode 循环     │ 每步生成一个词元，直到结束                        │
├──────────────────┼──────────────────────────────────────────────────┤
│ 9. 完成检查        │ EOS 词元？长度上限？停止字符串？                  │
├──────────────────┼──────────────────────────────────────────────────┤
│ 10. 写入缓存       │ （可选）将 token 序列写入 radix cache            │
├──────────────────┼──────────────────────────────────────────────────┤
│ 11. 内存释放       │ 释放 KV cache（或保留在缓存中）                   │
├──────────────────┼──────────────────────────────────────────────────┤
│ 12. 反分词         │ token_ids → text                                 │
├──────────────────┼──────────────────────────────────────────────────┤
│ 13. API 响应       │ 将生成文本返回给客户端                             │
└──────────────────┴──────────────────────────────────────────────────┘
```

## 详细流程：单进程（nano-vllm）

```python
# === Stage 1-3: Client side ===
engine = LLMEngine(config)
seq = Sequence(prompt_tokens, sampling_params)
engine.scheduler.add(seq)  # → waiting queue

# === Stage 4-9: Engine loop ===
while not seq.is_finished():
    # Stage 4: Schedule
    batch, is_prefill = engine.scheduler.schedule()
    #   - 优先检查 waiting queue（prefill 优先）
    #   - 检查 block_manager 是否有足够空闲块
    #   - 不够则抢占 running 序列

    # Stage 5-6: 内存分配（在 scheduler.schedule 内）
    #   - block_manager.allocate(seq) 分配物理块
    #   - 前缀缓存场景下，检查 hash_to_block_id 命中

    # Stage 7 or 8: 前向
    next_tokens = engine.model_runner.run(batch, is_prefill)
    #   - prepare_prefill/decode：构建输入张量
    #   - model forward：执行全部 transformer 层
    #   - 注意力层通过 context.slot_mapping 写 KV cache
    #   - sampler：logits 转 tokens

    # Stage 9: 后处理
    engine.scheduler.postprocess(batch, next_tokens)
    #   - 将 token 追加到序列
    #   - 检查 EOS / max_tokens
    #   - 若完成：block_manager.deallocate(seq)

# === Stage 12-13: Client side ===
output_text = tokenizer.decode(seq.output_tokens)
```

## 详细流程：多进程（nano-sglang）

```
Process 1: TokenizerManager（主进程）
├── 收到 HTTP POST /generate
├── 分词：text → token_ids
├── 创建带唯一 RID 的请求
├── 通过 ZMQ PUSH 发送 TokenizedGenerateReqInput 到 Router
├── 在 asyncio.Event 上等待该 RID
│
Process 2: Router（ModelRpcServer）
├── 通过 ZMQ PULL 接收请求
├── 创建 Req 对象，加入 forward_queue
├── 在 RadixCache 上 match_prefix(input_ids)
│   → 返回 (prefix_indices, last_node)
│   → 仅未缓存词元需要计算
├── 调度：检查是否满足内存预算
│   → 从 ReqToTokenPool + TokenToKVPool 分配
├── 执行前向（extend 模式）
│   → model_runner.forward(batch, EXTEND)
│   → 注意力层将 KV 写入 TokenToKVPool
│   → 采样下一个词元
├── 切换到 running_batch（decode 模式）
├── Decode 循环：
│   ├── forward(batch, DECODE) → 每请求一个词元
│   ├── 完成检查（EOS, max_tokens, stop_string）
│   ├── 若完成：
│   │   ├── 插入 RadixCache
│   │   ├── 释放 ReqToTokenPool 槽位
│   │   └── 减少 TokenToKVPool 引用计数
│   └── 通过 ZMQ PUSH 发送部分/最终结果到 Detokenizer
│
Process 3: Detokenizer
├── 通过 ZMQ PULL 接收 token IDs
├── 批量反分词：token_ids → text
├── 如有需要，裁剪 stop strings
├── 通过 ZMQ PUSH 发送 BatchStrOut 到 TokenizerManager
│
Process 1: TokenizerManager（继续）
├── 接收解码文本
├── 为对应 RID 设置 asyncio.Event
└── 返回 HTTP 响应给客户端
```

## 详细流程：重叠调度（mini-sglang）

```python
# Scheduler.overlap_loop():
next_batch = schedule_next_batch()

while next_batch is not None:
    current_batch = next_batch

    # === GPU: 执行当前 batch（异步） ===
    with engine.context.forward_batch(current_batch):
        if current_batch.phase == "prefill":
            logits = engine.model.forward(current_batch)
        else:
            logits = engine.graph_runner.replay(current_batch)
        tokens = sampler(logits)

    # === CPU: GPU 运行同时准备下一批 ===
    #   这部分在独立 CUDA stream 上执行
    next_batch = schedule_next_batch()
    #   1. 检查上一轮完成请求
    #   2. 释放已完成请求资源
    #   3. 接收 tokenizer 新请求
    #   4. 在 radix cache 匹配前缀
    #   5. 为新 batch 分配内存
    #   6. 构建 batch 张量

    # === 同步 ===
    # GPU 完成后处理结果
    postprocess(current_batch, tokens)
```

## 关键交互点

### Scheduler ↔ KV Cache Manager

- **调度时**："这个请求能放得下吗？" → 检查空闲块/词元
- **prefill 时**："为该请求分配 N 个块" → 预留物理内存
- **每个 decode 步**："再分配 1 个槽位" → 扩展分配
- **完成时**："释放该请求内存" → 归还块到内存池

### Model Runner ↔ KV Cache

- **前向时**：Model Runner 向注意力层提供 `slot_mapping`
- 注意力层用 `slot_mapping` **写入** 新 KV 数据
- 注意力层用 `block_tables` **读取** 历史 KV 数据

### Scheduler ↔ Radix Cache

- **调度时**："该请求共享了哪些前缀？" → `match_prefix()`
- **调度时**：增加命中节点 `ref_count` → 防止被驱逐
- **完成时**："将该请求 token 写入缓存" → `insert()`
- **完成时**：减少 `ref_count` → 无使用者后可被驱逐
- **OOM 时**："驱逐 LRU 条目" → `evict(num_tokens)` → 释放物理内存

## 终止条件

请求在任一条件满足时结束：

1. **EOS 词元**：模型生成 end-of-sequence 词元
2. **最大生成长度**：`output_len >= max_tokens`（由 sampling params 指定）
3. **停止字符串**：生成文本命中用户指定 stop string
4. **最大上下文长度**：总序列长度达到模型上下文上限

## 流式输出

对于流式响应，词元会增量发送：

```python
# nano-sglang: 每个 decode step 后
if req.stream:
    partial_output = BatchTokenIDOut(req.rid, [new_token_id])
    send_to_detokenizer(partial_output)
    # Detokenizer 再把部分文本回传给 TokenizerManager
    # TokenizerManager 通过 SSE 事件发给客户端
```

## 设计模板

实现完整生命周期至少需要：

1. **分词**：使用 HuggingFace tokenizer（或自定义）
2. **请求对象创建**：创建带状态跟踪的类型化对象
3. **队列管理**：waiting + running 两个队列
4. **内存检查**：调度前查询 KV cache 容量
5. **前向执行**：prefill 后进入 decode 循环
6. **完成检测**：检查 EOS、max_tokens、stop strings
7. **清理**：释放内存，并按需写入缓存复用
8. **反分词**：token 转回文本

