# 架构 — LLM 推理框架总体设计

## 核心认识

所有 LLM 推理框架都遵循同一种架构模式：**分层流水线**，把 HTTP 请求变成 GPU 计算再变回响应。框架之间的差异主要体现在每一层的**实现方式**，而非整体结构。

## 通用架构模式

每个推理框架从外到内都包含这些层次：

```
[API 层] → [请求管理] → [调度器] → [模型执行器] → [GPU 执行]
     ↑         ↓          ↓            ↓            ↓
  响应        分词     KV 缓存管理   权重加载     Attention/MLP
                        内存池      CUDA Graph    采样
```

## 进程模型

### 单进程（nano-vllm 风格）

最简单：所有逻辑跑在一个进程、一个主循环里。

```
主进程:
  while 仍有待处理请求:
    batch = scheduler.schedule()
    output = model_runner.run(batch)
    scheduler.postprocess(output)
```

**优点**：实现简单，无进程间通信开销。  
**缺点**：分词/反分词会阻塞 GPU。  
**适用**：离线批推理、简单基准测试。

### 多进程（nano-sglang / mini-sglang 风格）

把 CPU 密集工作与 GPU 密集工作拆到不同进程。

```
进程 1（Tokenizer）：HTTP → 分词 → ZMQ → Router
进程 2（Router/Engine）：ZMQ → 调度 → 前向 → ZMQ
进程 3（Detokenizer）：ZMQ → 反分词 → ZMQ → Tokenizer
```

**观测到的进程间通信机制**：

- **ZeroMQ (ZMQ)**：进程间传递请求/响应（nano-sglang、mini-sglang）
- **multiprocessing.Pipe**：初始状态同步
- **SharedMemory + Events**：TP 多卡协调（nano-vllm）
- **NCCL**：GPU 卡之间的张量级通信

**优点**：CPU 不阻塞 GPU；便于流式返回。  
**缺点**：IPC 复杂、调试更难。  
**适用**：高吞吐生产 serving。

### 重叠调度（mini-sglang 进阶）

最优化形态：用独立 CUDA 流让 CPU 调度与 GPU 执行重叠。

```
第 N 步：  [GPU：执行 batch N] [CPU：准备 batch N+1]
第 N+1 步：[GPU：执行 batch N+1] [CPU：处理 N 的结果，准备 N+2]
```

**关键技巧**：为「元数据准备」使用专用 CUDA 流，使其与主流上的模型前向并发执行。

## 核心组件（每个框架都需要）

### 1. 请求状态机

每个请求经历如下生命周期：

```
WAITING → RUNNING (prefill) → RUNNING (decode) → FINISHED
                ↑                                    |
                └── PREEMPTED（可选）←───────────────┘
```

每个请求跟踪的数据：

- `input_ids` / `token_ids`：完整 token 序列
- `status`：当前生命周期阶段
- `block_table` 或 `table_idx`：到物理 KV 缓存位置的映射
- `num_cached_tokens` / `cached_len`：已有多少 token 的 KV 被缓存
- `sampling_params`：temperature、top-k、top-p 等

### 2. 调度器

决定「下一步算什么」。核心决策：

- **Prefill 与 Decode**：哪个阶段优先？
- **批组成**：哪些请求进入下一批？
- **内存预算**：能否在不 OOM 的前提下塞进更多请求？

### 3. KV 缓存管理器

管理 K/V 张量的 GPU 内存：

- **块/页分配**：逻辑 token 位置到物理内存的映射
- **前缀缓存**（可选）：共享前缀的请求复用 KV
- **驱逐**：容量满时释放内存

### 4. 模型执行器

执行神经网络前向：

- **输入准备**：由调度好的 batch 构造张量
- **前向**：跑模型
- **CUDA graph 回放**（可选）：decode 批使用预捕获执行序列
- **采样**：logits → 下一 token

## 设计决策矩阵


| 决策         | 简单方案         | 性能方案             |
| ---------- | ------------ | ---------------- |
| 进程模型       | 单进程          | 多进程 + ZMQ        |
| KV 缓存      | 每请求连续内存      | 分页块              |
| 前缀缓存       | 无            | Radix 树          |
| Decode 执行  | 即时 PyTorch   | CUDA graph 回放    |
| 调度         | FCFS         | 前缀感知（LPM/Weight） |
| Prefill 策略 | 一次性整段 prompt | 分块 prefill       |
| 抢占         | 无（OOM 则拒绝）   | 换入等待队列           |


## 源码对照


| 组件    | nano-vllm                 | nano-sglang                       | mini-sglang              |
| ----- | ------------------------- | --------------------------------- | ------------------------ |
| 入口    | `llm.py`                  | `server.py`                       | `core.py`                |
| 引擎循环  | `engine/llm_engine.py`    | `managers/router/manager.py`      | `engine/engine.py`       |
| 调度器   | `engine/scheduler.py`     | `managers/router/scheduler.py`    | `scheduler/scheduler.py` |
| KV 缓存 | `engine/block_manager.py` | `managers/router/radix_cache.py`  | `kvcache/radix_cache.py` |
| 模型执行器 | `engine/model_runner.py`  | `managers/router/model_runner.py` | `engine/engine.py`       |
| 内存池   | （在 model_runner 内）        | `memory_pool.py`                  | `kvcache/mha_pool.py`    |
| 请求状态  | `engine/sequence.py`      | `managers/router/infer_batch.py`  | `core.py`                |


