# TP 通信原语 — API 契约

> 关联 notebooks: —

## 概述

TP 通信层包含 3 种 collective 操作: all_reduce_sum, all_gather_last_dim, CustomAR (P2P all-reduce)。
源实现文件: `engine/tp_layers/distributed.py`

---

## 接口签名

### all_reduce_sum

- **签名**: `def all_reduce_sum(x: Tensor) -> Tensor`
- **注册**: `@torch.library.custom_op('meta_infer::all_reduce_sum', mutates_args=())`
- **语义**: 跨 TP ranks 求和，不改变 shape
- **TP=1**: `return x.clone()` (custom_op 禁止输出别名输入)
- **TP>1**: CustomAR P2P 优先 → NCCL fallback

```python
def all_reduce_sum(x):
    if not is_tp_enabled(): return x.clone()  # tp_size=1, must return new tensor
    if _custom_ar_handle is not None:
        return _custom_ar_handle.all_reduce(x, registered=False)  # CustomAR P2P staging buffer
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)  # NCCL fallback
    return y

@all_reduce_sum.register_fake
def _(x): return torch.empty_like(x)
```

### all_gather_last_dim

- **签名**: `def all_gather_last_dim(x: Tensor[..., d]) -> Tensor[..., d * tp_size]`
- **语义**: 沿最后一维拼接各 rank 的分片
- **实现**: `dist.all_gather(out_list, x)` + `torch.cat(out_list, dim=-1)` (非 `all_gather_into_tensor`)
- **TP=1**: 返回 x 自身

```python
def all_gather_last_dim(x):
    tp = get_tp_group()
    if not tp.is_tp_enabled(): return x
    out_list = [torch.empty_like(x) for _ in range(tp.size)]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=-1)
```

### CustomAR (P2P All-Reduce)

- **语义**: 可选的 P2P all-reduce 优化，通过 CUDA IPC handle 交换实现跨进程的 GPU 内存直接访问。初始化失败时自动 fallback 到 NCCL/RCCL all_reduce。
- **约束**: out-of-place (输出新 tensor，非输入别名)
- **初始化时机**: `load_weights` 后、首次 forward 前
- **依赖**: gloo ProcessGroup (用于 IPC handle exchange)
- **两套 IPC buffer**: meta_ptrs (元数据+staging) + buf_ptrs (纯 staging)
- **Exchange 方式**: 两套都用 `dist.all_gather_object` (不用 `broadcast_object_list`)
- **workspace_size**: 16 MB

### Rank/Size 契约

```python
tp_rank = dist.get_rank()  # or env RANK
tp_size = dist.get_world_size()  # or env WORLD_SIZE
```

### init_state_machine

```python
# ⚠️ CRITICAL: 整个 init 必须在 try/except 中
try:
    dist.barrier()
    gloo_group = dist.new_group(backend='gloo')
    max_size = 16 * 1024 * 1024  # 16 MB
    raw_ptr, ipc_handle = ops.allocate_shared_buffer_and_handle(max_size)
    handles = [None] * world_size
    dist.all_gather_object(handles, ipc_handle, group=gloo_group)
    pointers = [ops.open_mem_handle(h) if i != rank else raw_ptr for i, h in enumerate(handles)]
    # Step 6: init CustomAR state
    self._ptr = ops.init_custom_ar(pointers, rank_data, rank, fully_connected)
    ops.register_buffer(self._ptr, buf_pointers)
    dist.barrier()
except Exception as e:
    _custom_ar_handle = None  # ← 确保 None，触发 NCCL fallback
```

### register_buffer detail

两套 buffer 使用同一个 `_allocate_and_exchange_handles` 函数，内部使用 `dist.all_gather_object`:

```python
def _allocate_and_exchange_handles(size, group, rank, world_size):
    raw_ptr, ipc_handle = ops.allocate_shared_buffer_and_handle(size)
    handles = [None] * world_size
    dist.all_gather_object(handles, ipc_handle, group=group)
    pointers = [ops.open_mem_handle(h) if i != rank else raw_ptr for i, h in enumerate(handles)]
    return pointers

meta_ptrs = _allocate_and_exchange_handles(meta_size + max_size, ...)  # 元数据+staging
buf_ptrs = _allocate_and_exchange_handles(max_size, ...)               # 纯 staging
ops.register_buffer(self._ptr, buf_ptrs)
```

### Required imports

```python
# 实现层通过 engine/kernels/vllm_wrappers.py 导入，契约层只定义接口语义：
# ops.allocate_shared_buffer_and_handle(size) -> (int, bytes)  — 分配共享内存并返回指针+IPC handle
# ops.open_mem_handle(ipc_handle) -> int                       — 打开远程 IPC handle 获取本地指针
# ops.init_custom_ar(meta_ptrs, rank_data, rank, fully_connected) -> int  — 初始化 P2P all-reduce 状态
# ops.register_buffer(ptr, buf_ptrs) -> None                   — 注册 staging buffer
# ops.all_reduce(ptr, inp, out, reg_buffer, reg_buffer_sz_bytes) -> None  — 执行 P2P all-reduce
# ops.dispose(ptr) -> None                                     — 释放 P2P all-reduce 资源
# ops.meta_size() -> int                                       — 获取元数据大小
```

---

## 陷阱与反模式

- **FM-005**: dist.all_gather_object 需要 gloo ProcessGroup — NCCL 不支持 object collectives
- **CUSTOMAR-001**: init_custom_ar 必须在 try/except 内 — IPC handle 交换可能因平台/容器限制而失败（如 "Cannot access data pointer" 异常）。失败后 `_custom_ar_handle = None` 触发 NCCL/RCCL fallback
- **CUSTOMAR-002**: world_size=1 时 init_custom_ar 立即 return (no-op)
- **COMM-001**: 各 rank `load_weights` 后必须 `dist.barrier()` 再 `init_custom_ar()`
- **COMM-002**: TP 采样协议 — 仅 rank 0 执行采样，`dist.broadcast` 给所有 rank。严禁各 rank 独立采样
- **COMM-003**: `broadcast_object_list` 仅在 CUDA Graph 路径的 `register_graph_buffers()` 中使用，eager nocompile 不能用
