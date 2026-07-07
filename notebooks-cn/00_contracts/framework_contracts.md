# 框架外壳 — API 契约

> 关联 notebooks: `01_framework_design/02_scheduler.md`, `01_framework_design/03_kv_cache.md`

## 概述

Sequence, Scheduler, Sampler, BlockManager 四个框架组件。
源实现文件: `engine/framework/sequence.py`, `engine/framework/scheduler.py`, `engine/framework/sampler.py`, `engine/framework/block_manager.py`

---

## Sequence

```python
class Sequence:
    seq_id: int
    prompt: str
    input_ids: list[int]      # 完整 token 序列 (prompt + generated)
    output_ids: list[int]      # 仅生成的 token
    status: SequenceStatus     # WAITING / PREFILL / DECODE / FINISHED / REJECTED
    max_tokens: int
    eos_token_id: int
    kv_len: int = 0            # 当前 KV cache 长度 (token 数)。用于 decode RoPE 位置编码
    block_table: Optional[Tensor]  # [max_blocks] int32 — TP 路径由 QwenAttentionTP 管理，框架层 None
    
    def append_token(self, token_id: int)
    def finish(self)
    def seq_len(self) -> int
    def output_len(self) -> int
```

**Status 转移**:
- `WAITING` → `PREFILL` (schedule 选中)
- `PREFILL` → `DECODE` (prefill 完成后 runner 内转换)
- `DECODE` → `FINISHED` (EOS 或 max_tokens 达到)

**TP 双轨**: block_table 在 TP 路径为 None (实际 block_table 由 QwenAttentionTP 内部 torch.arange 管理)

---

## Scheduler

```python
class Scheduler:
    def __init__(self, block_size: int, max_num_seqs: int)
    def add(self, seq: Sequence)
    def schedule(self, num_free_blocks: int) -> tuple[list[Sequence], list[Sequence]]
    def postprocess(self, seq: Sequence, token: int)
```

### postprocess() 行为

```python
def postprocess(self, seq: Sequence, token: int):
    """Called after model forward to update sequence state."""
    seq.append_token(token)
    if token == seq.eos_token_id or seq.output_len() >= seq.max_tokens:
        seq.finish()

    # Move to running if was prefill
    if seq.status == SequenceStatus.PREFILL:
        # CRITICAL: kv_len must be set to prefill token count.
        # input_ids already includes the decoded token (via append_token above),
        # so prefill length = len(input_ids) - 1.
        # Without this, _run_decode would use kv_len=0 for RoPE positions,
        # corrupting all FullAttention layer outputs.
        seq.kv_len = len(seq.input_ids) - 1
        seq.status = SequenceStatus.DECODE
        if seq not in self.running:
            self.running.append(seq)
```

### schedule() 签名

```python
def schedule(self, num_free_blocks: int) -> tuple[list[Sequence], list[Sequence]]:
    """
    返回 (prefill_seqs, decode_seqs)
    
    优先级:
    1. 从 waiting 组选择 prefill 序列 (需满足 max_num_batched_tokens 和 can_allocate)
    2. 从 running 组选择 decode 序列 (需满足 can_append_one_more)
    
    资源不足时不抢占 (preemption disabled)
    """
```

### can_allocate 公式

```
required_blocks = (seq_len + block_size - 1) // block_size
can_allocate = required_blocks <= num_free_blocks
```

### REJECTED 机制

超长 prompt 无法分配时: `seq.status = SequenceStatus.REJECTED`
防止永久 WAITING 死循环。

---

## Sampler

```python
class Sampler:
    def __init__(self, temperature: float = 0.0)
    def sample(self, logits: Tensor) -> Tensor  # [B, vocab_size] → [B]
```

**TP 采样协议**:
- 仅 rank 0 执行采样
- `dist.broadcast(token_ids, src=0)` 给所有 rank
- 严禁各 rank 独立采样 (CUDA 随机种子不同 → token 不一致 → NCCL 崩溃)

---

## BlockManager

```python
class BlockManager:
    def __init__(self, num_blocks: int, tp_mode: bool = False)
    def allocate(self, seq: Sequence, num_blocks: int) -> bool
    def free(self, seq: Sequence)
    def get_num_free_blocks(self) -> int
```

**TP 降级 (no-op)**:
- `tp_mode=True` 时 `allocate()` / `free()` 为 no-op (仅做容量计数)
- `get_num_free_blocks()` 保留但调用方改用 `runner.get_num_free_blocks()`
- 通过类内 `if self._tp_mode` 条件分支实现 (非继承或猴子补丁)

---

## Scheduler ↔ TP Runner 集成桥接

### block_size 注入

```python
# LLMEngine.__init__ 中根据 inference_backend 注入
if self.inference_backend in (None, 'hf'):
    scheduler._block_size = 16
else:
    scheduler._block_size = 256  # TP Runner 使用 paged KV cache
```

### num_free_blocks 来源路由

```python
# TP 路径: runner 提供 (从 _kv_len_gpu 推算)
num_free = runner.get_num_free_blocks()
# HF 路径: BlockManager 提供
num_free = block_manager.get_num_free_blocks()
```

---

## 陷阱与反模式

- **SCHED-001**: schedule() 先 prefill 后 decode，prefill 优先策略
- **SCHED-002**: REJECTED 机制防止永久 WAITING
- **SCHED-003**: TP 路径 block_size=256 (非默认 16)
- **TP-FW-001**: TP 路径 BlockManager allocate/free no-op
- **TP-FW-002**: TP 路径 block_table 由 QwenAttentionTP 管理，框架层为 None
- **TP-FW-003**: Sampler TP 协议 — rank 0 采样 + broadcast
