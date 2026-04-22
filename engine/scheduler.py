from collections import deque

from engine.memory_pool import KVMemoryPool
from engine.structs import Sequence, SequenceStatus


class Scheduler:
    """
    Prefill 优先；资源不足时等待（排队），不抢占正在运行的序列。
    """

    def __init__(self, memory_pool: KVMemoryPool, max_num_seqs: int, max_num_batched_tokens: int):
        self.memory_pool = memory_pool
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add_request(self, seq: Sequence) -> None:
        if seq.status != SequenceStatus.WAITING:
            raise ValueError("Only waiting sequence can be added")
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        prefill_batch = self._schedule_prefill()
        if prefill_batch:
            return prefill_batch, True
        return self._schedule_decode(), False

    def _schedule_prefill(self) -> list[Sequence]:
        batch: list[Sequence] = []
        budget = self.max_num_batched_tokens
        while self.waiting and len(batch) < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            need_tokens = seq.total_tokens - seq.num_cached_tokens
            if need_tokens <= 0:
                self.waiting.popleft()
                if seq.status == SequenceStatus.WAITING:
                    seq.transition_to(SequenceStatus.RUNNING_PREFILL)
                seq.transition_to(SequenceStatus.RUNNING_DECODE)
                self.running.append(seq)
                continue
            if need_tokens > budget:
                break
            seq.block_size = self.memory_pool.block_size
            if not self.memory_pool.can_allocate(seq):
                break

            self.waiting.popleft()
            seq.transition_to(SequenceStatus.RUNNING_PREFILL)
            self.memory_pool.allocate_for_sequence(seq, seq.total_tokens)
            self.running.append(seq)
            batch.append(seq)
            budget -= need_tokens
        return batch

    def _schedule_decode(self) -> list[Sequence]:
        batch: list[Sequence] = []
        if not self.running:
            return batch
        for seq in list(self.running):
            if len(batch) >= self.max_num_seqs:
                break
            if self.memory_pool.can_append_one_more(seq):
                batch.append(seq)
        return batch

    def postprocess(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        generated_tokens: list[int] | None = None,
    ) -> None:
        if is_prefill:
            # 可选：prefill 末尾已采样第一个生成 token（与 HF past_key_values 增量解码衔接）
            if generated_tokens:
                for seq, token in zip(seqs, generated_tokens):
                    seq.append_token(token)
                    self.memory_pool.ensure_capacity_for_sequence(seq)
            for seq in seqs:
                seq.num_cached_tokens = seq.total_tokens
                seq.transition_to(SequenceStatus.RUNNING_DECODE)
            return
        generated_tokens = generated_tokens or []
        for seq, token in zip(seqs, generated_tokens):
            seq.append_token(token)
            self.memory_pool.ensure_capacity_for_sequence(seq)
