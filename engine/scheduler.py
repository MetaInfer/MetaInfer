"""
Phase 8 — Scheduler: prefill-first continuous batching with injected block_size.

Nano-vllm overrides applied:
  - preempt() method DELETED (nano-vllm L66-69). TP path has no preemption;
    deallocate() in postprocess handles resource release.
  - block_size injected via self._block_size (default 16 HF, 256 TP).
    LLMEngine sets scheduler._block_size and scheduler._max_blocks.
  - schedule(num_free) accepts num_free_blocks from caller (BlockManager or runner).
  - TP path: can_allocate / can_append_one_more use num_free directly,
    not BlockManager.can_allocate / may_append.
  - REJECTED status for overlength prompts prevents infinite WAITING loop.
"""

from engine.structs import Sequence, SeqStatus


class Scheduler:
    """Prefill-first scheduler with injected block_size.

    preempt() is intentionally absent — the nano-vllm override requires its deletion.
    Resource release is handled via postprocess → _release → free block_table.
    """

    def __init__(self, memory_pool=None, max_num_seqs=4, max_num_batched_tokens=4096,
                 eos_token_id=None):
        self._block_size = 16         # DEFAULT (HF path), overridden by LLMEngine for TP
        self._max_blocks = 128        # DEFAULT, overridden by LLMEngine
        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens
        self._reserved_blocks = 0     # B=1 counter, reset in _release
        self._eos_token_id = eos_token_id
        # queues
        self.waiting = []   # list[Sequence]
        self.running = []   # list[Sequence]

    def add(self, seq: Sequence):
        """Enqueue a sequence for scheduling."""
        # overlength rejection
        req = (seq.seq_len() + self._block_size - 1) // self._block_size
        if req > self._max_blocks:
            seq.transition_to(SeqStatus.REJECTED)
            return
        self.waiting.append(seq)

    def schedule(self, num_free):
        """Schedule one batch of sequences.

        Phase 1 (prefill): pick from waiting if can_allocate and within token budget.
        Phase 2 (decode): if waiting empty, pick from running if can_append_one_more.

        Args:
            num_free: number of free blocks available (from BlockManager or runner).

        Returns:
            (batch: list[Sequence], is_prefill: bool)
        """
        batch = []
        reserved = 0
        current_tokens = 0

        # ---- Phase 1: Prefill (prefill-first) ----
        for seq in list(self.waiting):
            req = seq.required_blocks()
            # REJECTED: prompt too long (should have been caught in add, but guard here)
            if req > self._max_blocks:
                seq.transition_to(SeqStatus.REJECTED)
                self.waiting.remove(seq)
                continue
            # Can allocate?
            if reserved + req > num_free:
                break   # insufficient free blocks
            if current_tokens + seq.seq_len() > self._max_num_batched_tokens:
                break   # token budget exhausted
            batch.append(seq)
            reserved += req
            current_tokens += seq.seq_len()

        if batch:
            for seq in batch:
                self.waiting.remove(seq)
                seq.transition_to(SeqStatus.RUNNING_PREFILL)
                self.running.append(seq)
            self._reserved_blocks += reserved
            return batch, True

        # ---- Phase 2: Decode (waiting empty) ----
        for seq in list(self.running):
            # can_append_one_more: need at least 1 free block
            if num_free - reserved >= 1:
                batch.append(seq)
                reserved += 1

        if batch:
            # No status transition — decode sequences are already RUNNING_DECODE
            # from previous postprocess() call. transition_to(RUNNING_DECODE) again
            # would AssertionError (not in valid transition table).
            self._reserved_blocks += reserved
            return batch, False

        # Empty: nothing schedulable
        return [], False

    def postprocess(self, batch, is_prefill, generated_tokens):
        """Update output_ids, detect stop conditions, transition states.

        Called after ModelRunner.run() returns next tokens.

        Args:
            batch: list[Sequence] from schedule()
            is_prefill: True for prefill batch, False for decode
            generated_tokens: list[int] of next tokens per sequence
        """
        for i, seq in enumerate(batch):
            token = generated_tokens[i] if i < len(generated_tokens) else None

            if is_prefill:
                # Write first generated token
                if token is not None:
                    seq.append_token(token)
                seq.transition_to(SeqStatus.RUNNING_DECODE)
                # kv_len is set by runner during prefill
            else:
                # Decode: append token, check stop conditions
                if token is not None:
                    seq.append_token(token)
                    # EOS check
                    if self._eos_token_id is not None and token == self._eos_token_id:
                        seq.transition_to(SeqStatus.FINISHED)
                        self._release(seq)
                        continue
                    # max_tokens check
                    if seq.max_tokens > 0 and seq.num_completion_tokens >= seq.max_tokens:
                        seq.transition_to(SeqStatus.FINISHED)
                        self._release(seq)
                        continue

    def _release(self, seq):
        """Release a finished sequence's resources."""
        if seq in self.running:
            self.running.remove(seq)
        seq.block_table = []           # clear block list
        seq._block_table_tensor = None  # clear GPU tensor (next init will re-create)
        self._reserved_blocks = 0      # reset counter (B=1; B>1 needs per-seq tracking)

    def is_finished(self):
        return not self.waiting and not self.running

    # ---- preempt() intentionally DELETED ----
    # nano-vllm L66-69 preempt() causes sequence loss in TP path.
    # Resource release is handled by _release in postprocess.
