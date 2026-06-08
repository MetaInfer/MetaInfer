"""
Phase 8 — Scheduler: continuous batching with prefill-first priority.

Performs prefill/decode scheduling with overlength rejection.
block_size is injected externally (TP=256, else=16).
NO preempt() — nano-vllm preempt logic is explicitly deleted.
TP path: uses num_free_blocks parameter (not BlockManager internally).

All signatures must match inference_blueprint.json
  > components[0] Scheduler
  > data_flow_contracts.scheduler_tp_runner_bridge
"""

from typing import List, Optional

from engine.framework.sequence import Sequence, SequenceStatus


class ScheduleResult:
    """Result of a single schedule() call.

    At most one of scheduled_prefill / scheduled_decode is non-empty
    (prefill-first: either prefill batch or decode batch, never both).
    """

    def __init__(
        self,
        scheduled_prefill: Optional[List[Sequence]] = None,
        scheduled_decode: Optional[List[Sequence]] = None,
        rejected: Optional[List[Sequence]] = None,
    ):
        self.scheduled_prefill: List[Sequence] = scheduled_prefill or []
        self.scheduled_decode: List[Sequence] = scheduled_decode or []
        self.rejected: List[Sequence] = rejected or []

    @property
    def is_prefill(self) -> bool:
        return len(self.scheduled_prefill) > 0

    @property
    def batch(self) -> List[Sequence]:
        """Return the active batch (prefill takes priority)."""
        if self.scheduled_prefill:
            return self.scheduled_prefill
        return self.scheduled_decode

    def __repr__(self) -> str:
        return (
            f"ScheduleResult(prefill={len(self.scheduled_prefill)}, "
            f"decode={len(self.scheduled_decode)}, rejected={len(self.rejected)})"
        )


class Scheduler:
    """Continuous batching scheduler.

    Algorithm (prefill-first, nano-vllm style):
      1. Reject overlength waiting sequences (required_blocks > max_blocks).
      2. Prefill waiting sequences until free blocks exhausted.
      3. If no prefill scheduled, decode running sequences (each needs 1 block).
      4. NEVER preempt — no running.pop() / no deallocation.

    block_size is injected by LLMEngine (TP=256, else=16).
    max_blocks is injected for overlength rejection.
    """

    def __init__(self, block_size: int, max_blocks: int):
        self._block_size = block_size
        self._max_blocks = max_blocks
        self._reserved_blocks = 0

    # ------------------------------------------------------------------
    # schedule — main entry point
    # ------------------------------------------------------------------

    def schedule(
        self,
        waiting_seqs: List[Sequence],
        running_seqs: List[Sequence],
        num_free_blocks: int,
    ) -> ScheduleResult:
        """Schedule the next batch.

        Args:
            waiting_seqs: Sequences in WAITING status (new requests).
            running_seqs: Sequences in DECODE status (ongoing generation).
            num_free_blocks: Available KV cache blocks (from runner or BlockManager).

        Returns:
            ScheduleResult with scheduled prefill/decode batches and rejected sequences.
        """
        rejected: List[Sequence] = []
        remaining_waiting: List[Sequence] = []

        # Step 0: Overlength rejection — required_blocks > max_blocks
        for seq in waiting_seqs:
            req_blocks = self._required_blocks_for_prompt(seq.prompt_len)
            if req_blocks > self._max_blocks:
                seq.status = SequenceStatus.REJECTED
                rejected.append(seq)
            else:
                remaining_waiting.append(seq)

        # Step 1: Prefill first — try to schedule waiting sequences
        scheduled_prefill: List[Sequence] = []
        free = num_free_blocks - self._reserved_blocks
        for seq in remaining_waiting:
            req_blocks = self._required_blocks_for_prompt(seq.prompt_len)
            if req_blocks <= free:
                free -= req_blocks
                seq.num_blocks = req_blocks
                seq.status = SequenceStatus.PREFILL
                scheduled_prefill.append(seq)
            # else: not enough blocks, stays WAITING until next schedule() call

        if scheduled_prefill:
            self._reserved_blocks += sum(
                self._required_blocks_for_prompt(s.prompt_len)
                for s in scheduled_prefill
            )
            return ScheduleResult(
                scheduled_prefill=scheduled_prefill,
                scheduled_decode=[],
                rejected=rejected,
            )

        # Step 2: Decode — only when no waiting prefill scheduled
        scheduled_decode: List[Sequence] = []
        for seq in running_seqs:
            if seq.status == SequenceStatus.DECODE and free >= 1:
                free -= 1
                scheduled_decode.append(seq)
            # No preempt: if not enough blocks, skip silently.
            # Sequences that can't get a decode block will be retried
            # on the next schedule() call.

        self._reserved_blocks += len(scheduled_decode)
        return ScheduleResult(
            scheduled_prefill=[],
            scheduled_decode=scheduled_decode,
            rejected=rejected,
        )

    # ------------------------------------------------------------------
    # postprocess — advance sequence state after forward pass
    # ------------------------------------------------------------------

    def postprocess(
        self,
        batch: List[Sequence],
        is_prefill: bool,
        generated_tokens: List[int],
    ) -> None:
        """Postprocess after a forward step: advance state, check termination.

        Args:
            batch: Sequences that just completed a forward pass.
            is_prefill: Whether this was a prefill step.
            generated_tokens: Sampled token ids (one per sequence in batch).
        """
        eos_token_id = 151645  # Qwen3 EOS (standard for Qwen models)

        for seq, token_id in zip(batch, generated_tokens):
            if is_prefill:
                # First generated token after prefill → transition to DECODE
                seq.kv_len = seq.prompt_len
                seq.output_ids.append(token_id)
                seq.status = SequenceStatus.DECODE
            else:
                # Decode step
                seq.output_ids.append(token_id)
                seq.kv_len += 1

            # Termination checks
            if token_id == eos_token_id:
                seq.status = SequenceStatus.FINISHED
            elif len(seq.output_ids) >= seq.max_output_len:
                seq.status = SequenceStatus.FINISHED

        # Reset reservation counter after forward pass completes
        self._reserved_blocks = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _required_blocks_for_prompt(self, prompt_len: int) -> int:
        """Number of KV cache blocks needed for a prompt of given length."""
        return (prompt_len + self._block_size - 1) // self._block_size

    @property
    def block_size(self) -> int:
        return self._block_size
