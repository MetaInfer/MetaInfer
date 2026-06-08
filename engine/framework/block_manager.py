"""
Phase 8 — BlockManager: paged KV cache block allocator.

Supports two modes:
  - Normal mode: full block allocation with prefix caching via hash lookup.
  - TP mode (tp_mode=True): degraded to no-op counter. allocate() returns
    dummy block IDs, free() is no-op. Real KV cache management is handled
    by QwenAttentionTP internally.

All signatures must match inference_blueprint.json
  > components[2] BlockManager
  > _tp_degradation_fork_interface
"""

from typing import Dict, List, Optional, Set, Tuple


class BlockManager:
    """Paged KV cache block manager.

    Normal mode:
      - Maintains free/used block pools.
      - Prefix caching: hash(token_ids) → block_id mapping.
      - Reference counting for block sharing.

    TP mode (tp_mode=True):
      - allocate() returns placeholder list(range(n)).
      - free() is a no-op.
      - get_num_free_blocks() returns a very large number (capacity signal).
      - Real KV cache is managed by QwenAttentionTP internally.
    """

    def __init__(
        self,
        num_blocks: int,
        tp_mode: bool = False,
        block_size: int = 16,
    ):
        self._tp_mode = tp_mode
        self._block_size = block_size

        if tp_mode:
            # TP mode: no real block tracking needed
            self._num_blocks = num_blocks
            self._free_pool: Set[int] = set()
        else:
            self._free_pool: Set[int] = set(range(num_blocks))
            self._used_blocks: Set[int] = set()
            self._ref_count: Dict[int, int] = {}
            self._hash_to_block_id: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Allocate
    # ------------------------------------------------------------------

    def allocate(self, seq, num_blocks: int) -> List[int]:
        """Allocate KV cache blocks for a sequence.

        Normal mode: checks prefix caching (hash lookup), allocates new blocks,
        increments ref counts.

        TP mode: returns placeholder list [0, 1, ..., n-1].

        Args:
            seq: Sequence object (used for block_table and hash lookup).
            num_blocks: Number of blocks to allocate.

        Returns:
            List of allocated block IDs.
        """
        if self._tp_mode:
            # TP mode: dummy allocation — real blocks managed by QwenAttentionTP
            return list(range(num_blocks))

        allocated: List[int] = []
        for i in range(num_blocks):
            # Compute hash for prefix caching
            start = i * self._block_size
            end = min(start + self._block_size, len(seq.input_ids))
            token_ids = tuple(seq.input_ids[start:end])

            block_hash = hash(token_ids)

            # Prefix cache hit: reuse existing block
            if block_hash in self._hash_to_block_id:
                block_id = self._hash_to_block_id[block_hash]
                self._ref_count[block_id] = self._ref_count.get(block_id, 0) + 1
                allocated.append(block_id)
                continue

            # Allocate new block from free pool
            if not self._free_pool:
                # No free blocks — caller should have checked via
                # get_num_free_blocks() before calling allocate()
                break

            block_id = self._free_pool.pop()
            self._used_blocks.add(block_id)
            self._ref_count[block_id] = 1
            self._hash_to_block_id[block_hash] = block_id
            allocated.append(block_id)

        return allocated

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(self, block_id: int) -> None:
        """Release a block back to the free pool.

        Normal mode: decrements ref_count; returns to free pool when zero.

        TP mode: no-op (blocks are managed by QwenAttentionTP).

        Args:
            block_id: The block ID to release.
        """
        if self._tp_mode:
            return

        if block_id not in self._ref_count:
            return

        self._ref_count[block_id] -= 1
        if self._ref_count[block_id] <= 0:
            del self._ref_count[block_id]
            self._used_blocks.discard(block_id)
            self._free_pool.add(block_id)

    # ------------------------------------------------------------------
    # Free sequence — release all blocks for a sequence
    # ------------------------------------------------------------------

    def free_sequence(self, seq) -> None:
        """Release all blocks allocated to a sequence.

        Args:
            seq: Sequence whose blocks should be freed.
        """
        for block_id in seq.block_table:
            self.free(block_id)
        seq.block_table.clear()
        seq.num_blocks = 0

    # ------------------------------------------------------------------
    # Capacity query
    # ------------------------------------------------------------------

    def get_num_free_blocks(self) -> int:
        """Return the number of free blocks.

        TP mode: returns the total capacity (a very large number)
        so the Scheduler never blocks on capacity.
        """
        if self._tp_mode:
            # TP mode: always report a large capacity
            # Block management is handled internally by QwenAttentionTP
            return max(1, self._num_blocks)
        return len(self._free_pool)

    # ------------------------------------------------------------------
    # Prefix caching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(token_ids: Tuple[int, ...]) -> int:
        """Compute a hash for prefix caching.

        Uses Python builtin hash(tuple) — consistent within process,
        sufficient for single-process deployment.

        Args:
            token_ids: Tuple of token IDs for a block.

        Returns:
            Integer hash value.
        """
        return hash(token_ids)

    def may_append(self, seq) -> bool:
        """Check if a new block can be appended for the sequence.

        Formula: num_free_blocks >= 1

        Args:
            seq: The sequence to check.

        Returns:
            True if at least one free block is available.
        """
        if self._tp_mode:
            return True
        return len(self._free_pool) >= 1
