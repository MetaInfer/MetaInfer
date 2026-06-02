"""
Phase 8 — BlockManager: paged block allocator with TP degradation.

Nano-vllm overrides applied:
  - TP degradation via self._tp_mode flag (not subclassing).
  - allocate() / free() are no-op when _tp_mode=True.
  - get_num_free_blocks() always returns len(_free_pool).
  - compute_hash uses Python builtin hash(tuple(token_ids)) per blueprint.
  - may_append checks num_free >= 1.

Blueprint _tp_degradation_fork_interface:
  LLMEngine.__init__ injects tp_mode=True for qwen_tp/deepseek_tp backends.
  TP path: actual block_table allocated by QwenAttentionTP torch.arange.
"""


class BlockManager:
    """Paged block allocator with optional TP degradation.

    Normal mode (tp_mode=False): maintains free_pool, ref_count, prefix caching.
    TP mode (tp_mode=True): allocate/free are no-op; get_num_free_blocks still works.
    """

    def __init__(self, num_blocks, block_size=16, tp_mode=False):
        self._block_size = block_size
        self._tp_mode = tp_mode
        self._free_pool = set(range(num_blocks))
        self._ref_count = {}       # dict[int, int] — block_id → ref_count
        self._hash_to_block = {}   # dict[int, int] — hash → block_id (prefix caching)

    # ---- allocation ----

    def allocate(self, seq, num_blocks):
        """Allocate `num_blocks` blocks for a sequence.

        Normal mode: pops from free_pool, sets ref_count=1, returns block IDs.
        TP mode: returns list(range(num_blocks)) as placeholder (no-op).
        """
        if self._tp_mode:
            return list(range(num_blocks))  # placeholder allocation

        blocks = []
        for _ in range(num_blocks):
            if not self._free_pool:
                raise RuntimeError(f"BlockManager: no free blocks (requested {num_blocks})")
            bid = self._free_pool.pop()
            self._ref_count[bid] = 1
            blocks.append(bid)
        return blocks

    # ---- deallocation ----

    def free(self, block_id):
        """Decrement ref_count for a block. Return to free_pool when ref_count=0.

        TP mode: no-op.
        """
        if self._tp_mode:
            return

        if block_id not in self._ref_count:
            return
        self._ref_count[block_id] -= 1
        if self._ref_count[block_id] <= 0:
            self._ref_count.pop(block_id, None)
            self._free_pool.add(block_id)

    # ---- capacity queries ----

    def get_num_free_blocks(self):
        """Return number of free blocks. Works in both normal and TP modes."""
        return len(self._free_pool)

    def can_allocate(self, seq):
        """Check if enough free blocks exist for a sequence's required blocks."""
        return self.get_num_free_blocks() >= seq.required_blocks()

    def may_append(self, seq):
        """Check if at least 1 free block exists for decode extension."""
        return self.get_num_free_blocks() >= 1

    def can_append(self, seq):
        """Alias for may_append — legacy compatibility."""
        return self.may_append(seq)

    # ---- prefix caching (disabled in TP) ----

    def compute_hash(self, token_ids):
        """Compute hash for prefix caching.

        Uses Python builtin hash(tuple(token_ids)) per blueprint specification.
        nano-vllm uses xxhash; blueprint overrides to Python hash for simplicity.
        """
        return hash(tuple(token_ids))

    def lookup_prefix(self, token_ids):
        """Look up a block_id by token prefix hash.

        Returns block_id or None. TP mode always returns None (caching disabled).
        """
        if self._tp_mode:
            return None
        h = self.compute_hash(token_ids)
        return self._hash_to_block.get(h, None)

    def cache_prefix(self, token_ids, block_id):
        """Register a block under its prefix hash."""
        if self._tp_mode:
            return
        h = self.compute_hash(token_ids)
        self._hash_to_block[h] = block_id
