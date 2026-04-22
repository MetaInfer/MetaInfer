"""
Paged block 管理 + 前缀哈希共享（参考 nano-vllm BlockManager）。
哈希链：整块 token 的 xxhash 风格链式前缀（prefix 为上一块 hash）。
"""

from __future__ import annotations

import hashlib
from collections import deque

from engine.structs import Sequence


class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int = -1
        self.token_ids: list[int] = []

    def update(self, h: int, token_ids: list[int]) -> None:
        self.hash = h
        self.token_ids = list(token_ids)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        """链式前缀哈希；prefix 为上一整块 digest 的 64-bit 值（与 nano-vllm intdigest 范围一致）。"""
        h = hashlib.blake2b(digest_size=16)
        if prefix != -1:
            h.update((prefix & ((1 << 64) - 1)).to_bytes(8, "little", signed=False))
        for t in token_ids:
            h.update(t.to_bytes(4, "little", signed=False))
        digest = h.digest()[:8]
        return int.from_bytes(digest, "little", signed=False)

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        if seq.block_table:
            raise ValueError("allocate expects empty block_table")
        if not self.can_allocate(seq):
            raise MemoryError(
                f"Not enough KV blocks: need>={seq.num_blocks}, free={len(self.free_block_ids)}"
            )

        h = -1
        cache_miss = False
        for _i in range(seq.num_blocks):
            token_ids = seq.block(_i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 与 nano-vllm 一致：仅当新 token 落入新块时需要多占一块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence) -> None:
        block_table = seq.block_table
        if not block_table:
            return
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
