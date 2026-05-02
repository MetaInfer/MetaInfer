"""
Apple Silicon (MPS) 内存池：基于 psutil 估算统一内存，仅做逻辑块管理。
不分配物理 KV tensor（因为 use_cache=False，每次前向全量重计算）。
"""

from __future__ import annotations

from typing import Any

from mac_gpu.block_manager import BlockManager
from mac_gpu.structs import Sequence


class MPSMemoryPool:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self._manager = BlockManager(num_blocks, block_size)

    @property
    def num_free_blocks(self) -> int:
        return len(self._manager.free_block_ids)

    def required_blocks(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence) -> bool:
        seq.block_size = self.block_size
        return self._manager.can_allocate(seq)

    def allocate_for_sequence(self, seq: Sequence, num_tokens: int | None = None) -> list[int]:
        seq.block_size = self.block_size
        target = seq.total_tokens if num_tokens is None else num_tokens
        if seq.total_tokens < target:
            raise ValueError("num_tokens exceeds current sequence length")
        if seq.block_table:
            raise ValueError("block_table must be empty for allocate_for_sequence")
        self._manager.allocate(seq)
        return list(seq.block_table)

    def ensure_capacity_for_sequence(self, seq: Sequence) -> None:
        seq.block_size = self.block_size
        self._manager.may_append(seq)

    def can_append_one_more(self, seq: Sequence) -> bool:
        seq.block_size = self.block_size
        next_len = seq.total_tokens + 1
        need = self.required_blocks(next_len)
        have = len(seq.block_table)
        missing = need - have
        if missing <= 0:
            return True
        return len(self._manager.free_block_ids) >= missing

    def free_sequence(self, seq: Sequence) -> None:
        if not seq.block_table:
            return
        self._manager.deallocate(seq)

    @staticmethod
    def estimate_num_blocks(
        hf_config: Any,
        *,
        block_size: int,
        model_bytes: int,
        mem_utilization: float = 0.80,
    ) -> int:
        """用系统可用内存估算 KV 块数量（Apple Silicon 统一内存）。"""
        import psutil

        available = psutil.virtual_memory().available
        budget = max(0, int((available - model_bytes - 2 * 1024**3) * mem_utilization))

        head_dim = int(
            getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        )
        kv_heads = int(getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads))
        layers = int(hf_config.num_hidden_layers)
        # float16: 2 bytes per element; K+V: factor 2
        bytes_per_token = layers * kv_heads * head_dim * 2 * 2
        bytes_per_block = bytes_per_token * block_size
        return max(16, budget // max(bytes_per_block, 1))
