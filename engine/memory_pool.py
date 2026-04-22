"""
真实 KVMemoryPool：分页 BlockManager（前缀哈希共享）+ 按 HF MLA 维度估算的 GPU KV 占位张量。
MoE 不产生 KV；占位用于预留显存并与 nano-vllm「整块 buffer」思路对齐。
"""

from __future__ import annotations

from typing import Any

import torch

from engine.block_manager import BlockManager
from engine.kv_specs import hf_deepseek_v2_kv_bytes_per_block, hf_deepseek_v2_kv_bytes_per_token
from engine.structs import Sequence


class KVMemoryPool:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        hf_config: Any | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
        reserve_physical_kv: bool = True,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self.hf_config = hf_config
        self.dtype = dtype
        self.device = device

        self._manager = BlockManager(num_blocks, block_size)

        # 与 HF MLA 每 token 维数对齐的一维占位（便于 OOM 前预留；未接入自定义 kernel 写回）
        self.kv_bytes_per_token: int | None = None
        self.kv_storage: torch.Tensor | None = None
        if hf_config is not None and reserve_physical_kv and device is not None and device.type == "cuda":
            self.kv_bytes_per_token = hf_deepseek_v2_kv_bytes_per_token(hf_config, dtype)
            total_bytes = num_blocks * block_size * self.kv_bytes_per_token
            elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
            numel = max(1, total_bytes // elem)
            # 与预算一致的逻辑块数已用于 BlockManager；此处占位张量仅作显存对齐样例，避免再申请与逻辑容量 1:1 的巨型张量导致 OOM
            cap_numel = min(numel, (512 * 1024**2) // elem)
            self.kv_storage = torch.empty(cap_numel, dtype=dtype, device=device)
            print(
                f"[KVMemoryPool] KV placeholder tensor: numel={cap_numel:,} (~{cap_numel * elem / 1024**2:.0f}MiB), "
                f"logical KV bytes≈{total_bytes:,}, MLA bytes/token={self.kv_bytes_per_token}"
            )

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
        """首次为序列分配块表（含前缀缓存命中）；要求 block_table 为空。"""
        seq.block_size = self.block_size
        target = seq.total_tokens if num_tokens is None else num_tokens
        if seq.total_tokens < target:
            raise ValueError("num_tokens exceeds current sequence length")
        if seq.block_table:
            raise ValueError("block_table must be empty for allocate_for_sequence")
        self._manager.allocate(seq)
        return list(seq.block_table)

    def ensure_capacity_for_sequence(self, seq: Sequence) -> None:
        """增量扩展：每追加一个 token 后调用 may_append（与 nano-vllm 一致）。"""
        seq.block_size = self.block_size
        self._manager.may_append(seq)

    def can_append_one_more(self, seq: Sequence) -> bool:
        """再生成 1 个 token 前检查是否有足够空闲块（不抢占）。"""
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
        dtype: torch.dtype,
        free_bytes: int,
        reserve_bytes: int,
        mem_utilization: float,
    ) -> int:
        bytes_per_block = hf_deepseek_v2_kv_bytes_per_block(hf_config, dtype, block_size)
        available = max(0, int((free_bytes - reserve_bytes) * mem_utilization))
        n = max(16, available // max(1, bytes_per_block))
        return int(n)
