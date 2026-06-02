"""
Phase 8 — Request-level state container (Sequence) and status enum.
Physically independent from engine/models/. Uses dual-track block_table:
  - HF path: list[int] via block_table
  - TP path: torch.Tensor [1, max_blocks] int32 via block_table_tensor()
"""

from enum import Enum
from typing import Optional
import torch


class SeqStatus(Enum):
    WAITING = "WAITING"
    RUNNING_PREFILL = "RUNNING_PREFILL"
    RUNNING_DECODE = "RUNNING_DECODE"
    FINISHED = "FINISHED"
    REJECTED = "REJECTED"


class Sequence:
    """Request-level state container.

    block_table dual-track:
      - self.block_table: list[int] — HF path, dynamic growth
      - self._block_table_tensor: Tensor[1, max_blocks] int32 — TP path, lazy init

    Both representations exist simultaneously. The caller selects which to use
    based on inference_backend (HF → block_table list; TP → block_table_tensor()).
    """

    def __init__(self, request_id, input_ids, block_size=256, max_model_len=40960,
                 max_blocks=None, device=None):
        self.request_id = request_id
        self.input_ids = list(input_ids)
        self.output_ids = []
        self.status = SeqStatus.WAITING
        self.block_size = block_size
        if max_blocks is None:
            max_blocks = (max_model_len + block_size - 1) // block_size
        self._max_blocks = max_blocks
        self._device = device
        # dual-track block_table
        self.block_table = []            # list[int], HF path, dynamic growth
        self._block_table_tensor = None  # Tensor[1, max_blocks] int32, lazy init
        self.kv_len = 0                  # cached KV length (updated post-prefill)
        # sampling params (set by LLMEngine._enqueue)
        self.max_tokens = 0
        self.temperature = 0.0
        self.top_p = 1.0
        self.ignore_eos = False

    # ---- core properties ----

    def seq_len(self):
        return len(self.input_ids)

    def required_blocks(self):
        return (self.seq_len() + self.block_size - 1) // self.block_size

    def input_ids_tensor(self, device=None):
        return torch.tensor([self.input_ids], dtype=torch.long, device=device)

    # ---- status helpers ----

    def is_finished(self):
        return self.status in (SeqStatus.FINISHED, SeqStatus.REJECTED)

    def is_waiting(self):
        return self.status == SeqStatus.WAITING

    def is_running(self):
        return self.status in (SeqStatus.RUNNING_PREFILL, SeqStatus.RUNNING_DECODE)

    def transition_to(self, new_status: SeqStatus):
        """Centralized state transition with validation."""
        valid_transitions = {
            SeqStatus.WAITING: (SeqStatus.RUNNING_PREFILL, SeqStatus.REJECTED),
            SeqStatus.RUNNING_PREFILL: (SeqStatus.RUNNING_DECODE, SeqStatus.FINISHED, SeqStatus.REJECTED),
            SeqStatus.RUNNING_DECODE: (SeqStatus.FINISHED, SeqStatus.REJECTED),
            SeqStatus.FINISHED: (),
            SeqStatus.REJECTED: (),
        }
        assert new_status in valid_transitions[self.status], \
            f"Invalid status transition: {self.status.value} → {new_status.value}"
        self.status = new_status

    # ---- dual-track block_table accessors ----

    def block_table_tensor(self):
        """TP Runner path: return Tensor[1, max_blocks] int32 on device."""
        if self._block_table_tensor is None:
            if self._device is None:
                raise RuntimeError("Sequence block_table_tensor() requires device to be set.")
            self._block_table_tensor = torch.zeros(
                1, self._max_blocks, dtype=torch.int32, device=self._device)
        return self._block_table_tensor

    def block_table_list(self):
        """HF Runner path: return list[int]."""
        return self.block_table

    # ---- token helpers ----

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    @property
    def num_completion_tokens(self):
        return len(self.output_ids)

    def __repr__(self):
        return (f"Sequence(id={self.request_id}, status={self.status.value}, "
                f"input_len={self.seq_len()}, output_len={len(self.output_ids)}, "
                f"kv_len={self.kv_len}, blocks={self.block_table})")
