"""
Phase 8 — Sequence: request-level state container.

Maintains input/output tokens, block_table (list+Tensor dual-track),
status machine transitions, and block-level views.

All signatures must match inference_blueprint.json
  > components[5] Sequence
  > data_flow_contracts.request_level.sequence_fields
"""

import enum
from typing import List, Optional

import torch


# ===========================================================================
# SequenceStatus — state machine enum
# ===========================================================================

class SequenceStatus(enum.Enum):
    WAITING = 0
    PREFILL = 1
    DECODE = 2
    FINISHED = 3
    REJECTED = 4


# ===========================================================================
# Sequence — request-level state container
# ===========================================================================

class Sequence:
    """Request-level state container with dual-track block_table.

    block_table: Python list[int] (HF path) + torch.Tensor [1, max_blocks] int32 (TP path).
    Both representations are initialized simultaneously. The caller selects which
    one to use based on inference_backend — Sequence itself is a passive data dictionary.

    Attributes:
        seq_id: Unique sequence identifier.
        input_ids: Original prompt token ids.
        output_ids: Generated token ids.
        block_table: Python list[int] — logical block IDs (HF path).
        block_table_tensor: torch.Tensor [1, max_blocks] int32 (TP path, lazy init).
        status: Current SequenceStatus.
        prompt_len: Length of the original prompt (cached at construction).
        max_output_len: Maximum number of tokens to generate.
        num_blocks: Number of allocated KV cache blocks.
        block_size: KV cache block size (injected by caller).
        kv_len: Current KV cache length (updated by runner).
    """

    _next_seq_id = 0

    def __init__(
        self,
        input_ids: List[int],
        max_output_len: int = 256,
        seq_id: Optional[int] = None,
        block_size: int = 256,
        max_blocks: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        if seq_id is None:
            seq_id = Sequence._next_seq_id
            Sequence._next_seq_id += 1

        self.seq_id = seq_id
        self.input_ids = list(input_ids)
        self.output_ids: List[int] = []
        self.status = SequenceStatus.WAITING
        self.prompt_len = len(input_ids)
        self.max_output_len = max_output_len
        self.num_blocks = 0
        self.block_size = block_size
        self.kv_len = 0

        # Dual-track block_table
        self.block_table: List[int] = []
        self._max_blocks = max_blocks
        self._device = device
        self._block_table_tensor: Optional[torch.Tensor] = None

        # Sampling params (for postprocess finish checks)
        self.sampling_params = {"max_tokens": max_output_len}

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def transition_to(self, status: SequenceStatus) -> None:
        """State transition: WAITING → PREFILL → DECODE → FINISHED/REJECTED."""
        self.status = status

    # ------------------------------------------------------------------
    # Block table — dual-track access
    # ------------------------------------------------------------------

    def block_table_list(self) -> List[int]:
        """Return block table as list[int] — HF Runner path."""
        return self.block_table

    def block_table_tensor(self) -> torch.Tensor:
        """Return block table as Tensor [1, max_blocks] int32 — TP Runner path.

        Lazily initializes a zero-filled tensor if not yet created.
        """
        if self._block_table_tensor is None:
            if self._max_blocks is None:
                raise ValueError(
                    "block_table_tensor() requires max_blocks to be set at construction. "
                    "Pass max_blocks and device to Sequence.__init__()."
                )
            self._block_table_tensor = torch.zeros(
                1, self._max_blocks, dtype=torch.int32, device=self._device
            )
        return self._block_table_tensor

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    def input_ids_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return input_ids as torch.LongTensor.

        If device is provided, the tensor is placed on that device.
        This is critical for TP Runner which must pass device=self.device
        to avoid CPU/GPU tensor mismatch in torch.cat().
        """
        t = torch.tensor(self.input_ids, dtype=torch.long)
        if device is not None:
            t = t.to(device)
        return t

    def seq_len(self) -> int:
        """Total sequence length: prompt_len + generated tokens so far."""
        return self.prompt_len + len(self.output_ids)

    def required_blocks(self) -> int:
        """Number of KV cache blocks needed for the current sequence.

        Formula: ceil(seq_len / block_size)
        """
        return (self.seq_len() + self.block_size - 1) // self.block_size

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, status={self.status.name}, "
            f"seq_len={self.seq_len()}, prompt_len={self.prompt_len}, "
            f"output_len={len(self.output_ids)}, num_blocks={self.num_blocks})"
        )
