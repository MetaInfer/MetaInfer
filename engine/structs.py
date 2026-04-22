from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING_PREFILL = "running_prefill"
    RUNNING_DECODE = "running_decode"
    FINISHED = "finished"


@dataclass
class Sequence:
    request_id: str
    input_ids: list[int]
    sampling_params: dict[str, Any] = field(default_factory=dict)
    status: SequenceStatus = SequenceStatus.WAITING
    output_ids: list[int] = field(default_factory=list)
    block_size: int = 16
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    preemptions: int = 0
    past_key_values: Any = None

    @property
    def token_ids(self) -> list[int]:
        return self.input_ids + self.output_ids

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    def __len__(self) -> int:
        return self.total_tokens

    @property
    def block_ids(self) -> list[int]:
        """兼容旧字段名，与 block_table 相同。"""
        return self.block_table

    @property
    def num_blocks(self) -> int:
        if self.total_tokens == 0:
            return 0
        return (self.total_tokens + self.block_size - 1) // self.block_size

    def block(self, i: int) -> list[int]:
        if not (0 <= i < self.num_blocks):
            raise IndexError(f"block index {i} out of range for num_blocks={self.num_blocks}")
        start = i * self.block_size
        end = min(start + self.block_size, self.total_tokens)
        return self.token_ids[start:end]

    @property
    def last_block_num_tokens(self) -> int:
        if self.total_tokens == 0:
            return 0
        return self.total_tokens - (self.num_blocks - 1) * self.block_size

    def append_token(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    def transition_to(self, new_status: SequenceStatus) -> None:
        allowed: dict[SequenceStatus, set[SequenceStatus]] = {
            SequenceStatus.WAITING: {SequenceStatus.RUNNING_PREFILL},
            SequenceStatus.RUNNING_PREFILL: {SequenceStatus.RUNNING_DECODE},
            SequenceStatus.RUNNING_DECODE: {SequenceStatus.FINISHED},
            SequenceStatus.FINISHED: set(),
        }
        if new_status not in allowed[self.status]:
            raise ValueError(f"Invalid status transition: {self.status} -> {new_status}")
        self.status = new_status
