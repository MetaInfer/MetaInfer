import pytest

from engine.memory_pool import KVMemoryPool
from engine.structs import Sequence, SequenceStatus


def make_sequence(request_id: str, prompt_len: int) -> Sequence:
    return Sequence(request_id=request_id, input_ids=list(range(prompt_len)))


def test_allocate_blocks_success():
    pool = KVMemoryPool(num_blocks=8, block_size=4, hf_config=None, device=None, reserve_physical_kv=False)
    seq = make_sequence("req-1", prompt_len=10)

    allocated = pool.allocate_for_sequence(seq, num_tokens=10)

    assert allocated == [0, 1, 2]
    assert seq.block_table == [0, 1, 2]
    assert pool.num_free_blocks == 5


def test_allocate_blocks_reuses_freed_blocks_fifo():
    pool = KVMemoryPool(num_blocks=4, block_size=2, hf_config=None, device=None, reserve_physical_kv=False)
    seq_a = make_sequence("req-a", prompt_len=3)  # 2 blocks
    seq_b = make_sequence("req-b", prompt_len=1)  # 1 block

    pool.allocate_for_sequence(seq_a, num_tokens=3)
    pool.allocate_for_sequence(seq_b, num_tokens=1)
    pool.free_sequence(seq_a)

    # 使用与 req-a 不同的 token，避免整块前缀哈希命中复用块号（否则可能拿到已回收的 0 号块）
    seq_c = Sequence(request_id="req-c", input_ids=[10, 11])
    allocated_c = pool.allocate_for_sequence(seq_c, num_tokens=2)

    assert allocated_c == [3]
    assert pool.num_free_blocks == 2


def test_allocate_raises_when_capacity_insufficient():
    pool = KVMemoryPool(num_blocks=2, block_size=4, hf_config=None, device=None, reserve_physical_kv=False)
    seq = make_sequence("req-oom", prompt_len=9)  # needs 3 blocks

    with pytest.raises(MemoryError):
        pool.allocate_for_sequence(seq, num_tokens=9)

    assert pool.num_free_blocks == 2
    assert seq.block_table == []


def test_free_sequence_is_idempotent_and_updates_status():
    pool = KVMemoryPool(num_blocks=3, block_size=4, hf_config=None, device=None, reserve_physical_kv=False)
    seq = make_sequence("req-free", prompt_len=4)
    seq.status = SequenceStatus.RUNNING_DECODE

    pool.allocate_for_sequence(seq, num_tokens=4)
    pool.free_sequence(seq)
    pool.free_sequence(seq)

    assert seq.block_table == []
    assert pool.num_free_blocks == 3


def test_append_token_and_block_growth():
    pool = KVMemoryPool(num_blocks=4, block_size=2, hf_config=None, device=None, reserve_physical_kv=False)
    seq = make_sequence("req-grow", prompt_len=2)
    pool.allocate_for_sequence(seq, num_tokens=2)  # 1 block

    seq.append_token(99)
    pool.ensure_capacity_for_sequence(seq)
    assert seq.block_table == [0, 1]

    seq.append_token(100)
    pool.ensure_capacity_for_sequence(seq)
    assert seq.block_table == [0, 1]  # still 2 blocks

