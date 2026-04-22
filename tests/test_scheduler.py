from engine.memory_pool import KVMemoryPool
from engine.scheduler import Scheduler
from engine.structs import Sequence, SequenceStatus


def make_seq(i: int, length: int) -> Sequence:
    return Sequence(request_id=f"req-{i}", input_ids=list(range(length)))


def test_prefill_fills_batch_until_memory_or_token_budget():
    pool = KVMemoryPool(num_blocks=8, block_size=4, hf_config=None, device=None, reserve_physical_kv=False)
    scheduler = Scheduler(memory_pool=pool, max_num_seqs=8, max_num_batched_tokens=64)
    for i, L in enumerate([3, 7, 9, 2, 6]):
        scheduler.add_request(make_seq(i, L))

    batch, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert len(batch) >= 1
    assert all(s.status == SequenceStatus.RUNNING_PREFILL for s in batch)
    assert len(scheduler.running) == len(batch)
    assert len(scheduler.waiting) == 5 - len(batch)

    scheduler.postprocess(batch, is_prefill=True, generated_tokens=[1] * len(batch))
    assert all(s.status == SequenceStatus.RUNNING_DECODE for s in batch)


def test_decode_never_preempts():
    pool = KVMemoryPool(num_blocks=4, block_size=4, hf_config=None, device=None, reserve_physical_kv=False)
    scheduler = Scheduler(memory_pool=pool, max_num_seqs=8, max_num_batched_tokens=64)
    scheduler.add_request(make_seq(0, 4))
    scheduler.add_request(make_seq(1, 4))

    b, _ = scheduler.schedule()
    scheduler.postprocess(b, is_prefill=True, generated_tokens=[100] * len(b))
    decode_b, is_p = scheduler.schedule()
    assert is_p is False
    assert all(s.preemptions == 0 for s in b)


def test_after_finish_memory_waiting_can_prefill():
    # 仅 2 个物理块：req-0 占 1 块后，req-1 需 2 块无法同时进首包 → req-1 留在 waiting
    pool = KVMemoryPool(num_blocks=2, block_size=4, hf_config=None, device=None, reserve_physical_kv=False)
    scheduler = Scheduler(memory_pool=pool, max_num_seqs=8, max_num_batched_tokens=64)
    scheduler.add_request(make_seq(0, 4))
    scheduler.add_request(make_seq(1, 8))

    first, _ = scheduler.schedule()
    assert len(first) == 1
    scheduler.postprocess(first, is_prefill=True, generated_tokens=[1])

    done = first[0]
    scheduler.running.remove(done)
    done.transition_to(SequenceStatus.FINISHED)
    pool.free_sequence(done)

    nxt, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert len(nxt) == 1 and nxt[0].request_id == "req-1"
