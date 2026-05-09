import torch

from engine.memory_pool import KVMemoryPool
from engine.model_runner import FakeCausalLM, ModelRunner
from engine.scheduler import Scheduler
from engine.sampler import greedy_sample, top_p_sample
from engine.structs import Sequence, SequenceStatus


def test_greedy_sample_matches_argmax():
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])
    out = greedy_sample(logits)
    assert out.tolist() == [1, 2]


def test_top_p_sample_respects_distribution():
    torch.manual_seed(0)
    # Sharp distribution: token 0 dominates
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    out = top_p_sample(logits, top_p=0.95)
    assert out.item() == 0


def test_fake_causal_lm_output_shape():
    m = FakeCausalLM(vocab_size=32, hidden_size=16)
    x = torch.tensor([1, 2, 3, 4])
    y = m(x)
    assert y.shape == (4, 32)


def test_model_runner_prefill_and_decode_shapes():
    runner = ModelRunner(vocab_size=50, hidden_size=32, seed=123)
    seq_a = Sequence(request_id="a", input_ids=[10, 11, 12])
    seq_a.num_cached_tokens = 0
    seq_b = Sequence(request_id="b", input_ids=[5, 6])
    seq_b.num_cached_tokens = 0

    toks_prefill = runner.run([seq_a, seq_b], is_prefill=True, temperature=0.0)
    assert len(toks_prefill) == 2
    assert all(0 <= t < 50 for t in toks_prefill)

    # Simulate post-prefill: cache filled, decode one step from last prompt token
    seq_a.num_cached_tokens = len(seq_a.input_ids)
    seq_b.num_cached_tokens = len(seq_b.input_ids)
    toks_decode = runner.run([seq_a, seq_b], is_prefill=False, temperature=0.0)
    assert len(toks_decode) == 2


def test_scheduler_batch_end_to_end_greedy():
    pool = KVMemoryPool(num_blocks=16, block_size=4)
    sched = Scheduler(memory_pool=pool, max_num_seqs=8, max_num_batched_tokens=128)
    runner = ModelRunner(vocab_size=100, hidden_size=48, seed=7)

    seqs = [
        Sequence(request_id="r0", input_ids=[1, 2, 3]),
        Sequence(request_id="r1", input_ids=[4, 5, 6, 7, 8]),
    ]
    for s in seqs:
        sched.add_request(s)

    batch, is_prefill = sched.schedule()
    assert is_prefill is True
    assert len(batch) == 2
    assert all(s.status == SequenceStatus.RUNNING_PREFILL for s in batch)

    next_tokens = runner.run(batch, is_prefill=True, temperature=0.0)
    assert len(next_tokens) == len(batch)

    sched.postprocess(batch, is_prefill=True)
    assert all(s.status == SequenceStatus.RUNNING_DECODE for s in batch)

    decode_batch, is_dec = sched.schedule()
    assert is_dec is False
    assert len(decode_batch) == 2
    next_decode = runner.run(decode_batch, is_prefill=False, temperature=0.0)
    assert len(next_decode) == len(decode_batch)
