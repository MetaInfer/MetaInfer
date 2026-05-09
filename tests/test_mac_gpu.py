"""Mac GPU (MPS) 引擎测试。仅在 Apple Silicon Mac 上运行。"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS tests require Apple Silicon Mac",
)


def test_mps_device_available():
    device = torch.device("mps")
    t = torch.tensor([1.0, 2.0], device=device)
    assert t.device.type == "mps"


def test_structs_and_block_manager():
    """数据结构和块管理是纯 Python，不需要 GPU。"""
    from engine.block_manager import BlockManager
    from engine.structs import Sequence, SequenceStatus

    bm = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence(request_id="test", input_ids=[1, 2, 3, 4, 5])
    seq.block_size = 4
    bm.allocate(seq)
    assert len(seq.block_table) == 2
    assert bm.blocks[seq.block_table[0]].ref_count == 1
    bm.deallocate(seq)
    assert len(seq.block_table) == 0

    seq2 = Sequence(request_id="t2", input_ids=[10, 20])
    seq2.block_size = 4
    assert seq2.status == SequenceStatus.WAITING
    seq2.transition_to(SequenceStatus.RUNNING_PREFILL)
    seq2.transition_to(SequenceStatus.RUNNING_DECODE)
    assert seq2.status == SequenceStatus.RUNNING_DECODE


def test_sampler():
    """采样器在 MPS tensor 上工作。"""
    from engine.sampler import sample_next_tokens

    logits = torch.zeros(1, 100, device="mps")
    logits[0, 42] = 10.0
    tokens = sample_next_tokens(logits, temperature=0.0)
    assert tokens.item() == 42


def test_memory_pool():
    from engine.mac_gpu.memory_pool import MPSMemoryPool
    from engine.structs import Sequence

    pool = MPSMemoryPool(num_blocks=32, block_size=4)
    seq = Sequence(request_id="test", input_ids=[1, 2, 3, 4, 5, 6])
    assert pool.can_allocate(seq)
    pool.allocate_for_sequence(seq, seq.total_tokens)
    assert len(seq.block_table) > 0
    assert pool.num_free_blocks < 32
    pool.free_sequence(seq)
    assert pool.num_free_blocks == 32


def test_scheduler():
    from engine.mac_gpu.memory_pool import MPSMemoryPool
    from engine.mac_gpu.scheduler import Scheduler
    from engine.structs import Sequence, SequenceStatus

    pool = MPSMemoryPool(num_blocks=32, block_size=4)
    sched = Scheduler(memory_pool=pool, max_num_seqs=4, max_num_batched_tokens=512)

    seq = Sequence(request_id="test", input_ids=[1, 2, 3, 4, 5])
    sched.add_request(seq)
    batch, is_prefill = sched.schedule()
    assert len(batch) == 1
    assert is_prefill is True
    assert batch[0].status == SequenceStatus.RUNNING_PREFILL


def test_model_load_and_prefill():
    """加载小模型并运行 prefill（需要联网下载）。"""
    from engine.mac_gpu.model_runner import MPSModelRunner
    from engine.structs import Sequence

    runner = MPSModelRunner("Qwen/Qwen2.5-0.5B")
    seq = Sequence(
        request_id="test",
        input_ids=runner.tokenizer.encode("Hello", add_special_tokens=True),
        sampling_params={"temperature": 0.0},
    )
    tokens = runner.run_prefill([seq])
    assert len(tokens) == 1
    assert isinstance(tokens[0], int)
    assert tokens[0] >= 0


def test_end_to_end():
    """端到端推理测试（需要联网下载模型）。"""
    from engine.mac_gpu.engine import MacGPUEngine

    engine = MacGPUEngine("Qwen/Qwen2.5-0.5B", block_size=16, max_num_seqs=2)
    output = engine.generate("What is 2+2?", max_new_tokens=16, temperature=0.0)
    assert isinstance(output, str)
    assert len(output.strip()) > 0
