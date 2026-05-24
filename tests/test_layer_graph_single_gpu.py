"""Stage 2: Single-layer CUDA Graph capture + 10K replay stress (single GPU, no comm).

Loads layer from runner (CUDA_GRAPH=0 to skip runner's own compile), then
applies torch.compile + CUDAGraphWrapper manually — same pipeline as runner
but isolated to a single layer for DFT testing.
"""
import os, time, torch
import pytest

os.environ['META_INFER_CUDA_GRAPH'] = '0'  # skip runner compile; we do it manually

from engine.tp_layers.distributed import init_tp_distributed
from engine.tp_layers.cuda_graph_wrapper import CUDAGraphWrapper

MODEL_DIR = '/home/honglin/models/qwen/Qwen3-8B'


def _load_layer():
    """Load first QwenDecoderLayerTP via QwenTPModelRunner (CUDA_GRAPH=0)."""
    from engine.models.qwen import QwenTPModelRunner
    runner = QwenTPModelRunner(MODEL_DIR, device=torch.device('cuda:0'), dtype=torch.bfloat16)
    return runner.model.layers[0], runner.cfg


def _setup_kv_cache(attn, seq_len=4, max_seq_len=128):
    block_size = 256
    num_blocks = max(1, (max_seq_len + block_size - 1) // block_size)
    attn._key_cache = torch.zeros(num_blocks, block_size, attn.num_kv_heads,
                                   attn.head_dim, device='cuda:0', dtype=torch.bfloat16)
    attn._value_cache = torch.zeros(num_blocks, block_size, attn.num_kv_heads,
                                     attn.head_dim, device='cuda:0', dtype=torch.bfloat16)
    attn._block_table = torch.arange(num_blocks, dtype=torch.int32, device='cuda:0').unsqueeze(0)
    attn._kv_len_gpu[0] = seq_len
    if attn._cos_sin_cache_gpu is None:
        attn._cos_sin_cache_gpu = attn._cos_sin_cache_cpu.to(device='cuda:0')


class TestSingleLayerCUDAGraph:
    """Stage 2: Single-layer CUDA Graph isolation + 10K replay stress.

    Uses the same torch.compile + CUDAGraphWrapper pipeline as the runner —
    just isolated to a single layer. Compilation happens eagerly BEFORE
    CUDAGraphWrapper captures the graph (vLLM-aligned order).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        init_tp_distributed()
        self.layer, self.cfg = _load_layer()
        self.attn = self.layer.self_attn
        _setup_kv_cache(self.attn, seq_len=4, max_seq_len=128)

        self.hidden = torch.randn(1, 1, self.cfg.hidden_size, device='cuda:0', dtype=torch.bfloat16)
        self.residual = torch.randn(1, 1, self.cfg.hidden_size, device='cuda:0', dtype=torch.bfloat16)
        self.pos = torch.tensor([4], device='cuda:0', dtype=torch.long)

        # vLLM-aligned: compile eagerly FIRST, then wrap with CUDAGraphWrapper
        self.compiled = torch.compile(
            self.layer.forward_decode, fullgraph=True, dynamic=False,
        )
        # Eager warmup — triggers Dynamo + inductor (must NOT be inside torch.cuda.graph)
        self.attn._kv_len_gpu[0] = 4
        self.compiled(self.hidden, self.pos, 4, 128, self.residual)
        torch.cuda.synchronize()

        # Now wrap + capture
        self.wrapper = CUDAGraphWrapper(self.compiled, debug_mode=False)
        self.attn._kv_len_gpu[0] = 4
        self.wrapper(self.hidden, self.pos, 4, 128, self.residual)
        torch.cuda.synchronize()

    def _reset_kv(self):
        self.attn._kv_len_gpu[0] = 4

    # ---- tests ----

    def test_001_capture_succeeded(self):
        """CUDA Graph capture completed."""
        assert self.wrapper.is_captured, "Graph should be captured"

    def test_002_replay_matches_eager(self):
        """Replay output matches eager within rtol=1e-2."""
        self._reset_kv()
        eager_hs, eager_res = self.layer.forward_decode(
            self.hidden, self.pos, 4, 128, self.residual.clone(),
        )
        self._reset_kv()
        graph_hs, graph_res = self.wrapper(
            self.hidden, self.pos, 4, 128, self.residual,
        )
        torch.testing.assert_close(graph_hs, eager_hs, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(graph_res, eager_res, rtol=1e-2, atol=1e-2)

    def test_003_health_check_no_sync(self):
        """check_graph_health returns GPU tensors (no .item() sync)."""
        self._reset_kv()
        self.wrapper(self.hidden, self.pos, 4, 128, self.residual)
        health = self.wrapper.check_graph_health()
        for k, v in health.items():
            if 'has_nan' in k or 'has_inf' in k:
                assert isinstance(v, torch.Tensor), f"{k} should be Tensor, got {type(v)}"
                assert v.device.type == 'cuda'

    def test_004_10k_replay_no_crash(self):
        """10,000 consecutive replays without crash."""
        for i in range(10000):
            self._reset_kv()
            out = self.wrapper(self.hidden, self.pos, 4, 128, self.residual)
            if i % 2500 == 0:
                health = self.wrapper.check_graph_health()
                for k, v in health.items():
                    if isinstance(v, torch.Tensor) and ('has_nan' in k or 'has_inf' in k):
                        assert not v, f"{k} at step {i}"
        assert out is not None

    def test_005_profiler_launch_count(self):
        """torch.profiler confirms cudaGraphLaunch during replay."""
        self._reset_kv()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            self.wrapper(self.hidden, self.pos, 4, 128, self.residual)
        count = sum(1 for e in prof.key_averages() if 'cudaGraphLaunch' in e.key)
        assert count > 0, f"Expected cudaGraphLaunch, got {count}"

    def test_006_probes_show_launches(self):
        """Probes report graph launches after multiple replays."""
        for _ in range(100):
            self._reset_kv()
            self.wrapper(self.hidden, self.pos, 4, 128, self.residual)
        summary = self.wrapper.probes.summary()
        assert 'launches=' in summary
