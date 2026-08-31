from __future__ import annotations

from pathlib import Path


def test_python_graph_wrapper_is_staged_control_plane_code():
    wrapper = (
        Path(__file__).resolve().parent.parent
        / "assets" / "w8a8_baseline" / "w8a8_graph.py"
    )
    text = wrapper.read_text(encoding="utf-8")
    assert "def capture_w8a8_graph(" in text
    assert "api.w8a8_gemm_out(" in text
    assert "torch.cuda.CUDAGraph()" in text
    assert "torch.cuda.graph(graph, stream=capture_stream)" in text
    assert "def replay(self) -> torch.Tensor:" in text


def test_trusted_harness_requires_graph_replay_timing():
    harness = (
        Path(__file__).resolve().parent.parent
        / "assets" / "w8a8_bench.py"
    )
    text = harness.read_text(encoding="utf-8")
    assert "capture_candidate_graph(candidate, out)" in text
    assert '"timing_mode": "cuda_graph_replay"' in text
    assert "graph_runner.replay()" in text
