"""Python entry point for capturing the fixed W8A8 API in a CUDA/HIP Graph."""

from __future__ import annotations

from typing import Any

import torch


class W8A8GraphRunner:
    """Replay a captured W8A8 call and return its caller-owned output."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        output: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        self.graph = graph
        self.output = output
        self.stream = stream

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        torch.cuda.current_stream().wait_stream(self.stream)
        return self.output


def capture_w8a8_graph(
    api: Any,
    a: torch.Tensor,
    packed_weight: torch.Tensor,
    a_scale: torch.Tensor,
    packed_weight_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> W8A8GraphRunner:
    """Warm up, capture and return a Python-callable W8A8 Graph runner."""

    def invoke() -> None:
        returned = api.w8a8_gemm_out(
            a,
            packed_weight,
            a_scale,
            packed_weight_scale,
            out,
            workspace,
        )
        if returned.data_ptr() != out.data_ptr():
            raise RuntimeError(
                "w8a8_gemm_out must return the caller-provided out tensor"
            )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        invoke()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        invoke()
    runner = W8A8GraphRunner(graph, out, capture_stream)
    runner.replay()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return runner
