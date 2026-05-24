"""CUDAGraphWrapper — vLLM PIECEWISE CUDA Graph lifecycle manager.

Extracted and simplified from vllm/compilation/cuda_graph.py:145-356.
Stripped: forward_context, BatchDescriptor, VllmConfig, CUDAGraphMode,
offloader, graph_pool global management, weak_ref_tensors.
Preserved: core capture/replay state machine, input address validation,
non-blocking NaN/Inf health checks.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch


# ---- data classes ----

@dataclasses.dataclass
class CUDAGraphEntry:
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None
    input_addresses: list[int] | None = None


class CUDAGraphProfilingProbes:
    """Non-invasive CUDA Graph statistics."""

    def __init__(self) -> None:
        self.graph_launch_count: int = 0
        self.graph_capture_count: int = 0

    def record_launch(self) -> None:
        self.graph_launch_count += 1

    def record_capture(self) -> None:
        self.graph_capture_count += 1

    def summary(self) -> str:
        return (
            f"CUDAGraph: captures={self.graph_capture_count}, "
            f"launches={self.graph_launch_count}"
        )


# ---- main wrapper ----

class CUDAGraphWrapper:
    """Wraps a runnable (compiled forward_decode) with lazy CUDA Graph capture + replay.

    State machine (aligned with vLLM :233-356):
      __call__(*args):
        ├─ entry 不存在？ → _capture() → 缓存 entry → 返回 output
        └─ entry 已存在？ → 校验 input_addresses (debug) → replay → 返回 entry.output

    The runnable is expected to be a torch.compile(fullgraph=True)-wrapped
    forward_decode.  Inductor manages buffer allocation; this wrapper manages
    the CUDA Graph lifecycle.
    """

    def __init__(self, runnable: Callable[..., Any], debug_mode: bool = False) -> None:
        self.runnable = runnable
        self.debug_mode = debug_mode
        self._entry: CUDAGraphEntry | None = None
        self._probes = CUDAGraphProfilingProbes()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._entry is None or self._entry.cudagraph is None:
            return self._capture(*args, **kwargs)

        if self.debug_mode:
            new_addrs = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_addrs == self._entry.input_addresses, (
                f"Input address mismatch during replay. "
                f"Expected {self._entry.input_addresses}, got {new_addrs}"
            )

        self._entry.cudagraph.replay()
        self._probes.record_launch()
        return self._entry.output

    def _capture(self, *args: Any, **kwargs: Any) -> Any:
        input_addrs = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
        cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cudagraph):
            output = self.runnable(*args, **kwargs)
        self._entry = CUDAGraphEntry(
            cudagraph=cudagraph, output=output, input_addresses=input_addrs,
        )
        self._probes.record_capture()
        return output

    # ---- health check (non-blocking, no host-device sync) ----

    def check_graph_health(self) -> dict[str, Any]:
        """Non-blocking NaN/Inf detection.

        NEVER calls .item() or .cpu() — these trigger host-device sync and
        will crash CUDA Graph capture with 'operation not permitted'.
        """
        if torch.cuda.is_current_stream_capturing():
            return {"healthy": True, "reason": "skipped_during_capture"}
        if self._entry is None or self._entry.output is None:
            return {"healthy": True, "reason": "no_graph_yet"}

        out = self._entry.output
        tensors = list(out) if isinstance(out, (tuple, list)) else [out]
        tensors = [t for t in tensors if isinstance(t, torch.Tensor)]

        flags: dict[str, Any] = {}
        for i, t in enumerate(tensors):
            flags[f"out_{i}_has_nan"] = torch.isnan(t).any()   # GPU tensor — no sync
            flags[f"out_{i}_has_inf"] = torch.isinf(t).any()
        return flags

    # ---- lifecycle ----

    @property
    def is_captured(self) -> bool:
        return self._entry is not None and self._entry.cudagraph is not None

    def clear_graph(self) -> None:
        self._entry = None

    @property
    def probes(self) -> CUDAGraphProfilingProbes:
        return self._probes
