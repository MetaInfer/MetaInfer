"""Trace parser tests with synthetic trace fixtures."""

import json
from ..orchestrator.trace_parser import aggregate_kernels


def _synthetic_trace(kernels):
    """Build a minimal Chrome trace JSON."""
    events = []
    for name, dur, extra in kernels:
        evt = {"cat": "kernel", "name": name, "ph": "X", "dur": dur, "ts": 0}
        if extra:
            evt.setdefault("args", {}).update(extra)
        events.append(evt)
    return {"traceEvents": events}


def test_aggregate_empty_trace():
    result = aggregate_kernels(_synthetic_trace([]))
    assert result == []


def test_aggregate_single_kernel():
    trace = _synthetic_trace([("triton_gemm", 1000, {})])
    result = aggregate_kernels(trace)
    assert len(result) == 1
    assert result[0]["kernel_name"] == "triton_gemm"
    assert result[0]["total_dur_us"] == 1000
    assert result[0]["count"] == 1


def test_aggregate_multiple_same_kernel():
    trace = _synthetic_trace([
        ("triton_gemm", 500, {}),
        ("triton_gemm", 700, {}),
        ("flash_attn", 300, {}),
    ])
    result = aggregate_kernels(trace)
    assert len(result) == 2
    # triton_gemm aggregates: 500 + 700 = 1200
    assert result[0]["kernel_name"] == "triton_gemm"
    assert result[0]["total_dur_us"] == 1200
    assert result[0]["count"] == 2
    # flash_attn is second
    assert result[1]["kernel_name"] == "flash_attn"
    assert result[1]["total_dur_us"] == 300


def test_aggregate_ignores_non_kernel():
    trace = _synthetic_trace([
        ("triton_gemm", 500, {}),
        ("cpu_op", 200, {}),  # different cat
    ])
    # Make the second event non-kernel
    trace["traceEvents"][1]["cat"] = "cpu_op"
    result = aggregate_kernels(trace)
    assert len(result) == 1
    assert result[0]["kernel_name"] == "triton_gemm"


def test_aggregate_includes_call_stack():
    trace = _synthetic_trace([
        ("triton_gemm", 500, {"call stack": "model.layers.5.self_attn"}),
    ])
    result = aggregate_kernels(trace, include_call_stack=True)
    assert result[0]["call_stack"] == "model.layers.5.self_attn"
