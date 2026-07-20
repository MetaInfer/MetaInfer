"""Unit tests for trace_parser."""

from __future__ import annotations

import json
from pathlib import Path

from metainfer.tasks.find_low_hanging_kernel.orchestrator import trace_parser
from metainfer.tasks.find_low_hanging_kernel.tests._helpers import (
    make_event,
    make_small_trace_events,
    write_trace,
)


def test_parse_plain_json(tmp_path: Path):
    p = tmp_path / "trace.json"
    write_trace(p, make_small_trace_events())
    summary = trace_parser.parse_trace(p)

    assert summary["source"] == "trace.json"
    assert summary["event_count"] == 8
    assert summary["has_cpu_stack"] is True
    assert summary["cuda_graph_detected"] is False
    assert "__metadata" in summary["cats_present"]
    assert "kernel" in summary["cats_present"]

    # rms_norm_kernel aggregated the 3 X-events (bimodal).
    rms = next(
        r for r in summary["by_name_cat"]
        if r["name"] == "rms_norm_kernel" and r["cat"] == "kernel"
    )
    assert rms["count"] == 3
    assert rms["total_us"] == 122.0  # 20 + 80 + 22
    assert rms["bimodal_suspect"] is True

    # gemm_kernel was a B/E pair spanning ts=400..460 → dur=60.
    gemm = next(
        r for r in summary["by_name_cat"]
        if r["name"] == "gemm_kernel" and r["cat"] == "kernel"
    )
    assert gemm["count"] == 1
    assert gemm["total_us"] == 60.0


def test_parse_gzipped(tmp_path: Path):
    p = tmp_path / "trace.json.gz"
    write_trace(p, make_small_trace_events(), gzipped=True)
    summary = trace_parser.parse_trace(p)
    assert summary["event_count"] == 8


def test_write_summary_roundtrip(tmp_path: Path):
    src = tmp_path / "in.json"
    write_trace(src, make_small_trace_events())
    out = tmp_path / "out" / "trace_parsed.json"
    summary = trace_parser.write_summary(src, out)
    assert out.is_file()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["event_count"] == summary["event_count"]


def test_by_name_cat_sorted_by_total(tmp_path: Path):
    events = [
        make_event("a", "kernel", ts=0, dur=10),
        make_event("b", "kernel", ts=0, dur=1000),
        make_event("c", "kernel", ts=0, dur=100),
    ]
    p = tmp_path / "t.json"
    write_trace(p, events)
    summary = trace_parser.parse_trace(p)
    totals = [r["total_us"] for r in summary["by_name_cat"]]
    assert totals == sorted(totals, reverse=True)


def test_cuda_graph_detection(tmp_path: Path):
    events = [
        make_event("normal_kernel", "kernel", ts=0, dur=10),
        {"name": "cuda_graph_capture", "cat": "cuda_runtime", "ph": "X", "ts": 0, "dur": 1, "pid": 0, "tid": 0},
    ]
    p = tmp_path / "t.json"
    write_trace(p, events)
    summary = trace_parser.parse_trace(p)
    assert summary["cuda_graph_detected"] is True


def test_no_cpu_stack_when_absent(tmp_path: Path):
    events = [make_event("a", "kernel", ts=0, dur=10)]
    p = tmp_path / "t.json"
    write_trace(p, events)
    summary = trace_parser.parse_trace(p)
    assert summary["has_cpu_stack"] is False


def test_accepts_bare_list(tmp_path: Path):
    """Some trace emitters skip the ``traceEvents`` wrapper."""
    p = tmp_path / "bare.json"
    p.write_text(json.dumps(make_small_trace_events()), encoding="utf-8")
    summary = trace_parser.parse_trace(p)
    assert summary["event_count"] == 8
