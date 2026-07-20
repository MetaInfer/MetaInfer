"""Test helpers for find-low-hanging-kernel."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def make_requirements(
    task_id: str = "flhk-1",
    *,
    form: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_form = {
        "trace_file": "/tmp/fake/trace.json",
        "model_dir": "/tmp/fake/model",
        "framework_source_dir": "/tmp/fake/framework",
        "cli_args_and_env": "",
        "startup_log": "",
        "max_validator_rounds": 5,
    }
    if form:
        base_form.update(form)
    return {
        "task_id": task_id,
        "task_type": "find-low-hanging-kernel",
        "created_at": 0.0,
        "form": base_form,
    }


def make_event(
    name: str, cat: str, ts: int, dur: int, pid: int = 0, tid: int = 0,
    ph: str = "X", args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    e = {"name": name, "cat": cat, "ph": ph, "ts": ts, "pid": pid, "tid": tid}
    if ph == "X":
        e["dur"] = dur
    if args is not None:
        e["args"] = args
    return e


def write_trace(
    path: Path, events: Iterable[Dict[str, Any]], *, gzipped: bool = False
) -> Path:
    """Write a chrome-trace shaped JSON file (with ``traceEvents`` wrapper).
    Pass ``gzipped=True`` to write a .json.gz instead."""
    payload = {"traceEvents": list(events)}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload)
    if gzipped:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def make_small_trace_events() -> List[Dict[str, Any]]:
    """A small, realistic-ish trace with three kernel cats, a CPU event, and
    a flow event. Returns ~10 events."""
    return [
        # Two GPU kernels with the SAME name but different durs → bimodal.
        make_event("rms_norm_kernel", "kernel", ts=100, dur=20, pid=0, tid=1),
        make_event("rms_norm_kernel", "kernel", ts=200, dur=80, pid=0, tid=1),
        make_event("rms_norm_kernel", "kernel", ts=300, dur=22, pid=0, tid=1),
        # One CUDA runtime call.
        make_event("cudaLaunchKernel", "cuda_runtime", ts=150, dur=5, pid=0, tid=2),
        # CPU-side stack event (so has_cpu_stack should be True).
        make_event("RMSNorm.forward", "cpu", ts=99, dur=2, pid=0, tid=3),
        # Flow event for correlation.
        {"name": "M2F", "cat": "__metadata", "ph": "f", "ts": 105, "pid": 0, "tid": 1, "id": 1},
        # A B/E pair that exercises pairing logic.
        {"name": "gemm_kernel", "cat": "kernel", "ph": "B", "ts": 400, "pid": 0, "tid": 1},
        {"name": "gemm_kernel", "cat": "kernel", "ph": "E", "ts": 460, "pid": 0, "tid": 1},
    ]


def make_minimal_valid_graph() -> Dict[str, Any]:
    """A graph that passes the integrity check."""
    return {
        "schema_version": 1,
        "metadata": {
            "task_id": "t1",
            "model": "TestModel",
            "tp_size": 1,
            "vars": {"B": "batch", "M": "seq_len"},
        },
        "nodes": [
            {
                "id": "n01",
                "role": "entry",
                "operator": "embedding_kernel",
                "source_ref": {"file": "embed.py", "line": 10, "symbol": "Embed.forward"},
                "inputs": [{"name": "ids", "dtype": "int32", "shape": ["B", "M"]}],
                "outputs": [{"name": "x", "dtype": "fp16", "shape": ["B", "M", 4096]}],
                "stats": {"count": 4, "mean_us": 5.0, "std_us": 1.0, "total_us": 20.0, "p99_us": 6.0},
                "confidence": "high",
            },
            {
                "id": "n02",
                "role": "RMSNorm",
                "operator": "rms_norm_kernel",
                "source_ref": {"file": "norm.py", "line": 30, "symbol": "RMSNorm.forward"},
                "inputs": [{"name": "x", "dtype": "fp16", "shape": ["B", "M", 4096]}],
                "outputs": [{"name": "y", "dtype": "fp16", "shape": ["B", "M", 4096]}],
                "stats": {"count": 12, "mean_us": 30.0, "std_us": 5.0, "total_us": 360.0, "p99_us": 40.0},
                "confidence": "high",
            },
            {
                "id": "n03",
                "role": "exit",
                "operator": "logits_kernel",
                "source_ref": {"file": "head.py", "line": 50, "symbol": "Head.forward"},
                "inputs": [{"name": "h", "dtype": "fp16", "shape": ["B", "M", 4096]}],
                "outputs": [{"name": "logits", "dtype": "fp16", "shape": ["B", "M", 32000]}],
                "stats": {"count": 2, "mean_us": 50.0, "std_us": 3.0, "total_us": 100.0, "p99_us": 52.0},
                "confidence": "medium",
            },
        ],
        "edges": [
            {"from": "n01", "to": "n02", "label": "x"},
            {"from": "n02", "to": "n03", "label": "h"},
        ],
    }
