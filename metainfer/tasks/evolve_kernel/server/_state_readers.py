"""State-dir readers for the evolve-kernel task type .

Reads: iterations, charts, state-graph, kernel library, harnesses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import os

from ..orchestrator import phases as _phases


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def _read_run(state_dir: Path) -> Dict[str, Any]:
    return _load_json(state_dir / "run.json", {}) or {}


# --------------------------------------------------------------------------- #
# Iterations
# --------------------------------------------------------------------------- #


def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    iters_dir = state_dir / "iterations"
    if not iters_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(iters_dir.glob("*.json")):
        data = _load_json(p, None)
        if data is not None:
            out.append(data)
    return out


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "iterations" / f"{n:03d}.json", None)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def read_charts(state_dir: Path) -> Dict[str, Any]:
    recs = read_iterations(state_dir)
    durations = [
        {"x": r.get("iteration", 0), "y": round(r.get("duration_s", 0) or 0, 1)}
        for r in recs if r.get("duration_s")
    ]
    # Collect all perf metrics
    perf_keys = ["exec_time_ms", "speedup", "combined_score"]
    perf_series = []
    for k in perf_keys:
        series = [
            {"x": r.get("iteration", 0), "y": (r.get("perf") or {}).get(k)}
            for r in recs if r.get("perf") and k in r["perf"]
        ]
        if series:
            perf_series.append({"metric": k, "points": series})

    return {
        "durations": durations,
        "perf_series": perf_series,
        "iteration_status": [
            {
                "iteration": r.get("iteration", 0),
                "status": r.get("status", "running"),
                "goal": r.get("goal") or "",
            }
            for r in recs
        ],
    }


# --------------------------------------------------------------------------- #
# Retrospective
# --------------------------------------------------------------------------- #


def _find_retrospective_md(state_dir: Path, n: int) -> Optional[Path]:
    """Find retrospective.md for iteration n, trying multiple locations."""
    # 1) logs/<iter>/retrospective.md (where the agent writes it)
    logs_path = state_dir / "logs" / f"{n:03d}" / "retrospective.md"
    if logs_path.is_file():
        return logs_path
    # 2) state_dir / "logs" / str(n) / "retrospective.md" (unpadded variant)
    logs_path2 = state_dir / "logs" / str(n) / "retrospective.md"
    if logs_path2.is_file():
        return logs_path2
    return None


def read_retrospective(state_dir: Path, n: int) -> Dict[str, Any]:
    rec = read_iteration(state_dir, n)
    if rec is None:
        return {"has_retrospective": False, "markdown": "no such iteration",
                "path": None, "this_perf": {}, "prev_perf": {}, "iteration": n}
    prev = read_iteration(state_dir, n - 1) if n > 1 else None
    prev_perf = dict(prev.get("perf") or {}) if prev else {}
    this_perf = dict(rec.get("perf") or {})
    path_str = rec.get("retrospective_path")
    markdown = ""
    has = False
    actual_path: Optional[str] = None
    if path_str:
        p = Path(path_str)
        if p.is_file():
            try:
                markdown = p.read_text(encoding="utf-8", errors="replace")
                has = True
                actual_path = path_str
            except OSError:
                markdown = ""
    # Fallback: look for retrospective.md in logs dir (pipeline writes it there
    # but doesn't always record retrospective_path in the iteration record)
    if not has:
        found = _find_retrospective_md(state_dir, n)
        if found is not None:
            try:
                markdown = found.read_text(encoding="utf-8", errors="replace")
                has = True
                actual_path = str(found)
            except OSError:
                markdown = ""
    if not has:
        markdown = (
            f"# Iteration {n} — no retrospective available\n\n"
            f"Status: {rec.get('status', 'unknown')}\n"
        )
    return {
        "has_retrospective": has,
        "path": actual_path,
        "markdown": markdown,
        "this_perf": this_perf,
        "prev_perf": prev_perf,
        "iteration": n,
    }


# --------------------------------------------------------------------------- #
# State Graph
# --------------------------------------------------------------------------- #


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    run = _read_run(state_dir)
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome")
    last_label = run.get("last_transition_label")
    if hasattr(_phases, "graph_payload"):
        return _phases.graph_payload(current, last_outcome, last_label)
    return {"error": "phases module does not export graph_payload()"}


# --------------------------------------------------------------------------- #
# Kernel Library
# --------------------------------------------------------------------------- #


def read_optimizer_mode(state_dir: Path) -> str:
    """Read optimizer_mode from requirements.json. Returns display label or 'triton'."""
    req = _load_json(state_dir / "requirements.json", {})
    mode = req.get("optimizer_mode", "Triton (standard)")
    if isinstance(mode, list):
        mode = mode[0] if mode else "Triton (standard)"
    return mode


def read_kernel_library(workspace_dir: Path) -> Dict[str, Any]:
    path = workspace_dir / "kernel_library.json"
    if not path.is_file():
        return {"kernels": [], "size": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"kernels": [], "size": 0}
    if not isinstance(data, list):
        return {"kernels": [], "size": 0}

    # Build a lookup from kernel id → exec_time_ms for speedup calculation
    time_by_id: Dict[str, float] = {}
    for k in data:
        et = k.get("exec_time_ms", 0)
        if et > 0:
            time_by_id[k["id"]] = et

    # Enrich each kernel entry with a preview (first 20 lines) and speedup vs parent
    enriched = []
    for k in data:
        code = k.get("code", "")
        preview = "\n".join(code.splitlines()[:20]) if code else ""
        entry = {
            **k,
            "code_preview": preview,
            "code_lines": len(code.splitlines()) if code else 0,
            "speedup_vs_parent": None,
            # Headroom: extract from kernel entry or from nested dict
            "headroom": {
                "bottleneck": k.get("headroom_bottleneck"),
                "roofline_efficiency_pct": k.get("headroom_roofline_efficiency_pct",
                    # Fallback: compute from old bw_util_pct / compute_util_pct
                    max(k.get("headroom_bw_util_pct", 0), k.get("headroom_compute_util_pct", 0))),
                "headroom_pct": k.get("headroom_pct", 100 - max(k.get("headroom_bw_util_pct", 0), k.get("headroom_compute_util_pct", 0))),
                "p_max_tflops": k.get("headroom_p_max_tflops", 0),
                "p_bw_roof_tflops": k.get("headroom_p_bw_roof_tflops", 0),
                "ai_ridge": k.get("headroom_ai_ridge", 0),
                "bw_util_pct": k.get("headroom_bw_util_pct", 0),
                "compute_util_pct": k.get("headroom_compute_util_pct", 0),
                "suggestions": _parse_headroom_suggestions(k.get("headroom_suggestions_json")),
                "advice": k.get("headroom_advice"),
                "achieved_bw_gbps": k.get("headroom_achieved_bw_gbps", 0),
                "achieved_tflops": k.get("headroom_achieved_tflops", 0),
                "peak_bw_gbps": k.get("headroom_peak_bw_gbps", 0),
                "peak_tflops": k.get("headroom_peak_tflops", 0),
                "arithmetic_intensity": k.get("headroom_arithmetic_intensity", 0),
                "measured_ai": k.get("headroom_measured_ai", 0),
                "shape_label": k.get("headroom_shape_label", ""),
                "M": k.get("headroom_M", 0),
                "N": k.get("headroom_N", 0),
                "K": k.get("headroom_K", 0),
                "has_data": bool(k.get("headroom_bottleneck")),
            },
        }
        # Compute speedup vs parent
        pid = k.get("parent_id")
        child_time = k.get("exec_time_ms", 0)
        if pid and child_time > 0:
            parent_time = time_by_id.get(pid, 0)
            if parent_time > 0:
                entry["speedup_vs_parent"] = round(parent_time / child_time, 2)
        enriched.append(entry)
    return {"kernels": enriched, "size": len(enriched)}


# --------------------------------------------------------------------------- #
# Harnesses
# --------------------------------------------------------------------------- #


def read_harness(workspace_dir: Path, harness_type: str) -> Dict[str, Any]:
    """Read correctness or perf harness."""
    hdir = workspace_dir / "harnesses"
    fname = f"{harness_type}_harness.py"
    path = hdir / fname

    if not path.is_file():
        return {"exists": False, "code": "", "path": str(path)}

    try:
        code = path.read_text(encoding="utf-8")
    except OSError:
        return {"exists": False, "code": "", "path": str(path)}

    return {
        "exists": True,
        "code": code,
        "path": str(path),
        "lines": len(code.splitlines()),
    }


# --------------------------------------------------------------------------- #
# Reference Kernel
# --------------------------------------------------------------------------- #


def read_reference_kernel(workspace_dir: Path) -> Dict[str, Any]:
    path = workspace_dir / "reference" / "original_kernel.py"
    if not path.is_file():
        return {"exists": False, "code": "", "path": str(path)}

    try:
        code = path.read_text(encoding="utf-8")
    except OSError:
        return {"exists": False, "code": "", "path": str(path)}

    return {
        "exists": True,
        "code": code,
        "path": str(path),
        "lines": len(code.splitlines()),
    }


# --------------------------------------------------------------------------- #
# Kernel Lineage
# --------------------------------------------------------------------------- #


def _parse_headroom_suggestions(raw: Optional[str]) -> List[str]:
    """Parse JSON-encoded headroom suggestions list."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _auto_summary_items(kernel: Dict[str, Any], parent: Optional[Dict[str, Any]],
                        retro_md: Optional[str]) -> List[str]:
    """Generate improvement summary as a list of items from retrospective sections.

    Extracts bullet points and paragraph breaks from ``## What Changed`` and
    ``## Why Perf Moved`` sections. Falls back to metric-based summary if no
    retrospective is available.
    """
    items: List[str] = []

    if retro_md:
        lines = retro_md.splitlines()
        sections: Dict[str, List[str]] = {}
        current_section: Optional[str] = None
        for line in lines:
            if line.startswith("## "):
                current_section = line[3:].strip().lower()
                sections.setdefault(current_section, [])
            elif current_section:
                sections[current_section].append(line)

        # Process "What Changed"
        what_lines = sections.get("what changed", [])
        if what_lines:
            # Group consecutive non-empty lines into items, splitting on
            # bullet markers or double-newline gaps
            text = "\n".join(what_lines).strip()
            # Split by double-newline (paragraph breaks) or explicit bullet markers
            parts = re.split(r'\n\n+|\n(?=(?:- |\* |\d+\. ))', text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # If part contains `- ` bullets, split further
                if re.match(r'- ', part, re.MULTILINE):
                    sub = [s.strip()[2:].strip() for s in re.split(r'\n- ', part) if s.strip()]
                    items.extend(sub)
                elif re.match(r'\* ', part, re.MULTILINE):
                    sub = [s.strip()[2:].strip() for s in re.split(r'\n\* ', part) if s.strip()]
                    items.extend(sub)
                else:
                    items.append(part)

        # Process "Why Perf Moved"
        why_lines = sections.get("why perf moved", [])
        if why_lines:
            text = "\n".join(why_lines).strip()
            parts = re.split(r'\n\n+|\n(?=(?:- |\* |\d+\. ))', text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if re.match(r'- ', part, re.MULTILINE):
                    sub = [s.strip()[2:].strip() for s in re.split(r'\n- ', part) if s.strip()]
                    items.extend(sub)
                elif re.match(r'\* ', part, re.MULTILINE):
                    sub = [s.strip()[2:].strip() for s in re.split(r'\n\* ', part) if s.strip()]
                    items.extend(sub)
                else:
                    items.append(part)

    if items:
        return items

    # Fallback: metric-based summary
    if parent and parent.get("exec_time_ms", 0) > 0:
        parent_time = parent["exec_time_ms"]
        child_time = kernel.get("exec_time_ms", 0)
        if child_time > 0:
            delta = parent_time - child_time
            speedup = parent_time / child_time if child_time > 0 else 1.0
            faster = "faster" if delta > 0 else "slower"
            items.append(
                f"Exec time: {parent_time:.4f}ms → {child_time:.4f}ms "
                f"({abs(delta):.4f}ms {faster}, {speedup:.2f}× vs parent)"
            )
            return items

    if parent is None:
        items.append("Initial seed kernel — baseline for all subsequent optimizations.")
        return items

    items.append("No performance data available for comparison.")
    return items


def _auto_summary(kernel: Dict[str, Any], parent: Optional[Dict[str, Any]],
                  retro_md: Optional[str]) -> str:
    """Generate a human-readable improvement summary (legacy string format).

    Prefer ``_auto_summary_items`` for structured display; this is kept for
    backward compatibility with the string-based ``improvement.summary`` field.
    """
    items = _auto_summary_items(kernel, parent, retro_md)
    return "\n\n".join(items)


def read_kernel_lineage(workspace_dir: Path, state_dir: Path,
                        kernel_id: str) -> Dict[str, Any]:
    """Build lineage info for one kernel: parent, improvement metrics, retrospective."""
    lib_data = read_kernel_library(workspace_dir)
    kernels = lib_data.get("kernels", [])

    # Find the target kernel
    target: Optional[Dict[str, Any]] = None
    for k in kernels:
        if k.get("id") == kernel_id:
            target = k
            break

    if target is None:
        return {"error": f"kernel {kernel_id[:8]}… not found in library"}

    # Find parent
    parent: Optional[Dict[str, Any]] = None
    parent_id = target.get("parent_id")
    if parent_id:
        for k in kernels:
            if k.get("id") == parent_id:
                parent = k
                break

    # Build ancestor chain
    ancestors: List[str] = []
    current_pid = target.get("parent_id")
    seen: set = set()
    while current_pid and current_pid not in seen:
        seen.add(current_pid)
        ancestors.append(current_pid)
        # Find parent's parent
        found_parent = False
        for k in kernels:
            if k.get("id") == current_pid:
                current_pid = k.get("parent_id")
                found_parent = True
                break
        if not found_parent:
            break

    # Improvement metrics
    parent_time = parent.get("exec_time_ms", 0) if parent else 0
    child_time = target.get("exec_time_ms", 0)
    improvement: Dict[str, Any] = {
        "exec_time_delta_ms": round(child_time - parent_time, 6) if parent_time > 0 else None,
        "speedup_vs_parent": round(parent_time / child_time, 4) if parent_time > 0 and child_time > 0 else None,
        "complexity_delta": round(target.get("complexity_score", 0) - (parent.get("complexity_score", 0) if parent else 0), 2),
        "summary": "",
    }

    # Try retrospective for the iteration this kernel was added
    iter_num = target.get("iteration_added", 0)
    retro_md: Optional[str] = None
    if iter_num > 0:
        retro = read_retrospective(state_dir, iter_num)
        if retro.get("has_retrospective"):
            retro_md = retro.get("markdown", "")

    improvement["summary"] = _auto_summary(target, parent, retro_md)
    improvement["summary_items"] = _auto_summary_items(target, parent, retro_md)

    # Headroom data
    headroom = {
        "bottleneck": target.get("headroom_bottleneck"),
        "headroom_pct": target.get("headroom_pct", 0),
        "bw_util_pct": target.get("headroom_bw_util_pct", 0),
        "compute_util_pct": target.get("headroom_compute_util_pct", 0),
        "suggestions": _parse_headroom_suggestions(target.get("headroom_suggestions_json")),
        "advice": target.get("headroom_advice"),
        "achieved_bw_gbps": target.get("headroom_achieved_bw_gbps", 0),
        "achieved_tflops": target.get("headroom_achieved_tflops", 0),
        "peak_bw_gbps": target.get("headroom_peak_bw_gbps", 0),
        "peak_tflops": target.get("headroom_peak_tflops", 0),
        "arithmetic_intensity": target.get("headroom_arithmetic_intensity", 0),
        "has_data": bool(target.get("headroom_bottleneck")),
    }

    return {
        "kernel": target,
        "parent": parent,
        "improvement": improvement,
        "retrospective": {
            "has_retrospective": retro_md is not None,
            "markdown": retro_md or "",
            "iteration": iter_num,
        },
        "ancestor_chain": ancestors,
        "headroom": headroom,
    }


# --------------------------------------------------------------------------- #
# Failure Log
# --------------------------------------------------------------------------- #

PHASE_LABEL_MAP = {
    "A_gen_correctness_harness": "A: Gen Correctness Harness",
    "B_review_correctness_harness": "B: Review Correctness Harness",
    "C_gen_perf_harness": "C: Gen Perf Harness",
    "D_review_perf_harness": "D: Review Perf Harness",
    "E_select_kernel": "E: Select Kernel",
    "F_optimize": "F: Optimize",
    "G_verify_correctness": "G: Verify Correctness",
    "H_measure_perf": "H: Measure Perf",
}


def _read_correctness_failure_detail(state_dir: Path, n: int) -> Optional[str]:
    """Read detailed correctness failure from logs dir."""
    log_path = state_dir / "logs" / f"{n:03d}" / "correctness_failure.json"
    if not log_path.is_file():
        return None
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None

    lines: List[str] = []
    error_msg = data.get("error", "")
    if error_msg:
        lines.append(f"Error: {error_msg}")

    results = data.get("results", [])
    if results:
        lines.append(f"\nFailed test cases ({len(results)}):")
        for r in results:
            if isinstance(r, dict):
                name = r.get("name", r.get("test", "unknown"))
                passed = r.get("passed", False)
                err = r.get("error", "")
                status = "✓" if passed else "✗"
                lines.append(f"  {status} {name}")
                if err and not passed:
                    lines.append(f"    {err}")

    return "\n".join(lines) if lines else None


def read_failures(state_dir: Path) -> Dict[str, Any]:
    """Extract all failures from iteration records.

    Returns a list of failure entries, each with:
      - iteration, phase, phase_label, outcome, severity, summary, detail, attempts, timestamp
    """
    all_iters = read_iterations(state_dir)
    failures: List[Dict[str, Any]] = []

    for rec in all_iters:
        iter_num = rec.get("iteration", 0)
        phases = rec.get("phases", {})

        # 1) Iteration-level failures
        if rec.get("status") == "failed" and rec.get("failure_reason"):
            failures.append({
                "iteration": iter_num,
                "phase": None,
                "phase_label": "Iteration",
                "outcome": rec.get("outcome", "failed"),
                "severity": "error",
                "summary": _truncate(rec["failure_reason"], 120),
                "detail": rec["failure_reason"],
                "attempts": 1,
                "timestamp": rec.get("ended_at", rec.get("started_at", 0)),
            })

        # 2) Phase-level failures (including recovered ones where failure text persists)
        for phase_id, pdata in phases.items():
            if not isinstance(pdata, dict):
                continue
            failure_text = pdata.get("failure")
            if not failure_text:
                continue
            outcome = pdata.get("outcome", "unknown")
            attempts = pdata.get("attempts", 1)

            # Determine severity: error if final outcome is not ok, warning if recovered
            severity = "error" if outcome != "ok" else "warning"
            phase_label = PHASE_LABEL_MAP.get(phase_id, phase_id)

            # For correctness failures, try to read detailed test results
            detail = failure_text
            if phase_id == "G_verify_correctness":
                detail_extra = _read_correctness_failure_detail(state_dir, iter_num)
                if detail_extra:
                    detail = failure_text + "\n\n" + detail_extra

            failures.append({
                "iteration": iter_num,
                "phase": phase_id,
                "phase_label": phase_label,
                "outcome": outcome,
                "severity": severity,
                "summary": _truncate(failure_text, 120),
                "detail": detail,
                "attempts": attempts,
                "timestamp": pdata.get("ended_at", rec.get("started_at", 0)),
            })

    # Sort newest first
    failures.sort(key=lambda f: f["timestamp"], reverse=True)

    # Summary counts
    error_count = sum(1 for f in failures if f["severity"] == "error")
    warning_count = sum(1 for f in failures if f["severity"] == "warning")

    return {
        "failures": failures,
        "total": len(failures),
        "errors": error_count,
        "warnings": warning_count,
    }


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, adding ellipsis if needed."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Shape Benchmark
# --------------------------------------------------------------------------- #

from ._shape_bench import (
    parse_shapes_from_extra_notes,
    run_shape_benchmark,
    shape_results_to_dict,
    load_cached_benchmark,
    save_cached_benchmark,
)
from ._multi_gpu import (
    aggregate_shape_benchmarks,
    aggregate_best_kernels,
)


def read_shape_benchmark(state_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
    """Read or run shape benchmarks comparing best kernel vs reference.

    Returns cached results if available and best kernel hasn't changed.
    Otherwise runs benchmarks on all target shapes — this is slow (~30s-2min).
    """
    # 1. Find best kernel
    lib_path = workspace_dir / "kernel_library.json"
    if not lib_path.is_file():
        return {"results": [], "error": "No kernel library yet", "running": False}

    try:
        lib_data = json.loads(lib_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"results": [], "error": "Failed to read kernel library", "running": False}

    if not lib_data:
        return {"results": [], "error": "Kernel library is empty", "running": False}

    # Sort by exec_time_ms ascending
    lib_data.sort(key=lambda k: k.get("exec_time_ms", float("inf")))
    best_kernel = lib_data[0]
    best_id = best_kernel.get("id", "")
    best_iter = best_kernel.get("iteration_added", 0)

    # 2. Find reference kernel
    ref_path = workspace_dir / "reference" / "original_kernel.py"
    if not ref_path.is_file():
        return {"results": [], "error": "No reference kernel found", "running": False}

    # 3. Find best kernel code
    best_kernel_path = None
    # Try iteration dir first
    iter_path = workspace_dir / f"{best_iter:03d}" / "optimized_kernel.py"
    if iter_path.is_file():
        best_kernel_path = iter_path
    else:
        # Try shared kernels dir
        shared = workspace_dir / "optimized_kernels" / f"{best_id}.py"
        if shared.is_file():
            best_kernel_path = shared
    if best_kernel_path is None:
        return {"results": [], "error": f"Best kernel file not found (id={best_id[:8]})", "running": False}

    # 4. Check cache
    cache_path = workspace_dir / "shape_bench.json"
    cached = load_cached_benchmark(cache_path, best_id)
    if cached is not None:
        return {
            "results": cached,
            "best_kernel_id": best_id,
            "best_exec_time_ms": best_kernel.get("exec_time_ms", 0),
            "cached": True,
            "running": False,
        }

    # 5. Parse shapes from requirements
    req_path = state_dir / "requirements.json"
    extra_notes = ""
    kernel_fn_name = "matmul_int8"
    if req_path.is_file():
        try:
            req = json.loads(req_path.read_text(encoding="utf-8"))
            extra_notes = req.get("extra_notes", "")
            kernel_fn_name = req.get("kernel_function_name", "") or kernel_fn_name
        except (json.JSONDecodeError, OSError):
            pass

    shapes = parse_shapes_from_extra_notes(extra_notes)

    # 6. Run benchmarks
    try:
        results = run_shape_benchmark(
            ref_kernel_path=str(ref_path),
            best_kernel_path=str(best_kernel_path),
            kernel_fn_name=kernel_fn_name,
            shapes=shapes,
        )
    except Exception as e:
        return {"results": [], "error": f"Benchmark failed: {e}", "running": False,
                "best_kernel_id": best_id, "best_exec_time_ms": best_kernel.get("exec_time_ms", 0)}

    result_dicts = shape_results_to_dict(results)

    # 7. Cache
    save_cached_benchmark(cache_path, best_id, result_dicts)

    return {
        "results": result_dicts,
        "best_kernel_id": best_id,
        "best_exec_time_ms": best_kernel.get("exec_time_ms", 0),
        "cached": False,
        "running": False,
    }


def refresh_shape_benchmark(state_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
    """Force re-run shape benchmarks, ignoring cache."""
    cache_path = workspace_dir / "shape_bench.json"
    if cache_path.is_file():
        cache_path.unlink()
    return read_shape_benchmark(state_dir, workspace_dir)


# --------------------------------------------------------------------------- #
# Multi-GPU status
# --------------------------------------------------------------------------- #


def read_gpu_status(state_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
    """Read live status of all GPU workers from gpu_N/ subdirectories.

    Scans state_dir for gpu_0/, gpu_1/, etc. and returns each worker's
    current phase, iteration, exec_time, and running status.
    """
    workers: List[Dict[str, Any]] = []

    # Check parent run.json for multi-GPU metadata
    parent_run = _load_json(state_dir / "run.json", {})
    is_multi = parent_run.get("multi_gpu") or parent_run.get("multi_gpu_mode")

    if not is_multi:
        # Also check requirements
        req = _load_json(state_dir / "requirements.json", {})
        is_multi = req.get("multi_gpu") in ("All GPUs (auto)", "yes", "true", "1")

    if not is_multi:
        return {"workers": [], "is_multi_gpu": False}

    # Scan gpu_N directories
    for entry in sorted(state_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("gpu_"):
            continue
        try:
            gpu_idx = int(entry.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Read run.json
        run = _load_json(entry / "run.json", {})
        phase = run.get("current_phase", "idle")
        iteration = run.get("current_iteration", 0)

        # Check process liveness via PID file
        pid_data = _load_json(entry / "orchestrator.pid", {})
        pid = pid_data.get("pid", 0)
        running = False
        if pid and pid_data.get("finished_at") is None:
            try:
                os.kill(pid, 0)
                running = True
            except OSError:
                running = False

        # Read best time from per-GPU library
        exec_time_ms = 0.0
        speedup = 0.0
        kernel_count = 0
        best_kernel_id_short = ""
        gpu_workspace = workspace_dir / entry.name
        lib_path = gpu_workspace / "kernel_library.json"
        if lib_path.is_file():
            try:
                lib = json.loads(lib_path.read_text(encoding="utf-8"))
                kernel_count = len(lib)
                if lib:
                    lib.sort(key=lambda k: k.get("exec_time_ms", float("inf")))
                    best = lib[0]
                    exec_time_ms = best.get("exec_time_ms", 0)
                    best_kernel_id_short = best.get("id", "")[:8]
                    # Find seed (iteration_added=0) for speedup calc
                    for k in lib:
                        if k.get("iteration_added") == 0:
                            seed_time = k.get("exec_time_ms", 0)
                            if seed_time > 0 and exec_time_ms > 0:
                                speedup = seed_time / exec_time_ms
                            break
            except Exception:
                pass

        # Read agents from per-GPU agents.json
        agents_data = _load_json(entry / "agents.json", {"ts": 0, "agents": []})
        agents_summary: List[Dict[str, Any]] = []
        for a in agents_data.get("agents", []):
            if not isinstance(a, dict):
                continue
            agents_summary.append({
                "name": a.get("name", ""),
                "role": a.get("role", ""),
                "phase": a.get("phase", "unknown"),
                "elapsed_s": round(a.get("elapsed_s", 0), 1),
                "attempt": a.get("attempt", 1),
                "success": a.get("success"),
                "error": a.get("error"),
            })

        # Read shapes from requirements
        gpu_req = _load_json(entry / "requirements.json", {})
        shapes_notes = gpu_req.get("extra_notes", "")[:200]

        workers.append({
            "gpu_idx": gpu_idx,
            "label": f"GPU {gpu_idx}",
            "phase": phase,
            "iteration": iteration,
            "exec_time_ms": round(exec_time_ms, 4) if exec_time_ms else 0,
            "speedup": round(speedup, 2) if speedup else 0,
            "running": running,
            "pid": pid,
            "shapes": shapes_notes,
            "agents": agents_summary,
            "kernel_count": kernel_count,
            "best_kernel_id": best_kernel_id_short,
        })

    return {
        "workers": workers,
        "is_multi_gpu": True,
        "num_workers": len(workers),
    }


def read_aggregate_bench(state_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
    """Merge shape benchmarks from all per-GPU workspaces."""
    wds: List[str] = []
    for entry in sorted(workspace_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("gpu_"):
            wds.append(str(entry))

    if not wds:
        # Try parent workspace directly
        bench_path = workspace_dir / "shape_bench.json"
        if bench_path.is_file():
            try:
                data = json.loads(bench_path.read_text(encoding="utf-8"))
                return {
                    "results": data.get("results", []),
                    "kernels": [],
                    "is_multi_gpu": True,
                    "num_workers": 0,
                }
            except Exception:
                pass
        return {"results": [], "kernels": [], "is_multi_gpu": False}

    results = aggregate_shape_benchmarks(wds)
    kernels = aggregate_best_kernels(wds)

    return {
        "results": results,
        "kernels": kernels,
        "is_multi_gpu": True,
        "num_workers": len(wds),
    }


# --------------------------------------------------------------------------- #
# Combined timeline (across all GPU workers + parent)
# --------------------------------------------------------------------------- #

from metainfer.server.state_reader import read_timeline as _read_timeline_file


# --------------------------------------------------------------------------- #
# Per-GPU detail (state graph + kernel library for one GPU worker)
# --------------------------------------------------------------------------- #


def _gpu_state_dir(state_dir: Path, gpu_idx: int) -> Path:
    return state_dir / f"gpu_{gpu_idx}"


def _gpu_workspace_dir(workspace_dir: Path, gpu_idx: int) -> Path:
    return workspace_dir / f"gpu_{gpu_idx}"


def read_gpu_state_graph(state_dir: Path, gpu_idx: int) -> Dict[str, Any]:
    """Read state graph for a single GPU worker."""
    gpu_sd = _gpu_state_dir(state_dir, gpu_idx)
    run = _load_json(gpu_sd / "run.json", {})
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome")
    last_label = run.get("last_transition_label")
    if hasattr(_phases, "graph_payload"):
        return _phases.graph_payload(current, last_outcome, last_label)
    return {"error": "phases module does not export graph_payload()"}


def read_gpu_kernel_library(workspace_dir: Path, gpu_idx: int) -> Dict[str, Any]:
    """Read kernel library for a single GPU worker."""
    return read_kernel_library(_gpu_workspace_dir(workspace_dir, gpu_idx))


def read_gpu_detail(state_dir: Path, workspace_dir: Path, gpu_idx: int) -> Dict[str, Any]:
    """Aggregated per-GPU detail: state graph + kernel library + agents + phase info.

    Returns everything needed to render the single-GPU view for one worker.
    """
    gpu_sd = _gpu_state_dir(state_dir, gpu_idx)
    run = _load_json(gpu_sd / "run.json", {})

    # State graph
    graph = read_gpu_state_graph(state_dir, gpu_idx)

    # Kernel library
    library = read_gpu_kernel_library(workspace_dir, gpu_idx)

    # Agents
    agents_data = _load_json(gpu_sd / "agents.json", {"ts": 0, "agents": []})
    agents_summary: List[Dict[str, Any]] = []
    for a in agents_data.get("agents", []):
        if not isinstance(a, dict):
            continue
        agents_summary.append({
            "name": a.get("name", ""),
            "role": a.get("role", ""),
            "phase": a.get("phase", "unknown"),
            "elapsed_s": round(a.get("elapsed_s", 0), 1),
            "attempt": a.get("attempt", 1),
            "success": a.get("success"),
            "error": a.get("error"),
        })

    # Iteration data
    iteration = run.get("current_iteration", 0)
    phase = run.get("current_phase", "idle")

    # Check liveness
    pid_data = _load_json(gpu_sd / "orchestrator.pid", {})
    pid = pid_data.get("pid", 0)
    running = False
    if pid and pid_data.get("finished_at") is None:
        try:
            import os as _os
            _os.kill(pid, 0)
            running = True
        except OSError:
            running = False

    # Best perf from library
    exec_time_ms = 0.0
    speedup = 0.0
    kernel_count = 0
    if library.get("kernels"):
        kernels = library["kernels"]
        kernel_count = len(kernels)
        sorted_kernels = sorted(kernels, key=lambda k: k.get("exec_time_ms", float("inf")))
        best = sorted_kernels[0]
        exec_time_ms = best.get("exec_time_ms", 0)
        for k in kernels:
            if k.get("iteration_added") == 0:
                seed_time = k.get("exec_time_ms", 0)
                if seed_time > 0 and exec_time_ms > 0:
                    speedup = seed_time / exec_time_ms
                break

    # Read shapes from per-GPU requirements
    gpu_req = _load_json(gpu_sd / "requirements.json", {})
    shapes_notes = gpu_req.get("extra_notes", "")

    # Read optimizer mode from per-GPU requirements
    optimizer_mode = gpu_req.get("optimizer_mode", "Triton (standard)")
    if isinstance(optimizer_mode, list):
        optimizer_mode = optimizer_mode[0] if optimizer_mode else "Triton (standard)"

    return {
        "gpu_idx": gpu_idx,
        "label": f"GPU {gpu_idx}",
        "phase": phase,
        "iteration": iteration,
        "running": running,
        "pid": pid,
        "exec_time_ms": round(exec_time_ms, 4) if exec_time_ms else 0,
        "speedup": round(speedup, 2) if speedup else 0,
        "kernel_count": kernel_count,
        "shapes": shapes_notes,
        "graph": graph,
        "library": library,
        "agents": agents_summary,
        "optimizer_mode": optimizer_mode,
    }


def read_gpu_harness(workspace_dir: Path, gpu_idx: int, harness_type: str) -> Dict[str, Any]:
    """Read correctness or perf harness for a single GPU worker."""
    return read_harness(_gpu_workspace_dir(workspace_dir, gpu_idx), harness_type)


def read_gpu_kernel_lineage(workspace_dir: Path, state_dir: Path,
                            gpu_idx: int, kernel_id: str) -> Dict[str, Any]:
    """Read kernel lineage for a single GPU worker."""
    return read_kernel_lineage(
        _gpu_workspace_dir(workspace_dir, gpu_idx),
        _gpu_state_dir(state_dir, gpu_idx),
        kernel_id,
    )


def read_combined_timeline(state_dir: Path, since: float = 0.0) -> List[Dict[str, Any]]:
    """Merge timeline events from parent + all gpu_N/ subdirectories.

    Each event gets a ``source`` field (``"parent"`` or ``"gpu_N"``) so
    the frontend can color-code or filter by source.
    """
    all_events: List[Dict[str, Any]] = []

    # Parent timeline
    parent_events = _read_timeline_file(state_dir, since)
    for ev in parent_events:
        ev["source"] = "parent"
        all_events.append(ev)

    # Per-GPU timelines
    for entry in sorted(state_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("gpu_"):
            continue
        gpu_events = _read_timeline_file(entry, since)
        for ev in gpu_events:
            ev["source"] = entry.name
            all_events.append(ev)

    # Sort by timestamp
    all_events.sort(key=lambda e: e.get("ts", 0))
    return all_events


def read_summary(state_dir: Path, workspace_dir: Path) -> Dict[str, Any]:
    """Read the end-of-task summary report.

    Looks for:
      1. ``summary.txt`` in state_dir (multi-GPU orchestrator writes this)
      2. ``summary`` field in ``run.json``
      3. Falls back to auto-generating from kernel libraries
    """
    # Method 1: summary.txt file
    summary_path = state_dir / "summary.txt"
    if summary_path.is_file():
        return {
            "report": summary_path.read_text(encoding="utf-8"),
            "source": "summary.txt",
        }

    # Method 2: summary field in run.json
    run = _load_json(state_dir / "run.json", {})
    if run.get("summary"):
        return {
            "report": run["summary"],
            "source": "run.json",
        }

    # Method 3: auto-generate from available data
    gpu_summaries: List[Dict[str, Any]] = []
    for gpu_id in range(8):  # max 8 GPUs
        gpu_sd = state_dir / f"gpu_{gpu_id}"
        run_path = gpu_sd / "run.json"
        if not run_path.is_file():
            break
        gpu_run = _load_json(run_path, {})
        gpu_ws = workspace_dir / f"gpu_{gpu_id}"
        lib_path = gpu_ws / "kernel_library.json"

        seed_ms = 0.0
        best_ms = 0.0
        kernel_count = 0
        if lib_path.is_file():
            try:
                lib = json.loads(lib_path.read_text(encoding="utf-8"))
                kernel_count = len(lib)
                if lib:
                    best = min(lib, key=lambda k: k.get("exec_time_ms", float("inf")))
                    best_ms = best.get("exec_time_ms", 0)
                    for k in lib:
                        if k.get("iteration_added") == 0:
                            seed_ms = k.get("exec_time_ms", 0)
                            break
            except Exception:
                pass

        gpu_summaries.append({
            "gpu": gpu_id,
            "phase": gpu_run.get("current_phase", "?"),
            "status": gpu_run.get("final_status", "running"),
            "seed_ms": seed_ms,
            "best_ms": best_ms,
            "speedup": round(seed_ms / best_ms, 2) if seed_ms > 0 and best_ms > 0 else 0,
            "kernels": kernel_count,
        })

    # Build a simple text report
    lines = ["## Auto-generated summary (task still running or summary not yet written)", ""]
    lines.append("| GPU | Seed (ms) | Best (ms) | Speedup | Kernels | Status |")
    lines.append("|-----|----------|----------|---------|---------|--------|")
    for s in gpu_summaries:
        lines.append(
            f"| GPU {s['gpu']} | {s['seed_ms']:.4f} | {s['best_ms']:.4f} | "
            f"{s['speedup']:.2f}× | {s['kernels']} | {s['status']} |"
        )

    return {
        "report": "\n".join(lines),
        "source": "auto-generated",
        "gpu_summaries": gpu_summaries,
    }
