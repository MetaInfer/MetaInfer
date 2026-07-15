"""State-dir readers for the opt-kernel task type.

Same iteration record schema as gen-infer-framework (ABCDEF cycle,
perf dict, retrospective_path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def read_charts(state_dir: Path) -> Dict[str, Any]:
    recs = read_iterations(state_dir)
    durations = [
        {"x": r.get("iteration", 0), "y": round(r.get("duration_s", 0) or 0, 1)}
        for r in recs if r.get("duration_s")
    ]
    perf_keys: List[str] = []
    for r in recs:
        for k in (r.get("perf") or {}):
            if k not in perf_keys:
                perf_keys.append(k)
    perf_series = []
    for k in perf_keys:
        series = [
            {"x": r.get("iteration", 0), "y": (r.get("perf") or {}).get(k)}
            for r in recs if r.get("perf") and k in r["perf"]
        ]
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
    if path_str:
        p = Path(path_str)
        if p.is_file():
            try:
                markdown = p.read_text(encoding="utf-8", errors="replace")
                has = True
            except OSError:
                markdown = ""
    if not has:
        markdown = (
            f"# Iteration {n} — no retrospective available\n\n"
            f"Status: {rec.get('status', 'unknown')}\n"
        )
    return {
        "has_retrospective": has,
        "path": path_str,
        "markdown": markdown,
        "this_perf": this_perf,
        "prev_perf": prev_perf,
        "iteration": n,
    }


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    run = _read_run(state_dir)
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome")
    last_label = run.get("last_transition_label")
    if hasattr(_phases, "graph_payload"):
        return _phases.graph_payload(current, last_outcome, last_label)
    return {"error": "phases module does not export graph_payload()"}
