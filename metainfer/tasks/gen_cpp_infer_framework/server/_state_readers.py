"""State-dir readers that used to live in the shell's ``state_reader``.

These functions read the orchestrator's on-disk iteration records,
charts payload, retrospective markdown, and state-graph payload. They
were originally task-agnostic in the shell, but the ABCDEF iteration
record belongs to this task package. Future task types that don't fit
this shape ship their own reader.

Copy-first, don't pre-abstract. If patterns crystallize across ≥3
tasks, extraction back into a shared helper is fine.
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
    """All iteration records, sorted by iteration number."""
    iters_dir = state_dir / "iterations"
    if not iters_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    stable = _load_json(state_dir / "stable_candidate.json", {}) or {}
    stable_iteration = stable.get("iteration")
    for p in sorted(iters_dir.glob("*.json")):
        data = _load_json(p, None)
        if data is not None:
            data["is_stable_candidate"] = data.get("iteration") == stable_iteration
            out.append(data)
    return out


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "iterations" / f"{n:03d}.json", None)


def read_charts(state_dir: Path) -> Dict[str, Any]:
    """Aggregate perf-per-iteration + durations for the charts panel."""
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
    """Return the retrospective payload for iteration ``n``."""
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
        status = rec.get("status")
        if status == "running":
            reason = ("This iteration is still running — no retrospective "
                      "has been produced yet.")
        elif status == "failed":
            reason = (
                "This iteration failed and the postmortem agent didn't "
                f"produce a retrospective file. Failure reason: "
                f"`{rec.get('failure_reason') or 'unknown'}`."
            )
        elif rec.get("phases"):
            reason = ("This iteration hasn't finished all phases yet, "
                      "so no retrospective was written.")
        else:
            reason = ("The retrospective agent didn't produce a file. "
                      "Check the iteration's logs directory.")
        markdown = (
            f"# Iteration {n} — no retrospective available\n\n"
            f"{reason}\n\n"
            f"## Raw perf data\n\n"
            f"- this iteration: `{this_perf or 'no data'}`\n"
            f"- previous iteration: `{prev_perf or 'no data'}`\n"
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
    """Dispatch to this task's own ``phases.graph_payload``."""
    run = _read_run(state_dir)
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome")
    last_label = run.get("last_transition_label")
    if not hasattr(_phases, "graph_payload"):
        return {"error": "C++ task phases module does not export graph_payload()"}
    return _phases.graph_payload(current, last_outcome, last_label)
