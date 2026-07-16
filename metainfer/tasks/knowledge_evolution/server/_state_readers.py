"""Task-specific on-disk state readers for knowledge-evolution.

These functions read the task-private iteration schema, state graph,
knowledge-gained records, oracle reports, and log files.
Kept separate from the shell's generic ``state_reader.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.server.state_reader import read_run


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if missing or broken."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---- Iterations ----

def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    """Read all iteration records from ``<state_dir>/iterations/``."""
    iterations_dir = state_dir / "iterations"
    if not iterations_dir.is_dir():
        return []

    records: list[Dict[str, Any]] = []
    for fpath in sorted(iterations_dir.glob("*.json")):
        data = _load_json(fpath)
        if data is not None:
            records.append(data)

    records.sort(key=lambda r: r.get("n", r.get("iteration", 0)))
    return records


# ---- State graph ----

def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    """Build the state-graph payload for the current run.

    Reads ``run.json`` to get the current phase and last transition,
    then delegates to ``phases.graph_payload()`` from the orchestrator.
    """
    run = read_run(state_dir)
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome", "")
    last_label = run.get("last_transition_label", "")

    from metainfer.tasks.knowledge_evolution.orchestrator.phases import graph_payload

    return graph_payload(current, last_outcome, last_label)


# ---- Knowledge gained ----

def read_knowledge_gained(state_dir: Path) -> Dict[str, Any]:
    """Summarize what knowledge the consolidator wrote across iterations.

    Scans ``<state_dir>/logs/<nnn>/`` for ``consolidation.json`` files,
    which the C_consolidate phase writes after updating notebooks/.
    """
    logs_dir = state_dir / "logs"
    if not logs_dir.is_dir():
        return {"iterations": []}

    entries = []
    total_files = 0
    for iter_dir in sorted(logs_dir.iterdir()):
        if not iter_dir.is_dir():
            continue
        cf = iter_dir / "consolidation.json"
        data = _load_json(cf)
        if data is None:
            # Try failure_analysis.md as a fallback indicator
            fa = iter_dir / "failure_analysis.md"
            retro = iter_dir / "retrospective.md"
            if fa.exists() and retro.exists():
                try:
                    n = int(iter_dir.name)
                except ValueError:
                    continue
                entries.append({
                    "iteration": n,
                    "status": "failed",
                    "summary": "Iteration failed; see failure analysis and retrospective.",
                    "files": [],
                })
            continue
        total_files += len(data.get("files", []))
        entries.append(data)

    entries.sort(key=lambda e: e.get("iteration", 0))
    return {"iterations": entries, "total_files": total_files}


# ---- Oracle report ----

def read_oracle_report(state_dir: Path, iteration: int) -> Dict[str, Any]:
    """Read the oracle report for a specific iteration.

    Looks for ``<state_dir>/logs/<nnn>/oracle_report.md`` or ``c_repair.md``.
    """
    candidates = [
        state_dir / "logs" / f"{iteration:03d}" / "oracle_report.md",
        state_dir / "logs" / f"{iteration:03d}" / "c_repair.md",
    ]
    for path in candidates:
        if path.exists():
            return {
                "found": True,
                "iteration": iteration,
                "markdown": path.read_text(encoding="utf-8", errors="replace"),
                "file": path.name,
            }
    return {"found": False, "iteration": iteration}


# ---- Knowledge diff ----

def read_knowledge_diff(state_dir: Path, iteration: int, file: str) -> Dict[str, Any]:
    """Read a notebook file's content from a given iteration's code/logs dir.

    Looks in ``<state_dir>/code/<nnn>/`` or ``<state_dir>/logs/<nnn>/``
    for the requested file.
    """
    candidates = [
        state_dir / "code" / f"{iteration:03d}" / file,
        state_dir / "logs" / f"{iteration:03d}" / file,
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            return {
                "found": True,
                "iteration": iteration,
                "file": file,
                "text": text,
            }
    return {"found": False, "iteration": iteration, "file": file}


# ---- Retrospective ----

def read_retrospective(state_dir: Path, iteration: int) -> Dict[str, Any]:
    """Read the retrospective markdown for a given iteration.

    Looks for ``<state_dir>/logs/<nnn>/retrospective.md``.
    """
    path = state_dir / "logs" / f"{iteration:03d}" / "retrospective.md"
    if path.exists():
        return {
            "found": True,
            "iteration": iteration,
            "markdown": path.read_text(encoding="utf-8", errors="replace"),
        }
    return {"found": False, "iteration": iteration}


# ---- Log ----

def read_log(state_dir: Path) -> Dict[str, Any]:
    """Read the orchestrator log file."""
    log_path = state_dir / "orchestrator.log"
    if log_path.exists():
        return {
            "found": True,
            "text": log_path.read_text(encoding="utf-8", errors="replace"),
        }
    return {"found": False, "text": "(no log file yet)"}


# ---- Charts ----

def read_charts(state_dir: Path) -> Dict[str, Any]:
    """Aggregate perf-per-iteration + durations for the charts panel.

    Reads iteration records and reshapes them into series for Chart.js.
    For knowledge-evolution, the only perf metric surfaced is
    ``oracle_cases_passed``.
    """
    recs = read_iterations(state_dir)
    durations = [
        {"x": r.get("iteration", r.get("n", 0)), "y": round(r.get("duration_s", 0) or 0, 1)}
        for r in recs if r.get("duration_s")
    ]
    perf_keys: list[str] = []
    for r in recs:
        for k in (r.get("perf") or {}):
            if k not in perf_keys:
                perf_keys.append(k)
    perf_series = []
    for k in perf_keys:
        series = [
            {"x": r.get("iteration", r.get("n", 0)), "y": (r.get("perf") or {}).get(k)}
            for r in recs if r.get("perf") and k in r["perf"]
        ]
        perf_series.append({"metric": k, "points": series})
    return {
        "durations": durations,
        "perf_series": perf_series,
    }


def read_agent_status(state_dir: Path) -> Optional[str]:
    """Read the current agent activity string written by the pipeline."""
    path = state_dir / "agent_status"
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None
