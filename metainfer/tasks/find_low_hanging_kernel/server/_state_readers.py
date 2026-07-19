"""State-dir readers for find-low-hanging-kernel."""

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


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    run = _read_run(state_dir)
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome")
    last_label = run.get("last_transition_label")
    if hasattr(_phases, "graph_payload"):
        return _phases.graph_payload(current, last_outcome, last_label)
    return {"error": "phases module does not export graph_payload()"}


def read_flow_graph(workspace_dir: Path) -> Dict[str, Any]:
    """Load the validated flow_graph.json, or return a stub."""
    fg = workspace_dir / "flow_graph.json"
    data = _load_json(fg, None)
    if data is None:
        return {"ready": False, "reason": "flow_graph.json not written yet"}
    return {"ready": True, "graph": data}


def read_trace_summary(workspace_dir: Path) -> Dict[str, Any]:
    tp = workspace_dir / "trace_parsed.json"
    data = _load_json(tp, None)
    if data is None:
        return {"ready": False}
    return {"ready": True, "summary": data}


def read_memory_markdown(workspace_dir: Path, step: str) -> Dict[str, Any]:
    """Read a memory/*.md file for in-browser audit.

    `step` is one of: step1_code_analysis, step2_tracing_analysis,
    validation_warnings.
    """
    safe_name = Path(step).name
    p = workspace_dir / "memory" / f"{safe_name}.md"
    if not p.is_file():
        return {"ready": False, "markdown": "", "path": str(p)}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"ready": False, "markdown": "", "path": str(p)}
    return {"ready": True, "markdown": text, "path": str(p)}
