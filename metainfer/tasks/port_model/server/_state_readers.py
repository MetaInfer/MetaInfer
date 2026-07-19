"""Read-only access to port-model on-disk state.

These are called by the API routes. All reads are defensive: missing files
return None / empty defaults rather than raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.requirements import req_field
from metainfer.server import state_reader as _sr


def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    """Return all iteration records sorted by iteration number."""
    iters_dir = state_dir / "iterations"
    if not iters_dir.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(iters_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data.setdefault("iteration", int(p.stem))
            out.append(data)
        except (ValueError, OSError):
            continue
    return out


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    path = state_dir / "iterations" / f"{n:03d}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    """Build the state-graph payload from run.json + phases."""
    run = _sr.read_run(state_dir)
    try:
        from ..orchestrator.phases import graph_payload
    except ImportError:
        return {"current": "idle", "nodes": [], "edges": []}
    return graph_payload(
        current=run.get("current_phase", "idle"),
        last_outcome=run.get("last_outcome"),
        last_label=run.get("last_transition_label"),
    )


def read_memory_markdown(workspace_dir: Path, step: str) -> Optional[str]:
    """Read a memory/*.md file."""
    path = workspace_dir / "memory" / f"{step}.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def read_diff(workspace_dir: Path) -> Optional[str]:
    """Read the diff/model_port.patch file."""
    path = workspace_dir / "diff" / "model_port.patch"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def read_test_results(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the test/test_results.json file."""
    path = workspace_dir / "test" / "test_results.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
