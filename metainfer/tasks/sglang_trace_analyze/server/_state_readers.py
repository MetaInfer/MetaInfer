"""State-dir readers for sglang_trace_analyze.

Reads the authoritative analysis JSON files from
``<state_dir>/analysis/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def read_summary(state_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "analysis" / "summary.json")


def read_mapping(state_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "analysis" / "mapping.json")


def read_hints(state_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "analysis" / "hints.json")


def read_batch_detail(
    state_dir: Path, bs: int, stage: str
) -> Optional[Dict[str, Any]]:
    """Return the combined kernel_table + overlap + fuse for one
    (batch_size, stage) pair.
    """
    base = state_dir / "analysis" / "batches" / f"bs_{bs}" / stage
    kernel_table = _load_json(base / "kernel_table.json")
    overlap = _load_json(base / "overlap.json")
    fuse = _load_json(base / "fuse.json")
    if kernel_table is None and overlap is None and fuse is None:
        return None
    return {
        "kernel_table": kernel_table,
        "overlap": overlap,
        "fuse": fuse,
    }
