"""Bootstrap + entry point for the sglang_trace_analyze orchestrator.

Spawns as a child of the WebUI server per task. Reads requirements, sets
up state_dir / workspace_dir, and runs the linear phase pipeline:

    MAPPING -> BENCHMARK -> ANALYZE -> HINTS -> SUMMARIZE -> done

Unlike gen_infer_framework, this task has no complex transition table —
just five sequential phases with per-(bs, stage) iterations inside
ANALYZE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .pipeline import Pipeline
from .iteration_record import AnalyzeRecord
from metainfer.orchestrator.state import StateStore


def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    iter_limit: Optional[int] = None,
) -> int:
    """Per-task orchestrator entry point.

    Reads ``requirements.json``, runs the five-phase pipeline to
    completion, and exits.
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(
        requirements_path.read_text(encoding="utf-8")
    )
    task_id = req.get("task_id", "task")

    # Resolve state_dir + workspace_dir
    if state_dir is None or workspace_dir is None:
        from metainfer.server import paths as _web_paths
        if state_dir is None:
            state_dir = _web_paths.task_dir(task_id)
        if workspace_dir is None:
            workspace_dir = _web_paths.workspace_dir(task_id)

    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy requirements into state_dir for self-containment
    target_req = state_dir / "requirements.json"
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    store = StateStore(state_dir)
    pipe = Pipeline(
        req=req,
        store=store,
        state_dir=state_dir,
        workspace_dir=workspace_dir,
    )

    print(f"[metainfer:sglang_trace_analyze] task_id = {task_id}")
    print(f"[metainfer:sglang_trace_analyze] state_dir = {state_dir}")
    print(f"[metainfer:sglang_trace_analyze] workspace_dir = {workspace_dir}")

    pipe.run()
    return 0
