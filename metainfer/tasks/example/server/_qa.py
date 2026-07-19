"""QAConfig -- tells the generic QA engine how to find transcripts on disk.

  ``resolve_target(state_dir, payload)`` returns::

      {"events_file": Path, "target_workdir": Path | None, "target_label": str}

  The QA engine only calls this method -- it knows nothing about your task's
  directory layout (step dirs, agent names, iteration structure, ...).

  Provide the SMALLEST implementation that works. For a frontend-driven QA
  (where the browser uploads an events file and workdir directly), just
  propagate the ``events_file`` and ``target_workdir`` fields from the
  payload verbatim. See ``gen_infer_framework/server/_qa.py`` for a more
  complex example that supports tuple-lookup (iteration, agent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ExampleQAConfig:
    """Minimal QA pathsolver — frontend-driven only."""

    def resolve_target(
        self, state_dir: Path, payload: dict[str, Any],
    ) -> dict[str, Any]:
        events_file = payload.get("events_file")
        if not events_file:
            raise ValueError("payload must contain 'events_file'")
        target_workdir = payload.get("target_workdir")
        return {
            "events_file": Path(events_file),
            "target_workdir": (
                Path(target_workdir) if target_workdir else None
            ),
            "target_label": payload.get("target_label", "events_file"),
        }
