"""QA path-resolver config for knowledge-evolution tasks.

Tells the generic QA engine how to find transcript/event files on disk
for this task type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class KEEvolutionQAConfig:
    """Minimal QA config for knowledge-evolution.

    Frontend-driven QA: the browser uploads events files and workdir directly.
    """

    def resolve_target(self, state_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve QA target paths from a state directory and client payload.

        Args:
            state_dir: The task's ``.metainfer/tasks/<task_id>/`` directory.
            payload: Client-supplied payload with ``events_file`` and
                optional ``target_workdir``.

        Returns:
            Dict with ``events_file`` (Path), ``target_workdir`` (Path | None),
            and ``target_label`` (str).
        """
        events_file = Path(payload["events_file"])
        target_workdir = (
            Path(payload["target_workdir"]) if payload.get("target_workdir") else None
        )
        target_label = payload.get("target_label", str(events_file.name))
        return {
            "events_file": events_file,
            "target_workdir": target_workdir,
            "target_label": target_label,
        }
