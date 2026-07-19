"""QA config for port-model — maps QA session targets to agent event files.

Mirrors the pattern in opt_kernel/server/_qa.py and gen_infer_framework/server/_qa.py:
a QAConfig subclass whose ``resolve_target`` accepts either an explicit
``events_file`` path or a ``(step, agent)`` tuple that the method translates
into the actual events.jsonl on disk.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from metainfer.server._helpers import find_events_file


class PortModelQAConfig:
    def resolve_target(
        self,
        state_dir: Path,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        events_file = payload.get("events_file")
        if events_file:
            p = Path(events_file)
            if p.is_file():
                return {
                    "events_file": str(p),
                    "target_workdir": str(p.parent),
                    "target_label": f"events: {p.parent.name}/{p.name}",
                }
            raise FileNotFoundError(f"events_file not found: {events_file}")

        # Tuple lookup: (step, agent) → logs/<step>/<agent>.events.jsonl
        step = payload.get("step")
        agent = payload.get("agent")
        if step and agent:
            return self._resolve_by_step_agent(state_dir, step, agent)

        raise ValueError(
            "payload must have either 'events_file' or ('step', 'agent')"
        )

    def _resolve_by_step_agent(
        self, state_dir: Path, step: str, agent: str,
    ) -> Dict[str, Any]:
        logs_dir = state_dir / "logs" / step
        # Try exact match first.
        candidate = logs_dir / f"{agent}.events.jsonl"
        if candidate.is_file():
            return {
                "events_file": str(candidate),
                "target_workdir": str(logs_dir),
                "target_label": f"{step}/{agent}",
            }
        # Fallback: glob for agent*.events.jsonl (some steps use name-prefix variants).
        pattern = str(logs_dir / f"{agent}*.events.jsonl")
        matches = glob.glob(pattern, recursive=False)
        if matches:
            m = Path(matches[0])
            return {
                "events_file": str(m),
                "target_workdir": str(logs_dir),
                "target_label": f"{step}/{m.stem}",
            }
        raise FileNotFoundError(
            f"no events file found for step={step!r} agent={agent!r} in {logs_dir}"
        )


QA_CONFIG = PortModelQAConfig()
