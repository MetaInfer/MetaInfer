"""QA config for find-low-hanging-kernel.

Targets are addressed by (step, agent) tuples so the analyst can review any
of the analysis agents that ran during Step 1 / Step 2, or any of the 5
validation pool workers in Step 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.server._helpers import find_events_file

PLUGIN_TYPE = "find-low-hanging-kernel"


class FlhkQAConfig:
    def resolve_target(
        self, state_dir: Path, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        events_file_str = (payload.get("events_file") or "").strip()
        if events_file_str:
            return {
                "events_file": Path(events_file_str),
                "target_workdir": (
                    Path(payload["target_workdir"])
                    if payload.get("target_workdir")
                    else None
                ),
                "target_label": (
                    payload.get("target_label")
                    or f"events_file={Path(events_file_str).name}"
                ),
            }

        step = payload.get("step")
        agent = payload.get("agent")
        if step is not None and agent is not None:
            ef = _resolve_events_file(state_dir, str(step), str(agent))
            return {
                "events_file": ef,
                "target_workdir": None,
                "target_label": f"step={step} agent={agent}",
            }

        raise ValueError(
            "payload must contain either events_file, or (step, agent)"
        )


def _resolve_events_file(state_dir: Path, step: str, agent: str) -> Path:
    """Search a few common layouts for the agent's events.jsonl."""
    candidates = [
        state_dir / "logs" / step / agent,
        state_dir / "logs" / step / "pool" / agent,
        state_dir / step / agent,
    ]
    ef: Optional[Path] = find_events_file(candidates[0])
    if ef is not None:
        return ef
    # Fall back to globbing under the most likely root.
    for root in candidates[1:]:
        if root.exists():
            hits = sorted(root.rglob(f"{agent}.attempt*.events.jsonl"))
            if hits:
                return hits[0]
    # Final fallback: glob the whole logs/ tree.
    glob_root = state_dir / "logs"
    if glob_root.exists():
        hits = sorted(glob_root.rglob(f"{agent}.attempt*.events.jsonl"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no events.jsonl for agent {agent!r} (step={step!r}) under {state_dir}"
    )


CONFIG = FlhkQAConfig()
