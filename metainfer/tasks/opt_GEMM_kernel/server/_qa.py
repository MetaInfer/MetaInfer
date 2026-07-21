"""QA transcript resolver for GEMM arena agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from metainfer.server._helpers import find_events_file


class GemmArenaQAConfig:
    def resolve_target(self, state_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
        explicit = str(payload.get("events_file") or "").strip()
        if explicit:
            return {
                "events_file": Path(explicit),
                "target_workdir": None,
                "target_label": payload.get("target_label") or Path(explicit).name,
            }
        try:
            iteration = int(payload.get("iteration"))
        except (TypeError, ValueError) as exc:
            raise ValueError("iteration is required and must be an integer") from exc
        agent = str(payload.get("agent") or "").strip()
        if not agent:
            raise ValueError("agent is required")
        root = state_dir / "logs" / f"{iteration:03d}"
        events = find_events_file(root / agent)
        if events is None:
            hits = sorted(root.rglob(f"{agent}.attempt*.events.jsonl")) if root.exists() else []
            events = hits[0] if hits else None
        if events is None:
            raise FileNotFoundError(f"no events for {agent!r} in iteration {iteration}")
        return {
            "events_file": events,
            "target_workdir": None,
            "target_label": f"iteration={iteration} agent={agent}",
        }


CONFIG = GemmArenaQAConfig()
