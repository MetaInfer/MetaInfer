"""QA config for the evolve-kernel task type."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.server._helpers import find_events_file

PLUGIN_TYPE = "evolve-kernel"


class OptKernelQAConfig:
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

        iteration = payload.get("iteration")
        agent = payload.get("agent")
        if iteration is not None and agent is not None:
            ef = _resolve_events_file(state_dir, int(iteration), str(agent))
            return {
                "events_file": ef,
                "target_workdir": None,
                "target_label": f"iter={iteration} agent={agent}",
            }

        raise ValueError(
            "payload must contain either events_file, or (iteration, agent)"
        )


def _resolve_events_file(state_dir: Path, iteration: int, agent: str) -> Path:
    NNN = f"{iteration:03d}"
    for d in [
        state_dir / "iterations" / NNN / "logs" / agent,
        state_dir / "iterations" / NNN / ".metainfer-logs" / agent,
        state_dir / "logs" / NNN / agent,
    ]:
        ef: Optional[Path] = find_events_file(d)
        if ef is not None:
            return ef
    glob_root = state_dir / "logs" / NNN
    if glob_root.exists():
        hits = sorted(glob_root.rglob(f"{agent}.attempt*.events.jsonl"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no events.jsonl for agent {agent!r} in iteration {iteration} "
        f"under {state_dir}"
    )


CONFIG = OptKernelQAConfig()
