"""QA config for the gen-infer-framework-cpp task type.

The current QA flow is **frontend-driven**: the iterations reader
exposes ``events_file`` + ``target_workdir`` per agent entry, so the
generic ``/qa/start`` route just forwards those to
:mod:`metainfer.server.qa`. No server-side path resolution is needed for
the WebUI case.

This module is also the home for **server-side resolution** of
``{iteration, agent}`` tuples — useful for CLI/scripted QA callers
that don't have an explicit events_file path. The agent transcript
layout for gen-infer-framework-cpp is::

    <logs_root>/<NNN>/<agent>/<agent>.attempt0.events.jsonl

where ``<logs_root>`` defaults to ``<state_dir>/iterations/<NNN>/.metainfer-logs``
but can be relocated by the orchestrator's config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.server._helpers import find_events_file

PLUGIN_TYPE = "gen-infer-framework-cpp"


class GenInferQAConfig:
    """Resolve gen-infer-framework-cpp QA targets.

    Two resolution modes:

    1. **Explicit path** (default, used by current WebUI): payload
       contains ``events_file``. Validate + return it.
    2. **Tuple lookup**: payload contains ``{iteration, agent}``.
       Resolve via the per-iteration logs layout.
    """

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
            ef = _resolve_gf_events_file(state_dir, int(iteration), str(agent))
            return {
                "events_file": ef,
                "target_workdir": None,
                "target_label": f"iter={iteration} agent={agent}",
            }

        raise ValueError(
            "payload must contain either events_file, or (iteration, agent)"
        )


def _resolve_gf_events_file(
    state_dir: Path, iteration: int, agent: str,
) -> Path:
    """Locate ``events.jsonl`` for an (iteration, agent) tuple.

    The orchestrator writes per-agent transcripts under the iteration's
    logs dir. The current layout is::

        <state_dir>/iterations/<NNN>/.metainfer-logs/<agent>/<agent>.attempt0.events.jsonl

    but the orchestrator may relocate the logs root via config. We try
    the default location, then fall back to a glob.
    """
    NNN = f"{iteration:03d}"
    candidate_dirs = [
        state_dir / "iterations" / NNN / ".metainfer-logs" / agent,
        state_dir / "iterations" / NNN / "logs" / agent,
    ]
    for d in candidate_dirs:
        ef: Optional[Path] = find_events_file(d)
        if ef is not None:
            return ef
    # Last-resort glob across the iteration dir.
    glob_root = state_dir / "iterations" / NNN
    if glob_root.exists():
        hits = sorted(glob_root.rglob(f"{agent}.attempt*.events.jsonl"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no events.jsonl for agent {agent!r} in iteration {iteration} "
        f"under {state_dir}"
    )


# Singleton instance used by the plugin.
CONFIG = GenInferQAConfig()
