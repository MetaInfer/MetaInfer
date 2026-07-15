"""CLI for the gen-cpp-infer-framework orchestrator subprocess.

The orchestrator runs as a child of the WebUI server (see
:mod:`metainfer.server.launcher`). One orchestrator per task, spawned when
the user submits a new task via the web form, exits when the task
completes or stops.

Direct CLI usage (for debugging without the WebUI):

    python -m metainfer.tasks.gen_cpp_infer_framework.orchestrator.cli run requirements.json
    python -m metainfer.tasks.gen_cpp_infer_framework.orchestrator.cli run requirements.json --state-dir /path/to/state --workspace-dir /path/to/workspace
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# The Claude Code binary the sub-agent manager shells out to. Resolution:
#   1. --claude-bin CLI flag (highest priority)
#   2. METAINFER_CLAUDE_BIN env var
#   3. "ccb" (sensible default; override per environment if needed)
DEFAULT_CLAUDE_BIN = "ccb"

# Claude Code permission mode for sub-agents. Sub-agents are non-interactive
# (`-p` with stdin), so `default` mode hangs on every Edit/Write prompt.
# See the longer rationale in metainfer.server.forms / task docs.
DEFAULT_PERMISSION_MODE = "bypassPermissions"
_VALID_PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions", "auto")

# Claude Code effort level controls extended-thinking budget per turn.
DEFAULT_EFFORT = "max"
_VALID_EFFORTS = ("low", "medium", "high", "max")


def _resolve_claude_bin(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return os.environ.get("METAINFER_CLAUDE_BIN", DEFAULT_CLAUDE_BIN)


def _resolve_permission_mode(cli_value: str | None) -> str:
    if cli_value:
        v = cli_value
    else:
        v = os.environ.get("METAINFER_PERMISSION_MODE", DEFAULT_PERMISSION_MODE)
    if v not in _VALID_PERMISSION_MODES:
        raise SystemExit(
            f"invalid permission mode {v!r}; expected one of "
            f"{', '.join(_VALID_PERMISSION_MODES)}"
        )
    return v


def _resolve_effort(cli_value: str | None) -> str:
    if cli_value:
        v = cli_value
    else:
        v = os.environ.get("METAINFER_EFFORT", DEFAULT_EFFORT)
    if v not in _VALID_EFFORTS:
        raise SystemExit(
            f"invalid effort {v!r}; expected one of {', '.join(_VALID_EFFORTS)}"
        )
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="metainfer-orchestrator",
        description="MetaInfer per-task orchestrator (spawned by the WebUI).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the orchestrator on a requirements.json")
    run_p.add_argument("requirements", type=Path, help="Path to requirements.json")
    run_p.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(
            "Metadata dir (run.json, timeline.jsonl, logs, iteration records). "
            "The WebUI passes this explicitly."
        ),
    )
    run_p.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help=(
            "Generated-artifacts dir containing iteration trees (001, 002, ...). "
            "The WebUI passes this explicitly."
        ),
    )
    run_p.add_argument(
        "--claude-bin",
        default=None,
        help=(
            "Claude Code binary to shell out to for sub-agents "
            f"(default: env METAINFER_CLAUDE_BIN or {DEFAULT_CLAUDE_BIN!r})"
        ),
    )
    run_p.add_argument(
        "--permission-mode",
        default=None,
        choices=_VALID_PERMISSION_MODES,
        help=(
            "Claude Code permission mode for sub-agents (default: env "
            f"METAINFER_PERMISSION_MODE or {DEFAULT_PERMISSION_MODE!r})."
        ),
    )
    run_p.add_argument("--model", default=None, help="Override model for sub-agents")
    run_p.add_argument(
        "--effort",
        default=None,
        choices=_VALID_EFFORTS,
        help=(
            "Claude Code effort level, controls extended-thinking budget per "
            f"turn (default: env METAINFER_EFFORT or {DEFAULT_EFFORT!r})."
        ),
    )
    run_p.add_argument("--max-iterations", type=int, default=None,
                       help="Override max iterations (defaults to requirements.max_iterations or 20)")
    run_p.add_argument("--extra-claude-arg", action="append", default=[],
                       help="Extra arg(s) forwarded to claude -p")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from .orchestrator import run_with_requirements
        return run_with_requirements(
            requirements_path=args.requirements,
            state_dir=args.state_dir,
            workspace_dir=args.workspace_dir,
            claude_bin=_resolve_claude_bin(args.claude_bin),
            permission_mode=_resolve_permission_mode(args.permission_mode),
            model=args.model,
            max_iterations=args.max_iterations,
            extra_claude_args=args.extra_claude_arg,
            effort=_resolve_effort(args.effort),
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
