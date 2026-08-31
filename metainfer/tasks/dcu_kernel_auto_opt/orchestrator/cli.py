"""CLI entry point for dcu-kernel-auto-opt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcu-kernel-auto-opt")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("requirements", type=Path)
    run_p.add_argument("--state-dir", type=Path, required=True)
    run_p.add_argument("--workspace-dir", type=Path, required=True)
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument(
        "--claude-bin",
        default=None,
        help=(
            "Agent binary override. Defaults are resolved from the task's "
            "agent_framework answer: ccb -> METAINFER_CLAUDE_BIN (or 'ccb'), "
            "dsh -> the bundled bridge/dsh/dsh_agent.py wrapper."
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        from .config import resolve_agent_framework, resolve_claude_bin
        from .orchestrator import run_with_requirements

        req = json.loads(args.requirements.read_text(encoding="utf-8"))
        framework = resolve_agent_framework(req)
        claude_bin = resolve_claude_bin(framework, explicit=args.claude_bin)
        return run_with_requirements(
            args.requirements,
            state_dir=args.state_dir,
            workspace_dir=args.workspace_dir,
            dry_run=args.dry_run,
            claude_bin=claude_bin,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
