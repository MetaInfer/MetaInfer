"""CLI entry point for the port-model orchestrator.

Contract: ``python -m <this_module> run <req.json> --state-dir … --workspace-dir …``
See CLAUDE.md — all task orchestrator CLIs must accept this exact argv shape.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="metainfer-orchestrator-port-model",
        description="port-model — port a model to a target inference framework",
    )
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="start the orchestrator from a requirements file")
    run_p.add_argument("requirements", type=Path, help="path to requirements.json")
    run_p.add_argument("--state-dir", type=Path, required=True, help="state directory (.metainfer/tasks/<id>)")
    run_p.add_argument("--workspace-dir", type=Path, required=True, help="workspace directory for generated artifacts")
    run_p.add_argument("--claude-bin", type=str, default="ccb", help="claude-code binary (default: ccb)")
    run_p.add_argument("--permission-mode", type=str, default="bypassPermissions", help="permission mode for agents")
    run_p.add_argument("--model", type=str, default=None, help="override model for agents")
    run_p.add_argument("--effort", type=str, default="max", help="effort level (default: max)")
    run_p.add_argument("--extra-claude-arg", action="append", dest="extra_claude_args", default=[], help="extra arg to forward to claude-cli (repeatable)")

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        sys.exit(2)

    from .orchestrator import run_with_requirements

    run_with_requirements(
        req_path=args.requirements,
        state_dir=args.state_dir,
        workspace_dir=args.workspace_dir,
        claude_bin=args.claude_bin,
        permission_mode=args.permission_mode,
        model=args.model,
        effort=args.effort,
        extra_claude_args=args.extra_claude_args,
    )


if __name__ == "__main__":
    main()
