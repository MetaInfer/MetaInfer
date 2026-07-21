"""CLI entry point for ``opt_GEMM_kernel``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MetaInfer GEMM kernel arena")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("requirements", type=Path)
    run.add_argument("--state-dir", type=Path)
    run.add_argument("--workspace-dir", type=Path)
    run.add_argument("--claude-bin", default=os.environ.get("METAINFER_CLAUDE_BIN", "ccb"))
    run.add_argument("--permission-mode", default=os.environ.get("METAINFER_PERMISSION_MODE", "bypassPermissions"))
    run.add_argument("--model", default=None)
    run.add_argument("--effort", default=os.environ.get("METAINFER_EFFORT", "max"))
    run.add_argument("--max-iterations", type=int, default=None)
    run.add_argument("--extra-claude-arg", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "run":
        from .orchestrator import run_with_requirements
        return run_with_requirements(
            args.requirements,
            state_dir=args.state_dir,
            workspace_dir=args.workspace_dir,
            claude_bin=args.claude_bin,
            permission_mode=args.permission_mode,
            model=args.model,
            effort=args.effort,
            max_iterations=args.max_iterations,
            extra_claude_args=args.extra_claude_arg,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())

