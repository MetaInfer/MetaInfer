"""CLI entry point for the sglang_trace_analyze orchestrator subprocess.

The launcher spawns::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …

Contract required by the framework (§6d): ``run`` subcommand + ``--state-dir``
and ``--workspace-dir`` flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metainfer-orchestrator")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run")
    run_p.add_argument("requirements", type=Path,
                       help="Path to requirements.json")
    run_p.add_argument("--state-dir", type=Path, required=True)
    run_p.add_argument("--workspace-dir", type=Path, required=True)
    # Task-specific flags
    run_p.add_argument("--iter-limit", type=int, default=None,
                       help="Override max iterations (default: derive from batch count)")

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1

    from .orchestrator import run_with_requirements
    return run_with_requirements(
        requirements_path=args.requirements,
        state_dir=args.state_dir,
        workspace_dir=args.workspace_dir,
        iter_limit=args.iter_limit,
    )


if __name__ == "__main__":
    sys.exit(main())
