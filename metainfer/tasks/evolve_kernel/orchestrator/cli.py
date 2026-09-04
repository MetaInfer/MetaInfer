"""CLI for the evolve-kernel orchestrator subprocess."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_CLAUDE_BIN = "ccb"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
_VALID_PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions", "auto")
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
        raise SystemExit(f"invalid permission mode {v!r}; expected one of {', '.join(_VALID_PERMISSION_MODES)}")
    return v


def _resolve_effort(cli_value: str | None) -> str:
    if cli_value:
        v = cli_value
    else:
        v = os.environ.get("METAINFER_EFFORT", DEFAULT_EFFORT)
    if v not in _VALID_EFFORTS:
        raise SystemExit(f"invalid effort {v!r}; expected one of {', '.join(_VALID_EFFORTS)}")
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="metainfer-orchestrator",
        description="MetaInfer evolve-kernel orchestrator (spawned by WebUI).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run orchestrator on requirements.json")
    run_p.add_argument("requirements", type=Path, help="Path to requirements.json")
    run_p.add_argument("--state-dir", type=Path, default=None,
                       help="Metadata dir (run.json, timeline.jsonl, logs/, ...)")
    run_p.add_argument("--workspace-dir", type=Path, default=None,
                       help="Generated-artifacts dir (iteration code: 001/, 002/, ...)")
    run_p.add_argument("--claude-bin", default=None,
                       help=f"Claude Code binary (default: env METAINFER_CLAUDE_BIN or {DEFAULT_CLAUDE_BIN!r})")
    run_p.add_argument("--permission-mode", default=None, choices=_VALID_PERMISSION_MODES,
                       help=f"Claude Code permission mode (default: {DEFAULT_PERMISSION_MODE!r})")
    run_p.add_argument("--model", default=None, help="Override model for sub-agents")
    run_p.add_argument("--effort", default=None, choices=_VALID_EFFORTS,
                       help=f"Claude Code effort level (default: {DEFAULT_EFFORT!r})")
    run_p.add_argument("--gpu-device", type=str, default=None,
                       help="CUDA visible device(s), e.g. '0' or '0,1'. Sets CUDA_VISIBLE_DEVICES env var before importing torch.")
    run_p.add_argument("--max-iterations", type=int, default=None,
                       help="Override max iterations")
    run_p.add_argument("--extra-claude-arg", action="append", default=[],
                       help="Extra arg(s) forwarded to claude -p")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        # Set GPU device BEFORE importing torch/triton inside the orchestrator
        if args.gpu_device is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
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
