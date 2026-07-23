"""Claude Code PreToolUse hook for the task's B implementation phase."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def _load_evaluator():
    try:
        from metainfer.tasks.gen_cpp_infer_framework.orchestrator.execution_policy import (
            evaluate_pre_tool_use,
        )
        return evaluate_pre_tool_use
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(repo_root))
        from metainfer.tasks.gen_cpp_infer_framework.orchestrator.execution_policy import (
            evaluate_pre_tool_use,
        )
        return evaluate_pre_tool_use


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError) as exc:
        print(f"MetaInfer policy hook could not parse tool input: {exc}", file=sys.stderr)
        return 2
    decision = _load_evaluator()(payload)
    if decision["allowed"]:
        return 0

    audit_path = os.environ.get("METAINFER_COMMAND_POLICY_AUDIT")
    if audit_path:
        record = {
            "ts": time.time(),
            "tool_name": decision["tool_name"],
            "command_sha256": decision["command_sha256"],
            "rules": decision["rules"],
            "blocked": True,
        }
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    print(
        "MetaInfer blocked this command before execution: " + decision["message"],
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
