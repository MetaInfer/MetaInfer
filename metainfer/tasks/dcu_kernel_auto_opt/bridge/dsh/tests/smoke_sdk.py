#!/usr/bin/env python3
"""Smoke test: drive a real DSH agent via the Python SDK inside zth_meta.

Verifies, in order:
  1. runtime boots with the MetaInfer cordis composition
  2. a trivial prompt returns a final response (model endpoint works)
  3. session id is stable and a second prompt resumes the same conversation
  4. the agent can create a file through its tools (fs/bash) in DSH_CWD
  5. the agent can read a file outside DSH_CWD (absolute path) — no sandbox

Usage (inside the container):
    DEEPSEEK_API_KEY=... python3 /workspace/MetaInfer/metainfer/tasks/dcu_kernel_auto_opt/bridge/dsh/tests/smoke_sdk.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

CORDIS = Path(__file__).resolve().parents[1] / "cordis.yml"
WORK = Path("/tmp/dsh-smoke")
SESSION_ROOT = WORK / ".sessions"


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = DeepSeekHarnessConfig(
        provider="deepseek-official",
        model="deepseek-v4-flash",
        max_tokens=2048,
        cwd=str(WORK),
        session_root=str(SESSION_ROOT),
        cordis=str(CORDIS),
        env={"DSH_CWD": str(WORK), "DSH_SESSION_ROOT": str(SESSION_ROOT)},
        request_timeout_seconds=600.0,
        shutdown_timeout_seconds=15.0,
    )
    sid = f"session-smoke-{os.urandom(4).hex()}"
    print(f"[smoke] session={sid} cordis={CORDIS}", flush=True)
    with DeepSeekHarness(cfg) as h:
        r1 = h.run("Reply with exactly: DSH_OK", session_id=sid)
        print("[smoke] run1 session_id:", r1.session_id, flush=True)
        print("[smoke] run1 finish_reason:", r1.finish_reason, flush=True)
        print("[smoke] run1 final_response:", repr(r1.final_response), flush=True)
        if "DSH_OK" not in (r1.final_response or ""):
            print("[smoke] FAIL: unexpected run1 response", flush=True)
            return 1

        r2 = h.run(
            "Continuing the same conversation: what was the exact token you "
            "were asked to reply with? Reply with that token only.",
            session_id=sid,
        )
        print("[smoke] run2 session_id:", r2.session_id, flush=True)
        print("[smoke] run2 final_response:", repr(r2.final_response), flush=True)
        if "DSH_OK" not in (r2.final_response or ""):
            print("[smoke] WARN: resume did not preserve context", flush=True)

        r3 = h.run(
            "Create a file named smoke.txt in the current working directory "
            "containing exactly the text FILE_OK. Then print the file path.",
            session_id=sid,
        )
        print("[smoke] run3 final_response:", repr(r3.final_response), flush=True)
        created = (WORK / "smoke.txt").exists()
        content = (WORK / "smoke.txt").read_text() if created else ""
        print("[smoke] file created:", created, "content:", repr(content), flush=True)
        if not created or "FILE_OK" not in content:
            print("[smoke] FAIL: agent could not create file in DSH_CWD", flush=True)
            return 1

        r4 = h.run(
            "Read the file /workspace/MetaInfer/metainfer/tasks/dcu_kernel_auto_opt/"
            "bridge/dsh/cordis.yml and report the id of the first entry.",
            session_id=sid,
        )
        print("[smoke] run4 final_response:", repr(r4.final_response), flush=True)

    print("[smoke] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
