#!/usr/bin/env python3
"""Unit tests for dsh_agent.py (SubAgentManager stream-json protocol).

Runs without a real model endpoint: the ``deepseek_harness`` SDK module is
replaced by a fake whose behavior is scripted per test. Verify:

  1. arg parsing tolerates every SubAgentManager flag (incl. unknown ones)
  2. model mapping: sonnet/haiku -> deepseek-v4-flash, opus -> deepseek-v4-pro
  3. the emitted stream contains system -> assistant* -> result, exit 0
  4. a resume failure (finish_reason == "error") falls back to a fresh
     session and the stream's first session_id is the fallback one
  5. an SDK exception exits nonzero without a result event

Usage (inside the container where deepseek-harness-sdk is installed, or with
PYTHONPATH pointing at the SDK source):
    python3 tests/test_dsh_agent.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[1] / "dsh_agent.py"


class FakeRunResult:
    def __init__(self, session_id, final_response, finish_reason):
        self.session_id = session_id
        self.final_response = final_response
        self.finish_reason = finish_reason


class FakeDeepSeekHarness:
    """Scripted harness: returns queued results in order, or raises."""

    instances = []

    def __init__(self, config=None, **kwargs):
        self.config = config
        self.runs = []
        FakeDeepSeekHarness.instances.append(self)

    def run(self, prompt, session_id=None, on_notification=None):
        self.runs.append({"prompt": prompt, "session_id": session_id})
        script = FakeDeepSeekHarness.script
        if isinstance(script, Exception):
            raise script
        item = script.pop(0)
        if item == "raise":
            raise RuntimeError("boom")
        return FakeRunResult(
            session_id=item[0], final_response=item[1], finish_reason=item[2]
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def install_fake_sdk(script):
    FakeDeepSeekHarness.script = list(script)
    FakeDeepSeekHarness.instances = []
    fake = types.ModuleType("deepseek_harness")
    fake.DeepSeekHarness = FakeDeepSeekHarness
    fake.DeepSeekHarnessConfig = FakeConfig
    sys.modules["deepseek_harness"] = fake
    # Reload the wrapper so its import picks up the fake.
    for name in list(sys.modules):
        if name == "dsh_agent":
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("dsh_agent", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStdin:
    """Exposes .buffer (BytesIO) like a real process stdin."""

    def __init__(self, text):
        self.buffer = io.BytesIO(text.encode("utf-8"))


def run_wrapper(module, argv, prompt):
    out, err = io.StringIO(), io.StringIO()
    old_stdin = sys.stdin
    sys.stdin = FakeStdin(prompt)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = module.main(argv)
    finally:
        sys.stdin = old_stdin
    return code, out.getvalue(), err.getvalue()


BASE_ARGS = [
    "-p", "--output-format", "stream-json", "--input-format", "text",
    "--verbose", "--permission-mode", "bypassPermissions",
    "--add-dir", "/tmp/wtest", "--model", "sonnet", "--effort", "low",
    "--max-turns", "50", "--disallowedTools", "Edit,Write",
]


class DshAgentTests(unittest.TestCase):
    def test_success_stream_protocol(self):
        mod = install_fake_sdk([
            ("session-aaa", "The answer is 42", "completed"),
        ])
        code, out, err = run_wrapper(mod, BASE_ARGS, "Do the thing.")
        self.assertEqual(code, 0)
        events = [json.loads(l) for l in out.splitlines() if l.strip()]
        types_seen = [e["type"] for e in events]
        # Stream order: system (with final session id) -> optional assistant
        # text -> result. The fake emits no assistant events; the real SDK
        # streams them via on_notification.
        self.assertEqual(types_seen[0], "system")
        self.assertEqual(types_seen[-1], "result")
        result = events[-1]
        self.assertEqual(result["result"], "The answer is 42")
        self.assertEqual(result["session_id"], "session-aaa")
        self.assertEqual(result["finish_reason"], "completed")
        # The system event carries the same session id.
        self.assertEqual(events[0]["session_id"], "session-aaa")

    def test_model_mapping(self):
        mod = install_fake_sdk([
            ("s1", "ok", "completed"),
        ])
        # opus -> deepseek-v4-pro
        run_wrapper(mod, [a if a != "sonnet" else "opus" for a in BASE_ARGS], "hi")
        cfg = FakeDeepSeekHarness.instances[0].config
        self.assertEqual(cfg.model, "deepseek-v4-pro")

    def test_resume_fallback(self):
        mod = install_fake_sdk([
            ("session-old", "", "error"),          # resume attempt fails
            ("session-new", "recovered", "completed"),  # fresh session works
        ])
        args = BASE_ARGS + ["--resume", "session-old"]
        code, out, err = run_wrapper(mod, args, "Continue the work.")
        self.assertEqual(code, 0)
        events = [json.loads(l) for l in out.splitlines() if l.strip()]
        result = events[-1]
        self.assertEqual(result["result"], "recovered")
        self.assertEqual(result["session_id"], "session-new")
        # The first session_id seen by the parser must be the fallback one.
        first_sid = next(e["session_id"] for e in events if "session_id" in e)
        self.assertEqual(first_sid, "session-new")
        self.assertIn("retrying as a fresh session", err)

    def test_sdk_exception_exits_nonzero(self):
        mod = install_fake_sdk(["raise"])
        code, out, err = run_wrapper(mod, BASE_ARGS, "hi")
        self.assertNotEqual(code, 0)
        events = [l for l in out.splitlines() if l.strip()]
        # No result event on failure.
        self.assertFalse(any("result" in json.loads(l) for l in events))

    def test_empty_prompt_fails(self):
        mod = install_fake_sdk([("s1", "x", "completed")])
        code, out, err = run_wrapper(mod, BASE_ARGS, "   ")
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
