"""Focused tests for SubAgentManager backend command/parse behavior."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from metainfer.orchestrator.subagent_manager import (
    AgentHandle,
    AgentSpec,
    SubAgentManager,
)
from metainfer.orchestrator._bootstrap import make_subagent_manager


class _FakeProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def _spec(tmp: Path, *, resume_session_id: str | None = None) -> AgentSpec:
    workdir = tmp / "work"
    log_dir = tmp / "logs"
    workdir.mkdir()
    log_dir.mkdir()
    prompt = tmp / "prompt.txt"
    prompt.write_text("do work", encoding="utf-8")
    return AgentSpec(
        name="agent1",
        role="tester",
        prompt_file=prompt,
        workdir=workdir,
        log_dir=log_dir,
        resume_session_id=resume_session_id,
    )


def test_codex_command_uses_exec_json_and_current_config():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kb = tmp / "kb"
        kb.mkdir()
        mgr = SubAgentManager(
            agent_backend="codex",
            codex_bin="codex-dev",
            extra_add_dirs=[kb],
            default_model=None,
        )
        cmd = mgr._build_command(_spec(tmp))
        assert cmd[:5] == ["codex-dev", "exec", "--json", "--color", "never"]
        assert "--skip-git-repo-check" in cmd
        assert "--sandbox" in cmd
        assert "workspace-write" in cmd
        assert "-C" in cmd
        assert str(tmp / "work") in cmd
        assert "--add-dir" in cmd
        assert "/tmp" in cmd
        assert str(kb.resolve()) in cmd
        assert "--model" not in cmd
        assert "--ignore-user-config" not in cmd
        assert cmd[-1] == "-"


def test_codex_resume_command_threads_session():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = SubAgentManager(agent_backend="codex", codex_bin="codex-dev",
                              default_model="gpt-5-codex")
        cmd = mgr._build_command(_spec(tmp, resume_session_id="thread-123"))
        assert cmd == [
            "codex-dev", "exec", "resume", "--json",
            "--skip-git-repo-check", "--model", "gpt-5-codex",
            "thread-123", "-",
        ]


def test_materialize_codex_jsonl_result():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        spec = _spec(tmp)
        events = [
            {"type": "thread.started", "thread_id": "thread-123"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "done"},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 8,
                    "output_tokens": 2,
                    "reasoning_output_tokens": 1,
                },
            },
        ]
        ef = spec.events_file(1)
        ef.write_text(
            "Reading additional input from stdin...\n"
            + "\n".join(json.dumps(e) for e in events)
            + "\n",
            encoding="utf-8",
        )
        handle = AgentHandle(
            spec=spec,
            attempt=1,
            process=_FakeProcess(0),  # type: ignore[arg-type]
            started_at=time.time() - 1,
            last_output_at=time.time(),
        )
        mgr = SubAgentManager(agent_backend="codex")
        result = mgr._materialize_result(handle, spec, 1)
        assert result.success
        assert result.final_text == "done"
        assert result.session_id == "thread-123"
        assert result.usage == {**events[-1], "thread_id": "thread-123"}


def test_bootstrap_factory_reads_codex_env(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        monkeypatch.setenv("METAINFER_AGENT_BACKEND", "codex")
        monkeypatch.setenv("METAINFER_CODEX_BIN", "codex-dev")
        mgr = make_subagent_manager(
            claude_bin="ccb",
            model=None,
            permission_mode="bypassPermissions",
            effort="max",
            extra_add_dirs=[],
            snapshot_file=tmp / "agents.json",
        )
        try:
            assert mgr.agent_backend == "codex"
            assert mgr.codex_bin == "codex-dev"
        finally:
            mgr.shutdown()
