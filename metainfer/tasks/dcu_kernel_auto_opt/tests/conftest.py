"""Shared fixtures for dcu-kernel-auto-opt plugin route tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from metainfer.testing import isolated_env  # noqa: F401 — re-export as fixture
from metainfer.server import app as app_module
from metainfer.server import tasks as _tasks
from metainfer.server.tasks import TaskEntry


@pytest.fixture
def app(isolated_env):
    return app_module.create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


def register_dkao_task(
    state_dir, workspace_dir, task_id: str = "dkao-1"
) -> TaskEntry:
    """Register one dcu-kernel-auto-opt task in the WebUI registry."""
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    entry = TaskEntry(
        id=task_id,
        type="dcu-kernel-auto-opt",
        label="test dkao task",
        state_dir=str(state_dir),
        workspace_dir=str(workspace_dir),
        created_at=0.0,
    )
    _tasks.add_task(entry)
    return entry
