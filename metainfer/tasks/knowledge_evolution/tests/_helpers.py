"""Test helpers for knowledge-evolution task.

Shared mock tools come from :mod:`metainfer.testing`::

    from metainfer.testing import (
        MockAgentManager, FakeStore, FakeLauncher, isolated_env,
    )

Put task-specific helper factories here — NOT in ``metainfer/testing/``.
"""

from __future__ import annotations

import json
from pathlib import Path


def make_ke_requirements(task_id: str = "ke-test-1") -> dict:
    """Build a minimal requirements.json payload for knowledge-evolution testing."""
    return {
        "task_id": task_id,
        "task_type": "knowledge-evolution",
        "created_at": 0.0,
        "form": {
            "target_model": "test-model",
            "description": "Test knowledge evolution run",
            "max_iterations": 3,
            "max_verify_attempts": 1,
        },
    }


def make_ke_state_dir(tmp: Path, task_id: str = "ke-test-1") -> Path:
    """Create a state_dir and workspace_dir stub for tests."""
    sd = tmp / "state"
    sd.mkdir(parents=True)
    (sd / "requirements.json").write_text(
        json.dumps(make_ke_requirements(task_id))
    )
    (sd / "iterations").mkdir(exist_ok=True)
    wd = tmp / "workspace"
    wd.mkdir(parents=True)
    return sd
