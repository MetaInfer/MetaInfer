"""Kernel-repository rename: core logic + Web route tests."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from metainfer.tasks.dcu_kernel_auto_opt.orchestrator.rename_kernel_repo import (
    rename_kernel_repo,
)
from .conftest import register_dkao_task


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _make_repo(workspace: Path, repo_root: Path, name: str = "repo-old") -> dict:
    """Create a kernel-repos git repo with one linked worktree + main symlink."""
    repo = repo_root / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-qm", "init")
    worktree = workspace / "source"
    _git(repo, "worktree", "add", "-q", str(worktree))
    main = workspace / "main"
    main.symlink_to(
        os.path.relpath(repo, start=workspace), target_is_directory=True
    )
    return {"repo": repo, "worktree": worktree}


def _seed_task_files(workspace: Path, state: Path, repo: Path, task_id: str) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / "requirements.json").write_text(json.dumps({
        "task_id": task_id,
        "label": repo.name,
        "target_repo_path": repo.name,
    }), encoding="utf-8")
    (workspace / "plan.json").write_text(json.dumps({
        "kernel_repo": str(repo),
    }), encoding="utf-8")
    (state / "run.json").write_text(
        json.dumps({"finished": True}), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Core function
# --------------------------------------------------------------------------- #

def test_rename_kernel_repo_happy_path(tmp_path, monkeypatch):
    repo_root = tmp_path / "kernel-repos"
    workspace = tmp_path / "workspaces" / "task-1"
    state = tmp_path / "state" / "task-1"
    workspace.mkdir(parents=True)
    made = _make_repo(workspace, repo_root)
    _seed_task_files(workspace, state, made["repo"], "task-1")
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(repo_root))

    result = rename_kernel_repo(workspace, "repo-new", state_dir=state)

    assert result["renamed"] is True
    assert result["old_name"] == "repo-old"
    assert result["new_name"] == "repo-new"
    new_repo = repo_root / "repo-new"
    assert new_repo.is_dir()
    assert not (repo_root / "repo-old").exists()
    assert result["new_repo"] == str(new_repo)

    # symlink re-pointed
    assert workspace.joinpath("main").resolve() == new_repo

    # JSON references updated
    req = json.loads((state / "requirements.json").read_text(encoding="utf-8"))
    assert req["target_repo_path"] == "repo-new"
    assert req["label"] == "repo-new"
    plan = json.loads((workspace / "plan.json").read_text(encoding="utf-8"))
    assert plan["kernel_repo"] == str(new_repo)

    # linked worktree survives the move and still commits
    assert str(made["worktree"]) in _git(new_repo, "worktree", "list").stdout
    (made["worktree"] / "f.txt").write_text("hi\nmore\n", encoding="utf-8")
    _git(made["worktree"], "add", "f.txt")
    _git(made["worktree"], "commit", "-qm", "wt commit after move")

    # timeline event recorded
    lines = (state / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("kernel_repo_renamed" in line for line in lines)


@pytest.mark.parametrize("bad_name", [
    "", " ", "a/b", "../escape", "a b", ".hidden", "-lead",
])
def test_rejects_invalid_names(tmp_path, monkeypatch, bad_name):
    repo_root = tmp_path / "kernel-repos"
    workspace = tmp_path / "workspaces" / "task-1"
    workspace.mkdir(parents=True)
    made = _make_repo(workspace, repo_root)
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(repo_root))
    with pytest.raises(ValueError):
        rename_kernel_repo(workspace, bad_name)
    assert (repo_root / "repo-old").is_dir()  # untouched


def test_rejects_collision(tmp_path, monkeypatch):
    repo_root = tmp_path / "kernel-repos"
    workspace = tmp_path / "workspaces" / "task-1"
    workspace.mkdir(parents=True)
    made = _make_repo(workspace, repo_root)
    (repo_root / "repo-new").mkdir()
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(repo_root))
    with pytest.raises(ValueError, match="already exists"):
        rename_kernel_repo(workspace, "repo-new")


def test_rejects_missing_symlink(tmp_path):
    workspace = tmp_path / "workspaces" / "task-1"
    workspace.mkdir(parents=True)
    with pytest.raises(ValueError, match="not a symlink"):
        rename_kernel_repo(workspace, "repo-new")


def test_rejects_running_task(tmp_path, monkeypatch):
    repo_root = tmp_path / "kernel-repos"
    workspace = tmp_path / "workspaces" / "task-1"
    state = tmp_path / "state" / "task-1"
    workspace.mkdir(parents=True)
    made = _make_repo(workspace, repo_root)
    state.mkdir(parents=True)
    (state / "run.json").write_text(
        json.dumps({"finished": False}), encoding="utf-8"
    )
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(repo_root))
    with pytest.raises(RuntimeError, match="still running"):
        rename_kernel_repo(workspace, "repo-new", state_dir=state)
    assert (repo_root / "repo-old").is_dir()


def test_allows_when_no_run_state_and_no_pid(tmp_path, monkeypatch):
    repo_root = tmp_path / "kernel-repos"
    workspace = tmp_path / "workspaces" / "task-1"
    state = tmp_path / "state" / "task-1"
    workspace.mkdir(parents=True)
    made = _make_repo(workspace, repo_root)
    state.mkdir(parents=True)  # no run.json, no orchestrator.pid
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(repo_root))
    result = rename_kernel_repo(workspace, "repo-new", state_dir=state)
    assert result["renamed"] is True


def test_cli_smoke(tmp_path, monkeypatch):
    repo_root = tmp_path / "kernel-repos"
    workspace = tmp_path / "workspaces" / "task-1"
    state = tmp_path / "state" / "task-1"
    workspace.mkdir(parents=True)
    made = _make_repo(workspace, repo_root)
    _seed_task_files(workspace, state, made["repo"], "task-1")
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(repo_root))
    import sys
    from metainfer.tasks.dcu_kernel_auto_opt.orchestrator import (
        rename_kernel_repo as module,
    )
    proc = subprocess.run(
        [sys.executable, "-m",
         "metainfer.tasks.dcu_kernel_auto_opt.orchestrator.rename_kernel_repo",
         str(state), str(workspace), "repo-cli"],
        capture_output=True, text=True, check=True,
    )
    assert "repo-cli" in proc.stdout
    assert (repo_root / "repo-cli").is_dir()


# --------------------------------------------------------------------------- #
# Web route
# --------------------------------------------------------------------------- #

def test_rename_repo_route(client, isolated_env, monkeypatch, tmp_path):
    from metainfer.server import paths as _paths
    task_id = "dkao-rename-1"
    state_dir = isolated_env["home"] / "tasks" / task_id
    workspace_dir = _paths.workspace_dir(task_id)
    repo_root = tmp_path / "kernel-repos"
    monkeypatch.setenv("METAINFER_KERNEL_REPOS", str(repo_root))
    register_dkao_task(state_dir, workspace_dir, task_id)
    made = _make_repo(workspace_dir, repo_root)
    _seed_task_files(workspace_dir, state_dir, made["repo"], task_id)

    resp = client.post(
        f"/api/dcu-kernel-auto-opt/{task_id}/rename-repo",
        json={"new_name": "repo-new"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["new_name"] == "repo-new"
    assert (repo_root / "repo-new").is_dir()
    assert not (repo_root / "repo-old").exists()

    # Renaming while the orchestrator is running is refused.
    (state_dir / "run.json").write_text(
        json.dumps({"finished": False}), encoding="utf-8"
    )
    resp = client.post(
        f"/api/dcu-kernel-auto-opt/{task_id}/rename-repo",
        json={"new_name": "repo-while-running"},
    )
    assert resp.status_code == 400
    assert "still running" in resp.json()["detail"]
    assert not (repo_root / "repo-while-running").exists()

    # Invalid names are refused with a clear error.
    resp = client.post(
        f"/api/dcu-kernel-auto-opt/{task_id}/rename-repo",
        json={"new_name": "../bad"},
    )
    assert resp.status_code == 400
