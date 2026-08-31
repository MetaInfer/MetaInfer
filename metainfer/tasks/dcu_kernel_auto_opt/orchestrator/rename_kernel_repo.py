"""Rename the kernel repository of one task and repair every reference.

Mutates, in order:

* ``kernel-repos/<old>`` -> ``kernel-repos/<new>`` (the git repo directory);
* git worktree administrative files (the repo moved under its linked
  worktrees; ``git worktree repair`` rewrites the ``.git`` pointer files);
* ``<workspace>/main`` symlink (relative, same depth as before);
* task ``requirements.json``: ``label`` + ``target_repo_path``;
* workspace ``plan.json``: ``kernel_repo`` absolute path;
* state ``timeline.jsonl``: one ``kernel_repo_renamed`` event.

The task id (state/workspace directory names) never changes: the WebUI and
the orchestrator key on it, and the operator shapes inside the repo are
name-agnostic. Refuses to rename a task whose orchestrator is still running
(``run.json`` not finished or a live ``orchestrator.pid``).

The kernel-repos root is resolved exactly like ``config._kernel_repos_root()``
(``METAINFER_KERNEL_REPOS`` override, else sibling of the MetaInfer root), so
a rename stays consistent with how tasks resolve repositories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .config import _kernel_repos_root

# Safe repository directory names: no path separators, no traversal, and the
# same charset the New Task form accepts for repository names.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_timeline(state_dir: Path, event_type: str, payload: Dict[str, Any]) -> None:
    path = state_dir / "timeline.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"ts": time.time(), "type": event_type, "payload": payload},
                ensure_ascii=False,
            )
            + "\n"
        )


def _task_running(state_dir: Path) -> bool:
    """True when the orchestrator may still be executing this task."""
    run = _load_json(state_dir / "run.json", None)
    if run is None:
        # No run state yet: only a live orchestrator PID blocks a rename.
        pid_path = state_dir / "orchestrator.pid"
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
        if pid > 0:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                pass
        return False
    if run.get("finished") is True:
        return False
    # Started but not marked finished: the orchestrator may still be live,
    # so renaming the repo under it is unsafe.
    return True


def _repair_worktrees(repo: Path) -> None:
    """Fix git worktree admin files after the repo directory moved.

    ``git worktree repair`` (run in the moved repo) rewrites every linked
    worktree's ``.git`` pointer file, which embeds the old repo path. If git
    is unavailable or misses one, fall back to a deterministic rewrite: each
    linked worktree's ``.git`` file must contain
    ``gitdir: <repo>/.git/worktrees/<name>``.
    """
    try:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "repair"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    _repair_worktrees_manual(repo)


def _repair_worktrees_manual(repo: Path) -> None:
    """Deterministic fallback: rewrite every linked worktree ``.git`` file."""
    worktrees_dir = repo / ".git" / "worktrees"
    if not worktrees_dir.is_dir():
        return
    for entry in worktrees_dir.iterdir():
        gitdir_file = entry / "gitdir"
        if not gitdir_file.is_file():
            continue
        worktree_git = Path(gitdir_file.read_text(encoding="utf-8").strip())
        if not worktree_git.is_file():
            continue
        expected = f"gitdir: {repo / '.git' / 'worktrees' / entry.name}"
        if worktree_git.read_text(encoding="utf-8").strip() == expected:
            continue
        worktree_git.write_text(expected + "\n", encoding="utf-8")


def rename_kernel_repo(
    workspace_dir: Path | str,
    new_name: str,
    *,
    state_dir: Path | str | None = None,
    kernel_repos_root: Path | str | None = None,
) -> Dict[str, Any]:
    """Rename the kernel repository behind one task workspace.

    Returns a summary dict; raises ``ValueError`` on invalid input and
    ``RuntimeError`` when the task is still running or the repo is busy.
    """
    workspace = Path(workspace_dir)
    main = workspace / "main"
    if not main.is_symlink():
        raise ValueError(
            f"workspace main is not a symlink to a kernel repo: {main}"
        )
    old_repo = main.resolve()
    if not old_repo.is_dir():
        raise ValueError(f"kernel repo does not exist: {old_repo}")

    name = (new_name or "").strip()
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            "repository name must match [A-Za-z0-9][A-Za-z0-9._-]* "
            f"(no separators or traversal), got {new_name!r}"
        )

    root = (
        Path(kernel_repos_root).expanduser().resolve()
        if kernel_repos_root is not None
        else _kernel_repos_root()
    )
    new_repo = root / name
    if new_repo == old_repo:
        raise ValueError(f"repository is already named {name!r}")
    if new_repo.exists():
        raise ValueError(f"kernel repo already exists: {new_repo}")

    state: Path | None = Path(state_dir) if state_dir is not None else None
    if state is not None and _task_running(state):
        raise RuntimeError(
            "task is still running; stop or wait for it before renaming "
            "its kernel repository"
        )

    old_path = str(old_repo)
    new_path = str(new_repo)

    # 1. Move the repository directory (same filesystem).
    shutil.move(old_path, new_path)

    # 2. Repair git worktree administrative files.
    _repair_worktrees(new_repo)

    # 3. Re-point the workspace main symlink (relative, same depth).
    main.unlink(missing_ok=True)
    relative_target = os.path.relpath(new_repo, start=main.parent)
    main.symlink_to(relative_target, target_is_directory=True)

    updated: list[str] = []

    # 4. Task references: requirements.json (label + target_repo_path).
    if state is not None:
        requirements_path = state / "requirements.json"
        requirements = _load_json(requirements_path, {}) or {}
        changed = False
        if requirements.get("target_repo_path") == old_repo.name:
            requirements["target_repo_path"] = name
            changed = True
        if requirements.get("label") == old_repo.name:
            requirements["label"] = name
            changed = True
        if changed:
            _write_json(requirements_path, requirements)
            updated.append(str(requirements_path))

    # 5. Workspace plan.json: kernel_repo absolute path.
    plan_path = workspace / "plan.json"
    plan = _load_json(plan_path, {}) or {}
    if plan.get("kernel_repo") == old_path:
        plan["kernel_repo"] = new_path
        _write_json(plan_path, plan)
        updated.append(str(plan_path))

    # 6. Timeline event for the WebUI.
    if state is not None:
        _append_timeline(state, "kernel_repo_renamed", {
            "task_id": state.name,
            "old_name": old_repo.name,
            "new_name": name,
            "old_repo": old_path,
            "new_repo": new_path,
        })

    return {
        "renamed": True,
        "old_name": old_repo.name,
        "new_name": name,
        "old_repo": old_path,
        "new_repo": new_path,
        "updated_references": updated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dcu-kernel-auto-opt-rename-kernel-repo",
        description=(
            "Rename the kernel repository of one dcu-kernel-auto-opt task "
            "and repair every reference (worktrees, main symlink, "
            "requirements.json, plan.json)."
        ),
    )
    parser.add_argument("state_dir", type=Path)
    parser.add_argument("workspace_dir", type=Path)
    parser.add_argument("new_name", type=str)
    parser.add_argument(
        "--kernel-repos-root",
        type=Path,
        default=None,
        help=(
            "Override the kernel-repos root (defaults to "
            "METAINFER_KERNEL_REPOS or the sibling of the MetaInfer root)."
        ),
    )
    args = parser.parse_args()
    try:
        result = rename_kernel_repo(
            args.workspace_dir,
            args.new_name,
            state_dir=args.state_dir,
            kernel_repos_root=args.kernel_repos_root,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
