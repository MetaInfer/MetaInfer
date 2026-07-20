"""Cross-process file locking via ``fcntl.flock``.

Multiple processes (WebUI, orchestrator, QA reconciler) write the same JSON
files under ``<node>/.metainfer/``.  Without an IPC lock, concurrent
``tmp+replace`` or ``open(..., "a")`` operations can interleave or lose
data — see the SSOT audit findings for ``token_budget.json``,
``timeline.jsonl``, ``run.json``, and ``agents.json``.

This module provides a lightweight context manager patterned after
:func:`metainfer.server.tasks._lock` that acquires an exclusive flock on a
per-file ``.lock`` sibling.  Usage::

    from metainfer.server.filelock import lock_file

    with lock_file(state_dir / "token_budget.json"):
        data = _read()
        data["totals"]["total_cost_usd"] += amount
        _write(tmp, data)

NOTE: ``fcntl.flock`` works reliably on local filesystems and some NFSv4
configurations.  In this project's multi-node design, each node writes
only to its own ``nodes/<node_id>/`` subdirectory, so the lock *should
not* be acquired by a different host concurrently.  If you need cross-host
locking, use a higher-level coordinator.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def lock_file(target: Path) -> Iterator[None]:
    """Acquire an exclusive ``fcntl.LOCK_EX`` on ``<target>.lock``.

    The lock file is created (or opened) automatically.  The lock is
    released on context exit, even if an exception occurs inside the
    block.
    """
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()
