"""Durable, per-worker optimization guidance queue."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator


WORKER_IDS = tuple(f"worker_{index}" for index in range(4))


def _validate_worker(worker_id: str) -> None:
    if worker_id not in WORKER_IDS:
        raise ValueError(f"unknown worker: {worker_id}")


def _path(root: Path, worker_id: str) -> Path:
    return root / f"{worker_id}.json"


@contextmanager
def _locked(root: Path, worker_id: str) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f".{worker_id}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> list[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _write(path: Path, entries: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def list_guidance(root: Path, worker_id: str) -> list[Dict[str, Any]]:
    _validate_worker(worker_id)
    with _locked(root, worker_id):
        return _read(_path(root, worker_id))


def add_guidance(root: Path, worker_id: str, text: str) -> Dict[str, Any]:
    _validate_worker(worker_id)
    normalized = text.strip()
    if not normalized:
        raise ValueError("guidance cannot be empty")
    if len(normalized) > 4000:
        raise ValueError("guidance must be at most 4000 characters")
    entry: Dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "worker_id": worker_id,
        "text": normalized,
        "source": "manual",
        "status": "pending",
        "created_at": time.time(),
    }
    with _locked(root, worker_id):
        path = _path(root, worker_id)
        entries = _read(path)
        entries.append(entry)
        _write(path, entries)
    return entry


def claim_next_guidance(
    root: Path, worker_id: str, iteration: int
) -> Dict[str, Any] | None:
    """Atomically consume the oldest pending instruction for one worker."""
    _validate_worker(worker_id)
    with _locked(root, worker_id):
        path = _path(root, worker_id)
        entries = _read(path)
        for entry in entries:
            if entry.get("status") != "pending":
                continue
            entry["status"] = "consumed"
            entry["consumed_iteration"] = iteration
            entry["consumed_at"] = time.time()
            _write(path, entries)
            return dict(entry)
    return None
