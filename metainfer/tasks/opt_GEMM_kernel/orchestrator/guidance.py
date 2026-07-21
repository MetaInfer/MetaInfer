"""Durable task-local channel for live human optimization guidance."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List


class GuidanceError(ValueError):
    pass


class GuidanceStore:
    """Cross-process queue shared by the WebUI and GEMM orchestrator.

    Guidance is consumed only at planner/implementer launch boundaries. This
    preserves the non-interactive agent process model while ensuring a message
    submitted during compilation/evaluation survives until an agent can act on
    it.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "guidance.json"
        self.lock_path = root / "guidance.lock"

    def submit(self, text: str) -> Dict[str, Any]:
        value = str(text or "").strip()
        if not value:
            raise GuidanceError("guidance text is required")
        if len(value) > 8_000:
            raise GuidanceError("guidance text must not exceed 8000 characters")
        now = time.time()
        item = {
            "id": f"g-{time.time_ns()}-{secrets.token_hex(3)}",
            "text": value,
            "status": "pending",
            "created_at": now,
            "applied_at": None,
            "applied_iteration": None,
            "applied_phase": None,
            "applied_role": None,
        }
        with self._locked():
            data = self._read_unlocked()
            data["items"].append(item)
            self._write_unlocked(data)
        return dict(item)

    def snapshot(self) -> Dict[str, Any]:
        with self._locked():
            data = self._read_unlocked()
        items = list(data["items"])
        return {
            "schema_version": 1,
            "pending_count": sum(item.get("status") == "pending" for item in items),
            "items": items,
        }

    def consume(self, *, iteration: int, phase: str, role: str) -> List[Dict[str, Any]]:
        if role not in {"planner", "implementer"}:
            return []
        now = time.time()
        consumed: List[Dict[str, Any]] = []
        with self._locked():
            data = self._read_unlocked()
            for item in data["items"]:
                if item.get("status") != "pending":
                    continue
                item.update({
                    "status": "applied",
                    "applied_at": now,
                    "applied_iteration": int(iteration),
                    "applied_phase": phase,
                    "applied_role": role,
                })
                consumed.append(dict(item))
            if consumed:
                self._write_unlocked(data)
        return consumed

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "items": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GuidanceError(f"invalid guidance store: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise GuidanceError("guidance store must use schema_version=1")
        if not isinstance(data.get("items"), list):
            raise GuidanceError("guidance store items must be a list")
        return data

    def _write_unlocked(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
