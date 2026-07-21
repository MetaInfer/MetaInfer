"""Persistent champion/challenger selection for GEMM candidates."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ChampionStore:
    def __init__(self, root: Path, noise_threshold: float) -> None:
        self.root = root
        self.submission_dir = root / "submission"
        self.record_path = root / "champion.json"
        self.noise_threshold = noise_threshold

    def initialize(self, initial_submission: Optional[Path]) -> None:
        if self.record_path.exists():
            self.load()
            return
        self.root.mkdir(parents=True, exist_ok=True)
        if initial_submission and initial_submission.is_dir():
            shutil.copytree(initial_submission, self.submission_dir, dirs_exist_ok=True)
        self._write({
            "iteration": 0,
            "weighted_speedup": 1.0,
            "submission_sha256": _tree_digest(self.submission_dir),
            "promoted_at": time.time(),
            "reason": "initial baseline",
        })

    def load(self) -> Dict[str, Any]:
        if not self.record_path.exists():
            return {"iteration": 0, "weighted_speedup": 1.0}
        record = json.loads(self.record_path.read_text(encoding="utf-8"))
        expected = record.get("submission_sha256")
        if not expected or not self.submission_dir.is_dir():
            raise RuntimeError("champion submission or digest is missing")
        actual = _tree_digest(self.submission_dir)
        if actual != expected:
            raise RuntimeError("champion submission changed outside promotion")
        return record

    def consider(
        self,
        iteration: int,
        candidate_dir: Path,
        score: Dict[str, Any],
    ) -> tuple[bool, str, Dict[str, Any]]:
        current = self.load()
        candidate_speedup = float(score.get("weighted_speedup", 0.0))
        current_speedup = float(current.get("weighted_speedup", 1.0))
        if not bool(score.get("passed")):
            return False, "acceptance gates failed", current
        required = current_speedup * (1.0 + self.noise_threshold)
        if candidate_speedup < required:
            return False, (
                f"speedup {candidate_speedup:.6f} did not beat champion "
                f"{current_speedup:.6f} by noise threshold {self.noise_threshold:.2%}"
            ), current

        replacement = self.root / "submission.next"
        if replacement.exists():
            shutil.rmtree(replacement)
        shutil.copytree(candidate_dir, replacement)
        if self.submission_dir.exists():
            shutil.rmtree(self.submission_dir)
        os.replace(replacement, self.submission_dir)
        record = {
            "iteration": iteration,
            "weighted_speedup": candidate_speedup,
            "critical_regression": float(score.get("critical_regression", 0.0)),
            "submission_sha256": _tree_digest(self.submission_dir),
            "promoted_at": time.time(),
            "reason": "candidate passed all gates and beat the current champion",
        }
        self._write(record)
        return True, record["reason"], record

    def _write(self, data: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.record_path)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"champion submission contains symlink: {path}")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()
