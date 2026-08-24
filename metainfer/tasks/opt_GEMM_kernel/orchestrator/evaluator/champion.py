"""Persistent per-shape Champion selection for GEMM candidates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .scoring import compare_against_champion


ReportReference = Dict[str, str]


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def make_report_reference(state_root: Path, report_path: Path) -> ReportReference:
    root = state_root.resolve()
    path = report_path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"performance report is outside task state: {path}") from exc
    if not path.is_file():
        raise RuntimeError(f"performance report is missing: {path}")
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_report_reference(
    state_root: Path, reference: Dict[str, Any],
) -> Dict[str, Any]:
    root = state_root.resolve()
    relative = str(reference.get("path") or "").strip()
    expected = str(reference.get("sha256") or "").strip()
    if not relative or not expected:
        raise RuntimeError("performance report reference is incomplete")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("performance report reference escapes task state") from exc
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"performance report is missing: {relative}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"performance report changed: expected {expected}, got {actual}"
        )
    try:
        report = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"performance report is invalid: {relative}") from exc
    if not isinstance(report, dict):
        raise RuntimeError(f"performance report must be an object: {relative}")
    return report


def champion_report_reference(
    state_root: Path, record: Dict[str, Any],
) -> ReportReference:
    reference = record.get("measurement_report")
    if isinstance(reference, dict):
        load_report_reference(state_root, reference)
        return {"path": str(reference["path"]), "sha256": str(reference["sha256"])}

    kind = str(record.get("kind") or "hip")
    iteration = int(record.get("iteration") or 0)
    if kind == "triton":
        path = state_root / "baseline" / "baseline-benchmark-report.json"
    elif iteration == 0:
        path = (
            state_root / "certified" / "initial-hip"
            / "candidate-benchmark-report.json"
        )
    else:
        path = state_root / "logs" / f"{iteration:03d}" / "candidate-benchmark-report.json"
    return make_report_reference(state_root, path)


class ChampionStore:
    def __init__(
        self,
        root: Path,
        noise_threshold: float,
        expected_case_ids: Sequence[str],
    ) -> None:
        self.root = root
        self.state_root = root.parent
        self.submission_dir = root / "submission"
        self.record_path = root / "champion.json"
        self.noise_threshold = noise_threshold
        self.expected_case_ids = list(expected_case_ids)

    def initialize(self, initial_submission: Optional[Path]) -> None:
        """Legacy entry point retained for callers that seed a HIP Champion."""
        if self.record_path.exists():
            self.load()
            return
        if initial_submission is None:
            raise RuntimeError("initial HIP Champion requires a submission")
        report_path = (
            self.state_root / "certified" / "initial-hip"
            / "candidate-benchmark-report.json"
        )
        reference = make_report_reference(self.state_root, report_path)
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(initial_submission, self.submission_dir, dirs_exist_ok=True)
        self._write({
            "schema_version": 2,
            "kind": "hip",
            "iteration": 0,
            "measurement_report": reference,
            "submission_sha256": _tree_digest(self.submission_dir),
            "promoted_at": time.time(),
            "reason": "initial HIP Champion",
        })

    def initialize_triton(self, measurement_report: ReportReference) -> None:
        """Initialize the arena incumbent from frozen Triton measurements."""
        if self.record_path.exists():
            self.load()
            return
        load_report_reference(self.state_root, measurement_report)
        self.root.mkdir(parents=True, exist_ok=True)
        self._write({
            "schema_version": 2,
            "kind": "triton",
            "iteration": 0,
            "measurement_report": dict(measurement_report),
            "promoted_at": time.time(),
            "reason": "certified Triton baseline",
        })

    def load(self) -> Dict[str, Any]:
        if not self.record_path.exists():
            return {"schema_version": 2, "kind": "triton", "iteration": 0}
        record = json.loads(self.record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise RuntimeError("champion record must be an object")
        if record.get("kind", "hip") == "hip":
            expected = record.get("submission_sha256")
            if not expected or not self.submission_dir.is_dir():
                raise RuntimeError("champion submission or digest is missing")
            actual = _tree_digest(self.submission_dir)
            if actual != expected:
                raise RuntimeError("champion submission changed outside promotion")
        reference = champion_report_reference(self.state_root, record)
        load_report_reference(self.state_root, reference)
        return {**record, "measurement_report": reference}

    def consider(
        self,
        iteration: int,
        candidate_dir: Path,
        candidate_report: ReportReference,
        same_round_incumbent_report: ReportReference,
    ) -> tuple[bool, str, Dict[str, Any]]:
        current = self.load()
        candidate = load_report_reference(self.state_root, candidate_report)
        incumbent = load_report_reference(
            self.state_root, same_round_incumbent_report
        )
        promotion_gate = compare_against_champion(
            incumbent.get("cases") or [],
            candidate.get("cases") or [],
            self.expected_case_ids,
            self.noise_threshold,
            strict=True,
        )
        if not promotion_gate.passed:
            return False, "; ".join(promotion_gate.reasons), current

        replacement = self.root / "submission.next"
        if replacement.exists():
            shutil.rmtree(replacement)
        shutil.copytree(candidate_dir, replacement)
        if self.submission_dir.exists():
            shutil.rmtree(self.submission_dir)
        os.replace(replacement, self.submission_dir)
        record = {
            "schema_version": 2,
            "kind": "hip",
            "iteration": iteration,
            "measurement_report": dict(candidate_report),
            "promotion_incumbent_report": dict(same_round_incumbent_report),
            "submission_sha256": _tree_digest(self.submission_dir),
            "promoted_at": time.time(),
            "reason": "every shape beat the same-round Champion hipprof trace beyond the noise gate",
        }
        self._write(record)
        return True, record["reason"], record

    def _write(self, data: Dict[str, Any]) -> None:
        write_json_atomic(self.record_path, data)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"champion submission contains symlink: {path}")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()
