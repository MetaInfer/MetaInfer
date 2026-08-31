"""Cross-task experience derived only from trusted iteration records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _same_shape(record: Dict[str, Any], shape: Dict[str, Any]) -> bool:
    recorded = record.get("shape")
    if not isinstance(recorded, dict):
        return False
    try:
        return all(
            int(recorded[key]) == int(shape[key])
            for key in ("M", "N", "K")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _classification(record: Dict[str, Any]) -> str | None:
    if (
        record.get("accepted") is True
        and record.get("correctness_passed") is True
    ):
        return "accepted_correct"
    if (
        record.get("build_success") is True
        and record.get("correctness_passed") is False
        and isinstance(record.get("speedup"), (int, float))
        and float(record["speedup"]) > 1.0
    ):
        return "faster_incorrect_repairable"
    if (
        record.get("build_success") is True
        and record.get("correctness_passed") is True
    ):
        return "measured_correct_rejection"
    if record.get("build_success") is False:
        return "compile_or_agent_failure"
    return None


def load_verified_experience(
    kernel_repos_root: Path | None,
    shape: Dict[str, Any],
    *,
    exclude_repo: Path | None = None,
    limit: int = 12,
) -> list[Dict[str, Any]]:
    """Load exact-shape facts without trusting generated Skill prose."""
    if kernel_repos_root is None or not kernel_repos_root.is_dir():
        return []
    candidates: list[tuple[float, Path]] = []
    for path in kernel_repos_root.glob(
        "*/candidates/**/iteration.json"
    ):
        try:
            if exclude_repo is not None and path.is_relative_to(exclude_repo):
                continue
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    evidence: list[Dict[str, Any]] = []
    for _, path in sorted(candidates, reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or not _same_shape(record, shape):
            continue
        classification = _classification(record)
        if classification is None:
            continue
        evidence.append({
            "classification": classification,
            "shape_id": record.get("shape_id"),
            "iteration": record.get("iteration"),
            "proposed_change_not_verified_fact": record.get("hypothesis"),
            "metrics": record.get("metrics") or {},
            "baseline_us": record.get("baseline_us"),
            "speedup": record.get("speedup"),
            "accepted": record.get("accepted"),
            "build_success": record.get("build_success"),
            "correctness_passed": record.get("correctness_passed"),
            "record_path": str(path),
            "trust_rule": (
                "Only classification and measured fields are trusted. "
                "The proposed-change text may contain an Agent hypothesis."
            ),
        })
        if len(evidence) >= limit:
            break
    return evidence
