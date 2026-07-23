"""Task-local stable-candidate promotion for generated C++ frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional

from .acceptance import audit_iteration, performance_gate


STABLE_CANDIDATE_FILE = "stable_candidate.json"


@dataclass(frozen=True)
class StableCandidate:
    schema_version: int
    iteration: int
    promoted_at: float
    workspace_path: str
    logs_path: str
    oracle_report_path: str
    oracle_report_sha256: str
    performance_required: bool
    gates: Mapping[str, str]


def stable_candidate_path(state_dir: Path) -> Path:
    return state_dir / STABLE_CANDIDATE_FILE


def load_stable_candidate(state_dir: Path) -> Optional[Dict[str, Any]]:
    path = stable_candidate_path(state_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def promote_stable_candidate(
    req: Dict[str, Any],
    state_dir: Path,
    iteration: int,
    iter_dir: Path,
    logs_dir: Path,
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    """Promote an iteration only after its authoritative gates pass.

    Correctness-only tasks promote after C+D. Tasks with an explicit
    performance gate also require E. The iteration may still be open when this
    runs, so final iteration status is deliberately not part of promotion.
    """
    phases = record.get("phases", {}) if isinstance(record, Mapping) else {}
    required_gates = ["C_test", "D_review"]
    perf_required = bool(performance_gate(req, None)["required"])
    if perf_required:
        required_gates.append("E_perf_test")
    gates = {
        gate: str((phases.get(gate) or {}).get("outcome") or "")
        for gate in required_gates
    }
    missing = [gate for gate, outcome in gates.items() if outcome != "ok"]
    if missing:
        return {
            "promoted": False,
            "iteration": iteration,
            "failures": [f"stable gate {gate} is not ok" for gate in missing],
        }

    stages_path = logs_dir / "oracle-stages.json"
    try:
        stages_report = json.loads(stages_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stages_report = None
    stage_items = (
        stages_report.get("stages", [])
        if isinstance(stages_report, Mapping) else []
    )
    c4_stage = next(
        (
            item for item in stage_items
            if isinstance(item, Mapping) and item.get("id") == "C4_full"
        ),
        None,
    )
    if (
        not isinstance(stages_report, Mapping)
        or stages_report.get("full_oracle_completed") is not True
        or not isinstance(c4_stage, Mapping)
        or c4_stage.get("passed") is not True
    ):
        return {
            "promoted": False,
            "iteration": iteration,
            "failures": [
                "stable candidate requires a completed and passing C4 full oracle"
            ],
        }

    audit_record = dict(record)
    audit_record["status"] = "success"
    audit = audit_iteration(
        req, iter_dir, logs_dir, audit_record, require_success_status=False
    )
    if not audit["passed"]:
        return {
            "promoted": False,
            "iteration": iteration,
            "failures": list(audit.get("failures", [])),
        }

    oracle_path = logs_dir / "oracle-report.json"
    try:
        oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    except OSError:
        oracle_hash = ""
    candidate = StableCandidate(
        schema_version=1,
        iteration=iteration,
        promoted_at=time.time(),
        workspace_path=str(iter_dir.resolve()),
        logs_path=str(logs_dir.resolve()),
        oracle_report_path=str(oracle_path.resolve()),
        oracle_report_sha256=oracle_hash,
        performance_required=perf_required,
        gates=gates,
    )
    payload = asdict(candidate)
    path = stable_candidate_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return {"promoted": True, **payload, "audit": audit}


__all__ = [
    "STABLE_CANDIDATE_FILE",
    "load_stable_candidate",
    "promote_stable_candidate",
    "stable_candidate_path",
]
