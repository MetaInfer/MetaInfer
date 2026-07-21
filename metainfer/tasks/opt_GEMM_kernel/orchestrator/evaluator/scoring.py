"""Deterministic multi-shape scoring and promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Sequence

from .spec import AcceptanceSpec, BenchmarkCaseSpec


@dataclass(frozen=True)
class ScoreResult:
    passed: bool
    weighted_speedup: float
    critical_regression: float
    missing_case_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_benchmark(
    cases: Sequence[Dict[str, Any]],
    expected_case_ids: Sequence[str],
    acceptance: AcceptanceSpec,
) -> ScoreResult:
    by_id: Dict[str, Dict[str, Any]] = {}
    reasons: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for raw in cases:
        case_id = str(raw.get("id") or "").strip()
        if not case_id or case_id in by_id:
            reasons.append(f"invalid or duplicate benchmark case id: {case_id!r}")
            continue
        try:
            baseline_ms = float(raw["baseline_ms"])
            candidate_ms = float(raw["candidate_ms"])
            weight = float(raw.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            reasons.append(f"case {case_id!r} has invalid timing fields")
            continue
        if (
            not math.isfinite(baseline_ms)
            or not math.isfinite(candidate_ms)
            or not math.isfinite(weight)
            or baseline_ms <= 0
            or candidate_ms <= 0
            or weight <= 0
        ):
            reasons.append(f"case {case_id!r} timings and weight must be positive")
            continue
        item = {
            "id": case_id,
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
            "weight": weight,
            "critical": bool(raw.get("critical", False)),
            "speedup": baseline_ms / candidate_ms,
            "regression": candidate_ms / baseline_ms - 1.0,
        }
        by_id[case_id] = item
        normalized.append(item)

    missing = sorted(set(expected_case_ids) - set(by_id))
    unexpected = sorted(set(by_id) - set(expected_case_ids))
    if missing and acceptance.require_all_cases:
        reasons.append(f"missing benchmark cases: {missing}")
    if unexpected:
        reasons.append(f"unexpected benchmark cases: {unexpected}")

    expected = [by_id[cid] for cid in expected_case_ids if cid in by_id]
    base_work = sum(item["weight"] * item["baseline_ms"] for item in expected)
    candidate_work = sum(item["weight"] * item["candidate_ms"] for item in expected)
    weighted = base_work / candidate_work if candidate_work > 0 else 0.0
    critical = [item["regression"] for item in expected if item["critical"]]
    worst_critical = max(critical, default=0.0)

    if weighted < acceptance.min_weighted_speedup:
        reasons.append(
            f"weighted speedup {weighted:.6f} < minimum {acceptance.min_weighted_speedup:.6f}"
        )
    if worst_critical > acceptance.max_critical_regression:
        reasons.append(
            f"critical regression {worst_critical:.2%} exceeds "
            f"{acceptance.max_critical_regression:.2%}"
        )
    return ScoreResult(
        passed=not reasons,
        weighted_speedup=weighted,
        critical_regression=worst_critical,
        missing_case_ids=missing,
        reasons=reasons,
        cases=normalized,
    )


def compare_measurements(
    baseline_cases: Sequence[Dict[str, Any]],
    candidate_cases: Sequence[Dict[str, Any]],
    case_specs: Sequence[BenchmarkCaseSpec],
    acceptance: AcceptanceSpec,
) -> ScoreResult:
    """Compare independent baseline/candidate measurements.

    Weights and criticality come from the frozen task spec, never from either
    measurement report. This prevents per-iteration workload drift.
    """
    baseline, baseline_errors = _measurement_map(baseline_cases, "baseline")
    candidate, candidate_errors = _measurement_map(candidate_cases, "candidate")
    expected = [case.id for case in case_specs]
    missing = sorted(
        (set(expected) - set(baseline)) | (set(expected) - set(candidate))
    )
    unexpected = sorted((set(baseline) | set(candidate)) - set(expected))
    reasons = [*baseline_errors, *candidate_errors]
    if missing and acceptance.require_all_cases:
        reasons.append(f"missing benchmark cases: {missing}")
    if unexpected:
        reasons.append(f"unexpected benchmark cases: {unexpected}")

    normalized: List[Dict[str, Any]] = []
    for spec in case_specs:
        if spec.id not in baseline or spec.id not in candidate:
            continue
        base_ms = baseline[spec.id]
        cand_ms = candidate[spec.id]
        normalized.append({
            "id": spec.id,
            "baseline_ms": base_ms,
            "candidate_ms": cand_ms,
            "weight": spec.weight,
            "critical": spec.critical,
            "shape": spec.shape,
            "flops": spec.flops,
            "bytes": spec.bytes,
            "baseline_tflops": _rate(spec.flops, base_ms, 1e9),
            "candidate_tflops": _rate(spec.flops, cand_ms, 1e9),
            "baseline_bandwidth_gbps": _rate(spec.bytes, base_ms, 1e6),
            "candidate_bandwidth_gbps": _rate(spec.bytes, cand_ms, 1e6),
            "speedup": base_ms / cand_ms,
            "regression": cand_ms / base_ms - 1.0,
        })

    base_work = sum(item["weight"] * item["baseline_ms"] for item in normalized)
    candidate_work = sum(item["weight"] * item["candidate_ms"] for item in normalized)
    weighted = base_work / candidate_work if candidate_work > 0 else 0.0
    worst_critical = max(
        (item["regression"] for item in normalized if item["critical"]), default=0.0
    )
    if weighted < acceptance.min_weighted_speedup:
        reasons.append(
            f"weighted speedup {weighted:.6f} < minimum {acceptance.min_weighted_speedup:.6f}"
        )
    if worst_critical > acceptance.max_critical_regression:
        reasons.append(
            f"critical regression {worst_critical:.2%} exceeds "
            f"{acceptance.max_critical_regression:.2%}"
        )
    return ScoreResult(
        passed=not reasons,
        weighted_speedup=weighted,
        critical_regression=worst_critical,
        missing_case_ids=missing,
        reasons=reasons,
        cases=normalized,
    )


def _rate(work: float | None, latency_ms: float, scale: float) -> float | None:
    if work is None:
        return None
    return work / latency_ms / scale


def _measurement_map(
    cases: Sequence[Dict[str, Any]], label: str,
) -> tuple[Dict[str, float], List[str]]:
    values: Dict[str, float] = {}
    errors: List[str] = []
    for raw in cases:
        case_id = str(raw.get("id") or "").strip()
        if not case_id or case_id in values:
            errors.append(f"invalid or duplicate {label} case id: {case_id!r}")
            continue
        try:
            latency_ms = float(raw["latency_ms"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} case {case_id!r} has invalid latency_ms")
            continue
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            errors.append(f"{label} case {case_id!r} latency_ms must be positive")
            continue
        values[case_id] = latency_ms
    return values, errors
