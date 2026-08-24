"""Deterministic per-shape benchmark and promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Sequence

from .spec import AcceptanceSpec, BenchmarkCaseSpec


@dataclass(frozen=True)
class ScoreResult:
    passed: bool
    worst_case_speedup: float
    failed_case_ids: List[str] = field(default_factory=list)
    missing_case_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionResult:
    passed: bool
    noise_threshold: float
    failed_case_ids: List[str] = field(default_factory=list)
    missing_case_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compare_measurements(
    baseline_cases: Sequence[Dict[str, Any]],
    candidate_cases: Sequence[Dict[str, Any]],
    case_specs: Sequence[BenchmarkCaseSpec],
    acceptance: AcceptanceSpec,
) -> ScoreResult:
    """Require every candidate shape to beat the frozen Triton measurement."""
    del acceptance  # Every declared shape is an unconditional hard gate.
    baseline, baseline_errors = _measurement_map(baseline_cases, "baseline")
    candidate, candidate_errors = _measurement_map(candidate_cases, "candidate")
    expected = [case.id for case in case_specs]
    missing = sorted(
        (set(expected) - set(baseline)) | (set(expected) - set(candidate))
    )
    unexpected = sorted((set(baseline) | set(candidate)) - set(expected))
    reasons = [*baseline_errors, *candidate_errors]
    if missing:
        reasons.append(f"missing benchmark cases: {missing}")
    if unexpected:
        reasons.append(f"unexpected benchmark cases: {unexpected}")

    failed: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for spec in case_specs:
        if spec.id not in baseline or spec.id not in candidate:
            continue
        base_ms = baseline[spec.id]
        cand_ms = candidate[spec.id]
        speedup = base_ms / cand_ms
        if cand_ms >= base_ms:
            failed.append(spec.id)
        normalized.append({
            "id": spec.id,
            "baseline_ms": base_ms,
            "candidate_ms": cand_ms,
            "shape": spec.shape,
            "flops": spec.flops,
            "bytes": spec.bytes,
            "baseline_tflops": _rate(spec.flops, base_ms, 1e9),
            "candidate_tflops": _rate(spec.flops, cand_ms, 1e9),
            "baseline_bandwidth_gbps": _rate(spec.bytes, base_ms, 1e6),
            "candidate_bandwidth_gbps": _rate(spec.bytes, cand_ms, 1e6),
            "speedup": speedup,
            "regression": cand_ms / base_ms - 1.0,
            "passed": cand_ms < base_ms,
        })
    if failed:
        reasons.append(f"candidate did not beat baseline for cases: {failed}")
    worst = min((case["speedup"] for case in normalized), default=0.0)
    return ScoreResult(
        passed=not reasons,
        worst_case_speedup=worst,
        failed_case_ids=failed,
        missing_case_ids=missing,
        reasons=reasons,
        cases=normalized,
    )


def compare_against_champion(
    champion_cases: Sequence[Dict[str, Any]],
    candidate_cases: Sequence[Dict[str, Any]],
    expected_case_ids: Sequence[str],
    noise_threshold: float,
    *,
    strict: bool = False,
) -> PromotionResult:
    """Require every shape to improve on Champion beyond the noise floor."""
    champion, champion_errors = _measurement_map(champion_cases, "champion")
    candidate, candidate_errors = _measurement_map(candidate_cases, "candidate")
    expected = list(expected_case_ids)
    missing = sorted(
        (set(expected) - set(champion)) | (set(expected) - set(candidate))
    )
    unexpected = sorted((set(champion) | set(candidate)) - set(expected))
    reasons = [*champion_errors, *candidate_errors]
    if missing:
        reasons.append(f"missing champion comparison cases: {missing}")
    if unexpected:
        reasons.append(f"unexpected champion comparison cases: {unexpected}")

    failed: List[str] = []
    comparisons: List[Dict[str, Any]] = []
    for case_id in expected:
        if case_id not in champion or case_id not in candidate:
            continue
        champion_ms = champion[case_id]
        candidate_ms = candidate[case_id]
        required_ms = champion_ms * (1.0 - noise_threshold)
        passed = candidate_ms < required_ms if strict else candidate_ms <= required_ms
        if not passed:
            failed.append(case_id)
        comparisons.append({
            "id": case_id,
            "champion_ms": champion_ms,
            "candidate_ms": candidate_ms,
            "required_ms": required_ms,
            "speedup_vs_champion": champion_ms / candidate_ms,
            "improvement": 1.0 - candidate_ms / champion_ms,
            "passed": passed,
        })
    if failed:
        reasons.append(
            "candidate did not beat champion beyond noise threshold for cases: "
            f"{failed}"
        )
    return PromotionResult(
        passed=not reasons,
        noise_threshold=noise_threshold,
        failed_case_ids=failed,
        missing_case_ids=missing,
        reasons=reasons,
        cases=comparisons,
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
        if not isinstance(raw, dict):
            errors.append(f"{label} benchmark case must be an object")
            continue
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
