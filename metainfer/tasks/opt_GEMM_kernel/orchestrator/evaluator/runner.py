"""Run frozen evaluator commands and validate their structured reports."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .scoring import compare_measurements
from .spec import FrozenEvaluatorBundle


class EvaluationError(RuntimeError):
    pass


@dataclass
class EvaluationResult:
    phase: str
    passed: bool
    report: Dict[str, Any]
    failure: Optional[str] = None
    infra_failure: bool = False


class EvaluatorRunner:
    def __init__(
        self,
        bundle: FrozenEvaluatorBundle,
        *,
        private_env: Optional[Mapping[str, str]] = None,
        private_verifier: Optional[Callable[[], None]] = None,
    ) -> None:
        self.bundle = bundle
        self.private_env = dict(private_env or {})
        self.private_verifier = private_verifier

    def run(
        self,
        phase: str,
        submission_dir: Path,
        artifact_dir: Path,
        report_dir: Path,
        *,
        role: str,
        build_fingerprint: str,
        baseline_report: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        if phase not in self.bundle.spec.commands:
            raise EvaluationError(f"unknown evaluator phase: {phase}")
        if role not in {"baseline", "candidate"}:
            raise EvaluationError(f"invalid evaluation role: {role}")
        self.bundle.verify()
        if self.private_verifier is not None:
            self.private_verifier()
        _validate_submission_tree(submission_dir)
        if not artifact_dir.is_dir():
            raise EvaluationError(f"build artifact directory does not exist: {artifact_dir}")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{role}-{phase}-report.json"
        command = self.bundle.spec.commands[phase]
        values = {
            "bundle_dir": str(self.bundle.root),
            "submission_dir": str(submission_dir.resolve()),
            "report_path": str(report_path.resolve()),
            "phase": phase,
            "role": role,
            "artifact_dir": str(artifact_dir.resolve()),
            "build_fingerprint": build_fingerprint,
        }
        try:
            argv = [part.format_map(values) for part in command.argv]
        except KeyError as exc:
            raise EvaluationError(f"unsupported command placeholder: {exc}") from exc
        env = dict(os.environ)
        env.update({
            "METAINFER_EVALUATOR_BUNDLE": values["bundle_dir"],
            "METAINFER_SUBMISSION_DIR": values["submission_dir"],
            "METAINFER_REPORT_PATH": values["report_path"],
            "METAINFER_EVALUATION_PHASE": phase,
            "METAINFER_EVALUATION_ROLE": role,
            "METAINFER_BUILD_ARTIFACT_DIR": values["artifact_dir"],
            "METAINFER_BUILD_FINGERPRINT": build_fingerprint,
            "METAINFER_BENCHMARK_PROTOCOL": json.dumps(
                self.bundle.spec.benchmark_protocol, sort_keys=True
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        })
        env.update(self.private_env)
        stdout_path = report_dir / f"{role}-{phase}.stdout.log"
        stderr_path = report_dir / f"{role}-{phase}.stderr.log"
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.bundle.root),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=command.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return EvaluationResult(phase, False, {}, f"evaluator infrastructure failure: {exc}", True)
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        self.bundle.verify()
        if self.private_verifier is not None:
            self.private_verifier()
        try:
            _validate_submission_tree(submission_dir)
        except EvaluationError as exc:
            return EvaluationResult(phase, False, {}, str(exc), True)
        if proc.returncode != 0:
            return EvaluationResult(
                phase, False, {}, f"{phase} evaluator exited {proc.returncode}", False
            )
        if not report_path.is_file():
            return EvaluationResult(phase, False, {}, f"{phase} evaluator produced no report", True)
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EvaluationResult(phase, False, {}, f"invalid {phase} report: {exc}", True)
        if not isinstance(report, dict):
            return EvaluationResult(phase, False, {}, f"{phase} report must be an object", True)
        report["evaluation_role"] = role
        report["build_fingerprint"] = build_fingerprint
        return self._validate(phase, report, role=role, baseline_report=baseline_report)

    def _validate(
        self,
        phase: str,
        report: Dict[str, Any],
        *,
        role: str = "candidate",
        baseline_report: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        if phase == "correctness":
            raw_cases = report.get("cases")
            if not isinstance(raw_cases, list):
                return EvaluationResult(phase, False, report, "correctness report has no cases", True)
            ids = [str(case.get("id") or "") for case in raw_cases if isinstance(case, dict)]
            duplicate = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
            by_id = {
                str(case.get("id") or ""): case
                for case in raw_cases
                if isinstance(case, dict) and case.get("id")
            }
            expected = self.bundle.spec.correctness_case_ids
            missing = sorted(set(expected) - set(by_id))
            unexpected = sorted(set(by_id) - set(expected))
            failed = [cid for cid in expected if cid in by_id and by_id[cid].get("passed") is not True]
            passed = (
                report.get("passed") is True
                and not missing
                and not failed
                and not duplicate
                and not unexpected
            )
            report["summary"] = {
                "expected": len(expected),
                "missing": missing,
                "failed": failed,
                "duplicate": duplicate,
                "unexpected": unexpected,
            }
            reason = None if passed else _reason(
                report,
                f"missing={missing}, failed={failed}, duplicate={duplicate}, unexpected={unexpected}",
            )
            return EvaluationResult(phase, passed, report, reason)
        if phase == "benchmark":
            raw_cases = report.get("cases")
            if not isinstance(raw_cases, list):
                return EvaluationResult(phase, False, report, "benchmark report has no cases", True)
            if not isinstance(report.get("methodology"), dict) or not report["methodology"]:
                return EvaluationResult(
                    phase, False, report, "benchmark report requires methodology metadata", True
                )
            if report.get("methodology") != self.bundle.spec.benchmark_protocol:
                return EvaluationResult(
                    phase, False, report,
                    "benchmark methodology differs from frozen task protocol", False,
                )
            measurement_errors = _validate_measurement_cases(
                raw_cases, self.bundle.spec.benchmark_case_ids
            )
            if measurement_errors:
                return EvaluationResult(
                    phase, False, report, "; ".join(measurement_errors), False
                )
            if role == "baseline":
                passed = report.get("passed") is True
                report["summary"] = {
                    "expected": len(self.bundle.spec.benchmark_case_ids),
                    "measured": len(raw_cases),
                }
                return EvaluationResult(
                    phase, passed, report,
                    None if passed else _reason(report, "baseline benchmark failed"),
                )
            if not isinstance(baseline_report, dict):
                return EvaluationResult(
                    phase, False, report, "candidate benchmark has no frozen baseline", True
                )
            if baseline_report.get("methodology") != report.get("methodology"):
                return EvaluationResult(
                    phase, False, report, "benchmark methodology differs from frozen baseline", False
                )
            score = compare_measurements(
                baseline_report.get("cases") or [],
                raw_cases,
                self.bundle.spec.benchmark_cases,
                self.bundle.spec.acceptance,
            )
            report["score"] = score.to_dict()
            passed = report.get("passed") is True and score.passed
            return EvaluationResult(
                phase,
                passed,
                report,
                None if passed else "; ".join(score.reasons) or _reason(report, "benchmark failed"),
            )
        raise EvaluationError(f"unsupported evaluator phase: {phase}")


def _reason(report: Dict[str, Any], default: str) -> str:
    return str(report.get("reason") or report.get("error") or default)


def _validate_submission_tree(root: Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise EvaluationError(f"submission directory does not exist: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise EvaluationError(f"submission may not contain symlinks: {path.relative_to(root)}")


def _validate_measurement_cases(cases: list, expected_ids: list[str]) -> list[str]:
    seen: Dict[str, float] = {}
    errors = []
    for case in cases:
        if not isinstance(case, dict):
            errors.append("benchmark case must be an object")
            continue
        case_id = str(case.get("id") or "").strip()
        if not case_id or case_id in seen:
            errors.append(f"invalid or duplicate benchmark case id: {case_id!r}")
            continue
        try:
            latency = float(case["latency_ms"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"benchmark case {case_id!r} has invalid latency_ms")
            continue
        if not math.isfinite(latency) or latency <= 0:
            errors.append(f"benchmark case {case_id!r} latency_ms must be positive")
            continue
        seen[case_id] = latency
    missing = sorted(set(expected_ids) - set(seen))
    unexpected = sorted(set(seen) - set(expected_ids))
    if missing:
        errors.append(f"missing benchmark cases: {missing}")
    if unexpected:
        errors.append(f"unexpected benchmark cases: {unexpected}")
    return errors
