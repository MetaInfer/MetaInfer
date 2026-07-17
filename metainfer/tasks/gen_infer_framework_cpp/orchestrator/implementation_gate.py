"""Deterministic B-stage delivery gate for the native C++ task.

The implementer is not allowed to be its own only source of truth.  This gate
cross-checks its structured report against the validated A-stage manifest,
the files that actually exist, and (for incremental iterations) the inherited
baseline.  It intentionally stays lightweight: the immutable C oracle still
owns end-to-end correctness, while this gate catches incomplete B deliveries
before an expensive model boot.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


REPORT_NAME = "implementation_report.json"
REPORT_SCHEMA_VERSION = 1
_REQUIRED_VERIFICATION = ("build", "reference_differential", "end_to_end")
_IMPLEMENTED = {"implemented", "verified", "inherited_verified"}


@dataclass(frozen=True)
class ImplementationGateResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    expected_files: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "expected_files": list(self.expected_files),
            "changed_files": list(self.changed_files),
            "report_path": self.report_path,
        }

    def diagnostics(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class NativeBuildGateResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "steps": list(self.steps),
            "report_path": self.report_path,
        }

    def diagnostics(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def execute_native_build_gate(
    iter_dir: Path,
    report_dir: Path,
    *,
    build_type: str = "Release",
    timeout_s: int = 900,
) -> NativeBuildGateResult:
    """Independently configure, build, and run native CTest targets.

    Commands are fixed by the orchestrator rather than taken from the agent's
    report. This makes B's self-report useful metadata, not the source of
    truth for whether the tree actually compiles and its registered tests run.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "implementation-gate-execution.json"
    errors: List[str] = []
    steps: List[Dict[str, Any]] = []
    cmake = shutil.which("cmake")
    ctest = shutil.which("ctest")
    if cmake is None:
        errors.append("cmake executable is not available")
    if ctest is None:
        errors.append("ctest executable is not available")
    if not (iter_dir / "CMakeLists.txt").is_file():
        errors.append("CMakeLists.txt is missing")

    deadline = time.monotonic() + max(1, int(timeout_s))
    commands = []
    if not errors:
        commands = [
            (
                "configure",
                [cmake, "-S", ".", "-B", "build", f"-DCMAKE_BUILD_TYPE={build_type}"],
            ),
            ("build", [cmake, "--build", "build", "--parallel"]),
            (
                "ctest",
                [ctest, "--test-dir", "build", "--output-on-failure", "--no-tests=error"],
            ),
        ]

    for name, command in commands:
        remaining = max(1, int(deadline - time.monotonic()))
        log_path = report_dir / f"implementation-gate-{name}.log"
        started = time.monotonic()
        returncode: Optional[int] = None
        error: Optional[str] = None
        try:
            completed = subprocess.run(
                command,
                cwd=str(iter_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=remaining,
                check=False,
            )
            returncode = completed.returncode
            output = completed.stdout or ""
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            error = f"{name} timed out after {remaining}s"
        except OSError as exc:
            output = ""
            error = f"{name} could not start: {exc}"
        duration = time.monotonic() - started
        try:
            log_path.write_text(output, encoding="utf-8")
        except OSError as exc:
            error = error or f"cannot write {name} log: {exc}"
        step = {
            "name": name,
            "command": [str(part) for part in command],
            "returncode": returncode,
            "duration_s": round(duration, 3),
            "log_path": str(log_path),
            "log_bytes": len(output.encode("utf-8", errors="replace")),
            "error": error,
        }
        steps.append(step)
        if error is not None or returncode != 0:
            errors.append(error or f"{name} exited with code {returncode}")
            break

    result = NativeBuildGateResult(
        passed=not errors,
        errors=errors,
        steps=steps,
        report_path=str(report_path),
    )
    try:
        report_path.write_text(result.diagnostics(), encoding="utf-8")
    except OSError as exc:
        return NativeBuildGateResult(
            passed=False,
            errors=[*errors, f"cannot write native build gate report: {exc}"],
            steps=steps,
            report_path=str(report_path),
        )
    return result


def validate_implementation(iter_dir: Path, iteration: int) -> ImplementationGateResult:
    errors: List[str] = []
    manifest = _read_json(iter_dir / "plan_manifest.json", "plan_manifest.json", errors)
    report_path = iter_dir / REPORT_NAME
    report = _read_json(report_path, REPORT_NAME, errors)
    if not manifest or not report:
        return ImplementationGateResult(
            False, errors, report_path=str(report_path),
        )

    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append(
            f"{REPORT_NAME} schema_version must be {REPORT_SCHEMA_VERSION}"
        )
    if report.get("iteration") != iteration:
        errors.append(f"{REPORT_NAME} iteration must be {iteration}")
    if report.get("status") != "complete":
        errors.append(f"{REPORT_NAME} status must be 'complete'")

    expected = _expected_delivery_files(manifest, errors)
    for relative in sorted(expected):
        path = iter_dir / relative
        if not path.is_file():
            errors.append(f"planned delivery file is missing after B: {relative}")

    delivery_items = manifest.get("delivery_items") or []
    planned_items: Dict[str, Dict[str, Set[str]]] = {}
    if not isinstance(delivery_items, list) or not delivery_items:
        errors.append("plan_manifest.json delivery_items must be a non-empty array")
    else:
        for index, item in enumerate(delivery_items):
            if not isinstance(item, dict):
                errors.append(f"delivery_items[{index}] must be an object")
                continue
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                errors.append(f"delivery_items[{index}] must have a stable id")
                continue
            if item_id in planned_items:
                errors.append(f"duplicate planned delivery item id: {item_id}")
                continue
            files = item.get("files")
            tests = item.get("tests")
            if not _relative_paths(files, allow_empty=False):
                errors.append(f"planned delivery item {item_id!r} has invalid files")
                files = []
            if not isinstance(tests, list) or not any(
                str(test).strip() for test in tests
            ):
                errors.append(f"planned delivery item {item_id!r} has invalid tests")
                tests = []
            planned_items[item_id] = {
                "files": {str(path) for path in files},
                "tests": {str(test).strip() for test in tests if str(test).strip()},
            }

    plan_items = report.get("plan_items")
    reported_files: Set[str] = set()
    reported_ids: Set[str] = set()
    if not isinstance(plan_items, list) or not plan_items:
        errors.append(f"{REPORT_NAME} plan_items must be a non-empty array")
    else:
        for index, item in enumerate(plan_items):
            if not isinstance(item, dict):
                errors.append(f"plan_items[{index}] must be an object")
                continue
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                errors.append(f"plan_items[{index}] must have a stable id")
            elif item_id in reported_ids:
                errors.append(f"duplicate implementation report item id: {item_id}")
            else:
                reported_ids.add(item_id)
            status = str(item.get("status") or "")
            if status not in _IMPLEMENTED:
                errors.append(
                    f"plan item {item_id or index!r} is not implemented: {status!r}"
                )
            files = item.get("files")
            item_files: Set[str] = set()
            if not _relative_paths(files, allow_empty=False):
                errors.append(f"plan item {item_id or index!r} must name relative files")
            else:
                item_files = {str(path) for path in files}
                reported_files.update(item_files)
                for relative in sorted(item_files):
                    if not (iter_dir / relative).is_file():
                        errors.append(
                            f"implementation report names a missing file: {relative}"
                        )
            tests = item.get("tests")
            if not isinstance(tests, list) or not any(str(test).strip() for test in tests):
                errors.append(f"plan item {item_id or index!r} must name test evidence")
                item_tests: Set[str] = set()
            else:
                item_tests = {
                    str(test).strip() for test in tests if str(test).strip()
                }

            planned = planned_items.get(item_id)
            if item_id and planned is None:
                errors.append(f"implementation report has unplanned item id: {item_id}")
            elif planned is not None:
                missing_item_files = sorted(planned["files"] - item_files)
                if missing_item_files:
                    errors.append(
                        f"plan item {item_id!r} does not account for its planned files: "
                        + ", ".join(missing_item_files)
                    )
                missing_item_tests = sorted(planned["tests"] - item_tests)
                if missing_item_tests:
                    errors.append(
                        f"plan item {item_id!r} does not report its planned tests: "
                        + ", ".join(missing_item_tests)
                    )

    missing_from_report = sorted(expected - reported_files)
    if missing_from_report:
        errors.append(
            "implementation report does not account for planned files: "
            + ", ".join(missing_from_report)
        )

    expected_ids = set(planned_items)
    missing_ids = sorted(expected_ids - reported_ids)
    if missing_ids:
        errors.append(
            "implementation report is missing planned delivery item ids: "
            + ", ".join(missing_ids)
        )

    _validate_verification(report.get("verification"), errors)

    changed = _changed_files(iter_dir, manifest, expected)
    if manifest.get("gate_mode") == "incremental":
        required_changed = {
            str(path)
            for path in ((manifest.get("change_scope") or {}).get("changed_files") or [])
        }
        unchanged = sorted(required_changed - set(changed))
        if unchanged:
            errors.append(
                "incremental files declared for modification did not change from "
                "the verified baseline: " + ", ".join(unchanged)
            )

    return ImplementationGateResult(
        passed=not errors,
        errors=errors,
        expected_files=sorted(expected),
        changed_files=changed,
        report_path=str(report_path),
    )


def source_fingerprint(root: Path) -> Dict[str, str]:
    """Hash source-like delivery files while ignoring builds and logs."""
    result: Dict[str, str] = {}
    roots = ("include", "src", "tests")
    for dirname in roots:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not _ignored(path):
                result[path.relative_to(root).as_posix()] = _sha256(path)
    for name in ("CMakeLists.txt", "serve.sh", "LANGUAGE_BOUNDARY.md"):
        path = root / name
        if path.is_file():
            result[name] = _sha256(path)
    return result


def _expected_delivery_files(manifest: Mapping[str, Any], errors: List[str]) -> Set[str]:
    expected: Set[str] = set()
    capabilities = manifest.get("core_capabilities")
    if not isinstance(capabilities, dict):
        errors.append("plan_manifest.json core_capabilities must be an object")
    else:
        for capability, spec in capabilities.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("status") != "delivered_after_b":
                continue
            files = spec.get("files")
            if not _relative_paths(files, allow_empty=False):
                errors.append(
                    f"core capability {capability!r} has invalid delivery files"
                )
                continue
            expected.update(str(path) for path in files)

    scope = manifest.get("change_scope")
    if isinstance(scope, dict):
        for key in ("changed_files", "test_files"):
            values = scope.get(key) or []
            if not _relative_paths(values, allow_empty=True):
                errors.append(f"change_scope.{key} must contain relative paths")
            else:
                expected.update(str(path) for path in values)
    for item in manifest.get("delivery_items") or []:
        if isinstance(item, dict) and _relative_paths(
            item.get("files"), allow_empty=False,
        ):
            expected.update(str(path) for path in item["files"])
    return expected


def _validate_verification(value: Any, errors: List[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{REPORT_NAME} verification must be an object")
        return
    for key in _REQUIRED_VERIFICATION:
        check = value.get(key)
        if not isinstance(check, dict) or check.get("passed") is not True:
            errors.append(f"verification.{key}.passed must be true")
    reference = value.get("reference_differential")
    metrics = reference.get("metrics") if isinstance(reference, dict) else None
    cosine = metrics.get("prefill_decode_cosine") if isinstance(metrics, dict) else None
    if not isinstance(cosine, (int, float)):
        errors.append(
            "verification.reference_differential.metrics must include "
            "prefill_decode_cosine"
        )
    elif float(cosine) < 0.95:
        errors.append(
            "prefill/decode cosine must be >= 0.95; "
            f"reported {float(cosine):.6f}"
        )


def _changed_files(
    iter_dir: Path, manifest: Mapping[str, Any], expected: Iterable[str],
) -> List[str]:
    baseline_number = manifest.get("inherits_verified_iteration")
    if not isinstance(baseline_number, int) or baseline_number < 1:
        return sorted(path for path in expected if (iter_dir / path).is_file())
    baseline = iter_dir.parent / f"{baseline_number:03d}"
    changed: List[str] = []
    for relative in sorted(set(expected)):
        current = iter_dir / relative
        previous = baseline / relative
        if current.is_file() and (
            not previous.is_file() or _sha256(current) != _sha256(previous)
        ):
            changed.append(relative)
    return changed


def _read_json(path: Path, label: str, errors: List[str]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing required B-stage artifact: {label}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot parse {label}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return value


def _relative_paths(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(
            isinstance(item, str)
            and bool(item.strip())
            and not Path(item).is_absolute()
            and ".." not in Path(item).parts
            for item in value
        )
    )


def _ignored(path: Path) -> bool:
    return any(part in {"build", ".git", "__pycache__"} for part in path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ImplementationGateResult",
    "NativeBuildGateResult",
    "REPORT_NAME",
    "execute_native_build_gate",
    "source_fingerprint",
    "validate_implementation",
]
