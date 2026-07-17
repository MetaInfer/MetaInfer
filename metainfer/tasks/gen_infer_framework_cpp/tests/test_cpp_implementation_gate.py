from __future__ import annotations

import json
from pathlib import Path

from metainfer.tasks.gen_infer_framework_cpp.orchestrator.implementation_gate import (
    execute_native_build_gate,
    validate_implementation,
)


def _write_delivery(root: Path, *, iteration: int = 1, baseline: int | None = None):
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "src/main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (root / "tests/test_e2e.cpp").write_text("// assertions\n", encoding="utf-8")
    manifest = {
        "gate_mode": "incremental" if baseline else "full",
        "core_capabilities": {
            "native_cpp_runtime": {
                "status": "delivered_after_b",
                "files": ["src/main.cpp"],
            },
        },
        "delivery_items": [{
            "id": "native-runtime",
            "summary": "Deliver the runtime",
            "files": ["src/main.cpp", "tests/test_e2e.cpp"],
            "tests": ["ctest --test-dir build"],
        }],
    }
    if baseline:
        manifest["inherits_verified_iteration"] = baseline
        manifest["change_scope"] = {
            "changed_files": ["src/main.cpp"],
            "test_files": ["tests/test_e2e.cpp"],
        }
    (root / "plan_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "iteration": iteration,
        "status": "complete",
        "plan_items": [{
            "id": "native-runtime",
            "status": "implemented",
            "files": ["src/main.cpp", "tests/test_e2e.cpp"],
            "tests": ["ctest --test-dir build"],
        }],
        "verification": {
            "build": {"passed": True},
            "reference_differential": {
                "passed": True,
                "metrics": {"prefill_decode_cosine": 0.973},
            },
            "end_to_end": {"passed": True},
        },
    }
    (root / "implementation_report.json").write_text(
        json.dumps(report), encoding="utf-8",
    )


def test_complete_delivery_passes(tmp_path: Path):
    _write_delivery(tmp_path)

    result = validate_implementation(tmp_path, 1)

    assert result.passed, result.diagnostics()


def test_print_only_low_cosine_is_rejected(tmp_path: Path):
    _write_delivery(tmp_path)
    report_path = tmp_path / "implementation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["verification"]["reference_differential"]["metrics"][
        "prefill_decode_cosine"
    ] = 0.22
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_implementation(tmp_path, 1)

    assert not result.passed
    assert any("cosine must be >= 0.95" in error for error in result.errors)


def test_incremental_declared_file_must_change(tmp_path: Path):
    baseline = tmp_path / "001"
    current = tmp_path / "002"
    _write_delivery(baseline, iteration=1)
    _write_delivery(current, iteration=2, baseline=1)

    result = validate_implementation(current, 2)

    assert not result.passed
    assert any("did not change" in error for error in result.errors)


def test_delivery_ids_cannot_swap_their_files(tmp_path: Path):
    _write_delivery(tmp_path)
    manifest_path = tmp_path / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["delivery_items"] = [
        {
            "id": "native-runtime",
            "files": ["src/main.cpp"],
            "tests": ["runtime-test"],
        },
        {
            "id": "e2e-test",
            "files": ["tests/test_e2e.cpp"],
            "tests": ["e2e-test"],
        },
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report_path = tmp_path / "implementation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["plan_items"] = [
        {
            "id": "native-runtime",
            "status": "implemented",
            "files": ["tests/test_e2e.cpp"],
            "tests": ["runtime-test"],
        },
        {
            "id": "e2e-test",
            "status": "implemented",
            "files": ["src/main.cpp"],
            "tests": ["e2e-test"],
        },
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_implementation(tmp_path, 1)

    assert not result.passed
    assert sum("does not account for its planned files" in error for error in result.errors) == 2


def _write_native_cmake_project(root: Path, *, passing: bool) -> None:
    (root / "tests").mkdir(exist_ok=True)
    return_code = 0 if passing else 1
    (root / "tests/native_gate_test.cpp").write_text(
        f"int main() {{ return {return_code}; }}\n", encoding="utf-8",
    )
    (root / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.16)
project(native_gate_test LANGUAGES CXX)
enable_testing()
add_executable(native_gate_test tests/native_gate_test.cpp)
add_test(NAME native_gate_test COMMAND native_gate_test)
""",
        encoding="utf-8",
    )


def test_native_build_gate_executes_cmake_build_and_ctest(tmp_path: Path):
    _write_native_cmake_project(tmp_path, passing=True)
    logs = tmp_path / "gate-logs"

    result = execute_native_build_gate(tmp_path, logs, timeout_s=120)

    assert result.passed, result.diagnostics()
    assert [step["name"] for step in result.steps] == [
        "configure", "build", "ctest",
    ]
    assert all(step["returncode"] == 0 for step in result.steps)
    assert (logs / "implementation-gate-execution.json").is_file()


def test_native_build_gate_rejects_a_failing_ctest(tmp_path: Path):
    _write_native_cmake_project(tmp_path, passing=False)

    result = execute_native_build_gate(tmp_path, tmp_path / "logs", timeout_s=120)

    assert not result.passed
    assert result.steps[-1]["name"] == "ctest"
    assert result.steps[-1]["returncode"] != 0
