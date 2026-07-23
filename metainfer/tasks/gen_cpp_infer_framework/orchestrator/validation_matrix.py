"""Run the task's capability-combination regression matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .acceptance import AcceptanceContract, compile_suite_results
from .capabilities import CapabilityResolutionError, resolve_capabilities
from .knowledge import resolve_knowledge_route


TASK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = TASK_DIR / "validation_matrix.yaml"
NOTEBOOKS_DIR = TASK_DIR / "notebooks"
OPTIONAL_CAPABILITIES = {
    "paged_kv_cache", "continuous_batching", "tensor_parallelism",
    "speculative_decoding",
}
ALL_PROBE_VERDICTS = {
    "numeric-operator-contract": "pass",
    "capability-runtime-metadata": "pass",
    "capability-paged-kv-long-context": "pass",
    "capability-continuous-batching-concurrency": "pass",
    "capability-tp-paged-cb-integration": "pass",
}


def run_validation_matrix(path: Path = DEFAULT_MATRIX) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError(f"unsupported validation matrix: {path}")
    cases = raw.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("validation matrix cases must be a list")

    results = []
    for raw_case in cases:
        case = dict(raw_case)
        case_id = str(case.get("id", "<missing>"))
        req = {
            "task_id": f"matrix-{case_id}",
            "task_type": "gen-cpp-infer-framework",
            **dict(case.get("requirements", {})),
        }
        errors = []
        expected_error = case.get("expected_error")
        try:
            resolved = resolve_capabilities(req)
        except CapabilityResolutionError as exc:
            if not isinstance(expected_error, Mapping):
                errors.append(f"unexpected resolution error ({exc.field}): {exc}")
            else:
                if exc.field != expected_error.get("field"):
                    errors.append(
                        f"error field {exc.field!r} != {expected_error.get('field')!r}"
                    )
                expected_text = str(expected_error.get("contains", ""))
                if expected_text not in str(exc):
                    errors.append(f"error does not contain {expected_text!r}: {exc}")
            results.append({"id": case_id, "passed": not errors, "errors": errors})
            continue

        if expected_error:
            errors.append("expected resolution error but case resolved successfully")
        req["resolved_requirements"] = resolved
        optional_required = [
            capability_id for capability_id in resolved["required_capabilities"]
            if capability_id in OPTIONAL_CAPABILITIES
        ]
        _compare(errors, "optional required", optional_required,
                 case.get("expected_optional_required", []))
        _compare(errors, "disabled", resolved["disabled_capabilities"],
                 case.get("expected_disabled", []))
        _compare(errors, "combinations", resolved["active_combination_contracts"],
                 case.get("expected_combinations", []))
        for field, expected in dict(case.get("expected_resource", {})).items():
            actual = resolved["resource_contract"].get(field)
            if actual != expected:
                errors.append(
                    f"resource {field} mismatch: actual={actual!r} "
                    f"expected={expected!r}"
                )

        route = resolve_knowledge_route(
            req, NOTEBOOKS_DIR, role="implementer", context="",
        )
        route_ids = {document.id for document in route.required}
        for document_id in case.get("required_documents", []):
            if document_id not in route_ids:
                errors.append(f"required document {document_id!r} is absent")
        for document_id in case.get("forbidden_documents", []):
            if document_id in route_ids:
                errors.append(f"forbidden document {document_id!r} is routed")

        contract = AcceptanceContract.from_request(req)
        suites = compile_suite_results(
            contract,
            baseline_passed=True,
            probe_verdicts=ALL_PROBE_VERDICTS,
        )
        failed_suites = [item["suite"] for item in suites if not item["passed"]]
        if failed_suites:
            errors.append(f"suites lack passing immutable evidence: {failed_suites}")
        results.append({
            "id": case_id,
            "passed": not errors,
            "errors": errors,
            "required_capabilities": resolved["required_capabilities"],
            "active_combinations": resolved["active_combination_contracts"],
            "correctness_suites": resolved["correctness_suites"],
            "resource_contract": resolved["resource_contract"],
        })

    return {
        "schema_version": 1,
        "matrix": str(path),
        "passed": all(result["passed"] for result in results),
        "cases_total": len(results),
        "cases_passed": sum(1 for result in results if result["passed"]),
        "cases": results,
    }


def _compare(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if list(actual) != list(expected):
        errors.append(f"{name} mismatch: actual={list(actual)!r} expected={list(expected)!r}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_validation_matrix(args.matrix)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
