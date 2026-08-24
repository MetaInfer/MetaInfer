"""Read task-owned reports and derive public per-shape views."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.requirements import req_field

from ..orchestrator import phases
from ..orchestrator.evaluator.champion import (
    champion_report_reference,
    load_report_reference,
)
from ..orchestrator.evaluator.spec import BenchmarkCaseSpec, KernelTaskSpec, SpecError


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    spec = _task_spec(state_dir)
    baseline = _baseline_report(state_dir)
    records = [
        value
        for path in sorted((state_dir / "iterations").glob("*.json"))
        if isinstance((value := _json(path, None)), dict)
    ] if (state_dir / "iterations").is_dir() else []
    return [
        _public_record(state_dir, record, spec, baseline)
        for record in records
    ]


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    value = _json(state_dir / "iterations" / f"{n:03d}.json", None)
    if not isinstance(value, dict):
        return None
    return _public_record(state_dir, value, _task_spec(state_dir), _baseline_report(state_dir))


def read_champion(state_dir: Path) -> Dict[str, Any]:
    record = _json(state_dir / "champion" / "champion.json", {}) or {}
    if not isinstance(record, dict) or not record:
        return {}
    reference = champion_report_reference(state_dir, record)
    benchmark = load_report_reference(state_dir, reference)
    promotion_incumbent_ref = record.get("promotion_incumbent_report")
    promotion_incumbent = (
        load_report_reference(state_dir, promotion_incumbent_ref)
        if isinstance(promotion_incumbent_ref, dict) else {}
    )
    profile = _champion_profile(state_dir, record)
    return {
        **record,
        "measurement_report": reference,
        "benchmark": benchmark,
        "promotion_incumbent": promotion_incumbent,
        "profile": profile,
    }


def read_baseline(state_dir: Path) -> Dict[str, Any]:
    manifest = _json(state_dir / "baseline" / "baseline-manifest.json", {}) or {}
    initial_hip = _json(
        state_dir / "certified" / "initial-hip" / "initial-hip-manifest.json", {}
    ) or {}
    build_profile = _json(state_dir / "system_build" / "build_profile.json", {}) or {}
    requirements = _json(state_dir / "requirements.json", {}) or {}
    frozen_profiler = _json(
        state_dir / "system_profiler" / "profiler_profile.json", {}
    ) or {}
    benchmark = _baseline_report(state_dir)
    hardware_profile = _manifest_report(
        state_dir,
        manifest,
        "profile_report",
        state_dir / "baseline" / "baseline-hardware-profile.json",
    )
    initial_benchmark = _manifest_report(
        state_dir,
        initial_hip,
        "benchmark_report",
        state_dir / "certified" / "initial-hip" / "candidate-benchmark-report.json",
    )
    initial_profile = _manifest_report(
        state_dir,
        initial_hip,
        "profile_report",
        state_dir / "certified" / "initial-hip" / "candidate-hardware-profile.json",
    )
    spec = _task_spec(state_dir)
    cases = _baseline_cases(benchmark, spec)
    return {
        "certified": bool(manifest),
        "implementation": manifest.get("implementation", "legacy"),
        "certified_at": manifest.get("certified_at"),
        "build_fingerprint": manifest.get("build_fingerprint"),
        "backend": build_profile.get("backend"),
        "kernel_language": build_profile.get("kernel_language"),
        "target_hardware": build_profile.get("target_hardware"),
        "gpu_arch": build_profile.get("gpu_arch"),
        "detected_hardware": build_profile.get("detected_hardware"),
        "compiler": build_profile.get("compiler"),
        "compiler_version": build_profile.get("compiler_version"),
        "cmake_version": build_profile.get("cmake_version"),
        "profiler": {
            "profile_id": frozen_profiler.get("id"),
            "tool": frozen_profiler.get("tool_kind"),
            "tool_version": frozen_profiler.get("executable_version"),
            "fingerprint": frozen_profiler.get("fingerprint"),
            "representative_cases": frozen_profiler.get("representative_cases") or [],
            "counter_groups": frozen_profiler.get("counter_groups") or [],
            "passed": hardware_profile.get("passed"),
        },
        "correctness": (manifest.get("correctness") or {}).get("summary") or {},
        "initial_hip": {
            "certified": bool(initial_hip),
            "certified_at": initial_hip.get("certified_at"),
            "build_fingerprint": initial_hip.get("build_fingerprint"),
            "correctness": (initial_hip.get("correctness") or {}).get("summary") or {},
            "score": initial_benchmark.get("score") or {},
            "profile": initial_profile,
            "benchmark_report": initial_hip.get("benchmark_report") or {},
            "profile_report": initial_hip.get("profile_report") or {},
        },
        "task": {
            "kernel_path": req_field(requirements, "initial_submission"),
            "contract_source": "frozen evaluator" if spec else None,
            "public_contract": spec.agent_contract() if spec else {},
            "max_iterations": req_field(requirements, "max_iterations", 20),
        },
        "benchmark": {
            "methodology": benchmark.get("methodology") or {},
            "case_count": len(cases),
            "summary": _measurement_summary(cases),
            "cases": cases,
            "report": manifest.get("benchmark_report") or {},
        },
    }


def read_charts(state_dir: Path) -> Dict[str, Any]:
    spec = _task_spec(state_dir)
    baseline_report = _baseline_report(state_dir)
    baseline_manifest = _json(
        state_dir / "baseline" / "baseline-manifest.json", {}
    ) or {}
    baseline_profile = _manifest_report(
        state_dir,
        baseline_manifest,
        "profile_report",
        state_dir / "baseline" / "baseline-hardware-profile.json",
    )
    baseline_cases = _baseline_cases(baseline_report, spec)
    records = read_iterations(state_dir)
    champion = read_champion(state_dir)
    champion_cases = _comparison_cases(
        champion.get("promotion_incumbent") or baseline_report,
        champion.get("benchmark") or baseline_report,
        spec,
    )
    champion_profile = champion.get("profile") or baseline_profile
    profile_cases = _merge_hardware_cases(champion_cases, champion_profile)
    case_series = _case_series(
        baseline_cases,
        baseline_profile,
        records,
        spec,
    )
    duration_series: List[Dict[str, Any]] = []
    for record in records:
        _append_point(
            duration_series,
            int(record.get("iteration") or 0),
            record.get("duration_s"),
            bool(record.get("promoted")),
        )
    return {
        "series": {"duration_s": duration_series},
        "case_ids": list(case_series),
        "case_series": case_series,
        "baseline_summary": _measurement_summary(baseline_cases),
        "champion_summary": {
            **_gate_summary(champion_cases),
            "worst_case_speedup": min(
                (
                    float(case["speedup"])
                    for case in champion_cases
                    if case.get("speedup") is not None
                ),
                default=None,
            ),
            "iteration": int(champion.get("iteration") or 0),
            "kind": champion.get("kind", "triton"),
            "reason": champion.get("reason"),
        },
        "profile_cases": profile_cases,
    }


def read_retrospective(state_dir: Path, n: int) -> Dict[str, Any]:
    rec = read_iteration(state_dir, n)
    if rec is None:
        return {"iteration": n, "has_retrospective": False, "markdown": "no such iteration"}
    path = Path(str(rec.get("retrospective_path") or ""))
    markdown = ""
    if path.is_file():
        try:
            markdown = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return {
        "iteration": n,
        "has_retrospective": bool(markdown),
        "markdown": markdown or f"# Iteration {n}\n\nNo review was produced.",
        "score": rec.get("score") or {},
        "promoted": bool(rec.get("promoted")),
    }


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    run = _json(state_dir / "run.json", {}) or {}
    return phases.graph_payload(
        run.get("current_phase", "idle"),
        run.get("last_outcome"),
        run.get("last_transition_label"),
    )


def _task_spec(state_dir: Path) -> Optional[KernelTaskSpec]:
    path = state_dir / "system_evaluator" / "task.yaml"
    if not path.is_file():
        return None
    try:
        return KernelTaskSpec.load(path)
    except (SpecError, OSError):
        return None


def _manifest_report(
    state_dir: Path,
    manifest: Dict[str, Any],
    key: str,
    legacy_path: Path,
) -> Dict[str, Any]:
    reference = manifest.get(key)
    if isinstance(reference, dict):
        return load_report_reference(state_dir, reference)
    legacy_key = "benchmark" if key == "benchmark_report" else "hardware_profile"
    embedded = manifest.get(legacy_key)
    if isinstance(embedded, dict) and embedded:
        return embedded
    value = _json(legacy_path, {})
    return value if isinstance(value, dict) else {}


def _baseline_report(state_dir: Path) -> Dict[str, Any]:
    manifest = _json(state_dir / "baseline" / "baseline-manifest.json", {}) or {}
    return _manifest_report(
        state_dir,
        manifest,
        "benchmark_report",
        state_dir / "baseline" / "baseline-benchmark-report.json",
    )


def _champion_profile(state_dir: Path, champion: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(champion.get("kind") or "hip")
    iteration = int(champion.get("iteration") or 0)
    if kind == "triton":
        manifest = _json(state_dir / "baseline" / "baseline-manifest.json", {}) or {}
        return _manifest_report(
            state_dir,
            manifest,
            "profile_report",
            state_dir / "baseline" / "baseline-hardware-profile.json",
        )
    if iteration == 0:
        manifest = _json(
            state_dir / "certified" / "initial-hip" / "initial-hip-manifest.json", {}
        ) or {}
        return _manifest_report(
            state_dir,
            manifest,
            "profile_report",
            state_dir / "certified" / "initial-hip" / "candidate-hardware-profile.json",
        )
    record = _json(state_dir / "iterations" / f"{iteration:03d}.json", {}) or {}
    reference = record.get("profile_report")
    if isinstance(reference, dict):
        return load_report_reference(state_dir, reference)
    return _json(
        state_dir / "logs" / f"{iteration:03d}" / "candidate-hardware-profile.json",
        {},
    ) or {}


def _public_record(
    state_dir: Path,
    record: Dict[str, Any],
    spec: Optional[KernelTaskSpec],
    baseline_report: Dict[str, Any],
) -> Dict[str, Any]:
    result = copy.deepcopy(record)
    measurement_ref = result.get("measurement_report")
    if isinstance(measurement_ref, dict):
        benchmark = load_report_reference(state_dir, measurement_ref)
        result["benchmark"] = benchmark
        result["score"] = benchmark.get("score") or {}
    else:
        benchmark = {"score": result.get("score") or {}}
    profile_ref = result.get("profile_report")
    if isinstance(profile_ref, dict):
        result["profile"] = load_report_reference(state_dir, profile_ref)
    elif isinstance(result.get("hardware_profile"), dict):
        result["profile"] = result.get("hardware_profile")
    score = result.get("score")
    if isinstance(score, dict):
        if not score.get("cases") and benchmark.get("cases"):
            score["cases"] = _comparison_cases(
                baseline_report,
                benchmark,
                spec,
            )
        else:
            score["cases"] = _score_cases(score.get("cases") or [], spec)
        private = _private_ids(spec)
        score["reasons"] = [
            _redact(str(reason), private) for reason in score.get("reasons") or []
        ]
    result.pop("hardware_profile", None)
    return result


def _case_specs(spec: Optional[KernelTaskSpec]) -> Dict[str, BenchmarkCaseSpec]:
    return {case.id: case for case in spec.benchmark_cases} if spec else {}


def _private_ids(spec: Optional[KernelTaskSpec]) -> set[str]:
    return set(spec.private_case_ids) if spec else set()


def _baseline_cases(
    benchmark: Dict[str, Any], spec: Optional[KernelTaskSpec],
) -> List[Dict[str, Any]]:
    specs = _case_specs(spec)
    private = _private_ids(spec)
    cases: List[Dict[str, Any]] = []
    for raw in benchmark.get("cases") or []:
        if not isinstance(raw, dict):
            continue
        case_id = str(raw.get("id") or "")
        if not case_id or case_id in private:
            continue
        try:
            latency = float(raw["latency_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        cases.append(_profile_case(specs.get(case_id), case_id, latency, latency))
    return cases


def _comparison_cases(
    baseline_report: Dict[str, Any],
    candidate_report: Dict[str, Any],
    spec: Optional[KernelTaskSpec],
) -> List[Dict[str, Any]]:
    baseline = {
        str(case.get("id")): case
        for case in baseline_report.get("cases") or []
        if isinstance(case, dict) and case.get("id")
    }
    candidate = {
        str(case.get("id")): case
        for case in candidate_report.get("cases") or []
        if isinstance(case, dict) and case.get("id")
    }
    specs = _case_specs(spec)
    private = _private_ids(spec)
    cases: List[Dict[str, Any]] = []
    expected = [case.id for case in spec.benchmark_cases] if spec else list(baseline)
    for case_id in expected:
        if case_id in private or case_id not in baseline or case_id not in candidate:
            continue
        try:
            baseline_ms = float(baseline[case_id]["latency_ms"])
            candidate_ms = float(candidate[case_id]["latency_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        shown = _profile_case(specs.get(case_id), case_id, baseline_ms, candidate_ms)
        for key in (
            "latency_mean_ms", "latency_median_ms", "latency_stddev_ms",
            "latency_cv", "latency_min_ms", "latency_max_ms",
            "measurement_batches", "sample_count",
        ):
            if key in candidate[case_id]:
                shown[key] = candidate[case_id][key]
        cases.append(shown)
    return cases


def _score_cases(
    raw_cases: List[Any], spec: Optional[KernelTaskSpec],
) -> List[Dict[str, Any]]:
    specs = _case_specs(spec)
    private = _private_ids(spec)
    cases: List[Dict[str, Any]] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            continue
        case_id = str(raw.get("id") or "")
        if not case_id or case_id in private:
            continue
        try:
            baseline_ms = float(raw["baseline_ms"])
            candidate_ms = float(raw["candidate_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        cases.append(_profile_case(
            specs.get(case_id), case_id, baseline_ms, candidate_ms
        ))
    return cases


def _profile_case(
    spec: Optional[BenchmarkCaseSpec],
    case_id: str,
    baseline_ms: float,
    candidate_ms: float,
) -> Dict[str, Any]:
    flops = spec.flops if spec else None
    transferred = spec.bytes if spec else None
    return {
        "id": case_id,
        "shape": spec.shape if spec else None,
        "flops": flops,
        "bytes": transferred,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms if candidate_ms > 0 else None,
        "regression": candidate_ms / baseline_ms - 1.0 if baseline_ms > 0 else None,
        "baseline_tflops": _rate(flops, baseline_ms, 1e9),
        "candidate_tflops": _rate(flops, candidate_ms, 1e9),
        "baseline_bandwidth_gbps": _rate(transferred, baseline_ms, 1e6),
        "candidate_bandwidth_gbps": _rate(transferred, candidate_ms, 1e6),
    }


def _measurement_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    invalid = []
    for case in cases:
        try:
            latency_ms = float(case.get("baseline_ms"))
        except (TypeError, ValueError):
            latency_ms = math.nan
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            invalid.append(str(case.get("id")))
    return {
        "case_count": len(cases),
        "all_shapes_measured": bool(cases) and not invalid,
        "invalid_case_ids": invalid,
    }


def _gate_summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "case_count": len(cases),
        "all_shapes_passed": bool(cases) and all(
            float(case.get("candidate_ms") or 0)
            < float(case.get("baseline_ms") or 0)
            for case in cases
        ),
        "failed_case_ids": [
            str(case.get("id"))
            for case in cases
            if float(case.get("candidate_ms") or 0)
            >= float(case.get("baseline_ms") or 0)
        ],
    }


def _rate(work: Optional[float], latency_ms: float, scale: float) -> Optional[float]:
    return work / latency_ms / scale if work is not None and latency_ms > 0 else None


def _append_point(
    target: List[Dict[str, Any]], iteration: int, value: Any, promoted: bool,
) -> None:
    if value is None:
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    target.append({"x": iteration, "y": number, "promoted": promoted})


_HARDWARE_METRICS = (
    "measured_bandwidth_gbps",
    "hbm_read_gbps",
    "hbm_write_gbps",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "l2_hit_pct",
    "occupancy_pct",
    "vgpr_count",
    "agpr_count",
    "sgpr_count",
    "lds_bytes",
    "scratch_bytes",
    "dispatch_count",
)


def _case_series(
    baseline_cases: List[Dict[str, Any]],
    baseline_profile: Dict[str, Any],
    records: List[Dict[str, Any]],
    spec: Optional[KernelTaskSpec],
) -> Dict[str, Dict[str, Any]]:
    baseline_profiled = {
        str(case.get("id")): case
        for case in _merge_hardware_cases(baseline_cases, baseline_profile)
    }
    result: Dict[str, Dict[str, Any]] = {}
    for case in baseline_cases:
        case_id = str(case.get("id") or "")
        if not case_id:
            continue
        series = _empty_case_series()
        shown = baseline_profiled.get(case_id, case)
        _append_case_points(series, 0, shown, True, True)
        result[case_id] = {"case": shown, "series": series}

    for record in records:
        iteration = int(record.get("iteration") or 0)
        promoted = bool(record.get("promoted"))
        cases = _merge_hardware_cases(
            _score_cases((record.get("score") or {}).get("cases") or [], spec),
            record.get("profile") or {},
        )
        for case in cases:
            case_id = str(case.get("id") or "")
            if not case_id or case_id not in result:
                continue
            _append_case_points(
                result[case_id]["series"], iteration, case, promoted, False
            )
            _append_point(
                result[case_id]["series"]["duration_s"],
                iteration,
                record.get("duration_s"),
                promoted,
            )
    return result


def _empty_case_series() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "latency_ms": [],
        "speedup": [],
        "tflops": [],
        "bandwidth_gbps": [],
        "regression": [],
        "duration_s": [],
        **{metric: [] for metric in _HARDWARE_METRICS},
    }


def _append_case_points(
    series: Dict[str, List[Dict[str, Any]]],
    iteration: int,
    case: Dict[str, Any],
    promoted: bool,
    baseline: bool,
) -> None:
    prefix = "baseline" if baseline else "candidate"
    _append_point(series["latency_ms"], iteration, case.get(f"{prefix}_ms"), promoted)
    _append_point(series["speedup"], iteration, 1.0 if baseline else case.get("speedup"), promoted)
    _append_point(series["regression"], iteration, 0.0 if baseline else case.get("regression"), promoted)
    _append_point(series["tflops"], iteration, case.get(f"{prefix}_tflops"), promoted)
    _append_point(
        series["bandwidth_gbps"],
        iteration,
        case.get(f"{prefix}_bandwidth_gbps"),
        promoted,
    )
    for metric in _HARDWARE_METRICS:
        _append_point(series[metric], iteration, case.get(metric), promoted)


def _merge_hardware_cases(
    cases: List[Dict[str, Any]], report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    profiled = {
        str(case.get("id")): case
        for case in report.get("cases") or []
        if isinstance(case, dict) and case.get("id")
    }
    return [{**case, **profiled.get(str(case.get("id")), {})} for case in cases]


def _redact(value: str, private: set[str]) -> str:
    for case_id in sorted(private, key=len, reverse=True):
        value = value.replace(case_id, "<held-out>")
    return value
