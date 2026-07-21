"""Read the task-owned iteration, score and champion schemas."""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..orchestrator import phases
from ..orchestrator.evaluator.spec import BenchmarkCaseSpec, KernelTaskSpec, SpecError
from metainfer.orchestrator.requirements import req_field


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    spec = _task_spec(state_dir)
    records = [
        value
        for path in sorted((state_dir / "iterations").glob("*.json"))
        if isinstance((value := _json(path, None)), dict)
    ] if (state_dir / "iterations").is_dir() else []
    return [_public_record(record, spec) for record in records]


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    value = _json(state_dir / "iterations" / f"{n:03d}.json", None)
    return _public_record(value, _task_spec(state_dir)) if isinstance(value, dict) else None


def read_champion(state_dir: Path) -> Dict[str, Any]:
    return _json(state_dir / "champion" / "champion.json", {}) or {}


def read_baseline(state_dir: Path) -> Dict[str, Any]:
    manifest = _json(state_dir / "baseline" / "baseline-manifest.json", {}) or {}
    profile = _json(state_dir / "system_build" / "build_profile.json", {}) or {}
    requirements = _json(state_dir / "requirements.json", {}) or {}
    correctness = manifest.get("correctness") or {}
    benchmark = manifest.get("benchmark") or {}
    hardware_profile = manifest.get("hardware_profile") or {}
    frozen_profiler = _json(
        state_dir / "system_profiler" / "profiler_profile.json", {}
    ) or {}
    spec = _task_spec(state_dir)
    cases = _baseline_cases(benchmark, spec)
    summary = _aggregate(cases, "baseline_ms")
    return {
        "certified": bool(manifest),
        "certified_at": manifest.get("certified_at"),
        "build_fingerprint": manifest.get("build_fingerprint"),
        "backend": profile.get("backend"),
        "kernel_language": profile.get("kernel_language"),
        "target_hardware": profile.get("target_hardware"),
        "gpu_arch": profile.get("gpu_arch"),
        "detected_hardware": profile.get("detected_hardware"),
        "compiler": profile.get("compiler"),
        "compiler_version": profile.get("compiler_version"),
        "cmake_version": profile.get("cmake_version"),
        "profiler": {
            "profile_id": frozen_profiler.get("id"),
            "tool": frozen_profiler.get("tool_kind"),
            "tool_version": frozen_profiler.get("executable_version"),
            "fingerprint": frozen_profiler.get("fingerprint"),
            "representative_cases": frozen_profiler.get("representative_cases") or [],
            "counter_groups": frozen_profiler.get("counter_groups") or [],
            "passed": hardware_profile.get("passed"),
        },
        "correctness": correctness.get("summary") or {},
        "task": {
            "kernel_path": req_field(requirements, "initial_submission"),
            "contract_source": "frozen evaluator" if spec else None,
            "public_contract": spec.agent_contract() if spec else {},
            "max_iterations": req_field(requirements, "max_iterations", 20),
        },
        "benchmark": {
            "methodology": benchmark.get("methodology") or {},
            "case_count": len(benchmark.get("cases") or []),
            "summary": summary,
            "cases": cases,
        },
    }


def read_charts(state_dir: Path) -> Dict[str, Any]:
    records = read_iterations(state_dir)
    manifest = _json(state_dir / "baseline" / "baseline-manifest.json", {}) or {}
    spec = _task_spec(state_dir)
    baseline_cases = _baseline_cases(manifest.get("benchmark") or {}, spec)
    baseline_hardware = manifest.get("hardware_profile") or {}
    baseline_summary = _aggregate(baseline_cases, "baseline_ms")
    champion = read_champion(state_dir)

    series: Dict[str, List[Dict[str, Any]]] = {
        "latency_ms": [],
        "weighted_speedup": [],
        "tflops": [],
        "bandwidth_gbps": [],
        "critical_regression": [],
        "duration_s": [],
        "measured_bandwidth_gbps": [],
        "l2_hit_pct": [],
        "compute_busy_pct": [],
        "vgpr_count": [],
        "lds_bytes": [],
    }
    if baseline_cases:
        _append_point(series["latency_ms"], 0, baseline_summary.get("latency_ms"), True)
        _append_point(series["weighted_speedup"], 0, 1.0, True)
        _append_point(series["tflops"], 0, baseline_summary.get("tflops"), True)
        _append_point(
            series["bandwidth_gbps"], 0, baseline_summary.get("bandwidth_gbps"), True
        )
        _append_point(series["critical_regression"], 0, 0.0, True)
        _append_hardware_points(series, 0, baseline_hardware, True)

    candidate_cases_by_iteration: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        iteration = int(record.get("iteration") or 0)
        score = record.get("score") or {}
        hardware = record.get("hardware_profile") or {}
        cases = _score_cases(score.get("cases") or [], spec)
        if cases:
            candidate_cases_by_iteration[iteration] = cases
            summary = _aggregate(cases, "candidate_ms")
            _append_point(
                series["latency_ms"], iteration, summary.get("latency_ms"),
                bool(record.get("promoted")),
            )
            _append_point(
                series["tflops"], iteration, summary.get("tflops"),
                bool(record.get("promoted")),
            )
            _append_point(
                series["bandwidth_gbps"], iteration, summary.get("bandwidth_gbps"),
                bool(record.get("promoted")),
            )
        _append_hardware_points(
            series, iteration, hardware, bool(record.get("promoted"))
        )
        _append_point(
            series["weighted_speedup"], iteration, score.get("weighted_speedup"),
            bool(record.get("promoted")),
        )
        _append_point(
            series["critical_regression"], iteration,
            score.get("critical_regression"), bool(record.get("promoted")),
        )
        _append_point(
            series["duration_s"], iteration, record.get("duration_s"),
            bool(record.get("promoted")),
        )

    champion_iteration = int(champion.get("iteration") or 0)
    champion_cases = candidate_cases_by_iteration.get(champion_iteration, baseline_cases)
    champion_record = next(
        (record for record in records if int(record.get("iteration") or 0) == champion_iteration),
        None,
    )
    champion_hardware = (
        (champion_record or {}).get("hardware_profile") or baseline_hardware
    )
    champion_summary = (
        _aggregate(champion_cases, "candidate_ms")
        if champion_iteration in candidate_cases_by_iteration
        else baseline_summary
    )
    champion_summary = {
        **champion_summary,
        **_hardware_summary(champion_hardware),
        "weighted_speedup": float(champion.get("weighted_speedup", 1.0) or 1.0),
        "iteration": champion_iteration,
    }
    return {
        "series": series,
        "baseline_summary": baseline_summary,
        "champion_summary": champion_summary,
        "profile_cases": _merge_hardware_cases(champion_cases, champion_hardware),
        # Compatibility for early clients of this task-local endpoint.
        "weighted_speedup": series["weighted_speedup"],
        "critical_regression": series["critical_regression"],
        "durations": series["duration_s"],
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


def _public_record(record: Dict[str, Any], spec: Optional[KernelTaskSpec]) -> Dict[str, Any]:
    result = copy.deepcopy(record)
    score = result.get("score")
    if not isinstance(score, dict):
        return result
    score["cases"] = _score_cases(score.get("cases") or [], spec)
    private = _private_ids(spec)
    score["reasons"] = [
        _redact(str(reason), private) for reason in score.get("reasons") or []
    ]
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
        item = specs.get(case_id)
        try:
            latency = float(raw["latency_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        cases.append(_profile_case(item, case_id, latency, latency))
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
        cases.append(_profile_case(specs.get(case_id), case_id, baseline_ms, candidate_ms))
    return cases


def _profile_case(
    spec: Optional[BenchmarkCaseSpec], case_id: str,
    baseline_ms: float, candidate_ms: float,
) -> Dict[str, Any]:
    weight = float(spec.weight) if spec else 1.0
    critical = bool(spec.critical) if spec else False
    flops = spec.flops if spec else None
    transferred = spec.bytes if spec else None
    return {
        "id": case_id,
        "shape": spec.shape if spec else None,
        "weight": weight,
        "critical": critical,
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


def _aggregate(cases: List[Dict[str, Any]], latency_key: str) -> Dict[str, Any]:
    valid = [case for case in cases if float(case.get(latency_key) or 0) > 0]
    if not valid:
        return {"latency_ms": None, "tflops": None, "bandwidth_gbps": None}
    total_weight = sum(float(case.get("weight") or 1.0) for case in valid)
    weighted_ms = sum(
        float(case.get("weight") or 1.0) * float(case[latency_key]) for case in valid
    )
    flop_cases = [case for case in valid if case.get("flops") is not None]
    byte_cases = [case for case in valid if case.get("bytes") is not None]
    return {
        "latency_ms": weighted_ms / total_weight,
        "tflops": _aggregate_rate(flop_cases, latency_key, "flops", 1e9),
        "bandwidth_gbps": _aggregate_rate(byte_cases, latency_key, "bytes", 1e6),
    }


def _aggregate_rate(
    cases: List[Dict[str, Any]], latency_key: str, work_key: str, scale: float,
) -> Optional[float]:
    if not cases:
        return None
    work = sum(
        float(case.get("weight") or 1.0) * float(case[work_key]) for case in cases
    )
    elapsed = sum(
        float(case.get("weight") or 1.0) * float(case[latency_key]) for case in cases
    )
    return work / elapsed / scale if elapsed > 0 else None


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
    "measured_bandwidth_gbps", "l2_hit_pct", "compute_busy_pct",
    "vgpr_count", "lds_bytes",
)


def _hardware_summary(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    cases = [case for case in report.get("cases") or [] if isinstance(case, dict)]
    result: Dict[str, Optional[float]] = {}
    for metric in _HARDWARE_METRICS:
        values: List[float] = []
        for case in cases:
            try:
                value = float(case[metric])
            except (KeyError, TypeError, ValueError):
                continue
            values.append(value)
        result[metric] = sum(values) / len(values) if values else None
    return result


def _append_hardware_points(
    series: Dict[str, List[Dict[str, Any]]], iteration: int,
    report: Dict[str, Any], promoted: bool,
) -> None:
    summary = _hardware_summary(report)
    for metric in _HARDWARE_METRICS:
        _append_point(series[metric], iteration, summary.get(metric), promoted)


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
