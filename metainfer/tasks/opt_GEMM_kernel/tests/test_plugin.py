import ast
import json
import shutil
from pathlib import Path

from metainfer.orchestrator.tasks import get_task
from metainfer.server.forms import load_form_schema
from metainfer.server.registry import get

from ..server import _state_readers
from ._helpers import make_bundle


def test_plugins_and_form_are_auto_discovered():
    task = get_task("opt_GEMM_kernel")
    web = get("opt_GEMM_kernel")
    form = load_form_schema("opt_GEMM_kernel")
    assert task.cli_module.endswith("opt_GEMM_kernel.orchestrator.cli")
    assert web is not None
    assert form is not None
    assert {field["key"] for field in form["fields"]} >= {
        "evaluator_bundle", "initial_submission", "weight_bundle"
    }
    by_key = {field["key"]: field for field in form["fields"]}
    assert by_key["initial_submission"]["type"] == "file"
    assert by_key["evaluator_bundle"]["type"] == "file"
    assert by_key["evaluator_bundle"]["label"] == "Harness path"
    assert by_key["weight_bundle"]["type"] == "file"
    assert by_key["weight_bundle"]["label"] == "Weight directory"
    assert {"target_hardware", "gpu_arch"} <= set(by_key)
    assert by_key["target_hardware"]["default"] == "Hygon K100"
    assert by_key["gpu_arch"]["default"] == "gfx928"
    assert "compiler_path" not in by_key
    assert {"gemm_shapes", "dtype", "layout"}.isdisjoint(by_key)

    frontend = web.frontend_dir
    assert (frontend / "gemm-arena-detail.js").is_file()
    assert (frontend / "gemm-profiler-charts.js").is_file()


def test_task_does_not_import_other_task_packages():
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "metainfer.tasks.opt_kernel",
        "metainfer.tasks.gen_cpp_infer_framework",
        "metainfer.tasks.gen_infer_framework",
    )
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            module == name or module.startswith(name + ".")
            for module in imported
            for name in forbidden
        ), path


def test_profiler_chart_payload_uses_frozen_work_metadata(tmp_path):
    from ..orchestrator.evaluator.champion import (
        make_report_reference,
        write_json_atomic,
    )

    state = tmp_path / "state"
    evaluator = state / "system_evaluator"
    shutil.copytree(make_bundle(tmp_path / "bundle"), evaluator)
    (state / "baseline").mkdir(parents=True)
    (state / "iterations").mkdir(parents=True)
    (state / "champion").mkdir(parents=True)
    (state / "logs" / "001").mkdir(parents=True)

    protocol = {
        "warmup": 10,
        "samples": 100,
        "trace_calls": 110,
        "timer": "hipprof_gpu_kernel_duration_ns",
        "statistic": "arithmetic_mean",
        "operator_aggregation": "sum_gpu_kernel_duration_per_call",
        "synchronization": "hipprof_trace",
        "timed_scope": "operator_gpu_dispatches_only",
        "host_launch_time_included": False,
        "pmc_timing_used": False,
    }
    baseline_report_path = state / "baseline" / "baseline-benchmark-report.json"
    baseline_profile_path = state / "baseline" / "baseline-hardware-profile.json"
    candidate_report_path = state / "logs" / "001" / "candidate-benchmark-report.json"
    candidate_profile_path = state / "logs" / "001" / "candidate-hardware-profile.json"
    write_json_atomic(baseline_report_path, {
        "passed": True,
        "methodology": protocol,
        "cases": [
            {"id": "small", "latency_ms": 2.0},
            {"id": "large", "latency_ms": 4.0},
        ],
    })
    write_json_atomic(baseline_profile_path, {
        "passed": True,
        "cases": [],
    })
    write_json_atomic(candidate_report_path, {
        "passed": True,
        "methodology": protocol,
        "cases": [
            {"id": "small", "latency_ms": 1.0},
            {"id": "large", "latency_ms": 2.0},
        ],
        "score": {
            "passed": True,
            "worst_case_speedup": 2.0,
            "failed_case_ids": [],
            "cases": [
                {"id": "small", "baseline_ms": 2.0, "candidate_ms": 1.0},
                {"id": "large", "baseline_ms": 4.0, "candidate_ms": 2.0},
            ],
        },
    })
    write_json_atomic(candidate_profile_path, {
        "passed": True,
        "cases": [{
            "id": "small", "vgpr_count": 40, "agpr_count": 8,
            "sgpr_count": 32, "lds_bytes": 8192, "scratch_bytes": 0,
            "l2_hit_pct": 91.0, "occupancy_pct": None,
            "measured_bandwidth_gbps": 700.0,
            "hbm_read_gbps": 600.0, "hbm_write_gbps": 100.0,
            "dispatch_count": 2,
            "kernel_breakdown_us": {"splitk": 0.8, "reduce": 0.2},
        }],
    })
    baseline_ref = make_report_reference(state, baseline_report_path)
    baseline_profile_ref = make_report_reference(state, baseline_profile_path)
    candidate_ref = make_report_reference(state, candidate_report_path)
    candidate_profile_ref = make_report_reference(state, candidate_profile_path)
    (state / "baseline" / "baseline-manifest.json").write_text(
        json.dumps({
            "implementation": "triton",
            "benchmark_report": baseline_ref,
            "profile_report": baseline_profile_ref,
        }),
        encoding="utf-8",
    )
    record = {
        "iteration": 1,
        "duration_s": 12,
        "promoted": True,
        "measurement_report": candidate_ref,
        "profile_report": candidate_profile_ref,
    }
    (state / "iterations" / "001.json").write_text(json.dumps(record), encoding="utf-8")
    (state / "champion" / "champion.json").write_text(
        json.dumps({
            "schema_version": 2,
            "kind": "hip",
            "iteration": 1,
            "measurement_report": candidate_ref,
            "reason": "every shape passed baseline and Champion noise gates",
        }),
        encoding="utf-8",
    )

    payload = _state_readers.read_charts(state)
    assert payload["baseline_summary"] == {
        "case_count": 2,
        "all_shapes_measured": True,
        "invalid_case_ids": [],
    }
    assert payload["champion_summary"]["all_shapes_passed"] is True
    assert payload["champion_summary"]["worst_case_speedup"] == 2.0
    assert "weighted_speedup" not in payload
    assert "critical_regression" not in payload
    assert payload["profile_cases"][0]["candidate_tflops"] == 48 / 1.0 / 1e9
    assert payload["profile_cases"][0]["candidate_bandwidth_gbps"] == 100 / 1.0 / 1e6
    assert payload["profile_cases"][0]["measured_bandwidth_gbps"] == 700.0
    assert payload["profile_cases"][0]["vgpr_count"] == 40
    assert payload["profile_cases"][0]["dispatch_count"] == 2
    assert payload["profile_cases"][0]["kernel_breakdown_us"]["reduce"] == 0.2
    assert payload["case_ids"] == ["small", "large"]
    small = payload["case_series"]["small"]["series"]
    assert [point["y"] for point in small["latency_ms"]] == [2.0, 1.0]
    assert [point["y"] for point in small["speedup"]] == [1.0, 2.0]
    assert small["tflops"][1]["y"] == 48 / 1.0 / 1e9
    assert small["measured_bandwidth_gbps"][0]["x"] == 1
    large = payload["case_series"]["large"]["series"]
    assert [point["y"] for point in large["latency_ms"]] == [4.0, 2.0]
