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
    state = tmp_path / "state"
    evaluator = state / "system_evaluator"
    shutil.copytree(make_bundle(tmp_path / "bundle"), evaluator)
    (state / "baseline").mkdir(parents=True)
    (state / "iterations").mkdir(parents=True)
    (state / "champion").mkdir(parents=True)
    baseline = {
        "benchmark": {
            "methodology": {"warmup": 10, "samples": 100, "timer": "fake"},
            "cases": [
                {"id": "small", "latency_ms": 2.0},
                {"id": "large", "latency_ms": 4.0},
            ],
        },
    }
    (state / "baseline" / "baseline-manifest.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    record = {
        "iteration": 1,
        "duration_s": 12,
        "promoted": True,
        "score": {
            "weighted_speedup": 2.0,
            "critical_regression": 0.0,
            "cases": [
                {"id": "small", "baseline_ms": 2.0, "candidate_ms": 1.0,
                 "flops": 1, "bytes": 1},
                {"id": "large", "baseline_ms": 4.0, "candidate_ms": 2.0},
            ],
        },
        "hardware_profile": {
            "cases": [{"id": "small", "vgpr_count": 40, "lds_bytes": 8192,
                       "l2_hit_pct": 91.0, "compute_busy_pct": 82.0,
                       "measured_bandwidth_gbps": 700.0}],
        },
    }
    (state / "iterations" / "001.json").write_text(json.dumps(record), encoding="utf-8")
    (state / "champion" / "champion.json").write_text(
        json.dumps({"iteration": 1, "weighted_speedup": 2.0}), encoding="utf-8"
    )

    payload = _state_readers.read_charts(state)
    assert payload["champion_summary"]["weighted_speedup"] == 2.0
    assert payload["profile_cases"][0]["candidate_tflops"] == 48 / 1.0 / 1e9
    assert payload["profile_cases"][0]["candidate_bandwidth_gbps"] == 100 / 1.0 / 1e6
    assert payload["series"]["latency_ms"][0]["x"] == 0
    assert payload["series"]["latency_ms"][1]["x"] == 1
    assert payload["series"]["measured_bandwidth_gbps"][0]["y"] == 700.0
    assert payload["profile_cases"][0]["vgpr_count"] == 40
