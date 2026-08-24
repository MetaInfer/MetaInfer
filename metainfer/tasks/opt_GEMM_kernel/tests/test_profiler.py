from __future__ import annotations

import csv
import importlib.util
import subprocess
from pathlib import Path

import pytest

from ..orchestrator.hardware import HardwareProfileError, require_hardware_profile
from ..orchestrator.profiler import (
    FrozenProfilerProfile, ProfilerError, ProfilerRunner, _parse_case,
    _python_executable, _validate_harness_profile, _version,
)


def test_k100_selection_owns_build_and_profiler_routes():
    _, profile = require_hardware_profile({
        "target_hardware": "Hygon K100", "gpu_arch": "gfx928",
    })
    assert profile["build"]["generator"] == "Ninja"
    assert profile["build"]["release_flags"] == ["-O3"]
    assert profile["profiling"]["representative_cases"] == [
        "wq-b-tp4-m1", "wq-b-tp4-m16", "wq-b-tp4-m4096",
    ]
    assert profile["profiling"]["tool_candidates"][0] == "/opt/dtk/bin/hipprof"


def test_k100_rejects_ui_arch_override():
    with pytest.raises(HardwareProfileError, match="requires gpu_arch=gfx928"):
        require_hardware_profile({
            "target_hardware": "Hygon K100", "gpu_arch": "gfx942",
        })


def test_legacy_rocprof_help_is_stable_version_fallback(monkeypatch):
    def fake_run(argv, **_kwargs):
        if argv[1] in ("--version", "-v"):
            return subprocess.CompletedProcess(argv, 1, "Wrong option\n")
        return subprocess.CompletedProcess(
            argv, 1,
            "RPL: on 'dynamic timestamp' from '/opt/dtk/rocprofiler'\n"
            "ROCm Profiling Library (RPL) run script\nUsage: rocprof ...\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _version("/opt/dtk/rocprofiler/bin/rocprof") == (
        "ROCm Profiling Library (RPL) run script\nUsage: rocprof ..."
    )


def test_python_executable_ignores_launcher_argv0(monkeypatch):
    monkeypatch.setattr(
        "metainfer.tasks.opt_GEMM_kernel.orchestrator.profiler.sys.executable",
        "/usr/local/bin/metainfer-orchestrator",
    )
    assert Path(_python_executable()).name.startswith("python")


def test_rocprof_csv_is_normalized_for_ui_and_f_agent(tmp_path: Path):
    path = tmp_path / "pass_1" / "results.csv"
    path.parent.mkdir()
    fields = [
        "Kernel_Name", "Start_Timestamp", "End_Timestamp", "VGPR_Count",
        "LDS_Block_Size", "Counter_Name", "Counter_Value",
    ]
    rows = [
        ["w8a8_scaled_main", 100, 200, 40, 8192, "TCC_HIT", 90],
        ["w8a8_scaled_main", 100, 200, 40, 8192, "TCC_MISS", 10],
        ["w8a8_scaled_main", 100, 200, 40, 8192, "FETCH_SIZE", 50],
        ["w8a8_scaled_main", 100, 200, 40, 8192, "WRITE_SIZE", 10],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(fields); writer.writerows(rows)
    result = _parse_case("case", tmp_path, "w8a8_scaled_")
    assert result["duration_ns"] == 100
    assert result["vgpr_count"] == 40
    assert result["lds_bytes"] == 8192
    assert result["l2_hit_pct"] == 90.0
    assert result["measured_bandwidth_gbps"] == pytest.approx(614.4)


def test_k100_rocprofv3_command_is_system_owned(tmp_path: Path):
    profile = FrozenProfilerProfile(
        id="hygon-k100-gfx928", label="Hygon K100", backend="hip",
        gpu_arch="gfx928", executable="/opt/dtk/bin/rocprofv3",
        executable_version="test", tool_kind="rocprofv3",
        representative_cases=["wq-b-tp4-m1"], counter_groups=[["SQ_WAVES"]],
        kernel_name_contains="w8a8_scaled_", required=True,
        fingerprint="test",
    )
    command = ProfilerRunner(profile, private_env={})._command(
        [str(tmp_path / "metainfer_gemm_harness")], "wq-b-tp4-m1", ["SQ_WAVES"],
        tmp_path / "out",
    )
    assert command[:4] == [
        "/opt/dtk/bin/rocprofv3", "--pmc", "SQ_WAVES", "--output-format",
    ]
    assert command[-2:] == ["profile", "wq-b-tp4-m1"]
    assert "--kernel-include-regex" in command


def test_k100_hipprof_pmc_command_is_system_owned(tmp_path: Path):
    profile = FrozenProfilerProfile(
        id="hygon-k100-gfx928", label="Hygon K100", backend="hip",
        gpu_arch="gfx928", executable="/opt/dtk/bin/hipprof",
        executable_version="test", tool_kind="hipprof",
        representative_cases=["wq-b-tp4-m1"],
        counter_groups=[["HIPPROF_PMC_FULL"]], kernel_name_contains="",
        required=True, fingerprint="test",
    )
    command = ProfilerRunner(profile, private_env={})._command(
        ["python3", "evaluate.py"], "wq-b-tp4-m1",
        ["HIPPROF_PMC_FULL"], tmp_path / "out",
    )
    assert command[:5] == [
        "/opt/dtk/bin/hipprof", "--pmc", "--pmc-type", "3", "-o",
    ]
    assert command[-4:] == ["python3", "evaluate.py", "profile", "wq-b-tp4-m1"]


def test_hipprof_indexed_counters_are_summed(tmp_path: Path):
    path = tmp_path / "pass_1" / "counter_collection.csv"
    path.parent.mkdir()
    fields = [
        "KernelName", "BeginNs", "EndNs", "grd", "wgr",
        "TCC_HIT[0]", "TCC_HIT[1]", "TCC_MISS[0]", "TCC_MISS[1]",
        "GRBM_COUNT", "GRBM_GUI_ACTIVE", "SQ_LDS_BANK_CONFLICT",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerow(["small_m_sdot4_kernel", 100, 220, 64, 256,
                         40, 50, 5, 5, 1000, 750, 12])
    result = _parse_case("case", tmp_path, "")
    assert result["duration_ns"] == 120
    assert result["dispatch_count"] == 1
    assert result["counters"]["TCC_HIT"] == 90
    assert result["counters"]["TCC_MISS"] == 10
    assert result["l2_hit_pct"] == 90.0
    assert result["compute_busy_pct"] == 75.0
    assert result["counters"]["SQ_LDS_BANK_CONFLICT"] == 12


def test_harness_profile_must_match_requested_case(tmp_path: Path):
    path = tmp_path / "harness-profile.json"
    path.write_text(
        '{"passed": true, "case_id": "wq-b-tp4-m16"}', encoding="utf-8",
    )
    _validate_harness_profile(tmp_path, "wq-b-tp4-m16")
    with pytest.raises(ProfilerError, match="does not match"):
        _validate_harness_profile(tmp_path, "wq-b-tp4-m1")


def _analyzer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "harness" / "user_gemm" / "analyze_hipprof_suite.py"
    )
    spec = importlib.util.spec_from_file_location("gemm_hipprof_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_operator_time_sums_all_dispatches_per_final_sample():
    analyzer = _analyzer_module()
    case = {
        "id": "split-k",
        "host_epoch_begin_ns": 100,
        "host_epoch_end_ns": 1000,
    }
    rows = [
        {"begin_ns": 50, "duration_ns": 9999, "kernel_name": "prepare"},
        {"begin_ns": 100, "duration_ns": 900, "kernel_name": "split"},
        {"begin_ns": 110, "duration_ns": 100, "kernel_name": "split"},
        {"begin_ns": 120, "duration_ns": 100, "kernel_name": "reduce"},
        {"begin_ns": 200, "duration_ns": 100, "kernel_name": "split"},
        {"begin_ns": 210, "duration_ns": 200, "kernel_name": "split"},
        {"begin_ns": 220, "duration_ns": 50, "kernel_name": "reduce"},
        {"begin_ns": 300, "duration_ns": 150, "kernel_name": "split"},
        {"begin_ns": 310, "duration_ns": 250, "kernel_name": "split"},
        {"begin_ns": 320, "duration_ns": 100, "kernel_name": "reduce"},
        {"begin_ns": 1100, "duration_ns": 9999, "kernel_name": "post"},
    ]
    operator_us, breakdown, dispatches, samples = analyzer._trace_case_times(
        rows, case, calls=3, samples=2
    )
    assert operator_us == pytest.approx(0.425)
    assert breakdown == pytest.approx({"split": 0.35, "reduce": 0.075})
    assert dispatches == 3
    assert samples == pytest.approx([0.35, 0.5])


def test_trace_rejects_unstable_final_dispatch_pattern():
    analyzer = _analyzer_module()
    case = {
        "id": "unstable",
        "host_epoch_begin_ns": 100,
        "host_epoch_end_ns": 1000,
    }
    rows = [
        {"begin_ns": 100, "duration_ns": 10, "kernel_name": "main"},
        {"begin_ns": 110, "duration_ns": 10, "kernel_name": "reduce"},
        {"begin_ns": 200, "duration_ns": 10, "kernel_name": "main"},
        {"begin_ns": 210, "duration_ns": 10, "kernel_name": "other"},
    ]
    with pytest.raises(RuntimeError, match="unstable measured dispatch pattern"):
        analyzer._trace_case_times(rows, case, calls=2, samples=2)


def test_pmc_is_normalized_per_call_and_replay_duration_is_not_latency():
    analyzer = _analyzer_module()
    case = {
        "id": "pmc",
        "host_monotonic_begin_ns": 100,
        "host_monotonic_end_ns": 1000,
    }
    rows = [
        {
            "BeginNs": str(begin),
            "DurationNs": str(duration),
            "KernelName": kernel,
            "TCC_EA_RDREQ[0]": "1",
            "TCC_EA_RDREQ_32B[0]": "1",
            "TCC_EA_WRREQ[0]": "1",
            "TCC_EA_WRREQ_64B[0]": "0",
        }
        for begin, duration, kernel in (
            (100, 1_000_000, "split"),
            (110, 2_000_000, "reduce"),
            (200, 3_000_000, "split"),
            (210, 4_000_000, "reduce"),
        )
    ]
    counters = analyzer._aggregate_counters(rows, case, calls=2)
    assert counters["dispatch_count"] == 2
    assert counters["hbm_read_bytes"] == 64
    assert counters["hbm_write_bytes"] == 64
    assert "duration" not in counters
