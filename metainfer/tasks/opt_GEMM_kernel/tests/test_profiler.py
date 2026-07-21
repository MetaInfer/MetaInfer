from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ..orchestrator.hardware import HardwareProfileError, require_hardware_profile
from ..orchestrator.profiler import FrozenProfilerProfile, ProfilerRunner, _parse_case


def test_k100_selection_owns_build_and_profiler_routes():
    _, profile = require_hardware_profile({
        "target_hardware": "Hygon K100", "gpu_arch": "gfx928",
    })
    assert profile["build"]["generator"] == "Ninja"
    assert profile["build"]["release_flags"] == ["-O3"]
    assert profile["profiling"]["representative_cases"] == [
        "wq-b-tp4-m1", "wq-b-tp4-m16", "wq-b-tp4-m4096",
    ]


def test_k100_rejects_ui_arch_override():
    with pytest.raises(HardwareProfileError, match="requires gpu_arch=gfx928"):
        require_hardware_profile({
            "target_hardware": "Hygon K100", "gpu_arch": "gfx942",
        })


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
        tmp_path / "metainfer_gemm_harness", "wq-b-tp4-m1", ["SQ_WAVES"],
        tmp_path / "out",
    )
    assert command[:4] == [
        "/opt/dtk/bin/rocprofv3", "--pmc", "SQ_WAVES", "--output-format",
    ]
    assert command[-2:] == ["profile", "wq-b-tp4-m1"]
    assert "--kernel-include-regex" in command
