from __future__ import annotations

from pathlib import Path

import pytest

from ..orchestrator.evaluator.spec import FrozenEvaluatorBundle, KernelTaskSpec, SpecError
from ._helpers import make_bundle


def test_task_owned_harness_starter_has_a_valid_protocol():
    harness = Path(__file__).resolve().parents[1] / "harness" / "user_gemm"
    spec = KernelTaskSpec.load(harness / "task.yaml")
    assert spec.name == "deepseek-w8a8-gemm-tp4-tp8"
    assert set(spec.commands) == {"correctness", "profile"}
    assert len(spec.benchmark_cases) == 60
    assert len(spec.correctness_case_ids) == 64
    assert len(spec.private_case_ids) == 4
    assert spec.agent_contract()["abi"]["entrypoint"] == "launch_w8a8_gemm"
    assert spec.benchmark_protocol["timer"] == "hipprof_gpu_kernel_duration_ns"
    assert spec.benchmark_protocol["operator_aggregation"] == (
        "sum_gpu_kernel_duration_per_call"
    )
    assert spec.benchmark_protocol["pmc_timing_used"] is False
    assert {case.shape["m"] for case in spec.benchmark_cases} == {1, 2, 4, 8, 16, 4096}
    assert {case.shape["batch"] for case in spec.benchmark_cases} == {1}
    assert all(
        set(item) == {"id", "shape"}
        for item in spec.agent_contract()["benchmark_shapes"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timer", "gpu_event"),
        ("statistic", "median"),
        ("operator_aggregation", "longest_kernel"),
        ("synchronization", "gpu_event"),
        ("timed_scope", "host_api_call"),
        ("host_launch_time_included", True),
        ("pmc_timing_used", True),
    ],
)
def test_benchmark_protocol_rejects_non_hipprof_timing(tmp_path, field, value):
    import yaml

    source = make_bundle(tmp_path / "source")
    task_path = source / "task.yaml"
    raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    raw["benchmark_protocol"][field] = value
    task_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SpecError, match=rf"benchmark_protocol\.{field}"):
        KernelTaskSpec.load(task_path)


def test_task_owned_hipprof_suite_is_self_contained():
    harness = Path(__file__).resolve().parents[1] / "harness" / "user_gemm"
    collector = (harness / "run_hipprof_suite.py").read_text(encoding="utf-8")
    analyzer = (harness / "analyze_hipprof_suite.py").read_text(encoding="utf-8")
    evaluator = (harness / "evaluate.py").read_text(encoding="utf-8")
    assert "/data/FF/kernel benchmark" not in collector + analyzer
    assert "METAINFER_BUILD_ARTIFACT_DIR" in collector
    assert "METAINFER_WEIGHT_BUNDLE" in collector
    assert '"--hip-trace", "--stats"' in collector
    assert '"--pmc-read"' in collector and '"--pmc-write"' in collector
    assert 'phase == "profile-batch"' in evaluator
    assert '"host_epoch_begin_ns"' in evaluator
    assert '"host_epoch_end_ns"' in evaluator
    assert '"host_epoch_begin_ns" in case' in analyzer


def test_mygemm_baseline_is_decoupled_from_harness_code():
    task_root = Path(__file__).resolve().parents[1]
    submission = task_root / "initial_submissions" / "myGEMM_kernel"
    source = (submission / "myGEMM_kernel.hip").read_text(encoding="utf-8")
    manifest = (submission / "submission.yaml").read_text(encoding="utf-8")
    assert 'extern "C" int launch_w8a8_gemm' in source
    assert "int main(" not in source
    assert "launch_quant" not in source
    assert "cpu_quant_ref" not in source
    assert "check_output" not in source
    assert "myGEMM_kernel.hip" in manifest


def test_load_and_freeze_bundle(tmp_path):
    source = make_bundle(tmp_path / "source")
    spec = KernelTaskSpec.load(source / "task.yaml")
    assert spec.name == "unit-gemm"
    assert spec.private_case_ids == ["heldout"]
    assert spec.public_contract["abi"]["entrypoint"] == "launch_gemm"
    assert spec.agent_contract()["benchmark_shapes"][0]["shape"]["m"] == 2
    assert spec.benchmark_cases[0].shape == {"m": 2, "n": 3, "k": 4, "batch": 1}
    assert spec.benchmark_cases[0].flops == 48.0
    assert spec.benchmark_cases[0].bytes == 100.0

    frozen = FrozenEvaluatorBundle.materialize(source, tmp_path / "state" / "evaluator")
    frozen.verify()
    assert frozen.root != source


def test_frozen_bundle_detects_mutation(tmp_path):
    frozen = FrozenEvaluatorBundle.materialize(
        make_bundle(tmp_path / "source"), tmp_path / "frozen"
    )
    (frozen.root / "evaluate.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(SpecError, match="changed"):
        frozen.verify()


def test_resume_uses_snapshot_when_original_bundle_is_gone(tmp_path):
    source = make_bundle(tmp_path / "source")
    destination = tmp_path / "frozen"
    FrozenEvaluatorBundle.materialize(source, destination)
    for path in sorted(source.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    source.rmdir()
    resumed = FrozenEvaluatorBundle.materialize(source, destination)
    resumed.verify()


def test_profiler_work_metadata_must_be_positive(tmp_path):
    source = make_bundle(tmp_path / "source")
    text = (source / "task.yaml").read_text(encoding="utf-8")
    (source / "task.yaml").write_text(text.replace("bytes: 100", "bytes: -1"), encoding="utf-8")
    with pytest.raises(SpecError, match="bytes must be a positive"):
        KernelTaskSpec.load(source / "task.yaml")


def test_public_contract_is_required(tmp_path):
    source = make_bundle(tmp_path / "source")
    import yaml
    raw = yaml.safe_load((source / "task.yaml").read_text(encoding="utf-8"))
    raw.pop("public_contract")
    (source / "task.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SpecError, match="public_contract"):
        KernelTaskSpec.load(source / "task.yaml")


def test_benchmark_shape_is_required(tmp_path):
    source = make_bundle(tmp_path / "source")
    import yaml
    raw = yaml.safe_load((source / "task.yaml").read_text(encoding="utf-8"))
    raw["cases"]["benchmark"][0].pop("shape")
    (source / "task.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SpecError, match="requires shape metadata"):
        KernelTaskSpec.load(source / "task.yaml")
