from __future__ import annotations

from pathlib import Path

import pytest

from ..orchestrator.evaluator.spec import FrozenEvaluatorBundle, KernelTaskSpec, SpecError
from ._helpers import make_bundle


def test_task_owned_harness_starter_has_a_valid_protocol():
    harness = Path(__file__).resolve().parents[1] / "harness" / "user_gemm"
    spec = KernelTaskSpec.load(harness / "task.yaml")
    assert spec.name == "deepseek-w8a8-gemm-tp4-tp8"
    assert set(spec.commands) == {"correctness", "benchmark", "profile"}
    assert len(spec.benchmark_cases) == 60
    assert len(spec.correctness_case_ids) == 64
    assert len(spec.private_case_ids) == 4
    assert spec.agent_contract()["abi"]["entrypoint"] == "launch_w8a8_gemm"
    assert {case.shape["m"] for case in spec.benchmark_cases} == {1, 2, 4, 8, 16, 4096}
    assert {case.shape["batch"] for case in spec.benchmark_cases} == {1}
    small = sum(case.weight for case in spec.benchmark_cases if case.shape["m"] <= 16)
    large = sum(case.weight for case in spec.benchmark_cases if case.shape["m"] == 4096)
    assert small == pytest.approx(large)


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
