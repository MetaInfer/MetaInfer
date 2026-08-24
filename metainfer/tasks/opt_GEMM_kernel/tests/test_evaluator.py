from ..orchestrator.evaluator.runner import EvaluatorRunner
from ..orchestrator.evaluator.spec import FrozenEvaluatorBundle
from ._helpers import make_bundle


def test_runner_executes_all_system_owned_gates(tmp_path):
    bundle = FrozenEvaluatorBundle.materialize(make_bundle(tmp_path / "source"), tmp_path / "frozen")
    submission = tmp_path / "submission"
    submission.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    runner = EvaluatorRunner(bundle)

    correctness = runner.run(
        "correctness", submission, artifacts, tmp_path / "reports",
        role="baseline", build_fingerprint="build-1",
    )
    assert correctness.passed
    assert correctness.report["summary"]["expected"] == 2
    protocol = bundle.spec.benchmark_protocol
    baseline = runner.validate_benchmark_report(
        {
            "passed": True,
            "methodology": protocol,
            "cases": [
                {"id": "small", "latency_ms": 1.0},
                {"id": "large", "latency_ms": 1.0},
            ],
        },
        role="baseline",
        build_fingerprint="build-1",
    )
    benchmark = runner.validate_benchmark_report(
        {
            "passed": True,
            "methodology": protocol,
            "cases": [
                {"id": "small", "latency_ms": 0.8},
                {"id": "large", "latency_ms": 0.8},
            ],
        },
        role="candidate",
        build_fingerprint="build-1",
        baseline_report=baseline.report,
    )
    assert baseline.passed
    assert benchmark.passed
    assert benchmark.report["score"]["worst_case_speedup"] == 1.25


def test_submission_symlink_is_rejected(tmp_path):
    bundle = FrozenEvaluatorBundle.materialize(make_bundle(tmp_path / "source"), tmp_path / "frozen")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "escape").symlink_to(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    try:
        EvaluatorRunner(bundle).run(
            "correctness", submission, artifacts, tmp_path / "reports",
            role="candidate", build_fingerprint="build-1",
        )
    except Exception as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("symlink submission should be rejected")


def test_duplicate_correctness_case_is_rejected(tmp_path):
    bundle = FrozenEvaluatorBundle.materialize(make_bundle(tmp_path / "source"), tmp_path / "frozen")
    runner = EvaluatorRunner(bundle)
    result = runner._validate("correctness", {
        "passed": True,
        "cases": [
            {"id": "public", "passed": True},
            {"id": "public", "passed": True},
            {"id": "heldout", "passed": True},
        ],
    })
    assert not result.passed
    assert result.report["summary"]["duplicate"] == ["public"]
