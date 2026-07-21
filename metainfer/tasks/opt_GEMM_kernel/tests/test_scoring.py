from ..orchestrator.evaluator.scoring import compare_measurements, score_benchmark
from ..orchestrator.evaluator.spec import AcceptanceSpec, BenchmarkCaseSpec


def test_trace_weighted_score_and_critical_gate():
    result = score_benchmark(
        [
            {"id": "hot", "baseline_ms": 10, "candidate_ms": 5, "weight": 9, "critical": True},
            {"id": "cold", "baseline_ms": 10, "candidate_ms": 20, "weight": 1},
        ],
        ["hot", "cold"],
        AcceptanceSpec(min_weighted_speedup=1.2, max_critical_regression=0.03),
    )
    assert result.passed
    assert result.weighted_speedup == 100 / 65


def test_missing_shape_is_a_hard_failure():
    result = score_benchmark(
        [{"id": "a", "baseline_ms": 1, "candidate_ms": 0.5}],
        ["a", "b"],
        AcceptanceSpec(),
    )
    assert not result.passed
    assert result.missing_case_ids == ["b"]


def test_critical_regression_blocks_good_average():
    result = score_benchmark(
        [
            {"id": "hot", "baseline_ms": 100, "candidate_ms": 50, "weight": 10},
            {"id": "critical", "baseline_ms": 1, "candidate_ms": 1.1, "weight": 1, "critical": True},
        ],
        ["hot", "critical"],
        AcceptanceSpec(max_critical_regression=0.03),
    )
    assert not result.passed
    assert result.critical_regression > 0.09


def test_non_finite_latency_is_rejected():
    result = score_benchmark(
        [{"id": "a", "baseline_ms": 1, "candidate_ms": float("nan")}],
        ["a"],
        AcceptanceSpec(),
    )
    assert not result.passed


def test_profiler_rates_are_derived_from_frozen_case_spec():
    result = compare_measurements(
        [{"id": "gemm", "latency_ms": 2.0}],
        [{"id": "gemm", "latency_ms": 1.0, "flops": 1}],
        [BenchmarkCaseSpec(
            "gemm", weight=1.0, critical=True,
            shape={"m": 1000, "n": 1000, "k": 1000, "batch": 1},
            flops=2_000_000_000.0,
            bytes=1_000_000_000.0,
        )],
        AcceptanceSpec(),
    )
    case = result.cases[0]
    assert case["candidate_tflops"] == 2.0
    assert case["candidate_bandwidth_gbps"] == 1000.0
