from ..orchestrator.evaluator.scoring import (
    compare_against_champion,
    compare_measurements,
)
from ..orchestrator.evaluator.spec import AcceptanceSpec, BenchmarkCaseSpec


def _spec(case_id: str) -> BenchmarkCaseSpec:
    return BenchmarkCaseSpec(
        case_id,
        shape={"m": 1, "n": 1, "k": 1, "batch": 1},
    )


def test_every_shape_must_beat_the_frozen_baseline():
    result = compare_measurements(
        [
            {"id": "hot", "latency_ms": 10.0},
            {"id": "cold", "latency_ms": 10.0},
        ],
        [
            {"id": "hot", "latency_ms": 5.0},
            {"id": "cold", "latency_ms": 20.0},
        ],
        [_spec("hot"), _spec("cold")],
        AcceptanceSpec(),
    )
    assert not result.passed
    assert result.failed_case_ids == ["cold"]
    assert result.worst_case_speedup == 0.5


def test_missing_shape_is_a_hard_failure():
    result = compare_measurements(
        [
            {"id": "a", "latency_ms": 1.0},
            {"id": "b", "latency_ms": 1.0},
        ],
        [{"id": "a", "latency_ms": 0.5}],
        [_spec("a"), _spec("b")],
        AcceptanceSpec(),
    )
    assert not result.passed
    assert result.missing_case_ids == ["b"]


def test_non_finite_latency_is_rejected():
    result = compare_measurements(
        [{"id": "a", "latency_ms": 1.0}],
        [{"id": "a", "latency_ms": float("nan")}],
        [_spec("a")],
        AcceptanceSpec(),
    )
    assert not result.passed
    assert any("positive" in reason for reason in result.reasons)


def test_profiler_rates_are_derived_from_frozen_case_spec():
    result = compare_measurements(
        [{"id": "gemm", "latency_ms": 2.0}],
        [{"id": "gemm", "latency_ms": 1.0, "flops": 1}],
        [BenchmarkCaseSpec(
            "gemm",
            shape={"m": 1000, "n": 1000, "k": 1000, "batch": 1},
            flops=2_000_000_000.0,
            bytes=1_000_000_000.0,
        )],
        AcceptanceSpec(),
    )
    case = result.cases[0]
    assert case["candidate_tflops"] == 2.0
    assert case["candidate_bandwidth_gbps"] == 1000.0


def test_champion_noise_gate_requires_every_shape_to_cross_threshold():
    result = compare_against_champion(
        [
            {"id": "a", "latency_ms": 1.0},
            {"id": "b", "latency_ms": 2.0},
        ],
        [
            {"id": "a", "latency_ms": 0.98},
            {"id": "b", "latency_ms": 1.99},
        ],
        ["a", "b"],
        0.01,
    )
    assert not result.passed
    assert result.failed_case_ids == ["b"]


def test_strict_baseline_gate_rejects_equal_latency():
    result = compare_against_champion(
        [{"id": "a", "latency_ms": 1.0}],
        [{"id": "a", "latency_ms": 1.0}],
        ["a"],
        0.0,
        strict=True,
    )
    assert not result.passed
    assert result.failed_case_ids == ["a"]
