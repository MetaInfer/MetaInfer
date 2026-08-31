from __future__ import annotations

import pytest

from ..orchestrator.w8a8_baselines import fixed_triton_graph_baseline


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ({"M": 2, "N": 1536, "K": 4096}, 66.924),
        ({"M": 16, "N": 1536, "K": 4096}, 66.597),
        ({"M": 2, "N": 8192, "K": 1024}, 47.181),
        ({"M": 16, "N": 8192, "K": 1024}, 48.049),
        ({"M": 2, "N": 4096, "K": 2048}, 54.453),
        ({"M": 16, "N": 4096, "K": 2048}, 54.617),
        ({"M": 2, "N": 1024, "K": 4096}, 60.389),
        ({"M": 16, "N": 1024, "K": 4096}, 60.965),
        ({"M": 2, "N": 4096, "K": 512}, 18.975),
        ({"M": 16, "N": 4096, "K": 512}, 21.372),
        ({"M": 3072, "N": 1536, "K": 4096}, 9488.0),
        ({"M": 3072, "N": 8192, "K": 1024}, 15437.0),
        ({"M": 3072, "N": 4096, "K": 2048}, 14680.0),
        ({"M": 3072, "N": 1024, "K": 4096}, 5985.0),
        ({"M": 3072, "N": 4096, "K": 512}, 4199.0),
        ({"M": 4096, "N": 1536, "K": 4096}, 13247.253),
        ({"M": 4096, "N": 8192, "K": 1024}, 20590.225),
        ({"M": 4096, "N": 4096, "K": 2048}, 19881.949),
        ({"M": 4096, "N": 1024, "K": 4096}, 8790.192),
        ({"M": 4096, "N": 4096, "K": 512}, 5545.821),
    ],
)
def test_fixed_triton_graph_baseline(shape, expected):
    record = fixed_triton_graph_baseline("shape", shape)
    assert record["median_us"] == expected
    assert record["baseline_kind"] == "triton_graph"
    expected_scope = (
        "prefill_graph_replay" if int(shape["M"]) > 16
        else "decode_graph_replay"
    )
    assert record["timing_scope"] == expected_scope
    assert record["distribution_stats_available"] is False


def test_prefill_baseline_has_prefill_timing_scope():
    record = fixed_triton_graph_baseline(
        "tp4_wqkv_a_m3072",
        {"M": 3072, "N": 1536, "K": 4096},
    )
    assert record["timing_scope"] == "prefill_graph_replay"


def test_fixed_baseline_rejects_unknown_shape():
    with pytest.raises(ValueError, match="no fixed Triton Graph baseline"):
        fixed_triton_graph_baseline(
            "m4_wqkv_a", {"M": 4, "N": 1536, "K": 4096}
        )


def test_bootstrap_metrics_are_kept_separate():
    bootstrap = {"passed": True, "median_us": 123.0}
    record = fixed_triton_graph_baseline(
        "m2_wqkv_a",
        {"M": 2, "N": 1536, "K": 4096},
        bootstrap_metrics=bootstrap,
    )
    assert record["median_us"] == 66.924
    assert record["bootstrap_metrics"] == bootstrap


@pytest.mark.parametrize(
    ("shape", "eager_median", "eager_p90", "graph_median", "graph_p90"),
    [
        ((16, 1536, 4096), 97.660, 98.404, 66.591, 67.347),
        ((16, 4096, 1024), 95.284, 96.092, 32.137, 35.969),
        ((16, 8192, 1024), 96.204, 96.796, 48.274, 50.794),
        ((16, 512, 4096), 96.056, 97.788, 59.210, 59.962),
        ((16, 4096, 256), 94.540, 95.740, 14.849, 18.001),
        ((3072, 1536, 4096), 9423.243, 9446.877, 9422.539, 9486.942),
        ((3072, 4096, 1024), 7637.209, 8033.128, 7617.880, 7686.395),
        ((3072, 8192, 1024), 15461.000, 15528.571, 15466.648, 15958.569),
        ((3072, 512, 4096), 2775.526, 2780.134, 2740.757, 2751.685),
        ((3072, 4096, 256), 2414.377, 2424.569, 2370.583, 2376.343),
    ],
)
def test_tp8_hot_cache_baselines(
    shape, eager_median, eager_p90, graph_median, graph_p90
):
    m, n, k = shape
    record = fixed_triton_graph_baseline(
        "tp8_shape", {"tp_size": 8, "M": m, "N": n, "K": k}
    )
    assert record["tp_size"] == 8
    assert record["median_us"] == graph_median
    assert record["p90_us"] == graph_p90
    assert record["eager_median_us"] == eager_median
    assert record["eager_p90_us"] == eager_p90
    assert record["cache_state"] == "hot"
    assert record["distribution_stats_available"] is True
    expected_protocol = (
        {"warmups": 50, "samples": 50, "launches_per_sample": 20}
        if m == 16
        else {"warmups": 10, "samples": 20, "launches_per_sample": 5}
    )
    assert record["measurement_protocol"] == expected_protocol


def test_same_shape_uses_tp_specific_baseline():
    shape = {"M": 16, "N": 1536, "K": 4096}
    assert fixed_triton_graph_baseline("tp4", {**shape, "tp_size": 4})[
        "median_us"
    ] == 66.597
    assert fixed_triton_graph_baseline("tp8", {**shape, "tp_size": 8})[
        "median_us"
    ] == 66.591
