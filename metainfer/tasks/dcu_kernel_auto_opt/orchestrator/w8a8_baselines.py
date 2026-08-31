"""Control-plane-owned W8A8 Triton Graph comparison baselines."""

from __future__ import annotations

from typing import Any, Dict, Mapping


# User-supplied, fixed decode/prefill Graph measurements. These values are
# comparison targets, not measurements of the child-generated bootstrap HIP
# source. Latencies are stored in microseconds.
_TRITON_GRAPH_BASELINES_US = {
    (4, 2, 1536, 4096): ("wqkv_a", 66.924),
    (4, 16, 1536, 4096): ("wqkv_a", 66.597),
    (4, 2, 8192, 1024): ("wq_b", 47.181),
    (4, 16, 8192, 1024): ("wq_b", 48.049),
    (4, 2, 4096, 2048): ("wo_b", 54.453),
    (4, 16, 4096, 2048): ("wo_b", 54.617),
    (4, 2, 1024, 4096): ("shared_gate_up", 60.389),
    (4, 16, 1024, 4096): ("shared_gate_up", 60.965),
    (4, 2, 4096, 512): ("shared_down", 18.975),
    (4, 16, 4096, 512): ("shared_down", 21.372),
    # M=3072 prefill baselines supplied on 2026-08-03.
    (4, 3072, 1536, 4096): ("wqkv_a", 9488.0),
    (4, 3072, 8192, 1024): ("wq_b", 15437.0),
    (4, 3072, 4096, 2048): ("wo_b", 14680.0),
    (4, 3072, 1024, 4096): ("shared_gate_up", 5985.0),
    (4, 3072, 4096, 512): ("shared_down", 4199.0),
    # M=4096 prefill baselines measured on 2026-08-06 (worker29, gfx928) with
    # lmslim int8_utils.matmul_kernel (the W8A8 Triton baseline), GPU events,
    # CUDA Graph replay, hot cache, warmups=10, samples=20, launches/sample=5.
    # The (4096, 8192, 1024) entry also covers indexer.wq_b, which shares the
    # same logical (K, N) shape as wq_b.
    (4, 4096, 1536, 4096): ("wqkv_a", 13247.253),
    (4, 4096, 8192, 1024): ("wq_b", 20590.225),
    (4, 4096, 4096, 2048): ("wo_b", 19881.949),
    (4, 4096, 1024, 4096): ("shared_gate_up", 8790.192),
    (4, 4096, 4096, 512): ("shared_down", 5545.821),
    # Model-catalog baselines measured on 2026-08-12 (worker29, gfx928) with
    # lmslim int8_utils.matmul_kernel (the W8A8 Triton baseline), GPU events,
    # CUDA Graph replay, hot cache, warmups=10, samples=20, launches/sample=5.
    # Config follows int8_utils.matmul_int8 defaults per M (decode M<=32 uses
    # BM16/BN32/BK256; M>1024 uses BM256/BN256/BK64). Covers the DeepSeek TP1
    # and TP8 M=2 additions plus the Hy3 (Hunyuan 3), MiniMax M3 and GLM5.2
    # catalogs. Keys are shared across models when (tp, M, N, K) coincide,
    # e.g. (1, *, 8192, 4096) is DeepSeek wo_b / Hy3 o_proj and (4, *, 8192,
    # 1024) covers indexer.wq_b.
    (1, 2, 1536, 4096): ("wqkv_a", 78.046),
    (1, 2, 2624, 6144): ("fused_qkv_a_proj", 96.623),
    (1, 2, 3072, 4096): ("shared_gate_up_proj", 81.854),
    (1, 2, 4096, 1536): ("shared_down_proj", 47.839),
    (1, 2, 4096, 2048): ("shared_down_proj", 62.351),
    (1, 2, 4096, 4096): ("shared_gate_up_proj", 109.998),
    (1, 2, 4096, 6144): ("shared_gate_up_proj", 151.045),
    (1, 2, 4096, 8192): ("wo_b", 194.748),
    (1, 2, 6144, 2048): ("shared_down_proj", 66.098),
    (1, 2, 6144, 3072): ("shared_down_proj", 92.174),
    (1, 2, 6144, 6144): ("shared_gate_up_proj", 155.341),
    (1, 2, 6144, 8192): ("o_proj", 190.044),
    (1, 2, 6144, 16384): ("o_proj", 371.539),
    (1, 2, 8192, 1024): ("wq_b", 55.599),
    (1, 2, 9216, 6144): ("qkv_proj", 232.987),
    (1, 2, 9856, 6144): ("qkv_proj_and_indexer_qk", 216.684),
    (1, 2, 10240, 4096): ("qkv_proj", 175.532),
    (1, 2, 16384, 2048): ("q_b_proj", 145.087),
    (1, 2, 28672, 512): ("kv_b_proj", 82.624),
    (1, 2, 32768, 1024): ("wq_b", 155.197),
    (1, 16, 1536, 4096): ("wqkv_a", 80.574),
    (1, 16, 2624, 6144): ("fused_qkv_a_proj", 98.463),
    (1, 16, 3072, 4096): ("shared_gate_up_proj", 82.638),
    (1, 16, 4096, 1536): ("shared_down_proj", 49.439),
    (1, 16, 4096, 2048): ("shared_down_proj", 65.567),
    (1, 16, 4096, 4096): ("shared_gate_up_proj", 110.606),
    (1, 16, 4096, 6144): ("shared_gate_up_proj", 151.685),
    (1, 16, 4096, 8192): ("wo_b", 205.644),
    (1, 16, 6144, 2048): ("shared_down_proj", 67.346),
    (1, 16, 6144, 3072): ("shared_down_proj", 91.838),
    (1, 16, 6144, 6144): ("shared_gate_up_proj", 156.653),
    (1, 16, 6144, 8192): ("o_proj", 192.412),
    (1, 16, 6144, 16384): ("o_proj", 358.435),
    (1, 16, 8192, 1024): ("wq_b", 56.847),
    (1, 16, 9216, 6144): ("qkv_proj", 235.083),
    (1, 16, 9856, 6144): ("qkv_proj_and_indexer_qk", 219.564),
    (1, 16, 10240, 4096): ("qkv_proj", 171.069),
    (1, 16, 16384, 2048): ("q_b_proj", 146.719),
    (1, 16, 28672, 512): ("kv_b_proj", 83.504),
    (1, 16, 32768, 1024): ("wq_b", 158.989),
    (1, 3072, 1536, 4096): ("wqkv_a", 9795.766),
    (1, 3072, 2624, 6144): ("fused_qkv_a_proj", 28944.543),
    (1, 3072, 3072, 4096): ("shared_gate_up_proj", 21047.242),
    (1, 3072, 4096, 1536): ("shared_down_proj", 11363.852),
    (1, 3072, 4096, 2048): ("shared_down_proj", 14921.346),
    (1, 3072, 4096, 4096): ("shared_gate_up_proj", 29325.629),
    (1, 3072, 4096, 6144): ("shared_gate_up_proj", 43830.049),
    (1, 3072, 4096, 8192): ("wo_b", 58484.25),
    (1, 3072, 6144, 2048): ("shared_down_proj", 22320.748),
    (1, 3072, 6144, 3072): ("shared_down_proj", 33673.799),
    (1, 3072, 6144, 6144): ("shared_gate_up_proj", 67255.209),
    (1, 3072, 6144, 8192): ("o_proj", 90545.355),
    (1, 3072, 6144, 16384): ("o_proj", 182662.598),
    (1, 3072, 8192, 1024): ("wq_b", 15305.526),
    (1, 3072, 9216, 6144): ("qkv_proj", 103111.029),
    (1, 3072, 9856, 6144): ("qkv_proj_and_indexer_qk", 110813.373),
    (1, 3072, 10240, 4096): ("qkv_proj", 74301.431),
    (1, 3072, 16384, 2048): ("q_b_proj", 59796.738),
    (1, 3072, 28672, 512): ("kv_b_proj", 29183.929),
    (1, 3072, 32768, 1024): ("wq_b", 62264.438),
    (4, 2, 768, 4096): ("shared_gate_up_proj", 71.711),
    (4, 2, 1024, 6144): ("shared_gate_up_proj", 108.278),
    (4, 2, 1536, 6144): ("shared_gate_up_proj", 109.822),
    (4, 2, 2304, 6144): ("qkv_proj", 113.134),
    (4, 2, 2560, 4096): ("qkv_proj", 77.854),
    (4, 2, 2560, 6144): ("qkv_proj_and_indexer_qk", 111.79),
    (4, 2, 2624, 6144): ("fused_qkv_a_proj", 98.756),
    (4, 2, 4096, 384): ("shared_down_proj", 21.664),
    (4, 2, 6144, 512): ("shared_down_proj", 23.553),
    (4, 2, 6144, 768): ("shared_down_proj", 33.407),
    (4, 2, 6144, 2048): ("o_proj", 65.135),
    (4, 2, 6144, 4096): ("o_proj", 110.501),
    (4, 2, 7168, 512): ("kv_b_proj", 25.009),
    (4, 16, 768, 4096): ("shared_gate_up_proj", 73.262),
    (4, 16, 1024, 6144): ("shared_gate_up_proj", 108.422),
    (4, 16, 1536, 6144): ("shared_gate_up_proj", 113.006),
    (4, 16, 2304, 6144): ("qkv_proj", 113.422),
    (4, 16, 2560, 4096): ("qkv_proj", 80.19),
    (4, 16, 2560, 6144): ("qkv_proj_and_indexer_qk", 112.27),
    (4, 16, 2624, 6144): ("fused_qkv_a_proj", 98.708),
    (4, 16, 4096, 384): ("shared_down_proj", 23.216),
    (4, 16, 6144, 512): ("shared_down_proj", 24.977),
    (4, 16, 6144, 768): ("shared_down_proj", 34.063),
    (4, 16, 6144, 2048): ("o_proj", 67.535),
    (4, 16, 6144, 4096): ("o_proj", 112.198),
    (4, 16, 7168, 512): ("kv_b_proj", 27.793),
    (4, 3072, 768, 4096): ("shared_gate_up_proj", 4988.476),
    (4, 3072, 1024, 6144): ("shared_gate_up_proj", 9708.601),
    (4, 3072, 1536, 6144): ("shared_gate_up_proj", 14650.233),
    (4, 3072, 2304, 6144): ("qkv_proj", 22634.012),
    (4, 3072, 2560, 4096): ("qkv_proj", 16984.091),
    (4, 3072, 2560, 6144): ("qkv_proj_and_indexer_qk", 25396.649),
    (4, 3072, 2624, 6144): ("fused_qkv_a_proj", 28947.94),
    (4, 3072, 4096, 384): ("shared_down_proj", 3366.413),
    (4, 3072, 6144, 512): ("shared_down_proj", 6270.013),
    (4, 3072, 6144, 768): ("shared_down_proj", 8900.2),
    (4, 3072, 6144, 2048): ("o_proj", 22307.655),
    (4, 3072, 6144, 4096): ("o_proj", 44517.157),
    (4, 3072, 7168, 512): ("kv_b_proj", 7308.89),
    (4, 4096, 768, 4096): ("shared_gate_up_proj", 6678.282),
    (4, 4096, 1024, 6144): ("shared_gate_up_proj", 13036.336),
    (4, 4096, 1536, 6144): ("shared_gate_up_proj", 19776.488),
    (4, 4096, 2304, 6144): ("qkv_proj", 31556.192),
    (4, 4096, 2560, 4096): ("qkv_proj", 24045.12),
    (4, 4096, 2560, 6144): ("qkv_proj_and_indexer_qk", 35918.768),
    (4, 4096, 2624, 6144): ("fused_qkv_a_proj", 40458.051),
    (4, 4096, 4096, 384): ("shared_down_proj", 4381.289),
    (4, 4096, 6144, 512): ("shared_down_proj", 8243.047),
    (4, 4096, 6144, 768): ("shared_down_proj", 11759.473),
    (4, 4096, 6144, 2048): ("o_proj", 29702.473),
    (4, 4096, 6144, 4096): ("o_proj", 58712.341),
    (4, 4096, 7168, 512): ("kv_b_proj", 9702.386),
    (8, 2, 384, 4096): ("shared_gate_up_proj", 64.511),
    (8, 2, 512, 4096): ("shared_gate_up_proj", 72.687),
    (8, 2, 512, 6144): ("shared_gate_up_proj", 97.606),
    (8, 2, 768, 6144): ("shared_gate_up_proj", 104.351),
    (8, 2, 1280, 4096): ("qkv_proj", 73.535),
    (8, 2, 1280, 6144): ("qkv_proj", 110.638),
    (8, 2, 1536, 4096): ("wqkv_a", 74.767),
    (8, 2, 1536, 6144): ("qkv_proj_and_indexer_qk", 110.19),
    (8, 2, 2048, 2048): ("q_b_proj", 40.371),
    (8, 2, 2624, 6144): ("fused_qkv_a_proj", 95.43),
    (8, 2, 3584, 512): ("kv_b_proj", 17.873),
    (8, 2, 4096, 192): ("shared_down_proj", 16.144),
    (8, 2, 4096, 256): ("shared_down_proj", 19.344),
    (8, 2, 4096, 1024): ("wq_b_or_wo_b", 36.271),
    (8, 2, 6144, 256): ("shared_down_proj", 16.881),
    (8, 2, 6144, 384): ("shared_down_proj", 21.968),
    (8, 2, 6144, 1024): ("o_proj", 37.407),
    (8, 2, 6144, 2048): ("o_proj", 64.948),
    (8, 2, 8192, 1024): ("wq_b", 56.239),
    (8, 16, 384, 4096): ("shared_gate_up_proj", 66.527),
    (8, 16, 512, 6144): ("shared_gate_up_proj", 99.318),
    (8, 16, 768, 6144): ("shared_gate_up_proj", 104.527),
    (8, 16, 1280, 4096): ("qkv_proj", 74.782),
    (8, 16, 1280, 6144): ("qkv_proj", 110.478),
    (8, 16, 1536, 6144): ("qkv_proj_and_indexer_qk", 111.486),
    (8, 16, 2048, 2048): ("q_b_proj", 42.035),
    (8, 16, 2624, 6144): ("fused_qkv_a_proj", 97.99),
    (8, 16, 3584, 512): ("kv_b_proj", 18.833),
    (8, 16, 4096, 192): ("shared_down_proj", 17.312),
    (8, 16, 6144, 256): ("shared_down_proj", 18.177),
    (8, 16, 6144, 384): ("shared_down_proj", 23.152),
    (8, 16, 6144, 1024): ("o_proj", 38.655),
    (8, 16, 6144, 2048): ("o_proj", 66.98),
    (8, 3072, 384, 4096): ("shared_gate_up_proj", 3009.348),
    (8, 3072, 512, 6144): ("shared_gate_up_proj", 4355.572),
    (8, 3072, 768, 6144): ("shared_gate_up_proj", 7087.451),
    (8, 3072, 1280, 4096): ("qkv_proj", 8489.847),
    (8, 3072, 1280, 6144): ("qkv_proj", 12553.842),
    (8, 3072, 1536, 6144): ("qkv_proj_and_indexer_qk", 14614.569),
    (8, 3072, 2048, 2048): ("q_b_proj", 6821.235),
    (8, 3072, 2624, 6144): ("fused_qkv_a_proj", 29185.017),
    (8, 3072, 3584, 512): ("kv_b_proj", 3685.827),
    (8, 3072, 4096, 192): ("shared_down_proj", 2043.847),
    (8, 3072, 6144, 256): ("shared_down_proj", 3593.156),
    (8, 3072, 6144, 384): ("shared_down_proj", 4925.667),
    (8, 3072, 6144, 1024): ("o_proj", 11664.713),
    (8, 3072, 6144, 2048): ("o_proj", 22305.402),
}


# TP8 hot-cache measurements supplied in milliseconds and normalized here to
# microseconds. Graph median is the optimization comparison target; eager and
# both P90 values are retained as measurement evidence.
_TRITON_TP8_BASELINES_US = {
    (8, 16, 1536, 4096): ("wqkv_a", 97.660, 98.404, 66.591, 67.347),
    (8, 16, 4096, 1024): ("wq_b_or_wo_b", 95.284, 96.092, 32.137, 35.969),
    (8, 16, 8192, 1024): ("indexer.wq_b", 96.204, 96.796, 48.274, 50.794),
    (8, 16, 512, 4096): ("shared_gate_up", 96.056, 97.788, 59.210, 59.962),
    (8, 16, 4096, 256): ("shared_down", 94.540, 95.740, 14.849, 18.001),
    (8, 3072, 1536, 4096): ("wqkv_a", 9423.243, 9446.877, 9422.539, 9486.942),
    (8, 3072, 4096, 1024): ("wq_b_or_wo_b", 7637.209, 8033.128, 7617.880, 7686.395),
    (8, 3072, 8192, 1024): ("indexer.wq_b", 15461.000, 15528.571, 15466.648, 15958.569),
    (8, 3072, 512, 4096): ("shared_gate_up", 2775.526, 2780.134, 2740.757, 2751.685),
    (8, 3072, 4096, 256): ("shared_down", 2414.377, 2424.569, 2370.583, 2376.343),
}


def fixed_triton_graph_baseline(
    shape_id: str,
    shape: Mapping[str, Any],
    *,
    bootstrap_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return the immutable Graph baseline for one exact M/N/K shape.

    ``bootstrap_metrics`` remains separate because it describes the current
    child HIP source used for PMC, rollback and iterative improvement.
    """
    try:
        # Legacy callers predate TP-aware shape metadata and are TP4-only.
        tp_size = int(shape.get("tp_size", 4))
        key = (
            tp_size,
            int(shape["M"]),
            int(shape["N"]),
            int(shape["K"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid W8A8 shape for fixed baseline: {shape_id}: {shape}"
        ) from exc
    tp8_measurement = _TRITON_TP8_BASELINES_US.get(key)
    fixed_measurement = _TRITON_GRAPH_BASELINES_US.get(key)
    if tp8_measurement is None and fixed_measurement is None:
        raise ValueError(
            "no fixed Triton Graph baseline for "
            f"{shape_id} with TP={key[0]}, M={key[1]}, "
            f"N={key[2]}, K={key[3]}"
        )

    if tp8_measurement is not None:
        case, eager_median_us, eager_p90_us, latency_us, graph_p90_us = (
            tp8_measurement
        )
    else:
        assert fixed_measurement is not None
        case, latency_us = fixed_measurement

    record: Dict[str, Any] = {
        "median_us": latency_us,
        "baseline_us": latency_us,
        "baseline_kind": "triton_graph",
        "case": case,
        "tp_size": tp_size,
        "shape": {"M": key[1], "N": key[2], "K": key[3]},
        "source": "user_supplied_fixed_table",
        "timing_scope": (
            "prefill_graph_replay" if key[1] > 16
            else "decode_graph_replay"
        ),
        "distribution_stats_available": tp8_measurement is not None,
    }
    if tp8_measurement is not None:
        record.update(
            {
                "p90_us": graph_p90_us,
                "eager_median_us": eager_median_us,
                "eager_p90_us": eager_p90_us,
                "cache_state": "hot",
                "measurement_protocol": {
                    "warmups": 50 if key[1] == 16 else 10,
                    "samples": 50 if key[1] == 16 else 20,
                    "launches_per_sample": 20 if key[1] == 16 else 5,
                },
            }
        )
    if bootstrap_metrics is not None:
        record["bootstrap_metrics"] = dict(bootstrap_metrics)
    return record
