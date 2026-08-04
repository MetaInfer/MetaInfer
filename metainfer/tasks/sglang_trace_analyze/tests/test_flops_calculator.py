"""FLOPs calculator tests."""

from ..orchestrator.gpu_specs import GpuSpec
from ..orchestrator.flops_calculator import (
    _estimate_flops,
    _estimate_bytes,
    calculate_mfu,
)


K100 = GpuSpec(
    label="K100",
    fp32_tflops=49,
    tf32_tflops=98,
    bf16_tflops=192,
    fp16_tflops=192,
    int8_tops=392,
    bandwidth_gb_s=700,
)


def test_estimate_flops_gemm_3d():
    # M=4096, K=2048, N=512 → 2*4096*2048*512 = 8,589,934,592
    flops = _estimate_flops("GEMM", [[4096, 2048, 512]], batch_size=1)
    assert flops == 2 * 4096 * 2048 * 512


def test_estimate_flops_gemm_batched():
    # B=4, M=1024, N=512, K=2048 → 2*4*1024*2048*512
    flops = _estimate_flops("GEMM", [[4, 1024, 512, 2048]], batch_size=4)
    assert flops == 2 * 4 * 1024 * 2048 * 512


def test_estimate_flops_no_dims():
    assert _estimate_flops("GEMM", [], batch_size=8) == 0


def test_estimate_bytes_gemm():
    bytes_moved = _estimate_bytes("GEMM", [[4096, 2048, 512]], batch_size=1)
    # (4096*2048 + 2048*512 + 4096*512) * 2 bytes
    expected = (4096 * 2048 + 2048 * 512 + 4096 * 512) * 2
    assert bytes_moved == expected


def test_calculate_mfu_basic():
    # 2*4096*2048*512 = 8.59e9 FLOPs. At 10 us this is ~859 TFLOPS
    # (far above K100 peak), but this is synthetic — we just verify
    # the fields are populated and reasonable.
    kernels = [
        {
            "kernel_name": "triton_gemm",
            "total_dur_us": 50,  # 50 us for 8.6e9 FLOPs = 172 TFLOPS
            "count": 1,
            "input_dims": [[4096, 2048, 512]],
            "op_type": "GEMM",
        }
    ]
    result = calculate_mfu(kernels, K100, batch_size=1, dtype="bf16")
    k = result[0]
    assert k["tflops_theoretical"] == 192
    assert k["tflops_actual"] > 0
    assert k["mfu"] > 0
    assert k["bound"] in ("compute", "memory")


def test_calculate_mfu_no_dims():
    kernels = [
        {
            "kernel_name": "cuda_graph_replay",
            "total_dur_us": 500_000,
            "count": 1,
            "input_dims": [],
            "op_type": "Other",
        }
    ]
    result = calculate_mfu(kernels, K100, batch_size=8, dtype="bf16")
    assert result[0]["tflops_actual"] == 0
    assert result[0]["mfu"] == 0
