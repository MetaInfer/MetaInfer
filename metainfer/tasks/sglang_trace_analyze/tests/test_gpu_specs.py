"""GPU specs lookup tests."""

from ..orchestrator.gpu_specs import GPU_SPECS, GpuSpec


def test_gpu_specs_known():
    for label in ("K100", "A100_80G", "H100", "B200"):
        spec = GPU_SPECS.get(label)
        assert spec is not None, f"missing spec for {label}"
        assert spec.bf16_tflops > 0
        assert spec.bandwidth_gb_s > 0


def test_gpu_specs_values_reasonable():
    k100 = GPU_SPECS["K100"]
    assert k100.bf16_tflops == 192
    assert k100.bandwidth_gb_s == 700
