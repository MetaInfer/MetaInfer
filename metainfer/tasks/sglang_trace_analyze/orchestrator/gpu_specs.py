"""GPU theoretical-peak lookup table.

Used by ``flops_calculator.py`` to compute MFU:
  MFU = actual_TFLOPS / theoretical_peak_TFLOPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class GpuSpec:
    """Theoretical peak numbers for one GPU model."""

    label: str
    fp32_tflops: float
    tf32_tflops: float
    bf16_tflops: float
    fp16_tflops: float
    int8_tops: float
    bandwidth_gb_s: float


GPU_SPECS: Dict[str, GpuSpec] = {
    "K100": GpuSpec(
        label="K100",
        fp32_tflops=49,
        tf32_tflops=98,
        bf16_tflops=192,
        fp16_tflops=192,
        int8_tops=392,
        bandwidth_gb_s=700,
    ),
    "A100_80G": GpuSpec(
        label="A100_80G",
        fp32_tflops=19.5,
        tf32_tflops=156,
        bf16_tflops=312,
        fp16_tflops=312,
        int8_tops=624,
        bandwidth_gb_s=2039,
    ),
    "H100": GpuSpec(
        label="H100",
        fp32_tflops=67,
        tf32_tflops=989,
        bf16_tflops=989,
        fp16_tflops=989,
        int8_tops=1979,
        bandwidth_gb_s=3350,
    ),
    "B200": GpuSpec(
        label="B200",
        fp32_tflops=90,
        tf32_tflops=2250,
        bf16_tflops=2250,
        fp16_tflops=2250,
        int8_tops=4500,
        bandwidth_gb_s=8000,
    ),
}
