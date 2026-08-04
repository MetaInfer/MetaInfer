"""Compute TFLOPS, bandwidth, and MFU for aggregated kernel entries.

Uses:
- ``gpu_specs.py`` for theoretical peak values
- kernel ``input_dims`` (from MAPPING trace) or shape rules (for CUDA Graph
  formal traces) to derive actual FLOP counts per invocation
- kernel ``total_dur_us`` to compute actual TFLOPS/bandwidth
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .gpu_specs import GpuSpec


def calculate_mfu(
    kernels: List[Dict[str, Any]],
    gpu_spec: GpuSpec,
    *,
    batch_size: int,
    dtype: str = "bf16",
) -> List[Dict[str, Any]]:
    """Augment each kernel entry with TFLOPS, bandwidth, MFU, and bound classification.

    Args:
        kernels: Aggregated kernel list. Each entry must have ``total_dur_us``
            and ``count``. Entries from a non-CUDA Graph trace may also have
            ``input_dims``, which are used for FLOP/byte estimation where available.
        gpu_spec: GPU theoretical peak specification.
        batch_size: Decode batch size used for this trace.
        dtype: Compute dtype — determines which TFLOPS peak to use.
            One of ``fp32``, ``tf32``, ``bf16``, ``fp16``, ``int8``.

    Returns:
        The same kernel list with added fields: ``tflops_actual``,
        ``bandwidth_gb_s``, ``mfu``, ``bound``, ``flops_per_invocation``.
    """
    theoretical_tflops = _theoretical_peak(gpu_spec, dtype)
    theoretical_bw = gpu_spec.bandwidth_gb_s

    for k in kernels:
        dur_s = k["total_dur_us"] / 1e6
        count = k.get("count", 1)
        dur_per_invocation_s = dur_s / count if count else dur_s
        dims = k.get("input_dims", [])
        op_type = k.get("op_type", "Other")

        flops = _estimate_flops(op_type, dims, batch_size)
        bytes_moved = _estimate_bytes(op_type, dims, batch_size)

        tflops_actual = (flops / dur_s / 1e12) if dur_s > 0 else 0
        bandwidth_gb_s = (bytes_moved / dur_s / 1e9) if dur_s > 0 else 0
        mfu = (tflops_actual / theoretical_tflops * 100) if theoretical_tflops > 0 else 0

        # Compute-bound vs memory-bound heuristic
        ops_per_byte = flops / bytes_moved if bytes_moved > 0 else float("inf")
        # "Roofline" crossover point = peak_flops / peak_bw ops/byte
        if theoretical_bw > 0:
            crossover = theoretical_tflops * 1e12 / (theoretical_bw * 1e9)
        else:
            crossover = float("inf")
        bound = "compute" if ops_per_byte > crossover else "memory"

        k["tflops_actual"] = round(tflops_actual, 3)
        k["tflops_theoretical"] = theoretical_tflops
        k["bandwidth_gb_s"] = round(bandwidth_gb_s, 1)
        k["bandwidth_theoretical"] = theoretical_bw
        k["mfu"] = round(mfu, 1)
        k["bound"] = bound
        k["flops_per_invocation"] = int(flops)

    return kernels


def _theoretical_peak(spec: GpuSpec, dtype: str) -> float:
    """Return theoretical peak TFLOPS for the given dtype."""
    return {
        "fp32": spec.fp32_tflops,
        "tf32": spec.tf32_tflops,
        "bf16": spec.bf16_tflops,
        "fp16": spec.fp16_tflops,
        "int8": spec.int8_tops,  # TOPS → TFLOPS approximate
    }.get(dtype, spec.bf16_tflops)


def _estimate_flops(
    op_type: str,
    dims: List[Any],
    batch_size: int,
) -> float:
    """Estimate FLOPs for one kernel invocation.

    For GEMM: 2*M*N*K (or 2*B*M*N*K for batched).
    For Attention: approximately 4*B*seq_len*head_dim*num_heads^2.
    For ElementWise: 2*num_elements.

    Returns 0 if dims are unavailable (CUDA Graph trace).
    """
    if not dims:
        return 0

    # Use the first observed dim list
    d = dims[0]

    if op_type == "GEMM":
        if isinstance(d, list) and len(d) >= 2:
            if len(d) == 3:
                M, K, N = int(d[0]), int(d[1]), int(d[2])
                return 2 * M * N * K
            B, M, N, K = _unpack_4d(d, batch_size)
            return 2 * B * M * N * K

    elif op_type == "Attention":
        if isinstance(d, list) and len(d) >= 3:
            seq_len = int(d[0])
            num_heads = int(d[1])
            head_dim = int(d[2])
            return 4 * seq_len * head_dim * num_heads * num_heads * batch_size

    elif op_type == "MoE":
        if isinstance(d, list) and len(d) >= 3:
            M, K, N = int(d[0]), int(d[1]), int(d[2])
            return 2 * M * N * K

    return 0


def _estimate_bytes(
    op_type: str,
    dims: List[Any],
    batch_size: int,
) -> float:
    """Estimate bytes moved (reads + writes) for one kernel invocation.

    Simple heuristic: for GEMM, input_bytes ≈ (M*K + K*N) * dtype_size,
    output_bytes ≈ M*N * dtype_size. For elementwise, ≈ 3 * num_elements.

    Returns 0 if dims are unavailable.
    """
    if not dims:
        return 0

    d = dims[0]
    dtype_size = 2  # bf16/fp16 default

    if op_type == "GEMM":
        if isinstance(d, list):
            if len(d) == 3:
                M, K, N = int(d[0]), int(d[1]), int(d[2])
                return (M * K + K * N + M * N) * dtype_size
            B, M, N, K = _unpack_4d(d, batch_size)
            return B * (M * K + K * N + M * N) * dtype_size

    elif op_type == "Attention":
        if isinstance(d, list) and len(d) >= 3:
            seq_len = int(d[0])
            num_heads = int(d[1])
            head_dim = int(d[2])
            # Q, K, V reads + output write (approximate)
            return batch_size * seq_len * num_heads * head_dim * 4 * dtype_size

    return 0


def _unpack_4d(
    dims: list,
    batch_size: int,
) -> tuple:
    """Unpack a 4-element dim list into (B, M, N, K), defaulting B to batch_size."""
    if len(dims) >= 4:
        return int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])
    if len(dims) == 3:
        return batch_size, int(dims[0]), int(dims[1]), int(dims[2])
    return batch_size, int(dims[0]), 1, 1
