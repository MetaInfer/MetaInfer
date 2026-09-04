"""Roofline-model headroom analysis for GPU kernel optimization.

Computes the kernel's position on the roofline using the standard model:

    P_achieved = FLOPs / T
    AI = FLOPs / Bytes_HBM
    P_bandwidth_roof = BW_peak × AI
    P_max = min(P_compute_peak, P_bandwidth_roof)
    roofline_efficiency = P_achieved / P_max

where FLOPs and Bytes are the *theoretical minimum* for the algorithm
(algorithmic FLOPs and compulsory HBM traffic).  When hipprof profiling
is enabled, measured HBM bytes can replace the theoretical estimate for
a measured-AI comparison.

Reference: Williams, Waterman, Patterson.  "Roofline: an insightful visual
performance model for multicore architectures."  CACM 52(4), 2009.

Integrated into Phase H (measure perf) — runs after the perf harness but
before the kernel enters the library.  Results are stored in KernelEntry
and displayed in the WebUI to guide future optimization iterations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================== #
# GPU Spec Database
# =========================================================================== #


@dataclass
class GpuSpec:
    name: str
    peak_hbm_bw_gbps: float        # peak HBM bandwidth (GB/s)
    peak_tflops_fp32: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_int8: float = 0.0
    warp_size: int = 64
    smem_per_cu_kb: int = 64
    cu_count: int = 120
    # Approximate effective L2 bandwidth (GB/s) — used for hierarchical roofline
    l2_bw_gbps: float = 2000.0
    # Approximate shared memory bandwidth per CU (TB/s)
    smem_bw_tbps: float = 10.0


GPU_SPECS: Dict[str, GpuSpec] = {
    "gfx928": GpuSpec(
        name="AMD DCU K500SM_AI (gfx928)",
        peak_hbm_bw_gbps=700.0,
        peak_tflops_fp32=55.0,
        peak_tflops_fp16=110.0,
        peak_tflops_bf16=110.0,
        peak_tflops_int8=220.0,
        warp_size=64,
        smem_per_cu_kb=64,
        cu_count=120,
        l2_bw_gbps=2000.0,
    ),
    "gfx942": GpuSpec(
        name="AMD Instinct MI300X (gfx942)",
        peak_hbm_bw_gbps=5300.0,
        peak_tflops_fp32=81.7,
        peak_tflops_fp16=163.4,
        peak_tflops_bf16=163.4,
        peak_tflops_int8=326.8,
        warp_size=64,
        smem_per_cu_kb=64,
        cu_count=304,
    ),
    "gfx90a": GpuSpec(
        name="AMD Instinct MI250X (gfx90a)",
        peak_hbm_bw_gbps=1600.0,
        peak_tflops_fp32=47.9,
        peak_tflops_fp16=191.6,
        peak_tflops_bf16=191.6,
        peak_tflops_int8=383.2,
        warp_size=64,
        smem_per_cu_kb=64,
        cu_count=220,
    ),
    "sm90": GpuSpec(
        name="NVIDIA H100 (sm90)",
        peak_hbm_bw_gbps=3350.0,
        peak_tflops_fp32=67.0,
        peak_tflops_fp16=989.0,
        peak_tflops_bf16=989.0,
        peak_tflops_int8=1979.0,
        warp_size=32,
        smem_per_cu_kb=228,
        cu_count=132,
    ),
    "sm89": GpuSpec(
        name="NVIDIA RTX 4090 (sm89)",
        peak_hbm_bw_gbps=1008.0,
        peak_tflops_fp32=82.6,
        peak_tflops_fp16=165.2,
        peak_tflops_bf16=165.2,
        peak_tflops_int8=330.3,
        warp_size=32,
        smem_per_cu_kb=128,
        cu_count=128,
    ),
}

_FALLBACK_SPEC = GpuSpec(
    name="Unknown GPU",
    peak_hbm_bw_gbps=500.0,
    peak_tflops_fp32=10.0,
    peak_tflops_fp16=20.0,
    peak_tflops_bf16=20.0,
    peak_tflops_int8=40.0,
)


def detect_gpu(req: Dict[str, Any], kernel_code: str = "") -> GpuSpec:
    """Detect the GPU model from requirements or kernel code."""
    search_text = ""
    for key in ("extra_notes", "notes", "target_hardware", "gpu_model"):
        val = req.get(key, "")
        if val:
            search_text += str(val) + "\n"
    search_text += kernel_code[:2000]

    for gpu_id, spec in GPU_SPECS.items():
        if gpu_id.lower() in search_text.lower():
            return spec

    name_map = {
        "mi300x": "gfx942", "mi250x": "gfx90a", "mi250": "gfx90a",
        "h100": "sm90", "h200": "sm90", "h800": "sm90",
        "rtx 4090": "sm89", "rtx4090": "sm89",
        "dcu": "gfx928", "k500sm": "gfx928",
    }
    for name_key, gpu_id in name_map.items():
        if name_key in search_text.lower():
            return GPU_SPECS[gpu_id]

    return _FALLBACK_SPEC


# =========================================================================== #
# Shape extraction
# =========================================================================== #

_PLAIN_SHAPE_PATTERN = re.compile(r'\b(\d+)\s*[×x]\s*(\d+)\s*[×x]\s*(\d+)\b')
_PERGPU_SHAPE_PATTERN = re.compile(
    r'\(\s*(?:M|m)\s*,\s*(\d+)\s*\)\s*@\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
)


@dataclass
class ProblemShape:
    M: Optional[int]   # None means variable (benchmarked across multiple M)
    N: int
    K: int
    label: str = ""


def extract_shapes(req: Dict[str, Any]) -> List[ProblemShape]:
    """Extract target problem shapes from requirements text."""
    shapes: List[ProblemShape] = []
    seen: set = set()

    search_text = ""
    for key in ("extra_notes", "notes", "problem_shapes", "target_shapes"):
        val = req.get(key, "")
        if val:
            search_text += str(val) + "\n"

    if not search_text:
        return shapes

    # Pattern "M×K × K×N" (variable M)
    for match in re.finditer(
        r'(?:M|m)\s*[×x]\s*(\d+)\s*[×x]\s*(\d+)',
        search_text
    ):
        k_val = int(match.group(1))
        n_val = int(match.group(2))
        key = (None, k_val, n_val)
        if key not in seen:
            seen.add(key)
            shapes.append(ProblemShape(M=None, N=n_val, K=k_val,
                                       label=f"M×{k_val} (K)×{n_val}"))

    # Pattern "(M, K) @ (K, N)"  (per-GPU splitter format)
    for match in _PERGPU_SHAPE_PATTERN.finditer(search_text):
        k_val = int(match.group(1))
        n_val = int(match.group(3))
        key = (None, k_val, n_val)
        if key not in seen:
            seen.add(key)
            shapes.append(ProblemShape(M=None, N=n_val, K=k_val,
                                       label=f"M×{k_val} (K)×{n_val}"))

    # Pattern "m×k×n"
    for match in re.finditer(
        r'(?:m)\s*[×x]\s*(\d+)\s*[×x]\s*(\d+)',
        search_text
    ):
        k_val = int(match.group(1))
        n_val = int(match.group(2))
        key = (None, k_val, n_val)
        if key not in seen:
            seen.add(key)
            shapes.append(ProblemShape(M=None, N=n_val, K=k_val,
                                       label=f"m×{k_val} (K)×{n_val}"))

    # Plain M×N×K
    for match in _PLAIN_SHAPE_PATTERN.finditer(search_text):
        m_val = int(match.group(1))
        n_val = int(match.group(2))
        k_val = int(match.group(3))
        key = (m_val, n_val, k_val)
        if key not in seen:
            seen.add(key)
            shapes.append(ProblemShape(M=m_val, N=n_val, K=k_val,
                                       label=f"{m_val}×{n_val}×{k_val}"))

    return shapes


def _resolve_shape_dim(val, default: int) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# =========================================================================== #
# FLOPs & Memory Traffic
# =========================================================================== #


def estimate_gemm_flops(M: int, N: int, K: int) -> int:
    """GEMM FLOPs: 2*M*N*K (one multiply + one add per output element).

    For int8 w8a8 GEMM, the core dot-product is int8→int32 with fp32 scaling.
    We count the matmul FLOPs; scaling adds ~M*N FLOPs (negligible for large K).
    """
    return 2 * M * N * K


def estimate_hbm_traffic(M: int, N: int, K: int,
                         a_dtype_bytes: int = 1,
                         b_dtype_bytes: int = 1,
                         out_dtype_bytes: int = 2,
                         has_scales: bool = True) -> Tuple[int, int, int]:
    """Theoretical *minimum* HBM traffic for a GEMM kernel.

    Assumes each matrix element is read from HBM exactly once and the output
    is written once.  Real kernels will exceed this due to cache misses,
    non-coalesced access, write-allocate, and instruction overhead.

    Returns (bytes_read, bytes_write, total_bytes).
    """
    bytes_read = M * K * a_dtype_bytes + K * N * b_dtype_bytes
    if has_scales:
        bytes_read += M * 4 + N * 4
    bytes_write = M * N * out_dtype_bytes
    return bytes_read, bytes_write, bytes_read + bytes_write


# =========================================================================== #
# Roofline Analysis
# =========================================================================== #
# Notation:
#   P_achieved = FLOPs / T                   (achieved TFLOPS)
#   AI = FLOPs / Bytes_HBM                   (arithmetic intensity, FLOP/byte)
#   P_bw_roof = BW_peak × AI                 (bandwidth ceiling, in TFLOPS)
#   AI_ridge = P_compute_peak / BW_peak      (ridge point, FLOP/byte)
#   P_max = min(P_compute_peak, P_bw_roof)   (roofline ceiling)
#   η = P_achieved / P_max                   (roofline efficiency)


@dataclass
class HeadroomResult:
    """Result of roofline headroom analysis for one representative shape."""

    # -- Input / problem --
    shape_label: str = ""
    M: int = 0
    N: int = 0
    K: int = 0
    exec_time_ms: float = 0.0

    # -- Algorithmic quantities (theoretical minimum) --
    total_flops: int = 0
    total_bytes_hbm: int = 0          # theoretical compulsory HBM bytes
    bytes_read: int = 0
    bytes_write: int = 0
    arithmetic_intensity: float = 0.0  # FLOP / byte (HBM-level, theoretical)

    # -- Achieved (derived from exec_time + theoretical bytes/FLOPs) --
    achieved_tflops: float = 0.0      # P_achieved
    achieved_bw_gbps: float = 0.0     # total_bytes / T  (not from profiler)

    # -- When hipprof data is available, these hold measured HBM bytes --
    measured_hbm_bytes: float = 0.0   # profiler-measured HBM traffic (0 = no data)
    measured_ai: float = 0.0          # AI from measured bytes

    # -- Hardware peaks --
    peak_bw_gbps: float = 0.0         # HBM bandwidth
    peak_tflops: float = 0.0          # compute peak (matched to dtype)
    gpu_name: str = ""

    # -- Roofline model --
    ai_ridge: float = 0.0             # P_compute / BW_hbm  (FLOP/byte)
    p_bandwidth_roof_tflops: float = 0.0   # BW_peak × AI
    p_max_tflops: float = 0.0         # min(P_compute, P_bw_roof)
    roofline_efficiency_pct: float = 0.0   # P_achieved / P_max × 100

    # -- Bottleneck classification --
    bottleneck: str = "unknown"
    headroom_pct: float = 0.0         # 100 - roofline_efficiency (simplified)
    shape_is_approximate: bool = False

    # -- Recommendations --
    suggestions: List[str] = field(default_factory=list)
    optimization_advice: str = ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_headroom(
    kernel_code: str,
    kernel_fn_name: str,
    exec_time_ms: float,
    req: Dict[str, Any],
    gpu_spec: Optional[GpuSpec] = None,
    a_dtype_bytes: int = 1,
    b_dtype_bytes: int = 1,
    out_dtype_bytes: int = 2,
    profiled_hbm_bytes: float = 0.0,   # from hipprof (0 = no data)
) -> HeadroomResult:
    """Run roofline headroom analysis for a kernel.

    Uses the original Roofline model:
      - Algorithmic FLOPs and theoretical minimum HBM bytes define AI.
      - Exec time gives P_achieved.
      - P_max = min(P_compute, BW_hbm × AI).
      - Efficiency = P_achieved / P_max.

    Args:
        kernel_code: Optimized kernel source (for shape fallback).
        kernel_fn_name: Kernel wrapper function name.
        exec_time_ms: Measured execution time from the perf harness.
        req: Task requirements dict (shapes, GPU model).
        gpu_spec: Pre-detected GPU spec.
        a_dtype_bytes, b_dtype_bytes: Element sizes for A, B matrices.
        out_dtype_bytes: Element size for output.
        profiled_hbm_bytes: Actual HBM bytes from hipprof (0 = use theoretical).

    Returns:
        HeadroomResult with all computed metrics.
    """
    # 1. Detect GPU
    if gpu_spec is None:
        gpu_spec = detect_gpu(req, kernel_code)

    # 2. Extract shapes — pick the largest-FLOPs representative shape
    shapes = extract_shapes(req)
    if not shapes:
        shapes = [_guess_shape_from_code(kernel_code)]

    explicit_shapes = [s for s in shapes if s.M is not None]
    candidates = explicit_shapes if explicit_shapes else shapes

    best_shape = candidates[0]
    best_flops = 0
    for s in candidates:
        m = _resolve_shape_dim(s.M, 4096)
        n = s.N
        k = s.K
        flops = estimate_gemm_flops(m, n, k)
        if flops > best_flops:
            best_flops = flops
            best_shape = s

    M = _resolve_shape_dim(best_shape.M, 4096)
    N = best_shape.N
    K = best_shape.K

    # 3. Pick compute peak for the kernel's dtype
    if a_dtype_bytes == 1 and b_dtype_bytes == 1:
        peak_tflops = gpu_spec.peak_tflops_int8
    elif out_dtype_bytes == 2:
        peak_tflops = gpu_spec.peak_tflops_fp16
    else:
        peak_tflops = gpu_spec.peak_tflops_fp32
    if peak_tflops <= 0:
        peak_tflops = gpu_spec.peak_tflops_fp32

    peak_bw = gpu_spec.peak_hbm_bw_gbps

    # 4. Algorithmic quantities
    total_flops = estimate_gemm_flops(M, N, K)
    bytes_read, bytes_write, total_bytes = estimate_hbm_traffic(
        M, N, K,
        a_dtype_bytes=a_dtype_bytes,
        b_dtype_bytes=b_dtype_bytes,
        out_dtype_bytes=out_dtype_bytes,
    )

    # Arithmetic intensity: FLOP / byte (HBM level, theoretical minimum)
    ai = total_flops / max(total_bytes, 1)

    # ----------------------------------------------------------------
    # 5. Roofline model
    # ----------------------------------------------------------------
    time_s = max(exec_time_ms / 1000.0, 1e-9)

    # Achieved throughput (from theoretical FLOPs / measured time)
    p_achieved = total_flops / time_s / 1e12          # TFLOPS
    bw_achieved = total_bytes / time_s / 1e9          # GB/s

    # Bandwidth ceiling expressed as TFLOPS
    p_bw_roof = peak_bw * ai / 1e3                    # TFLOPS
    # Overall roofline ceiling
    p_max = min(peak_tflops, p_bw_roof)

    # Roofline efficiency: how close is P_achieved to P_max?
    roofline_efficiency = (p_achieved / p_max * 100.0) if p_max > 0 else 0.0
    roofline_efficiency = min(100.0, roofline_efficiency)

    # Ridge point: where the bandwidth ceiling meets the compute ceiling
    ai_ridge = peak_tflops * 1e3 / peak_bw if peak_bw > 0 else 0.0

    # ----------------------------------------------------------------
    # 6. Measured AI (if profiler data available)
    # ----------------------------------------------------------------
    measured_ai = 0.0
    if profiled_hbm_bytes > 0:
        measured_ai = total_flops / profiled_hbm_bytes

    # ----------------------------------------------------------------
    # 7. Bottleneck classification
    # ----------------------------------------------------------------
    bottleneck, headroom_pct = _classify_bottleneck(
        ai=ai,
        ai_ridge=ai_ridge,
        roofline_efficiency=roofline_efficiency,
        p_achieved=p_achieved,
        p_max=p_max,
        peak_bw=peak_bw,
        peak_tflops=peak_tflops,
        bw_achieved=bw_achieved,
    )

    # ----------------------------------------------------------------
    # 8. Suggestions
    # ----------------------------------------------------------------
    suggestions, optimization_advice = _generate_suggestions(
        bottleneck=bottleneck,
        roofline_efficiency=roofline_efficiency,
        ai=ai, ai_ridge=ai_ridge,
        p_achieved=p_achieved, p_max=p_max,
        p_bw_roof=p_bw_roof,
        bw_achieved=bw_achieved, peak_bw=peak_bw,
        peak_tflops=peak_tflops,
        measured_ai=measured_ai,
        M=M, N=N, K=K,
        gpu_spec=gpu_spec,
    )

    shape_is_approximate = best_shape.M is None

    return HeadroomResult(
        shape_label=best_shape.label,
        M=M, N=N, K=K,
        exec_time_ms=exec_time_ms,
        total_flops=total_flops,
        total_bytes_hbm=total_bytes,
        bytes_read=bytes_read,
        bytes_write=bytes_write,
        arithmetic_intensity=round(ai, 2),
        achieved_tflops=round(p_achieved, 4),
        achieved_bw_gbps=round(bw_achieved, 2),
        measured_hbm_bytes=profiled_hbm_bytes,
        measured_ai=round(measured_ai, 2) if measured_ai > 0 else 0.0,
        peak_bw_gbps=peak_bw,
        peak_tflops=peak_tflops,
        gpu_name=gpu_spec.name,
        ai_ridge=round(ai_ridge, 2),
        p_bandwidth_roof_tflops=round(p_bw_roof, 4),
        p_max_tflops=round(p_max, 4),
        roofline_efficiency_pct=round(roofline_efficiency, 1),
        bottleneck=bottleneck,
        headroom_pct=round(headroom_pct, 1),
        shape_is_approximate=shape_is_approximate,
        suggestions=suggestions,
        optimization_advice=optimization_advice,
    )


def _guess_shape_from_code(kernel_code: str, default_M: int = 4096,
                           default_N: int = 4096, default_K: int = 4096) -> ProblemShape:
    """Fallback: guess shape from kernel comments or docstring."""
    for match in _PLAIN_SHAPE_PATTERN.finditer(kernel_code[:5000]):
        return ProblemShape(
            M=int(match.group(1)),
            N=int(match.group(2)),
            K=int(match.group(3)),
            label=f"{match.group(1)}×{match.group(2)}×{match.group(3)}",
        )
    return ProblemShape(M=default_M, N=default_N, K=default_K,
                        label=f"{default_M}×{default_N}×{default_K} (guessed)")


# =========================================================================== #
# Bottleneck Classification
# =========================================================================== #


def _classify_bottleneck(
    ai: float,
    ai_ridge: float,
    roofline_efficiency: float,
    p_achieved: float,
    p_max: float,
    peak_bw: float,
    peak_tflops: float,
    bw_achieved: float,
) -> Tuple[str, float]:
    """Classify the performance bottleneck using the roofline model.

    The regime is determined by AI relative to the ridge point:
      - AI < AI_ridge : memory-bound regime → ceiling is BW_peak × AI
      - AI ≥ AI_ridge : compute-bound regime → ceiling is P_compute_peak

    Within each regime, the achieved throughput relative to that ceiling
    determines whether the kernel is efficient or not.
    """
    NEAR_OPTIMAL = 90.0     # roofline efficiency ≥ 90% → near_optimal
    EFFICIENT = 60.0        # roofline efficiency ≥ 60% → well-bound
    INEFFICIENT = 30.0      # roofline efficiency < 30% → clearly inefficient

    if roofline_efficiency >= NEAR_OPTIMAL:
        bottleneck = "near_optimal"
    elif ai < ai_ridge:
        # Memory-bound regime — bottleneck is HBM bandwidth
        if roofline_efficiency >= EFFICIENT:
            bottleneck = "memory_bound"
        else:
            bottleneck = "inefficient"
    else:
        # Compute-bound regime — bottleneck is compute throughput
        if roofline_efficiency >= EFFICIENT:
            bottleneck = "compute_bound"
        else:
            bottleneck = "inefficient"

    headroom = max(0.0, 100.0 - roofline_efficiency)
    return bottleneck, headroom


# =========================================================================== #
# Suggestions
# =========================================================================== #


def _generate_suggestions(
    bottleneck: str,
    roofline_efficiency: float,
    ai: float,
    ai_ridge: float,
    p_achieved: float,
    p_max: float,
    p_bw_roof: float,
    bw_achieved: float,
    peak_bw: float,
    peak_tflops: float,
    measured_ai: float,
    M: int, N: int, K: int,
    gpu_spec: GpuSpec,
) -> Tuple[List[str], str]:
    """Generate optimization suggestions based on roofline analysis."""
    suggestions: List[str] = []
    parts: List[str] = []

    # Always show the roofline equation breakdown
    suggestions.append(
        f"Roofline: P_achieved = {p_achieved:.2f} TFLOPS, "
        f"AI = {ai:.1f} FLOP/byte, "
        f"P_bw_roof = BW×AI = {p_bw_roof:.2f} TFLOPS, "
        f"P_max = min(P_compute={peak_tflops:.0f}, P_bw_roof={p_bw_roof:.2f}) = {p_max:.2f} TFLOPS"
    )
    if measured_ai > 0:
        suggestions.append(
            f"Measured AI (from hipprof HBM counters): {measured_ai:.1f} FLOP/byte "
            f"(theoretical minimum: {ai:.1f}). "
            f"{'Memory access is efficient' if measured_ai >= ai * 0.7 else 'Memory access is suboptimal — consider coalescing and cache blocking'}"
        )

    if bottleneck == "near_optimal":
        suggestions.append(
            f"Kernel is at {roofline_efficiency:.0f}% of the hardware roofline "
            f"(P_max = {p_max:.2f} TFLOPS). "
            f"Further optimization on this problem size is unlikely to yield significant gains."
        )
        suggestions.append(
            "Consider: (a) radically different algorithm (e.g., split-K, persistent kernel), "
            "(b) larger problem size where this kernel's optimizations scale better, "
            "(c) rewriting in HIP C++ with hand-tuned assembly, or "
            "(d) declaring convergence."
        )
        parts.append(f"This kernel achieves {roofline_efficiency:.0f}% of the hardware roofline (P_max={p_max:.2f} TFLOPS).")
        parts.append("Near the limit. Consider a different algorithm, HIP C++ rewrite, or declaring convergence.")

    elif bottleneck == "memory_bound":
        bw_util_pct = bw_achieved / peak_bw * 100 if peak_bw > 0 else 0
        suggestions.append(
            f"HBM bandwidth is the bottleneck (AI={ai:.1f} < ridge={ai_ridge:.1f}). "
            f"Achieved BW: {bw_achieved:.0f} GB/s ({bw_util_pct:.0f}% of {peak_bw:.0f} GB/s peak). "
            f"Roofline efficiency: {roofline_efficiency:.0f}%."
        )
        suggestions.append(
            f"To improve: increase AI by raising data reuse. "
            f"Larger tile sizes (BLOCK_M, BLOCK_N) reduce HBM reads per FLOP."
        )
        suggestions.append(
            f"Current AI={ai:.1f}. Raising it to {ai_ridge:.1f} would move to the "
            f"compute-bound regime (needs {ai_ridge/ai:.1f}× more reuse)."
        )
        if gpu_spec.smem_per_cu_kb > 0:
            suggestions.append(
                f"Use shared memory ({gpu_spec.smem_per_cu_kb} KB/CU) to cache "
                f"B-tile across M iterations — avoids re-reading B from HBM."
            )
        parts.append("This kernel is memory-bound.")
        parts.append(f"AI={ai:.1f}, ridge={ai_ridge:.1f}. Increase data reuse to improve.")

    elif bottleneck == "compute_bound":
        suggestions.append(
            f"Compute throughput is the bottleneck (AI={ai:.1f} > ridge={ai_ridge:.1f}). "
            f"Roofline efficiency: {roofline_efficiency:.0f}%."
        )
        suggestions.append(
            f"Check tensor core utilization: ensure tile sizes (M={M}, N={N}, K={K}) "
            f"are compatible with gfx928 MMA instructions (16×16×32 int8)."
        )
        suggestions.append(
            f"Consider warp-level tuning: {gpu_spec.warp_size}-wide "
            f"operations for coalesced execution on {gpu_spec.cu_count} CUs."
        )
        parts.append("This kernel is compute-bound.")
        parts.append("Focus on instruction-level optimizations and tensor core utilization.")

    else:  # inefficient
        suggestions.append(
            f"Kernel is far from the roofline (efficiency={roofline_efficiency:.0f}%). "
            f"Achieved: {p_achieved:.2f} TFLOPS vs ceiling of {p_max:.2f} TFLOPS."
        )
        suggestions.append(
            "Check occupancy: may be limited by register pressure or shared memory allocation. "
            f"Target: {gpu_spec.cu_count} CUs × {gpu_spec.warp_size} threads."
        )
        suggestions.append(
            "Verify launch configuration: grid size, block size, wavefront occupancy. "
            "Profile with hardware counters to identify the specific stall reason."
        )
        parts.append("This kernel is inefficient — far from the roofline ceiling.")
        parts.append("Investigate occupancy, register pressure, tile alignment, and launch configuration.")

    optimization_advice = (
        f"**Bottleneck:** {bottleneck.replace('_', ' ').title()}\n\n"
        + " ".join(parts)
    )

    return suggestions, optimization_advice


# =========================================================================== #
# Serialization helpers for pipeline.py
# =========================================================================== #


def headroom_result_to_dict(result: HeadroomResult) -> Dict[str, Any]:
    """Serialize HeadroomResult to a JSON-serializable dict."""
    return asdict(result)


def headroom_dict_to_summary(d: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary for storage in kernel_library.json.

    Key roofline fields:
      roofline_efficiency_pct : P_achieved / P_max × 100
      p_max_tflops            : the kernel's roofline ceiling
      p_bandwidth_roof_tflops : BW_peak × AI
    """
    return {
        "bottleneck": d.get("bottleneck", "unknown"),
        "roofline_efficiency_pct": d.get("roofline_efficiency_pct", 0.0),
        "headroom_pct": d.get("headroom_pct", 0),
        "achieved_tflops": d.get("achieved_tflops", 0),
        "achieved_bw_gbps": d.get("achieved_bw_gbps", 0),
        "peak_bw_gbps": d.get("peak_bw_gbps", 0),
        "peak_tflops": d.get("peak_tflops", 0),
        "ai_ridge": d.get("ai_ridge", 0),
        "p_bandwidth_roof_tflops": d.get("p_bandwidth_roof_tflops", 0),
        "p_max_tflops": d.get("p_max_tflops", 0),
        "arithmetic_intensity": d.get("arithmetic_intensity", 0),
        "measured_ai": d.get("measured_ai", 0),
        "bw_util_pct": 0.0,   # deprecated — use roofline_efficiency_pct
        "compute_util_pct": 0.0,  # deprecated — use roofline_efficiency_pct
        "shape_is_approximate": d.get("shape_is_approximate", False),
        "shape_label": d.get("shape_label", ""),
        "gpu_name": d.get("gpu_name", ""),
        "suggestions": d.get("suggestions", []),
        "optimization_advice": d.get("optimization_advice", ""),
    }
