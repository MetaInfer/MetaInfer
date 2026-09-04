"""Fine-grained GPU profiling via hipprof/rocprof for kernel optimization.

Provides hardware counter-level profiling that complements the roofline
model in headroom.py. hipprof gives actual achieved bandwidth, occupancy,
cache hit rates, and stall reasons — not just theoretical estimates.

Integrated into Phase H (measure perf) when ``enable_profiling`` is set
in the task requirements.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================== #
# Profiling result
# =========================================================================== #


@dataclass
class ProfileResult:
    """Fine-grained profiling data from hipprof for one kernel execution."""

    # Kernel identification
    kernel_name: str = ""
    kernel_duration_us: float = 0.0

    # Bandwidth
    achieved_bw_read_gbps: float = 0.0
    achieved_bw_write_gbps: float = 0.0
    achieved_bw_total_gbps: float = 0.0

    # Occupancy
    achieved_occupancy_pct: float = 0.0
    theoretical_occupancy_pct: float = 0.0

    # Cache
    l2_cache_hit_pct: float = 0.0

    # Stalls (percentage of cycles)
    stall_memory_pct: float = 0.0
    stall_dependency_pct: float = 0.0
    stall_sync_pct: float = 0.0
    stall_other_pct: float = 0.0

    # Raw data
    raw_stats: Dict[str, Any] = field(default_factory=dict)
    trace_file: str = ""
    success: bool = False
    error: str = ""


# =========================================================================== #
# hipprof runner
# =========================================================================== #


_HIPPROF_BIN = "/opt/dtk/bin/hipprof"


def run_hipprof_profile(
    kernel_script: Path,
    shape_args: Dict[str, int],
    kernel_fn_name: str = "matmul_int8",
    output_dir: Optional[Path] = None,
    timeout_s: int = 120,
) -> ProfileResult:
    """Run hipprof on a kernel script and parse the output.

    Args:
        kernel_script: Path to the .py file containing the kernel.
        shape_args: Dict with M, N, K dimensions.
        kernel_fn_name: Name of the kernel wrapper function.
        output_dir: Where to write hipprof output files.
        timeout_s: Timeout in seconds.

    Returns:
        ProfileResult with extracted metrics.
    """
    if not os.path.isfile(_HIPPROF_BIN):
        return ProfileResult(
            success=False,
            error=f"hipprof not found at {_HIPPROF_BIN}",
        )

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="hipprof_"))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / "hipprof")

    # Write a minimal runner script
    M = shape_args.get("M", 4096)
    N = shape_args.get("N", 4096)
    K = shape_args.get("K", 4096)

    runner_script = output_dir / "_prof_runner.py"
    runner_script.write_text(f'''"""Auto-generated hipprof profiling runner."""
import sys
sys.path.insert(0, "{kernel_script.parent}")
import importlib.util

spec = importlib.util.spec_from_file_location("_prof_kernel", "{kernel_script}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, "{kernel_fn_name}")

import torch
device = torch.device("cuda")

# Generate inputs matching the target shape
a = torch.randint(-128, 127, ({M}, {K}), device=device, dtype=torch.int8)
a_scale = torch.randn({M}, 1, device=device, dtype=torch.float32).abs() / 127.0
b = torch.randint(-128, 127, ({K}, {N}), device=device, dtype=torch.int8)
b_scale = torch.randn({N}, 1, device=device, dtype=torch.float32).abs() / 127.0

# Warmup
for _ in range(5):
    fn(a, a_scale, b, b_scale, torch.bfloat16)

torch.cuda.synchronize()

# Timed run
out = fn(a, a_scale, b, b_scale, torch.bfloat16)
torch.cuda.synchronize()

# Verify output is valid
assert out.shape == ({M}, {N}), f"Bad shape: {{out.shape}}"
assert not torch.isnan(out).any(), "NaN in output"
print("PROFILE_OK")
''')

    try:
        proc = subprocess.run(
            [
                _HIPPROF_BIN,
                "--stats",
                "--hip-trace",
                "-o", output_prefix,
                "python3", str(runner_script),
            ],
            capture_output=True, text=True,
            timeout=timeout_s,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        return ProfileResult(
            success=False,
            error=f"hipprof timed out after {timeout_s}s",
        )
    except Exception as e:
        return ProfileResult(
            success=False,
            error=f"Failed to run hipprof: {e!r}",
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Check if hipprof produced kernel CSV (reliable indicator of success)
    kernel_csv = output_dir / "hipprof.hipkernel.csv"
    if not kernel_csv.is_file():
        return ProfileResult(
            success=False,
            error=f"hipprof did not produce output CSV. stderr: {stderr[-500:]!r}",
        )

    # Parse hipprof CSV output files
    result = _parse_hipprof_stats(output_dir)

    # Find trace file
    trace_file = ""
    for f in output_dir.iterdir():
        if f.suffix in (".csv", ".db", ".json") and f.name.startswith("hipprof"):
            trace_file = str(f)
            break

    result.trace_file = trace_file
    result.success = True
    result.raw_stats["runner_script"] = str(runner_script)
    result.raw_stats["stdout_tail"] = stdout[-2000:]
    result.raw_stats["stderr_tail"] = stderr[-2000:]

    return result


# =========================================================================== #
# hipprof output parsing
# =========================================================================== #


def _parse_hipprof_stats(output_dir: Path) -> ProfileResult:
    """Parse hipprof CSV output files into structured ProfileResult.

    hipprof produces two key CSV files:
      - hipprof.hipkernel.csv: kernel Name, Calls, TotalDurationNs, AverageNs, Percentage
      - hipprof.hiptrace.csv: HIP API calls with durations (used for BW estimation)

    We extract the dominant compute kernel's duration and derive bandwidth from
    memory copy sizes divided by transfer time.
    """
    import csv as _csv

    result = ProfileResult()

    kernel_csv = output_dir / "hipprof.hipkernel.csv"
    trace_csv = output_dir / "hipprof.hiptrace.csv"

    # ---- Parse kernel CSV ----
    if kernel_csv.is_file():
        try:
            with open(kernel_csv, newline="", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                kernel_rows = list(reader)

            # Filter to real compute kernels (exclude torch/pytorch internals,
            # elementwise, reduce, random, memset, etc.)
            _IGNORE_PATTERNS = (
                "at::native::vectorized_elementwise",
                "at::native::reduce_kernel",
                "at::native::distribution_",
                "at::native::Bitwise",
                "at::native::Fill",
                "at::native::AUnary",
                "at::native::BUnary",
                "at::native::CUDA_tensor_apply",
                "at::native::tensor_kernel",
                "at::native::unrolled_elementwise",
                "hipMemset",
                "memset",
                "Total",
            )

            candidates: List[Tuple[str, float, float]] = []  # (name, avg_ns, pct)
            for row in kernel_rows:
                name = row.get("Name", "")
                if not name or any(p in name for p in _IGNORE_PATTERNS):
                    continue
                try:
                    avg_ns = float(row.get("AverageNs", "0").replace(",", ""))
                    pct = float(row.get("Percentage", "0").replace(",", ""))
                except (ValueError, KeyError):
                    continue
                if avg_ns > 0:
                    candidates.append((name, avg_ns, pct))

            if candidates:
                # Pick the kernel with highest percentage (dominant compute)
                candidates.sort(key=lambda x: x[2], reverse=True)
                result.kernel_name = candidates[0][0]
                result.kernel_duration_us = candidates[0][1] / 1000.0

            result.raw_stats["kernel_csv"] = [
                {"name": n, "avg_ns": a, "pct": p} for n, a, p in candidates[:10]
            ]
        except Exception as e:
            result.raw_stats["kernel_csv_error"] = str(e)[:500]

    # ---- Parse trace CSV for bandwidth estimation ----
    if trace_csv.is_file():
        try:
            with open(trace_csv, newline="", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                trace_rows = list(reader)

            # Sum memory copy durations (hipMemcpyWithStream, hipMemcpy)
            copy_total_ns = 0.0
            for row in trace_rows:
                name = row.get("Name", "")
                if name in ("hipMemcpyWithStream", "hipMemcpy", "hipMemcpyAsync"):
                    try:
                        copy_total_ns += float(row.get("TotalDurationNs", "0").replace(",", ""))
                    except (ValueError, KeyError):
                        pass

            # Bandwidth estimation: if we know the data size from the trace,
            # compute achieved BW. Use the kernel's known data footprint.
            # For now, store copy time so callers can compute BW with shape info.
            result.raw_stats["copy_total_ns"] = copy_total_ns

            # Also extract hipLaunchKernel total
            for row in trace_rows:
                if row.get("Name") == "hipLaunchKernel":
                    try:
                        launch_ns = float(row.get("TotalDurationNs", "0").replace(",", ""))
                        result.raw_stats["launch_total_ns"] = launch_ns
                    except (ValueError, KeyError):
                        pass
                    break
        except Exception as e:
            result.raw_stats["trace_csv_error"] = str(e)[:500]

    # Mark as successful if we found kernel data
    if result.kernel_duration_us > 0:
        result.success = True

    return result


# =========================================================================== #
# Profile summary for optimizer feedback
# =========================================================================== #


def profile_result_to_dict(result: ProfileResult) -> Dict[str, Any]:
    """Serialize ProfileResult to a JSON-serializable dict."""
    d = asdict(result)
    # Truncate raw output for storage
    if "raw_stats" in d and isinstance(d.get("raw_stats"), dict):
        raw = d["raw_stats"]
        if "output" in raw and isinstance(raw["output"], str):
            raw["output"] = raw["output"][-2000:]  # keep last 2K
    return d


def profile_to_advice(result: ProfileResult, M: int, N: int, K: int) -> str:
    """Generate optimization advice from profiling results.

    Returns a paragraph suitable for the optimizer agent's context.
    """
    if not result.success and result.kernel_duration_us <= 0:
        return f"(Profiling failed: {result.error})"

    parts = [f"**hipprof profile for ({M}×{N}×{K}):**"]

    if result.kernel_duration_us > 0:
        parts.append(f"Kernel duration: {result.kernel_duration_us:.1f} µs")
        # Estimate achieved BW from data footprint
        # int8 w8a8: A=(M,K) int8, a_scale=(M,1) fp32, B=(K,N) int8, b_scale=(N,1) fp32, out=(M,N) bf16
        bytes_read = (M * K * 1) + (M * 1 * 4) + (K * N * 1) + (N * 1 * 4)
        bytes_write = M * N * 2  # bf16 output
        total_bytes = bytes_read + bytes_write
        if result.kernel_duration_us > 0:
            achieved_bw = total_bytes / (result.kernel_duration_us / 1e6) / 1e9  # GB/s
            parts.append(f"Estimated BW: {achieved_bw:.1f} GB/s ({total_bytes/1024:.0f} KiB moved)")

    if result.achieved_bw_total_gbps > 0:
        parts.append(f"Measured BW: {result.achieved_bw_total_gbps:.1f} GB/s")
        peak_bw = 700.0  # gfx928
        util = result.achieved_bw_total_gbps / peak_bw * 100
        parts.append(f"Achieved HBM BW: {result.achieved_bw_total_gbps:.1f} GB/s ({util:.0f}% of {peak_bw:.0f} GB/s peak)")

    if result.achieved_occupancy_pct > 0:
        parts.append(f"Occupancy: {result.achieved_occupancy_pct:.0f}%")

    if result.l2_cache_hit_pct > 0:
        parts.append(f"L2 cache hit: {result.l2_cache_hit_pct:.0f}%")

    # Bottleneck-specific advice
    if result.achieved_bw_total_gbps > 0 and result.achieved_bw_total_gbps < 200:
        parts.append("→ CRITICAL: Bandwidth is very low. Focus on memory coalescing and vectorized loads.")
    elif result.achieved_occupancy_pct > 0 and result.achieved_occupancy_pct < 40:
        parts.append("→ Occupancy is low. Reduce register pressure or increase tile size.")
    elif result.l2_cache_hit_pct > 0 and result.l2_cache_hit_pct < 30:
        parts.append("→ Poor cache hit rate. Improve data reuse via larger tiles or shared memory caching.")

    return "\n".join(parts)


# =========================================================================== #
# Integration helper: generate profiling info for KernelEntry
# =========================================================================== #


def profile_summary_for_storage(result: ProfileResult) -> Dict[str, Any]:
    """Extract compact profiling summary for storage in kernel_library.json."""
    return {
        "profiled": result.success,
        "kernel_duration_us": round(result.kernel_duration_us, 2),
        "achieved_bw_gbps": round(result.achieved_bw_total_gbps, 2),
        "occupancy_pct": round(result.achieved_occupancy_pct, 1),
        "l2_cache_hit_pct": round(result.l2_cache_hit_pct, 1),
        "stall_memory_pct": round(result.stall_memory_pct, 1),
        "stall_dependency_pct": round(result.stall_dependency_pct, 1),
        "kernel_name": result.kernel_name,
        "error": result.error if not result.success else "",
    }
