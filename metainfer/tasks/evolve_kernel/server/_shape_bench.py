"""On-demand shape benchmark for the evolve-kernel WebUI.

Runs the best kernel from the library against the reference kernel on a set
of target shapes extracted from the task's extra_notes. Results are cached
in workspace/shape_bench.json and refreshed when the best kernel changes.

Parsed shape formats:
  - Table rows:  wq_b: (M, 1024) @ (1024, 4096)  # TP=8
  - Multi-line:  gate_up_proj: (M, 4096) @ (4096, 1024) → M=1,4,16,4096
  - Inline:      shapes=(M,1024,4096) (M,4096,1024)
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # torch optional so CPU-only CI can import/collect; runtime requires it
    import torch
except ImportError:  # pragma: no cover - CPU-only environment
    torch = None  # type: ignore[assignment]


# =========================================================================== #
# Shape spec
# =========================================================================== #


@dataclass
class ShapeSpec:
    label: str           # e.g. "gate_up_proj (TP=4)"
    M_values: List[int]  # e.g. [1, 2, 4, 8, 16, 4096]
    K: int
    N: int


# Default M values to benchmark per shape
_DEFAULT_M_VALUES = [1, 2, 4, 8, 16, 4096]

# Commonly-used shapes from DeepSeek V4 Flash TP=4/TP=8
_PRESET_SHAPES: List[ShapeSpec] = [
    ShapeSpec("wq_b (TP=4)", [1, 2, 4, 8, 16, 4096], K=1024, N=8192),
    ShapeSpec("wq_b (TP=8)", [1, 2, 4, 8, 16, 4096], K=1024, N=4096),
    ShapeSpec("wo_b (TP=4)", [1, 2, 4, 8, 16, 4096], K=2048, N=4096),
    ShapeSpec("wo_b (TP=8)", [1, 2, 4, 8, 16, 4096], K=1024, N=4096),
    ShapeSpec("gate_up_proj (TP=4)", [1, 2, 4, 8, 16, 4096], K=4096, N=1024),
    ShapeSpec("gate_up_proj (TP=8)", [1, 2, 4, 8, 16, 4096], K=4096, N=512),
    ShapeSpec("down_proj (TP=4)", [1, 2, 4, 8, 16, 4096], K=512, N=4096),
    ShapeSpec("down_proj (TP=8)", [1, 2, 4, 8, 16, 4096], K=256, N=4096),
]


# =========================================================================== #
# Shape parsing
# =========================================================================== #


def parse_shapes_from_extra_notes(extra_notes: str) -> List[ShapeSpec]:
    """Parse target shapes from extra_notes or other requirements text.

    Recognized formats:

    1. Named rows (table format):
       wq_b: (M, 1024) @ (1024, 8192)
       gate_up_proj: (M, 4096) @ (4096, 1024)

    2. Explicit M values on same line:
       gate_up_proj: M=1,4,8,16,4096  (M, 4096) @ (4096, 1024)

    3. Compact format:
       (M,1024,4096)  →  K=1024, N=4096

    If no shapes are found, returns the PRESET_SHAPES.
    """
    if not extra_notes:
        return list(_PRESET_SHAPES)

    shapes: List[ShapeSpec] = []

    # Pattern 1: named shapes with (M, K) @ (K, N) notation
    named_pattern = re.compile(
        r'([\w_]+(?:\s*\([^)]*\))?)\s*[:：]\s*'
        r'\(?\s*M\s*,\s*(\d+)\s*\)\s*@\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',
        re.IGNORECASE,
    )
    for match in named_pattern.finditer(extra_notes):
        label = match.group(1).strip()
        k_val = int(match.group(2))
        n_val = int(match.group(4))  # group 3 is K from @(K, N), group 4 is N

        # Make label descriptive if it's generic
        if label.upper() in ("SHAPE", "TARGET SHAPE", "TARGET", "SHAPES"):
            label = f"({k_val}×{n_val})"

        # Collect M values from nearby context (up to 500 chars)
        nearby = extra_notes[match.start():match.start() + 500]
        m_vals_set = set()
        # Match patterns: M=1,2,4,8  M=4096  M <=16  M=1,4,8,16,4096
        for m_match in re.finditer(
            r'(?:M|m)\s*[=：:<=]+\s*([\d,\s]+)',
            nearby,
        ):
            for num_str in m_match.group(1).split(","):
                num_str = num_str.strip()
                if num_str.isdigit():
                    m_vals_set.add(int(num_str))

        if m_vals_set:
            m_vals = sorted(m_vals_set)
        else:
            # Fallback: look for "M <=N" patterns in full text
            m_range = re.search(
                r'(?:M|m)\s*<=\s*(\d+)',
                extra_notes,
            )
            if m_range:
                max_m = int(m_range.group(1))
                m_vals = [1, 2, 4, 8, 16, max_m]
            else:
                m_vals = list(_DEFAULT_M_VALUES)

        shapes.append(ShapeSpec(label=label, M_values=m_vals, K=k_val, N=n_val))

    # Pattern 2: compact (M, K, N) format — pick the first one that looks significant
    if not shapes:
        compact = re.compile(r'\(\s*(?:M|m)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')
        for match in compact.finditer(extra_notes):
            k_val = int(match.group(1))
            n_val = int(match.group(2))
            shapes.append(ShapeSpec(
                label=f"({k_val}×{n_val})",
                M_values=[1, 2, 4, 8, 16, 4096],
                K=k_val, N=n_val,
            ))

    if not shapes:
        return list(_PRESET_SHAPES)

    return shapes


# =========================================================================== #
# Benchmark runner
# =========================================================================== #


@dataclass
class ShapeResult:
    shape_label: str
    M: int
    N: int
    K: int
    ref_ms: float = 0.0
    best_ms: float = 0.0
    speedup: float = 1.0
    error: str = ""


def _load_kernel_fn(filepath: str, fn_name: str):
    spec = importlib.util.spec_from_file_location("_shape_bench_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, fn_name):
        raise ValueError(f"Function {fn_name!r} not found in {filepath}")
    return getattr(mod, fn_name)


def _measure_one(kernel_fn, M: int, N: int, K: int,
                 warmup: int = 5, repeat: int = 20) -> float:
    """Measure median execution time in ms for one (M,N,K) shape."""
    device = torch.device("cuda")

    a = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
    a_scale = torch.randn(M, 1, device=device, dtype=torch.float32).abs() / 127.0
    b = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
    b_scale = torch.randn(N, 1, device=device, dtype=torch.float32).abs() / 127.0

    # Warmup
    for _ in range(warmup):
        kernel_fn(a, a_scale, b, b_scale, torch.bfloat16)
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(repeat):
        start.record()
        kernel_fn(a, a_scale, b, b_scale, torch.bfloat16)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.median(times_ms)


def _measure_one_safe(kernel_fn, M: int, N: int, K: int,
                      warmup: int = 2, repeat: int = 5) -> float:
    """Measure with adaptive repetitions; catches OOM and crashes.

    Uses more repetitions for small M (fast) and fewer for large M (slow).
    """
    try:
        # Adaptive: more repeats for fast kernels (M <= 16), fewer for slow ones
        actual_repeat = repeat
        if M <= 16:
            actual_repeat = 10  # small M is sub-ms, so more repeats for accuracy
            actual_warmup = max(warmup, 3)
        else:
            actual_repeat = 3   # large M is slower, fewer repeats
            actual_warmup = max(warmup, 1)

        device = torch.device("cuda")
        a = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
        a_scale = torch.randn(M, 1, device=device, dtype=torch.float32).abs() / 127.0
        b = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
        b_scale = torch.randn(N, 1, device=device, dtype=torch.float32).abs() / 127.0

        for _ in range(actual_warmup):
            kernel_fn(a, a_scale, b, b_scale, torch.bfloat16)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(actual_repeat):
            start.record()
            kernel_fn(a, a_scale, b, b_scale, torch.bfloat16)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        return statistics.median(times_ms)
    except Exception:
        return 0.0


def run_shape_benchmark(
    ref_kernel_path: str,
    best_kernel_path: str,
    kernel_fn_name: str,
    shapes: List[ShapeSpec],
) -> List[ShapeResult]:
    """Run benchmarks across all shapes for both reference and best kernel.

    Returns one ShapeResult per (shape, M_value) combination.
    """
    try:
        ref_fn = _load_kernel_fn(ref_kernel_path, kernel_fn_name)
    except Exception as e:
        return [ShapeResult(shape_label="error", M=0, N=0, K=0, error=f"Failed to load ref kernel: {e}")]

    try:
        best_fn = _load_kernel_fn(best_kernel_path, kernel_fn_name)
    except Exception as e:
        return [ShapeResult(shape_label="error", M=0, N=0, K=0, error=f"Failed to load best kernel: {e}")]

    results: List[ShapeResult] = []

    for spec in shapes:
        for m in spec.M_values:
            # Skip shapes that are too large for this M
            if m * spec.N > 2 ** 25:  # > 32M elements output
                continue

            ref_ms = _measure_one_safe(ref_fn, m, spec.N, spec.K)
            best_ms = _measure_one_safe(best_fn, m, spec.N, spec.K)

            if ref_ms <= 0 or best_ms <= 0:
                results.append(ShapeResult(
                    shape_label=spec.label, M=m, N=spec.N, K=spec.K,
                    ref_ms=ref_ms, best_ms=best_ms,
                    error="Measurement failed (OOM or kernel crash)",
                ))
                continue

            speedup = ref_ms / max(best_ms, 1e-6)
            results.append(ShapeResult(
                shape_label=spec.label, M=m, N=spec.N, K=spec.K,
                ref_ms=ref_ms, best_ms=best_ms, speedup=speedup,
            ))

    return results


def shape_results_to_dict(results: List[ShapeResult]) -> List[Dict[str, Any]]:
    return [
        {
            "shape_label": r.shape_label,
            "M": r.M,
            "N": r.N,
            "K": r.K,
            "ref_ms": round(r.ref_ms, 4),
            "best_ms": round(r.best_ms, 4),
            "speedup": round(r.speedup, 3),
            "error": r.error,
        }
        for r in results
    ]


# =========================================================================== #
# Cache management
# =========================================================================== #


def load_cached_benchmark(cache_path: Path, best_kernel_id: str) -> Optional[List[Dict[str, Any]]]:
    """Load cached benchmark results if they match the current best kernel."""
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("best_kernel_id") == best_kernel_id:
            return data.get("results", [])
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def save_cached_benchmark(cache_path: Path, best_kernel_id: str,
                          results: List[Dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "best_kernel_id": best_kernel_id,
        "results": results,
        "timestamp": time.time(),
    }, indent=2), encoding="utf-8")
