"""Rule-based fuse pattern matcher.

Scans the kernel table (ordered by GPU time or timeline order) for known
sequences that indicate a missing fusion opportunity, and reports each
match with a description and estimated saving.

The catalog is hard-coded — each pattern has a name, the kernel names
that must appear consecutively (or within a short window), and a
suggestion for what the fused replacement would be.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ------------------------------------------------------------------ #
#  Fuse pattern catalog
# ------------------------------------------------------------------ #

FUSE_PATTERNS: List[Dict[str, Any]] = [
    {
        "pattern": "rms_norm + gemm",
        "kernels": ["rms_norm", "gemm"],
        "match_mode": "consecutive",
        "suggestion": "Replace separate rms_norm + gemm with fused_rms_norm_gemm (e.g. triton kernel or sglang fused op).",
        "estimated_saving_us": 180,
        "confidence": "high",
    },
    {
        "pattern": "silu + mul + gemm",
        "kernels": ["silu", "mul", "gemm"],
        "match_mode": "consecutive",
        "suggestion": "Fuse into silu_and_mul + gemm, or a single fused MoE activation+gemm kernel.",
        "estimated_saving_us": 250,
        "confidence": "high",
    },
    {
        "pattern": "add + rms_norm",
        "kernels": ["add", "rms_norm"],
        "match_mode": "consecutive",
        "suggestion": "Fuse residual add + rms_norm into a single kernel to avoid a separate memory round-trip.",
        "estimated_saving_us": 120,
        "confidence": "medium",
    },
    {
        "pattern": "quant + gemm",
        "kernels": ["quant", "gemm"],
        "match_mode": "consecutive",
        "suggestion": "Integrate FP8 quantization into the GEMM launch to eliminate a precursor kernel.",
        "estimated_saving_us": 200,
        "confidence": "medium",
    },
    {
        "pattern": "nccl_allreduce + gemm (no overlap)",
        "kernels": ["ncclAllReduce", "gemm"],
        "match_mode": "consecutive",
        "suggestion": (
            "AllReduce and gemm are serialized. Try overlapping: issue AllReduce "
            "on a separate CUDA stream, or restructure to compute on one output "
            "shard while communicating another."
        ),
        "estimated_saving_us": 300,
        "confidence": "medium",
    },
]


def match_fuse_patterns(
    kernels: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Scan a kernel list for known fuse patterns.

    Args:
        kernels: List of kernel entries. Must contain ``kernel_name`` and
            preferably be in timeline order. If only duration-ordered, set
            ``match_mode`` to ``"unordered"`` for pattern matching.

    Returns:
        List of matched patterns, each with ``pattern``, ``kernels``,
        ``suggestion``, ``estimated_saving_us``, ``confidence``.
    """
    kernel_names = [k.get("kernel_name", "") for k in kernels]
    matches = []

    for pat in FUSE_PATTERNS:
        found = _match_consecutive(kernel_names, pat["kernels"])
        if found:
            matches.append({
                "pattern": pat["pattern"],
                "kernels": found,
                "suggestion": pat["suggestion"],
                "estimated_saving_us": pat["estimated_saving_us"],
                "confidence": pat["confidence"],
            })

    return matches


def build_fuse_report(
    kernels: List[Dict[str, Any]],
    batch_size: int,
    stage: str,
) -> Dict[str, Any]:
    """Produce the full fuse.json payload."""
    matches = match_fuse_patterns(kernels)
    return {
        "batch_size": batch_size,
        "stage": stage,
        "matches": matches,
    }


def _match_consecutive(
    names: List[str],
    pattern_kernels: List[str],
) -> List[str]:
    """Check if ``pattern_kernels`` appear consecutively (in order) within
    ``names``.

    Returns the matched kernel names if found, empty list otherwise.
    """
    if len(pattern_kernels) > len(names):
        return []

    patterns_lower = [p.lower() for p in pattern_kernels]
    names_lower = [n.lower() for n in names]

    for i in range(len(names_lower) - len(patterns_lower) + 1):
        match = True
        for j, pat in enumerate(patterns_lower):
            if pat not in names_lower[i + j]:
                match = False
                break
        if match:
            return names[i: i + len(patterns_lower)]
    return []
