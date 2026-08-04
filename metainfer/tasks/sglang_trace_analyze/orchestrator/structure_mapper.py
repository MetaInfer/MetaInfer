"""Map kernel names (via call stacks) to model structural elements.

Takes the aggregated kernel list from :mod:`trace_parser` and the model's
``config.json``, then assigns each kernel to:
- ``model_layer`` — e.g. ``layer_{2..58}/attn/qkv_proj``
- ``op_type`` — GEMM / Attention / Norm / ElementWise / MoE / NCCL / ...
- ``category`` — for grouping (MLA, MoE, GEMM, NCCL, etc.)

Mapping is done by parsing the Python source location from the call stack
and matching it against known sglang layer source patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_mapping(
    kernels: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build the kernel-to-model-structure mapping.

    Args:
        kernels: Aggregated kernel list from ``aggregate_kernels()`` with
            ``include_call_stack=True``.
        config: Model ``config.json`` as a dict.

    Returns:
        List of mapping entries with ``kernel_name``, ``model_layer``,
        ``op_type``, ``category``, ``call_stack``, ``confidence``.
    """
    entries = []
    for k in kernels:
        call_stack = k.get("call_stack", "")
        entry = _map_one(k["kernel_name"], call_stack, config)
        entries.append(entry)
    return entries


# ------------------------------------------------------------------ #
#  Internal: pattern-based mapping
# ------------------------------------------------------------------ #

def _map_one(
    kernel_name: str,
    call_stack: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Map a single kernel to a model layer by inspecting its call stack."""
    layer = _infer_layer(call_stack, kernel_name, config)
    op_type = _infer_op_type(kernel_name, call_stack)

    confidence = "high"
    if not call_stack:
        confidence = "low"
    elif layer is None:
        confidence = "medium"

    return {
        "kernel_name": kernel_name,
        "model_layer": layer,
        "op_type": op_type,
        "category": _op_type_to_category(op_type),
        "call_stack": call_stack,
        "confidence": confidence,
    }


def _infer_layer(
    call_stack: str,
    kernel_name: str,
    config: Dict[str, Any],
) -> Optional[str]:
    """Extract layer information from the call stack.

    Looks for patterns like:
    - ``sglang/srt/layers/...``
    - ``layer_forward``
    - ``model.py``, ``decoder.py``, ``encoder.py``
    - Module names like ``model.layers.5.self_attn``

    Returns ``None`` if no layer info can be inferred.
    """
    if not call_stack:
        return None

    # Heuristic: look for sglang/srt/layers or model.layers.N patterns
    lines = call_stack.strip().split("\n")

    # Pattern 1: model.layers.N in the call stack
    import re
    layer_pat = re.compile(r"model\.layers\.(\d+)")
    # Pattern 2: sglang source files under layers/
    sglang_layer_pat = re.compile(
        r"sglang/srt/layers/(attn|moe|mla|linear|norm|embed|sampler|router)"
    )

    for line in lines:
        m = layer_pat.search(line)
        if m:
            return f"layer_{m.group(1)}"
        m = sglang_layer_pat.search(line)
        if m:
            return f"layers/{m.group(1)}"

    # Fallback: use kernel name heuristics
    if "attn" in kernel_name.lower() or "attention" in kernel_name.lower():
        return "attention (unknown layer)"
    if "moe" in kernel_name.lower():
        return "moe (unknown layer)"
    if "gemm" in kernel_name.lower() or "linear" in kernel_name.lower():
        return "linear (unknown layer)"

    return None


def _infer_op_type(kernel_name: str, call_stack: str) -> str:
    """Infer the op type from kernel name and call stack."""
    name_lower = kernel_name.lower()
    if any(k in name_lower for k in ("attn", "attention", "flash_fwd", "flash_attn")):
        return "Attention"
    if any(k in name_lower for k in ("moe", "fused_moe")):
        return "MoE"
    if any(k in name_lower for k in ("gemm", "linear", "matmul", "w8a8", "fp8")):
        return "GEMM"
    if any(k in name_lower for k in ("rms", "norm", "layernorm", "layer_norm")):
        return "Norm"
    if any(k in name_lower for k in ("nccl", "allreduce", "allgather", "broadcast")):
        return "NCCL"
    if any(k in name_lower for k in ("hadamard", "rotate", "rope")):
        return "Transform"
    if any(k in name_lower for k in ("copy", "memcpy", "memset")):
        return "Memory"
    if any(k in name_lower for k in ("silu", "gelu", "swiglu", "activation", "act_and_mul")):
        return "Activation"
    if any(k in name_lower for k in ("topk", "top_k", "index", "gather", "scatter", "sort")):
        return "Indexing"
    if any(k in name_lower for k in ("quant", "dequant", "fp8_scale")):
        return "Quantization"
    return "Other"


def _op_type_to_category(op_type: str) -> str:
    """Map an op_type to a display category."""
    mapping = {
        "Attention": "Attention",
        "MoE": "MoE",
        "GEMM": "GEMM",
        "Norm": "Norm",
        "NCCL": "NCCL",
        "Transform": "Transform",
        "Memory": "Memory",
        "Activation": "Activation",
        "Indexing": "Indexing",
        "Quantization": "Quantization",
    }
    return mapping.get(op_type, "Other")
