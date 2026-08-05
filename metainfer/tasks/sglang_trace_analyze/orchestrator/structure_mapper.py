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
    cpu_ops: list | None = None,
) -> Dict[str, Any]:
    """Map a single kernel to a model layer by inspecting its call stack
    and correlated CPU ops."""
    layer = _infer_layer(call_stack, kernel_name, config, cpu_ops)
    op_type = _infer_op_type(kernel_name, call_stack, cpu_ops)

    confidence = "high"
    if not call_stack:
        # Without call stacks, we use kernel name + CPU op correlation
        has_cpu_hint = bool(cpu_ops)
        if has_cpu_hint and _is_ck_gemm(kernel_name):
            confidence = "medium"  # CK GEMM is unambiguous even without stack
        elif has_cpu_hint:
            confidence = "medium"
        else:
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


def _is_ck_gemm(name: str) -> bool:
    """CK (composable_kernel) GEMM kernels have Cijk_ prefix."""
    return name.lower().startswith("cijk_")


def _infer_layer(
    call_stack: str,
    kernel_name: str,
    config: Dict[str, Any],
    cpu_ops: list | None = None,
) -> Optional[str]:
    """Extract layer information from the call stack and kernel name."""
    name_lower = kernel_name.lower()
    cpu_lower = " ".join(cpu_ops or []).lower()

    if call_stack:
        import re
        lines = call_stack.strip().split("\n")
        layer_pat = re.compile(r"model\.layers\.(\d+)")
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

    # Fallback (no call stack): kernel name + CPU op heuristics
    if "flash_fwd" in name_lower or "flash_attn" in name_lower:
        return "all_layers/attention"
    if "fused_moe" in name_lower or "moe" in cpu_lower:
        return "moe_layers/experts"
    if name_lower.startswith("cijk_"):
        return "all_layers/linear"
    if "rms_norm" in cpu_lower or "rmsnorm" in name_lower:
        return "all_layers/norm"
    if "reduce_kernel" in name_lower:
        return "all_layers/allreduce"
    if "allgather" in cpu_lower or "nccl" in name_lower:
        return "all_layers/communication"
    if "elementwise" in name_lower or "vectorized" in name_lower:
        return "all_layers/elementwise"

    return None


def _infer_op_type(kernel_name: str, call_stack: str, cpu_ops: list | None = None) -> str:
    """Infer the op type from kernel name, call stack, and correlated CPU ops.

    Priority: kernel name patterns > CPU op hints > name substring heuristics.
    """
    name_lower = kernel_name.lower()
    cpu_lower = " ".join(cpu_ops or []).lower()

    # ── Strong kernel name patterns (highest priority) ──

    # CK GEMM kernels (HIP/ROCm composable_kernel)
    if name_lower.startswith("cijk_"):
        return "GEMM"

    # GPU kernel name patterns — unambiguous from the kernel name itself
    if "nccl" in name_lower:
        return "NCCL"
    if any(k in name_lower for k in ("flash_fwd", "flash_attn")):
        return "Attention"
    if "fused_moe" in name_lower:
        return "MoE"

    # ── Kernel name substring heuristics (medium priority) ──
    if "reduce_kernel" in name_lower or "cross_device_reduce" in name_lower:
        return "Reduce"
    if "elementwise" in name_lower:
        return "ElementWise"
    if "vectorized" in name_lower:
        return "ElementWise"
    if "gather" in name_lower:
        return "Indexing"

    # ── CPU op hints for torch-compiled/fused kernels ──
    if "all_reduce" in cpu_lower:
        return "Reduce"  # CustomAllReduce, not NCCL
    if "allgather" in cpu_lower:
        return "NCCL"
    if "rms_norm" in cpu_lower:
        return "Norm"

    # ── Remaining kernel name patterns (lower priority) ──
    if any(k in name_lower for k in ("attn", "attention")):
        return "Attention"
    if any(k in name_lower for k in ("moe",)):
        return "MoE"
    if any(k in name_lower for k in ("gemm", "linear", "matmul", "w8a8", "fp8")):
        return "GEMM"
    if any(k in name_lower for k in ("rmsnorm", "rms_norm", "layernorm")):
        return "Norm"
    if any(k in name_lower for k in ("allreduce", "allgather", "broadcast")):
        return "NCCL"
    if any(k in name_lower for k in ("hadamard", "rotate", "rope")):
        return "Transform"
    if any(k in name_lower for k in ("copy", "memcpy", "memset")):
        return "Memory"
    if any(k in name_lower for k in ("silu", "gelu", "swiglu", "activation", "act_and_mul")):
        return "Activation"
    if any(k in name_lower for k in ("topk", "top_k", "gather", "scatter", "sort")):
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
        "Reduce": "Reduce",
        "ElementWise": "ElementWise",
    }
    return mapping.get(op_type, "Other")
