"""Stable Python contract for TP=4/8 INT8 W8A8 GEMM.

This file is the boundary between benchmarks/framework code and an optimizable
backend.  Kernel authors may change packing, dispatch, HIP/DUMMA kernels, and
workspace use, but should not change the public function signatures or tensor
semantics in this file.

Logical operation:
    out[m, n] = bf16(
        int32_dot(x_q[m, :], raw_weight[:, n])
        * x_scale[m, 0]
        * weight_scale[n, 0]
    )

Only ``w8a8_gemm_out`` belongs inside the timed/CUDA-Graph region.
``prepare_weight`` and ``allocate_workspace`` must run before graph capture and
are excluded from GEMM timing.

Required backend operator schema:
    zth_w8a8::gemm_out(
        Tensor x_q,
        Tensor packed_weight,
        Tensor x_scale,
        Tensor packed_weight_scale,
        Tensor(a!) out,
        Tensor(b!) workspace
    ) -> Tensor(a!)

Optional backend packing schema:
    zth_w8a8::pack_weight(
        Tensor raw_weight,
        Tensor weight_scale
    ) -> (Tensor, Tensor)
"""

from __future__ import annotations

from typing import Final, Mapping

try:
    import torch
except ModuleNotFoundError:  # Metadata validation runs in CPU-only CI.
    torch = None  # type: ignore[assignment]


def _torch_no_grad():
    """Return PyTorch's decorator, or an import-only CI fallback."""
    if torch is None:
        return lambda function: function
    return torch.no_grad()


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "the INT8 W8A8 runtime API requires PyTorch in the DCU environment"
        )


# These are logical (K, N) dimensions after TP partitioning, not checkpoint
# storage shapes. Operator identity is retained even when two operators share
# the same numerical shape, so TP=4 and TP=8 task metadata cannot be confused.
TP4_OPERATOR_KN: Final[Mapping[str, tuple[int, int]]] = {
    "wqkv_a": (4096, 1536),
    "wq_b": (1024, 8192),
    "indexer.wq_b": (1024, 8192),
    "wo_b": (2048, 4096),
    "shared_gate_up_proj": (4096, 1024),
    "shared_down_proj": (512, 4096),
}
TP8_OPERATOR_KN: Final[Mapping[str, tuple[int, int]]] = {
    "wqkv_a": (4096, 1536),
    "wq_b": (1024, 4096),
    "indexer.wq_b": (1024, 8192),
    "wo_b": (1024, 4096),
    "shared_gate_up_proj": (4096, 512),
    "shared_down_proj": (256, 4096),
}
TP_OPERATOR_KN: Final[Mapping[int, Mapping[str, tuple[int, int]]]] = {
    4: TP4_OPERATOR_KN,
    8: TP8_OPERATOR_KN,
}
TP4_W8A8_KN: Final[frozenset[tuple[int, int]]] = frozenset(
    TP4_OPERATOR_KN.values()
)
TP8_W8A8_KN: Final[frozenset[tuple[int, int]]] = frozenset(
    TP8_OPERATOR_KN.values()
)
SUPPORTED_W8A8_KN: Final[frozenset[tuple[int, int]]] = (
    TP4_W8A8_KN | TP8_W8A8_KN
)
# Compatibility alias for repositories generated with the older API.
TP4_DECODE_KN: Final[frozenset[tuple[int, int]]] = TP4_W8A8_KN

DEFAULT_OPTIMIZATION_M_VALUES: Final[tuple[int, ...]] = (2, 16, 3072)
# TP4-only large-prefill boundary added on 2026-08-06. The TP8 workload keeps
# the original three M values, so each TP4 operator gets exactly one M=4096
# shape while TP8 defaults are unchanged.
TP4_EXTRA_OPTIMIZATION_M_VALUES: Final[tuple[int, ...]] = (4096,)


def _default_optimization_shapes() -> tuple[dict[str, int | str], ...]:
    """Return logical TP/operator shapes with collision-free task IDs.

    Identical numerical GEMMs may appear more than once because the task ID and
    metadata deliberately preserve their model call site. Each shape still
    produces an independent accepted artifact, so no runtime dispatch conflict
    is introduced.
    """
    shapes: list[dict[str, int | str]] = []
    for tp_size, operators in TP_OPERATOR_KN.items():
        m_values = (
            DEFAULT_OPTIMIZATION_M_VALUES
            + TP4_EXTRA_OPTIMIZATION_M_VALUES
            if tp_size == 4
            else DEFAULT_OPTIMIZATION_M_VALUES
        )
        for operator, (k, n) in operators.items():
            operator_id = operator.replace(".", "_")
            for m in m_values:
                shapes.append(
                    {
                        "id": f"tp{tp_size}_{operator_id}_m{m}",
                        "tp_size": tp_size,
                        "operator": operator,
                        "M": m,
                        "N": n,
                        "K": k,
                    }
                )
    return tuple(shapes)


# Default workload used when MetaInfer New Task leaves shapes empty. It covers
# decode, short-token DUMMA, and the requested large-prefill boundary.
DEFAULT_OPTIMIZATION_SHAPES: Final[tuple[dict[str, int | str], ...]] = (
    _default_optimization_shapes()
)

MIN_M: Final[int] = 1
MAX_M: Final[int] = 4096
WORKSPACE_BUDGET_BYTES: Final[int] = 16 * 1024 * 1024
WORKSPACE_MAX_SPLIT_K: Final[int] = 16
# Backward-compatible name used by existing generated repositories. Capacity
# is now calculated from the byte budget for each shape rather than fixed at 8.
WORKSPACE_SPLIT_K_CAP: Final[int] = WORKSPACE_MAX_SPLIT_K
WORKSPACE_ALIGNMENT: Final[int] = 256
INT32_BYTES: Final[int] = 4


def _check_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    *,
    contiguous: bool = True,
) -> None:
    _require_torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on a CUDA/HIP device")
    if tensor.dtype != dtype:
        raise TypeError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_same_device(reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(
                f"{name}.device must be {reference.device}, got {tensor.device}"
            )


def _check_target_shape(m: int, n: int, k: int) -> None:
    if not MIN_M <= m <= MAX_M:
        raise ValueError(f"M must be in [{MIN_M}, {MAX_M}], got {m}")
    if (k, n) not in SUPPORTED_W8A8_KN:
        raise ValueError(
            f"unsupported TP4/TP8 logical shape (K, N)=({k}, {n}); "
            f"supported={sorted(SUPPORTED_W8A8_KN)}"
        )
    if k % 32 != 0:
        raise ValueError(f"K must be divisible by the INT8 DUMMA K tile 32, got {k}")
    if n % 16 != 0:
        raise ValueError(f"N must be divisible by the DUMMA N tile 16, got {n}")


def validate_optimization_shape(shape: Mapping[str, object]) -> None:
    """Validate one topology-qualified MetaInfer optimization shape."""
    try:
        tp_size = int(shape["tp_size"])
        operator = str(shape["operator"])
        m = int(shape["M"])
        n = int(shape["N"])
        k = int(shape["K"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "optimization shape requires tp_size, operator, M, N and K"
        ) from exc
    operators = TP_OPERATOR_KN.get(tp_size)
    if operators is None:
        raise ValueError(f"tp_size must be 4 or 8, got {tp_size}")
    expected = operators.get(operator)
    if expected is None:
        raise ValueError(
            f"unsupported TP={tp_size} operator {operator!r}; "
            f"supported={sorted(operators)}"
        )
    if (k, n) != expected:
        raise ValueError(
            f"TP={tp_size} {operator} requires (K, N)={expected}, "
            f"got ({k}, {n})"
        )
    _check_target_shape(m, n, k)


def _optional_op(namespace: str, name: str):
    _require_torch()
    try:
        return getattr(getattr(torch.ops, namespace), name)
    except AttributeError:
        return None


def _required_op(namespace: str, name: str):
    op = _optional_op(namespace, name)
    if op is None:
        raise RuntimeError(
            f"required custom operator torch.ops.{namespace}.{name} is not "
            "registered; load/build the W8A8 HIP extension before calling it"
        )
    return op


@_torch_no_grad()
def prepare_weight(
    raw_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare a logical [K, N] INT8 weight outside the timed region.

    The optional backend op may return any contiguous opaque packed layout.
    Until a packing op is registered, the identity contiguous layout is used.
    The returned tensors must remain alive and at stable addresses throughout
    CUDA/HIP Graph capture and replay.
    """
    _check_cuda_tensor("raw_weight", raw_weight, torch.int8)
    _check_cuda_tensor("weight_scale", weight_scale, torch.float32)
    if raw_weight.ndim != 2:
        raise ValueError(
            f"raw_weight must have logical shape [K, N], got {raw_weight.shape}"
        )
    k, n = raw_weight.shape
    _check_target_shape(MAX_M, n, k)
    if weight_scale.shape != (n, 1):
        raise ValueError(
            f"weight_scale must have shape ({n}, 1), got {weight_scale.shape}"
        )
    _check_same_device(raw_weight, weight_scale=weight_scale)

    pack_op = _optional_op("zth_w8a8", "pack_weight")
    if pack_op is None:
        return raw_weight.contiguous(), weight_scale.contiguous()

    packed_weight, packed_scale = pack_op(raw_weight, weight_scale)
    _check_cuda_tensor("packed_weight", packed_weight, torch.int8)
    _check_cuda_tensor("packed_weight_scale", packed_scale, torch.float32)
    _check_same_device(
        raw_weight,
        packed_weight=packed_weight,
        packed_weight_scale=packed_scale,
    )
    if packed_weight.numel() < k * n:
        raise ValueError(
            "packed_weight must contain at least K*N int8 elements; "
            f"got {packed_weight.numel()} for K*N={k*n}"
        )
    if packed_scale.numel() < n:
        raise ValueError(
            "packed_weight_scale must contain at least N fp32 elements; "
            f"got {packed_scale.numel()} for N={n}"
        )
    return packed_weight, packed_scale


def allocate_workspace(
    m: int,
    n: int,
    k: int,
    device: torch.device | str | int,
) -> torch.Tensor:
    """Allocate an opaque, reusable workspace before graph capture.

    Capacity is shape-aware and capped by a fixed byte budget. This permits
    non-power-of-two split-K choices while keeping allocation bounded.
    """
    _require_torch()
    _check_target_shape(m, n, k)
    required_bytes = max(
        WORKSPACE_ALIGNMENT,
        workspace_split_k_capacity(m, n, k) * m * n * INT32_BYTES,
    )
    aligned_bytes = (
        (required_bytes + WORKSPACE_ALIGNMENT - 1)
        // WORKSPACE_ALIGNMENT
        * WORKSPACE_ALIGNMENT
    )
    return torch.empty(aligned_bytes, dtype=torch.uint8, device=device)


def workspace_split_k_capacity(m: int, n: int, k: int) -> int:
    """Return legal split-K partial planes, or zero when none fit the budget."""
    _check_target_shape(m, n, k)
    bytes_per_partial = m * n * INT32_BYTES
    budget_capacity = WORKSPACE_BUDGET_BYTES // bytes_per_partial
    stage_capacity = k // 32
    return min(WORKSPACE_MAX_SPLIT_K, budget_capacity, stage_capacity)


def validate_gemm_out_inputs(
    x_q: torch.Tensor,
    packed_weight: torch.Tensor,
    x_scale: torch.Tensor,
    packed_weight_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the frozen logical contract and return (M, N, K)."""
    _check_cuda_tensor("x_q", x_q, torch.int8)
    _check_cuda_tensor("packed_weight", packed_weight, torch.int8)
    _check_cuda_tensor("x_scale", x_scale, torch.float32)
    _check_cuda_tensor("packed_weight_scale", packed_weight_scale, torch.float32)
    _check_cuda_tensor("out", out, torch.bfloat16)
    _check_cuda_tensor("workspace", workspace, torch.uint8)

    if x_q.ndim != 2:
        raise ValueError(f"x_q must have shape [M, K], got {x_q.shape}")
    m, k = x_q.shape
    if out.ndim != 2:
        raise ValueError(f"out must have shape [M, N], got {out.shape}")
    if out.shape[0] != m:
        raise ValueError(f"out.shape[0] must equal M={m}, got {out.shape[0]}")
    n = out.shape[1]
    _check_target_shape(m, n, k)

    if x_scale.shape != (m, 1):
        raise ValueError(f"x_scale must have shape ({m}, 1), got {x_scale.shape}")
    if packed_weight.numel() < k * n:
        raise ValueError(
            "packed_weight must contain at least K*N int8 elements; "
            f"got {packed_weight.numel()} for K*N={k*n}"
        )
    if packed_weight_scale.numel() < n:
        raise ValueError(
            "packed_weight_scale must contain at least N fp32 elements; "
            f"got {packed_weight_scale.numel()} for N={n}"
        )
    minimum_workspace_bytes = (
        workspace_split_k_capacity(m, n, k) * m * n * INT32_BYTES
    )
    if workspace.numel() < minimum_workspace_bytes:
        raise ValueError(
            f"workspace requires at least {minimum_workspace_bytes} uint8 "
            f"elements, got {workspace.numel()}"
        )
    _check_same_device(
        x_q,
        packed_weight=packed_weight,
        x_scale=x_scale,
        packed_weight_scale=packed_weight_scale,
        out=out,
        workspace=workspace,
    )
    return m, n, k


@_torch_no_grad()
def w8a8_gemm_out(
    x_q: torch.Tensor,
    packed_weight: torch.Tensor,
    x_scale: torch.Tensor,
    packed_weight_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Run the graph-capturable W8A8 GEMM into preallocated ``out``.

    Backend requirements:
      * launch on PyTorch's current stream;
      * perform no allocation, compilation, autotuning, packing, or host sync;
      * include every required split-K/combine/epilogue kernel in this call;
      * return the exact same tensor/storage as ``out``.
    """
    validate_gemm_out_inputs(
        x_q,
        packed_weight,
        x_scale,
        packed_weight_scale,
        out,
        workspace,
    )
    gemm_op = _required_op("zth_w8a8", "gemm_out")
    result = gemm_op(
        x_q,
        packed_weight,
        x_scale,
        packed_weight_scale,
        out,
        workspace,
    )
    if not isinstance(result, torch.Tensor):
        raise TypeError("torch.ops.zth_w8a8.gemm_out must return a Tensor")
    if result.data_ptr() != out.data_ptr():
        raise RuntimeError("gemm_out must return the same storage as out")
    return result


__all__ = [
    "DEFAULT_OPTIMIZATION_M_VALUES",
    "DEFAULT_OPTIMIZATION_SHAPES",
    "MAX_M",
    "MIN_M",
    "TP4_EXTRA_OPTIMIZATION_M_VALUES",
    "TP4_DECODE_KN",
    "TP4_OPERATOR_KN",
    "TP4_W8A8_KN",
    "TP8_OPERATOR_KN",
    "TP8_W8A8_KN",
    "TP_OPERATOR_KN",
    "SUPPORTED_W8A8_KN",
    "WORKSPACE_BUDGET_BYTES",
    "WORKSPACE_MAX_SPLIT_K",
    "WORKSPACE_SPLIT_K_CAP",
    "allocate_workspace",
    "prepare_weight",
    "validate_optimization_shape",
    "validate_gemm_out_inputs",
    "w8a8_gemm_out",
    "workspace_split_k_capacity",
]
