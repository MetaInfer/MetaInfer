"""Resolve and stage user-owned, immutable operator API contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


W8A8_API_FILENAME = "int8_w8a8_gemm_api.py"
W8A8_BACKEND_FILENAME = "w8a8_backend.py"
W8A8_VARIANTS_FILENAME = "w8a8_gemm_variants.hip"
W8A8_VARIANTS_RELATIVE = Path("references") / W8A8_VARIANTS_FILENAME


@dataclass(frozen=True)
class OperatorAPIContract:
    operator: str
    dtype: str
    source: Path
    destination_name: str
    reference_sources: tuple[Path, ...] = ()


_CONTRACT_PATHS = {
    ("Quantized GEMM", "INT8 W8A8"): (
        Path("int8w8a8gemm") / W8A8_API_FILENAME
    ),
}


def _default_api_root() -> Path:
    override = os.environ.get("METAINFER_OPERATOR_API_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # Plugin-local integrated API files:
    # .../metainfer/tasks/dcu_kernel_auto_opt/api
    return Path(__file__).resolve().parents[1] / "api"


def _load_module(path: Path) -> ModuleType:
    name = "metainfer_operator_api_" + hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import operator API contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_operator_api(
    operator: str,
    dtype: str,
) -> OperatorAPIContract:
    relative = _CONTRACT_PATHS.get((operator, dtype))
    if relative is None:
        supported = ", ".join(
            f"{op} / {dt}" for op, dt in sorted(_CONTRACT_PATHS)
        )
        raise ValueError(
            f"no operator API contract registered for {operator} / {dtype}; "
            f"supported: {supported}"
        )
    source = (_default_api_root() / relative).resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"operator API contract not found: {source}; place the manually "
            "maintained interface under "
            "metainfer/tasks/dcu_kernel_auto_opt/api or set "
            "METAINFER_OPERATOR_API_ROOT"
        )
    module = _load_module(source)
    required = (
        "prepare_weight",
        "allocate_workspace",
        "validate_gemm_out_inputs",
        "w8a8_gemm_out",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(
            f"operator API contract {source} is missing callables: {missing}"
        )
    if not callable(getattr(module, "_check_target_shape", None)):
        raise ValueError(
            f"operator API contract {source} must expose a shape validator "
            "named _check_target_shape"
        )
    # Reference variant HIP code lives in the plugin's variant/ directory;
    # fall back to the legacy location next to the API file.
    variant_root = Path(__file__).resolve().parents[1] / "variant"
    variant_sources = tuple(
        path for path in (variant_root / W8A8_VARIANTS_FILENAME,)
        if path.is_file()
    ) or tuple(
        path for path in (source.parent / W8A8_VARIANTS_FILENAME,)
        if path.is_file()
    )
    return OperatorAPIContract(
        operator=operator,
        dtype=dtype,
        source=source,
        destination_name=W8A8_API_FILENAME,
        reference_sources=variant_sources,
    )


def validate_contract_shapes(
    contract: OperatorAPIContract,
    shapes: Mapping[str, Any],
) -> None:
    module = _load_module(contract.source)
    validate_shape = getattr(module, "_check_target_shape")
    validate_logical_shape = getattr(
        module, "validate_optimization_shape", None
    )
    for shape_id, shape in shapes.items():
        params = getattr(shape, "params", shape)
        try:
            m = int(params["M"])
            n = int(params["N"])
            k = int(params["K"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{shape_id}: operator API requires integer M, N and K"
            ) from exc
        try:
            if callable(validate_logical_shape) and (
                "tp_size" in params or "operator" in params
            ):
                validate_logical_shape(params)
            else:
                validate_shape(m, n, k)
        except Exception as exc:
            raise ValueError(
                f"{shape_id}: shape (M={m}, N={n}, K={k}) is outside "
                f"the fixed API contract {contract.source}: {exc}"
            ) from exc


def default_optimization_shapes(
    contract: OperatorAPIContract,
) -> list[dict[str, Any]]:
    module = _load_module(contract.source)
    raw = getattr(module, "DEFAULT_OPTIMIZATION_SHAPES", None)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(
            f"operator API contract {contract.source} must define "
            "DEFAULT_OPTIMIZATION_SHAPES as a sequence"
        )
    shapes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"DEFAULT_OPTIMIZATION_SHAPES[{index}] must be a mapping"
            )
        shape = {str(key): value for key, value in item.items()}
        shape_id = str(shape.get("id") or "").strip()
        if not shape_id:
            raise ValueError(
                f"DEFAULT_OPTIMIZATION_SHAPES[{index}] requires an id"
            )
        if shape_id in seen_ids:
            raise ValueError(
                f"duplicate default optimization shape id: {shape_id}"
            )
        seen_ids.add(shape_id)
        shapes.append(shape)
    if not shapes:
        raise ValueError("DEFAULT_OPTIMIZATION_SHAPES must not be empty")
    validate_contract_shapes(
        contract,
        {
            str(shape["id"]): {
                key: value for key, value in shape.items() if key != "id"
            }
            for shape in shapes
        },
    )
    return shapes


def stage_operator_api(
    contract: OperatorAPIContract,
    destination_dir: Path,
) -> Path:
    destination = destination_dir / contract.destination_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.chmod(0o644)
    # A New Task receives a new repository artifact.  Preserve the contract
    # bytes, but not the source asset's historical timestamp, so provenance
    # cannot be mistaken for code copied from an earlier task repository.
    shutil.copyfile(contract.source, destination)
    destination.chmod(0o444)
    return destination


def stage_operator_references(
    contract: OperatorAPIContract,
    destination_dir: Path,
) -> list[Path]:
    """Stage optional read-only evidence without adding it to the build."""
    staged: list[Path] = []
    references_dir = destination_dir / "references"
    for source in contract.reference_sources:
        references_dir.mkdir(parents=True, exist_ok=True)
        destination = references_dir / source.name
        if destination.exists():
            destination.chmod(0o644)
        shutil.copyfile(source, destination)
        destination.chmod(0o444)
        staged.append(destination)
    return staged


def file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
