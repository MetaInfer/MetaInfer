"""Validated, operator-agnostic configuration for the optimizer MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

MOCK_MODE = "Mock (no GPU)"
LEGACY_SMOKE_MODE = "Real agents + DCU (smoke harness)"
SMOKE_MODE = "Infrastructure smoke (not operator optimization)"
W8A8_MODE = "Real INT8 W8A8 GEMM"
GEN_AND_OPT_MODE = "Generate & optimize (auto-create kernel repo)"
CLAUDE_MODELS = {
    "Sonnet": "claude-sonnet-5",
    "Opus": "claude-opus-5",
}
# Agent frameworks selectable from the new-task form. Each framework maps a
# user-facing model label to the model id handed to SubAgentManager (and from
# there to the agent binary via --model). ccb keeps Claude Code semantics;
# dsh routes claude_bin to bridge/dsh/dsh_agent.py (see resolve_claude_bin).
CCB_FRAMEWORK = "ccb"
DSH_FRAMEWORK = "dsh"
DSH_DEFAULT_MODEL_ID = "deepseek/deepseek-v4-flash-0731"


def dsh_model_id() -> str:
    """Model id used for the dsh framework (env DSH_AGENT_MODEL overrides)."""
    override = os.environ.get("DSH_AGENT_MODEL")
    return (override.strip() if override and override.strip()
            else DSH_DEFAULT_MODEL_ID)


def agent_framework_models(framework: str) -> Dict[str, str]:
    """Label -> model id for one agent framework."""
    if framework == DSH_FRAMEWORK:
        return {"deepseek-v4-flash": dsh_model_id()}
    return dict(CLAUDE_MODELS)


def agent_framework_default_model(framework: str) -> str:
    return "deepseek-v4-flash" if framework == DSH_FRAMEWORK else "Opus"


def resolve_agent_framework(req: Mapping[str, Any]) -> str:
    """Read + validate the agent_framework answer (default ccb)."""
    answers = _answers(req)
    framework = str(
        answers.get("agent_framework") or CCB_FRAMEWORK
    ).strip().lower()
    if framework not in {CCB_FRAMEWORK, DSH_FRAMEWORK}:
        raise ValueError(
            "agent_framework must be one of "
            f"{sorted([CCB_FRAMEWORK, DSH_FRAMEWORK])}"
        )
    return framework


def resolve_model_id(
    req: Mapping[str, Any], framework: str | None = None
) -> str:
    """Resolve the model id for the chosen framework.

    Prefers the ``agent_model`` answer; falls back to the legacy
    ``claude_model`` answer (pre-framework requirements files) and finally to
    the framework default.
    """
    framework = framework or resolve_agent_framework(req)
    answers = _answers(req)
    models = agent_framework_models(framework)
    label = str(answers.get("agent_model") or "").strip()
    if not label:
        label = str(answers.get("claude_model") or "").strip()
    if not label:
        label = agent_framework_default_model(framework)
    try:
        return models[label]
    except KeyError as exc:
        raise ValueError(
            f"agent_model must be one of {sorted(models)} for "
            f"framework {framework!r}"
        ) from exc


def resolve_claude_bin(framework: str, explicit: str | None = None) -> str:
    """Pick the agent binary for a framework.

    An explicit ``--claude-bin`` always wins. Otherwise dsh uses the bundled
    ccb-compatible DSH wrapper (bridge/dsh/dsh_agent.py) and ccb falls back to
    ``METAINFER_CLAUDE_BIN`` (or ``ccb``).
    """
    if explicit:
        return explicit
    if framework == DSH_FRAMEWORK:
        return str(
            Path(__file__).resolve().parents[1]
            / "bridge" / "dsh" / "dsh_agent.py"
        )
    return os.environ.get("METAINFER_CLAUDE_BIN", "ccb")
# A small, stable gain may be accumulated across rounds. The user-facing
# minimum_improvement_percent is the final target versus the fixed baseline.
ROUND_ACCEPTANCE_IMPROVEMENT_PERCENT = 1.0


def _kernel_repos_root() -> Path:
    """Root directory for operator kernel repositories.

    ``kernel-repos/`` sits next to the MetaInfer root dir. When a user
    types a relative name in the Kernel repo field (e.g. "int8 test3"),
    it resolves to ``<kernel-repos>/int8 test3/``.
    """
    override = os.environ.get("METAINFER_KERNEL_REPOS")
    if override:
        return Path(override).expanduser().resolve()
    # Derive from METAINFER_ROOT or this installed MetaInfer package. Do not
    # use cwd: orchestrators run with their task state directory as cwd.
    meta_root = os.environ.get("METAINFER_ROOT")
    base = (
        Path(meta_root).expanduser().resolve()
        if meta_root
        else Path(__file__).resolve().parents[4]
    )
    return (base / ".." / "kernel-repos").resolve()


def _kernel_repo_from_name(name: str) -> Path:
    """Resolve one repository folder name below the sibling kernel-repos."""
    repo_name = name.strip()
    candidate = Path(repo_name)
    if (
        not repo_name
        or repo_name in {".", ".."}
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or "/" in repo_name
        or "\\" in repo_name
    ):
        raise ValueError(
            "Kernel repo must be a single folder name under kernel-repos "
            "(for example: int8 test2)"
        )
    kernel_root = _kernel_repos_root()
    target = (kernel_root / repo_name).resolve()
    if target.parent != kernel_root:
        raise ValueError("Kernel repo resolves outside kernel-repos")
    return target


@dataclass(frozen=True)
class ShapeSpec:
    id: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: str
    gpu: int
    shape_ids: List[str]


@dataclass(frozen=True)
class OptimizerConfig:
    operator: str
    dtype: str
    hardware: str
    kernel_language: str
    claude_model: str
    execution_mode: str
    target_repo_path: Path | None
    shapes: Dict[str, ShapeSpec]
    assignments: List[WorkerAssignment]
    assignment_mode: str
    shape_scope: str
    mock_iterations: int
    minimum_improvement_percent: float
    # Agent framework (ccb | dsh) that produced claude_model; the pipeline
    # uses it to pick the agent binary via resolve_claude_bin.
    agent_framework: str = CCB_FRAMEWORK


def _answers(req: Mapping[str, Any]) -> Mapping[str, Any]:
    value = req.get("answers")
    return value if isinstance(value, Mapping) else req


def _parse_shape_config(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        parsed = dict(raw)
    elif isinstance(raw, str):
        try:
            parsed = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"shape_config is not valid YAML: {exc}") from exc
    else:
        raise ValueError("shape_config must be YAML text or an object")
    if not isinstance(parsed, dict):
        raise ValueError("shape_config must decode to an object")
    # Backward compatibility for the first UI prototype, which nested full
    # shape objects under each worker:
    #
    # workers:
    #   worker_0: {gpu: 0, shapes: [{id: m2, M: 2, ...}]}
    #
    # Normalize it to the current canonical shapes + assignments schema.
    legacy_workers = parsed.get("workers")
    if (
        "shapes" not in parsed
        and "assignments" not in parsed
        and isinstance(legacy_workers, Mapping)
    ):
        shapes: list[Dict[str, Any]] = []
        assignments: Dict[str, Any] = {}
        for worker_id, worker in legacy_workers.items():
            if not isinstance(worker, Mapping):
                raise ValueError(f"worker {worker_id!r} must be an object")
            shape_ids: list[str] = []
            for shape in worker.get("shapes") or []:
                if not isinstance(shape, Mapping):
                    raise ValueError(
                        f"{worker_id}: legacy shapes must be full objects"
                    )
                item = dict(shape)
                shape_id = str(item.get("id") or "").strip()
                if not shape_id:
                    raise ValueError(
                        f"{worker_id}: every legacy shape requires an id"
                    )
                shapes.append(item)
                shape_ids.append(shape_id)
            assignments[str(worker_id)] = {
                "gpu": worker.get("gpu", -1),
                "shapes": shape_ids,
            }
        parsed = {"shapes": shapes, "assignments": assignments}
    return parsed


def load_config(req: Mapping[str, Any]) -> OptimizerConfig:
    answers = _answers(req)
    agent_framework = resolve_agent_framework(req)
    claude_model = resolve_model_id(req, agent_framework)
    mode = str(answers.get("execution_mode", MOCK_MODE))
    if mode not in {
        MOCK_MODE,
        LEGACY_SMOKE_MODE,
        SMOKE_MODE,
        W8A8_MODE,
        GEN_AND_OPT_MODE,
    }:
        raise ValueError("unsupported execution_mode")

    target_raw = str(answers.get("target_repo_path") or "").strip()
    target = Path(target_raw).expanduser() if target_raw else None

    # Smoke modes don't consume an external operator repository.
    if mode in {LEGACY_SMOKE_MODE, SMOKE_MODE}:
        target = None
    elif mode == GEN_AND_OPT_MODE:
        # Generate mode always owns a concrete repository under the
        # kernel-repos directory next to MetaInfer. The user-facing field is
        # a folder name, not an arbitrary filesystem path.
        repo_name = target_raw or str(req.get("task_id") or "generated-kernel")
        target = _kernel_repo_from_name(repo_name)
    elif target is not None:
        if target.is_absolute():
            # Absolute path: use as-is if it exists.
            if not target.is_dir():
                target = None
        else:
            target = _kernel_repo_from_name(target_raw)

    raw_shape_config = answers.get("shape_config")
    using_api_defaults = False
    if (
        raw_shape_config is None
        or (
            isinstance(raw_shape_config, str)
            and not raw_shape_config.strip()
        )
    ):
        using_api_defaults = True
        from .api_contracts import (
            default_optimization_shapes,
            resolve_operator_api,
        )

        contract = resolve_operator_api(
            str(answers.get("operator") or "Custom operator"),
            str(answers.get("dtype") or "Other"),
        )
        raw_shape_config = {
            "shapes": default_optimization_shapes(contract),
        }
    parsed = _parse_shape_config(raw_shape_config)
    shape_scope = str(
        parsed.get("shape_scope")
        or ("all" if using_api_defaults else "custom")
    ).strip().lower()
    if shape_scope not in {"all", "subset", "custom"}:
        raise ValueError(
            "shape_config.shape_scope must be all, subset or custom"
        )
    raw_shapes = parsed.get("shapes")
    if not isinstance(raw_shapes, list) or not raw_shapes:
        raise ValueError("shape_config.shapes must be a non-empty list")

    shapes: Dict[str, ShapeSpec] = {}
    for item in raw_shapes:
        if not isinstance(item, Mapping):
            raise ValueError("every shape must be an object")
        shape_id = str(item.get("id") or "").strip()
        if not shape_id:
            raise ValueError("every shape requires a non-empty id")
        if shape_id in shapes:
            raise ValueError(f"duplicate shape id: {shape_id}")
        shapes[shape_id] = ShapeSpec(
            id=shape_id,
            params={str(k): v for k, v in item.items() if k != "id"},
        )

    raw_assignments = parsed.get("assignments")
    assignment_mode = str(
        parsed.get("assignment_mode")
        or ("manual" if raw_assignments else "ai")
    ).strip().lower()
    if assignment_mode not in {"ai", "manual"}:
        raise ValueError("shape_config.assignment_mode must be ai or manual")
    assignments_omitted = (
        assignment_mode == "ai"
        and (mode == GEN_AND_OPT_MODE or using_api_defaults)
        and (not isinstance(raw_assignments, Mapping) or not raw_assignments)
    )
    if not assignments_omitted:
        if not isinstance(raw_assignments, Mapping) or not raw_assignments:
            raise ValueError("shape_config.assignments must be a non-empty object")
    if isinstance(raw_assignments, Mapping) and len(raw_assignments) > 4:
        raise ValueError("worker29 MVP supports at most four workers")

    assignments: List[WorkerAssignment] = []
    seen_gpus: set[int] = set()
    assigned_shapes: set[str] = set()

    if assignments_omitted:
        # API-default workloads do not carry machine-specific GPU placement.
        # Start with one deterministic worker. Generate mode replaces this
        # placeholder with the generate agent's validated proposal.
        assignments.append(
            WorkerAssignment("worker_0", 0, list(shapes.keys()))
        )
        assigned_shapes.update(shapes.keys())
    else:
        assert isinstance(raw_assignments, Mapping)
        for worker_id, item in raw_assignments.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"assignment {worker_id!r} must be an object")
            gpu = int(item.get("gpu", -1))
            if gpu not in range(4):
                raise ValueError(f"{worker_id}: gpu must be one of 0,1,2,3")
            if gpu in seen_gpus:
                raise ValueError(f"physical GPU {gpu} is assigned more than once")
            seen_gpus.add(gpu)
            shape_ids = [str(v) for v in (item.get("shapes") or [])]
            if not shape_ids:
                raise ValueError(f"{worker_id}: shapes must not be empty")
            unknown = [sid for sid in shape_ids if sid not in shapes]
            if unknown:
                raise ValueError(f"{worker_id}: unknown shapes: {unknown}")
            duplicate = [sid for sid in shape_ids if sid in assigned_shapes]
            if duplicate:
                raise ValueError(f"shapes assigned more than once: {duplicate}")
            assigned_shapes.update(shape_ids)
            assignments.append(
                WorkerAssignment(str(worker_id), gpu, shape_ids)
            )

    missing = sorted(set(shapes) - assigned_shapes)
    if missing:
        raise ValueError(f"unassigned shapes: {missing}")

    try:
        iterations = int(
            answers.get("max_iterations", answers.get("mock_iterations", 3))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("mock_iterations must be an integer") from exc
    if iterations < 1 or iterations > 50:
        raise ValueError("mock_iterations must be between 1 and 50")

    try:
        threshold = float(answers.get("minimum_improvement_percent", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_improvement_percent must be numeric") from exc
    if threshold < 0:
        raise ValueError("minimum_improvement_percent must be non-negative")

    return OptimizerConfig(
        operator=str(answers.get("operator") or "Custom operator"),
        dtype=str(answers.get("dtype") or "Other"),
        hardware=str(answers.get("target_hardware") or "Other DCU"),
        kernel_language=str(answers.get("kernel_language") or "HIP C++"),
        claude_model=claude_model,
        execution_mode=mode,
        target_repo_path=target,
        shapes=shapes,
        assignments=assignments,
        assignment_mode=assignment_mode,
        shape_scope=shape_scope,
        mock_iterations=iterations,
        minimum_improvement_percent=threshold,
        agent_framework=agent_framework,
    )


def replace_assignments(
    config: OptimizerConfig,
    assignments: List[WorkerAssignment],
) -> OptimizerConfig:
    """Return a new OptimizerConfig with the given assignments."""
    return OptimizerConfig(
        operator=config.operator,
        dtype=config.dtype,
        hardware=config.hardware,
        kernel_language=config.kernel_language,
        claude_model=config.claude_model,
        execution_mode=config.execution_mode,
        target_repo_path=config.target_repo_path,
        shapes=config.shapes,
        assignments=assignments,
        assignment_mode=config.assignment_mode,
        shape_scope=config.shape_scope,
        mock_iterations=config.mock_iterations,
        minimum_improvement_percent=config.minimum_improvement_percent,
        agent_framework=config.agent_framework,
    )


def validate_gpu_assignment(
    shapes: Dict[str, ShapeSpec],
    raw: Dict[str, Any],
) -> List[WorkerAssignment]:
    """Validate and parse a GPU assignment dict from the generate agent.

    Args:
        shapes: The known shape specs.
        raw: A dict mapping worker_id → {gpu: int, shapes: [str, ...]}.

    Returns:
        A validated list of WorkerAssignment.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError("gpu_assignment must be a non-empty object")
    if len(raw) > 4:
        raise ValueError("at most four workers supported")

    assignments: List[WorkerAssignment] = []
    seen_gpus: set[int] = set()
    assigned_shapes: set[str] = set()

    for worker_id, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"{worker_id}: must be an object")
        gpu = int(item.get("gpu", -1))
        if gpu not in range(4):
            raise ValueError(f"{worker_id}: gpu must be 0-3, got {gpu}")
        if gpu in seen_gpus:
            raise ValueError(f"GPU {gpu} assigned to more than one worker")
        seen_gpus.add(gpu)

        shape_ids = [str(s) for s in (item.get("shapes") or [])]
        if not shape_ids:
            raise ValueError(f"{worker_id}: shapes must not be empty")
        unknown = [sid for sid in shape_ids if sid not in shapes]
        if unknown:
            raise ValueError(f"{worker_id}: unknown shapes: {unknown}")
        duplicate = [sid for sid in shape_ids if sid in assigned_shapes]
        if duplicate:
            raise ValueError(f"shapes assigned more than once: {duplicate}")
        assigned_shapes.update(shape_ids)

        assignments.append(
            WorkerAssignment(str(worker_id), gpu, shape_ids)
        )

    missing = sorted(set(shapes) - assigned_shapes)
    if missing:
        raise ValueError(f"unassigned shapes: {missing}")

    return assignments
