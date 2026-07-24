"""Bootstrap the self-contained GEMM kernel orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.orchestrator._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from metainfer.orchestrator.requirements import req_field, req_field_int
from metainfer.orchestrator.state import StateStore

from .evaluator import FrozenEvaluatorBundle, FrozenWeightBundle, SpecError
from .build import BuildProfile, SystemBuilder
from .hardware import require_hardware_profile
from .pipeline import Orchestrator, OrchestratorConfig
from .profiler import FrozenProfilerProfile, ProfilerRunner


_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    max_iterations: Optional[int] = None,
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
) -> int:
    req: Dict[str, Any] = json.loads(requirements_path.read_text(encoding="utf-8"))
    task_id = str(req.get("task_id") or "task")
    state_dir = state_dir or Path.cwd() / "nodes" / "localhost" / ".metainfer" / "tasks" / task_id
    workspace_dir = workspace_dir or Path.cwd() / "nodes" / "localhost" / "workspaces" / task_id
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    logs_root = state_dir / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    target_req = state_dir / "requirements.json"
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(requirements_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Harness is the user-facing name; evaluator_bundle remains the persisted
    # requirements key for compatibility with existing tasks and API clients.
    bundle_value = str(req_field(req, "evaluator_bundle") or "").strip()
    if not bundle_value:
        raise SpecError(
            "Harness path (evaluator_bundle) is required for opt_GEMM_kernel"
        )
    bundle = FrozenEvaluatorBundle.materialize(
        Path(bundle_value), state_dir / "system_evaluator"
    )
    weight_value = str(req_field(req, "weight_bundle") or "").strip()
    if not weight_value:
        raise SpecError("Weight directory (weight_bundle) is required for opt_GEMM_kernel")
    weight_bundle = FrozenWeightBundle.materialize(
        Path(weight_value), state_dir / "system_weights"
    )
    _, hardware_profile = require_hardware_profile(req)
    build_profile = BuildProfile.from_requirements(req, hardware_profile)
    system_builder = SystemBuilder(
        build_profile, state_dir / "system_build", harness_source=None
    )
    profiler_profile = FrozenProfilerProfile.resolve(
        req, hardware_profile, state_dir / "system_profiler"
    )
    profile_cmd = bundle.spec.commands.get("profile")
    profiler_runner = None
    if profiler_profile is not None:
        harness_argv = None
        if profile_cmd is not None:
            values = {"bundle_dir": str(bundle.root.resolve())}
            harness_argv = [part.format_map(values) for part in profile_cmd.argv]
        profiler_runner = ProfilerRunner(
            profiler_profile,
            private_env={
                "METAINFER_WEIGHT_BUNDLE": str(weight_bundle.root.resolve()),
                "METAINFER_WEIGHT_SHA256": weight_bundle.digest,
            },
            harness_argv=harness_argv,
        )
    initial_value = str(req_field(req, "initial_submission") or "").strip()
    initial_submission = Path(initial_value).expanduser().resolve() if initial_value else None
    if initial_submission is not None and not initial_submission.is_dir():
        raise FileNotFoundError(f"initial_submission is not a directory: {initial_submission}")

    set_process_name("metainfer-gemm")
    pid_file = state_dir / "orchestrator.pid"
    write_pid_file(pid_file, task_id)
    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=[_NOTEBOOKS_DIR],
        snapshot_file=state_dir / "agents.json",
    )
    cfg = OrchestratorConfig(
        state_dir=state_dir,
        iterations_root=workspace_dir,
        logs_root=logs_root,
        notebooks_dir=_NOTEBOOKS_DIR,
        evaluator_bundle=bundle,
        weight_bundle=weight_bundle,
        system_builder=system_builder,
        profiler=profiler_runner,
        initial_submission=initial_submission,
        max_iterations=max_iterations or req_field_int(req, "max_iterations", 20),
        extra_claude_args=list(extra_claude_args or []),
    )
    orch = Orchestrator(req, StateStore(state_dir), cfg, manager)
    restore_signals = install_subagent_shutdown_handlers(manager, pid_file=pid_file)
    try:
        orch.run()
    finally:
        restore_signals()
        clear_pid_file(pid_file)
    return 0
