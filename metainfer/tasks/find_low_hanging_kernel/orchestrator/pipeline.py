"""Pipeline — find-low-hanging-kernel main control flow.

Phase dispatch loop driven by :mod:`phases`. Each phase runs in its own
fresh-agent scope (cross-validation pools spawn new agents per phase per the
spec: "每一个大步骤，都使用一个全新的 Agent 实例").

Resume semantics: each phase checks for its output artifact on disk and
skips if present (idempotent re-runs). Validation rounds also short-circuit
when the graph is already clean.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.orchestrator.agent_pool import AgentPool, PoolTask
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager

from . import phases as P
from . import prompts as PP
from .graph_validator import run_validation_loop
from .iteration_record import IterationRecord
from .trace_parser import write_summary as write_trace_summary
from .visualizer import (
    render_from_files as render_flow_html,
    write_graph_json as write_flow_graph_json,
)


# Per-phase agent timeout (seconds). Step 1/2 agents read code + traces; Step 3
# builds a whole graph. Allow generous wall time.
S1_AGENT_TIMEOUT_S = 1800
S1_SYNTH_TIMEOUT_S = 1200
S2_AGENT_TIMEOUT_S = 1800
S2_SYNTH_TIMEOUT_S = 1200
S3_BUILD_TIMEOUT_S = 2400
S3_VALIDATE_TIMEOUT_S = 600
CROSS_VAL_POOL_SIZE = 3


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class OrchestratorConfig:
    workspace_dir: Path
    memory_dir: Path
    validation_dir: Path
    inputs_snapshot_dir: Path
    repo_root: Path
    state_dir: Path
    logs_root: Path
    max_validator_rounds: int = 5
    claude_bin: str = "ccb"
    model: Optional[str] = None
    permission_mode: str = "bypassPermissions"
    extra_claude_args: List[str] = field(default_factory=list)
    effort: str = "max"
    user_paths: List[Path] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _failure_outcome(mode: Optional[str]) -> P.Outcome:
    return P.INFRA_FAIL if mode == "infra" else P.LOGIC_FAIL


def _read_prompt_out(agent_name: str, manager) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Fetch the AgentResult for ``agent_name`` and return (success, error,
    failure_mode, session_id)."""
    result = manager.result(agent_name)
    if result is None:
        return False, "no result recorded", "infra", None
    return result.success, result.error, result.failure_mode, result.session_id


def _write_prompt_file(logs_dir: Path, name: str, prompt: str) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    p = logs_dir / f"{name}.prompt.txt"
    p.write_text(prompt, encoding="utf-8")
    return p


def _run_single_agent(
    *,
    manager,
    name: str,
    role: str,
    workdir: Path,
    logs_dir: Path,
    prompt: str,
    timeout: int,
    cfg: OrchestratorConfig,
    resume_session_id: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Launch a blocking SubAgentManager.launch turn."""
    workdir.mkdir(parents=True, exist_ok=True)
    prompt_file = _write_prompt_file(logs_dir, name, prompt)
    spec = AgentSpec(
        name=name,
        role=role,
        prompt_file=prompt_file,
        workdir=workdir,
        log_dir=logs_dir,
        timeout_s=timeout,
        stuck_timeout_s=max(120, timeout // 3),
        extra_args=list(cfg.extra_claude_args),
        resume_session_id=resume_session_id,
    )
    manager.launch(spec)
    return _read_prompt_out(name, manager)


def _run_pool(
    *,
    manager,
    tasks: List[PoolTask],
    log_dir: Path,
    role: str,
    name_prefix: str,
    timeout: int,
) -> List:
    """Run an AgentPool batch. Returns results in input order."""
    pool = AgentPool(
        manager,
        n_workers=CROSS_VAL_POOL_SIZE,
        log_dir=log_dir,
        role=role,
        name_prefix=name_prefix,
        timeout_s=timeout,
        stuck_timeout_s=max(60, timeout // 3),
        max_retries=2,
    )
    return pool.run(tasks)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class Pipeline:
    def __init__(
        self,
        req: Dict[str, Any],
        store: StateStore,
        cfg: OrchestratorConfig,
        manager: Optional[SubAgentManager] = None,
    ) -> None:
        self.req = req
        self.form: Dict[str, Any] = req.get("form") or {}
        self.store = store
        self.cfg = cfg
        self.manager = manager or SubAgentManager(
            claude_bin=cfg.claude_bin,
            default_model=cfg.model,
            permission_mode=cfg.permission_mode,
            extra_add_dirs=[cfg.workspace_dir, *cfg.user_paths],
        )
        self.task_id = req.get("task_id", "task")
        self._stop = False

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.store.init_or_resume(task_id=self.task_id)
        self.store.append_timeline("orchestrator_start", {"task_id": self.task_id})

        # Build inputs snapshot first — cheap, makes the workspace self-contained.
        self._snapshot_inputs()

        phase: P.Phase = self._resume_phase()
        last_outcome: Optional[P.Outcome] = None

        try:
            while not self._stop and not P.is_terminal(phase):
                self._set_phase(phase)
                outcome, failure = self._dispatch(phase)
                last_outcome = outcome
                self.store.append_timeline("transition", {
                    "from": phase, "outcome": outcome, "failure": failure,
                })

                t = P.next_transition(phase, outcome)
                if t is None:
                    self._fail_run(f"no transition for ({phase}, {outcome})")
                    return

                self.store.update_run(
                    current_phase=t.to_phase,
                    last_outcome=outcome,
                    last_transition_label=t.label,
                )
                phase = t.to_phase

            final_status = "success" if last_outcome in (P.OK, P.CLEAN) else "stopped"
            self.store.update_run(
                finished=True,
                final_status=final_status,
                current_phase="finished",
                last_outcome=last_outcome,
            )
            self.store.append_timeline("orchestrator_end", {
                "task_id": self.task_id, "final_status": final_status,
            })
        except KeyboardInterrupt:
            self.store.append_timeline(
                "orchestrator_abort", {"reason": "keyboard-interrupt"}
            )
            self.store.update_run(
                finished=True, final_status="aborted", current_phase="finished"
            )
        finally:
            try:
                self.manager.shutdown()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #

    def _resume_phase(self) -> P.Phase:
        """Pick up at the first phase whose output is missing."""
        if (self.cfg.memory_dir / "step1_code_analysis.md").is_file():
            if (self.cfg.memory_dir / "step2_tracing_analysis.md").is_file():
                if (self.cfg.workspace_dir / "flow_graph.json").is_file():
                    if (self.cfg.workspace_dir / "flow_graph.html").is_file():
                        # Everything is done; mark terminal.
                        return "finished"
                    return "P4_visualize"
                return "P3_graph_build"
            return "P2_tracing_analysis"
        return "P1_code_analysis"

    # ------------------------------------------------------------------ #
    # Phase dispatcher
    # ------------------------------------------------------------------ #

    def _dispatch(
        self, phase: P.Phase,
    ) -> Tuple[P.Outcome, Optional[str]]:
        if phase == "P1_code_analysis":
            return self._do_step1()
        if phase == "P2_tracing_analysis":
            return self._do_step2()
        if phase == "P3_graph_build":
            return self._do_step3_build()
        if phase == "P3_graph_validate":
            return self._do_step3_validate()
        if phase == "P4_visualize":
            return self._do_step4()
        return P.LOGIC_FAIL, f"no handler for phase {phase!r}"

    # ------------------------------------------------------------------ #
    # Step 1: code + quant + runtime analysis (3 agents + synthesis)
    # ------------------------------------------------------------------ #

    def _do_step1(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "step1"
        workdir = self.cfg.memory_dir / "build" / "step1"
        workdir.mkdir(parents=True, exist_ok=True)

        if (self.cfg.memory_dir / "step1_code_analysis.md").is_file():
            self.store.append_timeline("phase_skip", {"phase": "P1_code_analysis"})
            return P.OK, None

        roles = ["arch_tracer", "quant_tracer", "runtime_tracer"]
        tasks: List[PoolTask] = []
        for role in roles:
            agent_workdir = workdir / role
            agent_workdir.mkdir(parents=True, exist_ok=True)
            prompt = PP.step1_agent_prompt(
                role=role,
                form=self.form,
                workdir=agent_workdir,
                inputs_snapshot_dir=self.cfg.inputs_snapshot_dir,
            )
            tasks.append(PoolTask(
                key=f"s1_{role}",
                prompt=prompt,
                workdir=agent_workdir,
                name=f"s1-{role}",
            ))

        self.store.append_timeline("step1_pool_launch", {
            "agents": [t.name for t in tasks],
            "n_workers": CROSS_VAL_POOL_SIZE,
        })
        pool_results = _run_pool(
            manager=self.manager,
            tasks=tasks,
            log_dir=logs_dir / "pool",
            role="step1_analyst",
            name_prefix="s1",
            timeout=S1_AGENT_TIMEOUT_S,
        )

        # Persist individual reports as agent_1.md / agent_2.md / agent_3.md
        # (regardless of success — the synthesizer can still read partial work).
        report_paths: Dict[str, Path] = {}
        for i, (role, pr) in enumerate(zip(roles, pool_results), start=1):
            out_path = self.cfg.memory_dir / f"step1_agent_{i}.md"
            # Each agent was instructed to write report.md in its workdir.
            src = workdir / role / "report.md"
            if src.is_file():
                try:
                    out_path.write_text(
                        src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                except OSError:
                    pass
            report_paths[role] = out_path
            self.store.append_timeline("step1_agent_done", {
                "role": role, "success": pr.success, "duration_s": pr.duration_s,
                "error": pr.error,
            })

        # Synthesis (single fresh agent).
        synth_workdir = workdir / "synthesizer"
        synth_workdir.mkdir(parents=True, exist_ok=True)
        out_memory = self.cfg.memory_dir / "step1_code_analysis.md"
        synth_prompt = PP.step1_synthesis_prompt(
            form=self.form,
            reports=report_paths,
            out_path=out_memory,
        )
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager,
            name="s1-synthesizer",
            role="step1_synthesizer",
            workdir=synth_workdir,
            logs_dir=logs_dir,
            prompt=synth_prompt,
            timeout=S1_SYNTH_TIMEOUT_S,
            cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"S1 synthesis failed: {err}"
        if not out_memory.is_file():
            return P.LOGIC_FAIL, "S1 synthesis produced no step1_code_analysis.md"
        return P.OK, None

    # ------------------------------------------------------------------ #
    # Step 2: tracing analysis (deterministic parse + 3 agents + synthesis)
    # ------------------------------------------------------------------ #

    def _do_step2(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "step2"
        workdir = self.cfg.memory_dir / "build" / "step2"
        workdir.mkdir(parents=True, exist_ok=True)
        out_memory = self.cfg.memory_dir / "step2_tracing_analysis.md"
        if out_memory.is_file():
            self.store.append_timeline("phase_skip", {"phase": "P2_tracing_analysis"})
            return P.OK, None

        # 2a. Deterministic parse.
        trace_file = (self.form.get("trace_file") or "").strip()
        if not trace_file:
            return P.LOGIC_FAIL, "trace_file not provided"
        trace_path = Path(trace_file)
        if not trace_path.is_file():
            return P.LOGIC_FAIL, f"trace_file not found: {trace_path}"
        parsed_path = self.cfg.workspace_dir / "trace_parsed.json"
        try:
            write_trace_summary(trace_path, parsed_path)
        except Exception as exc:  # noqa: BLE001
            return P.INFRA_FAIL, f"trace parse failed: {exc!r}"
        self.store.append_timeline("step2_parse_done", {
            "source": trace_path.name,
            "output": str(parsed_path),
        })

        # 2b. Three cross-validation agents.
        step1_memory = self.cfg.memory_dir / "step1_code_analysis.md"
        roles = ["stat_analyst", "source_mapper", "tp_shape_analyst"]
        tasks: List[PoolTask] = []
        for role in roles:
            agent_workdir = workdir / role
            agent_workdir.mkdir(parents=True, exist_ok=True)
            prompt = PP.step2_agent_prompt(
                role=role,
                form=self.form,
                workdir=agent_workdir,
                trace_parsed_path=parsed_path,
                step1_memory_path=step1_memory,
            )
            tasks.append(PoolTask(
                key=f"s2_{role}",
                prompt=prompt,
                workdir=agent_workdir,
                name=f"s2-{role}",
            ))
        pool_results = _run_pool(
            manager=self.manager,
            tasks=tasks,
            log_dir=logs_dir / "pool",
            role="step2_analyst",
            name_prefix="s2",
            timeout=S2_AGENT_TIMEOUT_S,
        )

        report_paths: Dict[str, Path] = {}
        for i, (role, pr) in enumerate(zip(roles, pool_results), start=1):
            out_path = self.cfg.memory_dir / f"step2_agent_{i}.md"
            src = workdir / role / "report.md"
            if src.is_file():
                try:
                    out_path.write_text(
                        src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                except OSError:
                    pass
            report_paths[role] = out_path
            self.store.append_timeline("step2_agent_done", {
                "role": role, "success": pr.success, "duration_s": pr.duration_s,
                "error": pr.error,
            })

        # Synthesis.
        synth_workdir = workdir / "synthesizer"
        synth_workdir.mkdir(parents=True, exist_ok=True)
        synth_prompt = PP.step2_synthesis_prompt(
            form=self.form,
            reports=report_paths,
            out_path=out_memory,
        )
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager,
            name="s2-synthesizer",
            role="step2_synthesizer",
            workdir=synth_workdir,
            logs_dir=logs_dir,
            prompt=synth_prompt,
            timeout=S2_SYNTH_TIMEOUT_S,
            cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"S2 synthesis failed: {err}"
        if not out_memory.is_file():
            return P.LOGIC_FAIL, "S2 synthesis produced no step2_tracing_analysis.md"
        return P.OK, None

    # ------------------------------------------------------------------ #
    # Step 3a: graph build (single agent)
    # ------------------------------------------------------------------ #

    def _do_step3_build(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "step3_build"
        workdir = self.cfg.workspace_dir / "build" / "step3"
        workdir.mkdir(parents=True, exist_ok=True)
        out_graph = self.cfg.workspace_dir / "flow_graph.json"
        if out_graph.is_file():
            self.store.append_timeline("phase_skip", {"phase": "P3_graph_build"})
            return P.OK, None

        prompt = PP.step3_build_prompt(
            form=self.form,
            workdir=workdir,
            step1_memory_path=self.cfg.memory_dir / "step1_code_analysis.md",
            step2_memory_path=self.cfg.memory_dir / "step2_tracing_analysis.md",
            out_graph_path=out_graph,
        )
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager,
            name="s3-builder",
            role="graph_builder",
            workdir=workdir,
            logs_dir=logs_dir,
            prompt=prompt,
            timeout=S3_BUILD_TIMEOUT_S,
            cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"S3 build failed: {err}"
        if not out_graph.is_file():
            return P.LOGIC_FAIL, "S3 build produced no flow_graph.json"
        self.store.append_timeline("step3_build_done", {"graph": str(out_graph)})
        return P.OK, None

    # ------------------------------------------------------------------ #
    # Step 3b: iterative validation (deterministic driver + 5-worker pool)
    # ------------------------------------------------------------------ #

    def _do_step3_validate(self) -> Tuple[P.Outcome, Optional[str]]:
        graph_path = self.cfg.workspace_dir / "flow_graph.json"
        if not graph_path.is_file():
            return P.LOGIC_FAIL, "flow_graph.json missing — Step 3a must run first"

        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return P.LOGIC_FAIL, f"flow_graph.json unparseable: {exc!r}"

        rounds, exhausted = run_validation_loop(
            graph=graph,
            manager=self.manager,
            step1_path=self.cfg.memory_dir / "step1_code_analysis.md",
            step2_path=self.cfg.memory_dir / "step2_tracing_analysis.md",
            framework_dir=Path(self.form.get("framework_source_dir") or "."),
            validation_root=self.cfg.validation_dir,
            logs_root=self.cfg.logs_root,
            max_rounds=self.cfg.max_validator_rounds,
            timeout_s=S3_VALIDATE_TIMEOUT_S,
        )

        # Persist the (possibly patched) graph back to disk.
        write_flow_graph_json(graph, out_path=graph_path)

        # Record one iteration per round.
        for r in rounds:
            rec = IterationRecord(
                iteration=r.round_num,
                goal=f"validation round {r.round_num}",
                started_at=time.time(),
                ended_at=time.time(),
                status="success" if r.outcome == "clean" else "needs_fix",
                round=r.round_num,
                integrity_fixes=r.integrity.fixes_applied,
                semantic_issues=[
                    {"round": r.round_num, "issue_count": r.issue_count}
                ],
                outcome=r.outcome,
                artifacts=[str(p) for p in r.group_result_paths],
            )
            self.store.write_iteration(r.round_num, asdict(rec))
            self.store.append_timeline("validate_round_done", {
                "round": r.round_num,
                "outcome": r.outcome,
                "issue_count": r.issue_count,
                "fixes": len(r.integrity.fixes_applied),
            })

        if exhausted:
            # Best-effort: write a warning and proceed.
            warnings_path = self.cfg.memory_dir / "validation_warnings.md"
            warnings_path.write_text(
                "# Validation warnings\n\n"
                f"Reached the cap of {self.cfg.max_validator_rounds} validation "
                "rounds without converging on a clean graph. The graph may "
                "still contain minor issues; review the validation/round_* "
                "directories for details.\n",
                encoding="utf-8",
            )
            self.store.append_timeline("validate_exhausted", {
                "max_rounds": self.cfg.max_validator_rounds,
            })
            return P.CLEAN, None

        return P.CLEAN, None

    # ------------------------------------------------------------------ #
    # Step 4: visualization (deterministic)
    # ------------------------------------------------------------------ #

    def _do_step4(self) -> Tuple[P.Outcome, Optional[str]]:
        graph_path = self.cfg.workspace_dir / "flow_graph.json"
        html_path = self.cfg.workspace_dir / "flow_graph.html"
        if html_path.is_file():
            self.store.append_timeline("phase_skip", {"phase": "P4_visualize"})
            return P.OK, None
        if not graph_path.is_file():
            return P.LOGIC_FAIL, "flow_graph.json missing — cannot render"
        try:
            render_flow_html(
                graph_path=graph_path,
                out_html_path=html_path,
            )
        except Exception as exc:  # noqa: BLE001
            return P.INFRA_FAIL, f"visualizer failed: {exc!r}"
        self.store.append_timeline("visualize_done", {"html": str(html_path)})
        return P.OK, None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _set_phase(self, phase: P.Phase) -> None:
        self.store.update_run(current_phase=phase)
        self.store.append_timeline("phase_start", {"phase": phase})

    def _fail_run(self, reason: str) -> None:
        self.store.update_run(
            finished=True, final_status="stopped",
            current_phase="finished", last_outcome=P.LOGIC_FAIL,
        )
        self.store.append_timeline("orchestrator_fail", {"reason": reason})

    def _snapshot_inputs(self) -> None:
        """Copy small user-provided inputs into workspace/inputs_snapshot/
        so the workspace is self-contained for audit."""
        snapshot = self.cfg.inputs_snapshot_dir
        snapshot.mkdir(parents=True, exist_ok=True)

        cli_env = (self.form.get("cli_args_and_env") or "").strip()
        if cli_env:
            (snapshot / "cli_args_and_env.txt").write_text(cli_env, encoding="utf-8")

        model_dir = Path(self.form.get("model_dir") or "")
        if model_dir.is_dir():
            cfg_json = model_dir / "config.json"
            if cfg_json.is_file():
                try:
                    shutil.copy2(cfg_json, snapshot / "config.json")
                except OSError:
                    pass
            # Best-effort weights index copy: try common names.
            for idx_name in (
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
                "model.npz.json",
            ):
                src_idx = model_dir / idx_name
                if src_idx.is_file():
                    try:
                        shutil.copy2(src_idx, snapshot / "weights_index.json")
                    except OSError:
                        pass
                    break

        startup_log = Path(self.form.get("startup_log") or "")
        if startup_log.is_file():
            try:
                shutil.copy2(startup_log, snapshot / "startup_log.txt")
            except OSError:
                pass

        self.store.append_timeline("inputs_snapshot_done", {"dir": str(snapshot)})
