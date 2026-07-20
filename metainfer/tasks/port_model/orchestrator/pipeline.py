"""Pipeline — port-model main control flow.

Phase dispatch loop driven by :mod:`phases`. Each phase gets a fresh agent
(sub-agent launch, not AgentPool — this task has one analysis angle per step).

Resume semantics: each phase checks for its output artifact on disk and
skips if present. P5 → P4 repair loop is capped at 3 retries.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec, SubAgentManager

from . import phases as P
from . import prompts as PP
from .iteration_record import IterationRecord


# Per-phase agent timeout (seconds).
S1_TIMEOUT_S = 600   # model analysis: read config.json + weight files
S2_TIMEOUT_S = 900   # source framework analysis: grep for model name
S3_TIMEOUT_S = 900   # target framework analysis
S4_TIMEOUT_S = 1800  # implementation: write registration + layers
S5_TIMEOUT_S = 1200  # testing: boot server + compare outputs

MAX_REPAIR_RETRIES = 3


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class OrchestratorConfig:
    workspace_dir: Path
    memory_dir: Path
    diff_dir: Path
    test_dir: Path
    inputs_snapshot_dir: Path
    repo_root: Path
    state_dir: Path
    logs_root: Path
    source_framework_dir: Path = field(default_factory=Path)
    target_framework_dir: Path = field(default_factory=Path)
    model_dir: Path = field(default_factory=Path)
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


def _read_prompt_out(
    agent_name: str, manager,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
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
        self._repair_count = 0

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.store.init_or_resume(task_id=self.task_id)
        self.store.append_timeline("orchestrator_start", {"task_id": self.task_id})

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

                # Repair loop counter: P5 → P4 retry.
                if outcome == P.TEST_FAIL:
                    self._repair_count += 1
                    if self._repair_count > MAX_REPAIR_RETRIES:
                        self.store.append_timeline("repair_exhausted", {
                            "retries": self._repair_count,
                        })
                        # Force stop with best-effort result.
                        self._fail_run("max repair retries exceeded")
                        return
                elif phase != "P5_test":
                    self._repair_count = 0

            final_status = (
                "success" if last_outcome in (P.OK, P.TEST_FAIL) else "stopped"
            )
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
        m = self.cfg.memory_dir
        if (m / "p1_model_analysis.md").is_file():
            if (m / "p2_source_analysis.md").is_file():
                if (m / "p3_target_analysis.md").is_file():
                    if (self.cfg.diff_dir / "model_port.patch").is_file():
                        test_results = self.cfg.test_dir / "test_results.json"
                        if test_results.is_file():
                            try:
                                tr = json.loads(test_results.read_text(encoding="utf-8"))
                                if tr.get("passed"):
                                    return "finished"
                            except (json.JSONDecodeError, OSError):
                                pass
                        return "P5_test"
                    return "P4_implement"
                return "P3_target_analysis"
            return "P2_source_analysis"
        return "P1_model_analysis"

    # ------------------------------------------------------------------ #
    # Phase dispatcher
    # ------------------------------------------------------------------ #

    def _dispatch(self, phase: P.Phase) -> Tuple[P.Outcome, Optional[str]]:
        if phase == "P1_model_analysis":
            return self._do_p1()
        if phase == "P2_source_analysis":
            return self._do_p2()
        if phase == "P3_target_analysis":
            return self._do_p3()
        if phase == "P4_implement":
            return self._do_p4()
        if phase == "P5_test":
            return self._do_p5()
        return P.LOGIC_FAIL, f"no handler for phase {phase!r}"

    # ------------------------------------------------------------------ #
    # P1: Model architecture analysis
    # ------------------------------------------------------------------ #

    def _do_p1(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "p1"
        workdir = self.cfg.memory_dir / "build" / "p1"
        out = self.cfg.memory_dir / "p1_model_analysis.md"
        if out.is_file():
            self.store.append_timeline("phase_skip", {"phase": "P1_model_analysis"})
            return P.OK, None

        prompt = PP.p1_model_analysis_prompt(form=self.form, workdir=workdir)
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager, name="p1-analyst", role="p1_analyst",
            workdir=workdir, logs_dir=logs_dir, prompt=prompt,
            timeout=S1_TIMEOUT_S, cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"P1 failed: {err}"

        # Agent writes to workdir; copy to memory if needed.
        src = workdir / "p1_model_analysis.md"
        if src.is_file():
            try:
                out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                return P.INFRA_FAIL, "failed to persist p1_model_analysis.md"
        if not out.is_file():
            return P.LOGIC_FAIL, "P1 produced no p1_model_analysis.md"
        return P.OK, None

    # ------------------------------------------------------------------ #
    # P2: Source framework analysis
    # ------------------------------------------------------------------ #

    def _do_p2(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "p2"
        workdir = self.cfg.memory_dir / "build" / "p2"
        out = self.cfg.memory_dir / "p2_source_analysis.md"
        if out.is_file():
            self.store.append_timeline("phase_skip", {"phase": "P2_source_analysis"})
            return P.OK, None

        p1_path = self.cfg.memory_dir / "p1_model_analysis.md"
        prompt = PP.p2_source_analysis_prompt(
            form=self.form, workdir=workdir, p1_path=p1_path,
        )
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager, name="p2-analyst", role="p2_analyst",
            workdir=workdir, logs_dir=logs_dir, prompt=prompt,
            timeout=S2_TIMEOUT_S, cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"P2 failed: {err}"

        src = workdir / "p2_source_analysis.md"
        if src.is_file():
            try:
                out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                return P.INFRA_FAIL, "failed to persist p2_source_analysis.md"
        if not out.is_file():
            return P.LOGIC_FAIL, "P2 produced no p2_source_analysis.md"
        return P.OK, None

    # ------------------------------------------------------------------ #
    # P3: Target framework analysis
    # ------------------------------------------------------------------ #

    def _do_p3(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "p3"
        workdir = self.cfg.memory_dir / "build" / "p3"
        out = self.cfg.memory_dir / "p3_target_analysis.md"
        if out.is_file():
            self.store.append_timeline("phase_skip", {"phase": "P3_target_analysis"})
            return P.OK, None

        p1_path = self.cfg.memory_dir / "p1_model_analysis.md"
        p2_path = self.cfg.memory_dir / "p2_source_analysis.md"
        prompt = PP.p3_target_analysis_prompt(
            form=self.form, workdir=workdir, p1_path=p1_path, p2_path=p2_path,
        )
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager, name="p3-analyst", role="p3_analyst",
            workdir=workdir, logs_dir=logs_dir, prompt=prompt,
            timeout=S3_TIMEOUT_S, cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"P3 failed: {err}"

        src = workdir / "p3_target_analysis.md"
        if src.is_file():
            try:
                out.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                return P.INFRA_FAIL, "failed to persist p3_target_analysis.md"
        if not out.is_file():
            return P.LOGIC_FAIL, "P3 produced no p3_target_analysis.md"
        return P.OK, None

    # ------------------------------------------------------------------ #
    # P4: Implementation
    # ------------------------------------------------------------------ #

    def _do_p4(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "p4"
        workdir = self.cfg.memory_dir / "build" / "p4"

        p1_path = self.cfg.memory_dir / "p1_model_analysis.md"
        p2_path = self.cfg.memory_dir / "p2_source_analysis.md"
        p3_path = self.cfg.memory_dir / "p3_target_analysis.md"

        prev_failure = None
        if self._repair_count > 0:
            test_results_path = self.cfg.test_dir / "test_results.json"
            if test_results_path.is_file():
                try:
                    prev_failure = test_results_path.read_text(encoding="utf-8")
                except OSError:
                    pass

        prompt = PP.p4_implement_prompt(
            form=self.form, workdir=workdir,
            p1_path=p1_path, p2_path=p2_path, p3_path=p3_path,
            prev_test_failure=prev_failure,
        )
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager, name="p4-implementer", role="p4_implementer",
            workdir=workdir, logs_dir=logs_dir, prompt=prompt,
            timeout=S4_TIMEOUT_S, cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"P4 failed: {err}"

        # Check that patch file exists.
        patch_path = self.cfg.diff_dir / "model_port.patch"
        if not patch_path.is_file():
            # Best-effort: generate patch ourselves.
            self._generate_patch()

        self.store.append_timeline("p4_implement_done", {
            "patch": str(patch_path) if patch_path.exists() else None,
        })
        return P.OK, None

    def _generate_patch(self) -> None:
        import subprocess
        target = self.cfg.target_framework_dir
        if not target.is_dir():
            return
        patch_path = self.cfg.diff_dir / "model_port.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["git", "diff"], cwd=str(target),
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                patch_path.write_text(result.stdout, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # P5: Test
    # ------------------------------------------------------------------ #

    def _do_p5(self) -> Tuple[P.Outcome, Optional[str]]:
        logs_dir = self.cfg.logs_root / "p5"
        workdir = self.cfg.memory_dir / "build" / "p5"

        p4_changes = self.cfg.memory_dir / "build" / "p4" / "p4_changes.md"
        prev_results = self.cfg.test_dir / "test_results.json"

        prompt = self._p5_test_prompt(workdir=workdir, p4_changes=p4_changes,
                                       prev_results=prev_results if prev_results.is_file() else None)
        ok, err, mode, _ = _run_single_agent(
            manager=self.manager, name="p5-tester", role="p5_tester",
            workdir=workdir, logs_dir=logs_dir, prompt=prompt,
            timeout=S5_TIMEOUT_S, cfg=self.cfg,
        )
        if not ok:
            return _failure_outcome(mode), f"P5 failed: {err}"

        # Read test verdict from the agent's output.
        results_path = workdir / "test_results.json"
        if results_path.is_file():
            try:
                shutil.copy2(results_path, self.cfg.test_dir / "test_results.json")
            except OSError:
                pass
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
                if results.get("passed"):
                    return P.OK, None
                return P.TEST_FAIL, results.get("error", "test not passed")
            except (json.JSONDecodeError, OSError):
                return P.LOGIC_FAIL, "P5 produced unparseable test_results.json"

        return P.LOGIC_FAIL, "P5 produced no test_results.json"

    def _p5_test_prompt(
        self, workdir: Path, p4_changes: Path,
        prev_results: Optional[Path],
    ) -> str:
        from metainfer.orchestrator.requirements import req_field, req_summary_lines

        model_dir = req_field(self.form, "model_dir") or ""
        source_fw = req_field(self.form, "source_framework_dir") or ""
        target_fw = req_field(self.form, "target_framework_dir") or ""
        fw_type = req_field(self.form, "target_framework_type") or "unknown"
        hw = req_field(self.form, "target_hardware") or "unknown"
        test_prompts = req_field(self.form, "test_prompts") or ""

        prompts_block = test_prompts.strip() if test_prompts.strip() else """\
- "What is the capital of France?"
- "Explain quantum computing in one paragraph."
- "Write a Python function to compute fibonacci numbers."\
"""

        prev_section = ""
        if prev_results and prev_results.is_file():
            try:
                prev_section = f"""\
# ⚠️ PREVIOUS TEST FAILURE

The previous P5 run failed. Here are the last test results for context:
```
{prev_results.read_text(encoding='utf-8')[:3000]}
```
"""
            except OSError:
                pass

        return f"""\
# ⛔ READ-ONLY

  - MODEL_DIR = {model_dir}
  - SOURCE_FRAMEWORK_DIR = {source_fw}

# Task: Test the model in the target framework

Verify that the model added to `{target_fw}` works correctly.

## Framework context

  - Target framework: {fw_type} at {target_fw}
  - Target hardware: {hw}
  - Model: {model_dir}

## Test plan

1. **Boot the target framework** with the new model. Depending on {fw_type}:
   - vLLM: `python -m vllm.entrypoints.openai.api_server --model {model_dir} --port <port>`
   - SGLang: `python -m sglang.launch_server --model-path {model_dir} --port <port>`
   - Use a free port (bind to 0, then read the assigned port).

2. **Send test prompts** and collect responses:
{prompts_block}

3. **Boot the source framework** at `{source_fw}` with the same model (if it
   has a server mode), and collect responses to the SAME prompts.

4. **Compare** the source and target framework outputs. They should be
   semantically equivalent (temperature=0 for deterministic output).

5. **Write results** to `{workdir}/test_results.json`:

```json
{{
  "passed": true|false,
  "total_cases": N,
  "passed_cases": N,
  "results": [
    {{
      "prompt": "...",
      "source_output": "...",
      "target_output": "...",
      "match": true|false,
      "reason": "why they match/differ"
    }}
  ],
  "error": "if !passed, summary of why"
}}
```

## Notes

- Use temperature=0 and fixed seed for deterministic comparison.
- If boot fails, record the error and return passed=false with the error.
- If {fw_type} requires a specific launch method, follow its conventions.
- Target hardware is {hw} — use any {hw}-specific flags if needed.

{prev_section}
"""

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
        snapshot = self.cfg.inputs_snapshot_dir
        snapshot.mkdir(parents=True, exist_ok=True)

        model_dir = Path(self.form.get("model_dir") or "")
        if model_dir.is_dir():
            cfg_json = model_dir / "config.json"
            if cfg_json.is_file():
                try:
                    shutil.copy2(cfg_json, snapshot / "config.json")
                except OSError:
                    pass
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

        self.store.append_timeline("inputs_snapshot_done",
                                   {"dir": str(snapshot)})
