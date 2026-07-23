"""Focused integration tests for the task-local A/B/D hard gates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from metainfer.orchestrator.state import StateStore
from metainfer.tasks.gen_cpp_infer_framework.orchestrator import phases
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.capabilities import (
    resolve_capabilities,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.pipeline import (
    CppInferIterationWorkspace,
    IterationContext,
    Orchestrator,
    OrchestratorConfig,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.execution_policy import (
    evaluate_pre_tool_use,
    validate_implementer_commands,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.promotion import (
    promote_stable_candidate,
)


TASK_DIR = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "requirements"


def _req():
    req = json.loads((FIXTURES / "base_q8.json").read_text(encoding="utf-8"))
    req["resolved_requirements"] = resolve_capabilities(req)
    return req


def _orchestrator(tmp_path: Path, req=None) -> Orchestrator:
    state_dir = tmp_path / "state"
    store = StateStore(state_dir)
    store.init_run("pipeline-gate-test")
    cfg = OrchestratorConfig(
        workdir=state_dir,
        repo_root=TASK_DIR.parents[2],
        notebooks_dir=TASK_DIR / "notebooks",
        iterations_root=tmp_path / "workspace",
        logs_root=state_dir / "logs",
        state_dir=state_dir,
    )
    return Orchestrator(req=req or _req(), store=store, cfg=cfg, manager=object())


def test_iteration_copy_excludes_ad_hoc_server_logs(tmp_path: Path):
    workspace = CppInferIterationWorkspace(tmp_path / "workspace")
    first = workspace.open_iteration(1)
    (first / "src").mkdir()
    (first / "src" / "engine.cpp").write_text("// source\n", encoding="utf-8")
    (first / "server_stderr6.log").write_text("stale smoke\n", encoding="utf-8")
    (first / "server.stdout.log").write_text("stale smoke\n", encoding="utf-8")
    (first / "benchmark.log").write_text("project artifact\n", encoding="utf-8")

    second = workspace.open_iteration(2)

    assert (second / "src" / "engine.cpp").is_file()
    assert not (second / "server_stderr6.log").exists()
    assert not (second / "server.stdout.log").exists()
    assert (second / "benchmark.log").is_file()


def _valid_plan(iter_dir: Path, req):
    resolved = req["resolved_requirements"]
    capabilities = " ".join(resolved["required_capabilities"])
    suites = " ".join(resolved["correctness_suites"])
    (iter_dir / "plan.md").write_text(
        f"# Overall architecture\n{capabilities}\n"
        f"# Capability matrix\n# Iteration roadmap\n# Test plan\n{suites}\n",
        encoding="utf-8",
    )
    (iter_dir / "test_spec.md").write_text(
        f"# Correctness tests\n{capabilities}\n{suites}\n",
        encoding="utf-8",
    )
    first_suites = list(resolved["first_iteration_suites"])
    remaining_suites = [
        suite for suite in resolved["correctness_suites"]
        if suite not in first_suites
    ]
    milestones = [{
        "iteration": 1,
        "capabilities": list(resolved["required_capabilities"]),
        "suites": first_suites,
        "deliverables": ["runnable vertical slice"],
    }]
    if remaining_suites:
        milestones.append({
            "iteration": 2,
            "capabilities": [],
            "suites": remaining_suites,
            "deliverables": ["remaining correctness suites"],
        })
    (iter_dir / "plan_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "iteration": 1,
        "required_capabilities": list(resolved["required_capabilities"]),
        "current_iteration_capabilities": list(resolved["required_capabilities"]),
        "current_suites": first_suites,
        "deferred_suites": remaining_suites,
        "milestones": milestones,
    }), encoding="utf-8")


def _valid_artifacts(iter_dir: Path):
    (iter_dir / "CMakeLists.txt").write_text("project(test)", encoding="utf-8")
    (iter_dir / "serve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (iter_dir / "serve.sh").chmod(0o755)
    (iter_dir / "include").mkdir(exist_ok=True)
    (iter_dir / "src").mkdir(exist_ok=True)
    (iter_dir / "src/main.cpp").write_text("int main(){}", encoding="utf-8")
    (iter_dir / "src/engine.cpp").write_text("", encoding="utf-8")
    (iter_dir / "src/model_loader.cpp").write_text(
        "auto data_offset = align_up(tensor_info_end, general.alignment);\n"
        "auto file_offset = data_offset + tensor.offset;\n",
        encoding="utf-8",
    )


def _write_passing_oracle(logs_dir: Path, req):
    logs_dir.mkdir(parents=True, exist_ok=True)
    resolved = req["resolved_requirements"]
    (logs_dir / "oracle-report.json").write_text(json.dumps({
        "passed": True,
        "acceptance": {
            "passed": True,
            "required_capabilities": resolved["required_capabilities"],
            "suite_results": [
                {"suite": suite, "passed": True}
                for suite in resolved["correctness_suites"]
            ],
        },
    }), encoding="utf-8")
    (logs_dir / "oracle-stages.json").write_text(json.dumps({
        "schema_version": 1,
        "target_route": None,
        "stages": [{"id": "C4_full", "passed": True, "duration_s": 1.0}],
        "full_oracle_completed": True,
    }), encoding="utf-8")


def test_a_and_b_handlers_enforce_task_contract(tmp_path: Path, monkeypatch):
    orch = _orchestrator(tmp_path)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    monkeypatch.setattr(
        orch, "_run_agent", lambda **_kwargs: (True, None, None, None)
    )

    (iter_dir / "plan.md").write_text("# tiny plan\n", encoding="utf-8")
    (iter_dir / "test_spec.md").write_text("# tiny test\n", encoding="utf-8")
    outcome, _perf, failure = orch._do_plan(1, iter_dir, IterationContext())
    assert outcome == phases.LOGIC_FAIL
    assert "plan validation failed" in failure

    _valid_plan(iter_dir, orch.req)
    outcome, _perf, failure = orch._do_plan(1, iter_dir, IterationContext())
    assert outcome == phases.OK
    assert failure is None

    outcome, _perf, failure = orch._do_implement(1, iter_dir, IterationContext())
    assert outcome == phases.LOGIC_FAIL
    assert "artifact validation failed" in failure

    _valid_artifacts(iter_dir)
    outcome, _perf, failure = orch._do_implement(1, iter_dir, IterationContext())
    assert outcome == phases.OK
    assert failure is None


def _write_bash_event(path: Path, command: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "type": "assistant",
        "message": {
            "content": [{
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": command},
            }],
        },
    }) + "\n", encoding="utf-8")


def _append_tool_event(path: Path, tool_name: str):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "type": "assistant",
            "message": {
                "content": [{
                    "type": "tool_use",
                    "name": tool_name,
                    "input": {"description": "unnecessary verifier"},
                }],
            },
        }) + "\n")


def test_implementer_command_policy_rejects_unsafe_global_actions(tmp_path: Path):
    logs = tmp_path / "logs"
    events = logs / "iter1-implementer.attempt1.events.jsonl"
    _write_bash_event(
        events,
        "cmake --build build\npkill -9 -f metainfer_cpp_server\n"
        "pgrep -f metainfer_cpp_server\nbash test.sh &\nkill -9 12345",
    )
    _append_tool_event(events, "Agent")

    errors = validate_implementer_commands(logs, "iter1-implementer")

    assert any("bypass-system-build-sh" in error for error in errors)
    assert any("global-process-kill" in error for error in errors)
    assert any("global-process-selection" in error for error in errors)
    assert any("background-test-sh" in error for error in errors)
    assert any("unowned-process-kill" in error for error in errors)
    assert any("subagent-delegation" in error for error in errors)
    assert all("attempt1.events.jsonl:" in error for error in errors)


def test_implementer_command_policy_allows_owned_smoke_lifecycle(tmp_path: Path):
    logs = tmp_path / "logs"
    events = logs / "iter1-implementer.attempt1.events.jsonl"
    _write_bash_event(
        events,
        "bash build.sh && ./build/metainfer_cpp_server --version\n"
        "bash serve.sh \"$PORT\" & server_pid=$!; "
        "trap 'kill -TERM \"$server_pid\" 2>/dev/null || true' EXIT; "
        "curl -f http://127.0.0.1:$PORT/v1/models; "
        "kill -TERM \"$server_pid\"; wait \"$server_pid\"; "
        "bash test.sh 2>&1",
    )

    assert validate_implementer_commands(logs, "iter1-implementer") == ()


def test_owned_pid_cannot_mask_an_unowned_kill_target():
    decision = evaluate_pre_tool_use({
        "tool_name": "Bash",
        "tool_input": {
            "command": "serve.sh & pid=$!; kill \"$pid\"; kill 12345"
        },
    })
    trapped = evaluate_pre_tool_use({
        "tool_name": "Bash",
        "tool_input": {"command": "serve.sh & pid=$!; trap 'kill 12345' EXIT"},
    })

    assert "unowned-process-kill" in decision["rules"]
    assert "unowned-process-kill" in trapped["rules"]


def test_pre_tool_policy_blocks_before_bash_and_explains_recovery():
    blocked = evaluate_pre_tool_use({
        "tool_name": "Bash",
        "tool_input": {"command": "cmake --build build && pkill server"},
    })
    allowed = evaluate_pre_tool_use({
        "tool_name": "Bash",
        "tool_input": {"command": "bash build.sh && bash test.sh"},
    })

    assert blocked["allowed"] is False
    assert blocked["rules"] == ["global-process-kill", "bypass-system-build-sh"]
    assert "bash build.sh" in blocked["message"]
    assert "pkill/killall" in blocked["message"]
    assert allowed["allowed"] is True
    assert allowed["rules"] == []


def test_denied_pre_tool_command_does_not_fail_clean_retry(tmp_path: Path):
    logs = tmp_path / "logs"
    command = "cmake --build build"
    decision = evaluate_pre_tool_use({
        "tool_name": "Bash",
        "tool_input": {"command": command},
    })
    _write_bash_event(
        logs / "iter1-implementer.attempt1.events.jsonl", command
    )
    (logs / "iter1-implementer.policy-denials.jsonl").write_text(
        json.dumps({
            "command_sha256": decision["command_sha256"],
            "rules": decision["rules"],
            "blocked": True,
        }) + "\n",
        encoding="utf-8",
    )

    assert validate_implementer_commands(logs, "iter1-implementer") == ()


def test_command_policy_uses_subagent_manager_final_attempt(tmp_path: Path):
    logs = tmp_path / "logs"
    name = "iter1-implementer"
    _write_bash_event(
        logs / f"{name}.attempt1.events.jsonl", "pkill stale-server"
    )
    _write_bash_event(
        logs / f"{name}.attempt2.events.jsonl", "bash build.sh && bash test.sh"
    )
    (logs / f"{name}.status.json").write_text(
        json.dumps({"attempt": 2, "success": True}), encoding="utf-8"
    )

    assert validate_implementer_commands(logs, name) == ()
    forensic_errors = validate_implementer_commands(
        logs, name, final_attempt_only=False
    )
    assert any("attempt1.events.jsonl" in error for error in forensic_errors)


def test_implementer_launch_installs_task_local_pre_tool_policy(tmp_path: Path):
    orch = _orchestrator(tmp_path)

    class CapturingManager:
        spec = None

        def launch(self, spec):
            self.spec = spec

        def result(self, _name):
            return SimpleNamespace(
                duration_s=0.1,
                success=True,
                error=None,
                failure_mode=None,
                attempts=1,
                final_text="done",
                session_id="session",
            )

    manager = CapturingManager()
    orch.manager = manager
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()

    result = orch._run_agent(
        name="iter1-implementer",
        role="implementer",
        iteration=1,
        iter_dir=iter_dir,
        prompt="implement",
        timeout=10,
    )

    assert result == (True, None, None, "session")
    assert manager.spec is not None
    settings_index = manager.spec.extra_args.index("--settings") + 1
    settings_path = Path(manager.spec.extra_args[settings_index])
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    hook = settings["hooks"]["PreToolUse"][0]
    assert hook["matcher"] == "Bash"
    assert "pre_tool_policy.py" in hook["hooks"][0]["command"]
    assert manager.spec.extra_args[-2:] == [
        "--disallowedTools", "Agent,Task,TaskOutput,Explore",
    ]
    assert manager.spec.env_overrides["METAINFER_COMMAND_POLICY_AUDIT"].endswith(
        "iter1-implementer.policy-denials.jsonl"
    )


def test_c_debugger_launch_has_policy_and_no_internal_retries(tmp_path: Path):
    orch = _orchestrator(tmp_path)

    class CapturingManager:
        spec = None

        def launch(self, spec):
            self.spec = spec

        def result(self, _name):
            return SimpleNamespace(
                duration_s=0.1,
                success=True,
                error=None,
                failure_mode=None,
                attempts=1,
                final_text="done",
                session_id="session",
            )

    manager = CapturingManager()
    orch.manager = manager
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()

    result = orch._run_agent(
        name="iter1-c-debugger.attempt1",
        role="c_debugger",
        iteration=1,
        iter_dir=iter_dir,
        prompt="repair",
        timeout=600,
    )

    assert result == (True, None, None, "session")
    assert manager.spec.max_retries == 0
    assert "--settings" in manager.spec.extra_args
    assert manager.spec.extra_args[-2:] == [
        "--disallowedTools", "Agent,Task,TaskOutput,Explore",
    ]
    assert manager.spec.env_overrides["METAINFER_COMMAND_POLICY_AUDIT"].endswith(
        "iter1-c-debugger.attempt1.policy-denials.jsonl"
    )


def test_b_handler_turns_command_policy_violation_into_logic_failure(
    tmp_path: Path, monkeypatch,
):
    orch = _orchestrator(tmp_path)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _valid_artifacts(iter_dir)
    _write_bash_event(
        orch._logs_dir_for(1) / "iter1-implementer.attempt1.events.jsonl",
        "pkill -9 -f metainfer_cpp_server",
    )
    monkeypatch.setattr(
        orch, "_run_agent", lambda **_kwargs: (True, None, None, "session")
    )

    outcome, _perf, failure = orch._do_implement(
        1, iter_dir, IterationContext()
    )

    assert outcome == phases.LOGIC_FAIL
    assert "B execution policy failed" in failure
    assert "global-process-kill" in failure


def test_repeated_c_failure_signature_replans_without_second_debugger(
    tmp_path: Path, monkeypatch,
):
    orch = _orchestrator(tmp_path)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    oracle_calls = 0
    debugger_calls = 0
    failure = (
        "server did not become healthy\n--- server.stderr.log ---\n"
        "insufficient VRAM after weight upload: 0 MiB free, 512 MiB required"
    )

    def fail_oracle(*_args, **_kwargs):
        nonlocal oracle_calls
        oracle_calls += 1
        return phases.LOGIC_FAIL, None, failure

    def repair_once(**kwargs):
        nonlocal debugger_calls
        debugger_calls += 1
        logs = orch._logs_dir_for(1)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "c-repair-attempt1.md").write_text(
            "# C-step repair\n## Verification\nbash build.sh passed\n",
            encoding="utf-8",
        )
        assert "vram_or_kv_allocation" in kwargs["prompt"]
        assert kwargs["timeout"] == 900
        return True, None, None, "debug-session"

    monkeypatch.setattr(orch, "_run_oracle_once", fail_oracle)
    monkeypatch.setattr(orch, "_run_agent", repair_once)

    outcome, _perf, returned_failure = orch._do_test(
        1, iter_dir, IterationContext()
    )

    assert outcome == phases.REPLAN
    assert "repeated failure signature" in returned_failure
    assert oracle_calls == 2
    assert debugger_calls == 1
    transition = phases.next_transition("C_test", phases.REPLAN)
    assert transition is not None
    assert transition.to_phase == "A_plan"
    assert transition.consume_iteration is True
    records = [
        json.loads(line)
        for line in (
            orch._logs_dir_for(1) / "c-repairs.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["test"]["outcome"] == phases.REPLAN
    assert records[-1]["failure_classification"]["route_id"] == (
        "vram_or_kv_allocation"
    )


def test_changed_c_failure_route_gets_second_targeted_repair(
    tmp_path: Path, monkeypatch,
):
    orch = _orchestrator(tmp_path)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    outcomes = iter([
        (phases.LOGIC_FAIL, None, "model file not found: /models/a.gguf"),
        (phases.LOGIC_FAIL, None, "vocab_size mismatch: expected 1 got 0"),
        (phases.OK, {"oracle_avg_http_latency_ms": 1.0}, None),
    ])
    prompts = []
    repair_routes = []

    def run_oracle(*_args, **kwargs):
        repair_routes.append(kwargs.get("repair_route"))
        return next(outcomes)

    monkeypatch.setattr(orch, "_run_oracle_once", run_oracle)

    def repair(**kwargs):
        prompts.append(kwargs["prompt"])
        attempt = len(prompts)
        logs = orch._logs_dir_for(1)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"c-repair-attempt{attempt}.md").write_text(
            "# C-step repair\n## Verification\nbash build.sh passed\n",
            encoding="utf-8",
        )
        return True, None, None, "debug-session"

    monkeypatch.setattr(orch, "_run_agent", repair)

    outcome, perf, failure = orch._do_test(1, iter_dir, IterationContext())

    assert outcome == phases.OK
    assert perf == {"oracle_avg_http_latency_ms": 1.0}
    assert failure is None
    assert len(prompts) == 2
    assert "model_path_or_serve_args" in prompts[0]
    assert "gguf_metadata_or_tensor" in prompts[1]
    assert "Updated knowledge route" in prompts[1]
    assert repair_routes[0] is None
    assert repair_routes[1]["route_id"] == "model_path_or_serve_args"
    assert repair_routes[2]["route_id"] == "gguf_metadata_or_tensor"


def test_c_debugger_contract_failure_does_not_rerun_oracle(
    tmp_path: Path, monkeypatch,
):
    orch = _orchestrator(tmp_path)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    oracle_calls = 0

    def fail_oracle(*_args, **_kwargs):
        nonlocal oracle_calls
        oracle_calls += 1
        return phases.LOGIC_FAIL, None, "C++ build failed with rc=2"

    monkeypatch.setattr(orch, "_run_oracle_once", fail_oracle)
    monkeypatch.setattr(
        orch, "_run_agent", lambda **_kwargs: (True, None, None, "session")
    )

    outcome, _perf, failure = orch._do_test(1, iter_dir, IterationContext())

    assert outcome == phases.REPLAN
    assert "did not write the required structured repair log" in failure
    assert oracle_calls == 1


def test_c_debugger_policy_violation_replans_without_rerunning_oracle(
    tmp_path: Path, monkeypatch,
):
    orch = _orchestrator(tmp_path)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    oracle_calls = 0

    def fail_oracle(*_args, **_kwargs):
        nonlocal oracle_calls
        oracle_calls += 1
        return phases.LOGIC_FAIL, None, "C++ build failed with rc=2"

    def unsafe_repair(**kwargs):
        logs = orch._logs_dir_for(1)
        logs.mkdir(parents=True, exist_ok=True)
        name = kwargs["name"]
        _write_bash_event(
            logs / f"{name}.attempt1.events.jsonl", "cmake --build build"
        )
        (logs / "c-repair-attempt1.md").write_text(
            "# C-step repair\n## Verification\nunsafe build\n",
            encoding="utf-8",
        )
        return True, None, None, "session"

    monkeypatch.setattr(orch, "_run_oracle_once", fail_oracle)
    monkeypatch.setattr(orch, "_run_agent", unsafe_repair)

    outcome, _perf, failure = orch._do_test(1, iter_dir, IterationContext())

    assert outcome == phases.REPLAN
    assert "C debugger execution policy failed" in failure
    assert "bypass-system-build-sh" in failure
    assert oracle_calls == 1


def test_stable_candidate_promotes_only_after_required_gates(tmp_path: Path):
    orch = _orchestrator(tmp_path)
    iteration = orch.workspace.open_iteration(1)
    _valid_artifacts(iteration)
    logs = orch._logs_dir_for(1)
    _write_passing_oracle(logs, orch.req)
    record = {
        "iteration": 1,
        "status": "running",
        "phases": {
            "C_test": {"outcome": phases.OK},
            "D_review": {"outcome": phases.LOGIC_FAIL},
        },
    }

    rejected = promote_stable_candidate(
        orch.req, orch.cfg.state_dir, 1, iteration, logs, record
    )
    assert rejected["promoted"] is False
    assert not (orch.cfg.state_dir / "stable_candidate.json").exists()

    record["phases"]["D_review"]["outcome"] = phases.OK
    promoted = promote_stable_candidate(
        orch.req, orch.cfg.state_dir, 1, iteration, logs, record
    )
    stable = json.loads(
        (orch.cfg.state_dir / "stable_candidate.json").read_text(encoding="utf-8")
    )

    assert promoted["promoted"] is True
    assert stable["iteration"] == 1
    assert stable["gates"] == {"C_test": "ok", "D_review": "ok"}
    assert stable["oracle_report_sha256"]


def test_stable_candidate_rejects_incomplete_full_oracle(tmp_path: Path):
    orch = _orchestrator(tmp_path)
    iteration = orch.workspace.open_iteration(1)
    _valid_artifacts(iteration)
    logs = orch._logs_dir_for(1)
    _write_passing_oracle(logs, orch.req)
    (logs / "oracle-stages.json").write_text(json.dumps({
        "schema_version": 1,
        "stages": [{"id": "C4_full", "passed": False}],
        "full_oracle_completed": False,
    }), encoding="utf-8")
    record = {
        "iteration": 1,
        "status": "running",
        "phases": {
            "C_test": {"outcome": phases.OK},
            "D_review": {"outcome": phases.OK},
        },
    }

    rejected = promote_stable_candidate(
        orch.req, orch.cfg.state_dir, 1, iteration, logs, record
    )

    assert rejected["promoted"] is False
    assert "completed and passing C4" in rejected["failures"][0]
    assert not (orch.cfg.state_dir / "stable_candidate.json").exists()


def test_performance_enforcement_delays_promotion_until_e(tmp_path: Path):
    req = _req()
    req["enforce_performance"] = True
    orch = _orchestrator(tmp_path, req=req)
    iteration = orch.workspace.open_iteration(1)
    _valid_artifacts(iteration)
    logs = orch._logs_dir_for(1)
    _write_passing_oracle(logs, req)
    (iteration / "perf_report.json").write_text(json.dumps({
        "tokens_per_sec": 10,
        "errors_total": 0,
        "num_requests": 1,
    }), encoding="utf-8")
    record = {
        "iteration": 1,
        "status": "running",
        "phases": {
            "C_test": {"outcome": phases.OK},
            "D_review": {"outcome": phases.OK},
        },
    }

    rejected = promote_stable_candidate(
        req, orch.cfg.state_dir, 1, iteration, logs, record
    )
    assert rejected["promoted"] is False
    assert any("E_perf_test" in error for error in rejected["failures"])

    record["phases"]["E_perf_test"] = {"outcome": phases.OK}
    promoted = promote_stable_candidate(
        req, orch.cfg.state_dir, 1, iteration, logs, record
    )
    assert promoted["promoted"] is True
    assert promoted["performance_required"] is True


def test_final_audit_falls_back_from_failed_trial_to_stable_candidate(
    tmp_path: Path,
):
    orch = _orchestrator(tmp_path)
    stable_dir = orch.workspace.open_iteration(1)
    _valid_artifacts(stable_dir)
    stable_logs = orch._logs_dir_for(1)
    _write_passing_oracle(stable_logs, orch.req)
    stable_record = {
        "iteration": 1,
        "status": "failed",
        "phases": {
            "C_test": {"outcome": phases.OK},
            "D_review": {"outcome": phases.OK},
        },
    }
    orch.store.write_iteration(1, stable_record)
    promoted = promote_stable_candidate(
        orch.req,
        orch.cfg.state_dir,
        1,
        stable_dir,
        stable_logs,
        stable_record,
    )
    assert promoted["promoted"] is True
    orch.workspace.mark_complete(1)

    failed_dir = orch.workspace.open_iteration(2)
    (failed_dir / "src/engine.cpp").unlink()
    orch.store.write_iteration(2, {
        "iteration": 2,
        "status": "failed",
        "failure_reason": "experimental scheduler regressed correctness",
        "phases": {
            "C_test": {"outcome": phases.LOGIC_FAIL},
            "D_review": {"outcome": phases.LOGIC_FAIL},
        },
    })

    audit = orch._run_final_audit()

    assert audit["passed"] is True
    assert audit["audited_iteration"] == 1
    assert audit["latest_iteration"] == 2
    assert audit["used_stable_candidate"] is True


def test_invalid_a_deliverables_retry_without_consuming_iteration():
    transition = phases.TRANSITIONS[("A_plan", phases.LOGIC_FAIL)]

    assert transition.to_phase == "A_plan"
    assert transition.consume_iteration is False
    assert transition.carry_failure is True


def test_resume_interrupted_a_plan_restores_preceding_failure(tmp_path: Path):
    orch = _orchestrator(tmp_path)
    first = orch.workspace.open_iteration(1)
    (first / "kept.cpp").write_text("int kept;", encoding="utf-8")
    orch.workspace.mark_complete(1)
    orch.store.write_iteration(1, {
        "iteration": 1,
        "start_phase": "A_plan",
        "status": "failed",
        "outcome": phases.INFRA_FAIL,
        "failure_reason": "B failed after non-finite logits",
    })

    orch.workspace.open_iteration(2)
    orch.store.write_iteration(2, {
        "iteration": 2,
        "start_phase": "A_plan",
        "status": "running",
        "phases": {"A_plan": {"started_at": 1.0}},
    })

    resumed = orch._prepare_resume()

    assert resumed["iter_num"] == 2
    assert resumed["start_phase"] == "A_plan"
    assert resumed["last_outcome"] == phases.INFRA_FAIL
    assert resumed["carried_failure"] == "B failed after non-finite logits"


def test_resume_interrupted_a_plan_prefers_latest_in_place_failure(tmp_path: Path):
    orch = _orchestrator(tmp_path)
    first = orch.workspace.open_iteration(1)
    (first / "kept.cpp").write_text("int kept;", encoding="utf-8")
    orch.workspace.mark_complete(1)
    orch.store.write_iteration(1, {
        "iteration": 1,
        "start_phase": "A_plan",
        "status": "failed",
        "outcome": phases.INFRA_FAIL,
        "failure_reason": "older B failure",
    })

    orch.workspace.open_iteration(2)
    orch.store.write_iteration(2, {
        "iteration": 2,
        "start_phase": "A_plan",
        "status": "running",
        "phases": {
            "A_plan": {
                "started_at": 1.0,
                "ended_at": 2.0,
                "outcome": phases.LOGIC_FAIL,
                "failure": "latest A manifest validation failure",
            },
            "planner": {"started_at": 1.1, "success": True},
        },
    })

    resumed = orch._prepare_resume()

    assert resumed["iter_num"] == 2
    assert resumed["start_phase"] == "A_plan"
    assert resumed["last_outcome"] == phases.LOGIC_FAIL
    assert resumed["carried_failure"] == "latest A manifest validation failure"


def test_resume_interrupted_a_retry_uses_record_level_failure_fallback(
    tmp_path: Path,
):
    orch = _orchestrator(tmp_path)
    first = orch.workspace.open_iteration(1)
    (first / "kept.cpp").write_text("int kept;", encoding="utf-8")
    orch.workspace.mark_complete(1)
    orch.store.write_iteration(1, {
        "iteration": 1,
        "start_phase": "A_plan",
        "status": "failed",
        "outcome": phases.INFRA_FAIL,
        "failure_reason": "older B failure",
    })

    latest_failure = "A plan validation failed: GGUF data_offset is missing"
    orch.workspace.open_iteration(2)
    orch.store.write_iteration(2, {
        "iteration": 2,
        "start_phase": "A_plan",
        "status": "running",
        "failure_reason": latest_failure,
        # The interrupted retry started but did not persist its own result.
        "phases": {"A_plan": {"started_at": 3.0}},
    })

    resumed = orch._prepare_resume()

    assert resumed["start_phase"] == "A_plan"
    assert resumed["carried_failure"] == latest_failure


def test_resume_interrupted_c_test_preserves_completed_implementation(
    tmp_path: Path,
):
    orch = _orchestrator(tmp_path)
    iteration = orch.workspace.open_iteration(1)
    generated = iteration / "src" / "engine.cpp"
    generated.parent.mkdir()
    generated.write_text("int completed_implementation;", encoding="utf-8")
    orch.store.write_iteration(1, {
        "iteration": 1,
        "start_phase": "A_plan",
        "status": "running",
        "phases": {
            "A_plan": {
                "started_at": 1.0,
                "ended_at": 2.0,
                "outcome": phases.OK,
            },
            "B_implement": {
                "started_at": 3.0,
                "ended_at": 4.0,
                "outcome": phases.OK,
            },
            "C_test": {"started_at": 5.0},
        },
    })

    resumed = orch._prepare_resume()

    assert resumed == {
        "iter_num": 1,
        "start_phase": "C_test",
        "carried_failure": None,
        "last_outcome": phases.OK,
        "resume_existing": True,
    }
    assert generated.read_text(encoding="utf-8") == "int completed_implementation;"
    assert orch.store.load_iteration(1)["status"] == "running"


def test_resume_interrupted_b_still_discards_partial_implementation(
    tmp_path: Path,
):
    orch = _orchestrator(tmp_path)
    iteration = orch.workspace.open_iteration(1)
    partial = iteration / "partial.cpp"
    partial.write_text("incomplete", encoding="utf-8")
    orch.store.write_iteration(1, {
        "iteration": 1,
        "start_phase": "A_plan",
        "status": "running",
        "phases": {
            "A_plan": {
                "started_at": 1.0,
                "ended_at": 2.0,
                "outcome": phases.OK,
            },
            "B_implement": {"started_at": 3.0},
        },
    })

    resumed = orch._prepare_resume()

    assert resumed["resume_existing"] is False
    assert resumed["start_phase"] == "A_plan"
    assert not partial.exists()


def test_resumed_iteration_persists_carried_failure_before_phase_finishes(
    tmp_path: Path, monkeypatch,
):
    orch = _orchestrator(tmp_path)
    observed = {}

    def inspect_open_record(phase, iteration, _iter_dir, _rec, _ctx):
        observed.update(orch.store.load_iteration(iteration))
        orch._stop = True
        return phases.INFRA_FAIL, None, "interrupted again"

    monkeypatch.setattr(orch, "_run_phase", inspect_open_record)
    orch._loop({
        "iter_num": 2,
        "start_phase": "A_plan",
        "carried_failure": "latest A validator failure",
        "last_outcome": phases.LOGIC_FAIL,
    })

    assert observed["failure_reason"] == "latest A validator failure"


def test_phase_result_is_persisted_without_erasing_agent_metadata(
    tmp_path: Path, monkeypatch,
):
    orch = _orchestrator(tmp_path)

    def fail_plan(phase, iteration, _iter_dir, _rec, _ctx):
        record = orch.store.load_iteration(iteration)
        record["phases"]["planner"] = {
            "success": True,
            "session_id": "planner-session",
        }
        orch.store.write_iteration(iteration, record)
        orch._stop = True
        return phases.LOGIC_FAIL, None, "A manifest validation failure"

    monkeypatch.setattr(orch, "_run_phase", fail_plan)

    orch._loop()

    record = orch.store.load_iteration(1)
    assert record["phases"]["planner"]["session_id"] == "planner-session"
    assert record["phases"]["A_plan"]["outcome"] == phases.LOGIC_FAIL
    assert record["phases"]["A_plan"]["failure"] == "A manifest validation failure"


def test_d_handler_gates_needs_fix_and_missing_verdict(tmp_path: Path, monkeypatch):
    orch = _orchestrator(tmp_path)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    ctx = IterationContext(last_outcome=phases.OK)
    review_path = orch._logs_dir_for(1) / "review.md"

    def write_review(verdict: str):
        def fake_agent(**_kwargs):
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(f"Verdict: {verdict}\n", encoding="utf-8")
            return True, None, None, None
        return fake_agent

    monkeypatch.setattr(orch, "_run_agent", write_review("NEEDS_FIX"))
    outcome, _perf, failure = orch._do_review(1, iter_dir, ctx)
    assert outcome == phases.LOGIC_FAIL
    assert "NEEDS_FIX" in failure

    monkeypatch.setattr(orch, "_run_agent", write_review("PASS"))
    outcome, _perf, failure = orch._do_review(1, iter_dir, ctx)
    assert outcome == phases.OK
    assert failure is None


def test_c_uses_current_manifest_suites_not_deferred_suites(tmp_path: Path):
    req = json.loads((FIXTURES / "full_f16.json").read_text(encoding="utf-8"))
    req["max_iterations"] = 10
    req["resolved_requirements"] = resolve_capabilities(req)
    orch = _orchestrator(tmp_path, req=req)
    iter_dir = tmp_path / "iteration"
    iter_dir.mkdir()
    _valid_plan(iter_dir, req)
    resolved = req["resolved_requirements"]
    active = set(resolved["first_iteration_suites"])

    class DeferredFailureOracle:
        task_type = "gen-cpp-infer-framework"

        def run(self, *, report_dir, **_kwargs):
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "oracle-report.json").write_text(json.dumps({
                "passed": False,
                "acceptance": {
                    "passed": False,
                    "required_capabilities": resolved["required_capabilities"],
                    "suite_results": [
                        {"suite": suite, "passed": suite in active, "evidence": []}
                        for suite in resolved["correctness_suites"]
                    ],
                },
            }), encoding="utf-8")
            return SimpleNamespace(
                passed=False,
                perf={},
                judge_mode="disabled",
                cases=[],
                failure_reason="only deferred suites failed",
            )

    outcome, _perf, failure = orch._run_oracle_once(
        1, iter_dir, IterationContext(), DeferredFailureOracle()
    )

    assert outcome == phases.OK
    assert failure is None
    report = json.loads(
        (orch._logs_dir_for(1) / "oracle-report.json").read_text(encoding="utf-8")
    )
    assert report["development_gate"]["passed"] is True
    assert report["development_gate"]["full_acceptance_passed"] is False
