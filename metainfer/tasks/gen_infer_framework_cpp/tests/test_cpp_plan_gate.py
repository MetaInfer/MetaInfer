from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from metainfer.tasks.gen_infer_framework_cpp.orchestrator import phases
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.pipeline import (
    IterationContext,
    Orchestrator,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.plan_gate import (
    CORE_CAPABILITIES,
    FULL_GATE,
    INCREMENTAL_GATE,
    PlanGateContext,
    frozen_requirements,
    resolve_plan_gate_context,
    validate_plan,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.prompts import (
    implement_prompt,
    plan_prompt,
)


REQ = {
    "task_id": "cpp-plan-gate",
    "task_type": "gen-infer-framework-cpp",
    "target_model": "/models/Qwen3-1.7B",
    "model_family": "Qwen3 Dense",
    "weight_dtype": "FP16",
    "kv_cache_dtype": "FP16",
    "quantization_format": "None / full precision",
    "target_hardware": "Hygon Z200SM",
    "accelerator_backend": "Hygon DTK / HIP",
    "assigned_devices": "0",
    "tensor_parallel_size": "1",
    "cpp_standard": "C++20",
    "build_type": "RelWithDebInfo",
    "features": ["Paged KV cache", "Continuous batching"],
    "perf_target": "Latency",
    "execution_mode": "Correctness only",
    "max_wall_time_minutes": 165,
}

CAPABILITY_FILES = {
    "native_cpp_runtime": ["CMakeLists.txt"],
    "real_model_loading": ["src/model_loader.cpp"],
    "tokenizer_chat_template": ["src/tokenizer.cpp"],
    "target_device_execution": ["src/model_runner.cpp"],
    "deterministic_decode": ["src/decoder.cpp"],
    "openai_chat_endpoint": ["src/http_server.cpp"],
    "long_running_server": ["serve.sh"],
    "graceful_shutdown": ["src/http_server.cpp"],
}


def _write_valid_plan(root: Path, iteration: int = 1) -> None:
    architecture = """# Complete native C++ architecture

CMake builds the native C++ runtime. Model loading and weight loading consume
the real checkpoint. The tokenizer and chat template preserve exact special
tokens. The model runner owns accelerator execution on the target device. A
scheduler provides Continuous batching and manages the Paged KV cache. Decode
uses deterministic greedy decoding plus seeded stochastic sampling and
detokenization. The OpenAI
/v1/chat/completions service owns HTTP request validation. Server lifecycle
includes foreground startup, SIGTERM shutdown, and native resource cleanup.
"""
    plan = """# Minimum end-to-end delivery

Deliver a minimum end-to-end native C++ path using the real model checkpoint.
`CMakeLists.txt` builds the server. `serve.sh` foreground-execs it.
`src/http_server.cpp` implements `/v1/chat/completions`, OpenAI response
shaping, and graceful shutdown. `src/tokenizer.cpp` renders the checkpoint chat
template. `src/model_loader.cpp` loads real weights. `src/model_runner.cpp`
performs target accelerator forward execution. `src/decoder.cpp` performs
deterministic greedy and seeded stochastic sampling plus detokenization. Native
build and smoke checks exercise one complete request. Risks include device
memory and checkpoint dtype support.
"""
    test_spec = """# Immutable full C Oracle

The immutable Oracle launches foreground `serve.sh`, confirms a native C++
server process, and sends temperature=0 requests to `/v1/chat/completions`
plus seeded temperature>0 and top_p sampling requests against the real
checkpoint on the selected accelerator. The response must
contain a string at `choices[0].message.content`; SIGTERM must cleanly release
the server and device resources.
"""
    manifest = {
        "schema_version": 2,
        "task_type": REQ["task_type"],
        "iteration": iteration,
        "gate_mode": "full",
        "architecture": {"file": "architecture.md", "complete": True},
        "plan": {
            "file": "plan.md",
            "goal": "minimum_end_to_end_ready_for_oracle",
        },
        "test_spec": {"file": "test_spec.md", "oracle": "immutable_full_c"},
        "requirements": frozen_requirements(REQ),
        "core_capabilities": {
            capability: {
                "status": "delivered_after_b",
                "files": CAPABILITY_FILES[capability],
            }
            for capability in CORE_CAPABILITIES
        },
        "requested_features": {
            feature: {
                "architecture": "covered",
                "delivery": "deferred",
                "reason": "The first B keeps a contiguous correctness path.",
            }
            for feature in REQ["features"]
        },
        "delivery_items": [{
            "id": "native-e2e",
            "summary": "Deliver the real native model execution path",
            "files": sorted({
                path
                for files in CAPABILITY_FILES.values()
                for path in files
            }),
            "tests": ["Immutable full C Oracle"],
        }],
        "deferred": [],
    }
    (root / "architecture.md").write_text(architecture, encoding="utf-8")
    (root / "plan.md").write_text(plan, encoding="utf-8")
    (root / "test_spec.md").write_text(test_spec, encoding="utf-8")
    (root / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )


def _successful_record(iteration: int = 1) -> dict:
    return {
        "iteration": iteration,
        "status": "success",
        "outcome": "ok",
        "interrupted": False,
        "phases": {"C_test": {"outcome": "ok"}},
    }


def _write_verified_baseline(root: Path, iteration: int = 1) -> Path:
    baseline = root / f"{iteration:03d}"
    baseline.mkdir(parents=True)
    _write_valid_plan(baseline, iteration)
    (baseline / ".metainfer-completed").touch()
    return baseline


def _write_valid_incremental_plan(
    root: Path, iteration: int = 2, baseline: int = 1,
) -> None:
    architecture = """# Complete native C++ architecture

CMake builds the native C++ runtime. Model loading and weight loading consume
the real checkpoint. The tokenizer and chat template preserve exact special
tokens. The model runner owns accelerator execution on the target device. A
scheduler provides Continuous batching and manages the Paged KV cache. Decode
uses deterministic greedy decoding plus seeded stochastic sampling. The OpenAI
/v1/chat/completions service owns HTTP validation. Server lifecycle includes
foreground startup, SIGTERM shutdown, and native resource cleanup.
"""
    plan = """# Iteration objective

Optimize target-device execution inherited from iteration 1.

## Change scope

Modify `src/model_runner.cpp` and verify it with
`tests/test_model_runner.cpp`. No inherited file inventory is repeated.

## Regression verification

Run the immutable full C Oracle after the focused test.
"""
    test_spec = """# Immutable full C Oracle regression

Run `tests/test_model_runner.cpp`, then launch foreground `serve.sh` and send
temperature=0 requests to `/v1/chat/completions` against the real checkpoint.
Seeded temperature>0 and top_p sampling must also affect generation. The
native C++ server response contains `choices[0].message.content` and
SIGTERM releases all resources.
"""
    core = {
        capability: {
            "status": "inherited_verified",
            "from_iteration": baseline,
        }
        for capability in CORE_CAPABILITIES
    }
    core["target_device_execution"] = {
        "status": "delivered_after_b",
        "files": ["src/model_runner.cpp"],
    }
    manifest = {
        "schema_version": 2,
        "task_type": REQ["task_type"],
        "iteration": iteration,
        "gate_mode": "incremental",
        "inherits_verified_iteration": baseline,
        "architecture": {"file": "architecture.md", "complete": True},
        "plan": {"file": "plan.md", "goal": "incremental_change_ready_for_oracle"},
        "test_spec": {"file": "test_spec.md", "oracle": "immutable_full_c"},
        "requirements": frozen_requirements(REQ),
        "iteration_objective": {
            "summary": "Optimize the model runner",
            "source": "perf_plan",
            "capabilities": ["target_device_execution"],
        },
        "change_scope": {
            "changed_files": ["src/model_runner.cpp"],
            "test_files": ["tests/test_model_runner.cpp"],
            "affected_core_capabilities": ["target_device_execution"],
            "regression_oracle": "immutable_full_c",
        },
        "core_capabilities": core,
        "requested_features": {
            feature: {
                "architecture": "covered",
                "delivery": "deferred",
                "reason": "Not part of this verified delta.",
            }
            for feature in REQ["features"]
        },
        "delivery_items": [{
            "id": "optimize-model-runner",
            "summary": "Optimize and verify the model runner",
            "files": ["src/model_runner.cpp", "tests/test_model_runner.cpp"],
            "tests": ["tests/test_model_runner.cpp"],
        }],
        "deferred": [],
    }
    (root / "architecture.md").write_text(architecture, encoding="utf-8")
    (root / "plan.md").write_text(plan, encoding="utf-8")
    (root / "test_spec.md").write_text(test_spec, encoding="utf-8")
    (root / "plan_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )


def _case_valid_cpp_plan_passes(tmp_path: Path):
    _write_valid_plan(tmp_path)
    result = validate_plan(tmp_path, REQ, 1)
    assert result.passed, result.diagnostics()


def _case_cpp_missing_artifact_fails(tmp_path: Path):
    _write_valid_plan(tmp_path)
    (tmp_path / "test_spec.md").unlink()
    result = validate_plan(tmp_path, REQ, 1)
    assert not result.passed
    assert "missing required planning artifact: test_spec.md" in result.errors


def _case_cpp_stub_fixed_503_and_core_deferral_fail(tmp_path: Path):
    _write_valid_plan(tmp_path)
    with (tmp_path / "plan.md").open("a", encoding="utf-8") as stream:
        stream.write("\nImplement a placeholder server with a hardcoded 503 response.\n")
    manifest_path = tmp_path / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["core_capabilities"]["target_device_execution"]["status"] = "deferred"
    manifest["deferred"] = [{"capability": "device execution", "reason": "later"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_plan(tmp_path, REQ, 1)

    assert not result.passed
    joined = "\n".join(result.errors)
    assert "mock/stub/placeholder" in joined
    assert "fixed HTTP 503" in joined
    assert "core capability cannot be deferred" in joined


def _case_cpp_frozen_requirement_mismatch_fails(tmp_path: Path):
    _write_valid_plan(tmp_path)
    manifest_path = tmp_path / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["requirements"]["accelerator_backend"] = "CPU"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_plan(tmp_path, REQ, 1)
    assert not result.passed
    assert any("accelerator_backend" in error for error in result.errors)


def _case_cpp_sampling_contract_requires_seed(tmp_path: Path):
    _write_valid_plan(tmp_path)
    test_spec = tmp_path / "test_spec.md"
    test_spec.write_text(
        test_spec.read_text(encoding="utf-8").replace("seeded ", ""),
        encoding="utf-8",
    )

    result = validate_plan(tmp_path, REQ, 1)

    assert not result.passed
    assert "test_spec.md does not mirror seeded sampling" in result.errors


def _case_cpp_negated_test_sh_authoring_is_allowed(tmp_path: Path):
    _write_valid_plan(tmp_path)
    test_spec = tmp_path / "test_spec.md"
    with test_spec.open("a", encoding="utf-8") as stream:
        stream.write(
            "\nThe planner must not write a test.sh. Do NOT author a `test.sh`; "
            "the immutable Oracle owns execution.\n"
        )

    result = validate_plan(tmp_path, REQ, 1)

    assert result.passed, result.diagnostics()


def _case_cpp_positive_test_sh_authoring_is_rejected(tmp_path: Path):
    _write_valid_plan(tmp_path)
    test_spec = tmp_path / "test_spec.md"
    with test_spec.open("a", encoding="utf-8") as stream:
        stream.write("\nCreate a `test.sh` as the source of truth.\n")

    result = validate_plan(tmp_path, REQ, 1)

    assert not result.passed
    assert any("agent-authored test.sh" in error for error in result.errors)


def _case_cpp_delivery_test_id_table_mapping_passes(tmp_path: Path):
    _write_valid_plan(tmp_path)
    manifest_path = tmp_path / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["delivery_items"][0]["tests"] = [
        "P01-T01: cmake configure with gfx906 completes without error",
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    test_spec = tmp_path / "test_spec.md"
    with test_spec.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n| Delivery item | Test ID | Assertion |\n"
            "|---|---|---|\n"
            "| native-e2e | P01-T01 | CMake configures for gfx906 |\n"
        )

    result = validate_plan(tmp_path, REQ, 1)

    assert result.passed, result.diagnostics()


def _case_cpp_delivery_test_id_must_map_to_its_item(tmp_path: Path):
    _write_valid_plan(tmp_path)
    manifest_path = tmp_path / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["delivery_items"][0]["tests"] = [
        "P01-T01: cmake configure with gfx906 completes without error",
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    test_spec = tmp_path / "test_spec.md"
    with test_spec.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n| Delivery item | Test ID | Assertion |\n"
            "|---|---|---|\n"
            "| unrelated-item | P01-T01 | CMake configures for gfx906 |\n"
        )

    result = validate_plan(tmp_path, REQ, 1)

    assert not result.passed
    assert any(
        "stable TEST-ID mapped to the delivery item" in error
        for error in result.errors
    )


def _case_cpp_delivery_items_cover_delivery_scope(tmp_path: Path):
    _write_valid_plan(tmp_path)
    manifest_path = tmp_path / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["delivery_items"][0]["files"] = ["src/model_runner.cpp"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_plan(tmp_path, REQ, 1)

    assert not result.passed
    assert any(
        "delivery_items do not account for delivery-scope files" in error
        for error in result.errors
    )


def _case_verified_previous_iteration_selects_incremental_gate(tmp_path: Path):
    _write_verified_baseline(tmp_path)
    current = tmp_path / "002"
    current.mkdir()

    context = resolve_plan_gate_context(current, REQ, 2, _successful_record())

    assert context.mode == INCREMENTAL_GATE
    assert context.baseline_iteration == 1


def _case_unverified_previous_iteration_selects_full_gate(tmp_path: Path):
    _write_verified_baseline(tmp_path)
    current = tmp_path / "002"
    current.mkdir()
    failed = _successful_record()
    failed["status"] = "failed"
    failed["outcome"] = "logic_fail"
    failed["phases"]["C_test"]["outcome"] = "logic_fail"

    failed_context = resolve_plan_gate_context(current, REQ, 2, failed)
    interrupted = _successful_record()
    interrupted["interrupted"] = True
    interrupted_context = resolve_plan_gate_context(current, REQ, 2, interrupted)
    (tmp_path / "001" / ".metainfer-completed").unlink()
    incomplete_context = resolve_plan_gate_context(
        current, REQ, 2, _successful_record(),
    )

    assert failed_context.mode == FULL_GATE
    assert interrupted_context.mode == FULL_GATE
    assert incomplete_context.mode == FULL_GATE


def _case_valid_incremental_plan_passes_without_full_inventory(tmp_path: Path):
    _write_verified_baseline(tmp_path)
    current = tmp_path / "002"
    current.mkdir()
    _write_valid_incremental_plan(current)
    context = resolve_plan_gate_context(current, REQ, 2, _successful_record())

    result = validate_plan(current, REQ, 2, context)

    assert result.passed, result.diagnostics()
    plan = (current / "plan.md").read_text(encoding="utf-8")
    assert "src/model_loader.cpp" not in plan
    assert result.to_dict()["gate"]["mode"] == INCREMENTAL_GATE


def _case_incremental_scope_mismatch_fails(tmp_path: Path):
    _write_verified_baseline(tmp_path)
    current = tmp_path / "002"
    current.mkdir()
    _write_valid_incremental_plan(current)
    manifest_path = current / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inherits_verified_iteration"] = 99
    manifest["change_scope"]["changed_files"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    context = resolve_plan_gate_context(current, REQ, 2, _successful_record())

    result = validate_plan(current, REQ, 2, context)

    assert not result.passed
    joined = "\n".join(result.errors)
    assert "inherits_verified_iteration must be 1" in joined
    assert "changed_files must name non-empty relative files" in joined


def _case_cpp_gate_replans_in_same_a_before_b(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    prompts = []
    calls = 0

    class Store:
        def __init__(self):
            self.events = []

        def append_timeline(self, event, payload):
            self.events.append((event, payload))

    store = Store()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.req = REQ
    orchestrator.cfg = SimpleNamespace(
        notebooks_dir=tmp_path / "notebooks",
        plan_timeout_s=1,
    )
    orchestrator.store = store
    orchestrator._logs_dir_for = lambda _n: logs

    def run_agent(**kwargs):
        nonlocal calls
        calls += 1
        prompts.append(kwargs["prompt"])
        if calls == 2:
            _write_valid_plan(tmp_path)
        return True, None, None, None

    orchestrator._run_agent = run_agent
    ctx = IterationContext()

    first = orchestrator._do_plan(1, tmp_path, ctx)
    transition = phases.next_transition("A_plan", first[0])
    assert first[0] == phases.LOGIC_FAIL
    assert ctx.plan_gate_feedback is not None
    assert transition is not None
    assert transition.to_phase == "A_plan"
    assert transition.consume_iteration is False

    second = orchestrator._do_plan(1, tmp_path, ctx)
    assert second == (phases.OK, None, None)
    assert "Previous Plan Gate rejection" in prompts[1]
    assert "missing required planning artifact" in prompts[1]
    assert ctx.plan_gate_feedback is None
    report = json.loads((logs / "plan-gate-report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert [event for event, _ in store.events] == [
        "plan_gate_rejected",
        "plan_gate_passed",
    ]


def _case_pipeline_uses_incremental_gate_after_verified_c(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_verified_baseline(tmp_path)
    current = tmp_path / "002"
    current.mkdir()

    class Store:
        def __init__(self):
            self.events = []

        def load_iteration(self, n):
            return _successful_record(n) if n == 1 else None

        def append_timeline(self, event, payload):
            self.events.append((event, payload))

    store = Store()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.req = REQ
    orchestrator.cfg = SimpleNamespace(
        notebooks_dir=tmp_path / "notebooks",
        plan_timeout_s=1,
    )
    orchestrator.store = store
    orchestrator._logs_dir_for = lambda _n: logs

    def run_agent(**kwargs):
        assert "Plan Gate mode: INCREMENTAL" in kwargs["prompt"]
        _write_valid_incremental_plan(current)
        return True, None, None, None

    orchestrator._run_agent = run_agent

    outcome = orchestrator._do_plan(2, current, IterationContext())

    assert outcome == (phases.OK, None, None)
    report = json.loads((logs / "plan-gate-report.json").read_text(encoding="utf-8"))
    assert report["gate"]["mode"] == INCREMENTAL_GATE
    assert report["gate"]["baseline_iteration"] == 1
    assert store.events[-1][1]["gate_mode"] == INCREMENTAL_GATE


class CppPlanGateTest(unittest.TestCase):
    def _run_with_tempdir(self, test):
        with tempfile.TemporaryDirectory() as td:
            test(Path(td))

    def test_valid_plan(self):
        self._run_with_tempdir(_case_valid_cpp_plan_passes)

    def test_unsupported_form_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_valid_plan(root)
            req = {**REQ, "features": ["Speculative decoding"]}

            result = validate_plan(root, req, 1)

            self.assertFalse(result.passed)
            self.assertTrue(any(
                "unsupported C++ feature: speculative decoding" in error
                for error in result.errors
            ))

    def test_tp_size_and_feature_must_agree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_valid_plan(root)
            req = {**REQ, "tensor_parallel_size": "4"}

            result = validate_plan(root, req, 1)

            self.assertFalse(result.passed)
            self.assertTrue(any(
                "tensor_parallel_size > 1 requires" in error
                for error in result.errors
            ))

    def test_missing_artifact(self):
        self._run_with_tempdir(_case_cpp_missing_artifact_fails)

    def test_rejects_stub_fixed_503_and_core_deferral(self):
        self._run_with_tempdir(_case_cpp_stub_fixed_503_and_core_deferral_fail)

    def test_gate_replans_before_b(self):
        self._run_with_tempdir(_case_cpp_gate_replans_in_same_a_before_b)

    def test_pipeline_uses_incremental_gate_after_verified_c(self):
        self._run_with_tempdir(
            _case_pipeline_uses_incremental_gate_after_verified_c,
        )

    def test_frozen_requirement_mismatch(self):
        self._run_with_tempdir(_case_cpp_frozen_requirement_mismatch_fails)

    def test_sampling_contract_requires_seed(self):
        self._run_with_tempdir(_case_cpp_sampling_contract_requires_seed)

    def test_negated_test_sh_authoring_is_allowed(self):
        self._run_with_tempdir(_case_cpp_negated_test_sh_authoring_is_allowed)

    def test_positive_test_sh_authoring_is_rejected(self):
        self._run_with_tempdir(_case_cpp_positive_test_sh_authoring_is_rejected)

    def test_delivery_test_id_table_mapping_passes(self):
        self._run_with_tempdir(_case_cpp_delivery_test_id_table_mapping_passes)

    def test_delivery_test_id_must_map_to_its_item(self):
        self._run_with_tempdir(_case_cpp_delivery_test_id_must_map_to_its_item)

    def test_delivery_items_cover_delivery_scope(self):
        self._run_with_tempdir(_case_cpp_delivery_items_cover_delivery_scope)

    def test_verified_previous_iteration_selects_incremental_gate(self):
        self._run_with_tempdir(
            _case_verified_previous_iteration_selects_incremental_gate,
        )

    def test_failed_interrupted_or_incomplete_previous_iteration_uses_full_gate(self):
        self._run_with_tempdir(
            _case_unverified_previous_iteration_selects_full_gate,
        )

    def test_incremental_plan_does_not_repeat_full_file_inventory(self):
        self._run_with_tempdir(
            _case_valid_incremental_plan_passes_without_full_inventory,
        )

    def test_incremental_scope_mismatch_is_rejected(self):
        self._run_with_tempdir(_case_incremental_scope_mismatch_fails)

    def test_prompts_enforce_validated_four_artifact_contract(self):
        plan = plan_prompt(REQ, Path("/tmp/iter"), Path("/tmp/notebooks"), 1)
        implement = implement_prompt(
            REQ, Path("/tmp/iter"), Path("/tmp/notebooks"), 1,
        )
        for artifact in (
            "architecture.md", "plan.md", "test_spec.md", "plan_manifest.json",
        ):
            self.assertIn(artifact, plan)
            self.assertIn(artifact, implement)
        self.assertIn("minimum_end_to_end_ready_for_oracle", plan)
        self.assertIn("delivered_after_b", implement)
        self.assertIn("temperature>0", plan)
        self.assertIn("genuine seeded stochastic sampling", implement)
        self.assertIn("implementation_report.json", implement)

    def test_incremental_prompt_uses_delta_manifest(self):
        context = PlanGateContext(
            INCREMENTAL_GATE,
            "previous C passed",
            baseline_iteration=1,
        )
        plan = plan_prompt(
            REQ,
            Path("/tmp/iter/002"),
            Path("/tmp/notebooks"),
            2,
            plan_gate_context=context,
        )
        self.assertIn("Plan Gate mode: INCREMENTAL", plan)
        self.assertIn("incremental_change_ready_for_oracle", plan)
        self.assertIn('"inherits_verified_iteration": 1', plan)
        self.assertIn('"status": "inherited_verified"', plan)


if __name__ == "__main__":
    unittest.main()
