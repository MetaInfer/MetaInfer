import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import time
import unittest

from metainfer.orchestrator.tasks import get_task
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.oracles.perf import (
    _scan_profile_dir,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator import phases as cpp_phases
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.capabilities import (
    CAPABILITIES,
    visible_feature_labels,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.iteration_record import (
    IterationRecord,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.pipeline import (
    IterationContext,
    Orchestrator,
    OrchestratorConfig,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.orchestrator import (
    _extract_execution_mode,
    _extract_max_wall_time_s,
)
from metainfer.tasks.gen_infer_framework_cpp.orchestrator.prompts import (
    _CPP_PHASE_NOTEBOOKS,
    _notebooks_hint,
    c_repair_prompt,
    implement_prompt,
    perf_plan_prompt,
    perf_review_prompt,
    plan_prompt,
    review_prompt,
)
from metainfer.server.forms import (
    list_task_types,
    load_form_schema,
    validate_submission,
)
from metainfer.server.registry import get as get_web_plugin
from metainfer.orchestrator.state import StateStore
from metainfer.tasks.gen_infer_framework_cpp.server import _state_readers


TASK_TYPE = "gen-infer-framework-cpp"

NATIVE_CPP_CONTRACTS = (
    "attention_kv_contracts.md",
    "engine_contracts.md",
    "framework_contracts.md",
    "hardware_profile_contracts.md",
    "memory_contracts.md",
    "native_service_contracts.md",
    "operator_contracts.md",
    "profiling_contracts.md",
    "qwen3_model_contracts.md",
    "testing_contracts.md",
    "tp_communication_contracts.md",
    "weight_loading_contracts.md",
)

IMPROVEMENT_PLAN_FILES = (
    "00_sources_and_governance.md",
    "01_sampling_and_generation.md",
    "02_continuous_batching.md",
    "03_paged_kv_and_prefix_cache.md",
    "04_hip_operator_optimization.md",
    "05_tensor_parallel_rccl.md",
    "06_quantization_and_weight_formats.md",
    "07_native_service_streaming.md",
    "08_profiling_and_benchmarking.md",
    "09_model_compatibility_and_loader.md",
)

REQUIRED_NOTEBOOK_LAYOUT = {
    "00_overview": ("README.md",),
    "01_framework_design": tuple(
        f"{index:02d}_{name}.md"
        for index, name in enumerate((
            "architecture", "scheduler", "kv_cache", "model_runner",
            "sampler", "memory_pool", "request_lifecycle",
            "tensor_ownership", "hardware_discovery", "cmake_dtk_toolchain",
        ), start=1)
    ),
    "03_operators": (
        "01_attention_ops.md", "02_rmsnorm_rope.md",
        "03_mlp_activation.md", "04_sampling_ops.md",
        "05_hip_blas_backend.md",
    ),
    "04_parallel_strategies": (
        "01_tensor_parallel.md", "02_qwen_dense_tp.md",
        "03_qwen_moe_tp.md", "04_rccl_collectives.md",
    ),
    "05_inference_service": (
        "01_long_running_native_service.md", "02_openai_http_api.md",
        "03_sse_streaming.md", "04_process_lifecycle.md",
    ),
    "06_profiling": (
        "00_overview.md", "01_native_trace.md", "02_roctx_rocprof.md",
        "03_gpu_event_benchmark.md",
    ),
    "07_improvementPlan": ("README.md", *IMPROVEMENT_PLAN_FILES),
}


def _requirements():
    return {
        "task_id": "cpp-test",
        "task_type": TASK_TYPE,
        "target_model": "/models/test",
        "target_hardware": "NVIDIA H100",
        "accelerator_backend": "CUDA",
        "cpp_standard": "C++20",
        "build_type": "RelWithDebInfo",
        "features": ["Paged KV cache", "Continuous batching"],
    }


class CppFrameworkTaskTest(unittest.TestCase):
    def test_all_routed_cpp_notebooks_exist(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        routed = {
            path
            for paths in _CPP_PHASE_NOTEBOOKS.values()
            for path in paths
        }
        for path in routed:
            self.assertTrue((notebooks / path).is_file(), path)

    def test_dynamic_cpp_notebook_routes_exist(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        variants = [
            {
                **_requirements(),
                "model_family": "Qwen3 Dense",
                "features": [],
                "tensor_parallel_size": "1",
            },
            {
                **_requirements(),
                "model_family": "Qwen3 MoE",
                "features": [
                    "Paged KV cache", "Continuous batching",
                    "Tensor parallelism", "Streaming responses",
                ],
                "tensor_parallel_size": "4",
            },
            {
                **_requirements(),
                "model_family": "Qwen3.5 Hybrid",
                "features": ["Streaming responses"],
                "tensor_parallel_size": "2",
            },
        ]
        for req in variants:
            for phase in _CPP_PHASE_NOTEBOOKS:
                hint = _notebooks_hint(req, phase)
                for relative in re.findall(r"- `([^`]+\.md)`", hint):
                    self.assertTrue((notebooks / relative).is_file(), relative)
                self.assertNotIn("gen_infer_framework/notebooks", hint)

    def test_native_contracts_do_not_embed_python_implementations(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        cpp_contracts = notebooks / "00_contracts"

        for filename in NATIVE_CPP_CONTRACTS:
            text = (cpp_contracts / filename).read_text(encoding="utf-8")
            self.assertNotIn("```python", text, filename)
            self.assertNotIn("import torch", text, filename)
            self.assertNotIn("torch.", text, filename)
            self.assertIn("## ", text, filename)
            self.assertGreater(len(text), 1000, filename)

    def test_cpp_notebook_information_architecture_is_complete(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        for dirname, filenames in REQUIRED_NOTEBOOK_LAYOUT.items():
            for filename in filenames:
                self.assertTrue((notebooks / dirname / filename).is_file())
        self.assertFalse((notebooks / "00_contracts" / "cpp").exists())
        self.assertFalse((notebooks / "09_cpp_inference").exists())
        self.assertTrue((notebooks / "02_model_specifics/02_qwen3/README.md").is_file())
        for dirname in ("06_experience", "07_improvementPlan", "08_issues"):
            self.assertTrue((notebooks / dirname / "README.md").is_file())

    def test_cpp_improvement_plans_are_pinned_and_self_contained(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        root = notebooks / "07_improvementPlan"
        source_text = (root / "00_sources_and_governance.md").read_text(
            encoding="utf-8"
        )
        expected_sources = {
            "ggml-org/llama.cpp": "0bd0ec60998d0f71ec45471b633bf2403ac81956",
            "vllm-project/vllm": "17fdd421009092a569e9ed22346b932e55824fb5",
            "NVIDIA/TensorRT-LLM": "ed1a0b9bfad2074b49452aa0597ee9a476cb47a5",
            "sgl-project/sglang": "bbd2a3fe4a267b1e5a2a49792a2273e2e519d881",
            "mlc-ai/mlc-llm": "a2bcc5c86678b72a86b7aadc29b643a5ce63c747",
            "ROCm/rocm-examples": "41dd7463e65e230af913db75d48a1d6c0dcff6bc",
            "ROCm/hipBLAS": "23b26a0093345264e7387481cbe01d1e1ae55fda",
            "ROCm/rocm-systems": "27b4e4dd4438e205c3c9163efe4084b890bbb08e",
            "safetensors/safetensors": "6eb4dc9a28ebce297606e0f4836bbf28839cacef",
        }
        for repository, commit in expected_sources.items():
            self.assertIn(repository, source_text)
            self.assertIn(commit, source_text)
        for license_name in ("MIT", "Apache-2.0", "复合License"):
            self.assertIn(license_name, source_text)
        self.assertIn("不得重新打开这些仓库", source_text)

        for filename in IMPROVEMENT_PLAN_FILES[1:]:
            text = (root / filename).read_text(encoding="utf-8")
            self.assertGreater(len(text), 2500, filename)
            self.assertIn("状态：", text, filename)
            self.assertIn("SRC-", text, filename)
            self.assertIn("验收", text, filename)
            self.assertIn("风险", text, filename)

    def test_cpp_improvement_plans_are_routed_by_phase_and_feature(self):
        req = {
            **_requirements(),
            "target_hardware": "Hygon Z200SM",
            "accelerator_backend": "Hygon DTK / HIP",
            "model_family": "Qwen3 MoE",
            "weight_dtype": "INT4 quantized",
            "features": [
                "Paged KV cache", "Continuous batching",
                "Tensor parallelism", "Streaming responses",
            ],
            "tensor_parallel_size": "4",
        }
        plan_hint = _notebooks_hint(req, "plan")
        for relative in (
            "07_improvementPlan/00_sources_and_governance.md",
            "07_improvementPlan/01_sampling_and_generation.md",
            "07_improvementPlan/02_continuous_batching.md",
            "07_improvementPlan/03_paged_kv_and_prefix_cache.md",
            "07_improvementPlan/04_hip_operator_optimization.md",
            "07_improvementPlan/05_tensor_parallel_rccl.md",
            "07_improvementPlan/06_quantization_and_weight_formats.md",
            "07_improvementPlan/07_native_service_streaming.md",
            "07_improvementPlan/09_model_compatibility_and_loader.md",
        ):
            self.assertIn(relative, plan_hint)

        perf_hint = _notebooks_hint(req, "perf")
        for relative in (
            "07_improvementPlan/02_continuous_batching.md",
            "07_improvementPlan/03_paged_kv_and_prefix_cache.md",
            "07_improvementPlan/04_hip_operator_optimization.md",
            "07_improvementPlan/05_tensor_parallel_rccl.md",
            "07_improvementPlan/06_quantization_and_weight_formats.md",
            "07_improvementPlan/08_profiling_and_benchmarking.md",
        ):
            self.assertIn(relative, perf_hint)

    def test_cpp_notebooks_have_no_python_templates_or_legacy_paths(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        forbidden = (
            "```python", "import torch", "torch.", "00_contracts/cpp/",
            "09_cpp_inference/", "gen_infer_framework/notebooks",
        )
        for path in notebooks.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, str(path.relative_to(notebooks)))

    def test_cpp_notebook_references_resolve(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        root_prefixes = {
            "00_overview", "00_contracts", "01_framework_design",
            "02_model_specifics", "03_operators", "04_parallel_strategies",
            "05_inference_service", "06_experience", "06_profiling",
            "07_improvementPlan", "08_issues",
        }
        generated_artifacts = {
            "plan.md", "perf-review.md", "LANGUAGE_BOUNDARY.md",
        }
        reference = re.compile(r"`([^`]+\.md)`")
        for path in notebooks.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            for relative in reference.findall(text):
                if relative in generated_artifacts:
                    continue
                first = Path(relative).parts[0]
                if first in root_prefixes:
                    target = notebooks / relative
                elif "/" not in relative:
                    target = path.parent / relative
                else:
                    continue
                self.assertTrue(
                    target.is_file(),
                    f"{path.relative_to(notebooks)} -> {relative}",
                )

    def test_card_schema_and_validation(self):
        task_ids = [entry["id"] for entry in list_task_types()]
        self.assertIn(TASK_TYPE, task_ids)

        schema = load_form_schema(TASK_TYPE)
        self.assertIsNotNone(schema)
        self.assertEqual(schema["label"], "构建 C++ 推理框架")
        field_keys = {field["key"] for field in schema["fields"]}
        self.assertTrue({
            "target_model",
            "model_family",
            "weight_dtype",
            "kv_cache_dtype",
            "quantization_format",
            "target_hardware",
            "accelerator_backend",
            "assigned_devices",
            "tensor_parallel_size",
            "cpp_standard",
            "build_type",
            "execution_mode",
            "max_wall_time_minutes",
            "target_tokens_per_sec",
        }.issubset(field_keys))
        hardware_field = next(field for field in schema["fields"] if field["key"] == "target_hardware")
        self.assertIn("Hygon Z200SM", {option["label"] for option in hardware_field["options"]})
        backend_field = next(field for field in schema["fields"] if field["key"] == "accelerator_backend")
        self.assertIn("Hygon DTK / HIP", {option["label"] for option in backend_field["options"]})
        feature_field = next(
            field for field in schema["fields"] if field["key"] == "features"
        )
        form_features = {option["label"] for option in feature_field["options"]}
        self.assertEqual(form_features, set(visible_feature_labels()))
        self.assertNotIn("Speculative decoding", form_features)
        for spec in CAPABILITIES.values():
            self.assertTrue(spec.notebooks, spec.label)
            self.assertTrue(spec.oracle_owners, spec.label)
            for relative in spec.notebooks:
                self.assertTrue(
                    (Path(__file__).resolve().parents[1] / "notebooks" / relative).is_file(),
                    f"{spec.label}: {relative}",
                )
        model_field = next(
            field for field in schema["fields"] if field["key"] == "model_family"
        )
        self.assertIn(
            "Qwen3.5 / Qwen3.6 Hybrid",
            {option["label"] for option in model_field["options"]},
        )
        iteration_field = next(
            field for field in schema["fields"] if field["key"] == "max_iterations"
        )
        self.assertTrue(
            {"1", "3", "5"}.issubset(
                {option["label"] for option in iteration_field["options"]}
            )
        )

        self.assertTrue(validate_submission(TASK_TYPE, _requirements())["ok"])
        valid_devices = {**_requirements(), "assigned_devices": "0, 1,2,3"}
        self.assertTrue(validate_submission(TASK_TYPE, valid_devices)["ok"])
        invalid = validate_submission(TASK_TYPE, {})
        self.assertFalse(invalid["ok"])
        self.assertIn("cpp_standard", invalid["errors"])

    def test_plugin_owns_its_orchestrator(self):
        plugin = get_task(TASK_TYPE)
        self.assertEqual(
            plugin.cli_module,
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.cli",
        )
        self.assertEqual(
            plugin.phases_module,
            "metainfer.tasks.gen_infer_framework_cpp.orchestrator.phases",
        )

        web_plugin = get_web_plugin(TASK_TYPE)
        self.assertIsNotNone(web_plugin)
        self.assertEqual(web_plugin.detail_view_module, "app/cpp-gf-detail")
        self.assertEqual(web_plugin.extra_stylesheets, ["cpp-gf.css"])
        self.assertTrue(
            (web_plugin.frontend_dir / "cpp-gf-detail.js").is_file()
        )
        route_paths = {
            route.path for route in web_plugin.build_router(web_plugin).routes
        }
        self.assertTrue(
            {"/iterations", "/charts", "/state-graph", "/qa/start"}
            .issubset(route_paths)
        )

    def test_card_overlays_cleared_pid_as_paused_iteration(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            (state / "iterations").mkdir()
            (state / "iterations/008.json").write_text(
                '{"iteration":8,"status":"running","started_at":100.0}',
                encoding="utf-8",
            )
            (state / "orchestrator.pid").write_text(
                '{"pid":null,"finished_at":160.0}', encoding="utf-8",
            )

            record = _state_readers.read_iteration(state, 8)

            self.assertEqual(record["status"], "paused")
            self.assertTrue(record["paused"])
            self.assertEqual(record["duration_s"], 60.0)

    def test_notebooks_live_inside_cpp_task_package(self):
        from metainfer.tasks.gen_infer_framework_cpp.orchestrator import (
            orchestrator as orchestrator_module,
        )

        notebooks = orchestrator_module._NOTEBOOKS_DIR
        self.assertEqual(notebooks.parent.name, "gen_infer_framework_cpp")
        self.assertTrue((notebooks / "00_contracts").is_dir())
        self.assertTrue((notebooks / "01_framework_design").is_dir())
        self.assertFalse((notebooks / "09_cpp_inference").exists())

    def test_cpp_task_uses_the_immutable_perf_oracle(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.req = {"task_type": TASK_TYPE}
        expected = ("ok", {"tokens_per_sec": 1.0}, None)
        orchestrator._do_perf_test_oracle = lambda *args: expected

        actual = orchestrator._do_perf_test(1, Path("/tmp/iter"), None, None)
        self.assertIs(actual, expected)

    def test_c_failure_is_carried_through_review_to_fresh_plan(self):
        c_transition = cpp_phases.next_transition(
            "C_test", cpp_phases.LOGIC_FAIL,
        )
        self.assertIsNotNone(c_transition)
        self.assertEqual(c_transition.to_phase, "D_review")
        self.assertTrue(c_transition.carry_failure)

        d_transition = cpp_phases.next_transition(
            "D_review", cpp_phases.LOGIC_FAIL,
        )
        self.assertIsNotNone(d_transition)
        self.assertEqual(d_transition.to_phase, "A_plan")
        self.assertTrue(d_transition.carry_failure)
        self.assertTrue(d_transition.consume_iteration)

        failed = IterationRecord(
            iteration=1,
            outcome=cpp_phases.LOGIC_FAIL,
            failure_reason="oracle output was incorrect",
        )
        self.assertEqual(Orchestrator._phase_after(failed), "A_plan")

    def test_perf_is_reviewed_only_after_measurement(self):
        e_transition = cpp_phases.next_transition(
            "E_perf_test", cpp_phases.OK,
        )
        self.assertIsNotNone(e_transition)
        self.assertEqual(e_transition.to_phase, "G_perf_review")
        g_transition = cpp_phases.next_transition(
            "G_perf_review", cpp_phases.OK,
        )
        self.assertIsNotNone(g_transition)
        self.assertEqual(g_transition.to_phase, "F_perf_plan")

        prompt = perf_review_prompt(
            _requirements(), Path("/tmp/iter"), Path("/tmp/notebooks"), 7,
            {"tokens_per_sec": 50.64}, "correctness passed", Path("/tmp/logs"),
        )
        self.assertIn("50.64", prompt)
        self.assertIn("do not reuse a previous iteration's number", prompt)

    def test_review_preserves_c_failure_for_iteration_close(self):
        class _Store:
            def append_timeline(self, *_args, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orchestrator = Orchestrator.__new__(Orchestrator)
            orchestrator.req = _requirements()
            orchestrator.store = _Store()
            orchestrator.cfg = SimpleNamespace(
                notebooks_dir=root / "notebooks",
                review_timeout_s=1,
            )
            orchestrator._logs_dir_for = lambda _n: root
            orchestrator._run_agent = lambda **_kwargs: (True, None, None, None)
            ctx = IterationContext(
                last_outcome=cpp_phases.LOGIC_FAIL,
                failure="oracle output was incorrect",
            )

            outcome, perf, failure = orchestrator._do_review(1, root, ctx)

            self.assertEqual(outcome, cpp_phases.LOGIC_FAIL)
            self.assertIsNone(perf)
            self.assertEqual(failure, "oracle output was incorrect")

    def test_c_repair_has_short_timeout_no_internal_retry_and_no_diff_fail_fast(self):
        class _Store:
            def __init__(self):
                self.events = []

            def append_timeline(self, event, payload):
                self.events.append((event, payload))

        class _Manager:
            @staticmethod
            def result(_name):
                return SimpleNamespace(final_text="")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logs = root / "logs"
            logs.mkdir()
            store = _Store()
            orchestrator = Orchestrator.__new__(Orchestrator)
            orchestrator.req = _requirements()
            orchestrator.store = store
            orchestrator.manager = _Manager()
            orchestrator.cfg = SimpleNamespace(
                max_c_retries=3,
                c_repair_timeout_s=17,
                primary_perf_metric=None,
                notebooks_dir=root / "notebooks",
                review_timeout_s=1,
            )
            orchestrator._logs_dir_for = lambda _n: logs
            oracle_calls = []
            orchestrator._run_oracle_once = lambda *args: (
                oracle_calls.append(args) or (
                    cpp_phases.LOGIC_FAIL, None, "same semantic failure"
                )
            )
            launches = []

            def run_agent(**kwargs):
                launches.append(kwargs)
                return True, None, None, "session"

            orchestrator._run_agent = run_agent

            outcome, _perf, failure = orchestrator._do_test(
                1, root, IterationContext(),
            )

            self.assertEqual(outcome, cpp_phases.LOGIC_FAIL)
            self.assertEqual(len(oracle_calls), 1)
            self.assertEqual(launches[0]["timeout"], 17)
            self.assertEqual(launches[0]["max_retries"], 0)
            self.assertIn("produced no source diff", failure)
            self.assertIn(
                "c_test_no_source_diff", [event for event, _ in store.events],
            )
            repair_records = [
                json.loads(line)
                for line in (logs / "c-repairs.jsonl").read_text(
                    encoding="utf-8",
                ).splitlines()
            ]
            self.assertEqual(repair_records[0]["changed_files"], [])

    def test_cpp_contract_is_present_across_agent_phases(self):
        req = _requirements()
        iter_dir = Path("/tmp/cpp-iteration")
        notebooks = Path("/tmp/notebooks")
        logs = Path("/tmp/logs")

        plan = plan_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)
        implement = implement_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)
        repair = c_repair_prompt(
            req, iter_dir, notebooks, 1, 1, 3, "compile failed", logs,
        )
        review = review_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)
        perf_plan = perf_plan_prompt(req, iter_dir, notebooks, 1, logs_dir=logs)
        perf_review = perf_review_prompt(
            req, iter_dir, notebooks, 1,
            {"tokens_per_sec": 1.0}, None, logs,
        )

        for prompt in (plan, implement, repair, review, perf_review, perf_plan):
            self.assertIn("NON-NEGOTIABLE implementation language: C++ first", prompt)
            self.assertIn("NON-NEGOTIABLE accelerator execution boundary", prompt)
            self.assertIn("nested-loop CPU GEMM", prompt)
            self.assertIn("exact checkpoint chat-template rendering", prompt)
            self.assertIn("LANGUAGE_BOUNDARY.md", prompt)

        self.assertIn("cmake -S . -B build", implement)
        self.assertIn("implementation_report.json", implement)
        self.assertIn("prefill_decode_cosine", implement)
        self.assertIn("CMakeLists.txt", implement)
        self.assertIn("compiled native server binary", implement)
        self.assertIn("Content-Length", implement)
        self.assertIn("survive SIGPIPE", implement)
        self.assertIn("Do NOT background an", implement)
        self.assertNotIn('python3 -c "import server"', implement)
        self.assertIn("native C++ source and access boundary", implement)
        self.assertIn("exact checkpoint directory", implement)
        self.assertIn("Do not embed CPython or use Torch", implement)
        self.assertNotIn("The ONLY allowed dependency surface is", implement)
        self.assertNotIn("HTTP server primitives (e.g. `fastapi`", implement)
        self.assertIn("Python runtime substitution", review)
        self.assertIn("Do not claim that performance improved", review)
        self.assertIn("cmake --build build --parallel", repair)
        self.assertIn(
            "00_contracts/profiling_contracts.md", perf_plan
        )

    def test_cpp_notebook_route_and_hardware_contract_are_deterministic(self):
        req = _requirements()
        req.update({
            "target_hardware": "Hygon Z200SM",
            "accelerator_backend": "Hygon DTK / HIP",
            "model_family": "Qwen3 MoE",
            "features": [
                "Paged KV cache",
                "Continuous batching",
                "Tensor parallelism",
                "Streaming responses",
            ],
            "hardware_profile_path": "/state/hardware_profile.json",
            "hardware_profile": {
                "vendor_family": "hygon",
                "device_count": 4,
                "hip_architectures": ["gfx906"],
                "validation": {"status": "matched", "blockers": []},
            },
        })
        prompt = plan_prompt(req, Path("/tmp/cpp"), Path("/tmp/notebooks"), 1)
        self.assertIn("C++ knowledge-base route", prompt)
        self.assertIn("REQUIRED phase bundle", prompt)
        self.assertIn("COMPONENT guides", prompt)
        self.assertIn("read only the 2-5 files", prompt)
        self.assertIn("00_contracts/hardware_profile_contracts.md", prompt)
        self.assertIn("00_contracts/attention_kv_contracts.md", prompt)
        self.assertIn(
            "00_contracts/tp_communication_contracts.md", prompt
        )
        self.assertIn("01_framework_design/03_kv_cache.md", prompt)
        self.assertIn("01_framework_design/02_scheduler.md", prompt)
        self.assertIn("04_parallel_strategies/02_qwen_dense_tp.md", prompt)
        self.assertIn("05_inference_service/03_sse_streaming.md", prompt)
        self.assertIn("02_model_specifics/02_qwen3/02_moe.md", prompt)
        self.assertIn("03_operators/05_hip_blas_backend.md", prompt)
        self.assertIn("NON-NEGOTIABLE detected hardware contract", prompt)
        self.assertIn('"hip_architectures": [', prompt)
        self.assertNotIn("Glob notebooks/**/*.md", prompt)
        self.assertNotIn("gen_infer_framework/notebooks", prompt)

    def test_python_framework_prompt_does_not_inherit_cpp_contract(self):
        req = {
            "task_id": "python-test",
            "task_type": "gen-infer-framework",
            "target_model": "/models/test",
            "target_hardware": "Hygon K100AI",
        }
        iter_dir = Path("/tmp/python-iteration")
        notebooks = Path("/tmp/notebooks")

        plan = plan_prompt(req, iter_dir, notebooks, 1)
        implement = implement_prompt(req, iter_dir, notebooks, 1)

        self.assertNotIn("NON-NEGOTIABLE implementation language: C++", plan)
        self.assertNotIn("CMake/build checks", plan)
        self.assertNotIn("CMakeLists.txt", implement)
        self.assertIn('python3 -c "import server"', implement)
        self.assertIn("Glob notebooks/**/*.md", plan)
        self.assertNotIn("native C++ source and access boundary", plan)

    def test_wall_time_and_execution_mode_resolution(self):
        self.assertEqual(
            _extract_execution_mode({"execution_mode": "Correctness only"}),
            "correctness_only",
        )
        self.assertEqual(
            _extract_execution_mode({
                "execution_mode": "Correctness + optimization",
            }),
            "optimize",
        )
        self.assertEqual(_extract_execution_mode({}), "optimize")
        self.assertEqual(
            _extract_max_wall_time_s({"max_wall_time_minutes": "2.5"}),
            150,
        )

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cfg = SimpleNamespace(max_wall_time_s=30)
        orchestrator._wall_deadline_epoch = time.time() + 2
        self.assertLessEqual(orchestrator._bounded_timeout(60), 2)

    def test_correctness_only_mode_stops_after_d_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"
            store = StateStore(state_dir)
            store.init_or_resume(task_id="bounded", task_type=TASK_TYPE)
            cfg = OrchestratorConfig(
                workdir=state_dir,
                repo_root=root,
                notebooks_dir=root / "notebooks",
                iterations_root=root / "workspace",
                state_dir=state_dir,
                logs_root=root / "logs",
                max_iterations=5,
                execution_mode="correctness_only",
            )

            class Manager:
                def shutdown(self):
                    return None

            orchestrator = Orchestrator(
                req={"task_id": "bounded", "task_type": TASK_TYPE},
                store=store,
                cfg=cfg,
                manager=Manager(),
            )
            phases_seen = []

            def run_phase(phase, *_args):
                phases_seen.append(phase)
                return cpp_phases.OK, None, None

            orchestrator._run_phase = run_phase
            orchestrator._write_retrospective = lambda **_kwargs: None
            orchestrator._loop()

            self.assertEqual(
                phases_seen,
                ["A_plan", "B_implement", "C_test", "D_review"],
            )
            run = store.load_run()
            self.assertTrue(run.finished)
            self.assertEqual(run.final_status, "success")

    def test_optimize_mode_stops_after_measured_target_and_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_dir = root / "state"
            store = StateStore(state_dir)
            store.init_or_resume(task_id="target", task_type=TASK_TYPE)
            cfg = OrchestratorConfig(
                workdir=state_dir,
                repo_root=root,
                notebooks_dir=root / "notebooks",
                iterations_root=root / "workspace",
                state_dir=state_dir,
                logs_root=root / "logs",
                max_iterations=5,
                execution_mode="optimize",
            )

            class Manager:
                def shutdown(self):
                    return None

            orchestrator = Orchestrator(
                req={
                    "task_id": "target",
                    "task_type": TASK_TYPE,
                    "target_tokens_per_sec": 40,
                },
                store=store,
                cfg=cfg,
                manager=Manager(),
            )
            phases_seen = []

            def run_phase(phase, *_args):
                phases_seen.append(phase)
                perf = {"tokens_per_sec": 50.0} if phase in {
                    "E_perf_test", "G_perf_review",
                } else None
                return cpp_phases.OK, perf, None

            orchestrator._run_phase = run_phase
            orchestrator._loop()

            self.assertEqual(
                phases_seen,
                [
                    "A_plan", "B_implement", "C_test", "D_review",
                    "E_perf_test", "G_perf_review",
                ],
            )
            self.assertEqual(store.load_run().final_status, "success")

    def test_native_profiler_artifacts_are_discovered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("trace.json.gz", "native.nsys-rep", "rocprof.csv"):
                (root / name).write_bytes(b"x" * 2048)
            (root / "ignored.bin").write_bytes(b"x" * 2048)

            artifacts = _scan_profile_dir(root)
            names = {Path(item["path"]).name for item in artifacts}
            self.assertEqual(
                names,
                {"trace.json.gz", "native.nsys-rep", "rocprof.csv"},
            )


if __name__ == "__main__":
    unittest.main()
